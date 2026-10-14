# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rank-local safetensors loading for frozen GLM-5.3-Flash weights."""

from __future__ import annotations

import json
import logging
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TensorSliceTarget:
    """Local storage and its offset inside a global native tensor."""

    tensor: torch.Tensor
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]


@dataclass(slots=True)
class ShardedLoadStats:
    """Small evidence bundle for rank-local HF loading."""

    tensors_loaded: int = 0
    tensors_skipped_lora: int = 0
    bytes_loaded: int = 0
    verified_keys: list[str] | None = None
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tensors_loaded": self.tensors_loaded,
            "tensors_skipped_lora": self.tensors_skipped_lora,
            "bytes_loaded": self.bytes_loaded,
            "verified_keys": list(self.verified_keys or []),
            "elapsed_seconds": self.elapsed_seconds,
        }


class SafetensorsSliceReader:
    """Open HF safetensors shards lazily and serve tensor slices by key."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.weight_map = _load_weight_map(self.checkpoint_path)
        self._stack = ExitStack()
        self._handles: dict[str, Any] = {}

    def __enter__(self) -> "SafetensorsSliceReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self._stack.close()

    def get_slice(self, hf_key: str) -> Any:
        try:
            shard_name = self.weight_map[hf_key]
        except KeyError as exc:
            raise KeyError(
                f"HF key {hf_key!r} is absent from safetensors index"
            ) from exc
        handle = self._handles.get(shard_name)
        if handle is None:
            shard_path = self.checkpoint_path / shard_name
            handle = self._stack.enter_context(
                safe_open(shard_path, framework="pt", device="cpu")
            )
            self._handles[shard_name] = handle
        return handle.get_slice(hf_key)


def load_rank_local_hf_safetensors(
    model: nn.Module | list[nn.Module],
    adapter: Any,
    checkpoint_path: str | Path,
    *,
    verify: bool = True,
) -> ShardedLoadStats:
    """Load only this rank's FSDP-local GLM tensors from HF safetensors.

    The model must already be materialized from meta and FSDP-sharded. Values
    that are DTensors are copied into their ``_local_tensor`` using the current
    rank's placement-derived offsets. LoRA tensors keep their random init.
    """

    begin = time.monotonic()
    stats = ShardedLoadStats(verified_keys=[])
    model_parts = [model] if isinstance(model, nn.Module) else model
    with torch.no_grad(), SafetensorsSliceReader(checkpoint_path) as reader:
        for model_part in model_parts:
            for native_key, value in _named_tensors(model_part).items():
                if _is_lora_key(native_key):
                    stats.tensors_skipped_lora += 1
                    continue
                target = _target_for_tensor(value)
                if target.tensor.numel() == 0:
                    continue
                if _is_gate_up_key(native_key):
                    _copy_grouped_gate_up(adapter, reader, native_key, target, stats)
                elif _is_down_key(native_key):
                    _copy_grouped_down(adapter, reader, native_key, target, stats)
                elif native_key.endswith(".dt_bias"):
                    hf_key = _native_to_hf_key(adapter, native_key)
                    _copy_flattened_tensor(reader, hf_key, target, stats)
                else:
                    hf_key = _native_to_hf_key(adapter, native_key)
                    _copy_one_to_one_tensor(reader, hf_key, target, stats)
                stats.tensors_loaded += 1

        if verify:
            stats.verified_keys = _verify_selected_slices(
                model_parts, adapter, reader, stats.verified_keys or []
            )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    stats.elapsed_seconds = time.monotonic() - begin
    logger.info("Loaded rank-local GLM HF shards: %s", stats.as_dict())
    return stats


def _named_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    tensors.update(dict(model.named_parameters()))
    tensors.update(dict(model.named_buffers()))
    return tensors


def _load_weight_map(checkpoint_path: Path) -> dict[str, str]:
    index_candidates = sorted(checkpoint_path.glob("*.safetensors.index.json"))
    if index_candidates:
        index = json.loads(index_candidates[0].read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"{index_candidates[0]} has no weight_map object")
        return {str(key): str(value) for key, value in weight_map.items()}

    weight_map: dict[str, str] = {}
    for shard_path in sorted(checkpoint_path.glob("*.safetensors")):
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                weight_map[key] = shard_path.name
    if not weight_map:
        raise FileNotFoundError(f"no safetensors shards found under {checkpoint_path}")
    return weight_map


def _target_for_tensor(value: torch.Tensor) -> TensorSliceTarget:
    if isinstance(value, DTensor):
        local_shape, global_offset = compute_local_shape_and_global_offset(
            value.shape,
            value.device_mesh,
            value.placements,
        )
        local_tensor = value._local_tensor
        if tuple(local_tensor.shape) != tuple(local_shape):
            raise ValueError(
                "DTensor local shape mismatch: "
                f"expected {tuple(local_shape)}, got {tuple(local_tensor.shape)}"
            )
        return TensorSliceTarget(
            tensor=local_tensor,
            global_shape=tuple(value.shape),
            global_offset=tuple(global_offset),
        )
    return TensorSliceTarget(
        tensor=value,
        global_shape=tuple(value.shape),
        global_offset=(0,) * value.ndim,
    )


def _native_to_hf_key(adapter: Any, native_key: str) -> str:
    hf_key = adapter._map_native_key(native_key)
    if hf_key is None:
        raise KeyError(f"unmapped GLM native key {native_key!r}")
    return hf_key


def _copy_one_to_one_tensor(
    reader: SafetensorsSliceReader,
    hf_key: str,
    target: TensorSliceTarget,
    stats: ShardedLoadStats,
) -> None:
    hf_slice = reader.get_slice(hf_key)
    hf_shape = tuple(hf_slice.get_shape())
    if hf_shape != target.global_shape:
        raise ValueError(
            f"shape mismatch for {hf_key}: HF {hf_shape} vs native "
            f"{target.global_shape}"
        )
    source = _read_slice(
        hf_slice,
        target.global_offset,
        tuple(target.tensor.shape),
    )
    _copy_into(target.tensor, source)
    stats.bytes_loaded += _num_bytes(source)


def _copy_flattened_tensor(
    reader: SafetensorsSliceReader,
    hf_key: str,
    target: TensorSliceTarget,
    stats: ShardedLoadStats,
) -> None:
    hf_slice = reader.get_slice(hf_key)
    hf_shape = tuple(hf_slice.get_shape())
    expected_numel = _shape_numel(target.global_shape)
    if hf_shape != (expected_numel,):
        raise ValueError(
            f"shape mismatch for {hf_key}: HF {hf_shape} vs flattened native "
            f"{target.global_shape}"
        )
    source = torch.empty(
        tuple(target.tensor.shape),
        dtype=target.tensor.dtype,
        device="cpu",
    )
    for local_row in range(target.tensor.shape[0]):
        global_row = target.global_offset[0] + local_row
        start = global_row * target.global_shape[1] + target.global_offset[1]
        end = start + target.tensor.shape[1]
        source[local_row].copy_(hf_slice[start:end].to(source.dtype))
    _copy_into(target.tensor, source)
    stats.bytes_loaded += _num_bytes(source)


def _copy_grouped_gate_up(
    adapter: Any,
    reader: SafetensorsSliceReader,
    native_key: str,
    target: TensorSliceTarget,
    stats: ShardedLoadStats,
) -> None:
    layer = _layer_from_grouped_key(native_key)
    intermediate_size = adapter.glm_config.moe_intermediate_size
    _copy_gate_or_up_rows(
        reader,
        layer=layer,
        projection="gate",
        row_base=0,
        intermediate_size=intermediate_size,
        target=target,
        stats=stats,
    )
    _copy_gate_or_up_rows(
        reader,
        layer=layer,
        projection="up",
        row_base=intermediate_size,
        intermediate_size=intermediate_size,
        target=target,
        stats=stats,
    )


def _copy_grouped_down(
    adapter: Any,
    reader: SafetensorsSliceReader,
    native_key: str,
    target: TensorSliceTarget,
    stats: ShardedLoadStats,
) -> None:
    del adapter
    layer = _layer_from_grouped_key(native_key)
    expert_start, row_start, col_start = target.global_offset
    local_experts, local_rows, local_cols = target.tensor.shape
    for local_expert in range(local_experts):
        expert = expert_start + local_expert
        hf_key = (
            f"model.language_model.layers.{layer}.mlp.experts.{expert}."
            "down_proj.weight"
        )
        source = _read_slice(
            reader.get_slice(hf_key),
            (row_start, col_start),
            (local_rows, local_cols),
        )
        _copy_into(target.tensor[local_expert], source)
        stats.bytes_loaded += _num_bytes(source)


def _copy_gate_or_up_rows(
    reader: SafetensorsSliceReader,
    *,
    layer: int,
    projection: str,
    row_base: int,
    intermediate_size: int,
    target: TensorSliceTarget,
    stats: ShardedLoadStats,
) -> None:
    expert_start, row_start, col_start = target.global_offset
    local_experts, local_rows, local_cols = target.tensor.shape
    row_end = row_start + local_rows
    projection_start = row_base
    projection_end = row_base + intermediate_size
    overlap_start = max(row_start, projection_start)
    overlap_end = min(row_end, projection_end)
    if overlap_start >= overlap_end:
        return
    target_row_start = overlap_start - row_start
    target_row_end = overlap_end - row_start
    hf_row_start = overlap_start - projection_start
    hf_rows = overlap_end - overlap_start
    for local_expert in range(local_experts):
        expert = expert_start + local_expert
        hf_key = (
            f"model.language_model.layers.{layer}.mlp.experts.{expert}."
            f"{projection}_proj.weight"
        )
        source = _read_slice(
            reader.get_slice(hf_key),
            (hf_row_start, col_start),
            (hf_rows, local_cols),
        )
        _copy_into(
            target.tensor[local_expert, target_row_start:target_row_end],
            source,
        )
        stats.bytes_loaded += _num_bytes(source)


def _verify_selected_slices(
    model_parts: list[nn.Module],
    adapter: Any,
    reader: SafetensorsSliceReader,
    verified_keys: list[str],
) -> list[str]:
    state_dict: dict[str, torch.Tensor] = {}
    for model_part in model_parts:
        state_dict.update(_named_tensors(model_part))

    candidates = [
        "tok_embeddings.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.3.self_attn.indexer.wk.weight",
    ]
    for native_key in candidates:
        value = state_dict.get(native_key)
        if value is None:
            continue
        target = _target_for_tensor(value)
        if target.tensor.numel() == 0:
            continue
        hf_key = _native_to_hf_key(adapter, native_key)
        expected = _read_slice(
            reader.get_slice(hf_key),
            target.global_offset,
            tuple(target.tensor.shape),
        )
        torch.testing.assert_close(
            target.tensor.detach().cpu(),
            expected.to(target.tensor.dtype),
            rtol=0,
            atol=0,
        )
        verified_keys.append(native_key)

    for native_key, value in state_dict.items():
        if _is_gate_up_key(native_key):
            target = _target_for_tensor(value)
            if target.tensor.numel() == 0:
                continue
            _verify_grouped_gate_up(adapter, reader, native_key, target)
            verified_keys.append(native_key)
            break
    return verified_keys


def _verify_grouped_gate_up(
    adapter: Any,
    reader: SafetensorsSliceReader,
    native_key: str,
    target: TensorSliceTarget,
) -> None:
    expected = torch.empty_like(target.tensor, device="cpu")
    scratch = TensorSliceTarget(
        tensor=expected,
        global_shape=target.global_shape,
        global_offset=target.global_offset,
    )
    stats = ShardedLoadStats(verified_keys=[])
    _copy_grouped_gate_up(adapter, reader, native_key, scratch, stats)
    torch.testing.assert_close(target.tensor.detach().cpu(), expected, rtol=0, atol=0)


def _read_slice(
    safetensor_slice: Any,
    offset: tuple[int, ...],
    shape: tuple[int, ...],
) -> torch.Tensor:
    slices = tuple(slice(start, start + size) for start, size in zip(offset, shape))
    return safetensor_slice[slices]


def _copy_into(target: torch.Tensor, source: torch.Tensor) -> None:
    if source.dtype != target.dtype:
        source = source.to(dtype=target.dtype)
    target.copy_(source, non_blocking=True)


def _num_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _shape_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _is_lora_key(native_key: str) -> bool:
    return native_key.endswith(".lora_a") or native_key.endswith(".lora_b")


def _is_gate_up_key(native_key: str) -> bool:
    return native_key.endswith(".mlp.experts.gate_up_proj")


def _is_down_key(native_key: str) -> bool:
    return native_key.endswith(".mlp.experts.down_proj")


def _layer_from_grouped_key(native_key: str) -> int:
    parts = native_key.split(".")
    if len(parts) < 5 or parts[0] != "layers":
        raise ValueError(f"invalid grouped expert key {native_key!r}")
    return int(parts[1])
