# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rank-local safetensors loading for the isolated GLM EP prototype."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan.experiments.glm53_flash.sharded_hf_load import (
    _copy_flattened_tensor,
    _copy_into,
    _copy_one_to_one_tensor,
    _is_lora_key,
    _named_tensors,
    _native_to_hf_key,
    _num_bytes,
    _read_slice,
    _target_for_tensor,
    SafetensorsSliceReader,
    ShardedLoadStats,
    TensorSliceTarget,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GlmEPShardedLoadStats(ShardedLoadStats):
    """Small evidence bundle including exact EP expert HF keys read."""

    expert_keys_loaded: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = ShardedLoadStats.as_dict(self)
        data["expert_keys_loaded"] = list(self.expert_keys_loaded)
        return data


def load_rank_local_ep_hf_safetensors(
    model: nn.Module | list[nn.Module],
    adapter: Any,
    checkpoint_path: str | Path,
    *,
    verify: bool = True,
) -> GlmEPShardedLoadStats:
    """Load only this rank's local GLM EP tensor slices from HF safetensors."""

    begin = time.monotonic()
    stats = GlmEPShardedLoadStats(verified_keys=[])
    model_parts = [model] if isinstance(model, nn.Module) else model
    with torch.no_grad(), SafetensorsSliceReader(checkpoint_path) as reader:
        for model_part in model_parts:
            for native_key, value in _named_tensors(model_part).items():
                if _is_lora_key(native_key):
                    stats.tensors_skipped_lora += 1
                    continue
                if native_key.endswith(".tokens_per_expert_E"):
                    continue
                target = _target_for_tensor(value)
                if target.tensor.numel() == 0:
                    continue
                if _is_ep_w13_key(native_key):
                    _copy_ep_w13(reader, native_key, target, stats)
                elif _is_ep_w2_key(native_key):
                    _copy_ep_w2(reader, native_key, target, stats)
                elif native_key.endswith(".dt_bias"):
                    hf_key = _native_to_hf_key(adapter, native_key)
                    _copy_flattened_tensor(reader, hf_key, target, stats)
                else:
                    hf_key = _native_to_hf_key(adapter, native_key)
                    _copy_one_to_one_tensor(reader, hf_key, target, stats)
                stats.tensors_loaded += 1

        if verify:
            stats.verified_keys = _verify_ep_expert_slice(model_parts, reader, stats)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    stats.elapsed_seconds = time.monotonic() - begin
    logger.info("Loaded rank-local GLM EP HF shards: %s", stats.as_dict())
    return stats


def _is_ep_w13_key(native_key: str) -> bool:
    return native_key.endswith(".mlp.experts.w13.weight")


def _is_ep_w2_key(native_key: str) -> bool:
    return native_key.endswith(".mlp.experts.w2.weight")


def _layer_from_ep_key(native_key: str) -> int:
    parts = native_key.split(".")
    if len(parts) < 6 or parts[0] != "layers":
        raise ValueError(f"invalid EP expert key {native_key!r}")
    return int(parts[1])


def _copy_ep_w13(
    reader: SafetensorsSliceReader,
    native_key: str,
    target: TensorSliceTarget,
    stats: GlmEPShardedLoadStats,
) -> None:
    layer = _layer_from_ep_key(native_key)
    expert_start, line_start, row_start, col_start = target.global_offset
    local_experts, local_lines, local_rows, local_cols = target.tensor.shape
    for local_expert in range(local_experts):
        expert = expert_start + local_expert
        for local_line in range(local_lines):
            line = line_start + local_line
            if line not in (0, 1):
                raise ValueError(f"invalid GLM EP w13 projection line {line}")
            projection = "gate" if line == 0 else "up"
            hf_key = (
                f"model.language_model.layers.{layer}.mlp.experts.{expert}."
                f"{projection}_proj.weight"
            )
            source = _read_slice(
                reader.get_slice(hf_key),
                (row_start, col_start),
                (local_rows, local_cols),
            )
            _copy_into(target.tensor[local_expert, local_line], source)
            stats.bytes_loaded += _num_bytes(source)
            stats.expert_keys_loaded.append(hf_key)


def _copy_ep_w2(
    reader: SafetensorsSliceReader,
    native_key: str,
    target: TensorSliceTarget,
    stats: GlmEPShardedLoadStats,
) -> None:
    layer = _layer_from_ep_key(native_key)
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
        stats.expert_keys_loaded.append(hf_key)


def _verify_ep_expert_slice(
    model_parts: list[nn.Module],
    reader: SafetensorsSliceReader,
    stats: GlmEPShardedLoadStats,
) -> list[str]:
    del stats
    state_dict: dict[str, torch.Tensor] = {}
    for model_part in model_parts:
        state_dict.update(_named_tensors(model_part))
    for native_key, value in state_dict.items():
        if not _is_ep_w13_key(native_key):
            continue
        target = _target_for_tensor(value)
        if target.tensor.numel() == 0:
            continue
        expected = torch.empty_like(target.tensor, device="cpu")
        scratch = TensorSliceTarget(
            tensor=expected,
            global_shape=target.global_shape,
            global_offset=target.global_offset,
        )
        scratch_stats = GlmEPShardedLoadStats(verified_keys=[])
        _copy_ep_w13(reader, native_key, scratch, scratch_stats)
        torch.testing.assert_close(
            target.tensor.detach().cpu(), expected, rtol=0, atol=0
        )
        return [native_key]
    return []
