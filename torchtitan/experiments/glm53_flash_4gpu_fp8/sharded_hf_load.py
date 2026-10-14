# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rank-local BF16-to-FP8 loading for GLM-5.3-Flash frozen weights."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan.experiments.glm53_flash.sharded_hf_load import (
    _copy_flattened_tensor,
    _copy_grouped_down,
    _copy_grouped_gate_up,
    _copy_one_to_one_tensor,
    _is_down_key,
    _is_gate_up_key,
    _is_lora_key,
    _native_to_hf_key,
    _named_tensors,
    _read_slice,
    _target_for_tensor,
    SafetensorsSliceReader,
    ShardedLoadStats,
    TensorSliceTarget,
)

from .fp8 import (
    blockwise_scale_shape,
    dequantize_blockwise_128,
    FP8_DTYPE,
    quantize_into_fp8_storage_,
)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FP8ShardedLoadStats:
    """Evidence bundle for rank-local BF16-to-FP8 loading."""

    tensors_copied_bf16: int = 0
    tensors_quantized_fp8: int = 0
    tensors_skipped_lora: int = 0
    tensors_skipped_fp8_scale: int = 0
    bf16_source_bytes_read: int = 0
    fp8_qdata_bytes_written: int = 0
    scale_bytes_written: int = 0
    verified_keys: list[str] | None = None
    max_abs_dequant_error: float = 0.0
    mean_abs_dequant_error: float = 0.0
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tensors_copied_bf16": self.tensors_copied_bf16,
            "tensors_quantized_fp8": self.tensors_quantized_fp8,
            "tensors_skipped_lora": self.tensors_skipped_lora,
            "tensors_skipped_fp8_scale": self.tensors_skipped_fp8_scale,
            "bf16_source_bytes_read": self.bf16_source_bytes_read,
            "fp8_qdata_bytes_written": self.fp8_qdata_bytes_written,
            "scale_bytes_written": self.scale_bytes_written,
            "verified_keys": list(self.verified_keys or []),
            "max_abs_dequant_error": self.max_abs_dequant_error,
            "mean_abs_dequant_error": self.mean_abs_dequant_error,
            "elapsed_seconds": self.elapsed_seconds,
        }


def load_rank_local_fp8_from_bf16_safetensors(
    model: nn.Module | list[nn.Module],
    adapter: Any,
    checkpoint_path: str | Path,
    *,
    verify: bool = True,
) -> FP8ShardedLoadStats:
    """Load this rank's HF slices, quantizing eligible frozen weights to FP8."""
    begin = time.monotonic()
    stats = FP8ShardedLoadStats(verified_keys=[])
    model_parts = [model] if isinstance(model, nn.Module) else model
    with torch.no_grad(), SafetensorsSliceReader(checkpoint_path) as reader:
        for model_part in model_parts:
            tensors = _named_tensors(model_part)
            for native_key, value in tensors.items():
                if _is_lora_key(native_key):
                    stats.tensors_skipped_lora += 1
                    continue
                if _is_fp8_scale_key(native_key):
                    stats.tensors_skipped_fp8_scale += 1
                    continue
                target = _target_for_tensor(value)
                if target.tensor.numel() == 0:
                    continue
                scale_key = _scale_key_for_fp8_key(native_key)
                if (
                    target.tensor.dtype == FP8_DTYPE
                    and scale_key is not None
                    and scale_key in tensors
                ):
                    scale_target = _target_for_tensor(tensors[scale_key])
                    _load_quantized_tensor(
                        adapter,
                        reader,
                        native_key,
                        target,
                        scale_target,
                        stats,
                    )
                    stats.tensors_quantized_fp8 += 1
                    continue
                if _is_gate_up_key(native_key):
                    _copy_grouped_gate_up(
                        adapter,
                        reader,
                        native_key,
                        target,
                        _bf16_stats_proxy(stats),
                    )
                elif _is_down_key(native_key):
                    _copy_grouped_down(
                        adapter,
                        reader,
                        native_key,
                        target,
                        _bf16_stats_proxy(stats),
                    )
                elif native_key.endswith(".dt_bias"):
                    hf_key = _native_to_hf_key(adapter, native_key)
                    _copy_flattened_tensor(
                        reader,
                        hf_key,
                        target,
                        _bf16_stats_proxy(stats),
                    )
                else:
                    hf_key = _native_to_hf_key(adapter, native_key)
                    _copy_one_to_one_tensor(
                        reader,
                        hf_key,
                        target,
                        _bf16_stats_proxy(stats),
                    )
                stats.tensors_copied_bf16 += 1

            if verify:
                _verify_selected_fp8_slices(model_part, adapter, reader, stats)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    stats.elapsed_seconds = time.monotonic() - begin
    logger.info("Loaded rank-local GLM BF16->FP8 shards: %s", stats.as_dict())
    return stats


def _bf16_stats_proxy(stats: FP8ShardedLoadStats) -> ShardedLoadStats:
    class _Proxy(ShardedLoadStats):
        @property
        def bytes_loaded(self) -> int:  # type: ignore[override]
            return stats.bf16_source_bytes_read

        @bytes_loaded.setter
        def bytes_loaded(self, value: int) -> None:
            stats.bf16_source_bytes_read = value

    return _Proxy(verified_keys=[])


def _load_quantized_tensor(
    adapter: Any,
    reader: SafetensorsSliceReader,
    native_key: str,
    qdata_target: TensorSliceTarget,
    scale_target: TensorSliceTarget,
    stats: FP8ShardedLoadStats,
) -> None:
    _validate_fp8_target_alignment(native_key, qdata_target, scale_target)
    source = _read_source_tensor(adapter, reader, native_key, qdata_target)
    quantize_into_fp8_storage_(qdata_target.tensor, scale_target.tensor, source)
    stats.bf16_source_bytes_read += source.numel() * source.element_size()
    stats.fp8_qdata_bytes_written += qdata_target.tensor.numel()
    stats.scale_bytes_written += scale_target.tensor.numel() * scale_target.tensor.element_size()


def _read_source_tensor(
    adapter: Any,
    reader: SafetensorsSliceReader,
    native_key: str,
    target: TensorSliceTarget,
) -> torch.Tensor:
    source = torch.empty(
        tuple(target.tensor.shape),
        dtype=torch.bfloat16,
        device="cpu",
    )
    scratch = TensorSliceTarget(
        tensor=source,
        global_shape=target.global_shape,
        global_offset=target.global_offset,
    )
    dummy_stats = ShardedLoadStats(verified_keys=[])
    if _is_gate_up_key(native_key):
        _copy_grouped_gate_up(adapter, reader, native_key, scratch, dummy_stats)
        return source
    if _is_down_key(native_key):
        _copy_grouped_down(adapter, reader, native_key, scratch, dummy_stats)
        return source
    if native_key.endswith(".dt_bias"):
        raise ValueError(f"dt_bias must stay BF16/FP32, got FP8 target {native_key}")
    hf_key = _native_to_hf_key(adapter, native_key)
    hf_slice = reader.get_slice(hf_key)
    hf_shape = tuple(hf_slice.get_shape())
    if hf_shape != target.global_shape:
        raise ValueError(
            f"shape mismatch for {hf_key}: HF {hf_shape} vs native "
            f"{target.global_shape}"
        )
    return _read_slice(
        hf_slice,
        target.global_offset,
        tuple(target.tensor.shape),
    ).to(torch.bfloat16)


def _validate_fp8_target_alignment(
    native_key: str,
    qdata_target: TensorSliceTarget,
    scale_target: TensorSliceTarget,
) -> None:
    expected_scale_shape = blockwise_scale_shape(tuple(qdata_target.tensor.shape))
    if tuple(scale_target.tensor.shape) != expected_scale_shape:
        raise ValueError(
            f"{native_key} local scale shape mismatch: expected "
            f"{expected_scale_shape}, got {tuple(scale_target.tensor.shape)}"
        )
    for dim_offset in qdata_target.global_offset[-2:]:
        if dim_offset % 128 != 0:
            raise ValueError(
                f"{native_key} FP8 shard starts inside a 128x128 block at "
                f"offset {qdata_target.global_offset}; unsupported for "
                "quantize-at-load."
            )


def _verify_selected_fp8_slices(
    model: nn.Module,
    adapter: Any,
    reader: SafetensorsSliceReader,
    stats: FP8ShardedLoadStats,
) -> None:
    tensors = _named_tensors(model)
    for native_key in (
        "layers.0.mlp.gate_proj.weight",
        "layers.3.self_attn.q_a_proj.weight",
        "layers.3.mlp.experts.gate_up_proj",
    ):
        value = tensors.get(native_key)
        scale = tensors.get(_scale_key_for_fp8_key(native_key) or "")
        if value is None or scale is None:
            continue
        qtarget = _target_for_tensor(value)
        starget = _target_for_tensor(scale)
        if qtarget.tensor.numel() == 0:
            continue
        expected = _read_source_tensor(adapter, reader, native_key, qtarget).float()
        actual = dequantize_blockwise_128(
            qtarget.tensor.detach().cpu(),
            starget.tensor.detach().cpu(),
            dtype=torch.float32,
        )
        diff = (actual - expected).abs()
        stats.max_abs_dequant_error = max(
            stats.max_abs_dequant_error,
            float(diff.max().item()) if diff.numel() else 0.0,
        )
        stats.mean_abs_dequant_error = max(
            stats.mean_abs_dequant_error,
            float(diff.mean().item()) if diff.numel() else 0.0,
        )
        assert stats.verified_keys is not None
        stats.verified_keys.append(native_key)


def _scale_key_for_fp8_key(native_key: str) -> str | None:
    if native_key.endswith(".weight"):
        return native_key.removesuffix(".weight") + ".weight_scale"
    if _is_gate_up_key(native_key):
        return native_key + "_scale"
    if _is_down_key(native_key):
        return native_key + "_scale"
    return None


def _is_fp8_scale_key(native_key: str) -> bool:
    return native_key.endswith(".weight_scale") or native_key.endswith("_proj_scale")

