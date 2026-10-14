# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small evidence helpers for frozen-base streaming estimates."""

from __future__ import annotations

from dataclasses import dataclass


BF16_BYTES = 2
GLM53_FLASH_BASE_PARAMS = 313_326_799_870
GLM53_FLASH_LORA_RANK16_PARAMS = 67_190_784


@dataclass(frozen=True, slots=True)
class StreamingEstimate:
    """Lower-bound bytes moved by FSDP CPU-offload parameter streaming."""

    world_size: int
    seq_len: int
    local_batch_size: int
    tokens_per_step: int
    gradient_accumulation_steps: int
    frozen_base_shard_gib: float
    trainable_lora_shard_gib: float
    weight_passes_per_microbatch: int
    h2d_gib_per_optimizer_step: float

    def h2d_bandwidth_gib_per_second(self, step_seconds: float) -> float:
        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        return self.h2d_gib_per_optimizer_step / step_seconds


def estimate_streaming_h2d(
    *,
    world_size: int,
    seq_len: int,
    local_batch_size: int,
    tokens_per_step: int,
    frozen_base_params: int = GLM53_FLASH_BASE_PARAMS,
    lora_params: int = GLM53_FLASH_LORA_RANK16_PARAMS,
    dtype_bytes: int = BF16_BYTES,
    weight_passes_per_microbatch: int = 2,
) -> StreamingEstimate:
    """Estimate FSDP CPU-offload H2D traffic for one optimizer step.

    FSDP owns the actual per-unit H2D transfers. This helper intentionally gives
    a transparent lower bound: frozen local shard bytes times forward/backward
    weight passes times gradient accumulation steps.
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if local_batch_size <= 0:
        raise ValueError("local_batch_size must be positive")
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be positive")
    if weight_passes_per_microbatch <= 0:
        raise ValueError("weight_passes_per_microbatch must be positive")
    tokens_per_accumulation = world_size * local_batch_size * seq_len
    if tokens_per_step % tokens_per_accumulation != 0:
        raise ValueError(
            "tokens_per_step must be divisible by "
            "world_size * local_batch_size * seq_len"
        )

    accumulation_steps = tokens_per_step // tokens_per_accumulation
    frozen_shard_gib = frozen_base_params * dtype_bytes / world_size / (1024**3)
    lora_shard_gib = lora_params * dtype_bytes / world_size / (1024**3)
    h2d_gib = frozen_shard_gib * weight_passes_per_microbatch * accumulation_steps
    return StreamingEstimate(
        world_size=world_size,
        seq_len=seq_len,
        local_batch_size=local_batch_size,
        tokens_per_step=tokens_per_step,
        gradient_accumulation_steps=accumulation_steps,
        frozen_base_shard_gib=frozen_shard_gib,
        trainable_lora_shard_gib=lora_shard_gib,
        weight_passes_per_microbatch=weight_passes_per_microbatch,
        h2d_gib_per_optimizer_step=h2d_gib,
    )
