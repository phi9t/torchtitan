# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mini Kimi K3 launch-recipe contracts.

These values describe the source recipe we intend to reproduce. They are not a
TorchTitan ``Trainer.Config`` yet, because the Mini-K3 KDA/MLA/SITU model path is
not implemented in this checkout.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MiniK3LaunchRecipe:
    model_variant: str
    target_tokens: int
    seq_len: int
    local_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    peak_lr: float
    optimizer: str
    adam_betas: tuple[float, float]
    adam_eps: float
    weight_decay: float
    excludes_router_parameters_from_optimizer: bool
    disables_weight_decay_for_1d_parameters: bool
    warmup_fraction: float
    decay_fraction: float
    min_lr_fraction: float
    checkpoint_interval: int
    dtype: str
    requires_gradient_checkpointing: bool

    def __post_init__(self) -> None:
        _require_positive("target_tokens", self.target_tokens)
        _require_positive("seq_len", self.seq_len)
        _require_positive("local_batch_size", self.local_batch_size)
        _require_positive(
            "gradient_accumulation_steps", self.gradient_accumulation_steps
        )
        _require_positive("world_size", self.world_size)
        _require_positive("checkpoint_interval", self.checkpoint_interval)

    @property
    def tokens_per_step(self) -> int:
        return (
            self.seq_len
            * self.local_batch_size
            * self.gradient_accumulation_steps
            * self.world_size
        )

    @property
    def optimizer_steps(self) -> int:
        return max(1, round(self.target_tokens / self.tokens_per_step))

    @property
    def planned_tokens(self) -> int:
        return self.optimizer_steps * self.tokens_per_step

    def to_dict(self) -> dict[str, object]:
        return {
            "model_variant": self.model_variant,
            "target_tokens": self.target_tokens,
            "seq_len": self.seq_len,
            "local_batch_size": self.local_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "world_size": self.world_size,
            "tokens_per_step": self.tokens_per_step,
            "optimizer_steps": self.optimizer_steps,
            "planned_tokens": self.planned_tokens,
            "peak_lr": self.peak_lr,
            "optimizer": self.optimizer,
            "adam_betas": self.adam_betas,
            "adam_eps": self.adam_eps,
            "weight_decay": self.weight_decay,
            "excludes_router_parameters_from_optimizer": (
                self.excludes_router_parameters_from_optimizer
            ),
            "disables_weight_decay_for_1d_parameters": (
                self.disables_weight_decay_for_1d_parameters
            ),
            "warmup_fraction": self.warmup_fraction,
            "decay_fraction": self.decay_fraction,
            "min_lr_fraction": self.min_lr_fraction,
            "checkpoint_interval": self.checkpoint_interval,
            "dtype": self.dtype,
            "requires_gradient_checkpointing": self.requires_gradient_checkpointing,
        }


def mini_k3_r1_launch_recipe(
    *,
    seq_len: int = 4096,
    local_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    world_size: int = 1,
    target_tokens: int = 5_000_000_000,
) -> MiniK3LaunchRecipe:
    return MiniK3LaunchRecipe(
        model_variant="r1",
        target_tokens=target_tokens,
        seq_len=seq_len,
        local_batch_size=local_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        world_size=world_size,
        peak_lr=6.0e-4,
        optimizer="AdamW",
        adam_betas=(0.9, 0.95),
        adam_eps=1.0e-8,
        weight_decay=0.1,
        excludes_router_parameters_from_optimizer=True,
        disables_weight_decay_for_1d_parameters=True,
        warmup_fraction=0.02,
        decay_fraction=0.15,
        min_lr_fraction=0.1,
        checkpoint_interval=100,
        dtype="bfloat16",
        requires_gradient_checkpointing=True,
    )


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
