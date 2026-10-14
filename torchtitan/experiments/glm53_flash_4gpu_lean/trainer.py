# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer wrapper that emits rank-local memory evidence for GLM lean probes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.trainer import Trainer

from .memory import (
    env_flag,
    env_int_set,
    maybe_enable_cuda_memory_history,
    write_memory_evidence,
)


class LeanMemoryTrainer(Trainer):
    """Trainer with optional CUDA memory summaries and snapshots."""

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def __init__(self, config: Config) -> None:
        maybe_enable_cuda_memory_history()
        super().__init__(config)
        if env_flag("GLM53_LEAN_RECORD_CUDA_MEMORY", False):
            write_memory_evidence(
                config.dump_folder,
                label="after_trainer_init",
                include_summary=True,
                include_snapshot=0 in env_int_set("GLM53_LEAN_MEMORY_SNAPSHOT_STEPS"),
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

    def train_step(self, data_iterator):
        if (
            env_flag("GLM53_LEAN_RECORD_CUDA_MEMORY", False)
            and self.engine.num_completed_steps == 0
        ):
            write_memory_evidence(
                self.config.dump_folder,
                label="after_checkpoint_load",
                include_summary=True,
                include_snapshot=0 in env_int_set("GLM53_LEAN_MEMORY_SNAPSHOT_STEPS"),
            )
        result = super().train_step(data_iterator)
        if env_flag("GLM53_LEAN_RECORD_CUDA_MEMORY", False):
            step = self.engine.num_completed_steps
            snapshot_steps = env_int_set("GLM53_LEAN_MEMORY_SNAPSHOT_STEPS")
            write_memory_evidence(
                self.config.dump_folder,
                label=f"after_step_{step:04d}",
                include_summary=True,
                include_snapshot=step in snapshot_steps,
            )
        return result

