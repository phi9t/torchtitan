# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint policy for the GLM FP8 frozen-base LoRA experiment."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.experiments.glm53_flash.checkpoint import (
    Glm53FlashLoRACheckpointManager,
)

from .sharded_hf_load import load_rank_local_fp8_from_bf16_safetensors


logger = logging.getLogger(__name__)


class Glm53FlashFP8LoRACheckpointManager(Glm53FlashLoRACheckpointManager):
    """Load the frozen HF base as FP8 qdata/scales and save only LoRA state."""

    @dataclass(kw_only=True, slots=True)
    class Config(Glm53FlashLoRACheckpointManager.Config):
        pass

    @torch.no_grad()
    def _load_frozen_base(self) -> None:
        if self._frozen_base_loaded:
            return
        if self.sd_adapter is None or self.sd_adapter.hf_assets_path is None:
            raise ValueError(
                "GLM FP8 frozen-base HF load requires a state-dict adapter and "
                "hf_assets_path."
            )

        checkpoint_id = self.sd_adapter.hf_assets_path
        logger.warning(
            "Checkpoint manifest integrity is NOT validated for Hugging Face "
            f"safetensors checkpoint {checkpoint_id}."
        )
        logger.info("Loading GLM frozen base as FP8 from %s.", checkpoint_id)
        begin = time.monotonic()

        stats = load_rank_local_fp8_from_bf16_safetensors(
            self._model_parts,
            self.sd_adapter,
            checkpoint_id,
            verify=self.verify_frozen_base_load,
        )
        self._frozen_base_loaded = True

        logger.info(
            "Finished loading GLM FP8 frozen base in %.2f seconds: %s.",
            time.monotonic() - begin,
            stats.as_dict(),
        )

