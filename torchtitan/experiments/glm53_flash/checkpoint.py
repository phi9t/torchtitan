# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5.3-Flash LoRA checkpoint policy."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from torchtitan.components.checkpointer import MODEL, ModelWrapper
from torchtitan.components.checkpointer.dcp import CheckpointManager

from .sharded_hf_load import load_rank_local_hf_safetensors

logger = logging.getLogger(__name__)


def _trainable_parameter_names(model: nn.Module | list[nn.Module]) -> set[str]:
    model_parts = [model] if isinstance(model, nn.Module) else model
    return {
        name
        for model_part in model_parts
        for name, param in model_part.named_parameters()
        if param.requires_grad
    }


class TrainableModelWrapper(ModelWrapper):
    """Stateful wrapper exposing only trainable model parameters."""

    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.trainable_parameter_names = _trainable_parameter_names(model)
        super().__init__(model)

    def _get_state_dict(self) -> dict[str, Any]:
        return {
            key: value
            for model in self.model
            for key, value in model.state_dict().items()
            if key in self.trainable_parameter_names
        }


class Glm53FlashLoRACheckpointManager(CheckpointManager):
    """Reload the frozen HF base, but save only LoRA model tensors."""

    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        load_frozen_base_from_hf: bool = True
        trainable_model_only: bool = True
        verify_frozen_base_load: bool = True

    def __init__(self, config: Config, **kwargs: Any) -> None:
        model_parts = kwargs["model_parts"]
        super().__init__(config, **kwargs)
        self.load_frozen_base_from_hf = config.load_frozen_base_from_hf
        self.verify_frozen_base_load = config.verify_frozen_base_load
        self._model_parts = model_parts
        self._frozen_base_loaded = False
        if config.trainable_model_only:
            self.states[MODEL] = TrainableModelWrapper(model_parts)

    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        if self.load_frozen_base_from_hf:
            self._load_frozen_base()
        return super().load(step=step)

    def _load_frozen_base(self) -> None:
        if self._frozen_base_loaded:
            return
        if self.sd_adapter is None or self.sd_adapter.hf_assets_path is None:
            raise ValueError(
                "GLM frozen-base HF load requires a state-dict adapter and "
                "hf_assets_path."
            )

        checkpoint_id = self.sd_adapter.hf_assets_path
        logger.warning(
            "Checkpoint manifest integrity is NOT validated for Hugging Face "
            f"safetensors checkpoint {checkpoint_id}."
        )
        logger.info("Loading GLM frozen base from %s.", checkpoint_id)
        begin = time.monotonic()

        stats = load_rank_local_hf_safetensors(
            self._model_parts,
            self.sd_adapter,
            checkpoint_id,
            verify=self.verify_frozen_base_load,
        )
        self._frozen_base_loaded = True

        logger.info(
            "Finished loading GLM frozen base in %.2f seconds: %s.",
            time.monotonic() - begin,
            stats.as_dict(),
        )
