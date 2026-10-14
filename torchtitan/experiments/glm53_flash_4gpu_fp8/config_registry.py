# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan configs for 4-GPU GLM FP8 frozen-base LoRA SFT."""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from torchtitan.experiments.glm53_flash import config_registry as bf16_registry
from torchtitan.experiments.glm53_flash.model import Glm53FlashForCausalLM

from .checkpoint import Glm53FlashFP8LoRACheckpointManager
from .model import Glm53FlashFP8ForCausalLM


def model_registry(flavor: str) -> Glm53FlashFP8ForCausalLM.Config:
    base = bf16_registry.model_registry(flavor)
    assert isinstance(base, Glm53FlashForCausalLM.Config)
    return _fp8_model_config(base)


def glm53_flash_debugmodel_lora():
    config = bf16_registry.glm53_flash_debugmodel_lora()
    config.model = _fp8_model_config(config.model)
    config.dump_folder = os.environ.get(
        "GLM53_FLASH_FP8_LORA_DUMP_FOLDER",
        "./outputs/glm53_flash_4gpu_fp8_debugmodel_lora",
    )
    return config


def glm53_flash_mini_countdown_lora():
    config = bf16_registry.glm53_flash_mini_countdown_lora()
    config.model = _fp8_model_config(config.model)
    config.dump_folder = os.environ.get(
        "GLM53_FLASH_FP8_MINI_DUMP_FOLDER",
        "./outputs/glm53_flash_4gpu_fp8_mini_countdown_lora",
    )
    return config


def glm53_flash_countdown_lora():
    config = bf16_registry.glm53_flash_countdown_lora()
    config.model = _fp8_model_config(config.model)
    config.checkpointer = _fp8_checkpointer_config(config.checkpointer)
    config.dump_folder = os.environ.get(
        "GLM53_FLASH_FP8_FULL_DUMP_FOLDER",
        "./outputs/glm53_flash_4gpu_fp8_countdown_lora",
    )
    return config


def glm53_flash_countdown_lora_truncated_fp8():
    """Full-width, first-N-layer FP8 config for T09b parity probes."""

    return _truncated_countdown_config(fp8=True)


def glm53_flash_countdown_lora_truncated_plain():
    """Full-width, first-N-layer BF16 baseline for T09b parity probes."""

    return _truncated_countdown_config(fp8=False)


def _truncated_countdown_config(*, fp8: bool):
    num_layers = bf16_registry._positive_env_int("GLM53_FP8_NUM_LAYERS", 8)
    layer_offset = bf16_registry._nonnegative_env_int("GLM53_FP8_LAYER_OFFSET", 0)
    if layer_offset != 0:
        raise ValueError(
            "GLM53_FP8_LAYER_OFFSET must stay 0: the inherited rank-local HF "
            "loader maps native layer ids directly to HF layer ids."
        )

    config = bf16_registry.glm53_flash_countdown_lora()
    assert isinstance(config.model, Glm53FlashForCausalLM.Config)
    glm_config = dataclasses.replace(
        config.model.glm_config,
        num_hidden_layers=num_layers,
        layer_offset=layer_offset,
    )
    glm_config.validate()

    flavor = f"glm53_flash_first_{num_layers}"
    plain_model_config = Glm53FlashForCausalLM.Config(
        glm_config=glm_config,
        experiment_flavor=flavor if fp8 else f"{flavor}_plain",
    )
    if fp8:
        config.model = _fp8_model_config(plain_model_config)
        config.checkpointer = _fp8_checkpointer_config(config.checkpointer)
    else:
        config.model = plain_model_config
    config.training.steps = bf16_registry._positive_env_int("GLM53_FP8_STEPS", 3)
    config.optim = bf16_registry._lora_optim_config(
        steps=config.training.steps,
        lr=bf16_registry._nonnegative_env_float("GLM53_FLASH_FULL_LR", 1.0e-4),
    )
    config.dump_folder = os.environ.get(
        "GLM53_FP8_DUMP_FOLDER",
        os.environ.get(
            "GLM53_FLASH_FULL_DUMP_FOLDER",
            (
                ".scratch/glm53-flash-sft-ladder/evidence/task-09b/"
                f"truncated_{'fp8' if fp8 else 'plain'}"
            ),
        ),
    )
    return config


def _fp8_model_config(model_config: Any) -> Glm53FlashFP8ForCausalLM.Config:
    if not isinstance(model_config, Glm53FlashForCausalLM.Config):
        raise TypeError(
            "GLM FP8 configs must wrap a Glm53FlashForCausalLM.Config, got "
            f"{type(model_config).__qualname__}."
        )
    return Glm53FlashFP8ForCausalLM.Config(
        glm_config=model_config.glm_config,
        experiment_flavor=model_config.experiment_flavor,
        local_compile_regions=list(model_config.local_compile_regions),
    )


def _fp8_checkpointer_config(checkpointer_config):
    if checkpointer_config is None:
        return None
    values = {
        field.name: getattr(checkpointer_config, field.name)
        for field in dataclasses.fields(Glm53FlashFP8LoRACheckpointManager.Config)
        if hasattr(checkpointer_config, field.name)
    }
    return Glm53FlashFP8LoRACheckpointManager.Config(**values)

