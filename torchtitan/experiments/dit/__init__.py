# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)

from .model import DiTModel
from .trainer import DiTTrainer

__all__ = ["DiTModel", "DiTTrainer", "dit_configs", "model_registry"]


def _dit_debug() -> DiTModel.Config:
    return DiTModel.Config(
        latent_channels=4,
        latent_height=16,
        latent_width=16,
        patch_size=2,
        hidden_size=192,
        num_layers=4,
        num_heads=3,
        mlp_ratio=4.0,
        num_classes=10,
    )


dit_configs = {
    "debug": _dit_debug,
}


def model_registry(
    flavor: str,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> DiTModel.Config:
    config = dit_configs[flavor]()
    if converters is not None:
        validate_converter_compatibility(converters)
        for converter in converters:
            converted_config = converter.build().convert(config)
            if not isinstance(converted_config, DiTModel.Config):
                raise ValueError(
                    "DiT experiment converters must preserve DiTModel.Config."
                )
            config = converted_config
    return config
