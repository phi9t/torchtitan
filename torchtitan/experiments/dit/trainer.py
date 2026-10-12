# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.trainer import Trainer

from .dataloader import SyntheticLatentDataLoader
from .model import DiTModel


class DiTTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        dataloader: SyntheticLatentDataLoader.Config
        diffusion_timesteps: int = 1000
        beta_start: float = 0.0001
        beta_end: float = 0.02

    def __init__(self, config: Config):
        model_config = config.model
        assert isinstance(model_config, DiTModel.Config)
        config.training.max_context_length = model_config.latent_patch_tokens
        super().__init__(config)

        betas_T = torch.linspace(
            config.beta_start,
            config.beta_end,
            config.diffusion_timesteps,
            device=self.device,
            dtype=torch.float32,
        )
        self.alpha_bar_T = torch.cumprod(1.0 - betas_T, dim=0)
        self.engine.preprocess_inputs_kwargs = {
            "alpha_bar": self.alpha_bar_T,
            "diffusion_timesteps": self.config.diffusion_timesteps,
            "training_dtype": self.config.training.dtype,
        }

    def train_step(self, data_iterator):
        if self.gradient_accumulation_steps > 1:
            raise ValueError("DiT synthetic experiment does not support grad accum.")
        return super().train_step(data_iterator)
