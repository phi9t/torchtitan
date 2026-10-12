# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import MSELoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.observability.metrics import MetricsProcessor

from . import model_registry
from .dataloader import SyntheticLatentDataLoader
from .model import DiTModel
from .trainer import DiTTrainer


def dit_debug_synthetic() -> DiTTrainer.Config:
    model_config = model_registry("debug")
    assert isinstance(model_config, DiTModel.Config)
    return DiTTrainer.Config(
        model=model_config,
        hf_assets_path="./tests/assets/tokenizer",
        dump_folder="./outputs/dit_debug_synthetic",
        loss=MSELoss.Config(),
        dataloader=SyntheticLatentDataLoader.Config(
            latent_channels=model_config.latent_channels,
            latent_height=model_config.latent_height,
            latent_width=model_config.latent_width,
            num_classes=model_config.num_classes,
            seed=17,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[
                    AdamW.Config(
                        pattern=r".*",
                        lr=8e-4,
                    ),
                ],
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8
            * model_config.latent_patch_tokens,
            max_context_length=model_config.latent_patch_tokens,
            steps=10,
            dtype="float32",
        ),
        checkpointer=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=None,
        diffusion_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
    )
