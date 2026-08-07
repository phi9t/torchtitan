# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import MSELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import TrainingConfig

from . import model_registry
from .dataloader import SyntheticLatentDataLoader
from .model import DiTModel
from .trainer import DiTTrainer


def dit_debug_synthetic() -> DiTTrainer.Config:
    model_spec = model_registry("debug")
    assert isinstance(model_spec.model, DiTModel.Config)
    model_config = model_spec.model
    return DiTTrainer.Config(
        model_spec=model_spec,
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
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=model_config.latent_patch_tokens,
            steps=10,
            dtype="float32",
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=None,
        diffusion_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
    )
