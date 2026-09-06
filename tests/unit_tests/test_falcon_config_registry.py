# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchtitan.config import ConfigManager
from torchtitan.experiments.falcon.config_registry import (
    falcon_science,
    falcon_tiny_overfit,
)


def test_falcon_tiny_overfit_config_is_synchronous_bank_training():
    config = falcon_tiny_overfit()

    assert config.model_spec.name == "falcon"
    assert config.model_spec.flavor == "tiny_overfit"
    assert config.training.local_batch_size == 8
    assert config.training.seq_len == 16
    assert config.training.steps >= 80
    assert config.dataloader.num_sequences == 8
    assert config.optimizer.param_groups[0].optimizer_kwargs["weight_decay"] == 0.0


def test_config_manager_resolves_falcon_tiny_overfit():
    manager = ConfigManager()
    config = manager.parse_args(
        ["--module", "falcon", "--config", "falcon_tiny_overfit"]
    )
    assert config.model_spec.flavor == "tiny_overfit"


def test_falcon_science_config_has_locked_science_shape():
    config = falcon_science()

    assert config.model_spec.name == "falcon"
    assert config.model_spec.flavor == "science"
    inner = config.model_spec.model.config
    # Science shape locked by ticket 05/06.
    assert inner.num_hidden_layers == 4
    assert inner.hidden_size == 256
    assert inner.num_heads == 8
    assert inner.head_dim == 32
    assert inner.vocab_size == 50257
    # Delayed Falcon default mixer must not silently change.
    assert inner.mixer == "falcon"
    assert inner.alignment == "delayed"

    assert config.training.seq_len == 512
    assert config.training.local_batch_size == 32
    # 8000-step science config is registered but the smoke never runs it.
    assert config.training.steps == 8000
    # AdamW betas (0.9, 0.95) per the paper-like science setup.
    betas = config.optimizer.param_groups[0].optimizer_kwargs["betas"]
    assert tuple(betas) == (0.9, 0.95)


def test_config_manager_resolves_falcon_science():
    manager = ConfigManager()
    config = manager.parse_args(["--module", "falcon", "--config", "falcon_science"])
    assert config.model_spec.flavor == "science"
    assert config.loss.global_vocab_size == 50257


def test_falcon_science_tokens_per_step_is_16384():
    config = falcon_science()
    tokens_per_step = config.training.local_batch_size * config.training.seq_len
    assert tokens_per_step == 16384
