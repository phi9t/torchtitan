# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchtitan.config import ConfigManager
from torchtitan.experiments.falcon.config_registry import (
    _ARM_KNOBS,
    apply_arm,
    apply_mechanism_arm,
    falcon_science,
    falcon_tiny_overfit,
)
from torchtitan.experiments.falcon.model import FalconConfig, FalconForCausalLM


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
    assert inner.qk_norm_eps == 1.0e-6
    assert inner.nlms_denom_eps == 1.0e-6
    assert inner.scale_compensation == "none"

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


def test_apply_mechanism_arm_materializes_m0_to_m4_without_changing_a_arms():
    expected = {
        "M0": ("delayed", "rms", "none"),
        "M1": ("same_step", "rms", "none"),
        "M2": ("delayed", "l2", "none"),
        "M3": ("same_step", "l2", "none"),
        "M4": ("delayed", "rms", "rms_to_l2"),
    }
    original_a_arms = {key: value.copy() for key, value in _ARM_KNOBS.items()}

    for arm_id, (alignment, phi, scale_compensation) in expected.items():
        config = apply_mechanism_arm(falcon_science(), arm_id)
        inner = config.model_spec.model.config
        assert inner.mixer == "falcon"
        assert inner.variant == "falcon1a"
        assert inner.alignment == alignment
        assert inner.phi == phi
        assert inner.scale_compensation == scale_compensation

    assert _ARM_KNOBS == original_a_arms


def test_apply_arm_preserves_legacy_a5_without_scale_compensation():
    config = apply_arm(falcon_science(), "A5")
    inner = config.model_spec.model.config

    assert inner.mixer == "falcon"
    assert inner.variant == "falcon1a"
    assert inner.alignment == "delayed"
    assert inner.phi == "l2"
    assert inner.scale_compensation == "none"


def test_apply_mechanism_arm_rejects_unknown_arm():
    try:
        apply_mechanism_arm(falcon_science(), "M9")
    except ValueError as exc:
        assert "mechanism arm" in str(exc)
    else:
        raise AssertionError("expected unknown mechanism arm to fail")


def test_falcon_config_rejects_invalid_scale_compensation_pairings():
    config = FalconConfig(mixer="gdn", scale_compensation="rms_to_l2")
    try:
        FalconForCausalLM(config)
    except ValueError as exc:
        assert "mixer" in str(exc)
    else:
        raise AssertionError("expected non-Falcon compensation to fail")

    config = FalconConfig(phi="l2", scale_compensation="rms_to_l2")
    try:
        FalconForCausalLM(config)
    except ValueError as exc:
        assert "phi" in str(exc)
    else:
        raise AssertionError("expected non-RMS compensation to fail")


def test_falcon_config_rejects_invalid_eps_with_exact_field_names():
    config = FalconConfig(qk_norm_eps=-1.0)
    try:
        FalconForCausalLM(config)
    except ValueError as exc:
        assert "qk_norm_eps" in str(exc)
    else:
        raise AssertionError("expected invalid qk_norm_eps to fail")

    config = FalconConfig(nlms_denom_eps=float("inf"))
    try:
        FalconForCausalLM(config)
    except ValueError as exc:
        assert "nlms_denom_eps" in str(exc)
    else:
        raise AssertionError("expected invalid nlms_denom_eps to fail")
