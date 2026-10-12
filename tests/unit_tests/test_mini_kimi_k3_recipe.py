# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from torchtitan.experiments.mini_kimi_k3.recipe import mini_k3_r1_launch_recipe


def test_r1_launch_recipe_matches_first_party_token_budget():
    recipe = mini_k3_r1_launch_recipe()

    assert recipe.model_variant == "r1"
    assert recipe.target_tokens == 5_000_000_000
    assert recipe.seq_len == 4096
    assert recipe.local_batch_size == 4
    assert recipe.gradient_accumulation_steps == 8
    assert recipe.world_size == 1
    assert recipe.tokens_per_step == 131_072
    assert recipe.optimizer_steps == 38_147
    assert recipe.planned_tokens == 5_000_003_584


def test_r1_launch_recipe_records_optimizer_schedule_contract():
    recipe = mini_k3_r1_launch_recipe()

    assert recipe.peak_lr == pytest.approx(6.0e-4)
    assert recipe.optimizer == "AdamW"
    assert recipe.adam_betas == (0.9, 0.95)
    assert recipe.adam_eps == pytest.approx(1.0e-8)
    assert recipe.weight_decay == pytest.approx(0.1)
    assert recipe.excludes_router_parameters_from_optimizer
    assert recipe.disables_weight_decay_for_1d_parameters
    assert recipe.warmup_fraction == pytest.approx(0.02)
    assert recipe.decay_fraction == pytest.approx(0.15)
    assert recipe.min_lr_fraction == pytest.approx(0.1)
    assert recipe.checkpoint_interval == 100
    assert recipe.dtype == "bfloat16"
    assert recipe.requires_gradient_checkpointing


def test_r1_launch_recipe_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="seq_len"):
        mini_k3_r1_launch_recipe(seq_len=0)

    with pytest.raises(ValueError, match="local_batch_size"):
        mini_k3_r1_launch_recipe(local_batch_size=0)

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        mini_k3_r1_launch_recipe(gradient_accumulation_steps=0)
