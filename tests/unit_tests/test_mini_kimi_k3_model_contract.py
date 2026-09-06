# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

import pytest

from torchtitan.experiments.mini_kimi_k3.model_contract import (
    compare_with_first_party_oracle,
    estimate_parameter_budget,
    mini_k3_flagship_config,
    mini_k3_r1_config,
    mini_k3_r1_parameter_targets,
    MiniK3Config,
    to_kimi_linear_config_dict,
)


def test_r1_config_matches_book_ladder_shape():
    config = mini_k3_r1_config()

    assert config.name == "r1"
    assert config.hidden_size == 512
    assert config.num_hidden_layers == 12
    assert config.num_attention_heads == 8
    assert config.num_experts == 256
    assert config.num_experts_per_token == 6
    assert config.moe_intermediate_size == 416
    assert config.linear_attn_num_heads == 4
    assert config.vocab_size == 163840
    assert config.max_position_embeddings == 4096
    assert config.full_attn_layers == (4, 8, 12)
    assert config.kda_layers == (1, 2, 3, 5, 6, 7, 9, 10, 11)


def test_flagship_config_matches_first_party_config_mini_shape():
    config = mini_k3_flagship_config()

    assert config.name == "flagship"
    assert config.hidden_size == 1536
    assert config.num_hidden_layers == 24
    assert config.num_attention_heads == 16
    assert config.num_experts == 512
    assert config.num_experts_per_token == 6
    assert config.num_shared_experts == 2
    assert config.moe_intermediate_size == 1280
    assert config.routed_expert_hidden_size == 768
    assert config.linear_attn_num_heads == 8
    assert config.full_attn_layers == (4, 8, 12, 16, 20, 24)
    assert config.kda_layers == (
        1,
        2,
        3,
        5,
        6,
        7,
        9,
        10,
        11,
        13,
        14,
        15,
        17,
        18,
        19,
        21,
        22,
        23,
    )


def test_config_rejects_overlapping_attention_layer_sets():
    with pytest.raises(ValueError, match="overlap"):
        MiniK3Config(
            name="bad",
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_experts=8,
            num_experts_per_token=2,
            moe_intermediate_size=64,
            routed_expert_hidden_size=64,
            linear_attn_num_heads=2,
            full_attn_layers=(2, 4),
            kda_layers=(1, 2, 3),
        )


def test_config_rejects_missing_attention_layer_coverage():
    with pytest.raises(ValueError, match="cover every layer"):
        MiniK3Config(
            name="bad",
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_experts=8,
            num_experts_per_token=2,
            moe_intermediate_size=64,
            routed_expert_hidden_size=64,
            linear_attn_num_heads=2,
            full_attn_layers=(4,),
            kda_layers=(1, 2),
        )


def test_config_estimates_active_routed_expert_parameters():
    config = mini_k3_r1_config()

    assert config.routed_expert_parameter_fraction == pytest.approx(6 / 256)
    assert config.routed_expert_parameters_per_layer == (2 * 512 * 416 * 256)
    assert config.active_routed_expert_parameters_per_layer == pytest.approx(
        2 * 512 * 416 * 6
    )


def test_r1_parameter_targets_record_source_claims():
    targets = mini_k3_r1_parameter_targets()

    assert targets.total_parameters == 1_020_000_000
    assert targets.active_parameters == 145_000_000
    assert targets.active_nonembedding_parameters == 61_000_000
    assert targets.source == "first_party_r1_ladder_claim"


def test_r1_parameter_estimate_uses_first_party_count_semantics():
    estimate = estimate_parameter_budget(mini_k3_r1_config())

    assert estimate.total_parameters == 1_023_206_628
    assert estimate.routed_expert_parameters == 899_678_208
    assert estimate.embedding_parameters == 83_886_080
    assert estimate.active_routed_expert_parameters == pytest.approx(21_086_208)
    assert estimate.active_parameters == pytest.approx(144_614_628)
    assert estimate.active_nonembedding_parameters == pytest.approx(60_728_548)
    assert estimate.sparsity == pytest.approx(7.0754019987521595)
    assert estimate.routed_share == pytest.approx(0.14580964797005183)


def test_r1_parameter_budget_matches_source_claims_within_tolerance():
    estimate = estimate_parameter_budget(mini_k3_r1_config())

    assert estimate.budget_status == "pass"
    assert estimate.target_total_parameters == 1_020_000_000
    assert estimate.total_delta_parameters == 3_206_628
    assert estimate.active_delta_parameters == pytest.approx(-385_372)
    assert estimate.active_nonembedding_delta_parameters == pytest.approx(-271_452)
    assert "matches" in estimate.budget_detail
    assert estimate.total_status == "pass"
    assert estimate.active_status == "pass"
    assert estimate.active_nonembedding_status == "pass"


def test_r1_config_exports_first_party_kimi_linear_shape():
    exported = to_kimi_linear_config_dict(mini_k3_r1_config())

    assert exported["hidden_size"] == 512
    assert exported["num_hidden_layers"] == 12
    assert exported["num_attention_heads"] == 8
    assert exported["num_key_value_heads"] == 8
    assert exported["qk_nope_head_dim"] == 128
    assert exported["qk_rope_head_dim"] == 64
    assert exported["v_head_dim"] == 128
    assert exported["kv_lora_rank"] == 64
    assert exported["q_lora_rank"] == 128
    assert exported["mla_use_nope"] is True
    assert exported["mla_use_output_gate"] is True
    assert exported["hidden_act"] == "situ"
    assert exported["topk_method"] == "noaux_tc"
    assert exported["linear_attn_config"] == {
        "head_dim": 128,
        "num_heads": 4,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "use_full_rank_gate": True,
        "full_attn_layers": [4, 8, 12],
        "kda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11],
    }


def test_flagship_config_export_matches_checked_config_mini_fields():
    exported = to_kimi_linear_config_dict(mini_k3_flagship_config())

    assert exported["hidden_size"] == 1536
    assert exported["num_hidden_layers"] == 24
    assert exported["kv_lora_rank"] == 128
    assert exported["q_lora_rank"] == 384
    assert exported["attn_res_block_size"] == 3
    assert exported["linear_attn_config"]["num_heads"] == 8
    assert exported["linear_attn_config"]["full_attn_layers"] == [4, 8, 12, 16, 20, 24]


def test_first_party_oracle_comparison_accepts_matching_files(tmp_path):
    oracle = tmp_path / "oracle"
    (oracle / "model").mkdir(parents=True)
    (oracle / "train").mkdir(parents=True)
    (oracle / "model" / "config_mini.json").write_text(
        json.dumps(
            to_kimi_linear_config_dict(mini_k3_flagship_config()), sort_keys=True
        )
    )
    (oracle / "train" / "ladder_r1.json").write_text(
        json.dumps(to_kimi_linear_config_dict(mini_k3_r1_config()), sort_keys=True)
    )

    result = compare_with_first_party_oracle(
        oracle,
        r1_config_path=oracle / "train" / "ladder_r1.json",
    )

    assert result == {
        "flagship": "match",
        "r1": "match",
    }


def test_first_party_oracle_comparison_reports_mismatch(tmp_path):
    oracle = tmp_path / "oracle"
    (oracle / "model").mkdir(parents=True)
    (oracle / "train").mkdir(parents=True)
    flagship = to_kimi_linear_config_dict(mini_k3_flagship_config())
    r1 = to_kimi_linear_config_dict(mini_k3_r1_config())
    r1["hidden_size"] = 768
    (oracle / "model" / "config_mini.json").write_text(json.dumps(flagship))
    (oracle / "train" / "ladder_r1.json").write_text(json.dumps(r1))

    result = compare_with_first_party_oracle(
        oracle,
        r1_config_path=oracle / "train" / "ladder_r1.json",
    )

    assert result["flagship"] == "match"
    assert result["r1"].startswith("mismatch:")
