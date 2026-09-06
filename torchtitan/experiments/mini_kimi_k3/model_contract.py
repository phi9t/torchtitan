# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mini Kimi K3 architecture contracts used before model implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class MiniK3ParameterTargets:
    source: str
    total_parameters: int
    active_parameters: int
    active_nonembedding_parameters: int
    tolerance_fraction: float


@dataclass(frozen=True)
class MiniK3ParameterEstimate:
    total_parameters: int
    routed_expert_parameters: int
    embedding_parameters: int
    active_routed_expert_parameters: float
    active_parameters: float
    active_nonembedding_parameters: float
    sparsity: float
    routed_share: float
    target_total_parameters: int
    target_active_parameters: int
    target_active_nonembedding_parameters: int
    tolerance_fraction: float

    @property
    def total_delta_parameters(self) -> float:
        return self.total_parameters - self.target_total_parameters

    @property
    def active_delta_parameters(self) -> float:
        return self.active_parameters - self.target_active_parameters

    @property
    def active_nonembedding_delta_parameters(self) -> float:
        return (
            self.active_nonembedding_parameters
            - self.target_active_nonembedding_parameters
        )

    @property
    def total_status(self) -> str:
        return _budget_status(
            measured=self.total_parameters,
            target=self.target_total_parameters,
            tolerance_fraction=self.tolerance_fraction,
        )

    @property
    def active_status(self) -> str:
        return _budget_status(
            measured=self.active_parameters,
            target=self.target_active_parameters,
            tolerance_fraction=self.tolerance_fraction,
        )

    @property
    def active_nonembedding_status(self) -> str:
        return _budget_status(
            measured=self.active_nonembedding_parameters,
            target=self.target_active_nonembedding_parameters,
            tolerance_fraction=self.tolerance_fraction,
        )

    @property
    def budget_status(self) -> str:
        statuses = {
            self.total_status,
            self.active_status,
            self.active_nonembedding_status,
        }
        return "pass" if statuses == {"pass"} else "fail"

    @property
    def budget_detail(self) -> str:
        failures = []
        if self.total_status != "pass":
            failures.append(
                "total parameters "
                f"{self.total_parameters} != {self.target_total_parameters}"
            )
        if self.active_status != "pass":
            failures.append(
                "active parameters "
                f"{self.active_parameters:.0f} != {self.target_active_parameters}"
            )
        if self.active_nonembedding_status != "pass":
            failures.append(
                "active non-embedding parameters "
                f"{self.active_nonembedding_parameters:.0f} != "
                f"{self.target_active_nonembedding_parameters}"
            )
        if not failures:
            return "parameter budget matches the source-derived r1 claim"
        return "; ".join(failures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.budget_status,
            "detail": self.budget_detail,
            "counts": {
                "total_parameters": self.total_parameters,
                "routed_expert_parameters": self.routed_expert_parameters,
                "embedding_parameters": self.embedding_parameters,
                "active_routed_expert_parameters": self.active_routed_expert_parameters,
                "active_parameters": self.active_parameters,
                "active_nonembedding_parameters": self.active_nonembedding_parameters,
                "sparsity": self.sparsity,
                "routed_share": self.routed_share,
            },
            "targets": {
                "total_parameters": self.target_total_parameters,
                "active_parameters": self.target_active_parameters,
                "active_nonembedding_parameters": (
                    self.target_active_nonembedding_parameters
                ),
                "tolerance_fraction": self.tolerance_fraction,
            },
            "deltas": {
                "total_parameters": self.total_delta_parameters,
                "active_parameters": self.active_delta_parameters,
                "active_nonembedding_parameters": (
                    self.active_nonembedding_delta_parameters
                ),
            },
        }


@dataclass(frozen=True)
class MiniK3Config:
    name: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_experts: int
    num_experts_per_token: int
    moe_intermediate_size: int
    routed_expert_hidden_size: int
    linear_attn_num_heads: int
    full_attn_layers: tuple[int, ...]
    kda_layers: tuple[int, ...]
    vocab_size: int = 163840
    num_key_value_heads: int | None = None
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    kv_lora_rank: int = 128
    q_lora_rank: int = 384
    num_shared_experts: int = 2
    first_k_dense_replace: int = 1
    intermediate_size: int = 7168
    hidden_act: str = "situ"
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = True
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True
    latent_moe_use_norm: bool = True
    moe_renormalize: bool = True
    num_expert_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    moe_layer_freq: int = 1
    attn_res_block_size: int | None = None
    rms_norm_eps: float = 1.0e-5
    num_nextn_predict_layers: int = 0
    initializer_range: float = 0.02
    linear_attn_head_dim: int = 128
    short_conv_kernel_size: int = 4
    gate_lower_bound: float = -5.0
    use_full_rank_gate: bool = True
    use_fla_kda: bool = False
    topk_method: str = "noaux_tc"
    moe_router_activation_func: str = "sigmoid"

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be > 0")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if self.num_experts_per_token <= 0:
            raise ValueError("num_experts_per_token must be > 0")
        if self.num_experts_per_token > self.num_experts:
            raise ValueError("num_experts_per_token must be <= num_experts")
        if self.hidden_act != "situ":
            raise ValueError("Mini-K3 contract currently requires SITU activation")
        if self.topk_method != "noaux_tc":
            raise ValueError("Mini-K3 contract currently requires noaux_tc routing")

        full = set(self.full_attn_layers)
        kda = set(self.kda_layers)
        overlap = full & kda
        if overlap:
            raise ValueError(f"attention layer sets overlap: {sorted(overlap)}")
        expected = set(range(1, self.num_hidden_layers + 1))
        covered = full | kda
        if covered != expected:
            missing = sorted(expected - covered)
            extra = sorted(covered - expected)
            raise ValueError(
                "full_attn_layers and kda_layers must cover every layer exactly "
                f"once; missing={missing}, extra={extra}"
            )

    @property
    def effective_num_key_value_heads(self) -> int:
        return (
            self.num_attention_heads
            if self.num_key_value_heads is None
            else self.num_key_value_heads
        )

    @property
    def routed_expert_parameter_fraction(self) -> float:
        return self.num_experts_per_token / self.num_experts

    @property
    def routed_expert_parameters_per_layer(self) -> int:
        return 2 * self.hidden_size * self.moe_intermediate_size * self.num_experts

    @property
    def active_routed_expert_parameters_per_layer(self) -> float:
        return (
            self.routed_expert_parameters_per_layer
            * self.routed_expert_parameter_fraction
        )

    @property
    def effective_attn_res_block_size(self) -> int:
        return (
            max(1, self.num_hidden_layers // 8)
            if self.attn_res_block_size is None
            else self.attn_res_block_size
        )


def mini_k3_r1_parameter_targets() -> MiniK3ParameterTargets:
    """Return the source-derived r1 parameter claims.

    The first-party ladder describes r1 as "1.02B total / 145M active"; its
    handoff table additionally records 61M active non-embedding parameters.
    """

    return MiniK3ParameterTargets(
        source="first_party_r1_ladder_claim",
        total_parameters=1_020_000_000,
        active_parameters=145_000_000,
        active_nonembedding_parameters=61_000_000,
        tolerance_fraction=0.05,
    )


def estimate_parameter_budget(
    config: MiniK3Config,
    *,
    targets: MiniK3ParameterTargets | None = None,
) -> MiniK3ParameterEstimate:
    """Estimate source-compatible total and active parameters on meta tensors."""

    from torchtitan.experiments.mini_kimi_k3.model import MiniK3ForCausalLM

    if targets is None:
        targets = mini_k3_r1_parameter_targets()
    with torch.device("meta"):
        model = MiniK3ForCausalLM(config)

    total = 0
    routed_expert = 0
    embedding = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if ".experts." in name:
            routed_expert += count
        if "embed_tokens" in name or "lm_head" in name:
            embedding += count

    active_routed = routed_expert * (config.num_experts_per_token / config.num_experts)
    active = total - routed_expert + active_routed
    return MiniK3ParameterEstimate(
        total_parameters=total,
        routed_expert_parameters=routed_expert,
        embedding_parameters=embedding,
        active_routed_expert_parameters=active_routed,
        active_parameters=active,
        active_nonembedding_parameters=active - embedding,
        sparsity=total / active,
        routed_share=active_routed / active,
        target_total_parameters=targets.total_parameters,
        target_active_parameters=targets.active_parameters,
        target_active_nonembedding_parameters=(targets.active_nonembedding_parameters),
        tolerance_fraction=targets.tolerance_fraction,
    )


def mini_k3_r1_config() -> MiniK3Config:
    return MiniK3Config(
        name="r1",
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_experts=256,
        num_experts_per_token=6,
        moe_intermediate_size=416,
        routed_expert_hidden_size=256,
        linear_attn_num_heads=4,
        kv_lora_rank=64,
        q_lora_rank=128,
        intermediate_size=2368,
        full_attn_layers=(4, 8, 12),
        kda_layers=(1, 2, 3, 5, 6, 7, 9, 10, 11),
        attn_res_block_size=1,
        use_fla_kda=True,
    )


def mini_k3_flagship_config() -> MiniK3Config:
    return MiniK3Config(
        name="flagship",
        hidden_size=1536,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_experts=512,
        num_experts_per_token=6,
        moe_intermediate_size=1280,
        routed_expert_hidden_size=768,
        linear_attn_num_heads=8,
        full_attn_layers=(4, 8, 12, 16, 20, 24),
        kda_layers=(
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
        ),
    )


def to_kimi_linear_config_dict(config: MiniK3Config) -> dict[str, object]:
    """Export the contract in the first-party KimiLinearConfig shape."""

    return {
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.effective_num_key_value_heads,
        "qk_nope_head_dim": config.qk_nope_head_dim,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "v_head_dim": config.v_head_dim,
        "kv_lora_rank": config.kv_lora_rank,
        "q_lora_rank": config.q_lora_rank,
        "mla_use_nope": config.mla_use_nope,
        "mla_use_output_gate": config.mla_use_output_gate,
        "num_experts": config.num_experts,
        "num_experts_per_token": config.num_experts_per_token,
        "num_shared_experts": config.num_shared_experts,
        "moe_intermediate_size": config.moe_intermediate_size,
        "routed_expert_hidden_size": config.routed_expert_hidden_size,
        "latent_moe_use_norm": config.latent_moe_use_norm,
        "moe_renormalize": config.moe_renormalize,
        "moe_router_activation_func": config.moe_router_activation_func,
        "topk_method": config.topk_method,
        "num_expert_group": config.num_expert_group,
        "topk_group": config.topk_group,
        "routed_scaling_factor": config.routed_scaling_factor,
        "moe_layer_freq": config.moe_layer_freq,
        "first_k_dense_replace": config.first_k_dense_replace,
        "intermediate_size": config.intermediate_size,
        "attn_res_block_size": config.effective_attn_res_block_size,
        "hidden_act": config.hidden_act,
        "activation_situ_beta": config.activation_situ_beta,
        "activation_situ_linear_beta": config.activation_situ_linear_beta,
        "vocab_size": config.vocab_size,
        "tie_word_embeddings": config.tie_word_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "max_position_embeddings": config.max_position_embeddings,
        "num_nextn_predict_layers": config.num_nextn_predict_layers,
        "initializer_range": config.initializer_range,
        "linear_attn_config": {
            "head_dim": config.linear_attn_head_dim,
            "num_heads": config.linear_attn_num_heads,
            "short_conv_kernel_size": config.short_conv_kernel_size,
            "gate_lower_bound": config.gate_lower_bound,
            "use_full_rank_gate": config.use_full_rank_gate,
            "full_attn_layers": list(config.full_attn_layers),
            "kda_layers": list(config.kda_layers),
        },
    }


def compare_with_first_party_oracle(
    oracle_root: str | Path,
    *,
    r1_config_path: str | Path | None = None,
) -> dict[str, str]:
    """Compare exported contracts with first-party KimiLinearConfig JSON files."""

    root = Path(oracle_root)
    expected_flagship = _normalized_json(root / "model" / "config_mini.json")
    actual_flagship = to_kimi_linear_config_dict(mini_k3_flagship_config())
    result = {
        "flagship": _comparison_status(
            expected=expected_flagship,
            actual=actual_flagship,
        )
    }
    if r1_config_path is not None:
        expected_r1 = _normalized_json(r1_config_path)
        actual_r1 = to_kimi_linear_config_dict(mini_k3_r1_config())
        result["r1"] = _comparison_status(expected=expected_r1, actual=actual_r1)
    return result


def _normalized_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _comparison_status(*, expected: dict[str, Any], actual: dict[str, Any]) -> str:
    if expected == actual:
        return "match"
    mismatched = sorted(
        key
        for key in set(expected) | set(actual)
        if expected.get(key) != actual.get(key)
    )
    return "mismatch: " + ", ".join(mismatched)


def _budget_status(
    *,
    measured: float,
    target: int,
    tolerance_fraction: float,
) -> str:
    tolerance = target * tolerance_fraction
    return "pass" if abs(measured - target) <= tolerance else "fail"
