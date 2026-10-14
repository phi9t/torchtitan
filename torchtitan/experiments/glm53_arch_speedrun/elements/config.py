# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Switch contracts for GLM-5.3-Flash speedrun architecture elements."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(kw_only=True, slots=True)
class SpeedrunElementsConfig:
    """Element switches sized for the modded-nanogpt speedrun substrate."""

    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 6
    head_dim: int = 128
    mlp_hidden_size: int = 3072
    vocab_size: int = 50_304
    max_seq_len: int = 4096
    rope_theta: float = 10_000.0
    rope_enabled: bool = True
    mhc_enabled: bool = False
    mhc_streams: int = 4
    mhc_sinkhorn_iters: int = 20
    mhc_eps: float = 1.0e-6
    mhc_identity_at_init: bool = True
    kda_enabled: bool = False
    kda_hybrid_3_to_1: bool = False
    kda_conv_kernel_size: int = 4
    kda_gate_lower_bound: float = -5.0
    sparse_attention_enabled: bool = False
    sparse_top_k: int = 2048
    indexer_num_heads: int = 6
    indexer_head_dim: int = 128
    index_pool_size: int = 1
    index_pool_always_select_tail: bool = True
    mtp_enabled: bool = False
    mtp_aux_loss_weight: float = 0.0

    def validate(self) -> None:
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim for the "
                "speedrun substrate."
            )
        if self.indexer_head_dim != self.head_dim:
            raise ValueError("indexer_head_dim must match head_dim for A4/A5.")
        if self.index_pool_size < 1:
            raise ValueError("index_pool_size must be >= 1.")
        if self.sparse_top_k < 1:
            raise ValueError("sparse_top_k must be >= 1.")
        if self.mhc_streams < 1:
            raise ValueError("mhc_streams must be >= 1.")
        if not -5.0 <= self.kda_gate_lower_bound < 0.0:
            raise ValueError("kda_gate_lower_bound must be in [-5, 0).")
        if self.mtp_aux_loss_weight < 0.0:
            raise ValueError("mtp_aux_loss_weight must be non-negative.")

    def attention_kind(self, layer_idx: int) -> str:
        if not self.kda_enabled:
            return "softmax"
        if self.kda_hybrid_3_to_1 and (layer_idx + 1) % 4 == 0:
            return "softmax"
        return "kda"


def b0_config() -> SpeedrunElementsConfig:
    cfg = SpeedrunElementsConfig()
    cfg.validate()
    return cfg


def arm_config(arm: str) -> SpeedrunElementsConfig:
    """Return the one-knob architecture-element arm requested by the brief."""

    arm = arm.upper()
    if arm == "B0":
        return b0_config()
    return compose_arms(arm)


def compose_arms(
    *arms: str,
    base: SpeedrunElementsConfig | None = None,
    sparse_top_k: int | None = None,
    index_pool_size: int | None = None,
    mtp_aux_loss_weight: float | None = None,
    mhc_sinkhorn_iters: int | None = None,
    kda_conv_kernel_size: int | None = None,
) -> SpeedrunElementsConfig:
    """Compose GLM element factors and optional dose knobs for S03 ablations."""

    cfg = b0_config() if base is None else base
    requested = {arm.upper() for arm in arms}
    unknown = requested - {"A1", "A2", "A3", "A4", "A5", "A6"}
    if unknown:
        raise ValueError(f"unknown GLM-5.3 speedrun arms: {sorted(unknown)}")
    if "A1" in requested:
        cfg = replace(cfg, mhc_enabled=True, mhc_identity_at_init=True)
    if "A2" in requested:
        cfg = replace(cfg, rope_enabled=False)
    if "A3" in requested:
        cfg = replace(cfg, kda_enabled=True, kda_hybrid_3_to_1=True)
    if "A4" in requested:
        cfg = replace(cfg, sparse_attention_enabled=True)
    if "A5" in requested:
        cfg = replace(
            cfg,
            sparse_attention_enabled=True,
            index_pool_size=4,
        )
    if "A6" in requested:
        cfg = replace(cfg, mtp_enabled=True, mtp_aux_loss_weight=0.1)

    dose_updates = {}
    if sparse_top_k is not None:
        dose_updates["sparse_top_k"] = sparse_top_k
    if index_pool_size is not None:
        dose_updates["index_pool_size"] = index_pool_size
        dose_updates["sparse_attention_enabled"] = True
    if mtp_aux_loss_weight is not None:
        dose_updates["mtp_aux_loss_weight"] = mtp_aux_loss_weight
        dose_updates["mtp_enabled"] = mtp_aux_loss_weight > 0.0
    if mhc_sinkhorn_iters is not None:
        dose_updates["mhc_sinkhorn_iters"] = mhc_sinkhorn_iters
    if kda_conv_kernel_size is not None:
        dose_updates["kda_conv_kernel_size"] = kda_conv_kernel_size
    if dose_updates:
        cfg = replace(cfg, **dose_updates)
    cfg.validate()
    return cfg
