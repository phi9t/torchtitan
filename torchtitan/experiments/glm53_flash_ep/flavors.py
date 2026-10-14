# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-size contracts for the isolated GLM EP prototype."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from torchtitan.experiments.glm53_flash.flavors import (
    debugmodel as _debugmodel,
    Glm53FlashConfig,
    mini as _mini,
    with_lora,
)


def debugmodel_ep() -> Glm53FlashConfig:
    """Debug GLM flavor with routed experts on the common EP path."""

    cfg = _debugmodel()
    cfg.name = "debugmodel_ep"
    cfg.validate()
    return cfg


def debugmodel_ep_lora() -> Glm53FlashConfig:
    """LoRA debug flavor used for trainable-only checkpoint checks."""

    cfg = with_lora(debugmodel_ep(), rank=8, alpha=32.0)
    cfg.name = "debugmodel_ep_lora"
    cfg.validate()
    return cfg


def mini_ep() -> Glm53FlashConfig:
    """Mini GLM flavor with the same EP adapter and 64 routed experts."""

    cfg = _mini()
    cfg.name = "mini_ep"
    cfg.validate()
    return cfg


def tiny_ep() -> Glm53FlashConfig:
    """Tiny CPU parity flavor with multiple MoE layers and four experts."""

    cfg = replace(
        _debugmodel(),
        name="tiny_ep",
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=4,
        intermediate_size=64,
        first_k_dense_replace=1,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        n_shared_experts=1,
        num_attention_heads=4,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        v_head_dim=8,
        linear_num_heads=2,
        linear_head_dim=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        index_kpool=2,
        hc_mult=2,
        sparse_attention_mode="dense_causal",
        max_position_embeddings=1024,
    )
    cfg.validate()
    return cfg


MODEL_FLAVORS: dict[str, Callable[[], Glm53FlashConfig]] = {
    "debugmodel_ep": debugmodel_ep,
    "debugmodel_ep_lora": debugmodel_ep_lora,
    "mini_ep": mini_ep,
    "tiny_ep": tiny_ep,
}


def build_model_config(flavor: str) -> Glm53FlashConfig:
    try:
        return MODEL_FLAVORS[flavor]()
    except KeyError as exc:
        raise ValueError(
            f"unsupported GLM-5.3-Flash EP flavor {flavor!r}; "
            f"available: {sorted(MODEL_FLAVORS)}"
        ) from exc
