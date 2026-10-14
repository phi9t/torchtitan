# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Speedrun-scale mini flavors for the architecture ladder.

Shape suffixes:
    T = packed tokens, D = model dimension, H = attention heads,
    K = query/key head dimension, F = feed-forward hidden dimension,
    E = experts, R = latent rank.
"""

from __future__ import annotations

from dataclasses import replace

from torchtitan.experiments.glm53_flash.flavors import (
    build_model_config as build_glm53_flash_config,
    Glm53FlashConfig,
)
from torchtitan.experiments.glm53_flash.model import Glm53FlashForCausalLM

from torchtitan.models.common import CosSinRoPE, Embedding, Linear
from torchtitan.models.kimi_k3.flavors import _kimi_k3_config
from torchtitan.models.qwen3.flavors import (
    _build_qwen3_layers,
    _build_qwen3_moe_layers,
    _EMBEDDING_INIT,
    _EMBEDDING_SKIP_INIT,
    _LINEAR_INIT,
    _qwen3_norm,
)
from torchtitan.models.qwen3.model import Qwen3Model


GPT2_VOCAB_SIZE = 50257
TARGET_ACTIVE_NON_EMBEDDING_PARAMS = 85_000_000


def qwen3_dense_l1(seq_len: int, *, attn_backend: str = "varlen") -> Qwen3Model.Config:
    """L1 Qwen3-style dense mini flavor."""

    dim = 768
    head_dim = 128
    return Qwen3Model.Config(
        max_context_length=seq_len,
        vocab_size=GPT2_VOCAB_SIZE,
        dim=dim,
        norm=_qwen3_norm(dim),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=GPT2_VOCAB_SIZE,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=GPT2_VOCAB_SIZE,
            param_init=_LINEAR_INIT,
        ),
        layers=_build_qwen3_layers(
            n_layers=14,
            dim=dim,
            n_heads=6,
            n_kv_heads=2,
            head_dim=head_dim,
            hidden_dim=2048,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=seq_len,
                theta=1_000_000.0,
            ),
        ),
    )


def qwen3_moe_l2(seq_len: int, *, attn_backend: str = "varlen") -> Qwen3Model.Config:
    """L2 Qwen3-MoE-style mini flavor."""

    dim = 768
    head_dim = 128
    return Qwen3Model.Config(
        max_context_length=seq_len,
        vocab_size=GPT2_VOCAB_SIZE,
        dim=dim,
        norm=_qwen3_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=GPT2_VOCAB_SIZE,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=GPT2_VOCAB_SIZE,
            param_init=_LINEAR_INIT,
        ),
        layers=_build_qwen3_moe_layers(
            n_layers=14,
            dim=dim,
            n_heads=6,
            n_kv_heads=2,
            head_dim=head_dim,
            moe_hidden_dim=1024,
            num_experts=8,
            top_k=2,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=seq_len,
                theta=1_000_000.0,
            ),
        ),
    )


def kimi_k3_l3(seq_len: int, *, attn_backend: str = "varlen"):
    """L3 mini-Kimi-K3 flavor using the real Kimi K3 block builder."""

    dim = 768
    return _kimi_k3_config(
        max_context_length=seq_len,
        dim=dim,
        vocab_size=GPT2_VOCAB_SIZE,
        num_layers=17,
        full_attention_layers={3, 7, 11, 15, 16},
        attn_res_block_size=4,
        num_heads=6,
        q_lora_rank=256,
        kv_lora_rank=256,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=2048,
        latent_dim=256,
        expert_hidden_dim=512,
        num_experts=8,
        top_k=2,
        num_shared_experts=1,
        vision_encoder=None,
        attn_backend=attn_backend,
    )


def glm53_flash_l4(
    seq_len: int,
    *,
    attn_impl: str = "fast",
    sparse_attention_mode: str = "indexer",
) -> Glm53FlashForCausalLM.Config:
    """L4 mini-GLM-5.3-Flash flavor using the GLM experiment model."""

    cfg: Glm53FlashConfig = replace(
        build_glm53_flash_config("mini"),
        name="arch_ladder_l4",
        vocab_size=GPT2_VOCAB_SIZE,
        hidden_size=768,
        num_hidden_layers=16,
        intermediate_size=1536,
        first_k_dense_replace=2,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=448,
        n_shared_experts=1,
        num_attention_heads=6,
        q_lora_rank=256,
        kv_lora_rank=256,
        qk_nope_head_dim=64,
        v_head_dim=128,
        linear_num_heads=6,
        linear_head_dim=128,
        index_n_heads=6,
        index_head_dim=128,
        index_topk=max(4, seq_len // 8),
        index_kpool=4,
        max_position_embeddings=max(seq_len, 4096),
        attn_impl=attn_impl,
        sparse_attention_mode=sparse_attention_mode,
        use_mtp=False,
    )
    cfg.validate()
    return Glm53FlashForCausalLM.Config(
        glm_config=cfg,
        experiment_name="glm53_arch_ladder",
        experiment_flavor="l4_glm53_flash",
    )


def build_ladder_model_config(
    rung: str,
    *,
    seq_len: int,
    attn_backend: str = "varlen",
    glm_attn_impl: str = "fast",
    glm_sparse_attention_mode: str = "indexer",
):
    canonical = rung.upper()
    if canonical == "L1":
        return qwen3_dense_l1(seq_len, attn_backend=attn_backend)
    if canonical == "L2":
        return qwen3_moe_l2(seq_len, attn_backend=attn_backend)
    if canonical == "L3":
        return kimi_k3_l3(seq_len, attn_backend=attn_backend)
    if canonical == "L4":
        return glm53_flash_l4(
            seq_len,
            attn_impl=glm_attn_impl,
            sparse_attention_mode=glm_sparse_attention_mode,
        )
    raise ValueError(f"unknown architecture-ladder rung {rung!r}")
