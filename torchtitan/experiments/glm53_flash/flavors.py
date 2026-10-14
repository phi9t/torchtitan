# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-size contracts for the GLM-5.3-Flash experiment.

Shape suffixes:
    B = batch, L = sequence length, T = packed tokens, D = model dimension,
    H = attention heads, K = query/key head dimension, V = value head dimension,
    R = latent rank, E = experts, F = feed-forward hidden dimension.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace


GLM53_FLASH_SPECIAL_TOKENS = {
    "pad_token_id": 154820,
    "eos_token_ids": (154820, 154827, 154829),
    "image_token_id": 154854,
    "video_token_id": 154855,
}


@dataclass(kw_only=True, slots=True)
class Glm53FlashConfig:
    name: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    first_k_dense_replace: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    n_shared_experts: int
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    num_attention_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    v_head_dim: int = 256
    linear_num_heads: int = 64
    linear_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float = -5.0
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_kpool: int = 4
    index_kpool_always_select_tail: bool = True
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1.0e-6
    rms_norm_eps: float = 1.0e-5
    layer_norm_eps: float = 1.0e-6
    swiglu_limit: float = 10.0
    max_position_embeddings: int = 1_048_576
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    attn_impl: str = "reference"
    sparse_attention_mode: str = "indexer"
    use_mtp: bool = False
    lora_rank: int | None = None
    lora_alpha: float = 64.0
    lora_target_modules: tuple[str, ...] = ()
    layer_offset: int = 0

    @property
    def sparse_layers(self) -> tuple[int, ...]:
        return tuple(
            i
            for i in range(self.num_hidden_layers)
            if (i + self.layer_offset + 1) % 4 == 0
        )

    @property
    def kda_layers(self) -> tuple[int, ...]:
        sparse = set(self.sparse_layers)
        return tuple(i for i in range(self.num_hidden_layers) if i not in sparse)

    @property
    def layer_types(self) -> tuple[str, ...]:
        sparse = set(self.sparse_layers)
        return tuple(
            "deepseek_sparse_attention" if i in sparse else "linear_attention"
            for i in range(self.num_hidden_layers)
        )

    @property
    def mlp_layer_types(self) -> tuple[str, ...]:
        return tuple(
            "dense" if i + self.layer_offset < self.first_k_dense_replace else "sparse"
            for i in range(self.num_hidden_layers)
        )

    def validate(self) -> None:
        if self.layer_offset < 0:
            raise ValueError("layer_offset must be non-negative")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.index_topk % self.index_kpool != 0:
            raise ValueError("index_topk must be divisible by index_kpool")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")
        if self.n_group != 1 or self.topk_group != 1:
            raise ValueError("this experiment currently supports one router group")
        if self.use_mtp:
            raise ValueError("MTP is intentionally skipped for SFT in this seam")
        if self.attn_impl not in {"reference", "fast"}:
            raise ValueError("attn_impl must be 'reference' or 'fast'")
        if self.sparse_attention_mode not in {"indexer", "dense_causal"}:
            raise ValueError(
                "sparse_attention_mode must be 'indexer' or 'dense_causal'"
            )


def glm53_flash() -> Glm53FlashConfig:
    """Exact released text-decoder contract, excluding MTP and vision."""

    cfg = Glm53FlashConfig(
        name="glm53_flash",
        vocab_size=154_880,
        hidden_size=4096,
        num_hidden_layers=45,
        intermediate_size=12_288,
        first_k_dense_replace=3,
        num_experts=288,
        num_experts_per_tok=8,
        moe_intermediate_size=2048,
        n_shared_experts=1,
        num_attention_heads=64,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        v_head_dim=256,
        linear_num_heads=64,
        linear_head_dim=128,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
    )
    cfg.validate()
    return cfg


def debugmodel() -> Glm53FlashConfig:
    """Small launchable model preserving the released 3:1 KDA:sparse schedule."""

    cfg = Glm53FlashConfig(
        name="debugmodel",
        vocab_size=512,
        hidden_size=256,
        num_hidden_layers=4,
        intermediate_size=768,
        first_k_dense_replace=1,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        n_shared_experts=1,
        num_attention_heads=4,
        q_lora_rank=128,
        kv_lora_rank=512,
        qk_nope_head_dim=64,
        v_head_dim=64,
        linear_num_heads=2,
        linear_head_dim=128,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=16,
        max_position_embeddings=8192,
    )
    cfg.validate()
    return cfg


def mini() -> Glm53FlashConfig:
    """Intermediate ladder rung with the same layer and MoE semantics."""

    cfg = Glm53FlashConfig(
        name="mini",
        vocab_size=32_000,
        hidden_size=1536,
        num_hidden_layers=16,
        intermediate_size=4096,
        first_k_dense_replace=2,
        num_experts=64,
        num_experts_per_tok=4,
        moe_intermediate_size=768,
        n_shared_experts=1,
        num_attention_heads=16,
        q_lora_rank=512,
        kv_lora_rank=512,
        qk_nope_head_dim=96,
        v_head_dim=96,
        linear_num_heads=12,
        linear_head_dim=128,
        index_n_heads=16,
        index_head_dim=128,
        index_topk=512,
        max_position_embeddings=65_536,
    )
    cfg.validate()
    return cfg


LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "f_a_proj",
    "f_b_proj",
    "g_a_proj",
    "g_b_proj",
    "shared_experts.gate_proj",
    "shared_experts.up_proj",
    "shared_experts.down_proj",
)


def with_lora(
    cfg: Glm53FlashConfig,
    *,
    rank: int = 16,
    alpha: float = 64.0,
    targets: tuple[str, ...] = LORA_TARGET_MODULES,
) -> Glm53FlashConfig:
    cfg = replace(
        cfg,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_target_modules=targets,
    )
    cfg.validate()
    return cfg


MODEL_FLAVORS: dict[str, Callable[[], Glm53FlashConfig]] = {
    "glm53_flash": glm53_flash,
    "debugmodel": debugmodel,
    "mini": mini,
}


def build_model_config(flavor: str) -> Glm53FlashConfig:
    try:
        return MODEL_FLAVORS[flavor]()
    except KeyError as exc:
        raise ValueError(
            f"unsupported GLM-5.3-Flash flavor {flavor!r}; "
            f"available: {sorted(MODEL_FLAVORS)}"
        ) from exc
