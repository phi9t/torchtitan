# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from torchtitan.experiments.mini_kimi_k3.kda import MiniK3DeltaAttention
from torchtitan.experiments.mini_kimi_k3.mla import MiniK3MLAAttention
from torchtitan.experiments.mini_kimi_k3.model import (
    MiniK3DecoderLayer,
    MiniK3ForCausalLM,
)
from torchtitan.experiments.mini_kimi_k3.model_contract import MiniK3Config
from torchtitan.experiments.mini_kimi_k3.moe import MiniK3DenseMLP, MiniK3SparseMoeBlock


def _tiny_config(
    *,
    tie_word_embeddings: bool = True,
    routed_expert_hidden_size: int = 8,
    attn_res_block_size: int | None = None,
) -> MiniK3Config:
    return MiniK3Config(
        name="tiny",
        hidden_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=4,
        routed_expert_hidden_size=routed_expert_hidden_size,
        linear_attn_num_heads=2,
        full_attn_layers=(2,),
        kda_layers=(1, 3),
        vocab_size=32,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=3,
        kv_lora_rank=4,
        q_lora_rank=4,
        num_shared_experts=1,
        first_k_dense_replace=1,
        intermediate_size=16,
        linear_attn_head_dim=4,
        short_conv_kernel_size=1,
        tie_word_embeddings=tie_word_embeddings,
        attn_res_block_size=attn_res_block_size,
    )


def test_mini_k3_decoder_layer_selects_kda_or_mla_from_contract():
    config = _tiny_config()

    first = MiniK3DecoderLayer(config, layer_idx=0)
    second = MiniK3DecoderLayer(config, layer_idx=1)

    assert isinstance(first.self_attn, MiniK3DeltaAttention)
    assert isinstance(second.self_attn, MiniK3MLAAttention)


def test_mini_k3_decoder_layer_uses_dense_first_layer_then_sparse_moe():
    config = _tiny_config()

    first = MiniK3DecoderLayer(config, layer_idx=0)
    second = MiniK3DecoderLayer(config, layer_idx=1)

    assert isinstance(first.feed_forward, MiniK3DenseMLP)
    assert not hasattr(first, "block_sparse_moe")
    assert isinstance(second.block_sparse_moe, MiniK3SparseMoeBlock)
    assert second.block_sparse_moe.experts[0].w1.weight.shape == (4, 8)
    assert isinstance(second.block_sparse_moe.shared_experts, MiniK3DenseMLP)
    assert second.block_sparse_moe.routed_expert_down_proj is None


def test_mini_k3_decoder_layer_uses_latent_moe_when_routed_width_is_narrower():
    config = _tiny_config(routed_expert_hidden_size=4)
    layer = MiniK3DecoderLayer(config, layer_idx=1)
    x = torch.randn(2, 3, config.hidden_size)

    out = layer(x)

    assert out.shape == x.shape
    assert layer.block_sparse_moe.routed_expert_down_proj is not None
    assert layer.block_sparse_moe.routed_expert_down_proj.weight.shape == (4, 8)
    assert layer.block_sparse_moe.routed_expert_up_proj.weight.shape == (8, 4)


def test_mini_k3_decoder_layer_uses_first_party_attention_residual_path():
    config = _tiny_config(attn_res_block_size=1)
    layer = MiniK3DecoderLayer(config, layer_idx=1)
    x = torch.randn(2, 3, config.hidden_size)
    block_residual = x.new_zeros(x.shape[0] * x.shape[1], 0, x.shape[2])

    out, next_block_residual = layer(x, block_residual=block_residual)

    assert out.shape == x.shape
    assert next_block_residual.shape == (6, 1, config.hidden_size)
    assert layer.self_attention_res_norm.weight.shape == (config.hidden_size,)
    assert layer.self_attention_res_proj.weight.shape == (1, config.hidden_size)
    assert layer.mlp_res_norm.weight.shape == (config.hidden_size,)
    assert layer.mlp_res_proj.weight.shape == (1, config.hidden_size)


def test_mini_k3_causal_lm_returns_logits_and_ties_output_weight():
    config = _tiny_config(tie_word_embeddings=True)
    model = MiniK3ForCausalLM(config)
    input_ids = torch.tensor([[1, 2, 3, 4]])

    logits = model(input_ids)

    assert logits.shape == (1, 4, config.vocab_size)
    assert model.lm_head.weight is model.embed_tokens.weight


def test_mini_k3_causal_lm_applies_output_attention_residual():
    config = _tiny_config(tie_word_embeddings=True, attn_res_block_size=1)
    model = MiniK3ForCausalLM(config)
    input_ids = torch.tensor([[1, 2, 3, 4]])

    logits = model(input_ids)

    assert logits.shape == (1, 4, config.vocab_size)
    assert model.output_attn_res_norm.weight.shape == (config.hidden_size,)
    assert model.output_attn_res_proj.weight.shape == (1, config.hidden_size)


def test_mini_k3_causal_lm_supports_untied_output_weight_and_backward():
    config = _tiny_config(tie_word_embeddings=False)
    model = MiniK3ForCausalLM(config)
    input_ids = torch.tensor([[1, 2, 3]])

    logits = model(input_ids)
    logits.sum().backward()

    assert logits.shape == (1, 3, config.vocab_size)
    assert model.lm_head.weight is not model.embed_tokens.weight
    assert model.embed_tokens.weight.grad is not None
    assert model.lm_head.weight.grad is not None
