# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tiny Mini Kimi K3 decoder assembly for oracle and plumbing tests."""

from __future__ import annotations

import torch
from torch import nn

from torchtitan.experiments.mini_kimi_k3.kda import MiniK3DeltaAttention
from torchtitan.experiments.mini_kimi_k3.mla import MiniK3MLAAttention, MiniK3RMSNorm
from torchtitan.experiments.mini_kimi_k3.model_contract import MiniK3Config
from torchtitan.experiments.mini_kimi_k3.moe import (
    MiniK3BlockSparseMLP,
    MiniK3DenseMLP,
    MiniK3SparseMoeBlock,
)
from torchtitan.experiments.mini_kimi_k3.router import NoAuxTcRouter


def apply_attention_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: MiniK3RMSNorm,
) -> torch.Tensor:
    """Apply the first-party attention-residual weighted average."""

    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_float = values.float()
    variance = values_float.pow(2).mean(-1, keepdim=True)
    keys = values_float * torch.rsqrt(variance + norm.eps)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (keys * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, values_float).squeeze(1).to(values.dtype)


class MiniK3DecoderLayer(nn.Module):
    """One Mini-K3 decoder layer assembled from experiment-owned primitives."""

    def __init__(self, config: MiniK3Config, *, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        first_party_layer_idx = layer_idx + 1
        if first_party_layer_idx in config.kda_layers:
            self.self_attn = MiniK3DeltaAttention(
                hidden_size=config.hidden_size,
                num_heads=config.linear_attn_num_heads,
                head_dim=config.linear_attn_head_dim,
                conv_kernel_size=config.short_conv_kernel_size,
                gate_lower_bound=config.gate_lower_bound,
                use_full_rank_gate=config.use_full_rank_gate,
                use_fla_backend=config.use_fla_kda,
                rms_norm_eps=config.rms_norm_eps,
            )
        elif first_party_layer_idx in config.full_attn_layers:
            self.self_attn = MiniK3MLAAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                rms_norm_eps=config.rms_norm_eps,
                use_output_gate=config.mla_use_output_gate,
            )
        else:
            raise ValueError(
                f"layer {first_party_layer_idx} is not in any attention set"
            )

        self.input_layernorm = MiniK3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MiniK3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.use_attn_residuals = config.attn_res_block_size is not None
        if self.use_attn_residuals:
            self.attn_res_block_size = config.attn_res_block_size
            self.self_attention_res_norm = MiniK3RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp_res_norm = MiniK3RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        if self._uses_sparse_moe(config, layer_idx):
            router = NoAuxTcRouter(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_token,
                routed_scaling_factor=config.routed_scaling_factor,
                activation=config.moe_router_activation_func,
                num_expert_group=config.num_expert_group,
                topk_group=config.topk_group,
                moe_renormalize=config.moe_renormalize,
            )
            experts = [
                MiniK3BlockSparseMLP(
                    hidden_size=config.routed_expert_hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    activation_beta=config.activation_situ_beta,
                    activation_linear_beta=config.activation_situ_linear_beta,
                )
                for _ in range(config.num_experts)
            ]
            shared_experts = (
                MiniK3DenseMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size
                    * config.num_shared_experts,
                    activation_beta=config.activation_situ_beta,
                    activation_linear_beta=config.activation_situ_linear_beta,
                )
                if config.num_shared_experts is not None
                else None
            )
            self.block_sparse_moe = MiniK3SparseMoeBlock(
                router=router,
                experts=experts,
                shared_experts=shared_experts,
                hidden_size=config.hidden_size,
                routed_expert_hidden_size=config.routed_expert_hidden_size,
                latent_moe_use_norm=config.latent_moe_use_norm,
            )
        else:
            self.feed_forward = MiniK3DenseMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                activation_beta=config.activation_situ_beta,
                activation_linear_beta=config.activation_situ_linear_beta,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        block_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_attn_residuals:
            if block_residual is None:
                raise ValueError(
                    "block_residual is required when attn_res_block_size is set"
                )
            return self._forward_attention_residual(
                hidden_states,
                positions=positions,
                block_residual=block_residual,
            )

        residual = hidden_states
        attn_input = self.input_layernorm(hidden_states)
        if isinstance(self.self_attn, MiniK3MLAAttention):
            attn_out = self.self_attn(attn_input, positions=positions)
        else:
            attn_out = self.self_attn(attn_input)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        hidden_states = residual + attn_out

        residual = hidden_states
        ffn_input = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            ffn_out = self.block_sparse_moe(ffn_input)
        else:
            ffn_out = self.feed_forward(ffn_input)
        return residual + ffn_out

    def _forward_attention_residual(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor | None,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix_sum = hidden_states

        if block_residual.shape[1] > 0:
            hidden_states = apply_attention_residual(
                prefix_sum.view(-1, hidden_size),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            ).view(batch_size, seq_len, hidden_size)

        assert self.attn_res_block_size is not None
        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual = torch.cat(
                [block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)],
                dim=1,
            )
            prefix_sum = None

        attn_input = self.input_layernorm(hidden_states)
        if isinstance(self.self_attn, MiniK3MLAAttention):
            attn_out = self.self_attn(attn_input, positions=positions)
        else:
            attn_out = self.self_attn(attn_input)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]

        if prefix_sum is None:
            prefix_sum = attn_out
        else:
            prefix_sum = prefix_sum + attn_out

        hidden_states = apply_attention_residual(
            prefix_sum.view(-1, hidden_size),
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        ).view(batch_size, seq_len, hidden_size)

        ffn_input = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            ffn_out = self.block_sparse_moe(ffn_input)
        else:
            ffn_out = self.feed_forward(ffn_input)

        if prefix_sum is None:
            prefix_sum = ffn_out
        else:
            prefix_sum = prefix_sum + ffn_out
        return prefix_sum, block_residual

    @staticmethod
    def _uses_sparse_moe(config: MiniK3Config, layer_idx: int) -> bool:
        return (
            config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )


class MiniK3ForCausalLM(nn.Module):
    """Minimal causal LM wrapper for Mini-K3 decoder plumbing tests."""

    def __init__(self, config: MiniK3Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            MiniK3DecoderLayer(config, layer_idx=layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = MiniK3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_attn_residuals = config.attn_res_block_size is not None
        if self.use_attn_residuals:
            self.output_attn_res_norm = MiniK3RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        hidden_states = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        block_residual = None
        if self.use_attn_residuals:
            block_residual = hidden_states.new_zeros(
                hidden_states.shape[0] * hidden_states.shape[1],
                0,
                hidden_states.shape[2],
            )
        for layer in self.layers:
            if self.use_attn_residuals:
                assert block_residual is not None
                hidden_states, block_residual = layer(
                    hidden_states,
                    positions=positions,
                    block_residual=block_residual,
                )
            else:
                hidden_states = layer(hidden_states, positions=positions)
        if self.use_attn_residuals:
            assert block_residual is not None
            hidden_states = apply_attention_residual(
                hidden_states.view(-1, hidden_states.shape[-1]),
                block_residual,
                self.output_attn_res_proj,
                self.output_attn_res_norm,
            ).view_as(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)
