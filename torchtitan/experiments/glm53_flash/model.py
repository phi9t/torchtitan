# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Text-only GLM-5.3-Flash experiment model.

Shape suffixes:
    B = batch, L = sequence length, T = packed tokens, D = model dimension,
    N = mHC streams, H = attention heads, K = query/key head dimension,
    V = value head dimension, R = latent rank, E = experts, F = FFN width,
    P = index pools, S = selected indices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
import torch_remat as remat
from torch import nn

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.base.model import BaseModel

from .flavors import debugmodel, Glm53FlashConfig


logger = logging.getLogger(__name__)


def _rms_norm(x_BLD: torch.Tensor, weight_D: torch.Tensor, eps: float) -> torch.Tensor:
    x_float_BLD = x_BLD.float()
    out_BLD = x_float_BLD * torch.rsqrt(
        x_float_BLD.square().mean(dim=-1, keepdim=True) + eps
    )
    return (out_BLD * weight_D.float()).to(x_BLD.dtype)


def _l2norm(x_BLHK: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    x_float_BLHK = x_BLHK.float()
    return (
        x_float_BLHK
        * torch.rsqrt(x_float_BLHK.square().sum(dim=-1, keepdim=True) + eps)
    ).to(x_BLHK.dtype)


class GlmRMSNorm(nn.Module):
    def __init__(self, dim: int, *, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x_BLD, self.weight, self.eps)


class GlmUnweightedRMSNorm(nn.Module):
    def __init__(self, *, eps: float) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        x_float_BLD = x_BLD.float()
        return (
            x_float_BLD
            * torch.rsqrt(x_float_BLD.square().mean(dim=-1, keepdim=True) + self.eps)
        ).to(x_BLD.dtype)


class GlmGatedRMSNorm(nn.Module):
    def __init__(self, dim: int, *, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps

    def forward(self, x_BLHK: torch.Tensor, gate_BLHK: torch.Tensor) -> torch.Tensor:
        x_float_BLHK = x_BLHK.float()
        x_float_BLHK = x_float_BLHK * torch.rsqrt(
            x_float_BLHK.square().mean(dim=-1, keepdim=True) + self.eps
        )
        x_float_BLHK = x_float_BLHK * self.weight.float()
        x_float_BLHK = x_float_BLHK * torch.sigmoid(gate_BLHK.float())
        return x_float_BLHK.to(x_BLHK.dtype)


class LoRALinear(nn.Linear):
    """Experiment-owned LoRA wrapper that preserves ``weight`` key names."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        rank: int | None = None,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.rank = rank
        self.alpha = alpha
        if rank is not None:
            self.weight.requires_grad_(False)
            if self.bias is not None:
                self.bias.requires_grad_(False)
            self.lora_a = nn.Parameter(torch.empty(rank, in_features))
            self.lora_b = nn.Parameter(torch.empty(out_features, rank))
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.rank is None:
            return out
        assert self.lora_a is not None and self.lora_b is not None
        adapter = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return out + adapter * (self.alpha / self.rank)


def _matches_lora_target(suffix: str, config: Glm53FlashConfig) -> bool:
    return config.lora_rank is not None and any(
        suffix == target or suffix.endswith(f".{target}")
        for target in config.lora_target_modules
    )


def _linear(
    in_features: int,
    out_features: int,
    *,
    config: Glm53FlashConfig,
    suffix: str,
    bias: bool = False,
) -> LoRALinear:
    return LoRALinear(
        in_features,
        out_features,
        bias=bias,
        rank=config.lora_rank if _matches_lora_target(suffix, config) else None,
        alpha=config.lora_alpha,
    )


class GlmHyperConnection(nn.Module):
    def __init__(self, config: Glm53FlashConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = GlmUnweightedRMSNorm(eps=config.rms_norm_eps)
        mix_size = (2 + config.hc_mult) * config.hc_mult
        self.fn = nn.Parameter(
            torch.empty(mix_size, config.hc_mult * config.hidden_size)
        )
        self.base = nn.Parameter(torch.empty(mix_size))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(
        self, hidden_streams_BLND: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = hidden_streams_BLND.shape[:2]
        hc = self.hc_mult
        flat_BLD = hidden_streams_BLND.view(batch_size, seq_len, -1)
        flat_BLD = self.input_norm(flat_BLD)
        mixes_BLW = F.linear(flat_BLD.float(), self.fn.float())
        pre_w_BLN, post_w_BLN, comb_w_BLNN = mixes_BLW.split([hc, hc, hc * hc], dim=-1)
        pre_b_N, post_b_N, comb_b_NN = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        comb_w_BLNN = comb_w_BLNN.view(batch_size, seq_len, hc, hc)
        comb_b_NN = comb_b_NN.view(hc, hc)

        pre_BLN = torch.sigmoid(pre_w_BLN * pre_scale + pre_b_N) + self.hc_eps
        post_BLN = 2 * torch.sigmoid(post_w_BLN * post_scale + post_b_N)
        comb_BLNN = (
            torch.softmax(comb_w_BLNN * comb_scale + comb_b_NN, dim=-1) + self.hc_eps
        )
        comb_BLNN = comb_BLNN / (comb_BLNN.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb_BLNN = comb_BLNN / (comb_BLNN.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb_BLNN = comb_BLNN / (comb_BLNN.sum(dim=-2, keepdim=True) + self.hc_eps)
        collapsed_BLD = (pre_BLN.unsqueeze(-1) * hidden_streams_BLND).sum(dim=2)
        return post_BLN, comb_BLNN, collapsed_BLD.to(hidden_streams_BLND.dtype)


def glm_hc_post_mix(
    x_BLD: torch.Tensor,
    residual_BLND: torch.Tensor,
    post_BLN: torch.Tensor,
    comb_BLNN: torch.Tensor,
) -> torch.Tensor:
    """HF GLM post-mix: output_j = post_j * x + sum_i comb_ij * residual_i."""

    mixed_BLND = torch.matmul(
        comb_BLNN.to(x_BLD.dtype).transpose(-1, -2), residual_BLND
    )
    return post_BLN.to(x_BLD.dtype).unsqueeze(-1) * x_BLD.unsqueeze(-2) + mixed_BLND


class GlmMLP(nn.Module):
    def __init__(
        self,
        config: Glm53FlashConfig,
        *,
        intermediate_size: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.swiglu_limit = config.swiglu_limit
        self.gate_proj = _linear(
            config.hidden_size,
            intermediate_size,
            config=config,
            suffix=f"{prefix}.gate_proj",
        )
        self.up_proj = _linear(
            config.hidden_size,
            intermediate_size,
            config=config,
            suffix=f"{prefix}.up_proj",
        )
        self.down_proj = _linear(
            intermediate_size,
            config.hidden_size,
            config=config,
            suffix=f"{prefix}.down_proj",
        )

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        gate_BLF = self.gate_proj(x_BLD).clamp(max=self.swiglu_limit)
        up_BLF = self.up_proj(x_BLD).clamp(
            min=-self.swiglu_limit, max=self.swiglu_limit
        )
        return self.down_proj(F.silu(gate_BLF) * up_BLF)


class GlmExperts(nn.Module):
    def __init__(self, config: Glm53FlashConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                config.num_experts, 2 * config.moe_intermediate_size, config.hidden_size
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                config.num_experts, config.hidden_size, config.moe_intermediate_size
            )
        )

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_index_TK: torch.Tensor,
        topk_weights_TK: torch.Tensor,
    ) -> torch.Tensor:
        out_TD = torch.zeros_like(x_TD)
        routing_map_ETK = F.one_hot(
            topk_index_TK, num_classes=self.num_experts
        ).permute(2, 0, 1)
        for expert_idx in routing_map_ETK.sum(dim=(1, 2)).nonzero().flatten():
            token_T, slot_T = torch.where(routing_map_ETK[expert_idx])
            gate_up_TF2 = F.linear(x_TD[token_T], self.gate_up_proj[expert_idx])
            gate_TF, up_TF = gate_up_TF2.chunk(2, dim=-1)
            hidden_TF = F.silu(gate_TF.clamp(max=self.swiglu_limit)) * up_TF.clamp(
                min=-self.swiglu_limit, max=self.swiglu_limit
            )
            contrib_TD = F.linear(hidden_TF, self.down_proj[expert_idx])
            contrib_TD = contrib_TD * topk_weights_TK[token_T, slot_T, None]
            out_TD.index_add_(0, token_T, contrib_TD.to(out_TD.dtype))
        return out_TD


class GlmTopKRouter(nn.Module):
    def __init__(self, config: Glm53FlashConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=True,
        )

    def forward(self, x_BLD: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_TD = x_BLD.view(-1, self.hidden_dim)
        scores_TE = F.linear(x_TD.float(), self.weight.float()).sigmoid()
        scores_for_choice_TE = scores_TE + self.e_score_correction_bias
        topk_index_TK = torch.topk(
            scores_for_choice_TE, k=self.top_k, dim=-1, sorted=False
        ).indices
        topk_weights_TK = scores_TE.gather(1, topk_index_TK)
        if self.norm_topk_prob:
            topk_weights_TK = topk_weights_TK / (
                topk_weights_TK.sum(dim=-1, keepdim=True) + 1.0e-20
            )
        return topk_index_TK, topk_weights_TK * self.routed_scaling_factor


class GlmMoE(nn.Module):
    def __init__(self, config: Glm53FlashConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.experts = GlmExperts(config)
        self.gate = GlmTopKRouter(config)
        self.shared_experts = GlmMLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            prefix="shared_experts",
        )

    @property
    def routed_experts(self) -> GlmExperts:
        return self.experts

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        original_shape = x_BLD.shape
        x_TD = x_BLD.view(-1, original_shape[-1])
        topk_index_TK, topk_weights_TK = self.gate(x_BLD)
        routed_TD = self.experts(x_TD, topk_index_TK, topk_weights_TK)
        routed_BLD = routed_TD.view(original_shape)
        return routed_BLD + self.shared_experts(x_BLD)


class GlmIndexPool(nn.Module):
    def __init__(self, config: Glm53FlashConfig) -> None:
        super().__init__()
        self.index_kpool = config.index_kpool
        self.head_dim = config.index_head_dim
        self.index_kpool_compress_ape = nn.Parameter(
            torch.empty(config.index_kpool, config.index_head_dim)
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(config.index_head_dim, config.hidden_size)
        )

    def forward(
        self,
        k_TK: torch.Tensor,
        gate_TK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if cu_seqlens is None:
            cu_seqlens = torch.tensor(
                [0, k_TK.shape[0]], dtype=torch.long, device=k_TK.device
            )
        pooled: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        valid: list[torch.Tensor] = []
        for doc_idx in range(cu_seqlens.numel() - 1):
            start = int(cu_seqlens[doc_idx].item())
            end = int(cu_seqlens[doc_idx + 1].item())
            doc_pooled, doc_indices, doc_valid = self._pool_one_doc(
                k_TK[start:end], gate_TK[start:end], token_offset=start
            )
            pooled.append(doc_pooled)
            indices.append(doc_indices)
            valid.append(doc_valid)
        if not pooled:
            return (
                k_TK.new_zeros(0, self.head_dim),
                torch.empty(0, self.index_kpool, dtype=torch.long, device=k_TK.device),
                torch.empty(0, dtype=torch.bool, device=k_TK.device),
            )
        return torch.cat(pooled), torch.cat(indices), torch.cat(valid)

    def _pool_one_doc(
        self,
        k_LK: torch.Tensor,
        gate_LK: torch.Tensor,
        *,
        token_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ratio = self.index_kpool
        num_pools = (k_LK.shape[0] + ratio - 1) // ratio
        if num_pools == 0:
            return (
                k_LK.new_zeros(0, self.head_dim),
                torch.empty(0, ratio, dtype=torch.long, device=k_LK.device),
                torch.empty(0, dtype=torch.bool, device=k_LK.device),
            )
        offsets_PR = torch.arange(num_pools * ratio, device=k_LK.device).view(
            num_pools, ratio
        )
        valid_PR = offsets_PR < k_LK.shape[0]
        safe_PR = offsets_PR.clamp(0, max(k_LK.shape[0] - 1, 0))
        grouped_keys_PRK = k_LK[safe_PR]
        grouped_gate_PRK = gate_LK[safe_PR]
        logits_PRK = grouped_gate_PRK.float() + self.index_kpool_compress_ape.float()
        logits_PRK = logits_PRK.masked_fill(~valid_PR[..., None], float("-inf"))
        probs_PRK = torch.nan_to_num(logits_PRK.softmax(dim=1)).to(k_LK.dtype)
        pooled_PK = (probs_PRK * grouped_keys_PRK).sum(dim=1)
        indices_PR = offsets_PR + token_offset
        indices_PR = indices_PR.masked_fill(~valid_PR, -1)
        pool_valid_P = valid_PR.all(dim=1)
        return pooled_PK, indices_PR, pool_valid_P


class GlmSparseIndexer(nn.Module):
    def __init__(self, config: Glm53FlashConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.index_kpool = config.index_kpool
        self.index_kpool_always_select_tail = config.index_kpool_always_select_tail
        self.softmax_scale = self.head_dim**-0.5
        self.wq_b = _linear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            config=config,
            suffix="indexer.wq_b",
        )
        self.wk = _linear(
            config.hidden_size,
            config.index_head_dim,
            config=config,
            suffix="indexer.wk",
        )
        self.k_norm = nn.LayerNorm(config.index_head_dim, eps=config.layer_norm_eps)
        self.weights_proj = _linear(
            config.hidden_size,
            config.index_n_heads,
            config=config,
            suffix="indexer.weights_proj",
        )
        self.index_pool = GlmIndexPool(config)

    @property
    def index_kpool_compress_ape(self) -> nn.Parameter:
        return self.index_pool.index_kpool_compress_ape

    @property
    def index_kpool_compress_gate(self) -> nn.Parameter:
        return self.index_pool.index_kpool_compress_gate

    @torch.no_grad()
    def forward(
        self,
        hidden_states_BLD: torch.Tensor,
        q_resid_BLR: torch.Tensor,
        attention_mask_BL: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states_BLD.shape[:2]
        q_BLHK = self.wq_b(q_resid_BLR).view(
            batch_size, seq_len, self.n_heads, self.head_dim
        )
        k_BLK = self.k_norm(self.wk(hidden_states_BLD))
        gate_BLK = F.linear(hidden_states_BLD, self.index_kpool_compress_gate)
        packed_BLW = torch.cat(
            [k_BLK, gate_BLK, attention_mask_BL.to(k_BLK.dtype).unsqueeze(-1)],
            dim=-1,
        )

        valid_keys_BL = packed_BLW[..., -1].bool()
        visible_BLL = self.get_visible_tokens(valid_keys_BL, seq_len)
        pool_keys_BPK, pool_indices_BPR, pool_valid_BP = self.get_pooled_states(
            packed_BLW
        )
        scores_BLHP = torch.matmul(
            q_BLHK.float(), pool_keys_BPK.transpose(-1, -2).float().unsqueeze(1)
        )
        scores_BLHP = F.relu(scores_BLHP * self.softmax_scale)
        weights_BLH = self.weights_proj(hidden_states_BLD).float() * (
            self.n_heads**-0.5
        )
        index_scores_BLP = torch.matmul(weights_BLH.unsqueeze(-2), scores_BLHP).squeeze(
            -2
        )

        if pool_indices_BPR.shape[-2] == 0:
            topk_BLS = torch.empty(
                batch_size,
                seq_len,
                0,
                dtype=torch.long,
                device=hidden_states_BLD.device,
            )
            valid_candidates_BLP = torch.empty(
                batch_size,
                seq_len,
                0,
                dtype=torch.bool,
                device=hidden_states_BLD.device,
            )
        else:
            pool_end_BP = pool_indices_BPR[..., -1].clamp(0, max(seq_len - 1, 0))
            pool_visible_BLP = visible_BLL.gather(
                dim=-1,
                index=pool_end_BP[:, None, :].expand(batch_size, seq_len, -1),
            )
            valid_candidates_BLP = pool_visible_BLP & pool_valid_BP[:, None, :]
            index_scores_BLP = index_scores_BLP.masked_fill(
                ~valid_candidates_BLP, torch.finfo(index_scores_BLP.dtype).min
            )
            select_k = min(
                self.index_topk // self.index_kpool, index_scores_BLP.shape[-1]
            )
            selected_BLS = index_scores_BLP.topk(select_k, dim=-1).indices
            batch_idx = torch.arange(
                hidden_states_BLD.shape[0], device=hidden_states_BLD.device
            )[:, None, None]
            selected_valid_BLS = valid_candidates_BLP.gather(-1, selected_BLS)
            selected_indices_BLSR = pool_indices_BPR[batch_idx, selected_BLS]
            topk_BLS = selected_indices_BLSR.flatten(-2)
            topk_BLS = topk_BLS.masked_fill(
                ~selected_valid_BLS[..., None]
                .expand_as(selected_indices_BLSR)
                .flatten(-2),
                -1,
            )

        output_width = self.index_topk
        if self.index_kpool_always_select_tail:
            topk_BLS = self.append_visible_tail(topk_BLS, visible_BLL, valid_keys_BL)
            output_width += self.index_kpool - 1
        topk_BLS = F.pad(topk_BLS, (0, output_width - topk_BLS.shape[-1]), value=-1)
        topk_BLS = topk_BLS[..., :output_width]
        topk_BLS = topk_BLS.masked_fill(~attention_mask_BL[..., None], -1)
        return topk_BLS.to(torch.int32)

    @staticmethod
    def get_visible_tokens(valid_keys_BL: torch.Tensor, q_length: int) -> torch.Tensor:
        device = valid_keys_BL.device
        kv_positions_L = torch.arange(valid_keys_BL.shape[-1], device=device)
        q_positions_L = torch.arange(q_length, device=device)
        causal_BLL = kv_positions_L[None, None, :] <= q_positions_L[None, :, None]
        return causal_BLL & valid_keys_BL[:, None, :]

    def get_pooled_states(
        self, packed_BLW: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys_BLK, gate_BLK, valid_BL1 = torch.split(
            packed_BLW, [self.head_dim, self.head_dim, 1], dim=-1
        )
        valid_BL = valid_BL1.bool().squeeze(-1)
        batch_size, seq_len = keys_BLK.shape[:2]
        num_pools = (seq_len + self.index_kpool - 1) // self.index_kpool
        device = keys_BLK.device
        first_key_B = torch.where(
            valid_BL.any(-1),
            valid_BL.long().argmax(-1),
            torch.full((batch_size,), seq_len, dtype=torch.long, device=device),
        )
        offsets_PR = torch.arange(num_pools * self.index_kpool, device=device).view(
            1, num_pools, self.index_kpool
        )
        pool_indices_BPR = first_key_B[:, None, None] + offsets_PR
        batch_idx = torch.arange(batch_size, device=device)[:, None, None]
        safe_BPR = pool_indices_BPR.clamp(0, max(seq_len - 1, 0))
        grouped_keys_BPRK = keys_BLK[batch_idx, safe_BPR]
        grouped_gate_BPRK = gate_BLK[batch_idx, safe_BPR]
        grouped_valid_BPR = valid_BL[batch_idx, safe_BPR] & (pool_indices_BPR < seq_len)
        pool_valid_BP = grouped_valid_BPR.all(-1)
        pool_indices_BPR = pool_indices_BPR.masked_fill(~grouped_valid_BPR, -1)
        logits_BPRK = (
            grouped_gate_BPRK.float()
            + self.index_kpool_compress_ape.float()[None, None]
        )
        logits_BPRK = logits_BPRK.masked_fill(
            ~grouped_valid_BPR[..., None], float("-inf")
        )
        probs_BPRK = torch.nan_to_num(logits_BPRK.softmax(dim=2)).to(keys_BLK.dtype)
        pool_keys_BPK = (probs_BPRK * grouped_keys_BPRK).sum(dim=2)
        keep_P = pool_valid_BP.any(0)
        return (
            pool_keys_BPK[:, keep_P],
            pool_indices_BPR[:, keep_P],
            pool_valid_BP[:, keep_P],
        )

    def append_visible_tail(
        self,
        topk_BLS: torch.Tensor,
        token_visible_BLL: torch.Tensor,
        key_valid_BL: torch.Tensor,
    ) -> torch.Tensor:
        max_tail_width = self.index_kpool - 1
        if max_tail_width == 0:
            return topk_BLS
        batch_size, _, kv_length = token_visible_BLL.shape
        device = token_visible_BLL.device
        first_key_B = torch.where(
            key_valid_BL.any(-1),
            key_valid_BL.long().argmax(-1),
            torch.full((batch_size,), kv_length, dtype=torch.long, device=device),
        )
        visible_count_BL = token_visible_BLL.long().sum(-1)
        tail_count_BL = visible_count_BL.remainder(self.index_kpool)
        tail_offsets_S = torch.arange(max_tail_width, device=device)
        tail_start_BL = first_key_B[:, None] + visible_count_BL - tail_count_BL
        tail_indices_BLS = tail_start_BL[..., None] + tail_offsets_S
        tail_valid_BLS = (tail_offsets_S[None, None, :] < tail_count_BL[..., None]) & (
            tail_indices_BLS < kv_length
        )
        kv_idx_BLS = tail_indices_BLS.clamp(0, max(kv_length - 1, 0))
        tail_visible_BLS = token_visible_BLL.gather(dim=-1, index=kv_idx_BLS)
        tail_indices_BLS = tail_indices_BLS.masked_fill(
            ~(tail_valid_BLS & tail_visible_BLS), -1
        )
        return torch.cat([topk_BLS, tail_indices_BLS], dim=-1)


class GlmKDAAttention(nn.Module):
    def __init__(self, config: Glm53FlashConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.qkv_dim = self.num_heads * self.head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.safe_gate_lower_bound = config.linear_lower_bound
        self.q_proj = _linear(
            config.hidden_size, self.qkv_dim, config=config, suffix="q_proj"
        )
        self.k_proj = _linear(
            config.hidden_size, self.qkv_dim, config=config, suffix="k_proj"
        )
        self.v_proj = _linear(
            config.hidden_size, self.qkv_dim, config=config, suffix="v_proj"
        )
        self.q_conv1d = nn.Conv1d(
            self.qkv_dim,
            self.qkv_dim,
            self.conv_kernel_size,
            groups=self.qkv_dim,
            bias=False,
        )
        self.k_conv1d = nn.Conv1d(
            self.qkv_dim,
            self.qkv_dim,
            self.conv_kernel_size,
            groups=self.qkv_dim,
            bias=False,
        )
        self.v_conv1d = nn.Conv1d(
            self.qkv_dim,
            self.qkv_dim,
            self.conv_kernel_size,
            groups=self.qkv_dim,
            bias=False,
        )
        self.f_a_proj = _linear(
            config.hidden_size, self.head_dim, config=config, suffix="f_a_proj"
        )
        self.f_b_proj = _linear(
            self.head_dim, self.qkv_dim, config=config, suffix="f_b_proj"
        )
        self.dt_bias = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.b_proj = _linear(
            config.hidden_size, self.num_heads, config=config, suffix="b_proj"
        )
        self.g_a_proj = _linear(
            config.hidden_size, self.head_dim, config=config, suffix="g_a_proj"
        )
        self.g_b_proj = _linear(
            self.head_dim, self.qkv_dim, config=config, suffix="g_b_proj"
        )
        self.o_norm = GlmGatedRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = _linear(
            self.qkv_dim, config.hidden_size, config=config, suffix="o_proj"
        )
        self.last_attention_path = "unrun"

    def forward(
        self,
        hidden_states_BLD: torch.Tensor,
        *,
        attention_mask_BL: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask_BL is not None:
            hidden_states_BLD = hidden_states_BLD * attention_mask_BL[..., None]
        query_BLC = self.q_proj(hidden_states_BLD)
        key_BLC = self.k_proj(hidden_states_BLD)
        value_BLC = self.v_proj(hidden_states_BLD)
        query_BLC = self._causal_conv(query_BLC, self.q_conv1d.weight)
        key_BLC = self._causal_conv(key_BLC, self.k_conv1d.weight)
        value_BLC = self._causal_conv(value_BLC, self.v_conv1d.weight)
        batch_size, seq_len = hidden_states_BLD.shape[:2]
        q_BLHK = query_BLC.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_BLHK = key_BLC.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_BLHK = value_BLC.view(batch_size, seq_len, self.num_heads, self.head_dim)
        raw_gate_BLHK = self.f_b_proj(self.f_a_proj(hidden_states_BLD)).view_as(q_BLHK)
        beta_BLH = torch.sigmoid(self.b_proj(hidden_states_BLD))

        if hidden_states_BLD.is_cuda:
            try:
                out_BLHK = self._fast_kda(
                    q_BLHK, k_BLHK, v_BLHK, raw_gate_BLHK, beta_BLH
                )
                self.last_attention_path = "attn_gym_kda"
            except Exception as exc:  # pragma: no cover - hardware/package dependent
                logger.warning("falling back to reference KDA: %s", exc)
                out_BLHK = self._reference_kda(
                    q_BLHK, k_BLHK, v_BLHK, raw_gate_BLHK, beta_BLH
                )
                self.last_attention_path = "reference"
        else:
            out_BLHK = self._reference_kda(
                q_BLHK, k_BLHK, v_BLHK, raw_gate_BLHK, beta_BLH
            )
            self.last_attention_path = "reference"

        gate_BLHK = self.g_b_proj(self.g_a_proj(hidden_states_BLD)).view_as(out_BLHK)
        out_BLC = self.o_norm(out_BLHK, gate_BLHK).reshape(batch_size, seq_len, -1)
        return self.o_proj(out_BLC)

    def _causal_conv(
        self, x_BLC: torch.Tensor, weight_C1W: torch.Tensor
    ) -> torch.Tensor:
        x_BCL = x_BLC.transpose(1, 2)
        out_BCL = F.conv1d(
            x_BCL,
            weight_C1W,
            padding=self.conv_kernel_size - 1,
            groups=x_BCL.shape[1],
        )
        out_BCL = out_BCL[..., : x_BCL.shape[-1]]
        return F.silu(out_BCL).transpose(1, 2)

    def _prepare_gate(self, raw_gate_BLHK: torch.Tensor) -> torch.Tensor:
        gate_BLHK = raw_gate_BLHK.float() + self.dt_bias.float().view(
            1, 1, self.num_heads, self.head_dim
        )
        decay_11H1 = torch.exp(self.A_log.float()).view(1, 1, self.num_heads, 1)
        return self.safe_gate_lower_bound * torch.sigmoid(decay_11H1 * gate_BLHK)

    def _fast_kda(
        self,
        q_BLHK: torch.Tensor,
        k_BLHK: torch.Tensor,
        v_BLHK: torch.Tensor,
        raw_gate_BLHK: torch.Tensor,
        beta_BLH: torch.Tensor,
    ) -> torch.Tensor:
        from torchtitan.models.common.attn_gym import bound_gate, chunk_kda, l2norm

        gate_BLHK = bound_gate(
            raw_gate_BLHK,
            self.A_log.float(),
            self.dt_bias.float(),
            lower_bound=self.safe_gate_lower_bound,
            impl="fused",
        )
        out_BLHK, _ = chunk_kda(
            l2norm(q_BLHK),
            l2norm(k_BLHK),
            v_BLHK,
            gate_BLHK,
            beta_BLH.float(),
            impl="fused",
            autotune=not torch.are_deterministic_algorithms_enabled(),
        )
        return out_BLHK

    def _reference_kda(
        self,
        q_BLHK: torch.Tensor,
        k_BLHK: torch.Tensor,
        v_BLHK: torch.Tensor,
        raw_gate_BLHK: torch.Tensor,
        beta_BLH: torch.Tensor,
    ) -> torch.Tensor:
        output_dtype = v_BLHK.dtype
        q_BLHK = _l2norm(q_BLHK).float()
        k_BLHK = _l2norm(k_BLHK).float()
        v_BLHK = v_BLHK.float()
        gate_BLHK = self._prepare_gate(raw_gate_BLHK)
        beta_BLH = beta_BLH.float()
        state_BHKK = torch.zeros(
            q_BLHK.shape[0],
            self.num_heads,
            self.head_dim,
            self.head_dim,
            dtype=torch.float32,
            device=q_BLHK.device,
        )
        outs: list[torch.Tensor] = []
        scale = self.head_dim**-0.5
        for t in range(q_BLHK.shape[1]):
            decay_BHK = torch.exp(gate_BLHK[:, t])
            state_BHKK = decay_BHK.unsqueeze(-1) * state_BHKK
            k_BHK = k_BLHK[:, t]
            v_BHV = v_BLHK[:, t]
            old_BHV = torch.einsum("bhkv,bhk->bhv", state_BHKK, k_BHK)
            delta_BHV = beta_BLH[:, t].unsqueeze(-1) * (v_BHV - old_BHV)
            state_BHKK = state_BHKK + torch.einsum("bhk,bhv->bhkv", k_BHK, delta_BHV)
            outs.append(torch.einsum("bhkv,bhk->bhv", state_BHKK, q_BLHK[:, t] * scale))
        return torch.stack(outs, dim=1).to(output_dtype)


class GlmSparseMLAAttention(nn.Module):
    def __init__(self, config: Glm53FlashConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.scaling = self.qk_nope_head_dim**-0.5
        self.q_a_proj = _linear(
            config.hidden_size, config.q_lora_rank, config=config, suffix="q_a_proj"
        )
        self.q_a_layernorm = GlmRMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = _linear(
            config.q_lora_rank,
            config.num_attention_heads * config.qk_nope_head_dim,
            config=config,
            suffix="q_b_proj",
        )
        self.kv_a_proj_with_mqa = _linear(
            config.hidden_size,
            config.kv_lora_rank,
            config=config,
            suffix="kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = GlmRMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = _linear(
            config.kv_lora_rank,
            config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
            config=config,
            suffix="kv_b_proj",
        )
        self.o_proj = _linear(
            config.num_attention_heads * config.v_head_dim,
            config.hidden_size,
            config=config,
            suffix="o_proj",
        )
        self.indexer = GlmSparseIndexer(config)
        self.last_attention_path = "unrun"

    def forward(
        self,
        hidden_states_BLD: torch.Tensor,
        *,
        attention_mask_BL: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states_BLD.shape[:2]
        q_resid_BLR = self.q_a_layernorm(self.q_a_proj(hidden_states_BLD))
        q_nope_BLHK = self.q_b_proj(q_resid_BLR).view(
            batch_size, seq_len, self.num_heads, self.qk_nope_head_dim
        )
        kv_latent_BLR = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states_BLD))
        topk_BLS = self.indexer(
            hidden_states_BLD.detach(),
            q_resid_BLR.detach(),
            attention_mask_BL,
        )

        kv_weight_HXR = self.kv_b_proj.weight.view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        k_weight_HKR = kv_weight_HXR[:, : self.qk_nope_head_dim]
        v_weight_HVR = kv_weight_HXR[:, self.qk_nope_head_dim :]
        q_latent_BLHR = torch.einsum("blhk,hkr->blhr", q_nope_BLHK, k_weight_HKR)
        if (
            self.config.attn_impl == "fast"
            and hidden_states_BLD.is_cuda
            and self.kv_lora_rank == 512
        ):
            try:
                out_latent_BLHR, path = self._fast_gather_attention(
                    q_latent_BLHR, kv_latent_BLR, topk_BLS
                )
                self.last_attention_path = path
            except Exception as exc:  # pragma: no cover - hardware/package dependent
                logger.warning("falling back to reference sparse MLA: %s", exc)
                out_latent_BLHR = self._reference_attention(
                    q_latent_BLHR, kv_latent_BLR, topk_BLS
                )
                self.last_attention_path = "reference"
        else:
            out_latent_BLHR = self._reference_attention(
                q_latent_BLHR, kv_latent_BLR, topk_BLS
            )
            self.last_attention_path = "reference"
        out_BLHV = torch.einsum("blhr,hvr->blhv", out_latent_BLHR, v_weight_HVR)
        return self.o_proj(out_BLHV.reshape(batch_size, seq_len, -1))

    def _reference_attention(
        self,
        q_latent_BLHR: torch.Tensor,
        kv_latent_BLR: torch.Tensor,
        topk_BLS: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, num_heads, _ = q_latent_BLHR.shape
        valid_BLS = topk_BLS.ge(0) & topk_BLS.lt(seq_len)
        safe_BLS = topk_BLS.clamp(0, max(seq_len - 1, 0)).long()
        selected_counts_BLL = torch.zeros(
            batch_size, seq_len, seq_len, dtype=torch.int32, device=q_latent_BLHR.device
        )
        selected_counts_BLL.scatter_add_(-1, safe_BLS, valid_BLS.to(torch.int32))
        selected_BLL = selected_counts_BLL.ne(0)
        scores_BHLL = torch.einsum(
            "blhr,bsr->bhls", q_latent_BLHR.float(), kv_latent_BLR.float()
        )
        scores_BHLL = scores_BHLL * self.scaling
        scores_BHLL = scores_BHLL.masked_fill(~selected_BLL[:, None], float("-inf"))
        probs_BHLL = torch.softmax(scores_BHLL, dim=-1, dtype=torch.float32).to(
            q_latent_BLHR.dtype
        )
        probs_BHLL = torch.nan_to_num(probs_BHLL)
        return torch.einsum("bhls,bsr->blhr", probs_BHLL, kv_latent_BLR)

    def _fast_gather_attention(
        self,
        q_latent_BLHR: torch.Tensor,
        kv_latent_BLR: torch.Tensor,
        topk_BLS: torch.Tensor,
    ) -> tuple[torch.Tensor, str]:
        from torchtitan.models.common.attn_gym import gather_attn

        query_BHLR = q_latent_BLHR.transpose(1, 2)
        for impl in ("fused", "reference"):
            try:
                out_BHLR = gather_attn(
                    query_BHLR,
                    kv_latent_BLR[:, None],
                    kv_latent_BLR[:, None],
                    topk_BLS,
                    attention_sink=None,
                    sliding_window_size=0,
                    scale=self.scaling,
                    impl=impl,
                )
            except Exception:
                if impl == "reference":
                    raise
                continue
            return out_BHLR.transpose(1, 2), f"attn_gym_gather_attn_{impl}"
        raise RuntimeError("unreachable gather_attn dispatch")


class Glm53FlashDecoderLayer(nn.Module):
    def __init__(self, config: Glm53FlashConfig, *, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.block_type = config.layer_types[layer_idx]
        self.self_attn = (
            GlmKDAAttention(config)
            if self.block_type == "linear_attention"
            else GlmSparseMLAAttention(config)
        )
        self.input_layernorm = GlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GlmRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_hc = GlmHyperConnection(config)
        self.ffn_hc = GlmHyperConnection(config)
        self.mlp = (
            GlmMoE(config)
            if config.mlp_layer_types[layer_idx] == "sparse"
            else GlmMLP(
                config, intermediate_size=config.intermediate_size, prefix="mlp"
            )
        )
        self.moe_enabled = isinstance(self.mlp, GlmMoE)

    @property
    def moe(self) -> GlmMoE | None:
        return self.mlp if self.moe_enabled else None

    def forward(
        self,
        hidden_streams_BLND: torch.Tensor,
        *,
        attention_mask_BL: torch.Tensor,
    ) -> torch.Tensor:
        residual_BLND = hidden_streams_BLND
        post_BLN, comb_BLNN, hidden_states_BLD = self.attn_hc(hidden_streams_BLND)
        hidden_states_BLD = self.input_layernorm(hidden_states_BLD)
        if isinstance(self.self_attn, GlmKDAAttention):
            hidden_states_BLD = self.self_attn(
                hidden_states_BLD, attention_mask_BL=attention_mask_BL
            )
        else:
            hidden_states_BLD = self.self_attn(
                hidden_states_BLD, attention_mask_BL=attention_mask_BL
            )
        hidden_streams_BLND = glm_hc_post_mix(
            hidden_states_BLD, residual_BLND, post_BLN, comb_BLNN
        )

        residual_BLND = hidden_streams_BLND
        post_BLN, comb_BLNN, hidden_states_BLD = self.ffn_hc(hidden_streams_BLND)
        hidden_states_BLD = self.post_attention_layernorm(hidden_states_BLD)
        hidden_states_BLD = self.mlp(hidden_states_BLD)
        return glm_hc_post_mix(hidden_states_BLD, residual_BLND, post_BLN, comb_BLNN)


class Glm53FlashForCausalLM(BaseModel):
    supports_pipeline_parallel: ClassVar[bool] = False
    _skip_lm_head: bool = False

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        glm_config: Glm53FlashConfig = field(default_factory=debugmodel)
        experiment_name: str = "glm53_flash"
        experiment_flavor: str = "debugmodel"

        def update_from_config(self, *, config, **kwargs: Any) -> None:
            del kwargs
            seq_len = config.training.max_context_length
            if seq_len > self.glm_config.max_position_embeddings:
                raise ValueError(
                    "training.max_context_length "
                    f"({seq_len}) exceeds GLM-5.3-Flash max positions "
                    f"({self.glm_config.max_position_embeddings})"
                )

        def set_sharding_(self, parallelism: ParallelismConfig) -> None:
            from .sharding import validate_glm53_flash_parallelism

            validate_glm53_flash_parallelism(self.glm_config, parallelism)

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams = sum(param.numel() for param in model.parameters())
            sparse_layers = len(self.glm_config.sparse_layers)
            kda_layers = len(self.glm_config.kda_layers)
            active_experts = max(1, self.glm_config.num_experts_per_tok)
            dense_flops = 6 * nparams
            attention_flops = (
                kda_layers
                * self.glm_config.linear_num_heads
                * self.glm_config.linear_head_dim
                * self.glm_config.linear_head_dim
                + sparse_layers
                * self.glm_config.num_attention_heads
                * seq_len
                * self.glm_config.kv_lora_rank
            )
            moe_discount = max(1, self.glm_config.num_experts // active_experts)
            return nparams, max(1, dense_flops // moe_discount + attention_flops)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        cfg = config.glm_config
        cfg.validate()
        self.glm_config = cfg
        self.enable_weight_tying = cfg.tie_word_embeddings
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): Glm53FlashDecoderLayer(cfg, layer_idx=layer_idx)
                for layer_idx in range(cfg.num_hidden_layers)
            }
        )
        self.norm = GlmRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if self.enable_weight_tying:
            self.lm_head.weight = self.tok_embeddings.weight

    def parallelize(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
        local_compile_regions: list[str],
        ac_config: ActivationCheckpointingConfig | None,
        dump_folder: str,
    ) -> "Glm53FlashForCausalLM":
        apply_local_compile(local_compile_regions)
        with parallelism_context.activate_spmd():
            if ac_config is not None:
                self._apply_activation_checkpoint(ac_config, dump_folder=dump_folder)
            self._apply_fsdp(
                parallelism_context=parallelism_context,
                training=training,
                parallelism=parallelism,
            )
        return self

    def _apply_activation_checkpoint(
        self,
        ac_config: ActivationCheckpointingConfig,
        *,
        dump_folder: str,
    ) -> None:
        del dump_folder
        if ac_config.debug:
            raise ValueError(
                "GLM-5.3-Flash experiment activation checkpointing does not "
                "support debug=True."
            )
        if ac_config.preserve_rng_state:
            raise ValueError(
                "GLM-5.3-Flash experiment activation checkpointing does not "
                "support preserve_rng_state=True."
            )
        for layer_id, layer in self.layers.items():
            layer.forward = remat.checkpoint(
                region_name=f"layers.{layer_id}",
                determinism_check=ac_config.determinism_check,
                preserve_rng_state=False,
            )(layer.forward)
        logger.info(
            "Applied full-block activation checkpointing to %d GLM layers",
            len(self.layers),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        if attention_mask is None:
            attention_mask = torch.ones_like(tokens, dtype=torch.bool)
        h_BLD = self.tok_embeddings(tokens)
        h_BLND = (
            h_BLD.unsqueeze(2).expand(-1, -1, self.glm_config.hc_mult, -1).contiguous()
        )
        for layer in self.layers.values():
            h_BLND = layer(h_BLND, attention_mask_BL=attention_mask)
        h_BLD = h_BLND.mean(dim=2)
        h_BLD = self.norm(h_BLD)
        h_TD = h_BLD.reshape(-1, h_BLD.shape[-1])
        if self._skip_lm_head:
            return h_TD
        return self.lm_head(h_TD)

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        del kwargs
        labels = input_dict.pop("labels")
        tokens = input_dict.pop("input")
        padding_mask = input_dict.pop("padding_mask", None)
        attention_mask = (
            torch.ones_like(tokens, dtype=torch.bool)
            if padding_mask is None
            else ~padding_mask.bool()
        )
        positions = input_dict.pop("positions", None)
        return (
            tokens,
            labels.reshape(-1),
            {
                "attention_mask": attention_mask,
                "positions": positions,
                **input_dict,
            },
        )

    def _apply_fsdp(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        from torchtitan.distributed.fsdp import (
            apply_fsdp_to_decoder,
            resolve_fsdp_mesh,
            resolve_sparse_fsdp_mesh,
        )

        dp_mesh, _ = resolve_fsdp_mesh(parallelism_context)
        edp_mesh, _ = resolve_sparse_fsdp_mesh(parallelism_context)
        apply_fsdp_to_decoder(
            self,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=False,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallelism_context.ep,
            edp_mesh=edp_mesh,
            symm_mem_scope=parallelism.fsdp_symm_mem_scope,
        )

    def verify_module_protocol(self) -> None:
        return

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        del buffer_device
        if self.enable_weight_tying:
            self.lm_head.weight = self.tok_embeddings.weight
        _initialize_glm53_flash_weights(self, self.glm_config)


def _initialize_glm53_flash_weights(model: nn.Module, config: Glm53FlashConfig) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if module.rank is not None:
                    assert module.lora_a is not None and module.lora_b is not None
                    nn.init.normal_(module.lora_a, mean=0.0, std=0.01)
                    nn.init.zeros_(module.lora_b)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (GlmRMSNorm, GlmGatedRMSNorm)):
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, GlmHyperConnection):
                nn.init.normal_(module.fn, mean=0.0, std=0.02)
                nn.init.zeros_(module.base)
                nn.init.ones_(module.scale)
            elif isinstance(module, GlmExperts):
                nn.init.normal_(
                    module.gate_up_proj, mean=0.0, std=config.initializer_range
                )
                nn.init.normal_(
                    module.down_proj, mean=0.0, std=config.initializer_range
                )
            elif isinstance(module, GlmTopKRouter):
                nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
                module.e_score_correction_bias.zero_()
            elif isinstance(module, GlmIndexPool):
                nn.init.zeros_(module.index_kpool_compress_ape)
                nn.init.ones_(module.index_kpool_compress_gate)
            elif isinstance(module, GlmKDAAttention):
                nn.init.zeros_(module.A_log)
                dt = torch.empty_like(module.dt_bias).uniform_(1.0e-3, 1.0e-1)
                module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))


from .state_dict_adapter import Glm53FlashStateDictAdapter


Glm53FlashForCausalLM.state_dict_adapter_cls = Glm53FlashStateDictAdapter
