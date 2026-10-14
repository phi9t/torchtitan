# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Softmax attention, GLM-style sparse top-k selection, and IndexPool.

Shape suffixes:
    B = batch, L = sequence length, H = attention heads, D = model dimension,
    K = head dimension, P = index pools, R = tokens per pool, S = selections.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import SpeedrunElementsConfig
from .positional import maybe_apply_rope


def _default_mask(
    x_BLD: torch.Tensor, attention_mask_BL: torch.Tensor | None
) -> torch.Tensor:
    if attention_mask_BL is None:
        return torch.ones(x_BLD.shape[:2], dtype=torch.bool, device=x_BLD.device)
    return attention_mask_BL.bool()


def causal_token_indices(
    batch_size: int,
    seq_len: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    positions_L = torch.arange(seq_len, device=device)
    indices_BLL = positions_L.view(1, 1, seq_len).expand(batch_size, seq_len, seq_len)
    valid_BLL = positions_L.view(1, seq_len, 1) >= positions_L.view(1, 1, seq_len)
    return indices_BLL.masked_fill(~valid_BLL, -1).to(torch.int32)


def dense_causal_attention(
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    v_BLHK: torch.Tensor,
    *,
    attention_mask_BL: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    batch_size, seq_len = q_BLHK.shape[:2]
    if scale is None:
        scale = q_BLHK.shape[-1] ** -0.5
    valid_BL = _default_mask(q_BLHK.flatten(-2), attention_mask_BL)
    positions_L = torch.arange(seq_len, device=q_BLHK.device)
    causal_LL = positions_L[:, None] >= positions_L[None, :]
    visible_BLL = causal_LL.unsqueeze(0) & valid_BL[:, None, :]
    scores_BHLL = torch.einsum("blhk,bshk->bhls", q_BLHK.float(), k_BLHK.float())
    scores_BHLL = scores_BHLL * scale
    scores_BHLL = scores_BHLL.masked_fill(~visible_BLL[:, None], float("-inf"))
    probs_BHLL = torch.nan_to_num(
        torch.softmax(scores_BHLL, dim=-1, dtype=torch.float32)
    )
    out_BLHK = torch.einsum("bhls,bshk->blhk", probs_BHLL.to(v_BLHK.dtype), v_BLHK)
    return out_BLHK * valid_BL[:, :, None, None].to(out_BLHK.dtype)


def sparse_causal_attention(
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    v_BLHK: torch.Tensor,
    selected_BLS: torch.Tensor,
    *,
    attention_mask_BL: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    batch_size, seq_len = q_BLHK.shape[:2]
    if selected_BLS.shape[:2] != (batch_size, seq_len):
        raise ValueError(
            "selected_BLS must have shape [B, L, S], "
            f"got {tuple(selected_BLS.shape)} for q {tuple(q_BLHK.shape)}."
        )
    if scale is None:
        scale = q_BLHK.shape[-1] ** -0.5
    valid_BL = _default_mask(q_BLHK.flatten(-2), attention_mask_BL)
    selected_BLS = selected_BLS.to(device=q_BLHK.device)
    valid_BLS = selected_BLS.ge(0) & selected_BLS.lt(seq_len)
    safe_BLS = selected_BLS.clamp(0, max(seq_len - 1, 0)).long()
    selected_counts_BLL = torch.zeros(
        batch_size,
        seq_len,
        seq_len,
        dtype=torch.int32,
        device=q_BLHK.device,
    )
    selected_counts_BLL.scatter_add_(-1, safe_BLS, valid_BLS.to(torch.int32))
    positions_L = torch.arange(seq_len, device=q_BLHK.device)
    causal_LL = positions_L[:, None] >= positions_L[None, :]
    selected_BLL = (
        selected_counts_BLL.ne(0) & causal_LL.unsqueeze(0) & valid_BL[:, None, :]
    )
    scores_BHLL = torch.einsum("blhk,bshk->bhls", q_BLHK.float(), k_BLHK.float())
    scores_BHLL = scores_BHLL * scale
    scores_BHLL = scores_BHLL.masked_fill(~selected_BLL[:, None], float("-inf"))
    probs_BHLL = torch.nan_to_num(
        torch.softmax(scores_BHLL, dim=-1, dtype=torch.float32)
    )
    out_BLHK = torch.einsum("bhls,bshk->blhk", probs_BHLL.to(v_BLHK.dtype), v_BLHK)
    return out_BLHK * valid_BL[:, :, None, None].to(out_BLHK.dtype)


class IndexPool(nn.Module):
    def __init__(self, head_dim: int, *, pool_size: int = 4) -> None:
        super().__init__()
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1.")
        self.head_dim = head_dim
        self.pool_size = pool_size
        self.slot_bias = nn.Parameter(torch.zeros(pool_size, head_dim))

    def forward(
        self,
        keys_BLK: torch.Tensor,
        gate_BLK: torch.Tensor,
        attention_mask_BL: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_BL = _default_mask(keys_BLK, attention_mask_BL)
        batch_size, seq_len = keys_BLK.shape[:2]
        if self.pool_size == 1:
            indices_BL1 = torch.arange(seq_len, device=keys_BLK.device).view(
                1, seq_len, 1
            )
            indices_BL1 = indices_BL1.expand(batch_size, seq_len, 1)
            return (
                keys_BLK,
                indices_BL1.masked_fill(~valid_BL[..., None], -1),
                valid_BL,
            )

        num_pools = math.ceil(seq_len / self.pool_size)
        pooled_keys = []
        pooled_indices = []
        pooled_valid = []
        offsets_PR = torch.arange(
            num_pools * self.pool_size,
            device=keys_BLK.device,
        ).view(num_pools, self.pool_size)
        for batch_idx in range(batch_size):
            first_key = (
                int(valid_BL[batch_idx].long().argmax().item())
                if valid_BL[batch_idx].any()
                else seq_len
            )
            raw_indices_PR = first_key + offsets_PR
            safe_indices_PR = raw_indices_PR.clamp(0, max(seq_len - 1, 0))
            valid_PR = (raw_indices_PR < seq_len) & valid_BL[batch_idx, safe_indices_PR]
            grouped_keys_PRK = keys_BLK[batch_idx, safe_indices_PR]
            grouped_gate_PRK = gate_BLK[batch_idx, safe_indices_PR]
            logits_PRK = grouped_gate_PRK.float() + self.slot_bias.float()
            logits_PRK = logits_PRK.masked_fill(~valid_PR[..., None], float("-inf"))
            probs_PRK = torch.nan_to_num(logits_PRK.softmax(dim=1)).to(keys_BLK.dtype)
            pooled_keys.append((probs_PRK * grouped_keys_PRK).sum(dim=1))
            pooled_indices.append(raw_indices_PR.masked_fill(~valid_PR, -1))
            pooled_valid.append(valid_PR.all(dim=1))
        return (
            torch.stack(pooled_keys),
            torch.stack(pooled_indices),
            torch.stack(pooled_valid),
        )


class LightningIndexer(nn.Module):
    """GLM/HF-style detached deterministic top-k selector for sparse attention."""

    def __init__(
        self,
        hidden_size: int,
        *,
        top_k: int,
        index_num_heads: int,
        index_head_dim: int,
        pool_size: int = 1,
        always_select_tail: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.index_num_heads = index_num_heads
        self.index_head_dim = index_head_dim
        self.pool_size = pool_size
        self.always_select_tail = always_select_tail
        self.softmax_scale = index_head_dim**-0.5
        self.q_proj = nn.Linear(
            hidden_size, index_num_heads * index_head_dim, bias=False
        )
        self.k_proj = nn.Linear(hidden_size, index_head_dim, bias=False)
        self.k_norm = nn.LayerNorm(index_head_dim, eps=1.0e-6)
        self.weights_proj = nn.Linear(hidden_size, index_num_heads, bias=False)
        self.gate_proj = nn.Linear(hidden_size, index_head_dim, bias=False)
        self.index_pool = IndexPool(index_head_dim, pool_size=pool_size)

    @torch.no_grad()
    def forward(
        self,
        hidden_BLD: torch.Tensor,
        attention_mask_BL: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_BLD.shape[:2]
        valid_BL = _default_mask(hidden_BLD, attention_mask_BL)
        if self.top_k >= seq_len:
            return causal_token_indices(batch_size, seq_len, device=hidden_BLD.device)

        q_BLHK = self.q_proj(hidden_BLD).view(
            batch_size,
            seq_len,
            self.index_num_heads,
            self.index_head_dim,
        )
        keys_BLK = self.k_norm(self.k_proj(hidden_BLD))
        gate_BLK = self.gate_proj(hidden_BLD)
        pool_keys_BPK, pool_indices_BPR, pool_valid_BP = self.index_pool(
            keys_BLK,
            gate_BLK,
            valid_BL,
        )
        num_pools = pool_keys_BPK.shape[1]
        if num_pools == 0:
            return torch.full(
                (batch_size, seq_len, self.top_k),
                -1,
                dtype=torch.int32,
                device=hidden_BLD.device,
            )

        scores_BLHP = torch.einsum(
            "blhk,bpk->blhp",
            q_BLHK.float(),
            pool_keys_BPK.float(),
        )
        scores_BLHP = F.relu(scores_BLHP * self.softmax_scale)
        head_weights_BLH = self.weights_proj(hidden_BLD).float()
        head_weights_BLH = head_weights_BLH * (self.index_num_heads**-0.5)
        scores_BLP = torch.einsum("blh,blhp->blp", head_weights_BLH, scores_BLHP)

        pool_last_BP = pool_indices_BPR[..., -1].clamp(0, max(seq_len - 1, 0))
        query_positions_L = torch.arange(seq_len, device=hidden_BLD.device)
        pool_visible_BLP = pool_last_BP[:, None, :] <= query_positions_L[None, :, None]
        valid_candidates_BLP = pool_valid_BP[:, None, :] & pool_visible_BLP
        scores_BLP = scores_BLP.masked_fill(
            ~valid_candidates_BLP,
            torch.finfo(scores_BLP.dtype).min,
        )
        num_pool_selections = min(math.ceil(self.top_k / self.pool_size), num_pools)
        selected_pools_BLS = scores_BLP.topk(num_pool_selections, dim=-1).indices
        batch_idx = torch.arange(batch_size, device=hidden_BLD.device)[:, None, None]
        selected_valid_BLS = valid_candidates_BLP.gather(-1, selected_pools_BLS)
        selected_indices_BLSR = pool_indices_BPR[batch_idx, selected_pools_BLS]
        selected_tokens_BLS = selected_indices_BLSR.flatten(-2)
        selected_tokens_BLS = selected_tokens_BLS.masked_fill(
            ~selected_valid_BLS[..., None].expand_as(selected_indices_BLSR).flatten(-2),
            -1,
        )
        selected_tokens_BLS = selected_tokens_BLS[..., : self.top_k]
        if self.always_select_tail and self.pool_size > 1:
            selected_tokens_BLS = self._append_tail(
                selected_tokens_BLS,
                valid_BL,
            )
        output_width = self.top_k
        if self.always_select_tail and self.pool_size > 1:
            output_width += self.pool_size - 1
        if selected_tokens_BLS.shape[-1] < output_width:
            selected_tokens_BLS = F.pad(
                selected_tokens_BLS,
                (0, output_width - selected_tokens_BLS.shape[-1]),
                value=-1,
            )
        selected_tokens_BLS = selected_tokens_BLS[..., :output_width]
        return selected_tokens_BLS.masked_fill(~valid_BL[..., None], -1).to(torch.int32)

    def _append_tail(
        self,
        selected_tokens_BLS: torch.Tensor,
        valid_BL: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = valid_BL.shape
        max_tail_width = self.pool_size - 1
        tail_offsets_S = torch.arange(max_tail_width, device=valid_BL.device)
        first_key_B = torch.where(
            valid_BL.any(-1),
            valid_BL.long().argmax(-1),
            torch.full(
                (batch_size,), seq_len, dtype=torch.long, device=valid_BL.device
            ),
        )
        visible_count_BL = valid_BL.long().cumsum(dim=-1)
        tail_count_BL = visible_count_BL.remainder(self.pool_size)
        tail_start_BL = first_key_B[:, None] + visible_count_BL - tail_count_BL
        tail_indices_BLS = tail_start_BL[..., None] + tail_offsets_S
        query_positions_L = torch.arange(seq_len, device=valid_BL.device)
        tail_valid_BLS = (tail_offsets_S[None, None, :] < tail_count_BL[..., None]) & (
            tail_indices_BLS <= query_positions_L[None, :, None]
        )
        tail_indices_BLS = tail_indices_BLS.masked_fill(~tail_valid_BLS, -1)
        return torch.cat([selected_tokens_BLS, tail_indices_BLS], dim=-1)


class TopKSparseCausalAttention(nn.Module):
    def __init__(self, config: SpeedrunElementsConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.indexer = LightningIndexer(
            config.hidden_size,
            top_k=config.sparse_top_k,
            index_num_heads=config.indexer_num_heads,
            index_head_dim=config.indexer_head_dim,
            pool_size=config.index_pool_size,
            always_select_tail=config.index_pool_always_select_tail,
        )

    def forward(
        self,
        x_BLD: torch.Tensor,
        *,
        attention_mask_BL: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = x_BLD.shape[:2]
        q_BLHK = self.q_proj(x_BLD).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k_BLHK = self.k_proj(x_BLD).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        v_BLHK = self.v_proj(x_BLD).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        q_BLHK, k_BLHK = maybe_apply_rope(
            q_BLHK,
            k_BLHK,
            rope_enabled=self.config.rope_enabled,
            theta=self.config.rope_theta,
            positions=positions,
        )
        if (
            not self.config.sparse_attention_enabled
            or self.config.sparse_top_k >= seq_len
        ):
            out_BLHK = dense_causal_attention(
                q_BLHK,
                k_BLHK,
                v_BLHK,
                attention_mask_BL=attention_mask_BL,
            )
        else:
            selected_BLS = self.indexer(x_BLD.detach(), attention_mask_BL)
            out_BLHK = sparse_causal_attention(
                q_BLHK,
                k_BLHK,
                v_BLHK,
                selected_BLS,
                attention_mask_BL=attention_mask_BL,
            )
        return self.o_proj(out_BLHK.reshape(batch_size, seq_len, self.hidden_size))
