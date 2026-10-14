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

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

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


def sparse_causal_attention_reference(
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


def _selected_token_mask(
    selected_BLS: torch.Tensor,
    valid_BL: torch.Tensor,
    *,
    seq_len: int,
) -> torch.Tensor:
    batch_size = selected_BLS.shape[0]
    selected_BLS = selected_BLS.to(device=valid_BL.device)
    valid_BLS = selected_BLS.ge(0) & selected_BLS.lt(seq_len)
    safe_BLS = selected_BLS.masked_fill(~valid_BLS, seq_len).long()
    selected_BLLX = torch.zeros(
        batch_size,
        seq_len,
        seq_len + 1,
        dtype=torch.bool,
        device=valid_BL.device,
    )
    selected_BLLX.scatter_(-1, safe_BLS, valid_BLS)
    selected_BLL = selected_BLLX[..., :seq_len]
    positions_L = torch.arange(seq_len, device=valid_BL.device)
    causal_LL = positions_L[:, None] >= positions_L[None, :]
    return selected_BLL & causal_LL.unsqueeze(0) & valid_BL[:, None, :]


def sparse_causal_attention_masked_dense(
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    v_BLHK: torch.Tensor,
    selected_BLS: torch.Tensor,
    *,
    attention_mask_BL: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Sparse causal attention through SDPA with a selected-key bool mask."""

    batch_size, seq_len = q_BLHK.shape[:2]
    if selected_BLS.shape[:2] != (batch_size, seq_len):
        raise ValueError(
            "selected_BLS must have shape [B, L, S], "
            f"got {tuple(selected_BLS.shape)} for q {tuple(q_BLHK.shape)}."
        )
    valid_BL = _default_mask(q_BLHK.flatten(-2), attention_mask_BL)
    selected_BLL = _selected_token_mask(selected_BLS, valid_BL, seq_len=seq_len)
    default_scale = q_BLHK.shape[-1] ** -0.5
    if scale is not None and scale != default_scale:
        q_BLHK = q_BLHK * (scale / default_scale)
    out_BHLK = F.scaled_dot_product_attention(
        q_BLHK.transpose(1, 2),
        k_BLHK.transpose(1, 2),
        v_BLHK.transpose(1, 2),
        attn_mask=selected_BLL[:, None],
        dropout_p=0.0,
    )
    out_BLHK = torch.nan_to_num(out_BHLK.transpose(1, 2).contiguous())
    return out_BLHK * valid_BL[:, :, None, None].to(out_BLHK.dtype)


def sparse_causal_attention(
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    v_BLHK: torch.Tensor,
    selected_BLS: torch.Tensor,
    *,
    attention_mask_BL: torch.Tensor | None = None,
    scale: float | None = None,
    query_chunk_size: int = 32,
) -> torch.Tensor:
    """Sparse causal attention by gathering selected K/V tokens in chunks."""

    batch_size, seq_len = q_BLHK.shape[:2]
    if selected_BLS.shape[:2] != (batch_size, seq_len):
        raise ValueError(
            "selected_BLS must have shape [B, L, S], "
            f"got {tuple(selected_BLS.shape)} for q {tuple(q_BLHK.shape)}."
        )
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be >= 1.")
    if scale is None:
        scale = q_BLHK.shape[-1] ** -0.5
    valid_BL = _default_mask(q_BLHK.flatten(-2), attention_mask_BL)
    selected_BLS = selected_BLS.to(device=q_BLHK.device)
    valid_BLS = selected_BLS.ge(0) & selected_BLS.lt(seq_len)
    safe_BLS = selected_BLS.clamp(0, max(seq_len - 1, 0)).long()
    batch_idx_B11 = torch.arange(batch_size, device=q_BLHK.device)[:, None, None]
    positions_L = torch.arange(seq_len, device=q_BLHK.device)
    chunks = []
    for start in range(0, seq_len, query_chunk_size):
        end = min(start + query_chunk_size, seq_len)
        safe_BCS = safe_BLS[:, start:end]
        selected_BCS = selected_BLS[:, start:end]
        query_positions_C = positions_L[start:end]
        key_valid_BCS = valid_BL[batch_idx_B11, safe_BCS]
        mask_BCS = (
            valid_BLS[:, start:end]
            & key_valid_BCS
            & (selected_BCS <= query_positions_C[None, :, None])
        )

        def _chunk_attention(
            q_BCHK: torch.Tensor,
            k_BLHK: torch.Tensor,
            v_BLHK: torch.Tensor,
        ) -> torch.Tensor:
            gathered_k_BCSHK = k_BLHK[batch_idx_B11, safe_BCS]
            gathered_v_BCSHK = v_BLHK[batch_idx_B11, safe_BCS]
            scores_BHCS = torch.einsum(
                "bchk,bcshk->bhcs",
                q_BCHK.float(),
                gathered_k_BCSHK.float(),
            )
            scores_BHCS = scores_BHCS * scale
            scores_BHCS = scores_BHCS.masked_fill(~mask_BCS[:, None], float("-inf"))
            probs_BHCS = torch.nan_to_num(
                torch.softmax(scores_BHCS, dim=-1, dtype=torch.float32)
            )
            return torch.einsum(
                "bhcs,bcshk->bchk",
                probs_BHCS.to(v_BLHK.dtype),
                gathered_v_BCSHK,
            )

        q_chunk_BCHK = q_BLHK[:, start:end]
        if torch.is_grad_enabled() and (
            q_chunk_BCHK.requires_grad or k_BLHK.requires_grad or v_BLHK.requires_grad
        ):
            out_BCHK = checkpoint(
                _chunk_attention,
                q_chunk_BCHK,
                k_BLHK,
                v_BLHK,
                use_reentrant=False,
            )
        else:
            out_BCHK = _chunk_attention(q_chunk_BCHK, k_BLHK, v_BLHK)
        chunks.append(out_BCHK)
    out_BLHK = torch.cat(chunks, dim=1)
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
        offsets_PR = torch.arange(
            num_pools * self.pool_size,
            device=keys_BLK.device,
        ).view(num_pools, self.pool_size)
        first_key_B = torch.where(
            valid_BL.any(dim=-1),
            valid_BL.long().argmax(dim=-1),
            torch.full((batch_size,), seq_len, dtype=torch.long, device=keys_BLK.device),
        )
        raw_indices_BPR = first_key_B[:, None, None] + offsets_PR[None]
        safe_indices_BPR = raw_indices_BPR.clamp(0, max(seq_len - 1, 0))
        flat_safe_BP_R = safe_indices_BPR.flatten(1)
        flat_valid_BP_R = valid_BL.gather(1, flat_safe_BP_R)
        valid_BPR = (
            raw_indices_BPR.lt(seq_len)
            & flat_valid_BP_R.view(batch_size, num_pools, self.pool_size)
        )
        gather_index_BPRK = safe_indices_BPR[..., None].expand(
            batch_size,
            num_pools,
            self.pool_size,
            self.head_dim,
        )
        grouped_keys_BPRK = keys_BLK[:, None].expand(
            batch_size,
            num_pools,
            seq_len,
            self.head_dim,
        ).gather(2, gather_index_BPRK)
        grouped_gate_BPRK = gate_BLK[:, None].expand(
            batch_size,
            num_pools,
            seq_len,
            self.head_dim,
        ).gather(2, gather_index_BPRK)
        logits_BPRK = grouped_gate_BPRK.float() + self.slot_bias.float()
        logits_BPRK = logits_BPRK.masked_fill(~valid_BPR[..., None], float("-inf"))
        probs_BPRK = torch.nan_to_num(logits_BPRK.softmax(dim=2)).to(keys_BLK.dtype)
        return (
            (probs_BPRK * grouped_keys_BPRK).sum(dim=2),
            raw_indices_BPR.masked_fill(~valid_BPR, -1),
            valid_BPR.all(dim=2),
        )


@dataclass(frozen=True)
class IndexerScoreState:
    scores: torch.Tensor
    pool_indices: torch.Tensor
    valid_candidates: torch.Tensor
    valid_queries: torch.Tensor


class LightningIndexer(nn.Module):
    """GLM/HF-style deterministic top-k selector for sparse attention.

    ``forward`` preserves the HF inference behavior and disables gradients.
    ``score_pools`` is differentiable for the indexer auxiliary objective.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        top_k: int,
        index_num_heads: int,
        index_head_dim: int,
        pool_size: int = 1,
        always_select_tail: bool = True,
        query_chunk_size: int = 64,
    ) -> None:
        super().__init__()
        if query_chunk_size < 1:
            raise ValueError("query_chunk_size must be >= 1.")
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.index_num_heads = index_num_heads
        self.index_head_dim = index_head_dim
        self.pool_size = pool_size
        self.always_select_tail = always_select_tail
        self.query_chunk_size = query_chunk_size
        self.softmax_scale = index_head_dim**-0.5
        self.q_proj = nn.Linear(
            hidden_size, index_num_heads * index_head_dim, bias=False
        )
        self.k_proj = nn.Linear(hidden_size, index_head_dim, bias=False)
        self.k_norm = nn.LayerNorm(index_head_dim, eps=1.0e-6)
        self.weights_proj = nn.Linear(hidden_size, index_num_heads, bias=False)
        self.gate_proj = nn.Linear(hidden_size, index_head_dim, bias=False)
        self.index_pool = IndexPool(index_head_dim, pool_size=pool_size)

    def score_pools(
        self,
        hidden_BLD: torch.Tensor,
        attention_mask_BL: torch.Tensor | None = None,
    ) -> IndexerScoreState:
        batch_size, seq_len = hidden_BLD.shape[:2]
        valid_BL = _default_mask(hidden_BLD, attention_mask_BL)
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
            return IndexerScoreState(
                scores=hidden_BLD.new_empty(batch_size, seq_len, 0),
                pool_indices=torch.empty(
                    batch_size,
                    0,
                    self.pool_size,
                    dtype=torch.long,
                    device=hidden_BLD.device,
                ),
                valid_candidates=torch.empty(
                    batch_size,
                    seq_len,
                    0,
                    dtype=torch.bool,
                    device=hidden_BLD.device,
                ),
                valid_queries=valid_BL,
            )

        head_weights_BLH = self.weights_proj(hidden_BLD).float()
        head_weights_BLH = head_weights_BLH * (self.index_num_heads**-0.5)
        scores_chunks = []
        for start in range(0, seq_len, self.query_chunk_size):
            end = min(start + self.query_chunk_size, seq_len)
            scores_BCHP = torch.einsum(
                "bchk,bpk->bchp",
                q_BLHK[:, start:end].float(),
                pool_keys_BPK.float(),
            )
            scores_BCHP = F.relu(scores_BCHP * self.softmax_scale)
            scores_chunks.append(
                torch.einsum(
                    "bch,bchp->bcp",
                    head_weights_BLH[:, start:end],
                    scores_BCHP,
                )
            )
        scores_BLP = torch.cat(scores_chunks, dim=1)

        pool_last_BP = pool_indices_BPR[..., -1].clamp(0, max(seq_len - 1, 0))
        query_positions_L = torch.arange(seq_len, device=hidden_BLD.device)
        pool_visible_BLP = pool_last_BP[:, None, :] <= query_positions_L[None, :, None]
        valid_candidates_BLP = pool_valid_BP[:, None, :] & pool_visible_BLP
        scores_BLP = scores_BLP.masked_fill(
            ~valid_candidates_BLP,
            torch.finfo(scores_BLP.dtype).min,
        )
        return IndexerScoreState(
            scores=scores_BLP,
            pool_indices=pool_indices_BPR,
            valid_candidates=valid_candidates_BLP,
            valid_queries=valid_BL,
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_BLD: torch.Tensor,
        attention_mask_BL: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_BLD.shape[:2]
        if self.top_k >= seq_len:
            return causal_token_indices(batch_size, seq_len, device=hidden_BLD.device)
        return self.select_from_scores(self.score_pools(hidden_BLD, attention_mask_BL))

    def select_from_scores(self, state: IndexerScoreState) -> torch.Tensor:
        batch_size, seq_len = state.valid_queries.shape
        pool_indices_BPR = state.pool_indices
        valid_candidates_BLP = state.valid_candidates
        scores_BLP = state.scores
        valid_BL = state.valid_queries
        num_pools = pool_indices_BPR.shape[1]
        num_pool_selections = min(math.ceil(self.top_k / self.pool_size), num_pools)
        selected_pools_BLS = scores_BLP.topk(num_pool_selections, dim=-1).indices
        batch_idx = torch.arange(batch_size, device=scores_BLP.device)[:, None, None]
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


def indexer_auxiliary_loss(
    indexer: LightningIndexer,
    hidden_BLD: torch.Tensor,
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    *,
    attention_mask_BL: torch.Tensor | None = None,
    target: str = "selected",
    query_chunk_size: int = 64,
    scale: float | None = None,
    state: IndexerScoreState | None = None,
) -> torch.Tensor:
    """KL from dense-attention teacher mass to differentiable indexer scores.

    The teacher path is detached. ``target="selected"`` trains only the current
    top-k pools; ``target="all"`` trains against every visible causal pool.
    """

    if target not in {"selected", "all"}:
        raise ValueError("target must be 'selected' or 'all'.")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be >= 1.")
    if scale is None:
        scale = q_BLHK.shape[-1] ** -0.5

    hidden_BLD = hidden_BLD.detach()
    q_teacher_BLHK = q_BLHK.detach().float()
    k_teacher_BLHK = k_BLHK.detach().float()
    if state is None:
        valid_BL = _default_mask(hidden_BLD, attention_mask_BL)
        state = indexer.score_pools(hidden_BLD, valid_BL)
    else:
        valid_BL = state.valid_queries
    batch_size, seq_len, num_pools = state.scores.shape
    if num_pools == 0:
        return state.scores.sum() * 0.0

    if target == "selected":
        num_pool_selections = min(
            math.ceil(indexer.top_k / indexer.pool_size),
            num_pools,
        )
        candidate_pools_BLC = state.scores.topk(
            num_pool_selections,
            dim=-1,
        ).indices
    else:
        candidate_pools_BLC = torch.arange(
            num_pools,
            device=hidden_BLD.device,
        ).view(1, 1, num_pools)
        candidate_pools_BLC = candidate_pools_BLC.expand(batch_size, seq_len, num_pools)

    positions_L = torch.arange(seq_len, device=hidden_BLD.device)
    pool_width = state.pool_indices.shape[-1]
    num_heads = q_BLHK.shape[2]
    batch_idx_B11 = torch.arange(batch_size, device=hidden_BLD.device).view(
        batch_size,
        1,
        1,
    )
    if target == "selected":
        losses = []
        batch_idx_B111 = batch_idx_B11[..., None]
        for start in range(0, seq_len, query_chunk_size):
            end = min(start + query_chunk_size, seq_len)
            chunk_size = end - start
            candidate_pools_BCP = candidate_pools_BLC[:, start:end]
            safe_pools_BCP = candidate_pools_BCP.clamp(0, num_pools - 1)
            candidate_logits_BCP = state.scores[:, start:end].gather(
                -1,
                safe_pools_BCP,
            )
            candidate_mask_BCP = state.valid_candidates[:, start:end].gather(
                -1,
                safe_pools_BCP,
            )
            pool_tokens_BCPR = state.pool_indices[batch_idx_B11, safe_pools_BCP]
            safe_tokens_BCPR = pool_tokens_BCPR.clamp(0, max(seq_len - 1, 0)).long()
            flat_safe_BC_PR = safe_tokens_BCPR.flatten(1)
            key_valid_BCPR = valid_BL.gather(1, flat_safe_BC_PR).view_as(
                safe_tokens_BCPR
            )
            query_positions_C = positions_L[start:end]
            token_mask_BCPR = (
                pool_tokens_BCPR.ge(0)
                & pool_tokens_BCPR.lt(seq_len)
                & key_valid_BCPR
                & (pool_tokens_BCPR <= query_positions_C[None, :, None, None])
            )
            gathered_k_BCPRHK = k_teacher_BLHK[batch_idx_B111, safe_tokens_BCPR]
            scores_BHCPR = torch.einsum(
                "bchk,bcprhk->bhcpr",
                q_teacher_BLHK[:, start:end],
                gathered_k_BCPRHK,
            )
            scores_BHCPR = (scores_BHCPR * scale).masked_fill(
                ~token_mask_BCPR[:, None],
                float("-inf"),
            )
            probs_BHCM = torch.nan_to_num(
                torch.softmax(
                    scores_BHCPR.flatten(-2),
                    dim=-1,
                    dtype=torch.float32,
                )
            ).detach()
            target_BCP = (
                probs_BHCM.view(
                    batch_size,
                    num_heads,
                    chunk_size,
                    safe_tokens_BCPR.shape[-2],
                    pool_width,
                )
                * token_mask_BCPR[:, None].to(probs_BHCM.dtype)
            ).sum(dim=(1, -1))
            target_BCP = target_BCP.masked_fill(~candidate_mask_BCP, 0.0)
            target_sum_BC1 = target_BCP.sum(dim=-1, keepdim=True)
            row_mask_BC = (
                state.valid_queries[:, start:end]
                & candidate_mask_BCP.any(dim=-1)
                & target_sum_BC1.squeeze(-1).gt(0.0)
            )
            target_BCP = target_BCP / target_sum_BC1.clamp_min(1.0e-12)
            log_probs_BCP = candidate_logits_BCP.float().masked_fill(
                ~candidate_mask_BCP,
                torch.finfo(torch.float32).min,
            )
            log_probs_BCP = torch.log_softmax(log_probs_BCP, dim=-1)
            loss_BC = torch.where(
                target_BCP > 0,
                target_BCP * (target_BCP.clamp_min(1.0e-12).log() - log_probs_BCP),
                torch.zeros_like(target_BCP),
            ).sum(dim=-1)
            if row_mask_BC.any():
                losses.append(loss_BC[row_mask_BC])

        if not losses:
            return state.scores.sum() * 0.0
        return torch.cat(losses).mean()

    safe_pools_BLC = candidate_pools_BLC.clamp(0, num_pools - 1)
    candidate_logits_BLC = state.scores.gather(-1, safe_pools_BLC)
    candidate_mask_BLC = state.valid_candidates.gather(-1, safe_pools_BLC)
    pool_tokens_BLCR = state.pool_indices[batch_idx_B11, safe_pools_BLC]
    safe_tokens_BLCR = pool_tokens_BLCR.clamp(0, max(seq_len - 1, 0)).long()
    flat_safe_BLC_R = safe_tokens_BLCR.flatten(1)
    key_valid_BLCR = valid_BL.gather(1, flat_safe_BLC_R).view_as(safe_tokens_BLCR)
    token_mask_BLCR = (
        pool_tokens_BLCR.ge(0)
        & pool_tokens_BLCR.lt(seq_len)
        & key_valid_BLCR
        & (pool_tokens_BLCR <= positions_L[None, :, None, None])
    )

    dense_scores_BHLL = torch.einsum(
        "blhk,bshk->bhls",
        q_teacher_BLHK,
        k_teacher_BLHK,
    )
    visible_BLL = positions_L[None, :, None] >= positions_L[None, None, :]
    visible_BLL = visible_BLL & valid_BL[:, None, :]
    dense_scores_BHLL = (dense_scores_BHLL * scale).masked_fill(
        ~visible_BLL[:, None],
        float("-inf"),
    )
    dense_probs_BHLL = torch.nan_to_num(
        torch.softmax(dense_scores_BHLL, dim=-1, dtype=torch.float32)
    ).detach()
    gather_index_BHLCR = safe_tokens_BLCR[:, None].expand(
        batch_size,
        q_BLHK.shape[2],
        seq_len,
        safe_tokens_BLCR.shape[-2],
        safe_tokens_BLCR.shape[-1],
    )
    token_probs_BHLCR = (
        dense_probs_BHLL[:, :, :, None]
        .expand(-1, -1, -1, safe_tokens_BLCR.shape[-2], -1)
        .gather(-1, gather_index_BHLCR)
    )
    target_BLC = (
        token_probs_BHLCR * token_mask_BLCR[:, None].to(token_probs_BHLCR.dtype)
    ).sum(dim=(1, -1))
    target_BLC = target_BLC.masked_fill(~candidate_mask_BLC, 0.0)
    target_sum_BL1 = target_BLC.sum(dim=-1, keepdim=True)
    row_mask_BL = (
        state.valid_queries
        & candidate_mask_BLC.any(dim=-1)
        & target_sum_BL1.squeeze(-1).gt(0.0)
    )
    if not row_mask_BL.any():
        return state.scores.sum() * 0.0
    target_BLC = target_BLC / target_sum_BL1.clamp_min(1.0e-12)
    log_probs_BLC = candidate_logits_BLC.float().masked_fill(
        ~candidate_mask_BLC,
        torch.finfo(torch.float32).min,
    )
    log_probs_BLC = torch.log_softmax(log_probs_BLC, dim=-1)
    loss_BL = torch.where(
        target_BLC > 0,
        target_BLC * (target_BLC.clamp_min(1.0e-12).log() - log_probs_BLC),
        torch.zeros_like(target_BLC),
    ).sum(dim=-1)
    return loss_BL[row_mask_BL].mean()


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
            query_chunk_size=config.sparse_attention_query_chunk_size,
        )
        self.train_step: int | None = None
        self.last_attention_path = "uninitialized"
        self.last_indexer_aux_loss: torch.Tensor | None = None

    def set_train_step(self, step: int | None) -> None:
        self.train_step = step

    def forward(
        self,
        x_BLD: torch.Tensor,
        *,
        attention_mask_BL: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        train_step: int | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = x_BLD.shape[:2]
        self.last_indexer_aux_loss = x_BLD.new_zeros(())
        selection_state: IndexerScoreState | None = None
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
        if self.config.indexer_aux_loss_weight > 0.0:
            selection_state = self.indexer.score_pools(x_BLD.detach(), attention_mask_BL)
            aux_loss = indexer_auxiliary_loss(
                self.indexer,
                x_BLD,
                q_BLHK,
                k_BLHK,
                attention_mask_BL=attention_mask_BL,
                target=self.config.indexer_aux_target,
                query_chunk_size=self.config.sparse_attention_query_chunk_size,
                state=selection_state,
            )
            self.last_indexer_aux_loss = self.config.indexer_aux_loss_weight * aux_loss
        active_step = self.train_step if train_step is None else train_step
        dense_warmup = (
            active_step is not None
            and active_step < self.config.indexer_dense_warmup_steps
        )
        if (
            not self.config.sparse_attention_enabled
            or self.config.sparse_top_k >= seq_len
            or dense_warmup
        ):
            out_BLHK = dense_causal_attention(
                q_BLHK,
                k_BLHK,
                v_BLHK,
                attention_mask_BL=attention_mask_BL,
            )
            self.last_attention_path = "dense_warmup" if dense_warmup else "dense"
        else:
            with torch.no_grad():
                if selection_state is None:
                    selection_state = self.indexer.score_pools(
                        x_BLD.detach(),
                        attention_mask_BL,
                    )
                selected_BLS = self.indexer.select_from_scores(selection_state)
            if self.config.sparse_attention_impl == "reference":
                out_BLHK = sparse_causal_attention_reference(
                    q_BLHK,
                    k_BLHK,
                    v_BLHK,
                    selected_BLS,
                    attention_mask_BL=attention_mask_BL,
                )
                self.last_attention_path = "reference"
            elif self.config.sparse_attention_impl == "gather":
                out_BLHK = sparse_causal_attention(
                    q_BLHK,
                    k_BLHK,
                    v_BLHK,
                    selected_BLS,
                    attention_mask_BL=attention_mask_BL,
                    query_chunk_size=self.config.sparse_attention_query_chunk_size,
                )
                self.last_attention_path = "gather"
            else:
                out_BLHK = sparse_causal_attention_masked_dense(
                    q_BLHK,
                    k_BLHK,
                    v_BLHK,
                    selected_BLS,
                    attention_mask_BL=attention_mask_BL,
                )
                self.last_attention_path = "masked_dense"
        return self.o_proj(out_BLHK.reshape(batch_size, seq_len, self.hidden_size))
