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

from torchtitan.distributed.local_compile import local_compile
from torchtitan.experiments.glm53_kernels.sparse_topk_attention import (
    sparse_topk_attention_blhd,
)

from .behavior import record, scalar, should_record
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


def _gather_batch_dim1(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``values[:, indices, ...]`` without advanced-index materialization."""

    if values.shape[0] != indices.shape[0]:
        raise ValueError("values and indices must have the same batch size.")
    gather_dim = indices.ndim
    value_view = values.view(
        values.shape[0],
        *((1,) * (indices.ndim - 1)),
        values.shape[1],
        *values.shape[2:],
    )
    value_expand = value_view.expand(
        values.shape[0],
        *indices.shape[1:],
        values.shape[1],
        *values.shape[2:],
    )
    index_view = indices.view(
        *indices.shape,
        *((1,) * (values.ndim - 1)),
    )
    index_expand = index_view.expand(
        *indices.shape,
        1,
        *values.shape[2:],
    )
    return value_expand.gather(gather_dim, index_expand).squeeze(gather_dim)


def _sample_query_indices(
    seq_len: int,
    *,
    device: torch.device,
    query_stride: int,
    query_seed: int,
    train_step: int | None,
) -> tuple[torch.Tensor, float]:
    if query_stride < 1:
        raise ValueError("query_stride must be >= 1.")
    if query_stride == 1:
        return torch.arange(seq_len, device=device), 1.0
    step = 0 if train_step is None else train_step
    offset = (query_seed + step) % query_stride
    return torch.arange(offset, seq_len, query_stride, device=device), float(
        query_stride
    )


def _indexer_aux_full_row_count(state: IndexerScoreState) -> torch.Tensor:
    row_mask_BL = state.valid_queries & state.valid_candidates.any(dim=-1)
    return row_mask_BL.to(torch.float32).sum()


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


def sparse_causal_attention_triton(
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    v_BLHK: torch.Tensor,
    selected_BLS: torch.Tensor,
    *,
    attention_mask_BL: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Sparse causal attention through the experiment-owned Triton custom op."""

    batch_size, seq_len = q_BLHK.shape[:2]
    if selected_BLS.shape[:2] != (batch_size, seq_len):
        raise ValueError(
            "selected_BLS must have shape [B, L, S], "
            f"got {tuple(selected_BLS.shape)} for q {tuple(q_BLHK.shape)}."
        )
    if scale is None:
        scale = q_BLHK.shape[-1] ** -0.5
    valid_BL = _default_mask(q_BLHK.flatten(-2), attention_mask_BL)
    return sparse_topk_attention_blhd(
        q_BLHK,
        k_BLHK,
        v_BLHK,
        selected_BLS,
        attention_mask_BL=valid_BL,
        scale=scale,
        index_granularity=1,
    )


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
            gathered_k_BCSHK = _gather_batch_dim1(k_BLHK, safe_BCS)
            gathered_v_BCSHK = _gather_batch_dim1(v_BLHK, safe_BCS)
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
            torch.full(
                (batch_size,), seq_len, dtype=torch.long, device=keys_BLK.device
            ),
        )
        raw_indices_BPR = first_key_B[:, None, None] + offsets_PR[None]
        safe_indices_BPR = raw_indices_BPR.clamp(0, max(seq_len - 1, 0))
        flat_safe_BP_R = safe_indices_BPR.flatten(1)
        flat_valid_BP_R = valid_BL.gather(1, flat_safe_BP_R)
        valid_BPR = raw_indices_BPR.lt(seq_len) & flat_valid_BP_R.view(
            batch_size, num_pools, self.pool_size
        )
        gather_index_BPRK = safe_indices_BPR[..., None].expand(
            batch_size,
            num_pools,
            self.pool_size,
            self.head_dim,
        )
        grouped_keys_BPRK = (
            keys_BLK[:, None]
            .expand(
                batch_size,
                num_pools,
                seq_len,
                self.head_dim,
            )
            .gather(2, gather_index_BPRK)
        )
        grouped_gate_BPRK = (
            gate_BLK[:, None]
            .expand(
                batch_size,
                num_pools,
                seq_len,
                self.head_dim,
            )
            .gather(2, gather_index_BPRK)
        )
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


@local_compile("glm53_speedrun_indexer_score", batch_invariant=True)
def _indexer_score_pools_region(
    hidden_BLD: torch.Tensor,
    valid_BL: torch.Tensor,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    k_norm: nn.LayerNorm,
    weights_proj: nn.Linear,
    gate_proj: nn.Linear,
    index_pool: IndexPool,
    *,
    index_num_heads: int,
    index_head_dim: int,
    pool_size: int,
    query_chunk_size: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len = hidden_BLD.shape[:2]
    q_BLHK = q_proj(hidden_BLD).view(
        batch_size,
        seq_len,
        index_num_heads,
        index_head_dim,
    )
    keys_BLK = k_norm(k_proj(hidden_BLD))
    gate_BLK = gate_proj(hidden_BLD)
    pool_keys_BPK, pool_indices_BPR, pool_valid_BP = index_pool(
        keys_BLK,
        gate_BLK,
        valid_BL,
    )
    num_pools = pool_keys_BPK.shape[1]
    if num_pools == 0:
        return (
            hidden_BLD.new_empty(batch_size, seq_len, 0),
            torch.empty(
                batch_size,
                0,
                pool_size,
                dtype=torch.long,
                device=hidden_BLD.device,
            ),
            torch.empty(
                batch_size,
                seq_len,
                0,
                dtype=torch.bool,
                device=hidden_BLD.device,
            ),
            valid_BL,
        )

    head_weights_BLH = weights_proj(hidden_BLD).float()
    head_weights_BLH = head_weights_BLH * (index_num_heads**-0.5)
    scores_chunks = []
    for start in range(0, seq_len, query_chunk_size):
        end = min(start + query_chunk_size, seq_len)
        scores_BCHP = torch.einsum(
            "bchk,bpk->bchp",
            q_BLHK[:, start:end].float(),
            pool_keys_BPK.float(),
        )
        scores_BCHP = F.relu(scores_BCHP * softmax_scale)
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
    return scores_BLP, pool_indices_BPR, valid_candidates_BLP, valid_BL


def _append_tail_tokens(
    selected_tokens_BLS: torch.Tensor,
    valid_BL: torch.Tensor,
    *,
    pool_size: int,
) -> torch.Tensor:
    batch_size, seq_len = valid_BL.shape
    max_tail_width = pool_size - 1
    tail_offsets_S = torch.arange(max_tail_width, device=valid_BL.device)
    first_key_B = torch.where(
        valid_BL.any(-1),
        valid_BL.long().argmax(-1),
        torch.full((batch_size,), seq_len, dtype=torch.long, device=valid_BL.device),
    )
    visible_count_BL = valid_BL.long().cumsum(dim=-1)
    tail_count_BL = visible_count_BL.remainder(pool_size)
    tail_start_BL = first_key_B[:, None] + visible_count_BL - tail_count_BL
    tail_indices_BLS = tail_start_BL[..., None] + tail_offsets_S
    query_positions_L = torch.arange(seq_len, device=valid_BL.device)
    tail_valid_BLS = (tail_offsets_S[None, None, :] < tail_count_BL[..., None]) & (
        tail_indices_BLS <= query_positions_L[None, :, None]
    )
    tail_indices_BLS = tail_indices_BLS.masked_fill(~tail_valid_BLS, -1)
    return torch.cat([selected_tokens_BLS, tail_indices_BLS], dim=-1)


@local_compile("glm53_speedrun_indexer_select", batch_invariant=True)
def _indexer_select_from_scores_region(
    scores_BLP: torch.Tensor,
    pool_indices_BPR: torch.Tensor,
    valid_candidates_BLP: torch.Tensor,
    valid_BL: torch.Tensor,
    *,
    top_k: int,
    pool_size: int,
    always_select_tail: bool,
) -> torch.Tensor:
    num_pools = pool_indices_BPR.shape[1]
    num_pool_selections = min(math.ceil(top_k / pool_size), num_pools)
    selected_pools_BLS = scores_BLP.topk(num_pool_selections, dim=-1).indices
    selected_valid_BLS = valid_candidates_BLP.gather(-1, selected_pools_BLS)
    selected_indices_BLSR = _gather_batch_dim1(pool_indices_BPR, selected_pools_BLS)
    selected_tokens_BLS = selected_indices_BLSR.flatten(-2)
    selected_tokens_BLS = selected_tokens_BLS.masked_fill(
        ~selected_valid_BLS[..., None].expand_as(selected_indices_BLSR).flatten(-2),
        -1,
    )
    selected_tokens_BLS = selected_tokens_BLS[..., :top_k]
    if always_select_tail and pool_size > 1:
        selected_tokens_BLS = _append_tail_tokens(
            selected_tokens_BLS,
            valid_BL,
            pool_size=pool_size,
        )
    output_width = top_k
    if always_select_tail and pool_size > 1:
        output_width += pool_size - 1
    if selected_tokens_BLS.shape[-1] < output_width:
        selected_tokens_BLS = F.pad(
            selected_tokens_BLS,
            (0, output_width - selected_tokens_BLS.shape[-1]),
            value=-1,
        )
    selected_tokens_BLS = selected_tokens_BLS[..., :output_width]
    return selected_tokens_BLS.masked_fill(~valid_BL[..., None], -1).to(torch.int32)


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
        valid_BL = _default_mask(hidden_BLD, attention_mask_BL)
        (
            scores_BLP,
            pool_indices_BPR,
            valid_candidates_BLP,
            valid_BL,
        ) = _indexer_score_pools_region(
            hidden_BLD,
            valid_BL,
            self.q_proj,
            self.k_proj,
            self.k_norm,
            self.weights_proj,
            self.gate_proj,
            self.index_pool,
            index_num_heads=self.index_num_heads,
            index_head_dim=self.index_head_dim,
            pool_size=self.pool_size,
            query_chunk_size=self.query_chunk_size,
            softmax_scale=self.softmax_scale,
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
        return _indexer_select_from_scores_region(
            state.scores,
            state.pool_indices,
            state.valid_candidates,
            state.valid_queries,
            top_k=self.top_k,
            pool_size=self.pool_size,
            always_select_tail=self.always_select_tail,
        )

    def _append_tail(
        self,
        selected_tokens_BLS: torch.Tensor,
        valid_BL: torch.Tensor,
    ) -> torch.Tensor:
        return _append_tail_tokens(
            selected_tokens_BLS,
            valid_BL,
            pool_size=self.pool_size,
        )


@local_compile("glm53_speedrun_indexer_aux", batch_invariant=True)
def _dense_teacher_indexer_auxiliary_loss(
    scores_BLP: torch.Tensor,
    pool_indices_BPR: torch.Tensor,
    valid_candidates_BLP: torch.Tensor,
    valid_queries_BL: torch.Tensor,
    q_teacher_BLHK: torch.Tensor,
    k_teacher_BLHK: torch.Tensor,
    query_indices_Q: torch.Tensor,
    full_row_count: torch.Tensor,
    *,
    top_k: int,
    pool_size: int,
    target_all: bool,
    scale: float,
    loss_scale: float,
) -> torch.Tensor:
    batch_size, seq_len, num_pools = scores_BLP.shape
    num_heads = q_teacher_BLHK.shape[2]
    pool_width = pool_indices_BPR.shape[-1]
    sampled_scores_BQP = scores_BLP.index_select(1, query_indices_Q)
    sampled_valid_candidates_BQP = valid_candidates_BLP.index_select(
        1,
        query_indices_Q,
    )
    if target_all:
        candidate_pools_BQC = torch.arange(
            num_pools,
            device=scores_BLP.device,
        ).view(1, 1, num_pools)
        candidate_pools_BQC = candidate_pools_BQC.expand(
            batch_size,
            query_indices_Q.shape[0],
            num_pools,
        )
    else:
        num_pool_selections = min(math.ceil(top_k / pool_size), num_pools)
        candidate_pools_BQC = sampled_scores_BQP.topk(
            num_pool_selections,
            dim=-1,
        ).indices

    safe_pools_BQC = candidate_pools_BQC.clamp(0, num_pools - 1)
    candidate_logits_BQC = sampled_scores_BQP.gather(-1, safe_pools_BQC)
    candidate_mask_BQC = sampled_valid_candidates_BQP.gather(-1, safe_pools_BQC)
    pool_tokens_BQCR = _gather_batch_dim1(pool_indices_BPR, safe_pools_BQC)
    safe_tokens_BQCR = pool_tokens_BQCR.clamp(0, max(seq_len - 1, 0)).long()
    flat_safe_BQ_CR = safe_tokens_BQCR.flatten(1)
    key_valid_BQCR = valid_queries_BL.gather(1, flat_safe_BQ_CR).view_as(
        safe_tokens_BQCR
    )
    token_mask_BQCR = (
        pool_tokens_BQCR.ge(0)
        & pool_tokens_BQCR.lt(seq_len)
        & key_valid_BQCR
        & (pool_tokens_BQCR <= query_indices_Q[None, :, None, None])
    )

    q_teacher_BQHK = q_teacher_BLHK.index_select(1, query_indices_Q)
    dense_scores_BHQL = torch.einsum(
        "bqhk,blhk->bhql",
        q_teacher_BQHK,
        k_teacher_BLHK,
    )
    dense_scores_BHQL = dense_scores_BHQL * scale
    key_positions_L = torch.arange(seq_len, device=scores_BLP.device)
    visible_BQL = key_positions_L[None, None, :] <= query_indices_Q[None, :, None]
    visible_BQL = visible_BQL & valid_queries_BL[:, None, :]
    dense_scores_BHQL = dense_scores_BHQL.masked_fill(
        ~visible_BQL[:, None],
        float("-inf"),
    )
    gather_index_BHQCR = safe_tokens_BQCR[:, None].expand(
        batch_size,
        num_heads,
        query_indices_Q.shape[0],
        safe_tokens_BQCR.shape[-2],
        pool_width,
    )
    token_scores_BHQCR = (
        dense_scores_BHQL[:, :, :, None]
        .expand(-1, -1, -1, safe_tokens_BQCR.shape[-2], -1)
        .gather(-1, gather_index_BHQCR)
    )
    if target_all:
        dense_lse_BHQ1 = torch.logsumexp(dense_scores_BHQL, dim=-1, keepdim=True)
    else:
        selected_scores_BHQCR = token_scores_BHQCR.masked_fill(
            ~token_mask_BQCR[:, None],
            float("-inf"),
        )
        dense_lse_BHQ1 = torch.logsumexp(
            selected_scores_BHQCR.flatten(-2),
            dim=-1,
            keepdim=True,
        )
    dense_lse_BHQ1 = torch.where(
        torch.isfinite(dense_lse_BHQ1),
        dense_lse_BHQ1,
        torch.zeros_like(dense_lse_BHQ1),
    )
    token_probs_BHQCR = torch.exp(token_scores_BHQCR - dense_lse_BHQ1[..., None])
    token_probs_BHQCR = torch.nan_to_num(token_probs_BHQCR)
    target_BQC = (
        token_probs_BHQCR * token_mask_BQCR[:, None].to(token_probs_BHQCR.dtype)
    ).sum(dim=(1, -1))
    target_BQC = target_BQC.masked_fill(~candidate_mask_BQC, 0.0)
    target_sum_BQ1 = target_BQC.sum(dim=-1, keepdim=True)
    row_mask_BQ = (
        valid_queries_BL.index_select(1, query_indices_Q)
        & candidate_mask_BQC.any(dim=-1)
        & target_sum_BQ1.squeeze(-1).gt(0.0)
    )
    target_BQC = target_BQC / target_sum_BQ1.clamp_min(1.0e-12)
    log_probs_BQC = candidate_logits_BQC.float().masked_fill(
        ~candidate_mask_BQC,
        torch.finfo(torch.float32).min,
    )
    log_probs_BQC = torch.log_softmax(log_probs_BQC, dim=-1)
    loss_BQ = torch.where(
        target_BQC > 0,
        target_BQC * (target_BQC.clamp_min(1.0e-12).log() - log_probs_BQC),
        torch.zeros_like(target_BQC),
    ).sum(dim=-1)
    row_mask_float_BQ = row_mask_BQ.to(loss_BQ.dtype)
    loss_sum = (loss_BQ * row_mask_float_BQ).sum()
    zero_loss = scores_BLP.sum() * 0.0
    return torch.where(
        full_row_count > 0,
        loss_sum * loss_scale / full_row_count.clamp_min(1.0),
        zero_loss,
    )


def indexer_auxiliary_loss(
    indexer: LightningIndexer,
    hidden_BLD: torch.Tensor,
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    *,
    attention_mask_BL: torch.Tensor | None = None,
    target: str = "selected",
    query_chunk_size: int = 64,
    query_stride: int = 1,
    query_seed: int = 0,
    train_step: int | None = None,
    use_dense_teacher: bool = False,
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
    if query_stride < 1:
        raise ValueError("query_stride must be >= 1.")
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

    if query_stride > 1 or use_dense_teacher:
        query_indices_Q, loss_scale = _sample_query_indices(
            seq_len,
            device=hidden_BLD.device,
            query_stride=query_stride,
            query_seed=query_seed,
            train_step=train_step,
        )
        return _dense_teacher_indexer_auxiliary_loss(
            state.scores,
            state.pool_indices,
            state.valid_candidates,
            state.valid_queries,
            q_teacher_BLHK,
            k_teacher_BLHK,
            query_indices_Q,
            _indexer_aux_full_row_count(state),
            top_k=indexer.top_k,
            pool_size=indexer.pool_size,
            target_all=target == "all",
            scale=scale,
            loss_scale=loss_scale,
        )

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
    if target == "selected":
        loss_sum = state.scores.new_zeros((), dtype=torch.float32)
        row_count = state.scores.new_zeros((), dtype=torch.float32)
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
            pool_tokens_BCPR = _gather_batch_dim1(
                state.pool_indices,
                safe_pools_BCP,
            )
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
            gathered_k_BCPRHK = _gather_batch_dim1(k_teacher_BLHK, safe_tokens_BCPR)
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
            row_mask_float_BC = row_mask_BC.to(loss_BC.dtype)
            loss_sum = loss_sum + (loss_BC * row_mask_float_BC).sum()
            row_count = row_count + row_mask_float_BC.sum()

        zero_loss = state.scores.sum() * 0.0
        return torch.where(
            row_count > 0,
            loss_sum / row_count.clamp_min(1.0),
            zero_loss,
        )

    safe_pools_BLC = candidate_pools_BLC.clamp(0, num_pools - 1)
    candidate_logits_BLC = state.scores.gather(-1, safe_pools_BLC)
    candidate_mask_BLC = state.valid_candidates.gather(-1, safe_pools_BLC)
    pool_tokens_BLCR = _gather_batch_dim1(state.pool_indices, safe_pools_BLC)
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
    row_mask_float_BL = row_mask_BL.to(loss_BL.dtype)
    loss_sum = (loss_BL * row_mask_float_BL).sum()
    row_count = row_mask_float_BL.sum()
    zero_loss = state.scores.sum() * 0.0
    return torch.where(
        row_count > 0,
        loss_sum / row_count.clamp_min(1.0),
        zero_loss,
    )


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
        self.behavior_tag = "sparse_attention"
        self._behavior_calls = 0

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
        active_step = self.train_step if train_step is None else train_step
        if self.config.indexer_aux_loss_weight > 0.0:
            selection_state = self.indexer.score_pools(
                x_BLD.detach(), attention_mask_BL
            )
            aux_loss = indexer_auxiliary_loss(
                self.indexer,
                x_BLD,
                q_BLHK,
                k_BLHK,
                attention_mask_BL=attention_mask_BL,
                target=self.config.indexer_aux_target,
                query_chunk_size=self.config.sparse_attention_query_chunk_size,
                query_stride=self.config.indexer_aux_query_stride,
                query_seed=self.config.indexer_aux_query_seed,
                train_step=active_step,
                use_dense_teacher=(
                    self.config.indexer_aux_compile
                    or self.config.indexer_aux_query_stride > 1
                ),
                state=selection_state,
            )
            self.last_indexer_aux_loss = self.config.indexer_aux_loss_weight * aux_loss
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
            elif self.config.sparse_attention_impl == "triton":
                out_BLHK = sparse_causal_attention_triton(
                    q_BLHK,
                    k_BLHK,
                    v_BLHK,
                    selected_BLS,
                    attention_mask_BL=attention_mask_BL,
                )
                self.last_attention_path = "triton"
            else:
                out_BLHK = sparse_causal_attention_masked_dense(
                    q_BLHK,
                    k_BLHK,
                    v_BLHK,
                    selected_BLS,
                    attention_mask_BL=attention_mask_BL,
                )
                self.last_attention_path = "masked_dense"
            self._behavior_calls += 1
            if should_record("indexer", self.behavior_tag, self._behavior_calls):
                with torch.no_grad():
                    selected_1LS = selected_BLS[:1].to(q_BLHK.device)
                    q_LK = q_BLHK[0, :, 0].detach().float()
                    k_LK = k_BLHK[0, :, 0].detach().float()
                    scores_LL = q_LK @ k_LK.transpose(0, 1)
                    positions_L = torch.arange(seq_len, device=q_BLHK.device)
                    scores_LL = scores_LL.masked_fill(
                        positions_L[:, None] < positions_L[None, :],
                        float("-inf"),
                    )
                    topk_width = min(self.config.sparse_top_k, seq_len)
                    dense_topk_LK = scores_LL.topk(topk_width, dim=-1).indices
                    dense_mask_LL = torch.zeros(
                        seq_len,
                        seq_len,
                        dtype=torch.bool,
                        device=q_BLHK.device,
                    )
                    dense_mask_LL.scatter_(-1, dense_topk_LK, True)
                    selected_mask_LL = torch.zeros_like(dense_mask_LL)
                    valid_LS = selected_1LS[0].ge(0) & selected_1LS[0].lt(seq_len)
                    safe_LS = selected_1LS[0].clamp(0, max(seq_len - 1, 0)).long()
                    selected_mask_LL.scatter_(-1, safe_LS, valid_LS)
                    visible_counts_L = torch.minimum(
                        positions_L + 1,
                        torch.full_like(positions_L, topk_width),
                    )
                    recall_L = (dense_mask_LL & selected_mask_LL).sum(
                        dim=-1
                    ).float() / visible_counts_L.clamp_min(1).float()
                    aux_loss = self.last_indexer_aux_loss
                    record(
                        {
                            "kind": "indexer",
                            "tag": self.behavior_tag,
                            "call_count": self._behavior_calls,
                            "attention_path": self.last_attention_path,
                            "top_k": self.config.sparse_top_k,
                            "index_pool_size": self.config.index_pool_size,
                            "sample_batch": 0,
                            "sample_head": 0,
                            "recall_vs_dense_topk_mean": scalar(recall_L.mean()),
                            "recall_vs_dense_topk_min": scalar(recall_L.min()),
                            "selection_width": int(selected_BLS.shape[-1]),
                            "selected_valid_fraction": scalar(valid_LS.float().mean()),
                            "indexer_aux_loss": (
                                None if aux_loss is None else scalar(aux_loss)
                            ),
                        }
                    )
        return self.o_proj(out_BLHK.reshape(batch_size, seq_len, self.hidden_size))
