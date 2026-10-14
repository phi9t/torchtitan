# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton kernels for selected-key sparse causal attention.

Shape suffixes:
    B = batch, L = sequence length, H = attention heads, D = head dimension,
    S = selected index slots, N = effective selected keys after granularity.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _selected_key_offsets(
    selected_ptr,
    slots_N,
    batch_idx,
    query_idx,
    stride_selected_b: tl.constexpr,
    stride_selected_l: tl.constexpr,
    stride_selected_s: tl.constexpr,
    seq_len: tl.constexpr,
    num_selected: tl.constexpr,
    index_granularity: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor]:
    selected_slot_N = slots_N // index_granularity
    within_granule_N = slots_N - selected_slot_N * index_granularity
    selected_mask_N = selected_slot_N < num_selected
    selected_N = tl.load(
        selected_ptr
        + batch_idx * stride_selected_b
        + query_idx * stride_selected_l
        + selected_slot_N * stride_selected_s,
        mask=selected_mask_N,
        other=-2147483648,
    )
    key_N = selected_N * index_granularity + within_granule_N
    valid_N = selected_mask_N & (selected_N >= 0) & (key_N >= 0)
    valid_N = valid_N & (key_N < seq_len) & (key_N <= query_idx)
    return key_N, valid_N


@triton.jit
def sparse_topk_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    selected_ptr,
    mask_ptr,
    out_ptr,
    lse_ptr,
    stride_q_b: tl.constexpr,
    stride_q_l: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_k_b: tl.constexpr,
    stride_k_l: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_k_d: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_l: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_d: tl.constexpr,
    stride_selected_b: tl.constexpr,
    stride_selected_l: tl.constexpr,
    stride_selected_s: tl.constexpr,
    stride_mask_b: tl.constexpr,
    stride_mask_l: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_l: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_d: tl.constexpr,
    stride_lse_b: tl.constexpr,
    stride_lse_h: tl.constexpr,
    stride_lse_l: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_selected: tl.constexpr,
    scale: tl.constexpr,
    index_granularity: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row_id = tl.program_id(0)
    query_idx = row_id % seq_len
    head_idx = (row_id // seq_len) % num_heads
    batch_idx = row_id // (seq_len * num_heads)

    offs_D = tl.arange(0, BLOCK_D)
    dim_mask_D = offs_D < head_dim
    q_D = tl.load(
        q_ptr
        + batch_idx * stride_q_b
        + query_idx * stride_q_l
        + head_idx * stride_q_h
        + offs_D * stride_q_d,
        mask=dim_mask_D,
        other=0.0,
    ).to(tl.float32)
    query_valid = tl.load(
        mask_ptr + batch_idx * stride_mask_b + query_idx * stride_mask_l
    )

    m_i = tl.full((), -float("inf"), tl.float32)
    denom_i = tl.full((), 0.0, tl.float32)
    acc_D = tl.zeros((BLOCK_D,), tl.float32)

    num_effective = num_selected * index_granularity
    for block_start in range(0, num_effective, BLOCK_N):
        slots_N = block_start + tl.arange(0, BLOCK_N)
        key_N, valid_N = _selected_key_offsets(
            selected_ptr,
            slots_N,
            batch_idx,
            query_idx,
            stride_selected_b,
            stride_selected_l,
            stride_selected_s,
            seq_len,
            num_selected,
            index_granularity,
        )
        safe_key_N = tl.minimum(tl.maximum(key_N, 0), seq_len - 1)
        key_valid_N = tl.load(
            mask_ptr + batch_idx * stride_mask_b + safe_key_N * stride_mask_l,
            mask=valid_N,
            other=0,
        )
        valid_N = valid_N & query_valid & key_valid_N

        k_ND = tl.load(
            k_ptr
            + batch_idx * stride_k_b
            + safe_key_N[:, None] * stride_k_l
            + head_idx * stride_k_h
            + offs_D[None, :] * stride_k_d,
            mask=valid_N[:, None] & dim_mask_D[None, :],
            other=0.0,
        ).to(tl.float32)
        scores_N = tl.sum(k_ND * q_D[None, :], axis=1) * scale
        scores_N = tl.where(valid_N, scores_N, -float("inf"))

        block_m = tl.max(scores_N, axis=0)
        new_m = tl.maximum(m_i, block_m)
        alpha = tl.exp(m_i - new_m)
        probs_N = tl.exp(scores_N - new_m)
        probs_N = tl.where(valid_N, probs_N, 0.0)

        v_ND = tl.load(
            v_ptr
            + batch_idx * stride_v_b
            + safe_key_N[:, None] * stride_v_l
            + head_idx * stride_v_h
            + offs_D[None, :] * stride_v_d,
            mask=valid_N[:, None] & dim_mask_D[None, :],
            other=0.0,
        ).to(tl.float32)
        acc_D = acc_D * alpha + tl.sum(probs_N[:, None] * v_ND, axis=0)
        denom_i = denom_i * alpha + tl.sum(probs_N, axis=0)
        m_i = new_m

    has_keys = denom_i > 0.0
    out_D = tl.where(has_keys, acc_D / denom_i, 0.0)
    lse = tl.where(has_keys, m_i + tl.log(denom_i), -float("inf"))
    tl.store(
        out_ptr
        + batch_idx * stride_out_b
        + query_idx * stride_out_l
        + head_idx * stride_out_h
        + offs_D * stride_out_d,
        out_D,
        mask=dim_mask_D,
    )
    tl.store(
        lse_ptr
        + batch_idx * stride_lse_b
        + head_idx * stride_lse_h
        + query_idx * stride_lse_l,
        lse,
    )


@triton.jit
def sparse_topk_attention_bwd_kernel(
    grad_out_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    selected_ptr,
    mask_ptr,
    out_ptr,
    lse_ptr,
    grad_q_ptr,
    grad_k_ptr,
    grad_v_ptr,
    stride_go_b: tl.constexpr,
    stride_go_l: tl.constexpr,
    stride_go_h: tl.constexpr,
    stride_go_d: tl.constexpr,
    stride_q_b: tl.constexpr,
    stride_q_l: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_k_b: tl.constexpr,
    stride_k_l: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_k_d: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_l: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_d: tl.constexpr,
    stride_selected_b: tl.constexpr,
    stride_selected_l: tl.constexpr,
    stride_selected_s: tl.constexpr,
    stride_mask_b: tl.constexpr,
    stride_mask_l: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_l: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_d: tl.constexpr,
    stride_lse_b: tl.constexpr,
    stride_lse_h: tl.constexpr,
    stride_lse_l: tl.constexpr,
    stride_gq_b: tl.constexpr,
    stride_gq_l: tl.constexpr,
    stride_gq_h: tl.constexpr,
    stride_gq_d: tl.constexpr,
    stride_gk_b: tl.constexpr,
    stride_gk_l: tl.constexpr,
    stride_gk_h: tl.constexpr,
    stride_gk_d: tl.constexpr,
    stride_gv_b: tl.constexpr,
    stride_gv_l: tl.constexpr,
    stride_gv_h: tl.constexpr,
    stride_gv_d: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_selected: tl.constexpr,
    scale: tl.constexpr,
    index_granularity: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row_id = tl.program_id(0)
    query_idx = row_id % seq_len
    head_idx = (row_id // seq_len) % num_heads
    batch_idx = row_id // (seq_len * num_heads)

    offs_D = tl.arange(0, BLOCK_D)
    dim_mask_D = offs_D < head_dim
    q_D = tl.load(
        q_ptr
        + batch_idx * stride_q_b
        + query_idx * stride_q_l
        + head_idx * stride_q_h
        + offs_D * stride_q_d,
        mask=dim_mask_D,
        other=0.0,
    ).to(tl.float32)
    grad_out_D = tl.load(
        grad_out_ptr
        + batch_idx * stride_go_b
        + query_idx * stride_go_l
        + head_idx * stride_go_h
        + offs_D * stride_go_d,
        mask=dim_mask_D,
        other=0.0,
    ).to(tl.float32)
    out_D = tl.load(
        out_ptr
        + batch_idx * stride_out_b
        + query_idx * stride_out_l
        + head_idx * stride_out_h
        + offs_D * stride_out_d,
        mask=dim_mask_D,
        other=0.0,
    ).to(tl.float32)
    lse = tl.load(
        lse_ptr
        + batch_idx * stride_lse_b
        + head_idx * stride_lse_h
        + query_idx * stride_lse_l
    )
    query_valid = tl.load(
        mask_ptr + batch_idx * stride_mask_b + query_idx * stride_mask_l
    )
    active = query_valid & (lse > -float("inf"))
    delta = tl.sum(out_D * grad_out_D, axis=0)
    grad_q_D = tl.zeros((BLOCK_D,), tl.float32)

    num_effective = num_selected * index_granularity
    for block_start in range(0, num_effective, BLOCK_N):
        slots_N = block_start + tl.arange(0, BLOCK_N)
        key_N, valid_N = _selected_key_offsets(
            selected_ptr,
            slots_N,
            batch_idx,
            query_idx,
            stride_selected_b,
            stride_selected_l,
            stride_selected_s,
            seq_len,
            num_selected,
            index_granularity,
        )
        safe_key_N = tl.minimum(tl.maximum(key_N, 0), seq_len - 1)
        key_valid_N = tl.load(
            mask_ptr + batch_idx * stride_mask_b + safe_key_N * stride_mask_l,
            mask=valid_N,
            other=0,
        )
        valid_N = valid_N & active & key_valid_N

        k_ND = tl.load(
            k_ptr
            + batch_idx * stride_k_b
            + safe_key_N[:, None] * stride_k_l
            + head_idx * stride_k_h
            + offs_D[None, :] * stride_k_d,
            mask=valid_N[:, None] & dim_mask_D[None, :],
            other=0.0,
        ).to(tl.float32)
        v_ND = tl.load(
            v_ptr
            + batch_idx * stride_v_b
            + safe_key_N[:, None] * stride_v_l
            + head_idx * stride_v_h
            + offs_D[None, :] * stride_v_d,
            mask=valid_N[:, None] & dim_mask_D[None, :],
            other=0.0,
        ).to(tl.float32)
        scores_N = tl.sum(k_ND * q_D[None, :], axis=1) * scale
        probs_N = tl.exp(scores_N - lse)
        probs_N = tl.where(valid_N, probs_N, 0.0)
        grad_probs_N = tl.sum(v_ND * grad_out_D[None, :], axis=1)
        grad_logits_N = probs_N * (grad_probs_N - delta)
        grad_scaled_N = grad_logits_N * scale

        grad_q_D += tl.sum(grad_scaled_N[:, None] * k_ND, axis=0)
        grad_k_ND = grad_scaled_N[:, None] * q_D[None, :]
        grad_v_ND = probs_N[:, None] * grad_out_D[None, :]

        tl.atomic_add(
            grad_k_ptr
            + batch_idx * stride_gk_b
            + safe_key_N[:, None] * stride_gk_l
            + head_idx * stride_gk_h
            + offs_D[None, :] * stride_gk_d,
            grad_k_ND,
            sem="relaxed",
            mask=valid_N[:, None] & dim_mask_D[None, :],
        )
        tl.atomic_add(
            grad_v_ptr
            + batch_idx * stride_gv_b
            + safe_key_N[:, None] * stride_gv_l
            + head_idx * stride_gv_h
            + offs_D[None, :] * stride_gv_d,
            grad_v_ND,
            sem="relaxed",
            mask=valid_N[:, None] & dim_mask_D[None, :],
        )

    tl.store(
        grad_q_ptr
        + batch_idx * stride_gq_b
        + query_idx * stride_gq_l
        + head_idx * stride_gq_h
        + offs_D * stride_gq_d,
        grad_q_D,
        mask=dim_mask_D,
    )


def block_size_for_head_dim(head_dim: int) -> int:
    if head_dim < 1 or head_dim > 256:
        raise ValueError(f"head_dim must be in [1, 256], got {head_dim}.")
    return triton.next_power_of_2(head_dim)

