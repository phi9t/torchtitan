# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch custom op wrapper for Triton selected-key sparse attention.

The custom op consumes BLHD tensors internally to match the speedrun element.
``sparse_topk_attention_bhld`` is provided for the brief's public B,H,L,D shape.
"""

from __future__ import annotations

import math

import torch

from .kernels import (
    block_size_for_head_dim,
    sparse_topk_attention_bwd_kernel,
    sparse_topk_attention_fwd_kernel,
)


_BLOCK_N = 64
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _validate_common(
    q_BLHD: torch.Tensor,
    k_BLHD: torch.Tensor,
    v_BLHD: torch.Tensor,
    selected_BLS: torch.Tensor,
    attention_mask_BL: torch.Tensor,
    index_granularity: int,
) -> tuple[int, int, int, int, int]:
    if not q_BLHD.is_cuda:
        raise ValueError("sparse_topk_attention requires CUDA tensors.")
    if q_BLHD.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "q, k, and v must use bf16, fp16, or fp32; "
            f"got {q_BLHD.dtype}."
        )
    if q_BLHD.shape != k_BLHD.shape or q_BLHD.shape != v_BLHD.shape:
        raise ValueError(
            "q, k, and v must have identical BLHD shapes, got "
            f"{tuple(q_BLHD.shape)}, {tuple(k_BLHD.shape)}, {tuple(v_BLHD.shape)}."
        )
    if q_BLHD.dtype != k_BLHD.dtype or q_BLHD.dtype != v_BLHD.dtype:
        raise TypeError("q, k, and v must have the same dtype.")
    if q_BLHD.device != k_BLHD.device or q_BLHD.device != v_BLHD.device:
        raise ValueError("q, k, and v must be on the same device.")
    if q_BLHD.ndim != 4:
        raise ValueError(f"q must have shape [B, L, H, D], got {q_BLHD.ndim}D.")
    batch_size, seq_len, num_heads, head_dim = q_BLHD.shape
    if min(batch_size, seq_len, num_heads, head_dim) <= 0:
        raise ValueError("q, k, and v dimensions must be positive.")
    if selected_BLS.shape[:2] != (batch_size, seq_len) or selected_BLS.ndim != 3:
        raise ValueError(
            "selected_BLS must have shape [B, L, S], got "
            f"{tuple(selected_BLS.shape)} for q {tuple(q_BLHD.shape)}."
        )
    if selected_BLS.device != q_BLHD.device:
        raise ValueError("selected_BLS must be on the same device as q.")
    if selected_BLS.dtype not in (torch.int32, torch.int64):
        raise TypeError("selected_BLS must use int32 or int64.")
    if attention_mask_BL.shape != (batch_size, seq_len):
        raise ValueError(
            "attention_mask_BL must have shape [B, L], got "
            f"{tuple(attention_mask_BL.shape)}."
        )
    if attention_mask_BL.device != q_BLHD.device:
        raise ValueError("attention_mask_BL must be on the same device as q.")
    if attention_mask_BL.dtype != torch.bool:
        raise TypeError("attention_mask_BL must be bool.")
    if index_granularity not in (1, 4):
        raise ValueError("index_granularity must be 1 or 4.")
    return batch_size, seq_len, num_heads, head_dim, selected_BLS.shape[-1]


@torch.library.custom_op(
    "glm53_sparse_topk::attention_fwd",
    mutates_args=(),
    device_types="cuda",
)
def _attention_fwd(
    q_BLHD: torch.Tensor,
    k_BLHD: torch.Tensor,
    v_BLHD: torch.Tensor,
    selected_BLS: torch.Tensor,
    attention_mask_BL: torch.Tensor,
    scale: float,
    index_granularity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, num_heads, head_dim, num_selected = _validate_common(
        q_BLHD,
        k_BLHD,
        v_BLHD,
        selected_BLS,
        attention_mask_BL,
        index_granularity,
    )
    selected_BLS = selected_BLS.contiguous()
    attention_mask_BL = attention_mask_BL.contiguous()
    out_BLHD = torch.empty_like(q_BLHD)
    lse_BHL = torch.empty(
        (batch_size, num_heads, seq_len),
        dtype=torch.float32,
        device=q_BLHD.device,
    )
    block_d = block_size_for_head_dim(head_dim)
    grid = (batch_size * num_heads * seq_len,)
    sparse_topk_attention_fwd_kernel[grid](
        q_BLHD,
        k_BLHD,
        v_BLHD,
        selected_BLS,
        attention_mask_BL,
        out_BLHD,
        lse_BHL,
        q_BLHD.stride(0),
        q_BLHD.stride(1),
        q_BLHD.stride(2),
        q_BLHD.stride(3),
        k_BLHD.stride(0),
        k_BLHD.stride(1),
        k_BLHD.stride(2),
        k_BLHD.stride(3),
        v_BLHD.stride(0),
        v_BLHD.stride(1),
        v_BLHD.stride(2),
        v_BLHD.stride(3),
        selected_BLS.stride(0),
        selected_BLS.stride(1),
        selected_BLS.stride(2),
        attention_mask_BL.stride(0),
        attention_mask_BL.stride(1),
        out_BLHD.stride(0),
        out_BLHD.stride(1),
        out_BLHD.stride(2),
        out_BLHD.stride(3),
        lse_BHL.stride(0),
        lse_BHL.stride(1),
        lse_BHL.stride(2),
        seq_len,
        num_heads,
        head_dim,
        num_selected,
        scale,
        index_granularity,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out_BLHD, lse_BHL


@_attention_fwd.register_fake
def _attention_fwd_fake(
    q_BLHD: torch.Tensor,
    k_BLHD: torch.Tensor,
    v_BLHD: torch.Tensor,
    selected_BLS: torch.Tensor,
    attention_mask_BL: torch.Tensor,
    scale: float,
    index_granularity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del k_BLHD, v_BLHD, selected_BLS, attention_mask_BL, scale, index_granularity
    batch_size, seq_len, num_heads, _ = q_BLHD.shape
    return (
        torch.empty_like(q_BLHD),
        q_BLHD.new_empty((batch_size, num_heads, seq_len), dtype=torch.float32),
    )


@torch.library.custom_op(
    "glm53_sparse_topk::attention_bwd",
    mutates_args=(),
    device_types="cuda",
)
def _attention_bwd(
    grad_out_BLHD: torch.Tensor,
    q_BLHD: torch.Tensor,
    k_BLHD: torch.Tensor,
    v_BLHD: torch.Tensor,
    selected_BLS: torch.Tensor,
    attention_mask_BL: torch.Tensor,
    out_BLHD: torch.Tensor,
    lse_BHL: torch.Tensor,
    scale: float,
    index_granularity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len, num_heads, head_dim, num_selected = _validate_common(
        q_BLHD,
        k_BLHD,
        v_BLHD,
        selected_BLS,
        attention_mask_BL,
        index_granularity,
    )
    if grad_out_BLHD.shape != q_BLHD.shape:
        raise ValueError(
            "grad_out_BLHD must have the same shape as q, got "
            f"{tuple(grad_out_BLHD.shape)} and {tuple(q_BLHD.shape)}."
        )
    selected_BLS = selected_BLS.contiguous()
    attention_mask_BL = attention_mask_BL.contiguous()
    grad_q = torch.empty_like(q_BLHD, dtype=torch.float32)
    grad_k = torch.zeros_like(k_BLHD, dtype=torch.float32)
    grad_v = torch.zeros_like(v_BLHD, dtype=torch.float32)
    block_d = block_size_for_head_dim(head_dim)
    grid = (batch_size * num_heads * seq_len,)
    sparse_topk_attention_bwd_kernel[grid](
        grad_out_BLHD,
        q_BLHD,
        k_BLHD,
        v_BLHD,
        selected_BLS,
        attention_mask_BL,
        out_BLHD,
        lse_BHL,
        grad_q,
        grad_k,
        grad_v,
        grad_out_BLHD.stride(0),
        grad_out_BLHD.stride(1),
        grad_out_BLHD.stride(2),
        grad_out_BLHD.stride(3),
        q_BLHD.stride(0),
        q_BLHD.stride(1),
        q_BLHD.stride(2),
        q_BLHD.stride(3),
        k_BLHD.stride(0),
        k_BLHD.stride(1),
        k_BLHD.stride(2),
        k_BLHD.stride(3),
        v_BLHD.stride(0),
        v_BLHD.stride(1),
        v_BLHD.stride(2),
        v_BLHD.stride(3),
        selected_BLS.stride(0),
        selected_BLS.stride(1),
        selected_BLS.stride(2),
        attention_mask_BL.stride(0),
        attention_mask_BL.stride(1),
        out_BLHD.stride(0),
        out_BLHD.stride(1),
        out_BLHD.stride(2),
        out_BLHD.stride(3),
        lse_BHL.stride(0),
        lse_BHL.stride(1),
        lse_BHL.stride(2),
        grad_q.stride(0),
        grad_q.stride(1),
        grad_q.stride(2),
        grad_q.stride(3),
        grad_k.stride(0),
        grad_k.stride(1),
        grad_k.stride(2),
        grad_k.stride(3),
        grad_v.stride(0),
        grad_v.stride(1),
        grad_v.stride(2),
        grad_v.stride(3),
        seq_len,
        num_heads,
        head_dim,
        num_selected,
        scale,
        index_granularity,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return (
        grad_q.to(q_BLHD.dtype),
        grad_k.to(k_BLHD.dtype),
        grad_v.to(v_BLHD.dtype),
    )


@_attention_bwd.register_fake
def _attention_bwd_fake(
    grad_out_BLHD: torch.Tensor,
    q_BLHD: torch.Tensor,
    k_BLHD: torch.Tensor,
    v_BLHD: torch.Tensor,
    selected_BLS: torch.Tensor,
    attention_mask_BL: torch.Tensor,
    out_BLHD: torch.Tensor,
    lse_BHL: torch.Tensor,
    scale: float,
    index_granularity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        grad_out_BLHD,
        selected_BLS,
        attention_mask_BL,
        out_BLHD,
        lse_BHL,
        scale,
        index_granularity,
    )
    return torch.empty_like(q_BLHD), torch.empty_like(k_BLHD), torch.empty_like(v_BLHD)


def _attention_setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
        int,
    ],
    output: tuple[torch.Tensor, torch.Tensor],
) -> None:
    q_BLHD, k_BLHD, v_BLHD, selected_BLS, attention_mask_BL, scale, index_granularity = (
        inputs
    )
    out_BLHD, lse_BHL = output
    ctx.save_for_backward(  # pyrefly: ignore[missing-attribute]
        q_BLHD,
        k_BLHD,
        v_BLHD,
        selected_BLS,
        attention_mask_BL,
        out_BLHD,
        lse_BHL,
    )
    ctx.scale = scale
    ctx.index_granularity = index_granularity
    ctx.mark_non_differentiable(lse_BHL)


def _attention_backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out_BLHD: torch.Tensor,
    grad_lse_BHL: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
    del grad_lse_BHL
    q_BLHD, k_BLHD, v_BLHD, selected_BLS, attention_mask_BL, out_BLHD, lse_BHL = (
        ctx.saved_tensors  # pyrefly: ignore[missing-attribute]
    )
    grad_q, grad_k, grad_v = _attention_bwd(
        grad_out_BLHD,
        q_BLHD,
        k_BLHD,
        v_BLHD,
        selected_BLS,
        attention_mask_BL,
        out_BLHD,
        lse_BHL,
        ctx.scale,  # pyrefly: ignore[missing-attribute]
        ctx.index_granularity,  # pyrefly: ignore[missing-attribute]
    )
    return grad_q, grad_k, grad_v, None, None, None, None


_attention_fwd.register_autograd(
    _attention_backward,
    setup_context=_attention_setup_context,
)


def _default_mask(q_BLHD: torch.Tensor) -> torch.Tensor:
    return torch.ones(q_BLHD.shape[:2], dtype=torch.bool, device=q_BLHD.device)


def sparse_topk_attention_blhd(
    q_BLHD: torch.Tensor,
    k_BLHD: torch.Tensor,
    v_BLHD: torch.Tensor,
    selected_BLS: torch.Tensor,
    *,
    attention_mask_BL: torch.Tensor | None = None,
    scale: float | None = None,
    index_granularity: int = 1,
) -> torch.Tensor:
    """Apply sparse causal attention over selected key indices.

    ``selected_BLS`` is shared across heads. Negative, out-of-range, masked, and
    non-causal keys are ignored. For ``index_granularity=4``, each selected index
    names a pool id and expands to ``pool * 4 + {0,1,2,3}``.
    """

    if scale is None:
        scale = q_BLHD.shape[-1] ** -0.5
    if not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError(f"scale must be positive and finite, got {scale}.")
    if attention_mask_BL is None:
        attention_mask_BL = _default_mask(q_BLHD)
    else:
        attention_mask_BL = attention_mask_BL.bool()
    out_BLHD, _lse_BHL = _attention_fwd(
        q_BLHD,
        k_BLHD,
        v_BLHD,
        selected_BLS,
        attention_mask_BL,
        float(scale),
        index_granularity,
    )
    return out_BLHD


def sparse_topk_attention_bhld(
    q_BHLD: torch.Tensor,
    k_BHLD: torch.Tensor,
    v_BHLD: torch.Tensor,
    selected_BLS: torch.Tensor,
    *,
    attention_mask_BL: torch.Tensor | None = None,
    scale: float | None = None,
    index_granularity: int = 1,
) -> torch.Tensor:
    """BHLD wrapper for the custom sparse attention op."""

    out_BLHD = sparse_topk_attention_blhd(
        q_BHLD.transpose(1, 2),
        k_BHLD.transpose(1, 2),
        v_BHLD.transpose(1, 2),
        selected_BLS,
        attention_mask_BL=attention_mask_BL,
        scale=scale,
        index_granularity=index_granularity,
    )
    return out_BLHD.transpose(1, 2)

