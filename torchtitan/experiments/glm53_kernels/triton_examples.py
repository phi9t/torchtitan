# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small Triton custom-op examples for the GLM kernel convention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scale_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha: tl.constexpr,
    numel: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = x * alpha + y
    tl.store(out_ptr + offsets, out, mask=mask)


def _validate_scale_add(x: torch.Tensor, y: torch.Tensor) -> None:
    if not x.is_cuda:
        raise ValueError("Triton scale_add requires CUDA tensors.")
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    if x.dtype != y.dtype:
        raise TypeError(f"x and y must have the same dtype, got {x.dtype} and {y.dtype}")
    if x.device != y.device:
        raise ValueError(f"x and y must be on the same device, got {x.device} and {y.device}")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported dtype for Triton scale_add: {x.dtype}")


@torch.library.custom_op(
    "glm53_triton_examples::scale_add",
    mutates_args=(),
    device_types="cuda",
)
def _scale_add_triton(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    _validate_scale_add(x, y)
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    block = 256
    grid = (triton.cdiv(x.numel(), block),)
    _scale_add_kernel[grid](x, y, out, float(alpha), x.numel(), BLOCK=block)
    return out


@_scale_add_triton.register_fake
def _scale_add_triton_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    del alpha
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)


def _scale_add_setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[torch.Tensor, torch.Tensor, float],
    output: torch.Tensor,
) -> None:
    del output
    _x, _y, alpha = inputs
    ctx.alpha = float(alpha)


def _scale_add_backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    alpha = ctx.alpha  # pyrefly: ignore[missing-attribute]
    return grad_out * alpha, grad_out, None


_scale_add_triton.register_autograd(
    _scale_add_backward,
    setup_context=_scale_add_setup_context,
)


def triton_scale_add(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Return ``x * alpha + y`` through a Triton torch.library custom op."""

    return _scale_add_triton(x, y, float(alpha))

