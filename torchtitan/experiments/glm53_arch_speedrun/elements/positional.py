# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RoPE helpers for the B0/A2 switch."""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def rotary_frequencies(
    seq_len: int,
    head_dim: int,
    *,
    device: torch.device,
    theta: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even.")
    positions_L = torch.arange(seq_len, device=device, dtype=torch.float32)
    dim_K = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq_K = theta ** (-dim_K / head_dim)
    freqs_LK = torch.outer(positions_L, inv_freq_K)
    freqs_LD = torch.repeat_interleave(freqs_LK, repeats=2, dim=-1)
    return freqs_LD.cos(), freqs_LD.sin()


def apply_rope(
    q_BLHD: torch.Tensor,
    k_BLHD: torch.Tensor,
    *,
    theta: float = 10_000.0,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if positions is not None and positions.shape != q_BLHD.shape[:2]:
        raise ValueError(
            "positions must have shape [B, L] when provided, "
            f"got {tuple(positions.shape)} for q {tuple(q_BLHD.shape)}."
        )
    cos_LD, sin_LD = rotary_frequencies(
        q_BLHD.shape[1],
        q_BLHD.shape[-1],
        device=q_BLHD.device,
        theta=theta,
    )
    if positions is None:
        cos_BLD = cos_LD[None]
        sin_BLD = sin_LD[None]
    else:
        cos_BLD = cos_LD[positions]
        sin_BLD = sin_LD[positions]
    cos_BL1D = cos_BLD[:, :, None, :].to(q_BLHD.dtype)
    sin_BL1D = sin_BLD[:, :, None, :].to(q_BLHD.dtype)
    return (
        q_BLHD * cos_BL1D + rotate_half(q_BLHD) * sin_BL1D,
        k_BLHD * cos_BL1D + rotate_half(k_BLHD) * sin_BL1D,
    )


def maybe_apply_rope(
    q_BLHD: torch.Tensor,
    k_BLHD: torch.Tensor,
    *,
    rope_enabled: bool,
    theta: float = 10_000.0,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rope_enabled:
        return q_BLHD, k_BLHD
    return apply_rope(q_BLHD, k_BLHD, theta=theta, positions=positions)
