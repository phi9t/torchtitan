# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


SUPPORTED_GEOMETRIES = ("anchor", "dot", "cosine", "gauss")


def validate_geometry(geometry: str) -> str:
    normalized = geometry.lower()
    if normalized not in SUPPORTED_GEOMETRIES:
        raise ValueError(
            f"unsupported attention geometry {geometry!r}; expected one of "
            f"{SUPPORTED_GEOMETRIES}"
        )
    return normalized


def rms_normalize(x_BTHD: Tensor, *, eps: float = 0.0) -> Tensor:
    return F.rms_norm(x_BTHD, (x_BTHD.size(-1),), eps=eps)


def l2_normalize(x_BTHD: Tensor, *, eps: float = 0.0) -> Tensor:
    return x_BTHD / x_BTHD.norm(dim=-1, keepdim=True).clamp_min(eps)


def geometry_log_param_for_heads(log_param_H: Tensor, num_heads: int) -> Tensor:
    """Return one log parameter per effective attention head.

    Upstream paired-head layers collapse adjacent heads into one effective head
    with doubled query/key width. The experiment owns one parameter per original
    head, so paired layers use the geometric mean of each adjacent pair.
    """

    if log_param_H.ndim != 1:
        raise ValueError("attention geometry log parameters must be 1-D")
    if log_param_H.numel() == num_heads:
        return log_param_H
    if log_param_H.numel() == 2 * num_heads:
        return log_param_H.view(num_heads, 2).mean(dim=1)
    raise ValueError(
        "attention geometry log parameter count must match the effective "
        f"heads ({num_heads}) or original paired heads ({2 * num_heads}); "
        f"found {log_param_H.numel()}"
    )


def broadcast_head_param(log_param_H: Tensor, q_BTHD: Tensor) -> Tensor:
    per_head_H = geometry_log_param_for_heads(log_param_H, q_BTHD.size(-2))
    return per_head_H.to(device=q_BTHD.device, dtype=torch.float32).view(
        *([1] * (q_BTHD.ndim - 2)), q_BTHD.size(-2), 1
    )


def next_multiple(value: int, *, n: int) -> int:
    return math.ceil(value / n) * n


def gaussian_augmented_qk(
    q_BTHD: Tensor,
    k_BTHD: Tensor,
    log_sigma_H: Tensor,
    *,
    pad_to_multiple: int = 8,
) -> tuple[Tensor, Tensor]:
    """Build augmented q/k for Gaussian logits.

    The attention logit is

        (q.k - ||k||^2 / 2) / sigma_h^2.

    This is represented as q' = [q, 1] and k' = [k, -||k||^2 / 2], with q'
    multiplied by 1 / sigma_h^2. Padding keeps GPU attention kernels on friendly
    head dimensions while preserving the dot product exactly.
    """

    inv_sigma2_BTH1 = torch.exp(-2.0 * broadcast_head_param(log_sigma_H, q_BTHD))
    key_penalty_BTH1 = -0.5 * k_BTHD.float().square().sum(dim=-1, keepdim=True)
    key_penalty_BTH1 = key_penalty_BTH1.to(dtype=k_BTHD.dtype)
    one_BTH1 = torch.ones_like(key_penalty_BTH1, dtype=q_BTHD.dtype)
    q_aug_BTHD = torch.cat((q_BTHD, one_BTH1), dim=-1)
    k_aug_BTHD = torch.cat((k_BTHD, key_penalty_BTH1), dim=-1)

    if pad_to_multiple > 1:
        padded_dim = next_multiple(q_aug_BTHD.size(-1), n=pad_to_multiple)
        pad = padded_dim - q_aug_BTHD.size(-1)
        if pad:
            q_aug_BTHD = F.pad(q_aug_BTHD, (0, pad))
            k_aug_BTHD = F.pad(k_aug_BTHD, (0, pad))

    return q_aug_BTHD * inv_sigma2_BTH1.to(dtype=q_aug_BTHD.dtype), k_aug_BTHD


def prepare_geometry_qk(
    q_BTHD: Tensor,
    k_BTHD: Tensor,
    log_param_H: Tensor,
    *,
    geometry: str,
    anchor_softmax_scale: float,
    eps: float = 0.0,
    gaussian_pad_to_multiple: int = 8,
) -> tuple[Tensor, Tensor, float]:
    """Prepare q/k and scalar SDPA scale for one geometry.

    Inputs and outputs use logical shape B, T, H, D.
    """

    geometry = validate_geometry(geometry)
    if geometry == "anchor":
        q_prepared = rms_normalize(q_BTHD, eps=eps)
        k_prepared = rms_normalize(k_BTHD, eps=eps)
        q_prepared = q_prepared * torch.exp(
            broadcast_head_param(log_param_H, q_BTHD)
        ).to(dtype=q_prepared.dtype)
        return q_prepared, k_prepared, anchor_softmax_scale
    if geometry == "dot":
        q_prepared = q_BTHD * torch.exp(broadcast_head_param(log_param_H, q_BTHD)).to(
            dtype=q_BTHD.dtype
        )
        return q_prepared, k_BTHD, 1.0
    if geometry == "cosine":
        q_prepared = l2_normalize(q_BTHD, eps=eps)
        k_prepared = l2_normalize(k_BTHD, eps=eps)
        q_prepared = q_prepared * torch.exp(
            broadcast_head_param(log_param_H, q_BTHD)
        ).to(dtype=q_prepared.dtype)
        return q_prepared, k_prepared, 1.0

    q_aug, k_aug = gaussian_augmented_qk(
        q_BTHD,
        k_BTHD,
        log_param_H,
        pad_to_multiple=gaussian_pad_to_multiple,
    )
    return q_aug, k_aug, 1.0


def geometry_sdpa(
    q_BTHD: Tensor,
    k_BTHD: Tensor,
    v_BTHD: Tensor,
    log_param_H: Tensor,
    *,
    geometry: str,
    anchor_softmax_scale: float,
    causal: bool = True,
    eps: float = 0.0,
    gaussian_pad_to_multiple: int = 8,
) -> Tensor:
    """Run dense SDPA for tests and CPU probes."""

    q_prepared, k_prepared, scale = prepare_geometry_qk(
        q_BTHD,
        k_BTHD,
        log_param_H,
        geometry=geometry,
        anchor_softmax_scale=anchor_softmax_scale,
        eps=eps,
        gaussian_pad_to_multiple=gaussian_pad_to_multiple,
    )
    y_BHTD = F.scaled_dot_product_attention(
        q_prepared.transpose(1, 2),
        k_prepared.transpose(1, 2),
        v_BTHD.transpose(1, 2),
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
    )
    return y_BHTD.transpose(1, 2)


def explicit_gaussian_attention(
    q_BTHD: Tensor,
    k_BTHD: Tensor,
    v_BTHD: Tensor,
    log_sigma_H: Tensor,
    *,
    causal: bool = True,
) -> Tensor:
    sigma2_BH11 = torch.exp(
        2.0
        * geometry_log_param_for_heads(log_sigma_H, q_BTHD.size(-2))
        .to(device=q_BTHD.device, dtype=torch.float32)
        .view(1, q_BTHD.size(-2), 1, 1)
    )
    q_BHTD = q_BTHD.transpose(1, 2).float()
    k_BHTD = k_BTHD.transpose(1, 2).float()
    v_BHTD = v_BTHD.transpose(1, 2)
    diff_BHTSD = q_BHTD[:, :, :, None, :] - k_BHTD[:, :, None, :, :]
    logits_BHTS = -0.5 * diff_BHTSD.square().sum(dim=-1) / sigma2_BH11
    if causal:
        t = q_BTHD.size(1)
        s = k_BTHD.size(1)
        mask_TS = torch.ones((t, s), device=q_BTHD.device, dtype=torch.bool).tril()
        logits_BHTS = logits_BHTS.masked_fill(~mask_TS.view(1, 1, t, s), -torch.inf)
    weights_BHTS = torch.softmax(logits_BHTS, dim=-1).to(dtype=v_BTHD.dtype)
    return torch.matmul(weights_BHTS, v_BHTD).transpose(1, 2)


def geometry_flash_varlen(
    q_BTHD: Tensor,
    k_BTHD: Tensor,
    v_BTHD: Tensor,
    log_param_H: Tensor,
    *,
    geometry: str,
    anchor_softmax_scale: float,
    cu_seqlens: Tensor,
    max_seqlen: int,
    flash_attn_varlen_func: Callable[..., Tensor],
    causal: bool = True,
    window_size: tuple[int | None, int] | None = None,
    eps: float = 0.0,
    gaussian_pad_to_multiple: int = 8,
) -> Tensor:
    """Run a geometry through FlashAttention varlen with upstream B=1 layout."""

    q_prepared, k_prepared, scale = prepare_geometry_qk(
        q_BTHD,
        k_BTHD,
        log_param_H,
        geometry=geometry,
        anchor_softmax_scale=anchor_softmax_scale,
        eps=eps,
        gaussian_pad_to_multiple=gaussian_pad_to_multiple,
    )
    v_for_attention = v_BTHD
    original_v_dim = v_BTHD.size(-1)
    if q_prepared.size(-1) != v_BTHD.size(-1):
        v_for_attention = F.pad(v_BTHD, (0, q_prepared.size(-1) - v_BTHD.size(-1)))

    kwargs: dict[str, object] = {
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_k": cu_seqlens.clone(),
        "max_seqlen_q": max_seqlen,
        "max_seqlen_k": max_seqlen,
        "causal": causal,
        "softmax_scale": scale,
    }
    if window_size is not None:
        kwargs["window_size"] = window_size

    y_THD = flash_attn_varlen_func(
        q_prepared[0], k_prepared[0], v_for_attention[0], **kwargs
    )
    return y_THD[..., :original_v_dim]


def upstream_rotary_factors(
    *,
    head_dim: int,
    seq_len: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    paired: bool = False,
) -> tuple[Tensor, Tensor]:
    angular_freq = (1 / 1024) ** torch.linspace(
        0,
        1,
        steps=head_dim // 4,
        dtype=torch.float32,
        device=device,
    )
    angular_freq = angular_freq.repeat_interleave(2)
    angular_freq = torch.cat((angular_freq, angular_freq.new_zeros(head_dim // 2)))
    t = torch.arange(2 * seq_len, dtype=torch.float32, device=device)
    if not paired:
        theta = torch.outer(t, angular_freq)
        factor1 = theta.cos().to(dtype=dtype)
        factor2 = theta.sin().to(dtype=dtype)
    else:
        t_even = 2 * t
        t_odd = t_even + 1
        theta1 = torch.outer(t_even, angular_freq)
        theta2 = torch.outer(t_odd, angular_freq)
        factor1 = torch.cat((theta1.cos(), theta2.cos()), dim=-1).to(dtype=dtype)
        factor2 = torch.cat((theta1.sin(), theta2.sin()), dim=-1).to(dtype=dtype)
    factor2[..., 1::2] *= -1
    return factor1, factor2


def apply_upstream_rotary(
    x_BTHD: Tensor, factor1_TD: Tensor, factor2_TD: Tensor
) -> Tensor:
    factor1 = factor1_TD[None, : x_BTHD.size(1), None, :]
    factor2 = factor2_TD[None, : x_BTHD.size(1), None, :]
    x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2)
    x_flip = x_flip.flip(-1).view(x_BTHD.shape)
    return factor1 * x_BTHD + factor2 * x_flip
