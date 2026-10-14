# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi Delta Attention element for the GLM-5.3-Flash speedrun arm A3.

Shape suffixes:
    B = batch, L = sequence length, H = attention heads, D = model dimension,
    K = head dimension, C = projection channels, W = convolution width.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common import attn_gym as common_attn_gym

from .config import SpeedrunElementsConfig


def l2norm(x: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    x_float = x.float()
    denom = torch.sqrt(x_float.square().sum(dim=-1, keepdim=True) + eps)
    return (x_float / denom).to(x.dtype)


def bound_kda_gate(
    raw_gate_BLHK: torch.Tensor,
    A_log_H: torch.Tensor,
    dt_bias_HK: torch.Tensor,
    *,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    gate_BLHK = raw_gate_BLHK.float() + dt_bias_HK.float().view(
        1,
        1,
        *dt_bias_HK.shape,
    )
    decay_11H1 = torch.exp(A_log_H.float()).view(1, 1, -1, 1)
    return lower_bound * torch.sigmoid(decay_11H1 * gate_BLHK)


def kda_recurrent(
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    v_BLHK: torch.Tensor,
    gate_BLHK: torch.Tensor,
    beta_BLH: torch.Tensor,
    *,
    initial_state_BHKK: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_dtype = v_BLHK.dtype
    q_BLHK = l2norm(q_BLHK).float()
    k_BLHK = l2norm(k_BLHK).float()
    v_BLHK = v_BLHK.float()
    gate_BLHK = gate_BLHK.float()
    beta_BLH = beta_BLH.float()
    batch_size, seq_len, num_heads, head_dim = q_BLHK.shape
    if initial_state_BHKK is None:
        state_BHKK = torch.zeros(
            batch_size,
            num_heads,
            head_dim,
            head_dim,
            dtype=torch.float32,
            device=q_BLHK.device,
        )
    else:
        state_BHKK = initial_state_BHKK.float()
    out_steps = []
    scale = head_dim**-0.5
    for idx in range(seq_len):
        state_BHKK = torch.exp(gate_BLHK[:, idx]).unsqueeze(-1) * state_BHKK
        k_BHK = k_BLHK[:, idx]
        old_BHV = torch.einsum("bhkv,bhk->bhv", state_BHKK, k_BHK)
        delta_BHV = beta_BLH[:, idx].unsqueeze(-1) * (v_BLHK[:, idx] - old_BHV)
        state_BHKK = state_BHKK + torch.einsum("bhk,bhv->bhkv", k_BHK, delta_BHV)
        out_steps.append(
            torch.einsum("bhkv,bhk->bhv", state_BHKK, q_BLHK[:, idx] * scale)
        )
    out_BLHK = torch.stack(out_steps, dim=1).to(output_dtype)
    return out_BLHK, state_BHKK


def kda_chunked_reference(
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    v_BLHK: torch.Tensor,
    gate_BLHK: torch.Tensor,
    beta_BLH: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    output_dtype = v_BLHK.dtype
    q_BHLK, k_BHLK, v_BHLK, gate_BHLK = [
        tensor.transpose(1, 2).contiguous().to(torch.float32)
        for tensor in (
            l2norm(q_BLHK),
            l2norm(k_BLHK),
            v_BLHK,
            gate_BLHK,
        )
    ]
    beta_BHL = beta_BLH.transpose(1, 2).contiguous().float()
    batch_size, num_heads, seq_len, k_head_dim = k_BHLK.shape
    v_head_dim = v_BHLK.shape[-1]
    scale = k_head_dim**-0.5
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    total_seq_len = seq_len + pad_size

    q_BHCLK = (F.pad(q_BHLK, (0, 0, 0, pad_size)) * scale).reshape(
        batch_size,
        num_heads,
        -1,
        chunk_size,
        k_head_dim,
    )
    k_BHCLK = F.pad(k_BHLK, (0, 0, 0, pad_size)).reshape(
        batch_size,
        num_heads,
        -1,
        chunk_size,
        k_head_dim,
    )
    v_BHCLK = F.pad(v_BHLK, (0, 0, 0, pad_size)).reshape(
        batch_size,
        num_heads,
        -1,
        chunk_size,
        v_head_dim,
    )
    gate_BHCLK = F.pad(gate_BHLK, (0, 0, 0, pad_size)).reshape(
        batch_size,
        num_heads,
        -1,
        chunk_size,
        k_head_dim,
    )
    beta_BHCL = F.pad(beta_BHL, (0, pad_size)).reshape(
        batch_size,
        num_heads,
        -1,
        chunk_size,
    )
    v_beta_BHCLK = v_BHCLK * beta_BHCL.unsqueeze(-1)
    k_beta_BHCLK = k_BHCLK * beta_BHCL.unsqueeze(-1)
    gate_cumsum_BHCLK = gate_BHCLK.cumsum(dim=-2)

    device = q_BLHK.device
    full_mask_TT = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device),
        diagonal=0,
    )
    strict_mask_TT = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device),
        diagonal=1,
    )
    decay_mask_BHCTTK = (
        gate_cumsum_BHCLK.unsqueeze(-2) - gate_cumsum_BHCLK.unsqueeze(-3)
    ).masked_fill(strict_mask_TT[..., None], float("-inf"))
    decay_mask_BHCTTK = decay_mask_BHCTTK.exp().float()
    attn_BHCTT = -(
        k_beta_BHCLK.unsqueeze(-2) * k_BHCLK.unsqueeze(-3) * decay_mask_BHCTTK
    ).sum(dim=-1)
    attn_BHCTT = attn_BHCTT.masked_fill(full_mask_TT, 0)
    for idx in range(1, chunk_size):
        row_BHCT = attn_BHCTT[..., idx, :idx].clone()
        sub_BHCUT = attn_BHCTT[..., :idx, :idx].clone()
        attn_BHCTT[..., idx, :idx] = row_BHCT + (
            row_BHCT.unsqueeze(-1) * sub_BHCUT
        ).sum(dim=-2)
    attn_BHCTT = attn_BHCTT + torch.eye(
        chunk_size,
        dtype=attn_BHCTT.dtype,
        device=device,
    )

    value_BHCLK = attn_BHCTT @ v_beta_BHCLK
    k_cumdecay_BHCLK = attn_BHCTT @ (k_beta_BHCLK * gate_cumsum_BHCLK.exp())
    state_BHKK = torch.zeros(
        batch_size,
        num_heads,
        k_head_dim,
        v_head_dim,
        dtype=torch.float32,
        device=device,
    )
    out_BHCLK = torch.zeros_like(value_BHCLK)
    upper_mask_TT = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device),
        diagonal=1,
    )
    for chunk_idx in range(total_seq_len // chunk_size):
        q_BHTK = q_BHCLK[:, :, chunk_idx]
        k_BHTK = k_BHCLK[:, :, chunk_idx]
        v_BHTK = value_BHCLK[:, :, chunk_idx]
        gate_BHTK = gate_cumsum_BHCLK[:, :, chunk_idx]
        attn_inter_BHTK = (q_BHTK * gate_BHTK.exp()) @ state_BHKK
        attn_intra_BHTT = (
            q_BHTK.unsqueeze(-2)
            * k_BHTK.unsqueeze(-3)
            * decay_mask_BHCTTK[:, :, chunk_idx]
        ).sum(dim=-1)
        attn_intra_BHTT = attn_intra_BHTT.masked_fill(upper_mask_TT, 0)
        v_prime_BHTK = k_cumdecay_BHCLK[:, :, chunk_idx] @ state_BHKK
        v_new_BHTK = v_BHTK - v_prime_BHTK
        out_BHCLK[:, :, chunk_idx] = attn_inter_BHTK + (attn_intra_BHTT @ v_new_BHTK)
        state_BHKK = state_BHKK * gate_BHTK[:, :, -1].exp().unsqueeze(-1)
        state_BHKK = (
            state_BHKK
            + (k_BHTK * (gate_BHTK[:, :, -1:] - gate_BHTK).exp()).transpose(-1, -2)
            @ v_new_BHTK
        )

    out_BHLK = out_BHCLK.reshape(batch_size, num_heads, total_seq_len, v_head_dim)
    return out_BHLK[:, :, :seq_len].transpose(1, 2).contiguous().to(output_dtype)


def kda_attention_gym(
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    v_BLHK: torch.Tensor,
    raw_gate_BLHK: torch.Tensor,
    raw_beta_BLH: torch.Tensor,
    A_log_H: torch.Tensor,
    dt_bias_HK: torch.Tensor,
    *,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    """Run the KDA recurrence through the shared Attention Gym chunk kernel."""

    output_dtype = v_BLHK.dtype
    gate_BLHK = common_attn_gym.bound_gate(
        raw_gate_BLHK,
        A_log_H.float(),
        dt_bias_HK.float(),
        lower_bound=lower_bound,
        impl="fused",
    )
    out_BLHK, _ = common_attn_gym.chunk_kda(
        common_attn_gym.l2norm(q_BLHK),
        common_attn_gym.l2norm(k_BLHK),
        v_BLHK,
        gate_BLHK,
        raw_beta_BLH.float().sigmoid(),
        autotune=not torch.are_deterministic_algorithms_enabled(),
    )
    return out_BLHK.to(output_dtype)


class GatedRMSNorm(nn.Module):
    def __init__(self, head_dim: int, *, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x_BLHK: torch.Tensor, gate_BLHK: torch.Tensor) -> torch.Tensor:
        x_float = x_BLHK.float()
        x_float = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        x_float = x_float * self.weight.float()
        x_float = x_float * torch.sigmoid(gate_BLHK.float())
        return x_float.to(x_BLHK.dtype)


class KDALayer(nn.Module):
    def __init__(self, config: SpeedrunElementsConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.qkv_dim = config.hidden_size
        self.conv_kernel_size = config.kda_conv_kernel_size
        self.q_proj = nn.Linear(config.hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.qkv_dim, bias=False)
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
        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.A_log = nn.Parameter(torch.zeros(self.num_heads))
        self.b_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.g_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = GatedRMSNorm(self.head_dim)
        self.o_proj = nn.Linear(self.qkv_dim, config.hidden_size, bias=False)
        self.last_impl = "uninitialized"

    def forward(self, x_BLD: torch.Tensor, *, impl: str | None = None) -> torch.Tensor:
        batch_size, seq_len = x_BLD.shape[:2]
        q_BLC = self._causal_conv(self.q_proj(x_BLD), self.q_conv1d.weight)
        k_BLC = self._causal_conv(self.k_proj(x_BLD), self.k_conv1d.weight)
        v_BLC = self._causal_conv(self.v_proj(x_BLD), self.v_conv1d.weight)
        q_BLHK = q_BLC.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_BLHK = k_BLC.view_as(q_BLHK)
        v_BLHK = v_BLC.view_as(q_BLHK)
        raw_gate_BLHK = self.f_b_proj(self.f_a_proj(x_BLD)).view_as(q_BLHK)
        gate_BLHK = bound_kda_gate(
            raw_gate_BLHK,
            self.A_log,
            self.dt_bias,
            lower_bound=self.config.kda_gate_lower_bound,
        )
        raw_beta_BLH = self.b_proj(x_BLD)
        beta_BLH = torch.sigmoid(raw_beta_BLH)
        effective_impl = self.config.kda_impl if impl is None else impl
        if effective_impl == "auto":
            effective_impl = "attn_gym" if x_BLD.is_cuda else "chunked"
        if effective_impl == "recurrent":
            out_BLHK, _ = kda_recurrent(q_BLHK, k_BLHK, v_BLHK, gate_BLHK, beta_BLH)
            self.last_impl = "recurrent"
        elif effective_impl == "chunked":
            out_BLHK = kda_chunked_reference(
                q_BLHK, k_BLHK, v_BLHK, gate_BLHK, beta_BLH
            )
            self.last_impl = "chunked"
        elif effective_impl == "attn_gym":
            try:
                out_BLHK = kda_attention_gym(
                    q_BLHK,
                    k_BLHK,
                    v_BLHK,
                    raw_gate_BLHK,
                    raw_beta_BLH,
                    self.A_log,
                    self.dt_bias,
                    lower_bound=self.config.kda_gate_lower_bound,
                )
                self.last_impl = "attn_gym"
            except (ImportError, RuntimeError):
                if impl == "attn_gym" or self.config.kda_impl == "attn_gym":
                    raise
                out_BLHK = kda_chunked_reference(
                    q_BLHK, k_BLHK, v_BLHK, gate_BLHK, beta_BLH
                )
                self.last_impl = "chunked_fallback"
        else:
            raise ValueError(
                "unknown KDA impl "
                f"{effective_impl!r}; expected 'auto', 'attn_gym', "
                "'chunked', or 'recurrent'."
            )
        out_gate_BLHK = self.g_b_proj(self.g_a_proj(x_BLD)).view_as(out_BLHK)
        out_BLC = self.o_norm(out_BLHK, out_gate_BLHK).reshape(batch_size, seq_len, -1)
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
        return F.silu(out_BCL[..., : x_BCL.shape[-1]]).transpose(1, 2)
