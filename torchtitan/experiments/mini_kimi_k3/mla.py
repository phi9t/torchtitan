# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mini Kimi K3 MLA attention primitives."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MiniK3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        return (self.weight.float() * x_float * torch.rsqrt(variance + self.eps)).to(
            dtype
        )


def apply_rotary_pos_emb(
    x: torch.Tensor,
    *,
    positions: torch.Tensor,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Apply complex-pair rotary embeddings to ``x``.

    ``x`` has shape ``[B, N, L, D]`` and ``positions`` has shape ``[B, L]``.
    """

    if x.ndim != 4:
        raise ValueError(f"x must have shape [B, N, L, D], got {tuple(x.shape)}")
    if positions.ndim != 2:
        raise ValueError(
            f"positions must have shape [B, L], got {tuple(positions.shape)}"
        )
    if positions.shape[0] != x.shape[0] or positions.shape[1] != x.shape[2]:
        raise ValueError(
            "positions shape must match x batch and sequence dimensions "
            f"({tuple(positions.shape)} vs {tuple(x.shape)})"
        )
    if x.shape[-1] % 2 != 0:
        raise ValueError("rotary dimension must be even")
    if rope_theta <= 0:
        raise ValueError("rope_theta must be > 0")

    dtype = x.dtype
    half_dim = x.shape[-1] // 2
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, half_dim, device=x.device, dtype=torch.float32) / half_dim)
    )
    angles = positions.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * inv_freq
    freqs_cis = torch.polar(torch.ones_like(angles), angles).unsqueeze(1)
    x_complex = torch.view_as_complex(
        x.float().contiguous().reshape(*x.shape[:-1], half_dim, 2)
    )
    return torch.view_as_real(x_complex * freqs_cis).flatten(-2).to(dtype)


class MiniK3MLAAttention(nn.Module):
    """Eager Mini-K3 multi-latent attention.

    This is a CPU-friendly attention primitive for oracle and assembly work. It
    follows the first-party MLA projection layout, output gate, RoPE over the
    rotary q/k slice, and asymmetric query/value head handling.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rms_norm_eps: float = 1.0e-5,
        use_output_gate: bool = True,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if kv_lora_rank <= 0:
            raise ValueError("kv_lora_rank must be > 0")
        if qk_nope_head_dim <= 0:
            raise ValueError("qk_nope_head_dim must be > 0")
        if qk_rope_head_dim < 0:
            raise ValueError("qk_rope_head_dim must be >= 0")
        if qk_rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be even for RoPE")
        if v_head_dim <= 0:
            raise ValueError("v_head_dim must be > 0")
        if q_lora_rank is not None and q_lora_rank <= 0:
            raise ValueError("q_lora_rank must be > 0 when provided")
        if rope_theta <= 0:
            raise ValueError("rope_theta must be > 0")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.scaling = self.q_head_dim**-0.5
        self.use_output_gate = use_output_gate
        self.rope_theta = rope_theta
        self.low_rank_norm_eps = 1.0e-6

        if q_lora_rank is None:
            self.q_proj = nn.Linear(
                hidden_size, num_heads * self.q_head_dim, bias=False
            )
            self.q_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
        else:
            self.q_proj = None
            self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_a_layernorm = MiniK3RMSNorm(
                q_lora_rank,
                eps=self.low_rank_norm_eps,
            )
            self.q_b_proj = nn.Linear(
                q_lora_rank, num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = MiniK3RMSNorm(
            kv_lora_rank,
            eps=self.low_rank_norm_eps,
        )
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)
        self.g_proj = (
            nn.Linear(hidden_size, num_heads * v_head_dim, bias=False)
            if use_output_gate
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states last dimension must match hidden_size "
                f"({hidden_states.shape[-1]} != {self.hidden_size})"
            )

        B, L, _ = hidden_states.shape
        q_rot, k_rot, q_pass, k_pass, value_states = self._project_qkv(hidden_states)
        del positions

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        attention_value_states = value_states
        if self.q_head_dim != self.v_head_dim:
            attention_value_states = F.pad(
                value_states,
                [0, self.q_head_dim - self.v_head_dim],
            )

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            attention_value_states,
            attn_mask=None,
            dropout_p=0.0,
            scale=self.scaling,
            is_causal=L > 1,
        )
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[..., : self.v_head_dim]

        attn_output = attn_output.transpose(1, 2).reshape(B, L, -1).contiguous()
        if self.g_proj is not None:
            attn_output = attn_output * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(attn_output)

    def rotary_projections_for_test(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_rot, k_rot, _, _, _ = self._project_qkv(hidden_states)
        if self.qk_rope_head_dim == 0:
            return q_rot, k_rot
        return (
            apply_rotary_pos_emb(
                q_rot, positions=positions, rope_theta=self.rope_theta
            ),
            apply_rotary_pos_emb(
                k_rot, positions=positions, rope_theta=self.rope_theta
            ),
        )

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, _ = hidden_states.shape
        if self.q_proj is not None:
            q_states = self.q_proj(hidden_states)
        else:
            assert self.q_a_proj is not None
            assert self.q_a_layernorm is not None
            assert self.q_b_proj is not None
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(B, L, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q_states,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_states = self.kv_b_proj(self.kv_a_layernorm(k_pass))
        kv_states = kv_states.view(
            B,
            L,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ).transpose(1, 2)
        k_pass, value_states = torch.split(
            kv_states,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        k_rot = k_rot.view(B, 1, L, self.qk_rope_head_dim).expand(
            B,
            self.num_heads,
            L,
            self.qk_rope_head_dim,
        )
        return q_rot, k_rot, q_pass, k_pass, value_states
