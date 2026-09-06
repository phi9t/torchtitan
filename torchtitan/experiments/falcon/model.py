# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tiny Falcon causal LM for the learnability overfit gate.

The decoder shell (Pre-Norm RMSNorm + SwiGLU) is fixed; only the token mixer
changes across science arms. ``FalconConfig.mixer`` selects one of:

- ``falcon``: the existing Falcon-1 / Falcon-1A fast-weight mixer (default,
  delayed pairing + QK-RMSNorm phi).
- ``gdn``: a Gated DeltaNet-style same-step residual write. The math mirrors
  the gated-delta recurrence used elsewhere in the repo
  (``state <- g S + k ((v - S^T k) beta)^T`` on ``k_t``, with L2-normed q/k),
  reimplemented here so the experiment stays self-contained. It does not import
  ``mini_kimi_k3`` or ``qwen3_5``.
- ``softmax``: causal softmax attention with RoPE on q/k, same head layout.

Legend: B batch, L sequence, N heads, K key/head dim, V value/head dim,
D hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.experiments.falcon.falcon import (
    falcon_masked_parallel_forward,
    FalconAlignment,
    FalconPhi,
    FalconVariant,
)

FalconMixerKind = Literal["falcon", "gdn", "softmax"]


@dataclass(kw_only=True)
class FalconConfig:
    vocab_size: int = 32
    hidden_size: int = 32
    num_hidden_layers: int = 2
    num_heads: int = 4
    head_dim: int = 8
    intermediate_size: int = 64
    seq_len: int = 16
    rms_norm_eps: float = 1.0e-5
    variant: FalconVariant = "falcon1a"
    alignment: FalconAlignment = "delayed"
    mixer: FalconMixerKind = "falcon"
    # QK feature map for the Falcon mixer: "rms" (QK-RMSNorm, paper default) or
    # "l2" (QK-L2, arm A5). Only consumed by the "falcon" mixer.
    phi: FalconPhi = "rms"
    rope_theta: float = 10000.0


def tiny_falcon_config(
    *,
    variant: FalconVariant = "falcon1a",
    alignment: FalconAlignment = "delayed",
    mixer: FalconMixerKind = "falcon",
    phi: FalconPhi = "rms",
) -> FalconConfig:
    return FalconConfig(variant=variant, alignment=alignment, mixer=mixer, phi=phi)


class FalconRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        x_float = x_BLD.float()
        normed = x_float * torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (self.weight.float() * normed).to(x_BLD.dtype)


class FalconMixer(nn.Module):
    """One multi-head Falcon-1 / Falcon-1A block."""

    def __init__(self, config: FalconConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        inner = config.num_heads * config.head_dim
        self.q_proj = nn.Linear(hidden, inner, bias=False)
        self.k_proj = nn.Linear(hidden, inner, bias=False)
        self.v_proj = nn.Linear(hidden, inner, bias=False)
        self.beta_proj = nn.Linear(hidden, config.num_heads, bias=True)
        self.lambda_proj = nn.Linear(hidden, config.num_heads, bias=True)
        self.o_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x_BLD.shape
        heads = self.config.num_heads
        head_dim = self.config.head_dim
        q_BLNK = self.q_proj(x_BLD).view(batch, seq_len, heads, head_dim)
        k_BLNK = self.k_proj(x_BLD).view(batch, seq_len, heads, head_dim)
        v_BLNV = self.v_proj(x_BLD).view(batch, seq_len, heads, head_dim)
        beta_BLN = 2.0 * torch.sigmoid(self.beta_proj(x_BLD))
        lambda_BLN = F.softplus(self.lambda_proj(x_BLD))
        # Use the masked-parallel view for training: it is bitwise-close to the
        # recurrent oracle (see test_falcon_kernel identity gate) but avoids the
        # Python-level sequential scan, which is prohibitively slow at seq 512
        # on GPU. Identity is proven for every variant x alignment x phi.
        out_BLNV, _ = falcon_masked_parallel_forward(
            q_BLNK,
            k_BLNK,
            v_BLNV,
            beta_BLN,
            lambda_BLN,
            variant=self.config.variant,
            alignment=self.config.alignment,
            phi=self.config.phi,
        )
        return self.o_proj(out_BLNV.reshape(batch, seq_len, heads * head_dim))


def _l2_normalize(x_BLND: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    return x_BLND * torch.rsqrt(x_BLND.pow(2).sum(dim=-1, keepdim=True) + eps)


def gdn_reference_forward(
    q_BLNK: torch.Tensor,
    k_BLNK: torch.Tensor,
    v_BLNV: torch.Tensor,
    beta_BLN: torch.Tensor,
    g_BLN: torch.Tensor,
) -> torch.Tensor:
    """Gated DeltaNet-style same-step residual recurrence.

    This is the strongest published recurrent neighbor control (arm A1). The
    recurrence is same-timestep (write and read at the same ``t``):

        S_t = g_t S_{t-1} + k_t ((v_t - S_{t-1}^T k_t) beta_t)^T
        o_t = S_t^T q_t

    with q, k L2-normalized per head, and q additionally scaled by ``K**-0.5``
    after L2 (matching ``kda_reference_forward``). This reimplements the
    gated-delta math
    (the same contract that Qwen3.5 / Mini-K3 use as a CPU oracle) so the Falcon
    experiment stays self-contained; it deliberately does not import those
    modules. ``g_t`` is a scalar per-head decay in ``(0, 1)``.

    Legend: B batch, L sequence, N heads, K key dim, V value dim.
    """
    q_work = _l2_normalize(q_BLNK.float())
    k_work = _l2_normalize(k_BLNK.float())
    # Scale q by K**-0.5 after L2, matching kda_reference_forward. This keeps
    # the read magnitude comparable across head dims (the gated-delta oracle
    # uses the same scaling on q only).
    key_dim = q_work.shape[-1]
    q_work = q_work * (key_dim**-0.5)
    v_work = v_BLNV.float()
    beta_work = beta_BLN.float()
    g_work = g_BLN.float()

    batch, seq_len, heads, key_dim = q_work.shape
    value_dim = v_work.shape[-1]
    out_BLNV = q_work.new_zeros(batch, seq_len, heads, value_dim)
    state_BNKV = q_work.new_zeros(batch, heads, key_dim, value_dim)

    for step in range(seq_len):
        q_BNK = q_work[:, step]
        k_BNK = k_work[:, step]
        v_BNV = v_work[:, step]
        decay_BN11 = g_work[:, step].unsqueeze(-1).unsqueeze(-1)
        beta_BN1 = beta_work[:, step].unsqueeze(-1)

        state_BNKV = state_BNKV * decay_BN11
        kv_mem_BNV = torch.einsum("bnkv,bnk->bnv", state_BNKV, k_BNK)
        delta_BNV = (v_BNV - kv_mem_BNV) * beta_BN1
        state_BNKV = state_BNKV + torch.einsum("bnk,bnv->bnkv", k_BNK, delta_BNV)
        out_BLNV[:, step] = torch.einsum("bnkv,bnk->bnv", state_BNKV, q_BNK)

    return out_BLNV.to(q_BLNK.dtype)


class FalconGDNMixer(nn.Module):
    """Gated DeltaNet-style same-step residual mixer (arm A1 control)."""

    def __init__(self, config: FalconConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        inner = config.num_heads * config.head_dim
        self.q_proj = nn.Linear(hidden, inner, bias=False)
        self.k_proj = nn.Linear(hidden, inner, bias=False)
        self.v_proj = nn.Linear(hidden, inner, bias=False)
        self.beta_proj = nn.Linear(hidden, config.num_heads, bias=True)
        # A per-head decay gate in (0, 1). Bias init leaves sigmoid near 1 so
        # the fresh state is nearly persistent, matching gated-delta defaults.
        self.gate_proj = nn.Linear(hidden, config.num_heads, bias=True)
        self.o_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x_BLD.shape
        heads = self.config.num_heads
        head_dim = self.config.head_dim
        q_BLNK = self.q_proj(x_BLD).view(batch, seq_len, heads, head_dim)
        k_BLNK = self.k_proj(x_BLD).view(batch, seq_len, heads, head_dim)
        v_BLNV = self.v_proj(x_BLD).view(batch, seq_len, heads, head_dim)
        beta_BLN = torch.sigmoid(self.beta_proj(x_BLD))
        g_BLN = torch.sigmoid(self.gate_proj(x_BLD))
        out_BLNV = gdn_reference_forward(q_BLNK, k_BLNK, v_BLNV, beta_BLN, g_BLN)
        return self.o_proj(out_BLNV.reshape(batch, seq_len, heads * head_dim))


def _build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) of shape (seq_len, head_dim) for interleaved RoPE."""
    if head_dim % 2 != 0:
        raise ValueError("softmax mixer head_dim must be even for RoPE")
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("l,k->lk", positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


class FalconSoftmaxMixer(nn.Module):
    """Causal softmax attention with RoPE (arm A0, paper Table 1 control)."""

    def __init__(self, config: FalconConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        inner = config.num_heads * config.head_dim
        self.q_proj = nn.Linear(hidden, inner, bias=False)
        self.k_proj = nn.Linear(hidden, inner, bias=False)
        self.v_proj = nn.Linear(hidden, inner, bias=False)
        self.o_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x_BLD.shape
        heads = self.config.num_heads
        head_dim = self.config.head_dim
        q_BLNK = self.q_proj(x_BLD).view(batch, seq_len, heads, head_dim)
        k_BLNK = self.k_proj(x_BLD).view(batch, seq_len, heads, head_dim)
        v_BLNV = self.v_proj(x_BLD).view(batch, seq_len, heads, head_dim)

        cos, sin = _build_rope_cache(
            seq_len,
            head_dim,
            self.config.rope_theta,
            x_BLD.device,
            torch.float32,
        )
        cos = cos.view(1, seq_len, 1, head_dim)
        sin = sin.view(1, seq_len, 1, head_dim)
        q_f = q_BLNK.float()
        k_f = k_BLNK.float()
        q_rot = q_f * cos + _rotate_half(q_f) * sin
        k_rot = k_f * cos + _rotate_half(k_f) * sin

        # (B, N, L, K) for scaled_dot_product_attention.
        q_BNLK = q_rot.transpose(1, 2)
        k_BNLK = k_rot.transpose(1, 2)
        v_BNLV = v_BLNV.float().transpose(1, 2)
        out_BNLV = F.scaled_dot_product_attention(
            q_BNLK, k_BNLK, v_BNLV, is_causal=True
        )
        out_BLNV = out_BNLV.transpose(1, 2).to(x_BLD.dtype)
        return self.o_proj(out_BLNV.reshape(batch, seq_len, heads * head_dim))


def build_mixer(config: FalconConfig) -> nn.Module:
    if config.mixer == "falcon":
        return FalconMixer(config)
    if config.mixer == "gdn":
        return FalconGDNMixer(config)
    if config.mixer == "softmax":
        return FalconSoftmaxMixer(config)
    raise ValueError(f"unsupported Falcon mixer: {config.mixer}")


class FalconDecoderLayer(nn.Module):
    def __init__(self, config: FalconConfig) -> None:
        super().__init__()
        self.input_norm = FalconRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mixer = build_mixer(config)
        self.post_norm = FalconRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        x_BLD = x_BLD + self.mixer(self.input_norm(x_BLD))
        ff_in = self.post_norm(x_BLD)
        return x_BLD + self.w2(F.silu(self.w1(ff_in)) * self.w3(ff_in))


class FalconForCausalLM(nn.Module):
    def __init__(self, config: FalconConfig) -> None:
        super().__init__()
        if config.hidden_size != config.num_heads * config.head_dim:
            raise ValueError("hidden_size must equal num_heads * head_dim")
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            FalconDecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = FalconRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: object | None = None,
    ) -> torch.Tensor:
        del positions, attention_masks
        hidden_BLD = self.embed_tokens(tokens)
        for layer in self.layers:
            hidden_BLD = layer(hidden_BLD)
        return self.lm_head(self.norm(hidden_BLD))
