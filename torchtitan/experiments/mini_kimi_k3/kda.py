# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mini Kimi K3 KDA support helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MiniK3RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        x_norm = self.weight.float() * x_float * torch.rsqrt(variance + self.eps)
        return (x_norm.to(dtype) * torch.sigmoid(gate.float())).to(dtype)


class MiniK3DeltaAttention(nn.Module):
    """CPU-reference KDA attention module for Mini-K3 assembly tests."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int = 4,
        gate_lower_bound: float | None = -5.0,
        use_full_rank_gate: bool = True,
        use_fla_backend: bool = False,
        rms_norm_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if head_dim <= 0:
            raise ValueError("head_dim must be > 0")
        if hidden_size != num_heads * head_dim:
            raise ValueError(
                "hidden_size must equal num_heads * head_dim "
                f"({hidden_size} != {num_heads * head_dim})"
            )
        if conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be > 0")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv_kernel_size = conv_kernel_size
        self.gate_lower_bound = gate_lower_bound
        self.use_full_rank_gate = use_full_rank_gate
        self.use_fla_backend = use_fla_backend

        projection_size = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.q_conv1d = nn.Conv1d(
            projection_size,
            projection_size,
            conv_kernel_size,
            groups=projection_size,
            bias=False,
        )
        self.k_conv1d = nn.Conv1d(
            projection_size,
            projection_size,
            conv_kernel_size,
            groups=projection_size,
            bias=False,
        )
        self.v_conv1d = nn.Conv1d(
            projection_size,
            projection_size,
            conv_kernel_size,
            groups=projection_size,
            bias=False,
        )
        self.A_log = nn.Parameter(
            torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, projection_size, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))
        initialize_dt_bias_(self.dt_bias)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        if use_full_rank_gate:
            self.g_proj = nn.Linear(hidden_size, projection_size, bias=False)
            self.g_a_proj = None
            self.g_b_proj = None
        else:
            self.g_proj = None
            self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
            self.g_b_proj = nn.Linear(head_dim, projection_size, bias=False)
        self.o_norm = MiniK3RMSNormGated(head_dim, eps=rms_norm_eps)
        self._fla_o_norm = None
        self.o_proj = nn.Linear(projection_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states last dimension must match hidden_size "
                f"({hidden_states.shape[-1]} != {self.hidden_size})"
            )

        B, L, _ = hidden_states.shape
        use_fla_backend = _should_use_fla_backend(self.use_fla_backend, hidden_states)
        q = self._causal_conv(
            self.q_proj(hidden_states),
            self.q_conv1d,
            use_fla_backend=use_fla_backend,
        )
        k = self._causal_conv(
            self.k_proj(hidden_states),
            self.k_conv1d,
            use_fla_backend=use_fla_backend,
        )
        v = self._causal_conv(
            self.v_proj(hidden_states),
            self.v_conv1d,
            use_fla_backend=use_fla_backend,
        )

        feature_gate = self.f_b_proj(self.f_a_proj(hidden_states)).view(
            B, L, self.num_heads, self.head_dim
        )
        beta = self.b_proj(hidden_states).float()

        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_heads, self.head_dim)
        v = v.view(B, L, self.num_heads, self.head_dim)
        if use_fla_backend:
            out, final_state = _load_fla_chunk_kda()(
                q=q,
                k=k,
                v=v,
                g=feature_gate,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=self.gate_lower_bound is not None,
                lower_bound=self.gate_lower_bound,
                transpose_state_layout=True,
            )
        else:
            out, final_state = kda_reference_forward(
                q,
                k,
                v,
                feature_gate,
                beta,
                a_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=initial_state,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=self.gate_lower_bound,
            )

        if self.g_proj is not None:
            output_gate = self.g_proj(hidden_states)
        else:
            assert self.g_a_proj is not None
            assert self.g_b_proj is not None
            output_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        output_gate = output_gate.view(B, L, self.num_heads, self.head_dim)
        if use_fla_backend:
            out = self._fla_output_norm(out, output_gate)
        else:
            out = self.o_norm(out, output_gate)
        out = out.reshape(B, L, self.hidden_size)
        return self.o_proj(out), final_state

    def _causal_conv(
        self,
        x_BLD: torch.Tensor,
        conv: nn.Conv1d,
        *,
        use_fla_backend: bool = False,
    ) -> torch.Tensor:
        if use_fla_backend:
            out, _ = _load_fla_causal_conv1d()(
                x=x_BLD,
                weight=conv.weight.squeeze(1),
                bias=conv.bias,
                output_final_state=False,
                activation="silu",
                backend="triton",
            )
            return out
        x_BDL = F.pad(x_BLD.transpose(1, 2), [self.conv_kernel_size - 1, 0])
        return F.silu(conv(x_BDL).transpose(1, 2))

    def _fla_output_norm(self, out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if self._fla_o_norm is None:
            norm_cls = _load_fla_rms_norm_gated()
            self._fla_o_norm = norm_cls(
                self.head_dim,
                eps=self.o_norm.eps,
                activation="sigmoid",
            )
        self._fla_o_norm = self._fla_o_norm.to(
            device=out.device,
            dtype=out.dtype,
        )
        with torch.no_grad():
            self._fla_o_norm.weight.copy_(self.o_norm.weight.to(out.device, out.dtype))
        return self._fla_o_norm(out, gate)


def kda_reference_forward(
    q_BLNK: torch.Tensor,
    k_BLNK: torch.Tensor,
    v_BLNV: torch.Tensor,
    g: torch.Tensor,
    beta_BLN: torch.Tensor,
    *,
    a_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    lower_bound: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slow torch reference for the KDA recurrence boundary.

    This mirrors the same gated-delta recurrence contract that TorchTitan's
    Qwen3.5 tests use as a CPU oracle. It is an oracle and shape contract, not a
    training backend.
    """

    g_BLNK = _prepare_kda_gate(
        g,
        q_BLNK=q_BLNK,
        v_BLNV=v_BLNV,
        a_log=a_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
    )
    _validate_kda_shapes(q_BLNK, k_BLNK, v_BLNV, g_BLNK, beta_BLN, initial_state)
    if q_BLNK.shape[2] != v_BLNV.shape[2]:
        repeat = v_BLNV.shape[2] // q_BLNK.shape[2]
        q_BLNK = q_BLNK.repeat_interleave(repeat, dim=2)
        k_BLNK = k_BLNK.repeat_interleave(repeat, dim=2)

    B, L, N, K = q_BLNK.shape
    V = v_BLNV.shape[-1]
    dtype = q_BLNK.dtype
    q_BLNK = _l2norm(q_BLNK.float(), dim=-1) * (K**-0.5)
    k_BLNK = _l2norm(k_BLNK.float(), dim=-1)
    v_BLNV = v_BLNV.float()
    g_BLNK = g_BLNK.float()
    beta_BLN = beta_BLN.float()
    if use_beta_sigmoid_in_kernel:
        beta_BLN = beta_BLN.sigmoid()
    if initial_state is None:
        state_BNKV = torch.zeros(
            B,
            N,
            K,
            V,
            dtype=torch.float32,
            device=q_BLNK.device,
        )
    else:
        state_BNKV = initial_state.float().clone()

    out_BLNV = torch.empty(B, L, N, V, dtype=torch.float32, device=q_BLNK.device)
    for seq_idx in range(L):
        q_BNK = q_BLNK[:, seq_idx]
        k_BNK = k_BLNK[:, seq_idx]
        v_BNV = v_BLNV[:, seq_idx]
        decay_BNK1 = g_BLNK[:, seq_idx].exp().unsqueeze(-1)
        beta_BN1 = beta_BLN[:, seq_idx].unsqueeze(-1)

        state_BNKV = state_BNKV * decay_BNK1
        kv_mem_BNV = torch.einsum("bnkv,bnk->bnv", state_BNKV, k_BNK)
        delta_BNV = (v_BNV - kv_mem_BNV) * beta_BN1
        state_BNKV = state_BNKV + torch.einsum("bnk,bnv->bnkv", k_BNK, delta_BNV)
        out_BLNV[:, seq_idx] = torch.einsum("bnkv,bnk->bnv", state_BNKV, q_BNK)

    return out_BLNV.to(dtype), state_BNKV


def _prepare_kda_gate(
    g: torch.Tensor,
    *,
    q_BLNK: torch.Tensor,
    v_BLNV: torch.Tensor,
    a_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
) -> torch.Tensor:
    if not use_gate_in_kernel:
        if g.ndim != 3:
            raise ValueError("precomputed g must have shape (B, L, N)")
        return g.unsqueeze(-1).expand(*g.shape, q_BLNK.shape[-1])

    if g.ndim != 4:
        raise ValueError("raw KDA gate must have shape (B, L, N, K)")
    if a_log is None:
        raise ValueError("a_log is required when use_gate_in_kernel=True")
    if dt_bias is None:
        raise ValueError("dt_bias is required when use_gate_in_kernel=True")
    value_heads = v_BLNV.shape[2]
    key_dim = q_BLNK.shape[-1]
    a = torch.exp(a_log.float()).view(1, 1, value_heads, 1)
    dt = dt_bias.float().view(1, 1, value_heads, key_dim)
    if lower_bound is None:
        return -a * F.softplus(g.float() + dt)
    return lower_bound * torch.sigmoid(a * (g.float() + dt))


def _should_use_fla_backend(enabled: bool, hidden_states: torch.Tensor) -> bool:
    return enabled and hidden_states.is_cuda


def _load_fla_chunk_kda():
    try:
        from fla.ops.kda import chunk_kda
    except ImportError as exc:
        raise RuntimeError(
            "Mini-K3 FLA KDA backend requested, but fla-core is not importable"
        ) from exc
    return chunk_kda


def _load_fla_causal_conv1d():
    try:
        from fla.modules.conv.causal_conv1d import causal_conv1d
    except ImportError as exc:
        raise RuntimeError(
            "Mini-K3 FLA causal convolution requested, but fla-core is not importable"
        ) from exc
    return causal_conv1d


def _load_fla_rms_norm_gated():
    try:
        from fla.modules import FusedRMSNormGated
    except ImportError as exc:
        raise RuntimeError(
            "Mini-K3 FLA gated RMSNorm requested, but fla-core is not importable"
        ) from exc
    return FusedRMSNormGated


def initialize_dt_bias_(
    dt_bias: torch.Tensor,
    *,
    dt_min: float = 1.0e-3,
    dt_max: float = 1.0e-1,
    floor: float = 1.0e-4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Initialize a KDA timestep bias in place.

    Mini-K3's first-party scratch-training patch samples timesteps uniformly in
    log-space over ``[dt_min, dt_max]``, clamps by ``floor``, then stores the
    inverse softplus. Applying ``softplus(dt_bias)`` recovers the sampled
    positive timestep.
    """

    if dt_min <= 0:
        raise ValueError("dt_min must be > 0")
    if dt_max <= dt_min:
        raise ValueError("dt_max must be > dt_min")
    if floor < 0:
        raise ValueError("floor must be >= 0")
    if not dt_bias.is_floating_point():
        raise ValueError("dt_bias must be a floating-point tensor")

    device = dt_bias.device
    sample = torch.rand(
        dt_bias.shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    log_min = torch.log(torch.tensor(dt_min, device=device, dtype=torch.float32))
    log_max = torch.log(torch.tensor(dt_max, device=device, dtype=torch.float32))
    dt = torch.exp(sample * (log_max - log_min) + log_min).clamp(min=floor)
    with torch.no_grad():
        dt_bias.copy_((dt + torch.log(-torch.expm1(-dt))).to(dt_bias.dtype))
    return dt_bias


def _l2norm(x: torch.Tensor, *, dim: int, eps: float = 1.0e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _validate_kda_shapes(
    q_BLNK: torch.Tensor,
    k_BLNK: torch.Tensor,
    v_BLNV: torch.Tensor,
    g_BLNK: torch.Tensor,
    beta_BLN: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> None:
    if q_BLNK.ndim != 4 or k_BLNK.ndim != 4 or v_BLNV.ndim != 4:
        raise ValueError("q, k, and v must have shapes (B, L, N, D)")
    if g_BLNK.ndim != 4:
        raise ValueError("g must have shape (B, L, N, K)")
    if beta_BLN.ndim != 3:
        raise ValueError("beta must have shape (B, L, N)")
    if q_BLNK.shape != k_BLNK.shape:
        raise ValueError("q and k must have identical shapes")
    if q_BLNK.shape[:2] != v_BLNV.shape[:2]:
        raise ValueError("q, k, and v must share batch and sequence dimensions")
    if v_BLNV.shape[2] % q_BLNK.shape[2] != 0:
        raise ValueError("value heads must be divisible by key heads")
    if g_BLNK.shape[:3] != v_BLNV.shape[:3] or g_BLNK.shape[-1] != q_BLNK.shape[-1]:
        raise ValueError("g dimensions must match value heads and key dimension")
    if beta_BLN.shape != v_BLNV.shape[:3]:
        raise ValueError("beta head dimensions must match value heads")
    if initial_state is not None:
        expected = (
            q_BLNK.shape[0],
            v_BLNV.shape[2],
            q_BLNK.shape[-1],
            v_BLNV.shape[-1],
        )
        if initial_state.shape != expected:
            raise ValueError(
                "initial_state must have shape "
                f"(B, value_heads, key_dim, value_dim), got {tuple(initial_state.shape)}"
            )
