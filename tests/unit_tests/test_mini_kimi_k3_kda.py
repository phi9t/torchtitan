# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch

import torchtitan.experiments.mini_kimi_k3.kda as kda_module
from torchtitan.experiments.mini_kimi_k3.kda import (
    initialize_dt_bias_,
    kda_reference_forward,
    MiniK3DeltaAttention,
)


def test_initialize_dt_bias_samples_positive_timesteps_in_log_range():
    dt_bias = torch.empty(2048, dtype=torch.float32)
    generator = torch.Generator().manual_seed(1234)

    initialize_dt_bias_(dt_bias, generator=generator)

    dt = torch.nn.functional.softplus(dt_bias)
    assert torch.all(dt >= 1.0e-4)
    assert torch.all(dt >= 1.0e-3)
    assert torch.all(dt <= 1.0e-1)
    assert dt.std() > 0


def test_initialize_dt_bias_preserves_parameter_dtype():
    dt_bias = torch.empty(32, dtype=torch.bfloat16)

    initialize_dt_bias_(dt_bias, generator=torch.Generator().manual_seed(1))

    assert dt_bias.dtype == torch.bfloat16
    dt = torch.nn.functional.softplus(dt_bias.float())
    assert torch.isfinite(dt).all()
    assert torch.all(dt >= 1.0e-4)


def test_initialize_dt_bias_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="dt_min"):
        initialize_dt_bias_(torch.empty(4), dt_min=0.0)

    with pytest.raises(ValueError, match="dt_max"):
        initialize_dt_bias_(torch.empty(4), dt_min=1.0e-1, dt_max=1.0e-2)

    with pytest.raises(ValueError, match="floor"):
        initialize_dt_bias_(torch.empty(4), floor=-1.0)


def test_kda_reference_forward_matches_manual_single_step_recurrence():
    q = torch.tensor([[[[2.0, 0.0]]]])
    k = torch.tensor([[[[3.0, 4.0]]]])
    v = torch.tensor([[[[5.0, 7.0, 11.0]]]])
    g = torch.tensor([[[-0.25]]])
    beta = torch.tensor([[[0.5]]])

    out, final_state = kda_reference_forward(q, k, v, g, beta)

    q_norm = q.float() * torch.rsqrt(
        (q.float() * q.float()).sum(dim=-1, keepdim=True) + 1e-6
    )
    q_norm = q_norm * (q.shape[-1] ** -0.5)
    k_norm = k.float() * torch.rsqrt(
        (k.float() * k.float()).sum(dim=-1, keepdim=True) + 1e-6
    )
    expected_state = torch.einsum("bnk,bnv->bnkv", k_norm[:, 0], v[:, 0].float() * 0.5)
    expected_out = torch.einsum(
        "bnkv,bnk->bnv", expected_state, q_norm[:, 0]
    ).unsqueeze(1)
    torch.testing.assert_close(out, expected_out.to(q.dtype))
    torch.testing.assert_close(final_state, expected_state)


def test_kda_reference_forward_accepts_initial_state_and_returns_final_state():
    q = torch.ones(1, 2, 1, 2)
    k = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    v = torch.tensor([[[[2.0, 0.0]], [[0.0, 4.0]]]])
    g = torch.full((1, 2, 1), -0.5)
    beta = torch.ones(1, 2, 1)
    initial_state = torch.ones(1, 1, 2, 2)

    out_with_state, final_state = kda_reference_forward(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
    )
    out_without_state, _ = kda_reference_forward(q, k, v, g, beta)

    assert out_with_state.shape == (1, 2, 1, 2)
    assert final_state.shape == (1, 1, 2, 2)
    assert not torch.allclose(out_with_state, out_without_state)


def test_kda_reference_forward_uses_feature_gate_and_per_channel_dt_bias():
    q = torch.ones(1, 2, 1, 2)
    k = torch.ones(1, 2, 1, 2)
    v = torch.ones(1, 2, 1, 2)
    beta = torch.ones(1, 2, 1)
    raw_gate_a = torch.full((1, 2, 1, 2), -10.0)
    raw_gate_b = torch.full((1, 2, 1, 2), 10.0)
    a_log = torch.zeros(1)
    dt_bias = torch.zeros(2)

    out_a, _ = kda_reference_forward(
        q,
        k,
        v,
        raw_gate_a,
        beta,
        a_log=a_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=-5.0,
    )
    out_b, _ = kda_reference_forward(
        q,
        k,
        v,
        raw_gate_b,
        beta,
        a_log=a_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=-5.0,
    )

    assert not torch.allclose(out_a, out_b)


def test_kda_reference_forward_expands_key_heads_to_value_heads():
    q = torch.ones(1, 3, 1, 2)
    k = torch.ones(1, 3, 1, 2)
    v = torch.ones(1, 3, 2, 4)
    g = torch.full((1, 3, 2), -0.5)
    beta = torch.ones(1, 3, 2)

    out, final_state = kda_reference_forward(q, k, v, g, beta)

    assert out.shape == (1, 3, 2, 4)
    assert final_state.shape == (1, 2, 2, 4)


def test_kda_reference_forward_rejects_incompatible_shapes():
    q = torch.ones(1, 3, 2, 2)
    k = torch.ones(1, 3, 2, 2)
    v = torch.ones(1, 3, 3, 4)
    g = torch.ones(1, 3, 3)
    beta = torch.ones(1, 3, 3)

    with pytest.raises(ValueError, match="value heads must be divisible"):
        kda_reference_forward(q, k, v, g, beta)


def test_mini_k3_delta_attention_preserves_input_shape_and_returns_state():
    attn = MiniK3DeltaAttention(
        hidden_size=6,
        num_heads=2,
        head_dim=3,
        conv_kernel_size=3,
        gate_lower_bound=-5.0,
        use_full_rank_gate=True,
    )
    x = torch.randn(2, 4, 6)

    out, state = attn(x)

    assert out.shape == x.shape
    assert state.shape == (2, 2, 3, 3)


def test_mini_k3_delta_attention_can_dispatch_to_fla_backend(monkeypatch):
    calls = []
    norm_calls = []
    conv_calls = []

    def fake_chunk_kda(**kwargs):
        calls.append(kwargs)
        q = kwargs["q"]
        v = kwargs["v"]
        return (
            torch.zeros(*q.shape[:3], v.shape[-1], dtype=q.dtype, device=q.device),
            torch.zeros(
                q.shape[0],
                v.shape[2],
                q.shape[-1],
                v.shape[-1],
                dtype=torch.float32,
                device=q.device,
            ),
        )

    monkeypatch.setattr(
        kda_module,
        "_should_use_fla_backend",
        lambda enabled, hidden_states: enabled,
    )
    monkeypatch.setattr(kda_module, "_load_fla_chunk_kda", lambda: fake_chunk_kda)

    def fake_causal_conv1d(**kwargs):
        conv_calls.append(
            {
                "shape": tuple(kwargs["x"].shape),
                "weight_shape": tuple(kwargs["weight"].shape),
                "activation": kwargs["activation"],
                "backend": kwargs["backend"],
            }
        )
        return torch.zeros_like(kwargs["x"]), None

    monkeypatch.setattr(
        kda_module, "_load_fla_causal_conv1d", lambda: fake_causal_conv1d
    )

    class FakeFlaNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps, activation):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.eps = eps
            self.activation = activation

        def forward(self, x, gate):
            norm_calls.append(
                {
                    "shape": tuple(x.shape),
                    "gate_shape": tuple(gate.shape),
                    "eps": self.eps,
                    "activation": self.activation,
                }
            )
            return torch.zeros_like(x)

    monkeypatch.setattr(kda_module, "_load_fla_rms_norm_gated", lambda: FakeFlaNorm)
    attn = MiniK3DeltaAttention(
        hidden_size=4,
        num_heads=1,
        head_dim=4,
        conv_kernel_size=1,
        use_fla_backend=True,
    )
    x = torch.randn(1, 2, 4)

    out, state = attn(x)

    assert out.shape == x.shape
    assert state.shape == (1, 1, 4, 4)
    assert len(calls) == 1
    assert conv_calls == [
        {
            "shape": (1, 2, 4),
            "weight_shape": (4, 1),
            "activation": "silu",
            "backend": "triton",
        },
        {
            "shape": (1, 2, 4),
            "weight_shape": (4, 1),
            "activation": "silu",
            "backend": "triton",
        },
        {
            "shape": (1, 2, 4),
            "weight_shape": (4, 1),
            "activation": "silu",
            "backend": "triton",
        },
    ]
    assert calls[0]["use_gate_in_kernel"] is True
    assert calls[0]["use_beta_sigmoid_in_kernel"] is True
    assert calls[0]["safe_gate"] is True
    assert calls[0]["lower_bound"] == -5.0
    assert calls[0]["output_final_state"] is True
    assert norm_calls == [
        {
            "shape": (1, 2, 1, 4),
            "gate_shape": (1, 2, 1, 4),
            "eps": 1.0e-5,
            "activation": "sigmoid",
        }
    ]


def test_mini_k3_delta_attention_initializes_a_log_like_first_party():
    torch.manual_seed(1234)

    attn = MiniK3DeltaAttention(
        hidden_size=8,
        num_heads=2,
        head_dim=4,
        conv_kernel_size=1,
        use_full_rank_gate=True,
    )

    a = attn.A_log.detach().exp()
    assert torch.all(a >= 1.0)
    assert torch.all(a < 16.0)
    assert a.std() > 0


def test_mini_k3_delta_attention_incremental_matches_chunk_reference_without_conv_context():
    attn = MiniK3DeltaAttention(
        hidden_size=4,
        num_heads=1,
        head_dim=4,
        conv_kernel_size=1,
        use_full_rank_gate=True,
    )
    x = torch.randn(1, 3, 4)

    chunk_out, _ = attn(x)
    state = None
    pieces = []
    for idx in range(x.shape[1]):
        out, state = attn(x[:, idx : idx + 1], initial_state=state)
        pieces.append(out)
    incremental_out = torch.cat(pieces, dim=1)

    torch.testing.assert_close(incremental_out, chunk_out)


def test_mini_k3_delta_attention_is_differentiable():
    attn = MiniK3DeltaAttention(
        hidden_size=4,
        num_heads=1,
        head_dim=4,
        conv_kernel_size=1,
        use_full_rank_gate=False,
    )
    x = torch.randn(1, 2, 4, requires_grad=True)

    out, _ = attn(x)
    out.sum().backward()

    assert x.grad is not None
    assert attn.q_proj.weight.grad is not None
    assert attn.o_proj.weight.grad is not None


def test_mini_k3_delta_attention_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="hidden_size must equal"):
        MiniK3DeltaAttention(hidden_size=8, num_heads=2, head_dim=3)
