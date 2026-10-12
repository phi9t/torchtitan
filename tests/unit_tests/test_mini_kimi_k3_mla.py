# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from torchtitan.experiments.mini_kimi_k3.mla import (
    apply_rotary_pos_emb,
    MiniK3MLAAttention,
)


def test_mini_k3_mla_preserves_shape_with_asymmetric_q_and_value_dims():
    attn = MiniK3MLAAttention(
        hidden_size=8,
        num_heads=2,
        q_lora_rank=4,
        kv_lora_rank=3,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=3,
        use_output_gate=True,
    )
    x = torch.randn(2, 5, 8)

    out = attn(x)

    assert out.shape == x.shape


def test_mini_k3_mla_output_gate_changes_result_when_gate_projection_changes():
    attn = MiniK3MLAAttention(
        hidden_size=4,
        num_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        use_output_gate=True,
    )
    x = torch.randn(1, 3, 4)

    with torch.no_grad():
        attn.g_proj.weight.fill_(8.0)
    high_gate = attn(x)
    with torch.no_grad():
        attn.g_proj.weight.fill_(-8.0)
    low_gate = attn(x)

    assert not torch.allclose(high_gate, low_gate)


def test_mini_k3_mla_supports_non_lora_query_projection():
    attn = MiniK3MLAAttention(
        hidden_size=4,
        num_heads=1,
        q_lora_rank=None,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
    )
    x = torch.randn(1, 2, 4, requires_grad=True)

    out = attn(x)
    out.sum().backward()

    assert out.shape == x.shape
    assert x.grad is not None
    assert attn.q_proj is not None


def test_mini_k3_mla_causal_mask_prevents_future_token_influence():
    attn = MiniK3MLAAttention(
        hidden_size=4,
        num_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
    )
    prefix = torch.randn(1, 2, 4)
    x_a = torch.cat([prefix, torch.ones(1, 1, 4)], dim=1)
    x_b = torch.cat([prefix, torch.full((1, 1, 4), -10.0)], dim=1)

    out_a = attn(x_a)
    out_b = attn(x_b)

    torch.testing.assert_close(out_a[:, :2], out_b[:, :2])


def test_mini_k3_mla_forward_ignores_positions_like_first_party_r1():
    attn = MiniK3MLAAttention(
        hidden_size=4,
        num_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
    )
    x = torch.randn(1, 3, 4)

    out_a = attn(x, positions=torch.tensor([[0, 1, 2]]))
    out_b = attn(x, positions=torch.tensor([[0, 2, 4]]))

    torch.testing.assert_close(out_a, out_b)


def test_mini_k3_mla_low_rank_norms_use_first_party_default_eps():
    attn = MiniK3MLAAttention(
        hidden_size=4,
        num_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        rms_norm_eps=1.0e-5,
    )

    assert attn.q_a_layernorm is not None
    assert attn.q_a_layernorm.eps == 1.0e-6
    assert attn.kv_a_layernorm.eps == 1.0e-6


def test_apply_rotary_pos_emb_rotates_complex_pairs_by_position():
    x = torch.tensor(
        [
            [
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0],
                ]
            ]
        ]
    )
    positions = torch.tensor([[0, 1]])

    out = apply_rotary_pos_emb(x, positions=positions, rope_theta=1.0)

    torch.testing.assert_close(out[:, :, 0], x[:, :, 0])
    expected = torch.tensor(
        [
            [
                [
                    [
                        torch.cos(torch.tensor(1.0)),
                        torch.sin(torch.tensor(1.0)),
                        -torch.sin(torch.tensor(1.0)),
                        torch.cos(torch.tensor(1.0)),
                    ]
                ]
            ]
        ]
    )
    torch.testing.assert_close(out[:, :, 1:], expected)


def test_mini_k3_mla_applies_rope_to_rotary_query_and_key_parts():
    attn = MiniK3MLAAttention(
        hidden_size=4,
        num_heads=1,
        q_lora_rank=None,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        rope_theta=1.0,
        use_output_gate=False,
    )
    x = torch.ones(1, 2, 4)
    positions = torch.tensor([[0, 1]])

    with torch.no_grad():
        for param in attn.parameters():
            param.fill_(0.0)
        attn.q_proj.weight[2, 0] = 1.0
        attn.q_proj.weight[3, 1] = 1.0
        attn.kv_a_proj_with_mqa.weight[2, 0] = 1.0
        attn.kv_a_proj_with_mqa.weight[3, 1] = 1.0

    q_rot, k_rot = attn.rotary_projections_for_test(x, positions=positions)

    torch.testing.assert_close(q_rot[:, :, 0], torch.tensor([[[1.0, 1.0]]]))
    torch.testing.assert_close(k_rot[:, :, 0], torch.tensor([[[1.0, 1.0]]]))
    assert not torch.allclose(q_rot[:, :, 1], torch.tensor([[[1.0, 1.0]]]))
    assert not torch.allclose(k_rot[:, :, 1], torch.tensor([[[1.0, 1.0]]]))


def test_mini_k3_mla_rejects_invalid_head_dimensions():
    try:
        MiniK3MLAAttention(
            hidden_size=4,
            num_heads=0,
            q_lora_rank=2,
            kv_lora_rank=2,
            qk_nope_head_dim=2,
            qk_rope_head_dim=1,
            v_head_dim=2,
        )
    except ValueError as exc:
        assert "num_heads" in str(exc)
    else:
        raise AssertionError("expected invalid num_heads to raise")
