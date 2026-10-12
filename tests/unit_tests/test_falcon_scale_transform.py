# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from torchtitan.experiments.falcon.falcon import falcon_recurrent_forward
from torchtitan.experiments.falcon.model import (
    FalconConfig,
    FalconForCausalLM,
    FalconMixer,
)


def _leaf_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def _m2_config() -> FalconConfig:
    return FalconConfig(
        vocab_size=37,
        hidden_size=32,
        num_hidden_layers=2,
        num_heads=1,
        head_dim=32,
        intermediate_size=48,
        seq_len=6,
        variant="falcon1a",
        alignment="delayed",
        mixer="falcon",
        phi="l2",
        qk_norm_eps=3.0e-4,
        nlms_denom_eps=3.0e-4,
        scale_compensation="none",
    )


def _m4_config() -> FalconConfig:
    config = _m2_config()
    config.phi = "rms"
    config.scale_compensation = "rms_to_l2"
    return config


def _copy_state(source: torch.nn.Module, target: torch.nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def _assert_named_grads_close(
    left: torch.nn.Module,
    right: torch.nn.Module,
    *,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-5,
) -> None:
    for (left_name, left_param), (right_name, right_param) in zip(
        left.named_parameters(), right.named_parameters(), strict=True
    ):
        assert left_name == right_name
        torch.testing.assert_close(
            left_param.grad,
            right_param.grad,
            rtol=rtol,
            atol=atol,
            msg=f"gradient mismatch for {left_name}",
        )


def test_rms_to_l2_kernel_scale_transform_preserves_outputs_state_and_gradients():
    # Break caught: using one shared epsilon for both QK normalization and the
    # NLMS denominator breaks the RMS/L2 scale transform at head_dim=32.
    generator = torch.Generator().manual_seed(20260908)
    batch, seq_len, heads, head_dim, value_dim = 2, 5, 3, 32, 7
    qk_base_eps = 3.0e-4
    nlms_base_eps = 7.0e-4

    q_base = torch.randn(
        batch, seq_len, heads, head_dim, generator=generator, dtype=torch.float32
    )
    k_base = torch.randn(
        batch, seq_len, heads, head_dim, generator=generator, dtype=torch.float32
    )
    v_base = torch.randn(
        batch, seq_len, heads, value_dim, generator=generator, dtype=torch.float32
    )
    beta_base = torch.rand(
        batch, seq_len, heads, generator=generator, dtype=torch.float32
    ).add_(0.25)
    lambda_base_value = torch.rand(
        batch, seq_len, heads, generator=generator, dtype=torch.float32
    ).add_(0.1)
    cotangent = torch.randn(
        batch, seq_len, heads, value_dim, generator=generator, dtype=torch.float32
    )

    q_l2 = _leaf_clone(q_base)
    k_l2 = _leaf_clone(k_base)
    v_l2 = _leaf_clone(v_base)
    beta_l2 = _leaf_clone(beta_base)
    lambda_l2 = _leaf_clone(lambda_base_value)
    out_l2, state_l2 = falcon_recurrent_forward(
        q_l2,
        k_l2,
        v_l2,
        beta_l2,
        lambda_l2,
        variant="falcon1a",
        alignment="delayed",
        phi="l2",
        qk_norm_eps=qk_base_eps,
        nlms_denom_eps=nlms_base_eps,
    )
    (out_l2 * cotangent).sum().backward()

    q_rms = _leaf_clone(q_base)
    k_rms = _leaf_clone(k_base)
    v_rms = _leaf_clone(v_base)
    beta_rms = _leaf_clone(beta_base)
    lambda_rms_base = _leaf_clone(lambda_base_value)
    lambda_rms = head_dim * lambda_rms_base
    out_rms, state_rms = falcon_recurrent_forward(
        q_rms,
        k_rms,
        v_rms,
        beta_rms,
        lambda_rms,
        variant="falcon1a",
        alignment="delayed",
        phi="rms",
        qk_norm_eps=qk_base_eps / head_dim,
        nlms_denom_eps=nlms_base_eps * head_dim,
    )
    (out_rms * cotangent).sum().backward()

    torch.testing.assert_close(out_rms, out_l2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        state_l2,
        math.sqrt(head_dim) * state_rms,
        rtol=1e-5,
        atol=1e-5,
    )
    for actual, expected in (
        (q_rms.grad, q_l2.grad),
        (k_rms.grad, k_l2.grad),
        (v_rms.grad, v_l2.grad),
        (beta_rms.grad, beta_l2.grad),
        (lambda_rms_base.grad, lambda_l2.grad),
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_rms_to_l2_mixer_preserves_outputs_input_gradients_and_parameter_gradients():
    # Break caught: M4 compensation must be a high-level mixer behavior, not
    # only a hand-built kernel theorem.
    torch.manual_seed(11)
    m2 = FalconMixer(_m2_config())
    m4 = FalconMixer(_m4_config())
    _copy_state(m2, m4)

    x_base = torch.randn(3, 6, 32)
    cotangent = torch.randn(3, 6, 32)
    x_m2 = _leaf_clone(x_base)
    x_m4 = _leaf_clone(x_base)

    out_m2 = m2(x_m2)
    out_m4 = m4(x_m4)
    torch.testing.assert_close(out_m4, out_m2, rtol=1e-5, atol=1e-5)

    (out_m2 * cotangent).sum().backward()
    (out_m4 * cotangent).sum().backward()
    torch.testing.assert_close(x_m4.grad, x_m2.grad, rtol=1e-5, atol=1e-5)
    _assert_named_grads_close(m4, m2)


def test_rms_to_l2_model_preserves_logits_loss_and_parameter_gradients():
    # Break caught: the full Falcon LM must preserve the M2 trajectory when M4
    # resolves compensation through the model config.
    torch.manual_seed(17)
    m2 = FalconForCausalLM(_m2_config())
    m4 = FalconForCausalLM(_m4_config())
    _copy_state(m2, m4)

    tokens = torch.tensor(
        [
            [1, 4, 7, 10, 13, 16],
            [2, 5, 8, 11, 14, 17],
            [3, 6, 9, 12, 15, 18],
        ],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [
            [4, 7, 10, 13, 16, 19],
            [5, 8, 11, 14, 17, 20],
            [6, 9, 12, 15, 18, 21],
        ],
        dtype=torch.long,
    )

    logits_m2 = m2(tokens)
    logits_m4 = m4(tokens)
    loss_m2 = F.cross_entropy(logits_m2.flatten(0, 1), labels.flatten())
    loss_m4 = F.cross_entropy(logits_m4.flatten(0, 1), labels.flatten())

    torch.testing.assert_close(logits_m4, logits_m2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(loss_m4, loss_m2, rtol=1e-5, atol=1e-5)
    loss_m2.backward()
    loss_m4.backward()
    _assert_named_grads_close(m4, m2)


def test_rms_to_l2_optimizer_trajectory_stays_identical_for_fixed_batches():
    # Break caught: even if a single forward/backward matches, optimizer state
    # must not diverge across several deterministic fp32 updates.
    torch.manual_seed(23)
    m2 = FalconForCausalLM(_m2_config())
    m4 = FalconForCausalLM(_m4_config())
    _copy_state(m2, m4)
    opt_m2 = torch.optim.AdamW(m2.parameters(), lr=2.0e-3, weight_decay=0.01)
    opt_m4 = torch.optim.AdamW(m4.parameters(), lr=2.0e-3, weight_decay=0.01)
    batches = [
        torch.tensor(
            [
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 3, 2, 1],
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [
                [7, 8, 9, 10, 11, 12],
                [12, 11, 10, 9, 8, 7],
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [
                [13, 14, 15, 16, 17, 18],
                [18, 17, 16, 15, 14, 13],
            ],
            dtype=torch.long,
        ),
    ]

    for tokens in batches:
        labels = (tokens + 1).remainder(m2.config.vocab_size)
        opt_m2.zero_grad(set_to_none=True)
        opt_m4.zero_grad(set_to_none=True)
        loss_m2 = F.cross_entropy(m2(tokens).flatten(0, 1), labels.flatten())
        loss_m4 = F.cross_entropy(m4(tokens).flatten(0, 1), labels.flatten())
        torch.testing.assert_close(loss_m4, loss_m2, rtol=1e-5, atol=1e-5)
        loss_m2.backward()
        loss_m4.backward()
        _assert_named_grads_close(m4, m2)
        opt_m2.step()
        opt_m4.step()
        for (name_m2, param_m2), (name_m4, param_m4) in zip(
            m2.named_parameters(), m4.named_parameters(), strict=True
        ):
            assert name_m2 == name_m4
            torch.testing.assert_close(
                param_m4,
                param_m2,
                rtol=1e-5,
                atol=1e-5,
                msg=f"parameter mismatch for {name_m2}",
            )
