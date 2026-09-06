# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch

from torchtitan.experiments.falcon.falcon import (
    falcon_masked_parallel_forward,
    falcon_recurrent_forward,
)


def test_falcon1a_first_step_is_noop_and_second_step_writes_shifted_key():
    # Legend: B=1, L=2, N=1, K=2, V=2. QK features are already phi(.).
    q_BLNK = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]).view(1, 2, 1, 2)
    k_BLNK = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]).view(1, 2, 1, 2)
    v_BLNV = torch.tensor([[[9.0, 9.0], [2.0, 3.0]]]).view(1, 2, 1, 2)
    beta_BLN = torch.ones(1, 2, 1)
    lambda_BLN = torch.zeros(1, 2, 1)

    out_BLNV, state_BNKV = falcon_recurrent_forward(
        q_BLNK,
        k_BLNK,
        v_BLNV,
        beta_BLN,
        lambda_BLN,
        variant="falcon1a",
        normalize_qk=False,
        eps=0.0,
    )

    torch.testing.assert_close(out_BLNV[:, 0], torch.zeros(1, 1, 2))
    torch.testing.assert_close(out_BLNV[:, 1], torch.tensor([[[2.0, 3.0]]]))
    torch.testing.assert_close(
        state_BNKV,
        torch.tensor([[[[2.0, 3.0], [0.0, 0.0]]]]),
    )


def test_falcon1_and_falcon1a_diverge_on_the_second_write():
    q_BLNK = torch.ones(1, 3, 1, 2)
    k_BLNK = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]).view(1, 3, 1, 2)
    v_BLNV = torch.tensor([[[0.0, 0.0], [2.0, 3.0], [4.0, 5.0]]]).view(1, 3, 1, 2)
    beta_BLN = torch.ones(1, 3, 1)
    lambda_BLN = torch.zeros(1, 3, 1)

    out_1a, state_1a = falcon_recurrent_forward(
        q_BLNK,
        k_BLNK,
        v_BLNV,
        beta_BLN,
        lambda_BLN,
        variant="falcon1a",
        normalize_qk=False,
        eps=0.0,
    )
    out_1, state_1 = falcon_recurrent_forward(
        q_BLNK,
        k_BLNK,
        v_BLNV,
        beta_BLN,
        lambda_BLN,
        variant="falcon1",
        normalize_qk=False,
        eps=0.0,
    )

    torch.testing.assert_close(out_1a[:, :2], out_1[:, :2])
    torch.testing.assert_close(
        state_1a,
        torch.tensor([[[[6.0, 8.0], [0.0, 0.0]]]]),
    )
    torch.testing.assert_close(
        state_1,
        torch.tensor([[[[4.0, 5.0], [0.0, 0.0]]]]),
    )
    assert not torch.allclose(out_1a[:, 2], out_1[:, 2])


def test_falcon_recurrent_forward_rejects_unknown_variant():
    ones = torch.ones(1, 1, 1, 2)
    with pytest.raises(ValueError, match="variant"):
        falcon_recurrent_forward(
            ones,
            ones,
            ones,
            torch.ones(1, 1, 1),
            torch.zeros(1, 1, 1),
            variant="falcon2",
        )


def test_delayed_alignment_first_write_uses_k0_with_v1():
    # Delayed default: t=0 is a no-op write, the first real write pairs
    # phi(k_0) with v_1. Legend: B=1, L=2, N=1, K=2, V=2.
    q_BLNK = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]).view(1, 2, 1, 2)
    k_BLNK = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]).view(1, 2, 1, 2)
    v_BLNV = torch.tensor([[[9.0, 9.0], [2.0, 3.0]]]).view(1, 2, 1, 2)
    beta_BLN = torch.ones(1, 2, 1)
    lambda_BLN = torch.zeros(1, 2, 1)

    out_BLNV, state_BNKV = falcon_recurrent_forward(
        q_BLNK,
        k_BLNK,
        v_BLNV,
        beta_BLN,
        lambda_BLN,
        variant="falcon1a",
        normalize_qk=False,
        eps=0.0,
        alignment="delayed",
    )

    # t=0 read sees an empty state; t=1 read sees k_0 -> v_1 association.
    torch.testing.assert_close(out_BLNV[:, 0], torch.zeros(1, 1, 2))
    torch.testing.assert_close(out_BLNV[:, 1], torch.tensor([[[2.0, 3.0]]]))
    torch.testing.assert_close(
        state_BNKV,
        torch.tensor([[[[2.0, 3.0], [0.0, 0.0]]]]),
    )


def test_same_step_alignment_first_write_uses_k0_with_v0():
    # Same-step ablation: at t=0 the write pairs phi(k_0) with v_0 and the
    # read at t=0 observes it (read-after-write). eta is computed normally.
    q_BLNK = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]).view(1, 2, 1, 2)
    k_BLNK = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]).view(1, 2, 1, 2)
    v_BLNV = torch.tensor([[[9.0, 9.0], [2.0, 3.0]]]).view(1, 2, 1, 2)
    beta_BLN = torch.ones(1, 2, 1)
    lambda_BLN = torch.zeros(1, 2, 1)

    out_BLNV, state_BNKV = falcon_recurrent_forward(
        q_BLNK,
        k_BLNK,
        v_BLNV,
        beta_BLN,
        lambda_BLN,
        variant="falcon1a",
        normalize_qk=False,
        eps=0.0,
        alignment="same_step",
    )

    # t=0 write is phi(k_0)=e_0 -> v_0; q_0=e_0 reads v_0 back immediately.
    torch.testing.assert_close(out_BLNV[:, 0], torch.tensor([[[9.0, 9.0]]]))
    # t=1 write is phi(k_1)=e_1 -> v_1; state holds both associations.
    torch.testing.assert_close(out_BLNV[:, 1], torch.tensor([[[9.0, 9.0]]]))
    torch.testing.assert_close(
        state_BNKV,
        torch.tensor([[[[9.0, 9.0], [2.0, 3.0]]]]),
    )


def test_falcon_recurrent_forward_rejects_unknown_alignment():
    ones = torch.ones(1, 1, 1, 2)
    with pytest.raises(ValueError, match="alignment"):
        falcon_recurrent_forward(
            ones,
            ones,
            ones,
            torch.ones(1, 1, 1),
            torch.zeros(1, 1, 1),
            alignment="lagged",
        )


def _random_falcon_inputs(
    *, batch: int, seq_len: int, heads: int, key_dim: int, value_dim: int, seed: int
):
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    q_BLNK = randn(batch, seq_len, heads, key_dim)
    k_BLNK = randn(batch, seq_len, heads, key_dim)
    v_BLNV = randn(batch, seq_len, heads, value_dim)
    # beta is a plasticity gain in (0, 2); lambda is a nonnegative ridge.
    beta_BLN = 2.0 * torch.sigmoid(randn(batch, seq_len, heads))
    lambda_BLN = torch.nn.functional.softplus(randn(batch, seq_len, heads))
    return q_BLNK, k_BLNK, v_BLNV, beta_BLN, lambda_BLN


@pytest.mark.parametrize("variant", ["falcon1", "falcon1a"])
@pytest.mark.parametrize("alignment", ["delayed", "same_step"])
@pytest.mark.parametrize("normalize_qk", [True, False])
def test_masked_parallel_matches_recurrent_on_random_inputs(
    variant, alignment, normalize_qk
):
    # The masked-parallel view must reproduce the recurrent oracle bitwise-close
    # in fp32 for every variant x alignment. This identity gate replaces the
    # Mercor logprob-diff < 0.03 train-inference check; there is no separate
    # train-inference stack for this mixer.
    q_BLNK, k_BLNK, v_BLNV, beta_BLN, lambda_BLN = _random_falcon_inputs(
        batch=2, seq_len=7, heads=3, key_dim=4, value_dim=5, seed=123
    )

    kwargs = dict(
        variant=variant,
        alignment=alignment,
        normalize_qk=normalize_qk,
    )
    ref_out, ref_state = falcon_recurrent_forward(
        q_BLNK, k_BLNK, v_BLNV, beta_BLN, lambda_BLN, **kwargs
    )
    par_out, par_state = falcon_masked_parallel_forward(
        q_BLNK, k_BLNK, v_BLNV, beta_BLN, lambda_BLN, **kwargs
    )

    torch.testing.assert_close(par_out, ref_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(par_state, ref_state, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("variant", ["falcon1", "falcon1a"])
@pytest.mark.parametrize("alignment", ["delayed", "same_step"])
def test_masked_parallel_matches_recurrent_on_worked_example(variant, alignment):
    # The same L=3 orthonormal-key worked example used by the recurrent tests,
    # verified against the parallel view for both alignments.
    q_BLNK = torch.ones(1, 3, 1, 2)
    k_BLNK = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]).view(1, 3, 1, 2)
    v_BLNV = torch.tensor([[[0.0, 0.0], [2.0, 3.0], [4.0, 5.0]]]).view(1, 3, 1, 2)
    beta_BLN = torch.ones(1, 3, 1)
    lambda_BLN = torch.zeros(1, 3, 1)

    kwargs = dict(
        variant=variant,
        alignment=alignment,
        normalize_qk=False,
        eps=0.0,
    )
    ref_out, ref_state = falcon_recurrent_forward(
        q_BLNK, k_BLNK, v_BLNV, beta_BLN, lambda_BLN, **kwargs
    )
    par_out, par_state = falcon_masked_parallel_forward(
        q_BLNK, k_BLNK, v_BLNV, beta_BLN, lambda_BLN, **kwargs
    )

    torch.testing.assert_close(par_out, ref_out, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(par_state, ref_state, rtol=1e-6, atol=1e-6)


def test_masked_parallel_rejects_unknown_variant():
    ones = torch.ones(1, 1, 1, 2)
    with pytest.raises(ValueError, match="variant"):
        falcon_masked_parallel_forward(
            ones,
            ones,
            ones,
            torch.ones(1, 1, 1),
            torch.zeros(1, 1, 1),
            variant="falcon2",
        )


def test_masked_parallel_rejects_unknown_alignment():
    ones = torch.ones(1, 1, 1, 2)
    with pytest.raises(ValueError, match="alignment"):
        falcon_masked_parallel_forward(
            ones,
            ones,
            ones,
            torch.ones(1, 1, 1),
            torch.zeros(1, 1, 1),
            alignment="lagged",
        )
