# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression coverage for DeepSeek-V4 mHC post mixing."""

import torch

from torchtitan.models.deepseek_v4.mhc import HcPost


def _reference_hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Official formula after folding the batch and sequence axes into T."""
    return post.unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(
        comb.transpose(-1, -2), residual
    )


def test_hc_post_matches_official_formula_after_flattening_batch_sequence():
    torch.manual_seed(0)
    tokens, hc_mult, dim = 7, 4, 5
    x = torch.randn(tokens, dim)
    residual = torch.randn(tokens, hc_mult, dim)
    post = torch.randn(tokens, hc_mult)

    comb = torch.rand(tokens, hc_mult, hc_mult)
    for _ in range(8):
        comb = comb / comb.sum(dim=-1, keepdim=True)
        comb = comb / comb.sum(dim=-2, keepdim=True)

    actual = HcPost.Config().build()(x, residual, post, comb)
    expected = _reference_hc_post(x, residual, post, comb)

    torch.testing.assert_close(actual, expected)


def test_hc_post_mixed_dtypes_and_batched_layout():
    """bf16 residual with fp32 weights, and an extra leading batch axis."""
    torch.manual_seed(1)
    batch, seq, hc_mult, dim = 2, 3, 4, 8
    x = torch.randn(batch, seq, dim, dtype=torch.bfloat16)
    residual = torch.randn(batch, seq, hc_mult, dim, dtype=torch.bfloat16)
    post = torch.rand(batch, seq, hc_mult)
    comb = torch.rand(batch, seq, hc_mult, hc_mult)

    actual = HcPost.Config().build()(x, residual, post, comb)
    expected = _reference_hc_post(x.float(), residual.float(), post, comb).to(
        torch.bfloat16
    )

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
