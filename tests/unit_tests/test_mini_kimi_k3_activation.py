# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch

from torchtitan.experiments.mini_kimi_k3.activation import SituAndMul


def test_situ_and_mul_matches_first_party_formula_and_preserves_dtype():
    x = torch.tensor(
        [
            [
                [-8.0, -1.5, 2.0, 5.0],
                [0.0, 1.25, -3.0, 7.0],
            ]
        ],
        dtype=torch.bfloat16,
    )

    actual = SituAndMul(beta=4.0)(x)

    gate = x[..., :2].to(torch.float32)
    up = x[..., 2:].to(torch.float32)
    expected = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * up).to(
        torch.bfloat16
    )
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.bfloat16


def test_situ_and_mul_applies_optional_linear_beta_to_up_half():
    x = torch.tensor([[[2.0, -4.0, 9.0, -16.0]]], dtype=torch.bfloat16)

    actual = SituAndMul(beta=4.0, linear_beta=25.0)(x)

    gate = x[..., :2].to(torch.float32)
    up = 25.0 * torch.tanh(x[..., 2:].to(torch.float32) / 25.0)
    expected = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * up).to(
        torch.bfloat16
    )
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.bfloat16


def test_situ_and_mul_rejects_odd_last_dimension():
    with pytest.raises(ValueError, match="last dimension must split evenly"):
        SituAndMul()(torch.zeros(2, 3, 5))
