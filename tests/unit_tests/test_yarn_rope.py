# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.rope import _yarn_inv_freq


class TestYaRNInvFreq(unittest.TestCase):
    def test_zero_clamped_cutoff_matches_reference_values(self):
        actual = _yarn_inv_freq(
            dim=128,
            base=10_000.0,
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=64,
            truncate=True,
        )

        selected = actual[torch.tensor([0, 1, 8, 16, 17, 63])]
        expected = torch.tensor(
            [
                1.0,
                0.8162987232208252,
                0.1711350381374359,
                0.008235293440520763,
                0.0021649107802659273,
                2.8869549169030506e-06,
            ]
        )
        torch.testing.assert_close(selected, expected, rtol=0, atol=0)

    def test_collapsed_cutoff_produces_finite_frequencies(self):
        actual = _yarn_inv_freq(
            dim=128,
            base=10_000.0,
            rope_factor=40.0,
            beta_fast=1.0,
            beta_slow=1.0,
            original_seq_len=64,
            truncate=False,
        )

        self.assertEqual(actual.shape, (64,))
        self.assertTrue(torch.isfinite(actual).all())

    def test_reversed_cutoff_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "reversed YaRN correction range"):
            _yarn_inv_freq(
                dim=128,
                base=10_000.0,
                rope_factor=40.0,
                beta_fast=1.0,
                beta_slow=32.0,
                original_seq_len=64,
                truncate=True,
            )
