# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from torchtitan.experiments.falcon.overfit import run_overfit


@pytest.mark.parametrize("variant", ["falcon1a", "falcon1"])
def test_falcon_overfits_the_fixed_sequence_bank(variant: str):
    report = run_overfit(variant=variant, steps=80, lr=3.0e-2, seed=0)

    assert report["status"] == "pass"
    assert report["final_loss"] < 0.5 * report["initial_loss"]
    assert report["final_loss"] < 1.0
    assert report["num_sequences"] == 8
    assert report["steps"] == 80


def test_overfit_gate_fails_when_learning_rate_is_zero():
    report = run_overfit(variant="falcon1a", steps=16, lr=0.0, seed=0)

    assert report["status"] == "fail"
    assert report["final_loss"] >= 0.5 * report["initial_loss"]
