# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util

import json
import math

import pytest

from experiments.modded_nanogpt_b200 import cpu_smoke


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="cpu smoke execution requires torch in the active environment",
)


def test_cpu_smoke_writes_end_to_end_artifacts(tmp_path):
    result = cpu_smoke.run_cpu_smoke(
        result_dir=tmp_path,
        run_id="cpu_fixture",
        steps=2,
        batch_size=2,
        seq_len=8,
        vocab_size=64,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        mlp_dim=32,
        seed=123,
    )

    assert result["ok"] is True
    assert result["device"] == "cpu"
    assert math.isfinite(result["initial_train_loss"])
    assert math.isfinite(result["final_train_loss"])
    assert math.isfinite(result["validation_loss"])
    assert result["validation_loss"] > 0

    summary = json.loads((tmp_path / "summary.json").read_text())
    attempt = json.loads((tmp_path / "attempt.json").read_text())
    log = (tmp_path / "run.log").read_text()

    assert summary["classification"]["mode"] == "cpu"
    assert summary["classification"]["claim_eligible"] is False
    assert summary["ok"] is True
    assert summary["final_validation_reached"] is True
    assert summary["final_metrics"]["val_loss"] == result["validation_loss"]
    assert attempt["command"] == ["cpu_smoke"]
    assert "cpu_smoke step=1" in log
    assert "cpu_smoke validation_loss=" in log
