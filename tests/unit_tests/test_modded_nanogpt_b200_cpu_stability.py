# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.modded_nanogpt_b200 import cpu_stability


def _fake_smoke_runner(
    *,
    result_dir: Path,
    run_id: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    mlp_dim: int,
    seed: int,
) -> dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    validation_loss = 4.0 + seed / 1000.0
    result = {
        "ok": True,
        "run_id": run_id,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "mlp_dim": mlp_dim,
        "seed": seed,
        "validation_loss": validation_loss,
        "elapsed_seconds": 0.25,
    }
    (result_dir / "run.log").write_text(
        f"cpu_smoke step=1/{steps} train_loss=4.2\n"
        f"cpu_smoke validation_loss={validation_loss:.6f}\n"
    )
    (result_dir / "summary.json").write_text(
        json.dumps(
            {
                "ok": True,
                "training_launched": True,
                "final_validation_reached": True,
                "included_in_baseline_stats": False,
                "final_metrics": {"val_loss": validation_loss},
            }
        )
        + "\n"
    )
    return result


def test_cpu_stability_runs_multiple_isolated_attempts_with_telemetry(tmp_path):
    summary = cpu_stability.run_cpu_stability(
        result_dir=tmp_path,
        run_id="stability_fixture",
        repeats=3,
        steps=2,
        batch_size=2,
        seq_len=8,
        vocab_size=64,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        mlp_dim=32,
        seed=100,
        smoke_runner=_fake_smoke_runner,
    )

    assert summary["ok"] is True
    assert summary["classification"]["mode"] == "cpu_stability"
    assert summary["classification"]["claim_eligible"] is False
    assert summary["included_in_baseline_stats"] is False
    assert summary["repeats"] == 3
    assert summary["completed_repeats"] == 3
    assert summary["failed_repeats"] == 0
    assert summary["teardown"]["ok"] is True
    assert len(summary["attempts"]) == 3

    for index, attempt in enumerate(summary["attempts"], start=1):
        attempt_dir = tmp_path / f"attempt_{index:03d}"
        assert attempt["ok"] is True
        assert attempt["attempt_index"] == index
        assert attempt["result_dir"] == str(attempt_dir)
        assert attempt["telemetry"]["artifact_size_bytes"] > 0
        assert attempt["telemetry"]["process"]["max_rss_kib"] >= 0
        assert (attempt_dir / "summary.json").exists()
        assert (attempt_dir / "run.log").exists()

    written = json.loads((tmp_path / "stability_summary.json").read_text())
    assert written == summary
    assert "repeat=1/3" in (tmp_path / "run.log").read_text()
    telemetry = json.loads((tmp_path / "telemetry" / "stability_status.json").read_text())
    assert telemetry["ok"] is True
    assert telemetry["completed_repeats"] == 3


def test_cpu_stability_preserves_failed_repeat_and_stops(tmp_path):
    def failing_runner(**kwargs):
        if kwargs["seed"] == 101:
            raise RuntimeError("synthetic repeat failure")
        return _fake_smoke_runner(**kwargs)

    summary = cpu_stability.run_cpu_stability(
        result_dir=tmp_path,
        run_id="stability_failure_fixture",
        repeats=3,
        steps=2,
        batch_size=2,
        seq_len=8,
        vocab_size=64,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        mlp_dim=32,
        seed=100,
        smoke_runner=failing_runner,
    )

    assert summary["ok"] is False
    assert summary["completed_repeats"] == 1
    assert summary["failed_repeats"] == 1
    assert len(summary["attempts"]) == 2
    assert summary["attempts"][1]["ok"] is False
    assert summary["attempts"][1]["error"]["type"] == "RuntimeError"
    assert "synthetic repeat failure" in summary["attempts"][1]["error"]["message"]
    assert (tmp_path / "attempt_002" / "failure.json").exists()
    assert "failed" in (tmp_path / "run.log").read_text()
