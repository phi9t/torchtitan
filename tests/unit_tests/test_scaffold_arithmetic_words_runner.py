# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wave F4 tests for the migrated arithmetic-words reference runner.

These verify that ``run_arithmetic_words_smoke.sh`` drives the typed
begin/stage/finish lifecycle plus the host_static profile doctor while keeping
its scientific conditions unchanged. They run on the host with no GPU and no
vLLM: TORCHTITAN_IN_ROOTFS=1 skips the bwrap re-exec, and host_static is an
ungated preflight profile.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    REPO_ROOT / "experiments" / "scaffold_to_policy" / "run_arithmetic_words_smoke.sh"
)


def _run_runner(tmp_path: Path) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """Run the migrated runner on the host under tmp roots.

    Returns the completed process plus the bundle and results-root paths.
    """

    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    env = dict(os.environ)
    env.update(
        {
            "TORCHTITAN_IN_ROOTFS": "1",
            "RUN_ID": "fixed-run-01",
            "ATTEMPT_ID": "attempt-01",
            "DATA_ROOT": str(data_root),
            "RESULTS_ROOT": str(results_root),
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    proc = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    bundle = results_root / "runs" / "fixed-run-01" / "attempt-01"
    return proc, bundle, results_root


def test_arithmetic_words_runner_syntax():
    result = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_arithmetic_words_runner_produces_attempt_bundle(tmp_path):
    proc, bundle, _ = _run_runner(tmp_path)
    assert proc.returncode == 0, proc.stderr

    for name in (
        "manifest.json",
        "outcome.json",
        "derived/report_input.json",
        "processes/coordinator/events.jsonl",
    ):
        assert (bundle / name).is_file(), f"missing bundle file {name}"

    outcome = json.loads((bundle / "outcome.json").read_text())
    assert outcome["execution_outcome"] == "completed"

    evaluations = outcome["evaluations"]
    assert set(evaluations) == {"dev", "ood_test"}
    for status in evaluations.values():
        assert status["measurement"] == "fixture"
        assert status["promotion"] == "not_evaluated"

    # Fixture rollouts must never be promotable evidence.
    assert outcome["run_gate"]["has_real_measurement"] is False


def test_arithmetic_words_runner_preflight_artifact_ready(tmp_path):
    proc, _, results_root = _run_runner(tmp_path)
    assert proc.returncode == 0, proc.stderr

    preflight = json.loads(
        (results_root / "manifests" / "preflight_fixed-run-01.json").read_text()
    )
    assert preflight["readiness"] == "ready"
    assert preflight["blocker_codes"] == []


def test_arithmetic_words_runner_report_input_parity(tmp_path):
    proc, bundle, results_root = _run_runner(tmp_path)
    assert proc.returncode == 0, proc.stderr

    derived = json.loads((bundle / "derived" / "report_input.json").read_text())
    standalone = json.loads(
        (results_root / "manifests" / "report_input_fixed-run-01.json").read_text()
    )
    # finish commits exactly the canonical report input the report stage built.
    assert derived == standalone
    assert all(derived["checks"].values()), derived["checks"]


def test_arithmetic_words_runner_writes_no_prototype_manifest(tmp_path):
    proc, _, results_root = _run_runner(tmp_path)
    assert proc.returncode == 0, proc.stderr

    # The prototype path wrote a per-run JSONL manifest; the migrated runner
    # writes only the typed attempt bundle plus JSON support files.
    stray = list((results_root / "manifests").glob("*.jsonl"))
    assert stray == [], f"unexpected prototype manifest(s): {stray}"


def test_arithmetic_words_runner_stage_events_all_succeed(tmp_path):
    proc, bundle, _ = _run_runner(tmp_path)
    assert proc.returncode == 0, proc.stderr

    rows = [
        json.loads(line)
        for line in (bundle / "processes" / "coordinator" / "events.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    started = [r["stage_id"] for r in rows if r["kind"] == "stage_started"]
    terminal_kinds = {r["kind"] for r in rows if r["kind"] != "stage_started"}

    assert started == [
        "gen-train",
        "gen-dev",
        "gen-ood_test",
        "validate-splits",
        "write-fixture-dev",
        "evaluate-dev",
        "write-fixture-ood_test",
        "evaluate-ood_test",
        "build-report-input",
    ]
    assert terminal_kinds == {"stage_succeeded"}
