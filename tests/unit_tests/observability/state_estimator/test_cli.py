# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from torchtitan.observability.state_estimator.cli import analyze_attempt
from torchtitan.observability.state_estimator.fixtures import (
    ProcessFixture,
    build_minimal_evidence_bundle,
)


def test_analyze_attempt_writes_graph_and_belief_outputs(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
                phases=[("train/step", 1, 1_000, 2_000)],
            )
        ],
    )

    result = analyze_attempt(attempt_path)

    assert result["ok"] is True
    assert (
        attempt_path / "derived" / "state_estimator" / "evidence_graph.json"
    ).exists()
    assert (
        attempt_path / "derived" / "state_estimator" / "belief_summary.json"
    ).exists()


def test_cli_runs_against_fixture_bundle(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
            )
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchtitan.observability.state_estimator.cli",
            "--attempt-path",
            str(attempt_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0
    assert json.loads(proc.stdout)["ok"] is True


def _run_cli(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torchtitan.observability.state_estimator.cli",
            *(str(arg) for arg in args),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_cli_analyze_attempt_emits_json_with_artifact_paths_and_output_dir(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
            )
        ],
    )
    raw_manifest = attempt_path / "manifest.json"
    before_manifest = raw_manifest.read_text()
    output_dir = tmp_path / "reports"

    proc = _run_cli(
        "analyze-attempt",
        "--attempt-path",
        attempt_path,
        "--output-dir",
        output_dir,
    )

    payload = json.loads(proc.stdout)
    assert proc.returncode == 0
    assert payload["ok"] is True
    assert payload["command"] == "analyze-attempt"
    assert payload["artifacts"]["operator_report"] == str(
        output_dir / "operator_report.md"
    )
    assert payload["artifacts"]["evidence_graph"] == str(
        output_dir / "evidence_graph.json"
    )
    assert raw_manifest.read_text() == before_manifest
    assert not (attempt_path / "derived").exists()


def test_cli_analyze_run_reports_every_attempt_under_run_path(tmp_path):
    first_attempt = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
            )
        ],
    )
    second_attempt = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-002",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000001",
                global_rank=1,
                local_rank=1,
                world_size=2,
                outcome="failed",
            )
        ],
    )
    run_path = first_attempt.parent
    output_dir = tmp_path / "run-report"

    proc = _run_cli("analyze-run", "--run-path", run_path, "--output-dir", output_dir)

    payload = json.loads(proc.stdout)
    assert proc.returncode == 0
    assert payload["ok"] is True
    assert payload["attempt_count"] == 2
    assert set(payload["attempts"]) == {"attempt-001", "attempt-002"}
    assert payload["artifacts"]["run_report"] == str(output_dir / "run_report.md")
    assert Path(
        payload["attempt_artifacts"]["attempt-001"]["operator_report"]
    ).exists()
    assert Path(
        payload["attempt_artifacts"]["attempt-002"]["operator_report"]
    ).exists()
    assert not (first_attempt / "derived").exists()
    assert not (second_attempt / "derived").exists()


def test_cli_compare_attempts_emits_json_with_comparison_report(tmp_path):
    first_attempt = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
            )
        ],
    )
    second_attempt = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-002",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
                outcome="failed",
            )
        ],
    )
    output_dir = tmp_path / "compare"

    proc = _run_cli(
        "compare-attempts",
        "--attempt-path",
        first_attempt,
        "--attempt-path",
        second_attempt,
        "--output-dir",
        output_dir,
    )

    payload = json.loads(proc.stdout)
    assert proc.returncode == 0
    assert payload["ok"] is True
    assert payload["artifacts"]["comparison_report"] == str(
        output_dir / "comparison_report.md"
    )
    assert "attempt-002" in (output_dir / "comparison_report.md").read_text()


def test_cli_evaluate_reads_manifest_and_writes_evaluation_artifacts(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
                outcome="failed",
            )
        ],
    )
    manifest_path = tmp_path / "evaluation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "id": "fixture-eval",
                "cases": [
                    {
                        "id": "case-001",
                        "attempt_path": str(attempt_path),
                        "expected_mode": "process_failure",
                        "expected_probe": None,
                        "nominal": False,
                    }
                ],
            },
            sort_keys=True,
        )
    )
    output_dir = tmp_path / "evaluation"

    proc = _run_cli(
        "evaluate", "--manifest-path", manifest_path, "--output-dir", output_dir
    )

    payload = json.loads(proc.stdout)
    assert proc.returncode == 0
    assert payload["ok"] is True
    assert payload["case_count"] == 1
    assert payload["artifacts"]["evaluation_summary"] == str(
        output_dir / "evaluation_summary.json"
    )
    assert payload["artifacts"]["evaluation_report"] == str(
        output_dir / "evaluation_report.md"
    )
    assert (
        json.loads((output_dir / "evaluation_summary.json").read_text())["case_count"]
        == 1
    )


def test_cli_malformed_required_evidence_exits_nonzero_with_actionable_json(tmp_path):
    attempt_path = tmp_path / "run_evidence" / "fixture-run" / "attempt-001"
    attempt_path.mkdir(parents=True)
    (attempt_path / "manifest.json").write_text("{not valid json")

    proc = _run_cli(
        "analyze-attempt",
        "--attempt-path",
        attempt_path,
        "--output-dir",
        tmp_path / "out",
    )

    payload = json.loads(proc.stdout)
    assert proc.returncode != 0
    assert payload["ok"] is False
    assert payload["error_type"] in {"JSONDecodeError", "ValueError"}
    assert "manifest.json" in payload["message"]
    assert payload["action"] == "repair required evidence and rerun offline analysis"


def test_cli_attempt_path_compatibility_still_writes_legacy_derived_outputs(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
            )
        ],
    )

    proc = _run_cli("--attempt-path", attempt_path)

    payload = json.loads(proc.stdout)
    assert proc.returncode == 0
    assert payload["ok"] is True
    assert payload["command"] == "compat-analyze-attempt"
    assert (
        attempt_path / "derived" / "state_estimator" / "evidence_graph.json"
    ).exists()
