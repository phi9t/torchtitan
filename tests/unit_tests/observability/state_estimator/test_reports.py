# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

from torchtitan.observability.state_estimator.fixtures import (
    ProcessFixture,
    build_minimal_evidence_bundle,
)
from torchtitan.observability.state_estimator.reports import (
    analyze_attempt_report,
    compare_attempt_reports,
)


REPORT_HEADINGS = [
    "## Evidence Quality",
    "## Topology",
    "## Timeline",
    "## Analytical Residuals",
    "## Mode Scores",
    "## Root-Cause Candidates",
    "## Transaction Risk",
    "## Probes",
    "## Learned Factors",
    "## Evaluation Metrics",
]


def _read_json(path):
    return json.loads(path.read_text())


def test_analyze_attempt_report_writes_derived_outputs_outside_raw_evidence(tmp_path):
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
    raw_manifest = attempt_path / "manifest.json"
    raw_log = (
        attempt_path / "structured_logs" / "trainer.core.global_rank_000000.jsonl"
    )
    before_manifest = raw_manifest.read_text()
    before_log = raw_log.read_text()
    output_dir = tmp_path / "operator-output"

    result = analyze_attempt_report(attempt_path, output_dir)

    assert result["ok"] is True
    artifacts = result["artifacts"]
    assert artifacts["evidence_graph"].startswith(str(output_dir))
    assert artifacts["belief_summary"].startswith(str(output_dir))
    assert artifacts["operator_report"].startswith(str(output_dir))
    assert (
        _read_json(output_dir / "belief_summary.json")["attempt_id"]
        == "attempt-001"
    )
    assert raw_manifest.read_text() == before_manifest
    assert raw_log.read_text() == before_log
    assert not (attempt_path / "derived").exists()


def test_operator_report_contains_required_operator_sections(tmp_path):
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

    result = analyze_attempt_report(attempt_path, tmp_path / "operator-output")

    report = (tmp_path / "operator-output" / "operator_report.md").read_text()
    for heading in REPORT_HEADINGS:
        assert heading in report
    assert "process_failure" in report
    assert result["artifacts"]["operator_report"].endswith("operator_report.md")


def test_compare_attempt_reports_writes_markdown_with_both_attempts(tmp_path):
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

    result = compare_attempt_reports(
        [first_attempt, second_attempt], tmp_path / "comparison"
    )

    report = (tmp_path / "comparison" / "comparison_report.md").read_text()
    assert result["ok"] is True
    assert "attempt-001" in report
    assert "attempt-002" in report
    assert result["artifacts"]["comparison_report"].startswith(
        str(tmp_path / "comparison")
    )


def test_compare_attempt_reports_keeps_same_attempt_id_from_different_runs(
    tmp_path,
):
    first_attempt = build_minimal_evidence_bundle(
        tmp_path,
        run_id="first-run",
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
        run_id="second-run",
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
    output_dir = tmp_path / "comparison"

    result = compare_attempt_reports([first_attempt, second_attempt], output_dir)

    assert result["attempt_count"] == 2
    assert len(result["attempts"]) == 2
    assert set(result["attempts"]) == {
        "first-run/attempt-001",
        "second-run/attempt-001",
    }
    artifact_dirs = {
        key: _read_json(output_dir / key / "evidence_graph.json")["run_id"]
        for key in result["attempts"]
    }
    assert artifact_dirs == {
        "first-run/attempt-001": "first-run",
        "second-run/attempt-001": "second-run",
    }
    for artifacts in result["attempt_artifacts"].values():
        assert artifacts["operator_report"].startswith(str(output_dir))
        assert artifacts["operator_report"].endswith("operator_report.md")
        assert artifacts["evidence_graph"].startswith(str(output_dir))
    assert (
        output_dir / "first-run" / "attempt-001" / "operator_report.md"
    ).exists()
    assert (
        output_dir / "second-run" / "attempt-001" / "operator_report.md"
    ).exists()
