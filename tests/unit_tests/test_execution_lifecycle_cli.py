# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Wave F1 begin/stage/finish CLI facade tests (roadmap Section 4).

The shell facade exposes the migration-compatible equivalent of the typed
lifecycle so run_common.sh and existing task runners can sequence shell
commands and still get the typed attempt bundle. Each subcommand is a separate
process, so state is reconstructed from the immutable/append-only bundle rather
than held in memory:

    python -m torchtitan.experiments.execution begin ...
    python -m torchtitan.experiments.execution stage ... -- COMMAND...
    python -m torchtitan.experiments.execution finish ...

These tests drive the facade directly (host-testable, no GPU) with commands
like ``true``/``false`` so no real runner is required.
"""

from __future__ import annotations

import json

import pytest

from torchtitan.experiments.execution import cli


def _begin(tmp_path, run_id="run-1", attempt_id="attempt-1"):
    return cli.main(
        [
            "begin",
            "--results-root",
            str(tmp_path),
            "--run-id",
            run_id,
            "--family",
            "reasoning",
            "--task",
            "math_style",
            "--lane",
            "reasoning",
            "--attempt-id",
            attempt_id,
        ]
    )


def _bundle_dir(tmp_path, run_id="run-1", attempt_id="attempt-1"):
    return tmp_path / "runs" / run_id / attempt_id


def test_begin_creates_bundle_with_manifest(tmp_path):
    rc = _begin(tmp_path)
    assert rc == 0
    manifest = json.loads((_bundle_dir(tmp_path) / "manifest.json").read_text())
    assert manifest["run"]["run_id"] == "run-1"
    assert manifest["attempt"]["attempt_id"] == "attempt-1"


def test_stage_runs_command_and_appends_events(tmp_path):
    _begin(tmp_path)
    rc = cli.main(
        [
            "stage",
            "--results-root",
            str(tmp_path),
            "--run-id",
            "run-1",
            "--attempt-id",
            "attempt-1",
            "--stage-id",
            "s1",
            "--name",
            "doctor",
            "--kind",
            "doctor",
            "--adapter",
            "host_test",
            "--",
            "true",
        ]
    )
    assert rc == 0
    stream = _bundle_dir(tmp_path) / "processes" / "coordinator" / "events.jsonl"
    kinds = [json.loads(line)["kind"] for line in stream.read_text().splitlines()]
    assert kinds == ["stage_started", "stage_succeeded"]


def test_stage_failure_returns_nonzero_and_records_failure(tmp_path):
    _begin(tmp_path)
    rc = cli.main(
        [
            "stage",
            "--results-root",
            str(tmp_path),
            "--run-id",
            "run-1",
            "--attempt-id",
            "attempt-1",
            "--stage-id",
            "s1",
            "--name",
            "verify",
            "--kind",
            "verify",
            "--adapter",
            "host_test",
            "--",
            "false",
        ]
    )
    assert rc != 0
    stream = _bundle_dir(tmp_path) / "processes" / "coordinator" / "events.jsonl"
    kinds = [json.loads(line)["kind"] for line in stream.read_text().splitlines()]
    assert kinds[-1] == "stage_failed"


def test_stage_can_map_return_code_to_blocked(tmp_path):
    _begin(tmp_path)
    rc = cli.main(
        [
            "stage",
            "--results-root",
            str(tmp_path),
            "--run-id",
            "run-1",
            "--attempt-id",
            "attempt-1",
            "--stage-id",
            "preflight",
            "--name",
            "preflight",
            "--kind",
            "preflight",
            "--adapter",
            "rootfs_cpu",
            "--terminal-kind-for-return-code",
            "1=stage_blocked",
            "--",
            "false",
        ]
    )
    assert rc == 1
    stream = _bundle_dir(tmp_path) / "processes" / "coordinator" / "events.jsonl"
    rows = [json.loads(line) for line in stream.read_text().splitlines()]
    assert rows[-1]["kind"] == "stage_blocked"
    assert rows[-1]["return_code"] == 1


def test_finish_reconstructs_invocations_from_event_stream(tmp_path):
    _begin(tmp_path)
    for stage_id in ("s1", "s2"):
        cli.main(
            [
                "stage",
                "--results-root",
                str(tmp_path),
                "--run-id",
                "run-1",
                "--attempt-id",
                "attempt-1",
                "--stage-id",
                stage_id,
                "--name",
                "doctor",
                "--kind",
                "doctor",
                "--adapter",
                "host_test",
                "--",
                "true",
            ]
        )
    report = tmp_path / "report_input.json"
    report.write_text(json.dumps({"schema_version": 1, "run": {"run_id": "run-1"}}))
    evaluations = tmp_path / "evaluations.json"
    evaluations.write_text(
        json.dumps(
            {
                "dev": {
                    "execution_outcome": "completed",
                    "measurement": "real",
                    "promotion": "hold",
                }
            }
        )
    )
    rc = cli.main(
        [
            "finish",
            "--results-root",
            str(tmp_path),
            "--run-id",
            "run-1",
            "--attempt-id",
            "attempt-1",
            "--attempt-outcome",
            "completed",
            "--report-input",
            str(report),
            "--evaluations",
            str(evaluations),
        ]
    )
    assert rc == 0
    outcome = json.loads((_bundle_dir(tmp_path) / "outcome.json").read_text())
    assert outcome["execution_outcome"] == "completed"
    # Both stages must be reconstructed from the append-only event stream.
    assert len(outcome["stage_invocation_ids"]) == 2
    assert outcome["evaluations"]["dev"]["measurement"] == "real"


def test_finish_is_immutable(tmp_path):
    _begin(tmp_path)
    report = tmp_path / "report_input.json"
    report.write_text(json.dumps({"schema_version": 1}))
    evaluations = tmp_path / "evaluations.json"
    evaluations.write_text(json.dumps({}))
    args = [
        "finish",
        "--results-root",
        str(tmp_path),
        "--run-id",
        "run-1",
        "--attempt-id",
        "attempt-1",
        "--attempt-outcome",
        "completed",
        "--report-input",
        str(report),
        "--evaluations",
        str(evaluations),
    ]
    assert cli.main(args) == 0
    with pytest.raises(FileExistsError):
        cli.main(args)
