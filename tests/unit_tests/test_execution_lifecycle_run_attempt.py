# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wave F1 RunAttempt lifecycle tests.

Cover RunAttempt.create/run_stage/finish (roadmap Section 4) with a fake
command executor: the attempt bundle layout (3.3), stage_started plus one
terminal event per stage, per-condition finish validation (not a run-wide
score), and immutable terminal outcome. Host-testable.
"""

from __future__ import annotations

import json

import pytest

from torchtitan.experiments.execution import models
from torchtitan.experiments.execution.executor import CommandResult, FakeExecutor
from torchtitan.experiments.execution.lifecycle import RunAttempt


def _declaration(run_id: str = "run-1") -> models.RunDeclaration:
    return models.RunDeclaration(
        run_id=run_id,
        family="reasoning",
        task="math_style",
        lane="reasoning",
        fields={"seed": 1},
    )


def _spec(
    stage_id: str = "s1", name: str = "doctor", kind: str = "doctor"
) -> models.StageSpec:
    return models.StageSpec(
        stage_id=stage_id, name=name, kind=kind, adapter="host_test", argv=["true"]
    )


def test_create_writes_manifest_with_declaration_digest(tmp_path):
    decl = _declaration()
    attempt = RunAttempt.create(decl, attempt_id="attempt-1", results_root=tmp_path)
    manifest_path = attempt.bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run"]["run_id"] == "run-1"
    assert manifest["attempt"]["attempt_id"] == "attempt-1"
    assert manifest["declaration_digest"] == decl.digest()


def test_run_stage_records_started_and_terminal_event(tmp_path):
    attempt = RunAttempt.create(
        _declaration(),
        attempt_id="attempt-1",
        results_root=tmp_path,
        executor=FakeExecutor({("true",): CommandResult(return_code=0)}),
    )
    event = attempt.run_stage(_spec())
    assert event.kind == "stage_succeeded"
    stream = attempt.bundle_dir / "processes" / "coordinator" / "events.jsonl"
    kinds = [json.loads(line)["kind"] for line in stream.read_text().splitlines()]
    assert kinds == ["stage_started", "stage_succeeded"]


def test_failed_stage_records_stage_failed(tmp_path):
    attempt = RunAttempt.create(
        _declaration(),
        attempt_id="attempt-1",
        results_root=tmp_path,
        executor=FakeExecutor({("false",): CommandResult(return_code=1)}),
    )
    spec = models.StageSpec(
        stage_id="s1", name="verify", kind="verify", adapter="host_test", argv=["false"]
    )
    event = attempt.run_stage(spec)
    assert event.kind == "stage_failed"
    assert event.return_code == 1


def test_finish_rejects_run_wide_measurement_pair(tmp_path):
    attempt = RunAttempt.create(
        _declaration(), attempt_id="attempt-1", results_root=tmp_path
    )
    # finish takes per-condition evaluations, not a single run-wide pair.
    with pytest.raises(TypeError):
        attempt.finish(
            canonical_report_input={"schema_version": 1},
            measurement="real",  # run-wide pair is not accepted
            promotion="promote",
        )


def test_finish_writes_immutable_outcome_and_report(tmp_path):
    attempt = RunAttempt.create(
        _declaration(), attempt_id="attempt-1", results_root=tmp_path
    )
    attempt.run_stage(_spec())
    evaluations = {
        "dev": models.ConditionStatus(
            execution_outcome="completed", measurement="real", promotion="hold"
        ),
    }
    outcome = attempt.finish(
        canonical_report_input={"schema_version": 1, "run": {"run_id": "run-1"}},
        evaluations=evaluations,
        attempt_outcome="completed",
    )
    assert outcome.execution_outcome == "completed"
    outcome_path = attempt.bundle_dir / "outcome.json"
    report_path = attempt.bundle_dir / "derived" / "report_input.json"
    assert outcome_path.is_file()
    assert report_path.is_file()
    # Terminal outcome is immutable: a second finish must not overwrite it.
    with pytest.raises(FileExistsError):
        attempt.finish(
            canonical_report_input={"schema_version": 1},
            evaluations=evaluations,
            attempt_outcome="completed",
        )
