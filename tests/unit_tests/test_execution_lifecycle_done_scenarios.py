# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Wave F1 done-scenario integration tests (roadmap Section 19 / F1).

F1 is done when a fake multi-stage run, a blocked run, a failed run, and an
interrupted attempt followed by a linked resume attempt all produce validated
immutable bundles. Each test drives the typed lifecycle with a deterministic
FakeExecutor (host-testable, no GPU) and validates the bundle: manifest,
coordinator event stream with one terminal event per stage, immutable
outcome.json, and derived report input.
"""

from __future__ import annotations

import json

import pytest

from torchtitan.experiments.execution import models
from torchtitan.experiments.execution.executor import CommandResult, FakeExecutor
from torchtitan.experiments.execution.lifecycle import RunAttempt


def _declaration(run_id="run-1"):
    return models.RunDeclaration(
        run_id=run_id,
        family="reasoning",
        task="math_style",
        lane="reasoning",
        fields={"seed": 1},
    )


def _spec(stage_id, kind="doctor", argv=("true",)):
    return models.StageSpec(
        stage_id=stage_id,
        name=kind,
        kind=kind,
        adapter="host_test",
        argv=list(argv),
    )


def _event_kinds(attempt):
    stream = attempt.bundle_dir / "processes" / "coordinator" / "events.jsonl"
    return [json.loads(line)["kind"] for line in stream.read_text().splitlines()]


def _assert_immutable_bundle(attempt, *, expected_outcome):
    """Every done scenario must leave a validated, immutable terminal bundle."""

    manifest = json.loads((attempt.bundle_dir / "manifest.json").read_text())
    assert manifest["kind"] == "attempt_manifest"
    outcome = json.loads((attempt.bundle_dir / "outcome.json").read_text())
    assert outcome["execution_outcome"] == expected_outcome
    assert (attempt.bundle_dir / "derived" / "report_input.json").is_file()
    # Terminal files are immutable: re-finishing must refuse to overwrite.
    with pytest.raises(FileExistsError):
        attempt.finish(
            canonical_report_input={"schema_version": 1},
            evaluations={},
            attempt_outcome=expected_outcome,
        )
    return outcome


def test_fake_multi_stage_run_produces_completed_bundle(tmp_path):
    attempt = RunAttempt.create(
        _declaration(),
        attempt_id="attempt-1",
        results_root=tmp_path,
        executor=FakeExecutor(),
    )
    for stage_id in ("doctor", "generate", "verify"):
        event = attempt.run_stage(_spec(stage_id, kind=stage_id))
        assert event.kind == "stage_succeeded"
    outcome = attempt.finish(
        canonical_report_input={"schema_version": 1, "run": {"run_id": "run-1"}},
        evaluations={
            "dev": models.ConditionStatus(
                execution_outcome="completed", measurement="real", promotion="promote"
            )
        },
        attempt_outcome="completed",
    )
    assert outcome.execution_outcome == "completed"
    assert _event_kinds(attempt) == [
        "stage_started",
        "stage_succeeded",
        "stage_started",
        "stage_succeeded",
        "stage_started",
        "stage_succeeded",
    ]
    committed = _assert_immutable_bundle(attempt, expected_outcome="completed")
    assert committed["run_gate"]["has_real_measurement"] is True


def test_blocked_run_records_blocked_and_no_fabricated_score(tmp_path):
    # A preflight gate blocks the launch; no measurement is fabricated.
    executor = FakeExecutor(
        {("preflight",): CommandResult(return_code=0, terminal_kind="stage_blocked")}
    )
    attempt = RunAttempt.create(
        _declaration(), attempt_id="attempt-1", results_root=tmp_path, executor=executor
    )
    event = attempt.run_stage(_spec("preflight", kind="preflight", argv=("preflight",)))
    assert event.kind == "stage_blocked"
    outcome = attempt.finish(
        canonical_report_input={"schema_version": 1},
        evaluations={
            "dev": models.ConditionStatus(
                execution_outcome="blocked",
                measurement="not_run",
                promotion="not_evaluated",
            )
        },
        attempt_outcome="blocked",
    )
    assert outcome.execution_outcome == "blocked"
    committed = _assert_immutable_bundle(attempt, expected_outcome="blocked")
    assert committed["run_gate"]["has_real_measurement"] is False


def test_failed_run_records_failed_and_invalid_measurement(tmp_path):
    executor = FakeExecutor({("false",): CommandResult(return_code=1)})
    attempt = RunAttempt.create(
        _declaration(), attempt_id="attempt-1", results_root=tmp_path, executor=executor
    )
    event = attempt.run_stage(_spec("verify", kind="verify", argv=("false",)))
    assert event.kind == "stage_failed"
    assert event.return_code == 1
    outcome = attempt.finish(
        canonical_report_input={"schema_version": 1},
        evaluations={
            "dev": models.ConditionStatus(
                execution_outcome="failed",
                measurement="invalid",
                promotion="not_evaluated",
            )
        },
        attempt_outcome="failed",
    )
    assert outcome.execution_outcome == "failed"
    _assert_immutable_bundle(attempt, expected_outcome="failed")


def test_interrupted_attempt_then_linked_resume(tmp_path):
    # First attempt is interrupted mid-stage; its evidence is retained.
    interrupted_executor = FakeExecutor(
        {("train",): CommandResult(return_code=130, terminal_kind="stage_interrupted")}
    )
    first = RunAttempt.create(
        _declaration(),
        attempt_id="attempt-1",
        results_root=tmp_path,
        executor=interrupted_executor,
    )
    event = first.run_stage(_spec("train", kind="train", argv=("train",)))
    assert event.kind == "stage_interrupted"
    first.finish(
        canonical_report_input={"schema_version": 1},
        evaluations={
            "dev": models.ConditionStatus(
                execution_outcome="interrupted",
                measurement="not_run",
                promotion="not_evaluated",
            )
        },
        attempt_outcome="interrupted",
    )
    _assert_immutable_bundle(first, expected_outcome="interrupted")

    # A resume is a new attempt that links its parent; it never reopens the old
    # attempt (roadmap 3.2). Both bundles coexist.
    resume = RunAttempt.create(
        _declaration(),
        attempt_id="attempt-2",
        results_root=tmp_path,
        parent_attempt_id="attempt-1",
        executor=FakeExecutor(),
    )
    resume.run_stage(_spec("train", kind="train", argv=("true",)))
    resume.finish(
        canonical_report_input={"schema_version": 1},
        evaluations={
            "dev": models.ConditionStatus(
                execution_outcome="completed", measurement="real", promotion="hold"
            )
        },
        attempt_outcome="completed",
    )
    resume_manifest = json.loads((resume.bundle_dir / "manifest.json").read_text())
    assert resume_manifest["attempt"]["parent_attempt_id"] == "attempt-1"
    # The interrupted parent bundle is retained, not overwritten.
    assert (first.bundle_dir / "outcome.json").is_file()
    _assert_immutable_bundle(resume, expected_outcome="completed")
