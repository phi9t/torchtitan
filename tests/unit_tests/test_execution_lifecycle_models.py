# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wave F1 typed-lifecycle model tests.

These cover the typed run-attempt lifecycle models defined by the runtime
preflight roadmap Sections 3-7: the run declaration and its normalized digest,
attempts and their parent linkage, stage declarations and terminal events, and
artifact references with work-status/freshness pairing. They are host-testable
and require no GPU, rootfs, or Docker.
"""

from __future__ import annotations

import pytest

from torchtitan.experiments.execution import models


def test_declaration_digest_is_deterministic_and_order_independent():
    a = models.RunDeclaration(
        run_id="run-1",
        family="reasoning",
        task="math_style",
        lane="reasoning",
        fields={"model": "qwen3-1.7b", "seed": 42},
    )
    b = models.RunDeclaration(
        run_id="run-1",
        family="reasoning",
        task="math_style",
        lane="reasoning",
        fields={"seed": 42, "model": "qwen3-1.7b"},
    )
    assert a.digest() == b.digest()


def test_declaration_digest_changes_with_fields():
    base = models.RunDeclaration(
        run_id="run-1", family="f", task="t", lane="l", fields={"model": "a"}
    )
    changed = models.RunDeclaration(
        run_id="run-1", family="f", task="t", lane="l", fields={"model": "b"}
    )
    assert base.digest() != changed.digest()


def test_attempt_requires_nonempty_ids():
    decl = models.RunDeclaration(
        run_id="run-1", family="f", task="t", lane="l", fields={}
    )
    with pytest.raises(ValueError):
        models.Attempt(attempt_id="", declaration=decl)


def test_attempt_records_parent_link_for_resume():
    decl = models.RunDeclaration(
        run_id="run-1", family="f", task="t", lane="l", fields={}
    )
    parent = models.Attempt(attempt_id="attempt-1", declaration=decl)
    resume = models.Attempt(
        attempt_id="attempt-2", declaration=decl, parent_attempt_id=parent.attempt_id
    )
    assert resume.parent_attempt_id == "attempt-1"
    assert parent.parent_attempt_id is None


def test_stage_spec_rejects_unknown_kind_and_adapter():
    with pytest.raises(ValueError):
        models.StageSpec(
            stage_id="s1",
            name="bad",
            kind="not_a_kind",
            adapter="host_test",
            argv=["true"],
        )
    with pytest.raises(ValueError):
        models.StageSpec(
            stage_id="s1",
            name="bad",
            kind="doctor",
            adapter="not_an_adapter",
            argv=["true"],
        )


def test_stage_spec_requires_argv():
    with pytest.raises(ValueError):
        models.StageSpec(
            stage_id="s1", name="empty", kind="doctor", adapter="host_test", argv=[]
        )


def test_stage_event_terminal_kinds_are_constrained():
    started = models.StageEvent(
        kind="stage_started", stage_id="s1", stage_invocation_id="inv-1"
    )
    assert not started.is_terminal
    for terminal in (
        "stage_succeeded",
        "stage_blocked",
        "stage_failed",
        "stage_interrupted",
    ):
        event = models.StageEvent(
            kind=terminal, stage_id="s1", stage_invocation_id="inv-1"
        )
        assert event.is_terminal
    with pytest.raises(ValueError):
        models.StageEvent(kind="mystery", stage_id="s1", stage_invocation_id="inv-1")


def test_artifact_ref_pairs_work_status_and_freshness():
    produced = models.ArtifactRef(
        artifact_id="a1",
        artifact_class="report",
        path="derived/report_input.json",
        work_status="produced",
        freshness="verified_new",
    )
    assert produced.work_status == "produced"
    # imported/external evidence may remain unknown freshness.
    imported = models.ArtifactRef(
        artifact_id="a2",
        artifact_class="checkpoint",
        path="checkpoints/x.json",
        work_status="imported",
        freshness="unknown",
    )
    assert imported.freshness == "unknown"


def test_artifact_ref_rejects_invalid_work_status_and_freshness():
    with pytest.raises(ValueError):
        models.ArtifactRef(
            artifact_id="a1",
            artifact_class="report",
            path="p",
            work_status="fresh",  # 'fresh' is not a work_status value (roadmap 6)
            freshness="verified_new",
        )
    with pytest.raises(ValueError):
        models.ArtifactRef(
            artifact_id="a1",
            artifact_class="report",
            path="p",
            work_status="produced",
            freshness="brand_new",
        )


def test_attempt_outcome_is_independent_terminal_summary():
    outcome = models.AttemptOutcome(
        attempt_id="attempt-1",
        execution_outcome="completed",
        stage_invocation_ids=["inv-1", "inv-2"],
    )
    assert outcome.execution_outcome == "completed"
    with pytest.raises(ValueError):
        models.AttemptOutcome(
            attempt_id="attempt-1",
            execution_outcome="not_a_terminal",
            stage_invocation_ids=[],
        )
