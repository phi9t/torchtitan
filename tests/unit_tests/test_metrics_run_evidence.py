# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorBoard native artifact lifecycle tests."""

import json
from pathlib import Path
from unittest import mock

import pytest
from torchtitan.components.metrics import TensorBoardLogger
from torchtitan.observability.run_evidence import (
    ArtifactState,
    record_artifact as record_run_artifact,
    RunEvidence,
)


@pytest.fixture
def active_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("TORCHTITAN_RUN_ID", "metrics-evidence-run")
    monkeypatch.setenv("TORCHTITAN_ATTEMPT_ID", "metrics-evidence-attempt")
    with RunEvidence(
        RunEvidence.Config(),
        dump_folder=str(tmp_path),
        job_config={"training": {"steps": 8}},
        role="trainer",
        actor_id="core",
    ) as evidence:
        yield evidence


def artifact_rows(active_evidence, *, kind):
    index_path = next(
        (
            Path(active_evidence.dump_folder)
            / "run_evidence"
            / active_evidence.run_id
            / active_evidence.attempt_id
            / "indexes"
        ).glob("artifacts.*.jsonl")
    )
    return [
        row
        for row in (json.loads(line) for line in index_path.read_text().splitlines())
        if row["kind"] == kind
    ]


def test_tensorboard_logger_records_evidence_lifecycle(tmp_path, active_evidence):
    """A missing close transition would make a TensorBoard event directory incomplete."""
    log_dir = tmp_path / "tensorboard"
    tensorboard = TensorBoardLogger(str(log_dir))

    declared_rows = artifact_rows(active_evidence, kind="tensorboard.event_stream")
    assert [row["state"] for row in declared_rows] == ["declared"]
    assert declared_rows[0]["producer"] == "tensorboard"
    assert declared_rows[0]["metadata"]["format"] == "event_directory"

    tensorboard.close()
    tensorboard.close()

    rows = artifact_rows(active_evidence, kind="tensorboard.event_stream")
    assert [row["state"] for row in rows] == ["declared", "complete"]
    assert log_dir.exists()


def test_tensorboard_logger_is_noop_without_active_evidence(tmp_path):
    """Normal TensorBoard use must not require the optional evidence session."""
    tensorboard = TensorBoardLogger(str(tmp_path / "tensorboard"))
    tensorboard.log({"loss": 1.0}, step=1)
    tensorboard.close()


def test_tensorboard_close_preserves_training_error_when_completion_append_fails(
    tmp_path, active_evidence
):
    """A completion-index failure cannot replace an already-active training error."""
    tensorboard = TensorBoardLogger(str(tmp_path / "tensorboard"))

    def fail_completion_transition(**kwargs):
        if kwargs["state"] is ArtifactState.COMPLETE:
            raise OSError("completion evidence append failed")
        return record_run_artifact(**kwargs)

    with pytest.raises(RuntimeError, match="training failed"):
        try:
            raise RuntimeError("training failed")
        except RuntimeError:
            with mock.patch(
                "torchtitan.components.metrics.record_artifact",
                side_effect=fail_completion_transition,
            ):
                tensorboard.close()
            raise
