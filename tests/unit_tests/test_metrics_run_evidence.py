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


class WriterAbort(BaseException):
    pass


class EvidenceAbort(BaseException):
    pass


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


@pytest.fixture
def relative_active_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("TORCHTITAN_RUN_ID", "metrics-relative-run")
    monkeypatch.setenv("TORCHTITAN_ATTEMPT_ID", "metrics-relative-attempt")
    with RunEvidence(
        RunEvidence.Config(),
        dump_folder="outputs",
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


def test_tensorboard_uses_relative_native_path_and_normalized_evidence_path(
    relative_active_evidence,
):
    """A relative event directory must not be rebased under outputs twice."""
    native_log_dir = "outputs/tensorboard"
    tensorboard = TensorBoardLogger(native_log_dir)

    assert tensorboard.writer.log_dir == native_log_dir
    tensorboard.close()

    rows = artifact_rows(relative_active_evidence, kind="tensorboard.event_stream")
    assert [row["state"] for row in rows] == ["declared", "complete"]
    assert {row["path"] for row in rows} == {"tensorboard"}
    assert {row["path_type"] for row in rows} == {"dump_relative"}
    assert Path(native_log_dir).exists()


def test_tensorboard_logger_is_noop_without_active_evidence(tmp_path):
    """Normal TensorBoard use must not require the optional evidence session."""
    tensorboard = TensorBoardLogger(str(tmp_path / "tensorboard"))
    tensorboard.log({"loss": 1.0}, step=1)
    tensorboard.close()


def test_tensorboard_constructor_failure_creates_no_evidence(tmp_path, active_evidence):
    """A writer that never opens has no native event stream to declare."""
    with mock.patch(
        "torchtitan.components.metrics.SummaryWriter",
        side_effect=WriterAbort("writer construction failed"),
    ):
        with pytest.raises(WriterAbort, match="writer construction failed"):
            TensorBoardLogger(str(tmp_path / "tensorboard"))

    assert artifact_rows(active_evidence, kind="tensorboard.event_stream") == []


def test_tensorboard_close_records_failure_for_writer_base_exception(
    tmp_path, active_evidence
):
    """A writer BaseException must transition the native event directory to failed."""
    tensorboard = TensorBoardLogger(str(tmp_path / "tensorboard"))
    with mock.patch.object(
        tensorboard.writer,
        "close",
        side_effect=WriterAbort("writer close failed"),
    ):
        with pytest.raises(WriterAbort, match="writer close failed"):
            tensorboard.close()

    rows = artifact_rows(active_evidence, kind="tensorboard.event_stream")
    assert [row["state"] for row in rows] == ["declared", "failed"]


def test_tensorboard_close_preserves_writer_base_exception_when_failed_append_raises(
    tmp_path, active_evidence
):
    """A failed-index BaseException cannot replace a writer-close BaseException."""
    tensorboard = TensorBoardLogger(str(tmp_path / "tensorboard"))
    failed_transitions = []

    def fail_failed_transition(**kwargs):
        if kwargs["state"] is ArtifactState.FAILED:
            failed_transitions.append(kwargs["state"])
            raise EvidenceAbort("failed evidence append")
        return record_run_artifact(**kwargs)

    with mock.patch.object(
        tensorboard.writer,
        "close",
        side_effect=WriterAbort("writer close failed"),
    ), mock.patch(
        "torchtitan.components.metrics.record_artifact",
        side_effect=fail_failed_transition,
    ):
        with pytest.raises(WriterAbort, match="writer close failed"):
            tensorboard.close()
    assert failed_transitions == [ArtifactState.FAILED]


def test_tensorboard_close_preserves_training_error_when_completion_append_fails(
    tmp_path, active_evidence
):
    """A completion-index failure cannot replace an already-active training error."""
    tensorboard = TensorBoardLogger(str(tmp_path / "tensorboard"))

    def fail_completion_transition(**kwargs):
        if kwargs["state"] is ArtifactState.COMPLETE:
            raise EvidenceAbort("completion evidence append failed")
        return record_run_artifact(**kwargs)

    with pytest.raises(WriterAbort, match="training failed"):
        try:
            raise WriterAbort("training failed")
        except WriterAbort:
            with mock.patch(
                "torchtitan.components.metrics.record_artifact",
                side_effect=fail_completion_transition,
            ):
                tensorboard.close()
            raise
