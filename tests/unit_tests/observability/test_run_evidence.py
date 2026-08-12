# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Behavior tests for the process-local core run evidence recorder."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from torchtitan.observability.run_evidence import (
    ArtifactState,
    bind_phase,
    event_context,
    EvidenceCollisionError,
    EvidenceContractError,
    EvidenceWriteError,
    record_artifact,
    RunEvidence,
)


@pytest.fixture
def launcher_identity(monkeypatch):
    identity = SimpleNamespace(run_id="research-run", attempt_id="launch-17")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("TORCHTITAN_RUN_ID", identity.run_id)
    monkeypatch.setenv("TORCHTITAN_ATTEMPT_ID", identity.attempt_id)
    monkeypatch.delenv("TORCHELASTIC_RESTART_COUNT", raising=False)
    return identity


def build_evidence(tmp_path):
    return RunEvidence(
        RunEvidence.Config(),
        dump_folder=str(tmp_path),
        job_config={"training": {"steps": 3}},
        role="trainer",
        actor_id="core",
    )


def read_artifact_rows(tmp_path, identity):
    index = next(
        (
            tmp_path
            / "run_evidence"
            / identity.run_id
            / identity.attempt_id
            / "indexes"
        ).glob("artifacts.*.jsonl")
    )
    return [json.loads(line) for line in index.read_text().splitlines()]


def test_config_defaults_to_enabled_relative_evidence_folder():
    config = RunEvidence.Config()

    assert config.enable is True
    assert config.folder == "run_evidence"


@pytest.mark.parametrize("folder", ("", "/absolute", "nested/folder", "../escape"))
def test_config_rejects_unsafe_evidence_folder(tmp_path, launcher_identity, folder):
    evidence = RunEvidence(
        RunEvidence.Config(folder=folder),
        dump_folder=str(tmp_path),
        job_config={"training": {"steps": 1}},
        role="trainer",
        actor_id="core",
    )

    with pytest.raises(ValueError, match="folder"):
        evidence.__enter__()


def test_enabled_evidence_rejects_remote_dump_folder(launcher_identity):
    evidence = RunEvidence(
        RunEvidence.Config(),
        dump_folder="s3://bucket/output",
        job_config={"training": {"steps": 1}},
        role="trainer",
        actor_id="core",
    )

    with pytest.raises(ValueError, match="dump_folder"):
        evidence.__enter__()


def test_single_process_run_generates_identifiers(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.delenv("TORCHTITAN_RUN_ID", raising=False)
    monkeypatch.delenv("TORCHTITAN_ATTEMPT_ID", raising=False)
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "none")

    with build_evidence(tmp_path) as evidence:
        context = event_context()

    assert len(context["run_id"]) == 32
    assert len(context["attempt_id"]) == 32
    assert evidence.run_id == context["run_id"]
    assert evidence.attempt_id == context["attempt_id"]


def test_launcher_identity_uses_environment_and_restart_suffix(
    tmp_path, launcher_identity, monkeypatch
):
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "2")

    with build_evidence(tmp_path) as evidence:
        pass

    assert evidence.run_id == launcher_identity.run_id
    assert evidence.attempt_id == "launch-17-restart-2"


def test_multi_process_run_requires_launcher_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("TORCHTITAN_RUN_ID", raising=False)
    monkeypatch.delenv("TORCHTITAN_ATTEMPT_ID", raising=False)
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "none")

    evidence = RunEvidence(
        RunEvidence.Config(),
        dump_folder=str(tmp_path),
        job_config={"training": {"steps": 1}},
        role="trainer",
        actor_id="core",
    )

    with pytest.raises(ValueError, match="TORCHTITAN_RUN_ID"):
        evidence.__enter__()


def test_manifest_and_process_index_are_created(tmp_path, launcher_identity):
    with RunEvidence(
        RunEvidence.Config(),
        dump_folder=str(tmp_path),
        job_config={"training": {"steps": 3}},
        role="trainer",
        actor_id="core",
    ):
        pass

    attempt = (
        tmp_path
        / "run_evidence"
        / launcher_identity.run_id
        / launcher_identity.attempt_id
    )
    manifest_path = attempt / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "research-run"
    assert manifest["attempt_id"] == "launch-17"
    assert manifest["config"]["normalized"] == {"training": {"steps": 3}}
    assert (
        manifest["config"]["sha256"]
        == hashlib.sha256(b'{"training":{"steps":3}}').hexdigest()
    )
    assert (
        manifest_path.read_text()
        == json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    assert manifest["source"]["revision"]
    assert isinstance(manifest["source"]["dirty"], bool)
    assert manifest["command"]
    assert manifest["runtime"]["python"]
    assert [path.name for path in (attempt / "indexes").glob("artifacts.*.jsonl")] == [
        "artifacts.trainer.core.global_rank_000000.jsonl"
    ]


def test_process_index_collision_fails_before_training(tmp_path, launcher_identity):
    with build_evidence(tmp_path):
        pass

    with pytest.raises(EvidenceCollisionError, match="process index"):
        build_evidence(tmp_path).__enter__()


def test_artifact_facade_is_noop_without_an_installed_recorder(tmp_path):
    assert event_context() == {}
    assert (
        record_artifact(producer="test", kind="test.file", path=tmp_path / "missing")
        is None
    )


def test_artifact_index_is_append_only(tmp_path, launcher_identity):
    with build_evidence(tmp_path):
        trace_path = tmp_path / "profiling" / "trace.json.gz"
        artifact_id = record_artifact(
            producer="pytorch_profiler",
            kind="pytorch.profiler.trace",
            path=trace_path,
            state=ArtifactState.DECLARED,
            step=4,
        )
        trace_path.parent.mkdir(parents=True)
        trace_path.write_bytes(b"compressed-trace")
        assert (
            record_artifact(
                producer="pytorch_profiler",
                kind="pytorch.profiler.trace",
                path=trace_path,
                state=ArtifactState.COMPLETE,
                artifact_id=artifact_id,
                step=4,
            )
            == artifact_id
        )

    rows = read_artifact_rows(tmp_path, launcher_identity)
    assert [row["state"] for row in rows] == ["declared", "complete"]
    assert rows[0]["artifact_id"] == rows[1]["artifact_id"]
    assert rows[0]["path"] == "profiling/trace.json.gz"
    assert rows[0]["path_type"] == "dump_relative"
    assert rows[1]["artifact_seq"] == rows[0]["artifact_seq"] + 1
    assert rows[1]["wall_time_ns"] >= rows[0]["wall_time_ns"]
    assert rows[1]["monotonic_ns"] >= rows[0]["monotonic_ns"]


def test_artifact_lifecycle_rejects_invalid_transition_and_metadata(
    tmp_path, launcher_identity
):
    with build_evidence(tmp_path):
        path = tmp_path / "checkpoint"
        artifact_id = record_artifact(
            producer="checkpoint",
            kind="checkpoint.directory",
            path=path,
            state=ArtifactState.DECLARED,
        )
        with pytest.raises(EvidenceContractError, match="transition"):
            record_artifact(
                producer="checkpoint",
                kind="checkpoint.directory",
                path=path,
                state=ArtifactState.DECLARED,
                artifact_id=artifact_id,
            )
        with pytest.raises(EvidenceContractError, match="metadata"):
            record_artifact(
                producer="checkpoint",
                kind="checkpoint.directory",
                path=path,
                state=ArtifactState.FAILED,
                artifact_id=artifact_id,
                metadata={"not_json": object()},
            )
        with pytest.raises(EvidenceContractError, match="does not exist"):
            record_artifact(
                producer="checkpoint",
                kind="checkpoint.directory",
                path=path,
                state=ArtifactState.COMPLETE,
                artifact_id=artifact_id,
            )
        assert (
            record_artifact(
                producer="checkpoint",
                kind="checkpoint.directory",
                path=path,
                state=ArtifactState.FAILED,
                artifact_id=artifact_id,
            )
            == artifact_id
        )
        with pytest.raises(EvidenceContractError, match="transition"):
            record_artifact(
                producer="checkpoint",
                kind="checkpoint.directory",
                path=path,
                state=ArtifactState.COMPLETE,
                artifact_id=artifact_id,
            )


def test_artifact_paths_preserve_external_locations(tmp_path, launcher_identity):
    external_path = tmp_path.parent / "external.bin"
    external_path.write_bytes(b"external")
    with build_evidence(tmp_path):
        record_artifact(
            producer="external",
            kind="artifact",
            path=external_path,
        )
        record_artifact(
            producer="remote",
            kind="artifact",
            path="s3://bucket/object",
        )

    external, remote = read_artifact_rows(tmp_path, launcher_identity)
    assert external["path"] == str(external_path)
    assert external["path_type"] == "external_absolute"
    assert remote["path"] == "s3://bucket/object"
    assert remote["path_type"] == "uri"


def test_event_context_tracks_phase_and_distributed_binding(
    tmp_path, launcher_identity
):
    with build_evidence(tmp_path) as evidence:
        first = event_context()
        with bind_phase("training"):
            phase_context = event_context()
            with bind_phase("checkpoint"):
                nested_context = event_context()
            restored_context = event_context()
        evidence.bind_distributed(
            SimpleNamespace(
                dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1
            ),
            torch.device("cpu"),
        )
        distributed_context = event_context()

    assert first["event_seq"] + 1 == phase_context["event_seq"]
    assert phase_context["phase"] == "training"
    assert nested_context["phase"] != phase_context["phase"]
    assert restored_context["phase"] == "training"
    assert "phase" not in first
    assert distributed_context["device_type"] == "cpu"
    assert distributed_context["tp"] == 1
    assert distributed_context["world_size"] == 1


def test_artifact_append_failure_is_fatal_without_an_active_exception(
    tmp_path, launcher_identity
):
    class FailingIndex:
        def write(self, value):
            raise OSError("no space")

        def flush(self):
            pass

        def close(self):
            pass

    with build_evidence(tmp_path) as evidence:
        evidence._index_file.close()
        evidence._index_file = FailingIndex()
        with pytest.raises(EvidenceWriteError, match="append"):
            record_artifact(producer="test", kind="artifact", path="s3://bucket/object")


def test_artifact_append_failure_preserves_active_exception(
    tmp_path, launcher_identity
):
    class FailingIndex:
        def write(self, value):
            raise OSError("no space")

        def flush(self):
            pass

        def close(self):
            pass

    with build_evidence(tmp_path) as evidence:
        evidence._index_file.close()
        evidence._index_file = FailingIndex()
        with pytest.raises(RuntimeError, match="training failed"):
            try:
                raise RuntimeError("training failed")
            except RuntimeError:
                assert (
                    record_artifact(
                        producer="test", kind="artifact", path="s3://bucket/object"
                    )
                    is None
                )
                raise


def test_process_outcome_records_success_and_failure(tmp_path, launcher_identity):
    with build_evidence(tmp_path):
        pass
    succeeded = json.loads(
        (
            tmp_path
            / "run_evidence"
            / launcher_identity.run_id
            / launcher_identity.attempt_id
            / "processes"
            / "trainer.core.global_rank_000000"
            / "outcome.json"
        ).read_text()
    )
    assert succeeded["outcome"] == "succeeded"
    assert succeeded["elapsed_monotonic_ns"] >= 0
