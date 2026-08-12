# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Behavior tests for the process-local core run evidence recorder."""

import hashlib
import json
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from torchtitan.observability.run_evidence import (
    ArtifactRelation,
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


def outcome_path(tmp_path, identity, process_id="trainer.core.global_rank_000000"):
    return (
        tmp_path
        / "run_evidence"
        / identity.run_id
        / identity.attempt_id
        / "processes"
        / process_id
        / "outcome.json"
    )


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


@pytest.mark.parametrize(
    ("titan_run_id", "titan_attempt_id", "elastic_run_id", "expected"),
    (
        (None, None, "elastic-42", ("elastic-42", "elastic-42")),
        (
            "preferred-run",
            "preferred-attempt",
            "elastic-42",
            ("preferred-run", "preferred-attempt"),
        ),
    ),
)
def test_identity_uses_elastic_fallback_and_titan_precedence(
    tmp_path,
    monkeypatch,
    titan_run_id,
    titan_attempt_id,
    elastic_run_id,
    expected,
):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", elastic_run_id)
    for name, value in (
        ("TORCHTITAN_RUN_ID", titan_run_id),
        ("TORCHTITAN_ATTEMPT_ID", titan_attempt_id),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    with build_evidence(tmp_path) as evidence:
        pass

    assert (evidence.run_id, evidence.attempt_id) == expected


@pytest.mark.parametrize("name", ("TORCHTITAN_RUN_ID", "TORCHTITAN_ATTEMPT_ID"))
def test_identity_rejects_invalid_identifier(tmp_path, monkeypatch, name):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("TORCHTITAN_RUN_ID", "valid-run")
    monkeypatch.setenv("TORCHTITAN_ATTEMPT_ID", "valid-attempt")
    monkeypatch.setenv(name, "invalid/id")

    with pytest.raises(ValueError, match="must match"):
        build_evidence(tmp_path).__enter__()


def test_multi_process_run_requires_launcher_attempt_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("TORCHTITAN_RUN_ID", "shared-run")
    monkeypatch.delenv("TORCHTITAN_ATTEMPT_ID", raising=False)
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "none")

    with pytest.raises(ValueError, match="TORCHTITAN_ATTEMPT_ID"):
        build_evidence(tmp_path).__enter__()


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    (
        ("role", "/absolute"),
        ("role", "nested/role"),
        ("role", "../escape"),
        ("actor_id", "/absolute"),
        ("actor_id", "nested/actor"),
        ("actor_id", "../escape"),
    ),
)
def test_process_identity_rejects_unsafe_path_components_before_creation(
    tmp_path, launcher_identity, field, unsafe_value
):
    values = {"role": "trainer", "actor_id": "core"}
    values[field] = unsafe_value
    evidence = RunEvidence(
        RunEvidence.Config(),
        dump_folder=str(tmp_path),
        job_config={"training": {"steps": 1}},
        **values,
    )

    with pytest.raises(ValueError, match=field):
        evidence.__enter__()

    assert not (tmp_path / "run_evidence").exists()


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


def test_shared_manifest_allows_distinct_rank_indexes(
    tmp_path, launcher_identity, monkeypatch
):
    with build_evidence(tmp_path):
        pass

    monkeypatch.setenv("RANK", "1")
    with build_evidence(tmp_path):
        pass

    index_names = sorted(
        path.name
        for path in (
            tmp_path
            / "run_evidence"
            / launcher_identity.run_id
            / launcher_identity.attempt_id
            / "indexes"
        ).glob("artifacts.*.jsonl")
    )
    assert index_names == [
        "artifacts.trainer.core.global_rank_000000.jsonl",
        "artifacts.trainer.core.global_rank_000001.jsonl",
    ]


@pytest.mark.parametrize(
    "mismatch",
    (
        "schema_version",
        "run_id",
        "attempt_id",
        "config",
        "source",
        "command",
        "runtime",
    ),
)
def test_shared_manifest_rejects_required_field_mismatch(
    tmp_path, launcher_identity, monkeypatch, mismatch
):
    with build_evidence(tmp_path):
        pass

    monkeypatch.setenv("RANK", "1")
    if mismatch in {"schema_version", "run_id", "attempt_id"}:
        manifest_path = (
            tmp_path
            / "run_evidence"
            / launcher_identity.run_id
            / launcher_identity.attempt_id
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        manifest[mismatch] = 2 if mismatch == "schema_version" else "other"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        evidence = build_evidence(tmp_path)
    elif mismatch == "config":
        evidence = RunEvidence(
            RunEvidence.Config(),
            dump_folder=str(tmp_path),
            job_config={"training": {"steps": 4}},
            role="trainer",
            actor_id="core",
        )
    elif mismatch == "command":
        monkeypatch.setattr(sys, "argv", ["different-command"])
        evidence = build_evidence(tmp_path)
    elif mismatch == "runtime":
        monkeypatch.setattr(torch, "__version__", "different-runtime")
        evidence = build_evidence(tmp_path)
    else:
        import torchtitan.observability.run_evidence as run_evidence

        monkeypatch.setattr(
            run_evidence, "_source_state", lambda: {"revision": "other", "dirty": False}
        )
        evidence = build_evidence(tmp_path)

    with pytest.raises(EvidenceContractError, match="manifest"):
        evidence.__enter__()


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


def test_concurrent_declared_artifacts_share_one_locked_lifecycle_transaction(
    tmp_path, launcher_identity
):
    class BarrierArtifacts(dict):
        def __init__(self, lock):
            super().__init__()
            self.lock = lock
            self.barrier = threading.Barrier(2)

        def get(self, key, default=None):
            if not self.lock.locked():
                self.barrier.wait(timeout=5)
            return super().get(key, default)

    with build_evidence(tmp_path) as evidence:
        evidence._artifacts = BarrierArtifacts(evidence._lock)
        start = threading.Barrier(3)
        results: list[str | BaseException] = []

        def record_declared():
            start.wait(timeout=5)
            try:
                results.append(
                    record_artifact(
                        producer="checkpoint",
                        kind="checkpoint.directory",
                        path="s3://bucket/checkpoint",
                        state=ArtifactState.DECLARED,
                    )
                )
            except BaseException as error:
                results.append(error)

        threads = [threading.Thread(target=record_declared) for _ in range(2)]
        for thread in threads:
            thread.start()
        start.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert sum(isinstance(result, str) for result in results) == 1
    assert sum(isinstance(result, EvidenceContractError) for result in results) == 1
    rows = read_artifact_rows(tmp_path, launcher_identity)
    assert [(row["state"], row["artifact_seq"]) for row in rows] == [("declared", 0)]


def test_artifact_identity_is_stable_across_ranks_and_relations(
    tmp_path, launcher_identity, monkeypatch
):
    with build_evidence(tmp_path):
        output_id = record_artifact(
            producer="checkpoint",
            kind="checkpoint.directory",
            path="s3://bucket/checkpoint",
            relation=ArtifactRelation.OUTPUT,
        )
        input_id = record_artifact(
            producer="checkpoint",
            kind="checkpoint.directory",
            path="s3://bucket/checkpoint",
            relation=ArtifactRelation.INPUT,
        )

    monkeypatch.setenv("RANK", "1")
    with build_evidence(tmp_path):
        rank_one_output_id = record_artifact(
            producer="checkpoint",
            kind="checkpoint.directory",
            path="s3://bucket/checkpoint",
            relation=ArtifactRelation.OUTPUT,
        )

    assert output_id == rank_one_output_id
    assert input_id != output_id


def test_retired_artifact_transitions_are_append_only(tmp_path, launcher_identity):
    with build_evidence(tmp_path):
        declared_id = record_artifact(
            producer="checkpoint",
            kind="checkpoint.directory",
            path="s3://bucket/declared",
            state=ArtifactState.DECLARED,
        )
        record_artifact(
            producer="checkpoint",
            kind="checkpoint.directory",
            path="s3://bucket/declared",
            state=ArtifactState.RETIRED,
            artifact_id=declared_id,
        )
        complete_id = record_artifact(
            producer="checkpoint",
            kind="checkpoint.directory",
            path="s3://bucket/complete",
        )
        record_artifact(
            producer="checkpoint",
            kind="checkpoint.directory",
            path="s3://bucket/complete",
            state=ArtifactState.RETIRED,
            artifact_id=complete_id,
        )

    assert [
        row["state"] for row in read_artifact_rows(tmp_path, launcher_identity)
    ] == [
        "declared",
        "retired",
        "complete",
        "retired",
    ]


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


def test_event_context_includes_schema_and_fresh_clocks(tmp_path, launcher_identity):
    with build_evidence(tmp_path):
        first = event_context()
        second = event_context()

    assert first["evidence_schema_version"] == 1
    assert second["event_seq"] == first["event_seq"] + 1
    assert second["wall_time_ns"] >= first["wall_time_ns"]
    assert second["monotonic_ns"] >= first["monotonic_ns"]


def test_bind_distributed_records_public_axis_coordinates_and_unknown_cpu_uuid(
    tmp_path, launcher_identity
):
    class AxisMesh:
        def size(self):
            return 4

        def get_local_rank(self):
            return 2

    class PublicParallelDims:
        dp_replicate = 1
        dp_shard = 1
        cp = 1
        tp = 4
        pp = 1
        ep = 1
        world_size = 4

        def get_all_one_dimensional_meshes(self):
            return {"tp": AxisMesh()}

    with build_evidence(tmp_path) as evidence:
        evidence.bind_distributed(PublicParallelDims(), torch.device("cpu"))
        context = event_context()

    assert context["device_uuid"] is None
    assert context["mesh_axis_tp_rank"] == 2
    assert context["mesh_axis_tp_size"] == 4


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


def test_process_outcome_records_success(tmp_path, launcher_identity):
    with build_evidence(tmp_path):
        pass
    succeeded = json.loads(outcome_path(tmp_path, launcher_identity).read_text())
    assert succeeded["outcome"] == "succeeded"
    assert succeeded["elapsed_monotonic_ns"] >= 0


@pytest.mark.parametrize(
    ("error", "expected_outcome"),
    ((RuntimeError("training failed"), "failed"), (KeyboardInterrupt(), "interrupted")),
)
def test_process_outcome_records_failure_and_interruption(
    tmp_path, launcher_identity, error, expected_outcome
):
    process_id = f"trainer.{expected_outcome}.global_rank_000000"
    evidence = RunEvidence(
        RunEvidence.Config(),
        dump_folder=str(tmp_path),
        job_config={"training": {"steps": 3}},
        role="trainer",
        actor_id=expected_outcome,
    )

    with pytest.raises(type(error)):
        with evidence:
            raise error

    outcome = json.loads(
        outcome_path(tmp_path, launcher_identity, process_id).read_text()
    )
    assert outcome["outcome"] == expected_outcome
    assert outcome["exception_type"] == type(error).__name__


def test_process_outcome_is_immutable_after_publication(tmp_path, launcher_identity):
    with build_evidence(tmp_path) as evidence:
        pass

    original = outcome_path(tmp_path, launcher_identity).read_bytes()
    with pytest.raises(EvidenceCollisionError, match="outcome"):
        evidence._write_outcome(None, None)
    assert outcome_path(tmp_path, launcher_identity).read_bytes() == original


def test_outcome_write_failure_preserves_active_training_exception(
    tmp_path, launcher_identity
):
    evidence = build_evidence(tmp_path)
    with pytest.raises(RuntimeError, match="training failed"):
        with evidence:
            destination = outcome_path(tmp_path, launcher_identity)
            destination.parent.mkdir(parents=True)
            destination.write_text("existing outcome\n")
            raise RuntimeError("training failed")
