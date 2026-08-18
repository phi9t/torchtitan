# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import replace

from torchtitan.observability.state_estimator.bundle import (
    RunEvidenceBundle,
    load_bundle,
)
from torchtitan.observability.state_estimator.fixtures import (
    ProcessFixture,
    build_minimal_evidence_bundle,
)
from torchtitan.observability.state_estimator.observation import (
    ClockQuality,
    ObservationKind,
    normalize_bundle_observations,
)
from torchtitan.observability.state_estimator.schema import canonical_json


def _bundle_with_one_process(tmp_path) -> RunEvidenceBundle:
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="run-observation",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
                phases=[("train/step", 7, 1_000, 500)],
            )
        ],
    )
    return load_bundle(attempt_path)


def test_normalize_bundle_observations_preserves_typed_identity_clocks_and_sources(
    tmp_path,
):
    bundle = _bundle_with_one_process(tmp_path)
    bundle.event_rows[0]["sensor_id"] = "structured-events/trainer-0"
    bundle.event_rows[0]["ingestion_time_ns"] = 1_100
    bundle.artifact_rows[0]["sensor_id"] = "artifact-index/trainer-0"
    bundle.artifact_rows[0]["ingestion_time_ns"] = 20
    bundle.outcome_rows[0]["sensor_id"] = "process-outcome/trainer-0"
    bundle.outcome_rows[0]["event_time_ns"] = 2_000
    bundle.outcome_rows[0]["ingestion_time_ns"] = 2_100

    result = normalize_bundle_observations(bundle)

    assert [observation.kind for observation in result.observations] == [
        ObservationKind.STRUCTURED_EVENT,
        ObservationKind.ARTIFACT,
        ObservationKind.PROCESS_OUTCOME,
    ]
    event = result.observations[0]
    assert event.envelope.record_type == "structured_event"
    assert event.envelope.process_id == "trainer.core.global_rank_000000"
    assert event.envelope.role == "trainer"
    assert event.envelope.actor_id == "core"
    assert event.envelope.host_name == "fixture-host"
    assert event.envelope.pid == 1000
    assert event.envelope.global_rank == 0
    assert event.envelope.local_rank == 0
    assert event.envelope.world_size == 1
    assert event.envelope.event_time_ns == 1_000
    assert event.envelope.ingestion_time_ns == 1_100
    assert event.envelope.monotonic_ns == 500
    assert event.envelope.event_seq == 0
    assert event.envelope.artifact_seq is None
    assert event.envelope.sensor_id == "structured-events/trainer-0"
    assert event.envelope.clock_quality == ClockQuality.EVENT_AND_MONOTONIC
    assert event.envelope.source_path.endswith(
        "structured_logs/trainer.core.global_rank_000000.jsonl"
    )
    assert event.envelope.raw_record_id.endswith(":1")
    assert event.raw_record is bundle.event_rows[0]
    assert event.payload["phase"] == "train/step"

    artifact = result.observations[1]
    assert artifact.envelope.record_type == "artifact"
    assert artifact.envelope.event_time_ns == 1
    assert artifact.envelope.ingestion_time_ns == 20
    assert artifact.envelope.monotonic_ns == 1
    assert artifact.envelope.event_seq is None
    assert artifact.envelope.artifact_seq == 0
    assert artifact.envelope.source_path.endswith(
        "indexes/artifacts.trainer.core.global_rank_000000.jsonl"
    )

    outcome = result.observations[2]
    assert outcome.envelope.record_type == "process_outcome"
    assert outcome.envelope.event_time_ns == 2_000
    assert outcome.envelope.ingestion_time_ns == 2_100
    assert outcome.envelope.source_path.endswith(
        "processes/trainer.core.global_rank_000000/outcome.json"
    )


def test_normalize_bundle_observations_reports_missing_sensor_identity(tmp_path):
    bundle = _bundle_with_one_process(tmp_path)

    result = normalize_bundle_observations(bundle)

    assert any(
        finding.kind == "missing_sensor_id"
        and finding.evidence_state == "uncollected"
        and finding.entity is not None
        and finding.entity.id == "trainer.core.global_rank_000000"
        for finding in result.quality_findings
    )


def test_normalize_bundle_observations_reports_missing_clocks(tmp_path):
    bundle = _bundle_with_one_process(tmp_path)
    event_without_clocks = {
        key: value
        for key, value in bundle.event_rows[0].items()
        if key not in {"wall_time_ns", "event_time_ns", "monotonic_ns"}
    }
    event_without_clocks["sensor_id"] = "structured-events/trainer-0"
    bundle = replace(bundle, event_rows=[event_without_clocks])

    result = normalize_bundle_observations(bundle)

    assert result.observations[0].envelope.clock_quality == ClockQuality.MISSING
    assert any(
        finding.kind == "missing_clocks"
        and finding.evidence_state == "unknown"
        and finding.entity is not None
        and finding.entity.id == "trainer.core.global_rank_000000"
        for finding in result.quality_findings
    )


def test_normalize_bundle_observations_reports_malformed_identity_fields(tmp_path):
    bundle = _bundle_with_one_process(tmp_path)
    malformed_event = dict(bundle.event_rows[0])
    malformed_event.update(
        {
            "sensor_id": "structured-events/trainer-0",
            "process_id": 0,
            "global_rank": "zero",
            "local_rank": "zero",
            "world_size": "one",
            "pid": "1000",
        }
    )
    bundle = replace(bundle, event_rows=[malformed_event])

    result = normalize_bundle_observations(bundle)

    assert result.observations[0].envelope.process_id is None
    assert result.observations[0].envelope.global_rank is None
    assert result.observations[0].envelope.local_rank is None
    assert result.observations[0].envelope.world_size is None
    assert result.observations[0].envelope.pid is None
    malformed_fields = {
        finding.metadata["field"]
        for finding in result.quality_findings
        if finding.kind == "malformed_identity_field"
    }
    assert malformed_fields == {
        "process_id",
        "global_rank",
        "local_rank",
        "world_size",
        "pid",
    }


def test_normalize_bundle_observations_reports_malformed_fields_in_stable_order(
    tmp_path,
):
    bundle = _bundle_with_one_process(tmp_path)
    malformed_event = dict(bundle.event_rows[0])
    malformed_event.update(
        {
            "process_id": 0,
            "role": 1,
            "actor_id": 2,
            "host_name": 3,
            "device_uuid": 4,
            "phase": 5,
            "sensor_id": 6,
            "pid": "1000",
            "global_rank": "zero",
            "local_rank": "zero",
            "world_size": "one",
            "device_index": "zero",
            "step": "seven",
            "ingestion_time_ns": "1100",
            "monotonic_ns": "500",
            "event_seq": "0",
            "artifact_seq": "0",
        }
    )
    bundle = replace(bundle, event_rows=[malformed_event])

    result = normalize_bundle_observations(bundle)

    malformed_fields = [
        finding.metadata["field"]
        for finding in result.quality_findings
        if finding.kind == "malformed_identity_field"
    ]
    assert malformed_fields == [
        "process_id",
        "role",
        "actor_id",
        "host_name",
        "device_uuid",
        "phase",
        "sensor_id",
        "pid",
        "global_rank",
        "local_rank",
        "world_size",
        "device_index",
        "step",
        "ingestion_time_ns",
        "monotonic_ns",
        "event_seq",
        "artifact_seq",
    ]


def test_normalize_bundle_observations_reports_one_malformed_event_time_finding(
    tmp_path,
):
    bundle = _bundle_with_one_process(tmp_path)
    malformed_event = dict(bundle.event_rows[0])
    malformed_event.update(
        {
            "sensor_id": "structured-events/trainer-0",
            "event_time_ns": "1000",
        }
    )
    bundle = replace(bundle, event_rows=[malformed_event])

    result = normalize_bundle_observations(bundle)

    malformed_event_time_findings = [
        finding
        for finding in result.quality_findings
        if finding.kind == "malformed_identity_field"
        and finding.metadata["field"] == "event_time_ns"
    ]
    assert len(malformed_event_time_findings) == 1


def test_normalize_bundle_observations_preserves_non_object_raw_record_quality(
    tmp_path,
):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="run-observation",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
                phases=[],
            )
        ],
    )
    bad_log = attempt_path / "structured_logs" / "bad.jsonl"
    bad_log.write_text(canonical_json(["not", "an", "object"]) + "\n")
    bundle = load_bundle(attempt_path)

    result = normalize_bundle_observations(bundle)

    assert any(
        finding.kind == "non_object_jsonl"
        and finding.evidence_state == "malformed"
        and finding.source_path == str(bad_log)
        for finding in result.quality_findings
    )
