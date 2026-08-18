# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Normalize raw run-evidence records into typed observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from torchtitan.observability.state_estimator.bundle import RunEvidenceBundle
from torchtitan.observability.state_estimator.schema import (
    EntityRef,
    EvidenceState,
    QualityFinding,
    SCHEMA_VERSION,
)


class ObservationKind(StrEnum):
    STRUCTURED_EVENT = "structured_event"
    ARTIFACT = "artifact"
    PROCESS_OUTCOME = "process_outcome"


class ClockQuality(StrEnum):
    EVENT_AND_MONOTONIC = "event_and_monotonic"
    EVENT_ONLY = "event_only"
    MONOTONIC_ONLY = "monotonic_only"
    INGESTION_ONLY = "ingestion_only"
    MISSING = "missing"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class ObservationEnvelope:
    run_id: str | None
    attempt_id: str | None
    schema_version: int | None
    record_type: str
    source_path: str
    raw_record_id: str
    process_id: str | None = None
    role: str | None = None
    actor_id: str | None = None
    host_name: str | None = None
    pid: int | None = None
    global_rank: int | None = None
    local_rank: int | None = None
    world_size: int | None = None
    device_index: int | None = None
    device_uuid: str | None = None
    step: int | None = None
    phase: str | None = None
    event_time_ns: int | None = None
    ingestion_time_ns: int | None = None
    monotonic_ns: int | None = None
    event_seq: int | None = None
    artifact_seq: int | None = None
    sensor_id: str | None = None
    clock_quality: ClockQuality = ClockQuality.MISSING
    quality: EvidenceState = "present"

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "record_schema_version": self.schema_version,
            "record_type": self.record_type,
            "source_path": self.source_path,
            "raw_record_id": self.raw_record_id,
            "process_id": self.process_id,
            "role": self.role,
            "actor_id": self.actor_id,
            "host_name": self.host_name,
            "pid": self.pid,
            "global_rank": self.global_rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "device_index": self.device_index,
            "device_uuid": self.device_uuid,
            "step": self.step,
            "phase": self.phase,
            "event_time_ns": self.event_time_ns,
            "ingestion_time_ns": self.ingestion_time_ns,
            "monotonic_ns": self.monotonic_ns,
            "event_seq": self.event_seq,
            "artifact_seq": self.artifact_seq,
            "sensor_id": self.sensor_id,
            "clock_quality": self.clock_quality.value,
            "quality": self.quality,
        }


@dataclass(frozen=True, slots=True)
class NormalizedObservation:
    id: str
    kind: ObservationKind
    envelope: ObservationEnvelope
    payload: Mapping[str, Any] = field(default_factory=dict)
    raw_record: Mapping[str, Any] = field(default_factory=dict)
    quality_findings: tuple[QualityFinding, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "kind": self.kind.value,
            "envelope": self.envelope.to_json(),
            "payload": dict(self.payload),
            "quality_findings": [
                finding.to_json() for finding in self.quality_findings
            ],
        }


@dataclass(frozen=True, slots=True)
class ObservationNormalizationResult:
    observations: list[NormalizedObservation]
    quality_findings: list[QualityFinding]


_STRING_IDENTITY_FIELDS = (
    "process_id",
    "role",
    "actor_id",
    "host_name",
    "device_uuid",
    "phase",
    "sensor_id",
)
_INT_IDENTITY_FIELDS = (
    "pid",
    "global_rank",
    "local_rank",
    "world_size",
    "device_index",
    "step",
)
_INT_CLOCK_FIELDS = (
    "event_time_ns",
    "ingestion_time_ns",
    "monotonic_ns",
    "event_seq",
    "artifact_seq",
)
_GENERIC_INT_FIELDS = _INT_IDENTITY_FIELDS + (
    "ingestion_time_ns",
    "monotonic_ns",
    "event_seq",
    "artifact_seq",
)
_CLOCK_FIELD_NAMES = frozenset(_INT_CLOCK_FIELDS + ("wall_time_ns",))


def normalize_bundle_observations(
    bundle: RunEvidenceBundle,
) -> ObservationNormalizationResult:
    observations: list[NormalizedObservation] = []
    quality_findings = list(bundle.quality_findings)

    for index, row in enumerate(bundle.event_rows):
        observation, findings = _normalize_row(
            bundle,
            row,
            kind=ObservationKind.STRUCTURED_EVENT,
            default_record_type="structured_event",
            fallback_source_path=f"structured_logs/event_rows/{index}",
            fallback_raw_record_id=f"event_rows:{index}",
        )
        observations.append(observation)
        quality_findings.extend(findings)

    for index, row in enumerate(bundle.artifact_rows):
        observation, findings = _normalize_row(
            bundle,
            row,
            kind=ObservationKind.ARTIFACT,
            default_record_type="artifact",
            fallback_source_path=f"indexes/artifact_rows/{index}",
            fallback_raw_record_id=f"artifact_rows:{index}",
        )
        observations.append(observation)
        quality_findings.extend(findings)

    for index, row in enumerate(bundle.outcome_rows):
        observation, findings = _normalize_row(
            bundle,
            row,
            kind=ObservationKind.PROCESS_OUTCOME,
            default_record_type="process_outcome",
            fallback_source_path=f"processes/outcome_rows/{index}",
            fallback_raw_record_id=f"outcome_rows:{index}",
        )
        observations.append(observation)
        quality_findings.extend(findings)

    return ObservationNormalizationResult(
        observations=observations,
        quality_findings=quality_findings,
    )


def _normalize_row(
    bundle: RunEvidenceBundle,
    row: Mapping[str, Any],
    *,
    kind: ObservationKind,
    default_record_type: str,
    fallback_source_path: str,
    fallback_raw_record_id: str,
) -> tuple[NormalizedObservation, list[QualityFinding]]:
    source_path = bundle.row_source_paths.get(id(row), fallback_source_path)
    raw_record_id = bundle.row_record_ids.get(id(row), fallback_raw_record_id)
    record_type = _read_string(row, "record_type") or default_record_type
    process_id = _read_string(row, "process_id")
    entity = EntityRef(kind="process", id=process_id) if process_id else None
    findings: list[QualityFinding] = []

    for field_name in _STRING_IDENTITY_FIELDS:
        if (
            _read_string(row, field_name) is None
            and field_name in row
            and row[field_name] is not None
        ):
            findings.append(
                _malformed_identity_field(
                    field_name,
                    row[field_name],
                    source_path=source_path,
                    entity=entity,
                )
            )
    for field_name in _GENERIC_INT_FIELDS:
        if (
            _read_int(row, field_name) is None
            and field_name in row
            and row[field_name] is not None
        ):
            findings.append(
                _malformed_identity_field(
                    field_name,
                    row[field_name],
                    source_path=source_path,
                    entity=entity,
                )
            )

    event_time_ns = _event_time_ns(row, findings, source_path=source_path, entity=entity)
    ingestion_time_ns = _read_int(row, "ingestion_time_ns")
    monotonic_ns = _read_int(row, "monotonic_ns")
    clock_quality = _clock_quality(
        event_time_ns=event_time_ns,
        ingestion_time_ns=ingestion_time_ns,
        monotonic_ns=monotonic_ns,
        malformed_clock=any(
            finding.kind == "malformed_identity_field"
            and finding.metadata.get("field") in _CLOCK_FIELD_NAMES
            for finding in findings
        ),
    )
    sensor_id = _read_string(row, "sensor_id")
    if sensor_id is None:
        findings.append(
            QualityFinding(
                kind="missing_sensor_id",
                severity="warning",
                evidence_state="uncollected",
                message=f"{record_type} record has no sensor_id",
                source_path=source_path,
                entity=entity,
                metadata={"record_type": record_type, "raw_record_id": raw_record_id},
            )
        )
    if clock_quality == ClockQuality.MISSING:
        findings.append(
            QualityFinding(
                kind="missing_clocks",
                severity="warning",
                evidence_state="unknown",
                message=f"{record_type} record has no event, ingestion, or monotonic clock",
                source_path=source_path,
                entity=entity,
                metadata={"record_type": record_type, "raw_record_id": raw_record_id},
            )
        )

    envelope = ObservationEnvelope(
        run_id=_read_string(row, "run_id") or _read_string(bundle.manifest, "run_id"),
        attempt_id=_read_string(row, "attempt_id")
        or _read_string(bundle.manifest, "attempt_id"),
        schema_version=_read_int(row, "schema_version")
        or _read_int(row, "evidence_schema_version"),
        record_type=record_type,
        source_path=source_path,
        raw_record_id=raw_record_id,
        process_id=process_id,
        role=_read_string(row, "role"),
        actor_id=_read_string(row, "actor_id"),
        host_name=_read_string(row, "host_name"),
        pid=_read_int(row, "pid"),
        global_rank=_read_int(row, "global_rank"),
        local_rank=_read_int(row, "local_rank"),
        world_size=_read_int(row, "world_size"),
        device_index=_read_int(row, "device_index"),
        device_uuid=_read_string(row, "device_uuid"),
        step=_read_int(row, "step"),
        phase=_read_string(row, "phase"),
        event_time_ns=event_time_ns,
        ingestion_time_ns=ingestion_time_ns,
        monotonic_ns=monotonic_ns,
        event_seq=_read_int(row, "event_seq"),
        artifact_seq=_read_int(row, "artifact_seq"),
        sensor_id=sensor_id,
        clock_quality=clock_quality,
        quality=(
            "malformed"
            if any(f.evidence_state == "malformed" for f in findings)
            else "present"
        ),
    )
    observation_id = _observation_id(kind, envelope)
    return (
        NormalizedObservation(
            id=observation_id,
            kind=kind,
            envelope=envelope,
            payload=dict(row),
            raw_record=row,
            quality_findings=tuple(findings),
        ),
        findings,
    )


def _observation_id(kind: ObservationKind, envelope: ObservationEnvelope) -> str:
    process_part = envelope.process_id or "unknown_process"
    sequence = envelope.event_seq if kind == ObservationKind.STRUCTURED_EVENT else None
    if kind == ObservationKind.ARTIFACT:
        sequence = envelope.artifact_seq
    if sequence is None:
        sequence = envelope.raw_record_id
    return f"{kind.value}:{process_part}:{sequence}"


def _event_time_ns(
    row: Mapping[str, Any],
    findings: list[QualityFinding],
    *,
    source_path: str,
    entity: EntityRef | None,
) -> int | None:
    if "event_time_ns" in row:
        event_time_ns = _read_int(row, "event_time_ns")
        if event_time_ns is None and row["event_time_ns"] is not None:
            findings.append(
                _malformed_identity_field(
                    "event_time_ns",
                    row["event_time_ns"],
                    source_path=source_path,
                    entity=entity,
                )
            )
        return event_time_ns
    wall_time_ns = _read_int(row, "wall_time_ns")
    if wall_time_ns is None and "wall_time_ns" in row and row["wall_time_ns"] is not None:
        findings.append(
            _malformed_identity_field(
                "wall_time_ns",
                row["wall_time_ns"],
                source_path=source_path,
                entity=entity,
            )
        )
    return wall_time_ns


def _clock_quality(
    *,
    event_time_ns: int | None,
    ingestion_time_ns: int | None,
    monotonic_ns: int | None,
    malformed_clock: bool,
) -> ClockQuality:
    if malformed_clock:
        return ClockQuality.MALFORMED
    if event_time_ns is not None and monotonic_ns is not None:
        return ClockQuality.EVENT_AND_MONOTONIC
    if event_time_ns is not None:
        return ClockQuality.EVENT_ONLY
    if monotonic_ns is not None:
        return ClockQuality.MONOTONIC_ONLY
    if ingestion_time_ns is not None:
        return ClockQuality.INGESTION_ONLY
    return ClockQuality.MISSING


def _read_string(row: Mapping[str, Any], field_name: str) -> str | None:
    value = row.get(field_name)
    return value if isinstance(value, str) else None


def _read_int(row: Mapping[str, Any], field_name: str) -> int | None:
    value = row.get(field_name)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _malformed_identity_field(
    field_name: str,
    value: Any,
    *,
    source_path: str,
    entity: EntityRef | None,
) -> QualityFinding:
    return QualityFinding(
        kind="malformed_identity_field",
        severity="warning",
        evidence_state="malformed",
        message=f"{field_name} has malformed value {value!r}",
        source_path=source_path,
        entity=entity,
        metadata={"field": field_name, "value_type": type(value).__name__},
    )
