# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dependency-light learned-feature segment extraction."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from torchtitan.observability.state_estimator.learned import TraceSegment
from torchtitan.observability.state_estimator.observation import (
    NormalizedObservation,
    ObservationEnvelope,
)


_MODALITY_RECORD_TYPES = {
    "scalar_metric": "scalar",
    "metric": "scalar",
    "scalar": "scalar",
    "structured_event": "event",
    "event": "event",
    "trace_summary": "trace_summary",
    "trace": "trace_summary",
    "communication": "communication",
    "collective": "communication",
    "hardware": "hardware",
    "dcgm": "hardware",
    "framework": "framework",
    "framework_state": "framework",
    "transaction": "transaction",
    "checkpoint_transaction": "transaction",
}
_MODALITY_HINT_FIELDS = {
    "metric_name": "scalar",
    "event_name": "event",
    "trace_id": "trace_summary",
    "collective": "communication",
    "gpu_utilization": "hardware",
    "hbm_used_bytes": "hardware",
    "component": "framework",
    "committed_step": "transaction",
    "speculative_step": "transaction",
}
_SCALAR_VALUE_TYPES = (str, int, float, bool, type(None))


def extract_feature_segments(
    observations: Sequence[NormalizedObservation | Mapping[str, Any]],
) -> tuple[TraceSegment, ...]:
    """Extract advisory learned-feature segments from normalized observations."""

    segments: list[TraceSegment] = []
    for observation in observations:
        row = _observation_row(observation)
        modality = _modality(row["record_type"], row["payload"])
        if modality is None:
            continue
        envelope = row["envelope"]
        features = _feature_values(
            row["payload"],
            envelope=envelope,
            observation_id=row["id"],
            record_type=row["record_type"],
        )
        segments.append(
            TraceSegment(
                id=_segment_id(modality, envelope, row["id"]),
                entity=_entity(envelope),
                modality=modality,
                start_time_ns=envelope.event_time_ns,
                end_time_ns=envelope.event_time_ns,
                features=features,
                source_observation_ids=(row["id"],),
            )
        )
    return tuple(segments)


def _observation_row(
    observation: NormalizedObservation | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(observation, NormalizedObservation):
        return {
            "id": observation.id,
            "record_type": observation.envelope.record_type,
            "envelope": observation.envelope,
            "payload": dict(observation.payload),
        }

    envelope = observation.get("envelope")
    if isinstance(envelope, Mapping):
        record_type = str(envelope.get("record_type") or observation.get("kind") or "")
        observation_id = str(observation.get("id") or "unknown_observation")
        payload = observation.get("payload", {})
        return {
            "id": observation_id,
            "record_type": record_type,
            "envelope": _mapping_envelope(
                envelope,
                record_type=record_type,
                source_path=str(observation.get("source_path") or ""),
                raw_record_id=observation_id,
            ),
            "payload": dict(payload) if isinstance(payload, Mapping) else {},
        }

    payload = observation.get("payload", {})
    record_type = str(observation.get("kind") or "")
    observation_id = str(observation.get("id") or "unknown_observation")
    return {
        "id": observation_id,
        "record_type": record_type,
        "envelope": _mapping_envelope(
            observation,
            record_type=record_type,
            source_path=str(observation.get("source_path") or ""),
            raw_record_id=observation_id,
        ),
        "payload": dict(payload) if isinstance(payload, Mapping) else {},
    }


def _mapping_envelope(
    mapping: Mapping[str, Any],
    *,
    record_type: str,
    source_path: str,
    raw_record_id: str,
) -> ObservationEnvelope:
    entity = mapping.get("entity")
    process_id = None
    if isinstance(entity, Mapping) and isinstance(entity.get("id"), str):
        process_id = entity["id"]
    return ObservationEnvelope(
        run_id=_string(mapping.get("run_id")),
        attempt_id=_string(mapping.get("attempt_id")),
        schema_version=_int(mapping.get("record_schema_version"))
        or _int(mapping.get("schema_version")),
        record_type=record_type,
        source_path=_string(mapping.get("source_path")) or source_path,
        raw_record_id=raw_record_id,
        process_id=_string(mapping.get("process_id")) or process_id,
        role=_string(mapping.get("role")),
        actor_id=_string(mapping.get("actor_id")),
        host_name=_string(mapping.get("host_name")),
        pid=_int(mapping.get("pid")),
        global_rank=_int(mapping.get("global_rank")),
        local_rank=_int(mapping.get("local_rank")),
        world_size=_int(mapping.get("world_size")),
        device_index=_int(mapping.get("device_index")),
        device_uuid=_string(mapping.get("device_uuid")),
        step=_int(mapping.get("step")),
        phase=_string(mapping.get("phase")),
        event_time_ns=_int(mapping.get("event_time_ns")),
        ingestion_time_ns=_int(mapping.get("ingestion_time_ns")),
        monotonic_ns=_int(mapping.get("monotonic_ns")),
        event_seq=_int(mapping.get("event_seq")),
        artifact_seq=_int(mapping.get("artifact_seq")),
        sensor_id=_string(mapping.get("sensor_id")),
    )


def _modality(record_type: str, payload: Mapping[str, Any]) -> str | None:
    normalized_record_type = record_type.strip().lower()
    if normalized_record_type in _MODALITY_RECORD_TYPES:
        return _MODALITY_RECORD_TYPES[normalized_record_type]
    for field_name, modality in _MODALITY_HINT_FIELDS.items():
        if field_name in payload:
            return modality
    return None


def _feature_values(
    payload: Mapping[str, Any],
    *,
    envelope: ObservationEnvelope,
    observation_id: str,
    record_type: str,
) -> dict[str, Any]:
    features = {
        key: _stable_value(value)
        for key, value in sorted(payload.items())
        if _is_feature_value(value)
    }
    features.update(
        {
            "observation_id": observation_id,
            "record_type": record_type,
            "process_id": envelope.process_id,
            "role": envelope.role,
            "actor_id": envelope.actor_id,
            "host_name": envelope.host_name,
            "global_rank": envelope.global_rank,
            "local_rank": envelope.local_rank,
            "world_size": envelope.world_size,
            "device_index": envelope.device_index,
            "device_uuid": envelope.device_uuid,
            "step": envelope.step,
            "phase": envelope.phase,
            "sensor_id": envelope.sensor_id,
            "clock_quality": envelope.clock_quality.value,
        }
    )
    return features


def _segment_id(
    modality: str, envelope: ObservationEnvelope, observation_id: str
) -> str:
    entity_id = envelope.process_id or envelope.actor_id or envelope.host_name or "unknown"
    step = envelope.step if envelope.step is not None else "unknown_step"
    return f"{modality}:{entity_id}:{step}:{observation_id}"


def _entity(envelope: ObservationEnvelope) -> dict[str, str]:
    if envelope.process_id:
        return {"kind": "process", "id": envelope.process_id}
    if envelope.actor_id:
        return {"kind": "actor", "id": envelope.actor_id}
    if envelope.host_name:
        return {"kind": "host", "id": envelope.host_name}
    return {"kind": "unknown", "id": "unknown"}


def _stable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _stable_value(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    if isinstance(value, tuple):
        return [_stable_value(item) for item in value]
    return value


def _is_feature_value(value: Any) -> bool:
    if isinstance(value, _SCALAR_VALUE_TYPES):
        return True
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _is_feature_value(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_is_feature_value(child) for child in value)
    return False


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
