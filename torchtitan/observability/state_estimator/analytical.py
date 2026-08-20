# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic analytical execution summaries for offline evidence graphs."""

from __future__ import annotations

from collections import defaultdict

from collections.abc import Mapping, Sequence
from typing import Any

from torchtitan.observability.state_estimator.schema import SCHEMA_VERSION


_ADVISORY_RESIDUAL = (
    "Residual is advisory and indicates model mismatch or missing factors; "
    "it does not identify a root cause by itself."
)


def build_analytical_summary(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Build advisory analytical summaries from already-normalized evidence."""

    observations = [
        observation
        for observation in graph.get("observations", [])
        if isinstance(observation, Mapping)
    ]
    phase_durations = _phase_duration_summaries(observations)
    checkpoint_durations = _checkpoint_duration_summaries(observations)
    collective_predictions, residuals = _collective_predictions_and_residuals(
        observations
    )
    collective_symptoms = _collective_symptoms(observations)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": graph.get("run_id"),
        "attempt_id": graph.get("attempt_id"),
        "claim_calibration": "heuristic",
        "phase_durations": phase_durations,
        "max_plus_predecessors": _max_plus_predecessors(observations),
        "collective_predictions": collective_predictions,
        "collective_symptoms": collective_symptoms,
        "checkpoint_durations": checkpoint_durations,
        "residuals": residuals,
    }


def predict_ring_all_reduce_duration(
    *,
    message_bytes: int,
    group_size: int,
    latency_ns: int,
    effective_bandwidth_bytes_per_ns: int | float,
) -> dict[str, Any]:
    """Predict ring all-reduce duration from explicit workload metadata."""

    _validate_non_negative("message_bytes", message_bytes)
    _validate_positive("group_size", group_size)
    _validate_non_negative("latency_ns", latency_ns)
    _validate_positive(
        "effective_bandwidth_bytes_per_ns", effective_bandwidth_bytes_per_ns
    )
    if group_size == 1:
        predicted_duration_ns = latency_ns
    else:
        transfer_bytes = 2 * (group_size - 1) / group_size * message_bytes
        transfer_ns = transfer_bytes / effective_bandwidth_bytes_per_ns
        latency_component_ns = 2 * (group_size - 1) * latency_ns
        predicted_duration_ns = latency_component_ns + transfer_ns
    return {
        "model": "ring_all_reduce",
        "message_bytes": message_bytes,
        "group_size": group_size,
        "latency_ns": latency_ns,
        "effective_bandwidth_bytes_per_ns": effective_bandwidth_bytes_per_ns,
        "predicted_duration_ns": _stable_number(predicted_duration_ns),
        "calibration": "heuristic",
        "quality": "present",
    }


def _phase_duration_summaries(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for observation in observations:
        if observation.get("kind") != "structured_event":
            continue
        payload = _payload(observation)
        duration = _duration_from_payload(payload, observation)
        if duration is None:
            continue
        summaries.append(duration)
    return sorted(summaries, key=_duration_sort_key)


def _checkpoint_duration_summaries(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for observation in observations:
        if observation.get("kind") != "artifact":
            continue
        payload = _payload(observation)
        if not _is_checkpoint_payload(payload):
            continue
        duration = _duration_from_payload(payload, observation)
        if duration is None:
            continue
        operation = _checkpoint_operation(payload)
        if operation is None:
            continue
        checkpoint_id = payload.get("checkpoint_id") or payload.get("artifact_id")
        checkpoint_summary = dict(duration)
        checkpoint_summary["checkpoint_operation"] = operation
        checkpoint_summary["checkpoint_id"] = (
            checkpoint_id if isinstance(checkpoint_id, str) else None
        )
        summaries.append(checkpoint_summary)
    return sorted(
        summaries,
        key=lambda item: (
            str(item.get("checkpoint_operation", "")),
            item.get("step") if item.get("step") is not None else -1,
            str(item.get("process", {}).get("id", "")),
            str(item.get("checkpoint_id", "")),
        ),
    )


def _max_plus_predecessors(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    for observation in observations:
        payload = _payload(observation)
        predecessor_phase = payload.get("predecessor_phase")
        predecessor_step = payload.get("predecessor_step")
        phase = payload.get("phase")
        step = payload.get("step")
        if (
            not isinstance(predecessor_phase, str)
            or not isinstance(predecessor_step, int)
            or not isinstance(phase, str)
            or not isinstance(step, int)
        ):
            continue
        relationships.append(
            {
                "process": _entity(observation),
                "global_rank": payload.get("global_rank"),
                "phase": phase,
                "step": step,
                "predecessor_phase": predecessor_phase,
                "predecessor_step": predecessor_step,
                "source_evidence": [_observation_id(observation)],
                "calibration": "heuristic",
                "quality": "present",
            }
        )
    return sorted(
        relationships,
        key=lambda item: (
            str(item.get("process", {}).get("id", "")),
            item["step"],
            item["phase"],
            item["predecessor_step"],
            item["predecessor_phase"],
        ),
    )


def _collective_predictions_and_residuals(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    residuals: list[dict[str, Any]] = []
    for observation in observations:
        if observation.get("kind") != "structured_event":
            continue
        payload = _payload(observation)
        if not _is_all_reduce_payload(payload):
            continue
        missing_fields = _missing_collective_prediction_fields(payload)
        if missing_fields:
            residuals.append(
                {
                    "kind": "collective_prediction_unavailable",
                    "status": "metadata_missing",
                    "quality": "uncollected",
                    "calibration": "heuristic",
                    "advisory": (
                        "No analytical all-reduce duration is predicted because "
                        "workload metadata is incomplete."
                    ),
                    "source_evidence": [_observation_id(observation)],
                    "metadata": {
                        "missing_fields": missing_fields,
                        "phase": payload.get("phase"),
                        "step": payload.get("step"),
                    },
                }
            )
            continue
        prediction = predict_ring_all_reduce_duration(
            message_bytes=payload["message_bytes"],
            group_size=payload["group_size"],
            latency_ns=payload["latency_ns"],
            effective_bandwidth_bytes_per_ns=payload[
                "effective_bandwidth_bytes_per_ns"
            ],
        )
        prediction = {
            **prediction,
            "phase": payload.get("phase"),
            "step": payload.get("step"),
            "process": _entity(observation),
            "global_rank": payload.get("global_rank"),
            "source_evidence": [_observation_id(observation)],
        }
        predictions.append(prediction)
        observed_duration_ns = _collective_observed_duration_ns(payload)
        if observed_duration_ns is None:
            continue
        predicted_duration_ns = _number_or_none(prediction.get("predicted_duration_ns"))
        if predicted_duration_ns is None:
            continue
        sigma = _residual_sigma_ns(payload, predicted_duration_ns)
        expected_sigma_ns = _number_or_none(sigma.get("expected_sigma_ns"))
        if expected_sigma_ns is None:
            continue
        residuals.append(
            {
                "kind": "collective_duration_residual",
                "status": "advisory",
                "quality": "present",
                "calibration": "heuristic",
                "phase": payload.get("phase"),
                "step": payload.get("step"),
                "process": _entity(observation),
                "global_rank": payload.get("global_rank"),
                "observed_duration_ns": observed_duration_ns,
                "predicted_duration_ns": predicted_duration_ns,
                "expected_sigma_ns": expected_sigma_ns,
                "sigma_calibration": sigma["sigma_calibration"],
                "normalized_residual": _normalized_residual(
                    observed_duration_ns,
                    predicted_duration_ns,
                    expected_sigma_ns,
                ),
                "advisory": _ADVISORY_RESIDUAL,
                "source_evidence": [_observation_id(observation)],
            }
        )
    return (
        sorted(predictions, key=_collective_sort_key),
        sorted(residuals, key=_residual_sort_key),
    )


def _collective_symptoms(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int | None], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for observation in observations:
        payload = _payload(observation)
        if not _is_all_reduce_payload(payload):
            continue
        phase = payload.get("phase")
        step = payload.get("step")
        group_size = payload.get("group_size")
        if isinstance(phase, str) and isinstance(step, int):
            groups[
                (phase, step, group_size if isinstance(group_size, int) else None)
            ].append(observation)

    symptoms: list[dict[str, Any]] = []
    for (phase, step, group_size), rows in sorted(groups.items()):
        launches: list[int] = []
        completions: list[int] = []
        evidence: list[str] = []
        for row in rows:
            payload = _payload(row)
            launch_time_ns = payload.get("launch_time_ns")
            completion_time_ns = payload.get("completion_time_ns")
            if not isinstance(launch_time_ns, int) or not isinstance(
                completion_time_ns, int
            ):
                continue
            launches.append(launch_time_ns)
            completions.append(completion_time_ns)
            evidence.append(_observation_id(row))
        if not launches or not completions:
            continue
        symptoms.append(
            {
                "phase": phase,
                "step": step,
                "group_size": group_size,
                "arrival_skew_ns": max(launches) - min(launches),
                "network_progress_ns": max(completions) - max(launches),
                "launch_clock": "time_ns",
                "completion_clock": "time_ns",
                "source_evidence": sorted(evidence),
                "calibration": "heuristic",
                "quality": "present",
            }
        )
    return symptoms


def _duration_from_payload(
    payload: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any] | None:
    phase = payload.get("phase")
    step = payload.get("step")
    if not isinstance(phase, str) or not isinstance(step, int):
        return None
    duration_ns = payload.get("duration_ns")
    clock = "duration_ns"
    start_time_ns = None
    end_time_ns = None
    if not isinstance(duration_ns, int):
        start_time_ns = payload.get("start_time_ns")
        end_time_ns = payload.get("end_time_ns")
        if not isinstance(start_time_ns, int) or not isinstance(end_time_ns, int):
            return None
        if end_time_ns < start_time_ns:
            return None
        duration_ns = end_time_ns - start_time_ns
        clock_value = payload.get("clock")
        clock = clock_value if isinstance(clock_value, str) else "time_ns"
    return {
        "process": _entity(observation),
        "global_rank": payload.get("global_rank"),
        "phase": phase,
        "step": step,
        "duration_ns": duration_ns,
        "clock": clock,
        "start_time_ns": start_time_ns,
        "end_time_ns": end_time_ns,
        "source_evidence": [_observation_id(observation)],
        "calibration": "heuristic",
        "quality": "present",
    }


def _missing_collective_prediction_fields(payload: Mapping[str, Any]) -> list[str]:
    missing_fields: list[str] = []
    for field in ("message_bytes", "group_size", "latency_ns"):
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            missing_fields.append(field)
    bandwidth = payload.get("effective_bandwidth_bytes_per_ns")
    if (
        isinstance(bandwidth, bool)
        or not isinstance(bandwidth, (int, float))
        or bandwidth <= 0
    ):
        missing_fields.append("effective_bandwidth_bytes_per_ns")
    return missing_fields


def _collective_observed_duration_ns(payload: Mapping[str, Any]) -> int | None:
    duration_ns = payload.get("duration_ns")
    if isinstance(duration_ns, int):
        return duration_ns
    launch_time_ns = payload.get("launch_time_ns")
    completion_time_ns = payload.get("completion_time_ns")
    if not isinstance(launch_time_ns, int) or not isinstance(completion_time_ns, int):
        return None
    if completion_time_ns < launch_time_ns:
        return None
    return completion_time_ns - launch_time_ns


def _normalized_residual(
    observed_duration_ns: int,
    predicted_duration_ns: int | float,
    expected_sigma_ns: int | float,
) -> float:
    return (observed_duration_ns - predicted_duration_ns) / expected_sigma_ns


def _residual_sigma_ns(
    payload: Mapping[str, Any], predicted_duration_ns: int | float
) -> dict[str, int | float | str]:
    workload_sigma = payload.get("expected_sigma_ns")
    if (
        not isinstance(workload_sigma, bool)
        and isinstance(workload_sigma, (int, float))
        and workload_sigma > 0
    ):
        return {
            "expected_sigma_ns": _stable_number(workload_sigma),
            "sigma_calibration": "workload_expected_sigma_ns",
        }
    return {
        "expected_sigma_ns": _stable_number(
            max(abs(float(predicted_duration_ns)), 1.0)
        ),
        "sigma_calibration": "heuristic_predicted_duration_scale",
    }


def _is_all_reduce_payload(payload: Mapping[str, Any]) -> bool:
    phase = payload.get("phase")
    collective = payload.get("collective")
    collective_kind = payload.get("collective_kind")
    return (
        phase == "collective/all_reduce"
        or collective == "all_reduce"
        or collective_kind == "all_reduce"
    )


def _is_checkpoint_payload(payload: Mapping[str, Any]) -> bool:
    kind = payload.get("kind")
    artifact_kind = payload.get("artifact_kind")
    phase = payload.get("phase")
    return any(
        isinstance(value, str) and "checkpoint" in value
        for value in (kind, artifact_kind, phase)
    )


def _checkpoint_operation(payload: Mapping[str, Any]) -> str | None:
    operation = payload.get("checkpoint_operation") or payload.get("operation")
    if isinstance(operation, str) and operation in {"stage", "save", "load"}:
        return operation
    phase = payload.get("phase")
    if isinstance(phase, str):
        suffix = phase.rsplit("/", maxsplit=1)[-1]
        if suffix in {"stage", "save", "load"}:
            return suffix
    state = payload.get("state")
    if isinstance(state, str) and state in {"stage", "save", "load"}:
        return state
    return None


def _payload(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = observation.get("payload", {})
    return payload if isinstance(payload, Mapping) else {}


def _entity(observation: Mapping[str, Any]) -> dict[str, str] | None:
    entity = observation.get("entity")
    if not isinstance(entity, Mapping):
        return None
    kind = entity.get("kind")
    entity_id = entity.get("id")
    if not isinstance(kind, str) or not isinstance(entity_id, str):
        return None
    return {"kind": kind, "id": entity_id}


def _observation_id(observation: Mapping[str, Any]) -> str:
    observation_id = observation.get("id")
    return observation_id if isinstance(observation_id, str) else "unknown"


def _duration_sort_key(item: Mapping[str, Any]) -> tuple[str, int, str, int]:
    return (
        str(item.get("phase", "")),
        _int_or_default(item.get("step")),
        str((item.get("process") or {}).get("id", "")),
        _int_or_default(item.get("duration_ns")),
    )


def _collective_sort_key(item: Mapping[str, Any]) -> tuple[int, str, str]:
    return (
        _int_or_default(item.get("step")),
        str((item.get("process") or {}).get("id", "")),
        str(item.get("phase", "")),
    )


def _residual_sort_key(item: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(item.get("kind", "")),
        _int_or_default(item.get("step")),
        str((item.get("process") or {}).get("id", "")),
    )


def _int_or_default(value: object, default: int = -1) -> int:
    return value if isinstance(value, int) else default


def _number_or_none(value: object) -> int | float | None:
    return value if isinstance(value, int | float) else None


def _stable_number(value: int | float) -> int | float:
    if isinstance(value, int):
        return value
    if value.is_integer():
        return int(value)
    return value


def _validate_positive(field_name: str, value: int | float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _validate_non_negative(field_name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
