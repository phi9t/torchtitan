# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Advisory transaction and checkpoint state summaries.

The functions in this module only read normalized evidence graph observations.
They do not call commit hooks, inspect checkpoint internals, or mutate training
state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torchtitan.observability.state_estimator.schema import SCHEMA_VERSION


CALIBRATION_STATE = "heuristic_uncalibrated"
_STATE_FIELDS = (
    "committed_step",
    "speculative_step",
    "process_group_epoch",
    "replica_epoch",
    "checkpoint_parent",
    "checkpoint_shard_inventory",
    "rng_evidence",
    "dataloader_evidence",
    "data_cursor",
    "save_status",
    "load_status",
    "staging_status",
    "restore_validation_status",
)
_FIELD_LABELS = {
    "committed_step": "committed step",
    "speculative_step": "speculative step",
    "process_group_epoch": "process-group epoch",
    "replica_epoch": "replica epoch",
    "checkpoint_parent": "checkpoint parent",
    "checkpoint_shard_inventory": "checkpoint shard inventory",
    "rng_evidence": "RNG evidence",
    "dataloader_evidence": "dataloader evidence",
    "data_cursor": "data cursor",
    "save_status": "save status",
    "load_status": "load status",
    "staging_status": "staging status",
    "restore_validation_status": "restore validation status",
}
_FIELD_ALIASES = {
    "committed_step": ("committed_step", "last_committed_step"),
    "speculative_step": ("speculative_step", "inflight_step", "current_step"),
    "process_group_epoch": ("process_group_epoch", "pg_epoch"),
    "replica_epoch": ("replica_epoch",),
    "checkpoint_parent": (
        "checkpoint_parent",
        "parent_checkpoint_id",
        "checkpoint_parent_id",
    ),
    "checkpoint_shard_inventory": (
        "checkpoint_shard_inventory",
        "checkpoint_shards",
        "shard_inventory",
    ),
    "rng_evidence": ("rng_evidence", "rng_state_id", "rng_state"),
    "dataloader_evidence": (
        "dataloader_evidence",
        "dataloader_state_id",
        "dataloader_state",
    ),
    "data_cursor": ("data_cursor", "data_position", "dataset_cursor"),
    "save_status": ("save_status",),
    "load_status": ("load_status",),
    "staging_status": ("staging_status", "stage_status"),
    "restore_validation_status": (
        "restore_validation_status",
        "restore_status",
        "restore_validation",
    ),
}


def summarize_transaction_state(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Return advisory transaction/checkpoint state from graph observations."""

    observations = [
        observation
        for observation in graph.get("observations", [])
        if isinstance(observation, Mapping)
    ]
    state = _extract_state(observations)
    advisories = _risk_advisories(state, observations)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": graph.get("run_id"),
        "attempt_id": graph.get("attempt_id"),
        "claim_calibration": CALIBRATION_STATE,
        "advisory": True,
        "state": state,
        "risk_advisories": advisories,
        "report_text": build_transaction_report_text(advisories),
    }


def build_transaction_report_text(advisories: Sequence[Mapping[str, Any]]) -> str:
    """Build compact operator text for advisory transaction findings."""

    lines = [
        "Transaction checkpoint state is advisory and uncalibrated.",
        (
            "A transaction-safety claim would require committed/speculative "
            "step, process-group epoch, replica epoch, checkpoint lineage, "
            "checkpoint shard inventory, RNG, dataloader, data cursor, "
            "save/load/staging, and restore-validation evidence."
        ),
    ]
    for advisory in advisories:
        message = advisory.get("message")
        if isinstance(message, str):
            lines.append(f"- {message}")
    return "\n".join(lines)


def _extract_state(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    state: dict[str, Any] = {field: None for field in _STATE_FIELDS}
    for observation in observations:
        payload = _payload(observation)
        for field in _STATE_FIELDS:
            if state[field] is not None:
                continue
            value = _field_value(payload, field)
            if value is None:
                continue
            state[field] = {
                "value": _stable_value(value),
                "source_evidence": [_observation_id(observation)],
            }
    return state


def _risk_advisories(
    state: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    advisories: list[dict[str, Any]] = []
    missing_fields = [field for field in _STATE_FIELDS if state.get(field) is None]
    if not missing_fields:
        advisories.append(
            _advisory(
                "enough_evidence_to_assess_risk",
                status="evidence_present",
                message=(
                    "advisory transaction-risk assessment has the expected "
                    "transaction, checkpoint, data-position, and restore "
                    "evidence fields; the conclusion remains advisory."
                ),
                missing_fields=[],
                source_evidence=_state_evidence(state),
            )
        )
    else:
        advisories.append(
            _advisory(
                "missing_transaction_evidence",
                status="evidence_missing",
                message=(
                    "advisory transaction-risk assessment lacks "
                    f"{_field_list(missing_fields)} evidence."
                ),
                missing_fields=missing_fields,
                source_evidence=[],
            )
        )

    for anomaly in _anomaly_observations(observations):
        anomaly_step = _anomaly_step(anomaly)
        committed = _state_value(state, "committed_step")
        if not isinstance(anomaly_step, int) or not isinstance(committed, int):
            continue
        if anomaly_step <= committed:
            advisories.append(
                _advisory(
                    "anomaly_before_likely_commit_boundary",
                    status="advisory_timing",
                    message=(
                        "advisory anomaly timing places evidence before the "
                        f"likely commit boundary at step {committed}."
                    ),
                    missing_fields=[],
                    source_evidence=[_observation_id(anomaly)],
                )
            )
        else:
            advisories.append(
                _advisory(
                    "anomaly_after_likely_commit_boundary",
                    status="advisory_timing",
                    message=(
                        "advisory anomaly timing places evidence after the "
                        f"likely commit boundary at step {committed}."
                    ),
                    missing_fields=[],
                    source_evidence=[_observation_id(anomaly)],
                )
            )

    if _has_checkpoint_evidence(observations):
        lineage_missing = [
            field
            for field in ("checkpoint_parent", "checkpoint_shard_inventory")
            if state.get(field) is None
        ]
        if lineage_missing:
            advisories.append(
                _advisory(
                    "checkpoint_lineage_incomplete",
                    status="evidence_missing",
                    message=(
                        "advisory checkpoint lineage assessment lacks "
                        f"{_field_list(lineage_missing)} evidence."
                    ),
                    missing_fields=lineage_missing,
                    source_evidence=_checkpoint_evidence(observations),
                )
            )
        if state.get("restore_validation_status") is None:
            advisories.append(
                _advisory(
                    "checkpoint_restore_validation_absent",
                    status="evidence_missing",
                    message=(
                        "advisory checkpoint restore assessment lacks restore "
                        "validation status evidence."
                    ),
                    missing_fields=["restore_validation_status"],
                    source_evidence=_checkpoint_evidence(observations),
                )
            )
    return advisories


def _advisory(
    kind: str,
    *,
    status: str,
    message: str,
    missing_fields: Sequence[str],
    source_evidence: Sequence[str],
) -> dict[str, Any]:
    return {
        "kind": kind,
        "status": status,
        "advisory": True,
        "calibration": CALIBRATION_STATE,
        "message": message,
        "missing_fields": list(missing_fields),
        "source_evidence": list(source_evidence),
    }


def _field_list(fields: Sequence[str]) -> str:
    labels = [_FIELD_LABELS[field] for field in fields]
    if not labels:
        return "no"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _state_evidence(state: Mapping[str, Any]) -> list[str]:
    evidence: list[str] = []
    for field in _STATE_FIELDS:
        item = state.get(field)
        if not isinstance(item, Mapping):
            continue
        for evidence_id in item.get("source_evidence", []):
            if isinstance(evidence_id, str) and evidence_id not in evidence:
                evidence.append(evidence_id)
    return evidence


def _state_value(state: Mapping[str, Any], field: str) -> Any:
    item = state.get(field)
    if isinstance(item, Mapping):
        return item.get("value")
    return None


def _payload(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = observation.get("payload", {})
    return payload if isinstance(payload, Mapping) else {}


def _field_value(payload: Mapping[str, Any], field: str) -> Any:
    for key in _FIELD_ALIASES[field]:
        if key in payload and payload[key] is not None:
            return payload[key]
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        for key in _FIELD_ALIASES[field]:
            if key in metadata and metadata[key] is not None:
                return metadata[key]
    return None


def _stable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _stable_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    if isinstance(value, tuple):
        return [_stable_value(item) for item in value]
    return value


def _observation_id(observation: Mapping[str, Any]) -> str:
    observation_id = observation.get("id")
    return observation_id if isinstance(observation_id, str) else "unknown"


def _anomaly_observations(
    observations: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    anomalies: list[Mapping[str, Any]] = []
    for observation in observations:
        payload = _payload(observation)
        record_type = payload.get("record_type")
        event = payload.get("event") or payload.get("incident_type")
        severity = payload.get("severity")
        outcome = payload.get("outcome")
        if (
            record_type == "incident"
            or isinstance(event, str)
            and event
            in {
                "rank_death",
                "nonfinite_loss",
                "checkpoint_corruption",
                "checkpoint_interruption",
                "collective_hang",
            }
        ):
            anomalies.append(observation)
        elif severity in {"error", "fatal", "critical"} or outcome == "failed":
            anomalies.append(observation)
    return anomalies


def _anomaly_step(observation: Mapping[str, Any]) -> int | None:
    payload = _payload(observation)
    step = payload.get("step")
    return step if isinstance(step, int) else None


def _has_checkpoint_evidence(observations: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        _is_checkpoint_payload(_payload(observation)) for observation in observations
    )


def _checkpoint_evidence(observations: Sequence[Mapping[str, Any]]) -> list[str]:
    evidence: list[str] = []
    for observation in observations:
        if _is_checkpoint_payload(_payload(observation)):
            evidence.append(_observation_id(observation))
    return evidence


def _is_checkpoint_payload(payload: Mapping[str, Any]) -> bool:
    for key in ("kind", "artifact_kind", "phase", "checkpoint_id", "artifact_id"):
        value = payload.get(key)
        if isinstance(value, str) and "checkpoint" in value:
            return True
    return any(
        _field_value(payload, field) is not None
        for field in (
            "checkpoint_parent",
            "checkpoint_shard_inventory",
            "save_status",
            "load_status",
            "staging_status",
            "restore_validation_status",
        )
    )
