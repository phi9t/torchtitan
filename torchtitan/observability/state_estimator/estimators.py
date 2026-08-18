# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic offline estimators for training run evidence graphs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from torchtitan.observability.state_estimator.analytical import build_analytical_summary
from torchtitan.observability.state_estimator.graph import build_evidence_graph
from torchtitan.observability.state_estimator.inference import score_fault_modes
from torchtitan.observability.state_estimator.probes import recommend_probes
from torchtitan.observability.state_estimator.timeline import build_incident_timeline
from torchtitan.observability.state_estimator.transactions import (
    summarize_transaction_state,
)
from torchtitan.observability.state_estimator.schema import (
    DerivedPaths,
    EntityRef,
    QualityFinding,
    SCHEMA_VERSION,
    write_json_atomic,
)


def _process_liveness(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entities = {entity["id"]: entity for entity in graph.get("entities", [])}
    liveness: dict[str, dict[str, Any]] = {}
    for edge in graph.get("edges", []):
        if edge.get("kind") != "process_outcome":
            continue
        process_id = edge["source"]
        outcome_id = edge["target"]
        outcome = entities.get(outcome_id, {}).get("attrs", {}).get("outcome", "unknown")
        liveness[process_id] = {"outcome": outcome}
    return liveness


def _peer_skew_findings(
    observations: list[Mapping[str, Any]], *, skew_threshold_ns: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        if observation.get("kind") != "structured_event":
            continue
        payload = observation.get("payload", {})
        phase = payload.get("phase")
        step = payload.get("step")
        if isinstance(phase, str) and isinstance(step, int):
            groups[(phase, step)].append(observation)
    findings: list[dict[str, Any]] = []
    for (phase, step), rows in sorted(groups.items()):
        times = [
            row.get("event_time_ns")
            for row in rows
            if isinstance(row.get("event_time_ns"), int)
        ]
        if len(times) < 2:
            continue
        skew_ns = max(times) - min(times)
        if skew_ns > skew_threshold_ns:
            latest = max(
                rows,
                key=lambda row: (
                    row.get("event_time_ns")
                    if isinstance(row.get("event_time_ns"), int)
                    else -1
                ),
            )
            findings.append(
                {
                    "phase": phase,
                    "step": step,
                    "skew_ns": skew_ns,
                    "latest_entity": latest.get("entity"),
                    "root_cause": "ambiguous",
                    "candidate_modes": [
                        "host_or_data_stall",
                        "compute_degradation",
                        "network_degradation",
                        "normal_workload_skew",
                    ],
                }
            )
    return findings


def _phase_event_state(payload: Mapping[str, Any]) -> str | None:
    for field in ("event", "phase_state", "state"):
        value = payload.get(field)
        if not isinstance(value, str):
            continue
        normalized = value.lower()
        if normalized in {"start", "started", "begin", "began"}:
            return "start"
        if normalized in {"end", "ended", "finish", "finished", "complete", "completed"}:
            return "end"
    message = payload.get("message")
    if isinstance(message, str):
        suffix = message.lower().rsplit("_", maxsplit=1)[-1]
        if suffix in {"start", "started", "begin", "began"}:
            return "start"
        if suffix in {"end", "ended", "finish", "finished", "complete", "completed"}:
            return "end"
    return None


def _phase_time(payload: Mapping[str, Any], observation: Mapping[str, Any]) -> tuple[str, int] | None:
    monotonic_ns = payload.get("monotonic_ns")
    if isinstance(monotonic_ns, int):
        return ("monotonic_ns", monotonic_ns)
    event_time_ns = observation.get("event_time_ns")
    if isinstance(event_time_ns, int):
        return ("wall_time_ns", event_time_ns)
    wall_time_ns = payload.get("wall_time_ns")
    if isinstance(wall_time_ns, int):
        return ("wall_time_ns", wall_time_ns)
    return None


def _duration_from_explicit_field(
    payload: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any] | None:
    duration_ns = payload.get("duration_ns")
    if not isinstance(duration_ns, int):
        duration_ns = payload.get("elapsed_monotonic_ns")
    if not isinstance(duration_ns, int):
        return None
    phase = payload.get("phase")
    step = payload.get("step")
    if not isinstance(phase, str) or not isinstance(step, int):
        return None
    return {
        "phase": phase,
        "step": step,
        "process": observation.get("entity"),
        "global_rank": payload.get("global_rank"),
        "duration_ns": duration_ns,
        "clock": "duration_ns",
        "start_time_ns": None,
        "end_time_ns": None,
        "calibration": "heuristic",
    }


def _duration_from_paired_fields(
    payload: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any] | None:
    phase = payload.get("phase")
    step = payload.get("step")
    start_time_ns = payload.get("start_time_ns")
    end_time_ns = payload.get("end_time_ns")
    if (
        not isinstance(phase, str)
        or not isinstance(step, int)
        or not isinstance(start_time_ns, int)
        or not isinstance(end_time_ns, int)
        or end_time_ns < start_time_ns
    ):
        return None
    clock = payload.get("clock")
    return {
        "phase": phase,
        "step": step,
        "process": observation.get("entity"),
        "global_rank": payload.get("global_rank"),
        "duration_ns": end_time_ns - start_time_ns,
        "clock": clock if isinstance(clock, str) else "time_ns",
        "start_time_ns": start_time_ns,
        "end_time_ns": end_time_ns,
        "calibration": "heuristic",
    }


def _phase_duration_missing_fields(
    payload: Mapping[str, Any], observation: Mapping[str, Any]
) -> list[str]:
    missing_fields: list[str] = []
    if not isinstance(payload.get("phase"), str):
        missing_fields.append("phase")
    if not isinstance(payload.get("step"), int):
        missing_fields.append("step")
    entity = observation.get("entity", {})
    process_id = entity.get("id") if isinstance(entity, Mapping) else None
    if not isinstance(process_id, str):
        missing_fields.append("process")
    duration_ns = payload.get("duration_ns")
    elapsed_monotonic_ns = payload.get("elapsed_monotonic_ns")
    start_time_ns = payload.get("start_time_ns")
    end_time_ns = payload.get("end_time_ns")
    state = _phase_event_state(payload)
    phase_time = _phase_time(payload, observation)
    has_duration_evidence = (
        isinstance(duration_ns, int)
        or isinstance(elapsed_monotonic_ns, int)
        or (isinstance(start_time_ns, int) and isinstance(end_time_ns, int))
        or (state in {"start", "end"} and phase_time is not None)
    )
    if not has_duration_evidence:
        missing_fields.append("duration_fields")
    return missing_fields


def _missing_phase_duration_warning(
    *,
    payload: Mapping[str, Any],
    observation: Mapping[str, Any],
    missing_fields: list[str],
) -> dict[str, Any]:
    entity = observation.get("entity", {})
    process_id = entity.get("id") if isinstance(entity, Mapping) else None
    entity_ref = (
        EntityRef(kind="process", id=process_id)
        if isinstance(process_id, str)
        else None
    )
    phase = payload.get("phase")
    step = payload.get("step")
    observation_id = observation.get("id")
    return QualityFinding(
        kind="missing_phase_duration_fields",
        severity="info",
        evidence_state="missing",
        message=(
            f"structured event id={observation_id} cannot produce an exact phase "
            f"duration; missing fields: {', '.join(missing_fields)}"
        ),
        source_path=str(observation.get("source_path", "structured_logs")),
        entity=entity_ref,
        metadata={
            "observation_id": observation_id,
            "phase": phase if isinstance(phase, str) else None,
            "step": step if isinstance(step, int) else None,
            "process": process_id if isinstance(process_id, str) else None,
            "duration_ns": payload.get("duration_ns"),
            "elapsed_monotonic_ns": payload.get("elapsed_monotonic_ns"),
            "start_time_ns": payload.get("start_time_ns"),
            "end_time_ns": payload.get("end_time_ns"),
            "missing_fields": missing_fields,
        },
    ).to_json()


def _phase_duration_summaries(
    observations: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    explicit: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for observation in observations:
        if observation.get("kind") != "structured_event":
            continue
        payload = observation.get("payload", {})
        if not isinstance(payload, Mapping):
            continue
        duration = _duration_from_explicit_field(payload, observation)
        if duration is not None:
            explicit.append(duration)
            continue
        duration = _duration_from_paired_fields(payload, observation)
        if duration is not None:
            explicit.append(duration)
            continue
        phase = payload.get("phase")
        step = payload.get("step")
        entity = observation.get("entity", {})
        process_id = entity.get("id") if isinstance(entity, Mapping) else None
        if isinstance(phase, str) and isinstance(step, int) and isinstance(process_id, str):
            groups[(phase, step, process_id)].append(observation)
            continue
        missing_fields = _phase_duration_missing_fields(payload, observation)
        if missing_fields:
            warnings.append(
                _missing_phase_duration_warning(
                    payload=payload,
                    observation=observation,
                    missing_fields=missing_fields,
                )
            )

    summaries = list(explicit)
    for (phase, step, process_id), rows in sorted(groups.items()):
        starts: list[tuple[int, Mapping[str, Any]]] = []
        ends: list[tuple[int, Mapping[str, Any]]] = []
        clocks: set[str] = set()
        for row in rows:
            payload = row.get("payload", {})
            if not isinstance(payload, Mapping):
                continue
            phase_time = _phase_time(payload, row)
            if phase_time is None:
                continue
            clock, time_ns = phase_time
            clocks.add(clock)
            state = _phase_event_state(payload)
            if state == "start":
                starts.append((time_ns, row))
            elif state == "end":
                ends.append((time_ns, row))
        if starts and ends:
            start_time_ns, start_row = min(starts, key=lambda item: item[0])
            end_time_ns, end_row = max(ends, key=lambda item: item[0])
            if end_time_ns >= start_time_ns:
                payload = end_row.get("payload", {})
                if not isinstance(payload, Mapping):
                    payload = start_row.get("payload", {})
                summaries.append(
                    {
                        "phase": phase,
                        "step": step,
                        "process": end_row.get("entity") or start_row.get("entity"),
                        "global_rank": payload.get("global_rank"),
                        "duration_ns": end_time_ns - start_time_ns,
                        "clock": "monotonic_ns" if "monotonic_ns" in clocks else "wall_time_ns",
                        "start_time_ns": start_time_ns,
                        "end_time_ns": end_time_ns,
                        "calibration": "heuristic",
                    }
                )
                continue
        warnings.append(
            QualityFinding(
                kind="missing_phase_duration_fields",
                severity="info",
                evidence_state="missing",
                message=(
                    f"phase duration for phase={phase} step={step} process={process_id} "
                    "requires paired start/end events or explicit duration fields"
                ),
                source_path="structured_logs",
                entity=EntityRef(kind="process", id=process_id),
                metadata={
                    "phase": phase,
                    "step": step,
                    "missing_fields": [
                        "duration_ns",
                        "phase_state",
                        "monotonic_ns_pair",
                    ],
                },
            ).to_json()
        )
    return (
        sorted(
            summaries,
            key=lambda item: (
                item["phase"],
                item["step"],
                str(item.get("process", {}).get("id", "")),
                item["duration_ns"],
            ),
        ),
        warnings,
    )


def estimate_belief(
    graph: Mapping[str, Any], *, skew_threshold_ns: int = 500_000_000
) -> dict[str, Any]:
    liveness = _process_liveness(graph)
    fault_hypotheses: list[dict[str, Any]] = []
    liveness_quality: list[dict[str, Any]] = []
    if any(item["outcome"] == "failed" for item in liveness.values()):
        fault_hypotheses.append(
            {
                "mode": "process_failure",
                "score": 1.0,
                "calibration": "heuristic",
                "evidence": "process outcome reported failed",
            }
        )
    for process_id, item in sorted(liveness.items()):
        if item["outcome"] != "failed":
            continue
        liveness_quality.append(
            QualityFinding(
                kind="process_outcome_failed",
                severity="error",
                evidence_state="failed",
                message="process outcome reported failed",
                source_path="process_outcome",
                entity=EntityRef(kind="process", id=process_id),
                metadata={"process": process_id, "outcome": "failed"},
            ).to_json()
        )
    peer_skew = _peer_skew_findings(
        list(graph.get("observations", [])), skew_threshold_ns=skew_threshold_ns
    )
    phase_durations, phase_duration_warnings = _phase_duration_summaries(
        list(graph.get("observations", []))
    )
    analytical = build_analytical_summary(graph)
    inference = score_fault_modes(
        {
            **analytical,
            "quality": [
                *list(graph.get("quality", [])),
                *liveness_quality,
                *phase_duration_warnings,
            ],
        }
    )
    if peer_skew:
        fault_hypotheses.append(
            {
                "mode": "peer_relative_skew",
                "score": 0.5,
                "calibration": "heuristic",
                "evidence": "phase arrival skew exceeded threshold",
            }
        )
    transactions = summarize_transaction_state(graph)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": graph.get("run_id"),
        "attempt_id": graph.get("attempt_id"),
        "claim_calibration": "heuristic",
        "process_liveness": liveness,
        "phase_durations": phase_durations,
        "analytical": analytical,
        "inference": inference,
        "transactions": transactions,
        "peer_skew_findings": peer_skew,
        "fault_hypotheses": fault_hypotheses,
        "observability_warnings": [
            *list(graph.get("quality", [])),
            *liveness_quality,
            *phase_duration_warnings,
        ],
    }


def write_belief_summary(attempt_path: Path) -> DerivedPaths:
    paths = DerivedPaths.from_attempt_path(attempt_path)
    graph = build_evidence_graph(attempt_path)
    summary = estimate_belief(graph)
    timeline_graph = {
        **graph,
        "observations": [
            observation
            for observation in graph.get("observations", [])
            if isinstance(observation, Mapping)
            and observation.get("kind") == "structured_event"
        ],
    }
    timeline = build_incident_timeline(
        timeline_graph,
        center_time_ns=None,
        window_before_ns=300_000_000_000,
        window_after_ns=30_000_000_000,
    )
    summary["incident_timeline"] = timeline
    summary["probe_recommendations"] = recommend_probes(summary)
    write_json_atomic(paths.belief_summary, summary)
    paths.diagnosis.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Training State Estimator Diagnosis",
        "",
        f"- run_id: {summary['run_id']}",
        f"- attempt_id: {summary['attempt_id']}",
        f"- claim_calibration: {summary['claim_calibration']}",
        f"- fault_hypotheses: {len(summary['fault_hypotheses'])}",
        f"- peer_skew_findings: {len(summary['peer_skew_findings'])}",
        "",
        "## Incident Timeline",
        "",
        *(
            f"- {row['relative_time_ns']} ns: {row.get('id')}"
            for row in timeline[:20]
        ),
        "",
        "## Probe Recommendations",
        "",
        *(
            f"- {probe['kind']}: {probe['information_target']}"
            for probe in summary["probe_recommendations"]
        ),
        "",
        "## Transaction Advisory",
        "",
        summary["transactions"]["report_text"],
        "",
    ]
    paths.diagnosis.write_text("\n".join(lines))
    return paths
