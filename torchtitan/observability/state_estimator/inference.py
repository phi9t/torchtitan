# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Robust advisory inference over offline analytical residual summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from torchtitan.observability.state_estimator.schema import SCHEMA_VERSION


MODE_LABELS = (
    "nominal",
    "compute_degradation",
    "network_degradation",
    "host_data_stall",
    "collective_desynchronization",
    "memory_fault",
    "numerical_corruption",
    "storage_fault",
    "control_plane_fault",
    "planned_intervention",
    "failed",
    "recovering",
)

CALIBRATION_STATE = "heuristic_uncalibrated"
_RESIDUAL_CAP = 6.0
_CUSUM_REFERENCE = 0.75
_CUSUM_CAP = 12.0


@dataclass(frozen=True)
class ModeScore:
    mode: str
    score: float
    calibration: str = CALIBRATION_STATE
    advisory: bool = True
    evidence: list[str] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "score": self.score,
            "calibration": self.calibration,
            "advisory": self.advisory,
            "evidence": self.evidence or [],
        }


@dataclass(frozen=True)
class SensorHealthState:
    sensor_id: str
    state: str
    calibration: str = CALIBRATION_STATE
    evidence: list[str] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "state": self.state,
            "calibration": self.calibration,
            "evidence": self.evidence or [],
        }


@dataclass(frozen=True)
class RootCauseCandidate:
    candidate: str
    score: float
    affected_scope: dict[str, Any]
    evidence_chain: list[dict[str, Any]]
    calibration: str = CALIBRATION_STATE
    advisory: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "score": self.score,
            "affected_scope": self.affected_scope,
            "evidence_chain": self.evidence_chain,
            "calibration": self.calibration,
            "advisory": self.advisory,
        }


@dataclass(frozen=True)
class ObservabilityWarning:
    kind: str
    message: str
    candidate_modes: list[str]
    evidence: list[str]
    calibration: str = CALIBRATION_STATE

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "candidate_modes": self.candidate_modes,
            "evidence": self.evidence,
            "calibration": self.calibration,
        }


def score_fault_modes(analytical_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Score advisory fault modes from residuals and evidence quality.

    Scores are deterministic heuristic severities. They are not probabilities
    and do not authorize control-plane action.
    """

    residuals = _residuals(analytical_summary)
    symptoms = _collective_symptoms(analytical_summary)
    quality = _quality_findings(analytical_summary)
    robust = _robust_residual_score(residuals)
    persistent = _persistent_degradation(residuals)
    residual_evidence = _residual_evidence(residuals)

    feature_scores = {
        "compute_degradation": _score_compute(residuals, robust, persistent),
        "network_degradation": _score_network(residuals, symptoms, robust),
        "host_data_stall": _score_host_data(residuals, quality),
        "collective_desynchronization": _score_collective_desync(symptoms),
        "memory_fault": _score_by_terms(residuals, quality, ("memory", "hbm", "oom")),
        "numerical_corruption": _score_by_terms(
            residuals, quality, ("nan", "inf", "numerical", "loss")
        ),
        "storage_fault": _score_by_terms(
            residuals, quality, ("checkpoint", "storage", "artifact")
        ),
        "control_plane_fault": _score_by_terms(
            residuals, quality, ("control", "launcher", "heartbeat", "scheduler")
        ),
        "planned_intervention": _score_by_terms(
            residuals, quality, ("planned", "intervention", "maintenance")
        ),
        "failed": _score_by_terms(residuals, quality, ("failed", "failure", "died")),
        "recovering": _score_by_terms(residuals, quality, ("recovering", "restart")),
    }
    max_non_nominal = max(feature_scores.values()) if feature_scores else 0.0
    mode_scores = [
        ModeScore(
            mode="nominal",
            score=_round_score(max(0.0, 1.0 - max_non_nominal)),
            evidence=[] if max_non_nominal else residual_evidence,
        ),
        *[
            ModeScore(
                mode=mode,
                score=_round_score(feature_scores[mode]),
                evidence=residual_evidence,
            )
            for mode in MODE_LABELS
            if mode != "nominal"
        ],
    ]
    warnings = _observability_warnings(residuals, symptoms, feature_scores)
    candidates = rank_root_causes(
        residuals=residuals,
        collective_symptoms=symptoms,
        mode_scores=[score.to_json() for score in mode_scores],
        warnings=warnings,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": analytical_summary.get("run_id"),
        "attempt_id": analytical_summary.get("attempt_id"),
        "claim_calibration": CALIBRATION_STATE,
        "mode_scores": [score.to_json() for score in mode_scores],
        "persistent_degradation": persistent,
        "sensor_health": [
            item.to_json() for item in _sensor_health_states(residuals, quality)
        ],
        "root_cause_candidates": [candidate.to_json() for candidate in candidates],
        "observability_warnings": [warning.to_json() for warning in warnings],
        "metadata": {
            "robust_residual_scoring": {
                "method": "winsorized_abs_residual_mean",
                "cap": _RESIDUAL_CAP,
                "note": (
                    "Absolute normalized residuals are capped before aggregation "
                    "so one outlier cannot dominate a mode score."
                ),
            },
            "score_semantics": (
                "Advisory heuristic severity score; not a calibrated probability."
            ),
        },
    }


def rank_root_causes(
    *,
    residuals: Sequence[Mapping[str, Any]],
    collective_symptoms: Sequence[Mapping[str, Any]],
    mode_scores: Sequence[Mapping[str, Any]],
    warnings: Sequence[ObservabilityWarning],
) -> list[RootCauseCandidate]:
    """Rank advisory root-cause candidates with scope and evidence chains."""

    if warnings:
        warning = warnings[0]
        return [
            RootCauseCandidate(
                candidate="ambiguous",
                score=_top_candidate_score(mode_scores),
                affected_scope=_affected_scope(residuals, collective_symptoms),
                evidence_chain=_evidence_chain(residuals, collective_symptoms),
            )
        ]

    mode = _top_non_nominal_mode(mode_scores)
    if mode is None:
        return []
    return [
        RootCauseCandidate(
            candidate=mode["mode"],
            score=mode["score"],
            affected_scope=_affected_scope(residuals, collective_symptoms),
            evidence_chain=_evidence_chain(residuals, collective_symptoms),
        )
    ]


def _residuals(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [item for item in summary.get("residuals", []) if isinstance(item, Mapping)]


def _collective_symptoms(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        item
        for item in summary.get("collective_symptoms", [])
        if isinstance(item, Mapping)
    ]


def _quality_findings(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [item for item in summary.get("quality", []) if isinstance(item, Mapping)]


def _robust_residual_score(residuals: Sequence[Mapping[str, Any]]) -> float:
    values = [_capped_abs_residual(item) for item in residuals]
    if not values:
        return 0.0
    return min(sum(values) / len(values) / _RESIDUAL_CAP, 1.0)


def _persistent_degradation(residuals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    running = 0.0
    peak = 0.0
    evidence: list[str] = []
    for residual in sorted(residuals, key=_residual_order):
        value = max(0.0, _capped_abs_residual(residual) - _CUSUM_REFERENCE)
        running = min(_CUSUM_CAP, max(0.0, running + value))
        peak = max(peak, running)
        if value > 0:
            evidence.extend(_source_evidence(residual))
    return {
        "method": "positive_cusum",
        "reference": _CUSUM_REFERENCE,
        "cap": _CUSUM_CAP,
        "score": _round_score(peak / _CUSUM_CAP),
        "calibration": CALIBRATION_STATE,
        "evidence": sorted(set(evidence)),
    }


def _score_compute(
    residuals: Sequence[Mapping[str, Any]],
    robust: float,
    persistent: Mapping[str, Any],
) -> float:
    non_collective = [
        item
        for item in residuals
        if "collective" not in _residual_text(item)
        and "dataloader" not in _residual_text(item)
    ]
    if not non_collective:
        return max(0.0, robust * 0.35)
    local_robust = _robust_residual_score(non_collective)
    return min(1.0, local_robust * 0.75 + float(persistent["score"]) * 0.35)


def _score_network(
    residuals: Sequence[Mapping[str, Any]],
    symptoms: Sequence[Mapping[str, Any]],
    robust: float,
) -> float:
    collective_residuals = [
        item for item in residuals if "collective" in _residual_text(item)
    ]
    symptom_score = 0.0
    for symptom in symptoms:
        progress = symptom.get("network_progress_ns")
        skew = symptom.get("arrival_skew_ns")
        if isinstance(progress, int) and progress > 0:
            symptom_score = max(symptom_score, min(progress / 1_000_000_000, 1.0))
        if isinstance(skew, int) and skew > 0:
            symptom_score = max(symptom_score, min(skew / 1_000_000_000, 1.0) * 0.5)
    if not collective_residuals:
        return min(1.0, symptom_score)
    return min(1.0, _robust_residual_score(collective_residuals) * 0.8 + symptom_score * 0.4)


def _score_host_data(
    residuals: Sequence[Mapping[str, Any]], quality: Sequence[Mapping[str, Any]]
) -> float:
    host_residuals = [
        item
        for item in residuals
        if any(term in _residual_text(item) for term in ("data", "dataloader", "host"))
    ]
    quality_score = _score_by_terms([], quality, ("data", "dataloader", "host"))
    if not host_residuals:
        return quality_score
    return min(1.0, _robust_residual_score(host_residuals) * 0.85 + quality_score)


def _score_collective_desync(symptoms: Sequence[Mapping[str, Any]]) -> float:
    score = 0.0
    for symptom in symptoms:
        skew = symptom.get("arrival_skew_ns")
        progress = symptom.get("network_progress_ns")
        if isinstance(skew, int) and skew > 0:
            denominator = max(progress if isinstance(progress, int) else 0, 1)
            score = max(score, min(skew / denominator, 1.0))
    return score


def _score_by_terms(
    residuals: Sequence[Mapping[str, Any]],
    quality: Sequence[Mapping[str, Any]],
    terms: Sequence[str],
) -> float:
    matched_residuals = [
        residual
        for residual in residuals
        if any(term in _residual_text(residual) for term in terms)
    ]
    if residuals:
        score = min(
            sum(_capped_abs_residual(residual) for residual in matched_residuals)
            / len(residuals)
            / _RESIDUAL_CAP,
            1.0,
        )
    else:
        score = 0.0
    for residual in residuals:
        text = _residual_text(residual)
        if (
            any(term in text for term in terms)
            and _numeric_residual(residual) is None
        ):
            score = max(score, 0.25)
    for finding in quality:
        text = _quality_text(finding)
        if any(term in text for term in terms):
            severity = finding.get("severity")
            evidence_state = finding.get("evidence_state")
            if severity == "error" or evidence_state == "failed":
                score = max(score, 0.85)
            elif severity == "warning" or evidence_state in {"missing", "delayed"}:
                score = max(score, 0.5)
            else:
                score = max(score, 0.25)
    return min(score, 1.0)


def _sensor_health_states(
    residuals: Sequence[Mapping[str, Any]], quality: Sequence[Mapping[str, Any]]
) -> list[SensorHealthState]:
    by_sensor: dict[str, list[Mapping[str, Any]]] = {}
    evidence_by_sensor: dict[str, list[str]] = {}
    explicit: dict[str, str] = {}
    for residual in residuals:
        sensor_id = _sensor_id_from_residual(residual)
        if sensor_id is None:
            continue
        by_sensor.setdefault(sensor_id, []).append(residual)
        evidence_by_sensor.setdefault(sensor_id, []).extend(_source_evidence(residual))
    for finding in quality:
        sensor_id = _sensor_id_from_quality(finding)
        if sensor_id is None:
            continue
        evidence_by_sensor.setdefault(sensor_id, []).extend(_source_evidence(finding))
        state = _state_from_quality(finding)
        if state is not None:
            explicit[sensor_id] = state

    states: list[SensorHealthState] = []
    for sensor_id in sorted(set(by_sensor) | set(explicit)):
        state = explicit.get(sensor_id)
        rows = by_sensor.get(sensor_id, [])
        if state is None:
            state = _state_from_residuals(rows)
        states.append(
            SensorHealthState(
                sensor_id=sensor_id,
                state=state,
                evidence=sorted(set(evidence_by_sensor.get(sensor_id, []))),
            )
        )
    return states


def _state_from_quality(finding: Mapping[str, Any]) -> str | None:
    evidence_state = finding.get("evidence_state")
    if evidence_state in {"missing", "uncollected"}:
        return "missing"
    if evidence_state == "delayed":
        return "delayed"
    text = _quality_text(finding)
    if "delayed" in text or "clock" in text:
        return "delayed"
    if "bias" in text:
        return "biased"
    if "stuck" in text:
        return "stuck"
    return None


def _state_from_residuals(rows: Sequence[Mapping[str, Any]]) -> str:
    values = [
        value
        for value in (_numeric_residual(row) for row in rows)
        if value is not None
    ]
    if len(values) >= 2 and sum(values) / len(values) >= 2.5:
        return "biased"
    return "healthy"


def _observability_warnings(
    residuals: Sequence[Mapping[str, Any]],
    symptoms: Sequence[Mapping[str, Any]],
    feature_scores: Mapping[str, float],
) -> list[ObservabilityWarning]:
    candidate_modes = [
        mode
        for mode in MODE_LABELS
        if mode != "nominal"
        if feature_scores.get(mode, 0.0) >= 0.3
    ]
    if len(candidate_modes) < 2:
        return []
    non_compute_candidates = [
        mode
        for mode in candidate_modes
        if mode != "compute_degradation"
    ]
    if len(non_compute_candidates) >= 2:
        candidate_modes = non_compute_candidates
    top_score = max(feature_scores.get(mode, 0.0) for mode in candidate_modes)
    candidate_modes = [
        mode
        for mode in candidate_modes
        if top_score - feature_scores.get(mode, 0.0) <= 0.1
    ]
    if (
        "network_degradation" in candidate_modes
        and "host_data_stall" in candidate_modes
        and "compute_degradation" not in candidate_modes
    ):
        candidate_modes.append("compute_degradation")
    if len(candidate_modes) < 2:
        return []
    return [
        ObservabilityWarning(
            kind="ambiguous_fault_mode",
            message=(
                "Available residual evidence cannot distinguish the candidate "
                "fault hypotheses without independent sensor or probe evidence."
            ),
            candidate_modes=candidate_modes,
            evidence=sorted(
                set(_residual_evidence(residuals) + _symptom_evidence(symptoms))
            ),
        )
    ]


def _affected_scope(
    residuals: Sequence[Mapping[str, Any]], symptoms: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    phases = sorted(
        {
            phase
            for phase in [item.get("phase") for item in [*residuals, *symptoms]]
            if isinstance(phase, str)
        }
    )
    processes = sorted(
        {
            process_id
            for process_id in [_process_id(item) for item in residuals]
            if process_id is not None
        }
    )
    steps = sorted(
        {
            step
            for step in [item.get("step") for item in [*residuals, *symptoms]]
            if isinstance(step, int)
        }
    )
    return {
        "phase": phases[0] if len(phases) == 1 else phases,
        "processes": processes,
        "steps": steps,
    }


def _evidence_chain(
    residuals: Sequence[Mapping[str, Any]], symptoms: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    for residual in residuals:
        value = _numeric_residual(residual)
        chain.append(
            {
                "kind": str(residual.get("kind", "residual")),
                "source_evidence": _source_evidence(residual),
                "summary": (
                    f"normalized_residual={value}"
                    if value is not None
                    else "normalized_residual=unknown"
                ),
            }
        )
    for symptom in symptoms:
        chain.append(
            {
                "kind": "collective_symptom",
                "source_evidence": _source_evidence(symptom),
                "summary": (
                    f"network_progress_ns={symptom.get('network_progress_ns')} "
                    f"arrival_skew_ns={symptom.get('arrival_skew_ns')}"
                ),
            }
        )
    return chain


def _top_non_nominal_mode(
    mode_scores: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    candidates = [item for item in mode_scores if item.get("mode") != "nominal"]
    if not candidates:
        return None
    top = max(candidates, key=lambda item: item.get("score", 0.0))
    if not isinstance(top.get("score"), (int, float)) or top["score"] <= 0:
        return None
    return top


def _top_candidate_score(mode_scores: Sequence[Mapping[str, Any]]) -> float:
    top = _top_non_nominal_mode(mode_scores)
    if top is None:
        return 0.0
    return float(top["score"])


def _capped_abs_residual(residual: Mapping[str, Any]) -> float:
    value = _numeric_residual(residual)
    if value is None:
        return 0.0
    return min(abs(value), _RESIDUAL_CAP)


def _numeric_residual(residual: Mapping[str, Any]) -> float | None:
    value = residual.get("normalized_residual")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _residual_order(residual: Mapping[str, Any]) -> tuple[int, str, str]:
    step = residual.get("step")
    return (
        step if isinstance(step, int) else -1,
        str(residual.get("phase", "")),
        str(_process_id(residual) or ""),
    )


def _residual_text(residual: Mapping[str, Any]) -> str:
    pieces = [
        residual.get("kind"),
        residual.get("phase"),
        residual.get("status"),
        residual.get("quality"),
    ]
    return " ".join(str(piece).lower() for piece in pieces if piece is not None)


def _quality_text(finding: Mapping[str, Any]) -> str:
    pieces = [
        finding.get("kind"),
        finding.get("severity"),
        finding.get("evidence_state"),
        finding.get("message"),
    ]
    metadata = finding.get("metadata")
    if isinstance(metadata, Mapping):
        pieces.extend(str(value) for value in metadata.values())
    return " ".join(str(piece).lower() for piece in pieces if piece is not None)


def _process_id(item: Mapping[str, Any]) -> str | None:
    process = item.get("process")
    if isinstance(process, Mapping) and isinstance(process.get("id"), str):
        return process["id"]
    entity = item.get("entity")
    if isinstance(entity, Mapping) and isinstance(entity.get("id"), str):
        return entity["id"]
    return None


def _sensor_id_from_residual(residual: Mapping[str, Any]) -> str | None:
    sensor_id = residual.get("sensor_id")
    if isinstance(sensor_id, str):
        return sensor_id
    return _process_id(residual)


def _sensor_id_from_quality(finding: Mapping[str, Any]) -> str | None:
    metadata = finding.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("sensor_id"), str):
        return metadata["sensor_id"]
    entity = finding.get("entity")
    if isinstance(entity, Mapping) and isinstance(entity.get("id"), str):
        return entity["id"]
    return None


def _residual_evidence(residuals: Sequence[Mapping[str, Any]]) -> list[str]:
    evidence: list[str] = []
    for residual in residuals:
        evidence.extend(_source_evidence(residual))
    return sorted(set(evidence))


def _symptom_evidence(symptoms: Sequence[Mapping[str, Any]]) -> list[str]:
    evidence: list[str] = []
    for symptom in symptoms:
        evidence.extend(_source_evidence(symptom))
    return sorted(set(evidence))


def _source_evidence(item: Mapping[str, Any]) -> list[str]:
    evidence = item.get("source_evidence")
    if isinstance(evidence, list):
        return [value for value in evidence if isinstance(value, str)]
    source_path = item.get("source_path")
    if isinstance(source_path, str):
        return [source_path]
    return []


def _round_score(value: float) -> float:
    return round(min(max(value, 0.0), 1.0), 6)
