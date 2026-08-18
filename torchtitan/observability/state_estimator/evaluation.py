# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluation helpers for offline training state-estimator outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from torchtitan.observability.state_estimator.schema import (
    SCHEMA_VERSION,
    write_json_atomic,
)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    attempt_path: str
    expected_mode: str | None
    expected_probe: str | None
    nominal: bool
    case_class: str = "legacy"
    onset_time_ns: int | None = None
    affected_scope: Mapping[str, Any] | None = None
    expected_root_cause: str | None = None
    expected_warning_kinds: Sequence[str] = ()
    expected_artifact_bytes: int | None = None
    overhead_metrics: Mapping[str, float] = field(default_factory=dict)
    real_training_labels: bool = False


@dataclass(frozen=True, slots=True)
class SyntheticEvaluationCase(EvaluationCase):
    case_class: str = "synthetic"


@dataclass(frozen=True, slots=True)
class CpuFakeBackendEvaluationCase(EvaluationCase):
    case_class: str = "cpu_fake_backend"


@dataclass(frozen=True, slots=True)
class RealSmallRunEvaluationCase(EvaluationCase):
    case_class: str = "real_small_run"
    real_training_labels: bool = True


@dataclass(frozen=True, slots=True)
class InjectedFaultEvaluationCase(EvaluationCase):
    injected_fault: str | None = None
    case_class: str = "injected_fault"


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    id: str
    cases: Sequence[EvaluationCase]
    score_label_families: Mapping[str, bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationMetricSummary:
    name: str
    metrics: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return dict(self.metrics)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    case_id: str
    case_class: str
    detected: bool
    expected_mode_found: bool
    recommended_probe_found: bool
    false_alert: bool
    observability_warning_count: int
    detection_latency_ns: int | None = None
    top_1_root_cause_found: bool | None = None
    top_3_root_cause_found: bool | None = None
    failure_domain_scope_score: float | None = None
    observability_warning_correct: bool | None = None
    artifact_bytes: int | None = None
    overhead_metrics: Mapping[str, float] = field(default_factory=dict)
    calibration_labels: Mapping[str, Sequence[Mapping[str, Any]]] = field(
        default_factory=dict
    )

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_class": self.case_class,
            "detected": self.detected,
            "expected_mode_found": self.expected_mode_found,
            "recommended_probe_found": self.recommended_probe_found,
            "false_alert": self.false_alert,
            "observability_warning_count": self.observability_warning_count,
            "detection_latency_ns": self.detection_latency_ns,
            "top_1_root_cause_found": self.top_1_root_cause_found,
            "top_3_root_cause_found": self.top_3_root_cause_found,
            "failure_domain_scope_score": self.failure_domain_scope_score,
            "observability_warning_correct": self.observability_warning_correct,
            "artifact_bytes": self.artifact_bytes,
            "overhead_metrics": dict(self.overhead_metrics),
            "calibration_labels": {
                family: [dict(item) for item in labels]
                for family, labels in self.calibration_labels.items()
            },
        }


def evaluate_case(
    case: EvaluationCase, summary: Mapping[str, Any]
) -> EvaluationResult:
    mode_items = _mode_items(summary)
    modes = {item.get("mode") for item in mode_items}
    probes = {item.get("kind") for item in summary.get("probe_recommendations", [])}
    root_causes = [
        item
        for item in summary.get("root_cause_candidates", [])
        if isinstance(item, Mapping)
    ]
    detected = bool(modes)
    expected_mode_found = (
        case.expected_mode in modes if case.expected_mode else not detected
    )
    recommended_probe_found = (
        case.expected_probe in probes if case.expected_probe else True
    )
    false_alert = case.nominal and detected
    expected_root_cause = case.expected_root_cause or case.expected_mode
    top_1_root_cause_found = _top_k_root_cause_found(
        root_causes, expected_root_cause, k=1
    )
    top_3_root_cause_found = _top_k_root_cause_found(
        root_causes, expected_root_cause, k=3
    )
    failure_domain_scope_score = _best_scope_score(
        root_causes, expected_root_cause, case.affected_scope
    )
    detection_latency_ns = _detection_latency_ns(case, summary, detected)
    warning_kinds = {
        item.get("kind")
        for item in summary.get("observability_warnings", [])
        if isinstance(item, Mapping)
    }
    expected_warning_kinds = set(case.expected_warning_kinds)
    observability_warning_correct = expected_warning_kinds.issubset(warning_kinds)
    if not expected_warning_kinds and warning_kinds:
        observability_warning_correct = False
    artifact_bytes = _int_or_none(summary.get("artifact_bytes"))
    return EvaluationResult(
        case_id=case.id,
        case_class=case.case_class,
        detected=detected,
        expected_mode_found=expected_mode_found,
        recommended_probe_found=recommended_probe_found,
        false_alert=false_alert,
        observability_warning_count=len(summary.get("observability_warnings", [])),
        detection_latency_ns=detection_latency_ns,
        top_1_root_cause_found=top_1_root_cause_found,
        top_3_root_cause_found=top_3_root_cause_found,
        failure_domain_scope_score=failure_domain_scope_score,
        observability_warning_correct=observability_warning_correct,
        artifact_bytes=artifact_bytes,
        overhead_metrics=_numeric_mapping(summary.get("overhead_metrics", {})),
        calibration_labels={
            "mode_scores": _mode_calibration_labels(case, mode_items),
        },
    )


def evaluate_manifest(
    manifest: EvaluationManifest,
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    results = [
        evaluate_case(case, summaries[case.id]).to_json()
        for case in manifest.cases
    ]
    return _aggregate_results(
        results, score_label_families=manifest.score_label_families
    )


def compute_calibration_metrics(
    labeled_scores: Sequence[Mapping[str, Any]],
    *,
    num_bins: int = 10,
) -> dict[str, Any]:
    if not labeled_scores:
        return {"state": "uncalibrated", "reason": "no labeled scores"}
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    pairs = [
        (float(item["score"]), 1.0 if bool(item["label"]) else 0.0)
        for item in labeled_scores
    ]
    brier_score = sum((score - label) ** 2 for score, label in pairs) / len(pairs)
    bins: list[list[tuple[float, float]]] = [[] for _ in range(num_bins)]
    for score, label in pairs:
        clamped_score = max(0.0, min(1.0, score))
        index = min(int(clamped_score * num_bins), num_bins - 1)
        bins[index].append((clamped_score, label))
    expected_calibration_error = 0.0
    for bucket in bins:
        if not bucket:
            continue
        confidence = sum(score for score, _ in bucket) / len(bucket)
        accuracy = sum(label for _, label in bucket) / len(bucket)
        expected_calibration_error += (
            len(bucket) / len(pairs) * abs(confidence - accuracy)
        )
    return {
        "state": "calibrated",
        "count": len(pairs),
        "brier_score": brier_score,
        "expected_calibration_error": expected_calibration_error,
    }


def write_evaluation_report(
    cases: Sequence[EvaluationCase],
    summaries: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    results = [evaluate_case(case, summaries[case.id]).to_json() for case in cases]
    aggregate = _aggregate_results(results, score_label_families={})
    write_json_atomic(output_dir / "evaluation_summary.json", aggregate)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = _report_lines(aggregate)
    (output_dir / "evaluation_report.md").write_text("\n".join(lines))
    return aggregate


def write_calibration_report(
    calibration_summary: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    output = {
        family: dict(metrics) for family, metrics in sorted(calibration_summary.items())
    }
    write_json_atomic(output_dir / "calibration_summary.json", output)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Training State Estimator Calibration", ""]
    for family, metrics in output.items():
        lines.extend(
            [
                f"## {family}",
                "",
                f"- state: {metrics.get('state')}",
                f"- brier_score: {metrics.get('brier_score')}",
                f"- expected_calibration_error: {metrics.get('expected_calibration_error')}",
                "",
            ]
        )
    lines.append("Calibration requires labels for each reported score family.")
    (output_dir / "calibration_report.md").write_text("\n".join(lines))
    return output


def _aggregate_results(
    results: Sequence[Mapping[str, Any]],
    *,
    score_label_families: Mapping[str, bool],
) -> dict[str, Any]:
    detection_latencies = [
        result["detection_latency_ns"]
        for result in results
        if result.get("detection_latency_ns") is not None
    ]
    scope_scores = [
        result["failure_domain_scope_score"]
        for result in results
        if result.get("failure_domain_scope_score") is not None
    ]
    artifact_bytes = [
        result["artifact_bytes"]
        for result in results
        if result.get("artifact_bytes") is not None
    ]
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "case_count": len(results),
        "detected_count": sum(1 for result in results if result["detected"]),
        "false_alert_count": sum(1 for result in results if result["false_alert"]),
        "expected_mode_found_count": sum(
            1 for result in results if result["expected_mode_found"]
        ),
        "recommended_probe_found_count": sum(
            1 for result in results if result["recommended_probe_found"]
        ),
        "detection": {
            "case_count": len(results),
            "detected_count": sum(1 for result in results if result["detected"]),
            "false_alert_count": sum(1 for result in results if result["false_alert"]),
            "expected_mode_found_count": sum(
                1 for result in results if result["expected_mode_found"]
            ),
            "detection_latency_ns": _number_summary(detection_latencies),
        },
        "localization": {
            "top_1_root_cause_count": sum(
                1 for result in results if result.get("top_1_root_cause_found")
            ),
            "top_3_root_cause_count": sum(
                1 for result in results if result.get("top_3_root_cause_found")
            ),
            "failure_domain_scope_score": _mean(scope_scores),
        },
        "probe_action": {
            "expected_probe_found_count": sum(
                1 for result in results if result["recommended_probe_found"]
            )
        },
        "observability": {
            "warning_correct_count": sum(
                1
                for result in results
                if result.get("observability_warning_correct") is True
            ),
            "warning_count": sum(
                int(result.get("observability_warning_count", 0))
                for result in results
            ),
        },
        "resource": {
            "artifact_bytes": {
                "total": sum(artifact_bytes),
                "mean_per_case": _mean(artifact_bytes),
            },
            "overhead_metrics": _overhead_summary(results),
        },
        "calibration": _calibration_summary(results, score_label_families),
        "cases": list(results),
        "results": list(results),
    }
    return aggregate


def _calibration_summary(
    results: Sequence[Mapping[str, Any]],
    score_label_families: Mapping[str, bool],
) -> dict[str, Any]:
    families = sorted(
        {
            family
            for result in results
            for family in result.get("calibration_labels", {})
        }
        | set(score_label_families)
    )
    summary: dict[str, Any] = {}
    for family in families:
        if not score_label_families.get(family, False):
            summary[family] = {
                "state": "uncalibrated",
                "reason": f"manifest lacks labels for score family {family}",
            }
            continue
        labels: list[Mapping[str, Any]] = []
        for result in results:
            labels.extend(result.get("calibration_labels", {}).get(family, []))
        metrics = compute_calibration_metrics(labels)
        if labels:
            case_classes = sorted(
                {
                    str(label["case_class"])
                    for label in labels
                    if label.get("case_class") is not None
                }
            )
            metrics["case_classes"] = case_classes
            if metrics.get("state") == "calibrated" and case_classes == ["synthetic"]:
                metrics["state"] = "synthetic_calibrated"
        summary[family] = metrics
    return summary


def _mode_calibration_labels(
    case: EvaluationCase, mode_items: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    labels = []
    for item in mode_items:
        mode = item.get("mode")
        score = item.get("score")
        if mode is None or score is None:
            continue
        labels.append(
            {
                "score": float(score),
                "label": mode == case.expected_mode,
                "case_class": case.case_class,
                "provenance": _calibration_label_provenance(case),
            }
        )
    return labels


def _calibration_label_provenance(case: EvaluationCase) -> str:
    if case.case_class == "synthetic":
        return "synthetic"
    if case.real_training_labels:
        return "real_training"
    return case.case_class


def _mode_items(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = summary.get("fault_hypotheses", [])
    if not candidates and isinstance(summary.get("inference"), Mapping):
        candidates = summary["inference"].get("mode_scores", [])
    return [item for item in candidates if isinstance(item, Mapping)]


def _top_k_root_cause_found(
    root_causes: Sequence[Mapping[str, Any]],
    expected_root_cause: str | None,
    *,
    k: int,
) -> bool | None:
    if expected_root_cause is None:
        return None
    return any(
        item.get("candidate") == expected_root_cause for item in root_causes[:k]
    )


def _best_scope_score(
    root_causes: Sequence[Mapping[str, Any]],
    expected_root_cause: str | None,
    expected_scope: Mapping[str, Any] | None,
) -> float | None:
    if expected_root_cause is None or not expected_scope:
        return None
    scores = [
        _scope_score(expected_scope, item.get("affected_scope", {}))
        for item in root_causes
        if item.get("candidate") == expected_root_cause
    ]
    if not scores:
        return 0.0
    return max(scores)


def _scope_score(expected_scope: Mapping[str, Any], actual_scope: Any) -> float:
    if not isinstance(actual_scope, Mapping):
        return 0.0
    expected_entities: set[tuple[str, Any]] = set()
    actual_entities: set[tuple[str, Any]] = set()
    for field, expected_values in expected_scope.items():
        expected_entities.update((field, item) for item in _as_set(expected_values))
        actual_entities.update(
            (field, item) for item in _as_set(actual_scope.get(field))
        )
    if not expected_entities and not actual_entities:
        return 0.0
    union = expected_entities | actual_entities
    if not union:
        return 0.0
    return len(expected_entities & actual_entities) / len(union)


def _as_set(value: Any) -> set[Any]:
    if value is None:
        return set()
    if isinstance(value, (str, int, float, bool)):
        return {value}
    try:
        return set(value)
    except TypeError:
        return {value}


def _detection_latency_ns(
    case: EvaluationCase, summary: Mapping[str, Any], detected: bool
) -> int | None:
    if not detected or case.onset_time_ns is None:
        return None
    detected_time_ns = _int_or_none(summary.get("detected_time_ns"))
    if detected_time_ns is None:
        return None
    return detected_time_ns - case.onset_time_ns


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _numeric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result = {}
    for key, item in value.items():
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            result[str(key)] = float(item)
    return result


def _number_summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": _mean(values),
    }


def _mean(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _overhead_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_name: dict[str, list[float]] = {}
    for result in results:
        for name, value in result.get("overhead_metrics", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                by_name.setdefault(str(name), []).append(float(value))
    return {
        name: {"count": len(values), "mean": _mean(values), "max": max(values)}
        for name, values in sorted(by_name.items())
    }


def _report_lines(aggregate: Mapping[str, Any]) -> list[str]:
    detection = aggregate["detection"]
    localization = aggregate["localization"]
    probe_action = aggregate["probe_action"]
    observability = aggregate["observability"]
    resource = aggregate["resource"]
    calibration = aggregate["calibration"]
    return [
        "# Training State Estimator Evaluation",
        "",
        "## Detection Metrics",
        "",
        f"- cases: {detection['case_count']}",
        f"- detected: {detection['detected_count']}",
        f"- false_alerts: {detection['false_alert_count']}",
        f"- expected_mode_found: {detection['expected_mode_found_count']}",
        f"- detection_latency_ns_mean: {detection['detection_latency_ns']['mean']}",
        "",
        "Detection counts do not imply localization accuracy.",
        "",
        "## Localization Metrics",
        "",
        f"- top_1_root_cause: {localization['top_1_root_cause_count']}",
        f"- top_3_root_cause: {localization['top_3_root_cause_count']}",
        f"- failure_domain_scope_score: {localization['failure_domain_scope_score']}",
        "",
        "Localization metrics do not imply probe safety or utility.",
        "",
        "## Probe And Action Metrics",
        "",
        f"- expected_probe_found: {probe_action['expected_probe_found_count']}",
        "",
        "Probe recommendations are advisory and are evaluated separately from detection.",
        "",
        "## Observability Metrics",
        "",
        f"- warning_correct: {observability['warning_correct_count']}",
        f"- warning_count: {observability['warning_count']}",
        "",
        "## Resource And Overhead Metrics",
        "",
        f"- artifact_bytes_total: {resource['artifact_bytes']['total']}",
        f"- artifact_bytes_mean_per_case: {resource['artifact_bytes']['mean_per_case']}",
        f"- overhead_metric_families: {len(resource['overhead_metrics'])}",
        "",
        "## Calibration Metrics",
        "",
        f"- score_families: {', '.join(sorted(calibration)) if calibration else 'none'}",
        "",
        "Calibration is reported only for score families with labels.",
        "",
    ]
