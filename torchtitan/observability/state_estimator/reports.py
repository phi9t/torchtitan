# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Operator-facing reports for offline state-estimator analysis."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from torchtitan.observability.state_estimator.bundle import load_bundle
from torchtitan.observability.state_estimator.estimators import estimate_belief
from torchtitan.observability.state_estimator.evaluation import (
    EvaluationCase,
    EvaluationManifest,
    evaluate_manifest,
)
from torchtitan.observability.state_estimator.graph import build_evidence_graph
from torchtitan.observability.state_estimator.probes import recommend_probes
from torchtitan.observability.state_estimator.schema import (
    SCHEMA_VERSION,
    read_json_object,
    write_json_atomic,
)
from torchtitan.observability.state_estimator.timeline import build_incident_timeline


@dataclass(frozen=True, slots=True)
class AttemptReportArtifacts:
    output_dir: Path
    evidence_graph: Path
    entities: Path
    timeline: Path
    quality: Path
    belief_summary: Path
    operator_report: Path

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> "AttemptReportArtifacts":
        return cls(
            output_dir=output_dir,
            evidence_graph=output_dir / "evidence_graph.json",
            entities=output_dir / "entities.json",
            timeline=output_dir / "timeline.jsonl",
            quality=output_dir / "quality.json",
            belief_summary=output_dir / "belief_summary.json",
            operator_report=output_dir / "operator_report.md",
        )

    def to_json(self) -> dict[str, str]:
        return {
            "evidence_graph": str(self.evidence_graph),
            "entities": str(self.entities),
            "timeline": str(self.timeline),
            "quality": str(self.quality),
            "belief_summary": str(self.belief_summary),
            "operator_report": str(self.operator_report),
        }


def analyze_attempt_report(attempt_path: Path, output_dir: Path) -> dict[str, Any]:
    artifacts = AttemptReportArtifacts.from_output_dir(output_dir)
    graph, summary = _build_attempt_analysis(attempt_path)
    _write_attempt_artifacts(graph, summary, artifacts)
    return {
        "ok": True,
        "command": "analyze-attempt",
        "schema_version": SCHEMA_VERSION,
        "run_id": graph.get("run_id"),
        "attempt_id": graph.get("attempt_id"),
        "attempt_path": str(attempt_path),
        "output_dir": str(output_dir),
        "artifacts": artifacts.to_json(),
    }


def analyze_run_report(run_path: Path, output_dir: Path) -> dict[str, Any]:
    attempt_paths = _discover_attempt_paths(run_path)
    attempts: dict[str, dict[str, Any]] = {}
    attempt_artifacts: dict[str, dict[str, str]] = {}
    for attempt_path in attempt_paths:
        attempt_output_dir = output_dir / attempt_path.name
        result = analyze_attempt_report(attempt_path, attempt_output_dir)
        attempt_id = str(result["attempt_id"])
        attempts[attempt_id] = result
        attempt_artifacts[attempt_id] = result["artifacts"]

    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "schema_version": SCHEMA_VERSION,
        "run_path": str(run_path),
        "attempt_count": len(attempt_paths),
        "attempts": sorted(attempts),
        "attempt_summaries": {
            attempt_id: _attempt_digest(result)
            for attempt_id, result in sorted(attempts.items())
        },
    }
    write_json_atomic(output_dir / "run_summary.json", run_summary)
    (output_dir / "run_report.md").write_text(
        "\n".join(_run_report_lines(run_summary))
    )
    return {
        "ok": True,
        "command": "analyze-run",
        "schema_version": SCHEMA_VERSION,
        "run_path": str(run_path),
        "output_dir": str(output_dir),
        "attempt_count": len(attempt_paths),
        "attempts": sorted(attempts),
        "attempt_artifacts": attempt_artifacts,
        "artifacts": {
            "run_summary": str(output_dir / "run_summary.json"),
            "run_report": str(output_dir / "run_report.md"),
        },
    }


def compare_attempt_reports(
    attempt_paths: Sequence[Path], output_dir: Path
) -> dict[str, Any]:
    if len(attempt_paths) < 2:
        raise ValueError("compare-attempts requires at least two --attempt-path values")
    identities = [
        _read_attempt_identity(attempt_path) for attempt_path in attempt_paths
    ]
    use_run_attempt_keys = (
        len({identity["attempt_id"] for identity in identities}) != len(identities)
        or len({attempt_path.name for attempt_path in attempt_paths})
        != len(attempt_paths)
    )
    comparisons: dict[str, dict[str, Any]] = {}
    attempt_artifacts: dict[str, dict[str, str]] = {}
    for attempt_path, identity in zip(attempt_paths, identities):
        attempt_key = _comparison_attempt_key(
            identity, use_run_attempt_keys=use_run_attempt_keys
        )
        result = analyze_attempt_report(attempt_path, output_dir / attempt_key)
        comparisons[attempt_key] = result
        attempt_artifacts[attempt_key] = result["artifacts"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "attempt_count": len(attempt_paths),
        "attempts": sorted(comparisons),
        "attempt_summaries": {
            attempt_id: _attempt_digest(result)
            for attempt_id, result in sorted(comparisons.items())
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "comparison_summary.json", summary)
    (output_dir / "comparison_report.md").write_text(
        "\n".join(_comparison_report_lines(summary))
    )
    return {
        "ok": True,
        "command": "compare-attempts",
        "schema_version": SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "attempt_count": len(attempt_paths),
        "attempts": sorted(comparisons),
        "attempt_artifacts": attempt_artifacts,
        "artifacts": {
            "comparison_summary": str(output_dir / "comparison_summary.json"),
            "comparison_report": str(output_dir / "comparison_report.md"),
        },
    }


def evaluate_report(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = _read_evaluation_manifest(manifest_path)
    summaries: dict[str, Mapping[str, Any]] = {}
    case_artifacts: dict[str, dict[str, str]] = {}
    for case in manifest.cases:
        case_output_dir = output_dir / "cases" / case.id
        result = analyze_attempt_report(Path(case.attempt_path), case_output_dir)
        case_artifacts[case.id] = result["artifacts"]
        summaries[case.id] = _read_json(case_output_dir / "belief_summary.json")
    aggregate = evaluate_manifest(manifest, summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "evaluation_summary.json", aggregate)
    (output_dir / "evaluation_report.md").write_text(
        "\n".join(_evaluation_report_lines(manifest, aggregate))
    )
    return {
        "ok": True,
        "command": "evaluate",
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "case_count": len(manifest.cases),
        "case_artifacts": case_artifacts,
        "artifacts": {
            "evaluation_summary": str(output_dir / "evaluation_summary.json"),
            "evaluation_report": str(output_dir / "evaluation_report.md"),
        },
    }


def _build_attempt_analysis(attempt_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        load_bundle(attempt_path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{attempt_path / 'manifest.json'}: {exc}") from exc
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
    return graph, summary


def _write_attempt_artifacts(
    graph: Mapping[str, Any],
    summary: Mapping[str, Any],
    artifacts: AttemptReportArtifacts,
) -> None:
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(artifacts.evidence_graph, graph)
    write_json_atomic(
        artifacts.entities,
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": graph.get("run_id"),
            "attempt_id": graph.get("attempt_id"),
            "entities": list(graph.get("entities", [])),
        },
    )
    write_json_atomic(
        artifacts.quality,
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": graph.get("run_id"),
            "attempt_id": graph.get("attempt_id"),
            "findings": list(graph.get("quality", [])),
        },
    )
    artifacts.timeline.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in graph.get("observations", [])
        )
    )
    write_json_atomic(artifacts.belief_summary, summary)
    artifacts.operator_report.write_text(
        "\n".join(_operator_report_lines(graph, summary))
    )


def _operator_report_lines(
    graph: Mapping[str, Any], summary: Mapping[str, Any]
) -> list[str]:
    topology_layers = graph.get("layers", [])
    analytical = summary.get("analytical", {})
    inference = summary.get("inference", {})
    transactions = summary.get("transactions", {})
    return [
        "# Training State Estimator Operator Report",
        "",
        f"- run_id: {graph.get('run_id')}",
        f"- attempt_id: {graph.get('attempt_id')}",
        "",
        "## Evidence Quality",
        "",
        f"- findings: {len(graph.get('quality', []))}",
        f"- warnings: {len(summary.get('observability_warnings', []))}",
        "",
        "## Topology",
        "",
        f"- entities: {len(graph.get('entities', []))}",
        f"- edges: {len(graph.get('edges', []))}",
        (
            "- topology_layers: "
            f"{len(topology_layers) if isinstance(topology_layers, list) else 0}"
        ),
        "",
        "## Timeline",
        "",
        f"- observations: {len(graph.get('observations', []))}",
        f"- incident_events: {len(summary.get('incident_timeline', []))}",
        "",
        "## Analytical Residuals",
        "",
        f"- phase_duration_count: {len(summary.get('phase_durations', []))}",
        (
            "- analytical_keys: "
            f"{', '.join(sorted(analytical)) if isinstance(analytical, Mapping) else 'none'}"
        ),
        "",
        "## Mode Scores",
        "",
        *_mode_score_lines(summary, inference),
        "",
        "## Root-Cause Candidates",
        "",
        *_root_cause_lines(summary),
        "",
        "## Transaction Risk",
        "",
        _transaction_text(transactions),
        "",
        "## Probes",
        "",
        *_probe_lines(summary),
        "",
        "## Learned Factors",
        "",
        "- state: not_requested",
        "- advisory: true",
        "",
        "## Evaluation Metrics",
        "",
        "- state: not_evaluated",
        "- use the evaluate subcommand with a labeled manifest for metrics",
        "",
    ]


def _mode_score_lines(
    summary: Mapping[str, Any], inference: Any
) -> list[str]:
    lines = [
        f"- {item.get('mode')}: {item.get('score')}"
        for item in summary.get("fault_hypotheses", [])
        if isinstance(item, Mapping)
    ]
    if isinstance(inference, Mapping):
        lines.extend(
            f"- {item.get('mode')}: {item.get('score')}"
            for item in inference.get("mode_scores", [])
            if isinstance(item, Mapping)
        )
    return lines or ["- none"]


def _root_cause_lines(summary: Mapping[str, Any]) -> list[str]:
    candidates = [
        item
        for item in summary.get("root_cause_candidates", [])
        if isinstance(item, Mapping)
    ]
    if not candidates and isinstance(summary.get("inference"), Mapping):
        candidates = [
            item
            for item in summary["inference"].get("root_cause_candidates", [])
            if isinstance(item, Mapping)
        ]
    return [
        f"- {item.get('candidate')}: {item.get('score')}"
        for item in candidates
    ] or ["- none"]


def _probe_lines(summary: Mapping[str, Any]) -> list[str]:
    return [
        f"- {probe.get('kind')}: {probe.get('information_target')}"
        for probe in summary.get("probe_recommendations", [])
        if isinstance(probe, Mapping)
    ] or ["- none"]


def _transaction_text(transactions: Any) -> str:
    if isinstance(transactions, Mapping):
        report_text = transactions.get("report_text")
        if isinstance(report_text, str):
            return report_text
    return "- state: unavailable"


def _discover_attempt_paths(run_path: Path) -> list[Path]:
    attempts = [
        path
        for path in sorted(run_path.iterdir())
        if path.is_dir() and (path / "manifest.json").exists()
    ]
    if not attempts:
        raise ValueError(f"{run_path}: no attempt directories with manifest.json found")
    return attempts


def _read_attempt_identity(attempt_path: Path) -> dict[str, str]:
    manifest = read_json_object(attempt_path / "manifest.json")
    return {
        "run_id": _required_manifest_string(manifest, "run_id", attempt_path),
        "attempt_id": _required_manifest_string(manifest, "attempt_id", attempt_path),
    }


def _required_manifest_string(
    manifest: Mapping[str, Any], field_name: str, attempt_path: Path
) -> str:
    value = manifest.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{attempt_path / 'manifest.json'}: {field_name} must be a non-empty string"
        )
    return value


def _comparison_attempt_key(
    identity: Mapping[str, str], *, use_run_attempt_keys: bool
) -> str:
    if not use_run_attempt_keys:
        return identity["attempt_id"]
    return str(Path(identity["run_id"]) / identity["attempt_id"])


def _attempt_digest(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = _read_json(Path(result["artifacts"]["belief_summary"]))
    return {
        "run_id": result.get("run_id"),
        "attempt_id": result.get("attempt_id"),
        "fault_hypotheses": len(summary.get("fault_hypotheses", [])),
        "observability_warnings": len(summary.get("observability_warnings", [])),
        "probe_recommendations": len(summary.get("probe_recommendations", [])),
    }


def _run_report_lines(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Training State Estimator Run Report",
        "",
        f"- run_path: {summary['run_path']}",
        f"- attempt_count: {summary['attempt_count']}",
        "",
        "## Evidence Quality",
        "",
        "- see per-attempt operator reports",
        "",
        "## Topology",
        "",
        "- see per-attempt operator reports",
        "",
        "## Timeline",
        "",
        "- see per-attempt operator reports",
        "",
        "## Analytical Residuals",
        "",
        "- see per-attempt operator reports",
        "",
        "## Mode Scores",
        "",
        "- see per-attempt operator reports",
        "",
        "## Root-Cause Candidates",
        "",
        "- see per-attempt operator reports",
        "",
        "## Transaction Risk",
        "",
        "- see per-attempt operator reports",
        "",
        "## Probes",
        "",
        "- see per-attempt operator reports",
        "",
        "## Learned Factors",
        "",
        "- see per-attempt operator reports",
        "",
        "## Evaluation Metrics",
        "",
        "- state: not_evaluated",
        "",
        "## Attempts",
        "",
    ]
    for attempt_id, digest in summary["attempt_summaries"].items():
        lines.append(
            f"- {attempt_id}: fault_hypotheses={digest['fault_hypotheses']} "
            f"warnings={digest['observability_warnings']} probes={digest['probe_recommendations']}"
        )
    return lines + [""]


def _comparison_report_lines(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Training State Estimator Attempt Comparison",
        "",
        f"- attempt_count: {summary['attempt_count']}",
        "",
        "## Evidence Quality",
        "",
        "- compare per-attempt warning counts below",
        "",
        "## Topology",
        "",
        "- see per-attempt operator reports",
        "",
        "## Timeline",
        "",
        "- see per-attempt operator reports",
        "",
        "## Analytical Residuals",
        "",
        "- see per-attempt operator reports",
        "",
        "## Mode Scores",
        "",
        "- compare per-attempt fault hypothesis counts below",
        "",
        "## Root-Cause Candidates",
        "",
        "- see per-attempt operator reports",
        "",
        "## Transaction Risk",
        "",
        "- see per-attempt operator reports",
        "",
        "## Probes",
        "",
        "- compare per-attempt probe recommendation counts below",
        "",
        "## Learned Factors",
        "",
        "- see per-attempt operator reports",
        "",
        "## Evaluation Metrics",
        "",
        "- state: not_evaluated",
        "",
        "## Attempts",
        "",
    ]
    for attempt_id, digest in summary["attempt_summaries"].items():
        lines.append(
            f"- {attempt_id}: fault_hypotheses={digest['fault_hypotheses']} "
            f"warnings={digest['observability_warnings']} probes={digest['probe_recommendations']}"
        )
    return lines + [""]


def _evaluation_report_lines(
    manifest: EvaluationManifest, aggregate: Mapping[str, Any]
) -> list[str]:
    detection = aggregate.get("detection", {})
    localization = aggregate.get("localization", {})
    calibration = aggregate.get("calibration", {})
    return [
        "# Training State Estimator Evaluation Report",
        "",
        f"- manifest_id: {manifest.id}",
        f"- case_count: {aggregate.get('case_count')}",
        "",
        "## Evidence Quality",
        "",
        "- state: evaluated per case artifact",
        "",
        "## Topology",
        "",
        "- state: evaluated per case artifact",
        "",
        "## Timeline",
        "",
        "- state: evaluated per case artifact",
        "",
        "## Analytical Residuals",
        "",
        "- state: evaluated per case artifact",
        "",
        "## Mode Scores",
        "",
        f"- expected_mode_found: {aggregate.get('expected_mode_found_count')}",
        "",
        "## Root-Cause Candidates",
        "",
        f"- top_1_root_cause: {localization.get('top_1_root_cause_count')}",
        f"- top_3_root_cause: {localization.get('top_3_root_cause_count')}",
        "",
        "## Transaction Risk",
        "",
        "- state: advisory only",
        "",
        "## Probes",
        "",
        f"- expected_probe_found: {aggregate.get('recommended_probe_found_count')}",
        "",
        "## Learned Factors",
        "",
        (
            "- calibration_families: "
            f"{', '.join(sorted(calibration)) if isinstance(calibration, Mapping) and calibration else 'none'}"
        ),
        "",
        "## Evaluation Metrics",
        "",
        f"- detected: {aggregate.get('detected_count')}",
        f"- false_alerts: {aggregate.get('false_alert_count')}",
        (
            "- detection_latency_ns_mean: "
            f"{detection.get('detection_latency_ns', {}).get('mean') if isinstance(detection, Mapping) else None}"
        ),
        "",
    ]


def _read_evaluation_manifest(path: Path) -> EvaluationManifest:
    data = _read_json(path)
    cases_value = data.get("cases")
    if not isinstance(cases_value, list):
        raise ValueError("evaluation manifest is missing cases list")
    cases = [_evaluation_case(item) for item in cases_value]
    manifest_id = data.get("id")
    if not isinstance(manifest_id, str) or not manifest_id:
        raise ValueError("evaluation manifest is missing id")
    score_label_families = data.get("score_label_families", {})
    if not isinstance(score_label_families, Mapping):
        raise ValueError("score_label_families must be a JSON object")
    return EvaluationManifest(
        id=manifest_id,
        cases=cases,
        score_label_families={
            str(key): bool(value) for key, value in score_label_families.items()
        },
    )


def _evaluation_case(data: Any) -> EvaluationCase:
    if not isinstance(data, Mapping):
        raise ValueError("evaluation manifest case must be a JSON object")
    required = ("id", "attempt_path")
    for field in required:
        if not isinstance(data.get(field), str) or not data.get(field):
            raise ValueError(f"evaluation manifest case is missing {field}")
    return EvaluationCase(
        id=data["id"],
        attempt_path=data["attempt_path"],
        expected_mode=_optional_string(data.get("expected_mode")),
        expected_probe=_optional_string(data.get("expected_probe")),
        nominal=bool(data.get("nominal", False)),
        case_class=str(data.get("case_class", "legacy")),
        onset_time_ns=_optional_int(data.get("onset_time_ns")),
        affected_scope=_optional_mapping(data.get("affected_scope")),
        expected_root_cause=_optional_string(data.get("expected_root_cause")),
        expected_warning_kinds=tuple(_string_list(data.get("expected_warning_kinds"))),
        expected_artifact_bytes=_optional_int(data.get("expected_artifact_bytes")),
        overhead_metrics=_float_mapping(data.get("overhead_metrics")),
        real_training_labels=bool(data.get("real_training_labels", False)),
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional string field must be a string or null")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("optional integer field must be an integer or null")
    return value


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("optional mapping field must be a JSON object or null")
    return dict(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("string list field must contain only strings")
    return list(value)


def _float_mapping(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("overhead_metrics must be a JSON object")
    output: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("overhead metric values must be numeric")
        output[str(key)] = float(item)
    return output


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value
