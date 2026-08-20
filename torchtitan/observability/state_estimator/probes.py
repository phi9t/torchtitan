# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recommendation-only active diagnostic probes for estimator ambiguity."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


_INFO_GAIN_RANK = {"high": 0, "medium": 1, "low": 2}
_RISK_RANK = {"low": 0, "medium": 1, "high": 2}
_COST_RANK = {"low": 0, "medium": 1, "high": 2}
_STOPPED_PRECONDITIONS = (
    "explicit operator approval",
    "training attempt is stopped",
)


@dataclass(frozen=True, slots=True)
class ProbeRecommendation:
    kind: str
    target_entities: tuple[str, ...]
    competing_hypotheses: tuple[str, ...]
    information_target: str
    expected_information_gain_class: str
    cost_class: str
    risk_class: str
    required_authority: str
    live_safe: bool
    requires_stopped_job: bool
    preconditions: tuple[str, ...]
    expected_artifacts: tuple[str, ...]
    source_evidence: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target_entities": list(self.target_entities),
            "competing_hypotheses": list(self.competing_hypotheses),
            "information_target": self.information_target,
            "expected_information_gain_class": self.expected_information_gain_class,
            "cost_class": self.cost_class,
            "risk_class": self.risk_class,
            "required_authority": self.required_authority,
            "live_safe": self.live_safe,
            "requires_stopped_job": self.requires_stopped_job,
            "preconditions": list(self.preconditions),
            "expected_artifacts": list(self.expected_artifacts),
            "source_evidence": list(self.source_evidence),
        }


def recommend_probes(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    recommendations: list[ProbeRecommendation] = []
    recommendations.extend(_peer_skew_recommendations(summary))
    recommendations.extend(_warning_recommendations(summary))
    recommendations.extend(_fault_hypothesis_recommendations(summary))
    recommendations.extend(_transaction_recommendations(summary))
    return [
        recommendation.to_json()
        for recommendation in sorted(
            _deduplicate(recommendations),
            key=_ranking_key,
        )
    ]


def _peer_skew_recommendations(
    summary: Mapping[str, Any],
) -> list[ProbeRecommendation]:
    recommendations: list[ProbeRecommendation] = []
    for finding in summary.get("peer_skew_findings", []):
        if not isinstance(finding, Mapping):
            continue
        candidates = tuple(
            str(candidate) for candidate in finding.get("candidate_modes", [])
        )
        if (
            finding.get("root_cause") == "ambiguous"
            and "network_degradation" in candidates
            and (
                "compute_degradation" in candidates
                or "host_or_data_stall" in candidates
            )
        ):
            source_evidence = _finding_evidence(finding)
            target_entities = _target_entities_from_finding(finding)
            hypotheses = _normalize_hypotheses(candidates)
            recommendations.append(
                ProbeRecommendation(
                    kind="separate_late_arrival_from_network",
                    target_entities=target_entities,
                    competing_hypotheses=hypotheses,
                    information_target="rank arrival time versus collective progress time",
                    expected_information_gain_class="medium",
                    cost_class="medium",
                    risk_class="high",
                    required_authority="operator_stopped_job",
                    live_safe=False,
                    requires_stopped_job=True,
                    preconditions=_STOPPED_PRECONDITIONS
                    + ("diagnostic collective target selected by operator",),
                    expected_artifacts=(
                        "rank-arrival-summary.json",
                        "collective-progress-summary.json",
                    ),
                    source_evidence=source_evidence,
                )
            )
            recommendations.append(
                _make_recommendation(
                    "isolated_collective",
                    target_entities=target_entities,
                    competing_hypotheses=hypotheses,
                    information_target="collective progress independent of model compute",
                    expected_information_gain_class="medium",
                    cost_class="medium",
                    risk_class="medium",
                    required_authority="operator_diagnostic_window",
                    live_safe=True,
                    requires_stopped_job=False,
                    preconditions=(
                        "diagnostic communication window reserved",
                        "matching process-group metadata available",
                    ),
                    expected_artifacts=(
                        "isolated-collective-latency.json",
                        "collective-rank-timing.json",
                    ),
                    source_evidence=source_evidence,
                )
            )
            recommendations.append(
                _make_recommendation(
                    "stack_snapshot",
                    target_entities=target_entities,
                    competing_hypotheses=hypotheses,
                    information_target="blocked stack frame versus active compute frame",
                    expected_information_gain_class="medium",
                    cost_class="low",
                    risk_class="low",
                    required_authority="operator_diagnostic_window",
                    live_safe=True,
                    requires_stopped_job=False,
                    preconditions=(
                        "target process remains observable",
                        "stack sampler access is available",
                    ),
                    expected_artifacts=(
                        "stack-signature-summary.json",
                        "rank-stack-snapshot.txt",
                    ),
                    source_evidence=source_evidence,
                )
            )
            recommendations.append(
                _make_recommendation(
                    "profiler_window",
                    target_entities=target_entities,
                    competing_hypotheses=hypotheses,
                    information_target="phase duration split across CPU, CUDA, and collective work",
                    expected_information_gain_class="medium",
                    cost_class="medium",
                    risk_class="medium",
                    required_authority="operator_diagnostic_window",
                    live_safe=True,
                    requires_stopped_job=False,
                    preconditions=(
                        "bounded profiler schedule configured",
                        "artifact storage budget confirmed",
                    ),
                    expected_artifacts=(
                        "profiler-trace.json",
                        "profiler-summary.json",
                    ),
                    source_evidence=source_evidence,
                )
            )
            recommendations.append(
                _make_recommendation(
                    "flight_recorder_dump_preservation",
                    target_entities=target_entities,
                    competing_hypotheses=hypotheses,
                    information_target="recent collective ordering and timeout context",
                    expected_information_gain_class="medium",
                    cost_class="low",
                    risk_class="low",
                    required_authority="operator_artifact_collection",
                    live_safe=True,
                    requires_stopped_job=False,
                    preconditions=(
                        "Flight Recorder buffer was enabled",
                        "evidence bundle can preserve diagnostic files",
                    ),
                    expected_artifacts=(
                        "flight-recorder-dump.json",
                        "flight-recorder-manifest.json",
                    ),
                    source_evidence=source_evidence,
                )
            )
            recommendations.append(
                _make_recommendation(
                    "stopped_job_nccl_tests",
                    target_entities=target_entities,
                    competing_hypotheses=(
                        "network_degradation",
                        "collective_desynchronization",
                    ),
                    information_target="fabric health after training process teardown",
                    expected_information_gain_class="medium",
                    cost_class="medium",
                    risk_class="high",
                    required_authority="operator_stopped_job",
                    live_safe=False,
                    requires_stopped_job=True,
                    preconditions=_STOPPED_PRECONDITIONS
                    + ("diagnostic fabric scope selected by operator",),
                    expected_artifacts=(
                        "nccl-tests-output.txt",
                        "nccl-tests-summary.json",
                    ),
                    source_evidence=source_evidence,
                )
            )
            recommendations.append(
                _make_recommendation(
                    "stopped_job_superbench_or_eud",
                    target_entities=target_entities,
                    competing_hypotheses=(
                        "network_degradation",
                        "memory_fault",
                        "compute_degradation",
                    ),
                    information_target="hardware health outside the failed training process",
                    expected_information_gain_class="low",
                    cost_class="high",
                    risk_class="high",
                    required_authority="operator_stopped_job",
                    live_safe=False,
                    requires_stopped_job=True,
                    preconditions=_STOPPED_PRECONDITIONS
                    + ("hardware diagnostic scope selected by operator",),
                    expected_artifacts=(
                        "superbench-or-eud-output.txt",
                        "hardware-diagnostic-summary.json",
                    ),
                    source_evidence=source_evidence,
                )
            )
    return recommendations


def _warning_recommendations(summary: Mapping[str, Any]) -> list[ProbeRecommendation]:
    recommendations: list[ProbeRecommendation] = []
    for warning in summary.get("observability_warnings", []):
        if not isinstance(warning, Mapping):
            continue
        candidates = _normalize_hypotheses(
            str(candidate) for candidate in warning.get("candidate_modes", [])
        )
        if len(candidates) < 2:
            continue
        evidence = _source_evidence(warning)
        targets = _target_entities_from_scope(warning)
        recommendations.append(
            _make_recommendation(
                "same_microbatch_replay_another_gpu",
                target_entities=targets,
                competing_hypotheses=candidates,
                information_target="whether the same microbatch follows the data or the device",
                expected_information_gain_class="high",
                cost_class="low",
                risk_class="low",
                required_authority="operator_offline_replay",
                live_safe=True,
                requires_stopped_job=False,
                preconditions=(
                    "microbatch identity and input artifact are available",
                    "compatible replay harness is available",
                ),
                expected_artifacts=(
                    "microbatch-replay-result.json",
                    "device-comparison-summary.json",
                ),
                source_evidence=evidence,
            )
        )
        recommendations.append(
            _make_recommendation(
                "same_data_replay_different_placement",
                target_entities=targets,
                competing_hypotheses=candidates,
                information_target="whether the same data changes behavior under a different placement",
                expected_information_gain_class="high",
                cost_class="medium",
                risk_class="low",
                required_authority="operator_offline_replay",
                live_safe=True,
                requires_stopped_job=False,
                preconditions=(
                    "data cursor and checkpoint reference are available",
                    "alternate placement is selected by operator",
                ),
                expected_artifacts=(
                    "placement-replay-result.json",
                    "placement-comparison-summary.json",
                ),
                source_evidence=evidence,
            )
        )
        recommendations.append(
            _make_recommendation(
                "independent_sensor_query",
                target_entities=targets,
                competing_hypotheses=candidates,
                information_target="independent hardware or host signal for the same incident window",
                expected_information_gain_class="high",
                cost_class="low",
                risk_class="low",
                required_authority="operator_artifact_collection",
                live_safe=True,
                requires_stopped_job=False,
                preconditions=(
                    "incident window is known",
                    "independent telemetry source is reachable",
                ),
                expected_artifacts=(
                    "independent-sensor-query.json",
                    "sensor-health-summary.json",
                ),
                source_evidence=evidence,
            )
        )
    return recommendations


def _fault_hypothesis_recommendations(
    summary: Mapping[str, Any],
) -> list[ProbeRecommendation]:
    recommendations: list[ProbeRecommendation] = []
    for hypothesis in summary.get("fault_hypotheses", []):
        if not isinstance(hypothesis, Mapping):
            continue
        mode = hypothesis.get("mode")
        if not isinstance(mode, str):
            continue
        targets = _target_entities_from_scope(hypothesis)
        evidence = _source_evidence(hypothesis)
        if mode == "compute_degradation":
            recommendations.append(
                _make_recommendation(
                    "local_compute_canary",
                    target_entities=targets,
                    competing_hypotheses=(
                        "compute_degradation",
                        "normal_workload_skew",
                    ),
                    information_target="local compute throughput versus expected device baseline",
                    expected_information_gain_class="medium",
                    cost_class="low",
                    risk_class="low",
                    required_authority="operator_diagnostic_window",
                    live_safe=True,
                    requires_stopped_job=False,
                    preconditions=(
                        "target device identity is known",
                        "bounded canary artifact path is available",
                    ),
                    expected_artifacts=(
                        "compute-canary-result.json",
                        "device-throughput-summary.json",
                    ),
                    source_evidence=evidence,
                )
            )
        elif mode == "numerical_corruption":
            recommendations.append(
                _make_recommendation(
                    "bf16_fp32_replay_numerical_instability",
                    target_entities=targets,
                    competing_hypotheses=(
                        "numerical_corruption",
                        "precision_instability",
                    ),
                    information_target="BF16 versus FP32 divergence for the suspected sample",
                    expected_information_gain_class="medium",
                    cost_class="medium",
                    risk_class="low",
                    required_authority="operator_offline_replay",
                    live_safe=True,
                    requires_stopped_job=False,
                    preconditions=(
                        "sample and model checkpoint references are available",
                        "full precision replay path is available",
                    ),
                    expected_artifacts=(
                        "bf16-fp32-replay.json",
                        "numeric-diff-summary.json",
                    ),
                    source_evidence=evidence,
                )
            )
        elif mode == "control_plane_fault":
            recommendations.append(
                _make_recommendation(
                    "prior_code_version_replay",
                    target_entities=targets,
                    competing_hypotheses=("code_regression", "environment_regression"),
                    information_target="whether prior code version reproduces the same control symptom",
                    expected_information_gain_class="low",
                    cost_class="high",
                    risk_class="medium",
                    required_authority="operator_offline_replay",
                    live_safe=True,
                    requires_stopped_job=False,
                    preconditions=(
                        "prior code version and configuration are known",
                        "artifact lineage is sufficient for comparison",
                    ),
                    expected_artifacts=(
                        "prior-version-replay.json",
                        "version-comparison-summary.json",
                    ),
                    source_evidence=evidence,
                )
            )
    return recommendations


def _transaction_recommendations(
    summary: Mapping[str, Any],
) -> list[ProbeRecommendation]:
    transactions = summary.get("transactions")
    if not isinstance(transactions, Mapping):
        return []
    advisories = [
        advisory
        for advisory in transactions.get("risk_advisories", [])
        if isinstance(advisory, Mapping)
        and advisory.get("kind")
        in {
            "checkpoint_lineage_incomplete",
            "checkpoint_restore_validation_absent",
        }
    ]
    if not advisories:
        return []
    kinds = {str(advisory.get("kind")) for advisory in advisories}
    hypotheses: list[str] = []
    if "checkpoint_lineage_incomplete" in kinds:
        hypotheses.append("checkpoint_lineage_gap")
    if "checkpoint_restore_validation_absent" in kinds:
        hypotheses.append("restore_validation_gap")
    source_evidence = _stable_unique(
        evidence for advisory in advisories for evidence in _source_evidence(advisory)
    )
    return [
        _make_recommendation(
            "checkpoint_lineage_validation",
            target_entities=("checkpoint_lineage",),
            competing_hypotheses=tuple(hypotheses),
            information_target="checkpoint parent, shard inventory, and restore validation evidence",
            expected_information_gain_class="medium",
            cost_class="low",
            risk_class="low",
            required_authority="operator_artifact_collection",
            live_safe=True,
            requires_stopped_job=False,
            preconditions=(
                "checkpoint artifact index is available",
                "transaction advisory remains evidence-only",
            ),
            expected_artifacts=(
                "checkpoint-lineage-validation.json",
                "restore-validation-summary.json",
            ),
            source_evidence=source_evidence,
        )
    ]


def _make_recommendation(
    kind: str,
    *,
    target_entities: Sequence[str],
    competing_hypotheses: Sequence[str],
    information_target: str,
    expected_information_gain_class: str,
    cost_class: str,
    risk_class: str,
    required_authority: str,
    live_safe: bool,
    requires_stopped_job: bool,
    preconditions: Sequence[str],
    expected_artifacts: Sequence[str],
    source_evidence: Sequence[str],
) -> ProbeRecommendation:
    return ProbeRecommendation(
        kind=kind,
        target_entities=tuple(_stable_unique(target_entities)) or ("unknown",),
        competing_hypotheses=tuple(_stable_unique(competing_hypotheses)),
        information_target=information_target,
        expected_information_gain_class=expected_information_gain_class,
        cost_class=cost_class,
        risk_class=risk_class,
        required_authority=required_authority,
        live_safe=live_safe,
        requires_stopped_job=requires_stopped_job,
        preconditions=tuple(_stable_unique(preconditions)),
        expected_artifacts=tuple(_stable_unique(expected_artifacts)),
        source_evidence=tuple(_stable_unique(source_evidence)) or ("summary",),
    )


def _deduplicate(
    recommendations: Sequence[ProbeRecommendation],
) -> list[ProbeRecommendation]:
    by_kind: dict[str, ProbeRecommendation] = {}
    for recommendation in recommendations:
        existing = by_kind.get(recommendation.kind)
        if existing is None:
            by_kind[recommendation.kind] = recommendation
            continue
        by_kind[recommendation.kind] = _make_recommendation(
            recommendation.kind,
            target_entities=(
                *existing.target_entities,
                *recommendation.target_entities,
            ),
            competing_hypotheses=(
                *existing.competing_hypotheses,
                *recommendation.competing_hypotheses,
            ),
            information_target=existing.information_target,
            expected_information_gain_class=min(
                existing.expected_information_gain_class,
                recommendation.expected_information_gain_class,
                key=lambda item: _INFO_GAIN_RANK[item],
            ),
            cost_class=min(
                existing.cost_class,
                recommendation.cost_class,
                key=lambda item: _COST_RANK[item],
            ),
            risk_class=min(
                existing.risk_class,
                recommendation.risk_class,
                key=lambda item: _RISK_RANK[item],
            ),
            required_authority=existing.required_authority,
            live_safe=existing.live_safe and recommendation.live_safe,
            requires_stopped_job=(
                existing.requires_stopped_job or recommendation.requires_stopped_job
            ),
            preconditions=(*existing.preconditions, *recommendation.preconditions),
            expected_artifacts=(
                *existing.expected_artifacts,
                *recommendation.expected_artifacts,
            ),
            source_evidence=(
                *existing.source_evidence,
                *recommendation.source_evidence,
            ),
        )
    return list(by_kind.values())


def _ranking_key(recommendation: ProbeRecommendation) -> tuple[int, int, int, str, str]:
    return (
        _INFO_GAIN_RANK[recommendation.expected_information_gain_class],
        _RISK_RANK[recommendation.risk_class],
        _COST_RANK[recommendation.cost_class],
        "|".join(recommendation.source_evidence),
        recommendation.kind,
    )


def _normalize_hypotheses(candidates: Iterable[str]) -> tuple[str, ...]:
    aliases = {
        "host_or_data_stall": "host_data_stall",
    }
    return tuple(
        _stable_unique(aliases.get(candidate, candidate) for candidate in candidates)
    )


def _target_entities_from_finding(finding: Mapping[str, Any]) -> tuple[str, ...]:
    latest = finding.get("latest_entity")
    if isinstance(latest, Mapping):
        entity_id = latest.get("id")
        if isinstance(entity_id, str) and entity_id:
            return (entity_id,)
    entity = finding.get("entity")
    if isinstance(entity, str) and entity:
        return (entity,)
    phase = finding.get("phase")
    step = finding.get("step")
    if isinstance(phase, str) and isinstance(step, int):
        return (f"{phase}:step-{step}",)
    return ("peer_skew_scope",)


def _target_entities_from_scope(item: Mapping[str, Any]) -> tuple[str, ...]:
    scope = item.get("affected_scope")
    if isinstance(scope, Mapping):
        entities = scope.get("entities")
        if isinstance(entities, Sequence) and not isinstance(entities, (str, bytes)):
            values = [str(entity) for entity in entities if str(entity)]
            if values:
                return tuple(_stable_unique(values))
    return ("diagnostic_scope",)


def _finding_evidence(finding: Mapping[str, Any]) -> tuple[str, ...]:
    evidence = _source_evidence(finding)
    if evidence != ("summary",):
        return evidence
    phase = finding.get("phase")
    step = finding.get("step")
    if isinstance(phase, str) and isinstance(step, int):
        return (f"peer skew in {phase} step {step}",)
    return evidence


def _source_evidence(item: Mapping[str, Any]) -> tuple[str, ...]:
    direct = item.get("source_evidence")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        values = [str(evidence) for evidence in direct if str(evidence)]
        if values:
            return tuple(_stable_unique(values))
    evidence = item.get("evidence")
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
        values = [str(entry) for entry in evidence if str(entry)]
        if values:
            return tuple(_stable_unique(values))
    chain = item.get("evidence_chain")
    if isinstance(chain, Sequence) and not isinstance(chain, (str, bytes)):
        values = []
        for entry in chain:
            if isinstance(entry, Mapping):
                observation_id = entry.get("observation_id")
                if isinstance(observation_id, str) and observation_id:
                    values.append(observation_id)
            elif str(entry):
                values.append(str(entry))
        if values:
            return tuple(_stable_unique(values))
    return ("summary",)


def _stable_unique(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value})
