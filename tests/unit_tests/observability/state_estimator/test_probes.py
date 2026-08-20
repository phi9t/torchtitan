# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchtitan.observability.state_estimator.probes import recommend_probes


REQUIRED_FIELDS = {
    "kind",
    "target_entities",
    "competing_hypotheses",
    "information_target",
    "expected_information_gain_class",
    "cost_class",
    "risk_class",
    "required_authority",
    "live_safe",
    "requires_stopped_job",
    "preconditions",
    "expected_artifacts",
    "source_evidence",
}


def test_peer_skew_recommends_compute_and_collective_separation():
    summary = {
        "peer_skew_findings": [
            {
                "phase": "train/step",
                "step": 1,
                "root_cause": "ambiguous",
                "candidate_modes": [
                    "host_or_data_stall",
                    "compute_degradation",
                    "network_degradation",
                ],
            }
        ],
        "observability_warnings": [],
        "fault_hypotheses": [],
    }

    recommendations = recommend_probes(summary)
    recommendation = _find(recommendations, "separate_late_arrival_from_network")

    assert recommendation["live_safe"] is False
    assert recommendation["required_authority"] == "operator_stopped_job"
    assert recommendation["requires_stopped_job"] is True


def test_no_ambiguity_yields_no_probe():
    summary = {
        "peer_skew_findings": [],
        "observability_warnings": [],
        "fault_hypotheses": [
            {"mode": "process_failure", "score": 1.0, "calibration": "heuristic"}
        ],
    }

    assert recommend_probes(summary) == []


def test_active_planner_recommends_all_initial_probe_families_with_metadata():
    summary = _summary_for_all_probe_families()

    recommendations = recommend_probes(summary)
    by_kind = {
        recommendation["kind"]: recommendation for recommendation in recommendations
    }

    assert {
        "local_compute_canary",
        "isolated_collective",
        "same_microbatch_replay_another_gpu",
        "same_data_replay_different_placement",
        "independent_sensor_query",
        "stack_snapshot",
        "profiler_window",
        "flight_recorder_dump_preservation",
        "checkpoint_lineage_validation",
        "bf16_fp32_replay_numerical_instability",
        "stopped_job_nccl_tests",
        "stopped_job_superbench_or_eud",
        "prior_code_version_replay",
    }.issubset(by_kind)

    for recommendation in recommendations:
        assert REQUIRED_FIELDS <= recommendation.keys()
        assert isinstance(recommendation["target_entities"], list)
        assert recommendation["target_entities"]
        assert recommendation["expected_information_gain_class"] in {
            "high",
            "medium",
            "low",
        }
        assert recommendation["cost_class"] in {"low", "medium", "high"}
        assert recommendation["risk_class"] in {"low", "medium", "high"}
        assert isinstance(recommendation["required_authority"], str)
        assert recommendation["required_authority"]
        assert isinstance(recommendation["live_safe"], bool)
        assert isinstance(recommendation["requires_stopped_job"], bool)
        assert isinstance(recommendation["preconditions"], list)
        assert recommendation["preconditions"]
        assert isinstance(recommendation["expected_artifacts"], list)
        assert recommendation["expected_artifacts"]
        assert isinstance(recommendation["source_evidence"], list)
        assert recommendation["source_evidence"]


def test_ranking_is_deterministic_and_prioritizes_ambiguity_risk_cost_and_evidence():
    summary = _summary_for_all_probe_families()

    first = recommend_probes(summary)
    second = recommend_probes(summary)

    assert first == second
    assert [_ranking_fields(recommendation) for recommendation in first] == sorted(
        _ranking_fields(recommendation) for recommendation in first
    )
    assert {
        first[0]["expected_information_gain_class"],
        first[1]["expected_information_gain_class"],
        first[2]["expected_information_gain_class"],
    } == {"high"}
    assert first.index(_find(first, "stopped_job_nccl_tests")) < first.index(
        _find(first, "stopped_job_superbench_or_eud")
    )


def test_stopped_job_probes_require_explicit_authority_and_are_not_live_safe():
    recommendations = recommend_probes(_summary_for_all_probe_families())
    stopped = [
        recommendation
        for recommendation in recommendations
        if recommendation["requires_stopped_job"]
    ]

    assert stopped
    assert {recommendation["kind"] for recommendation in stopped} >= {
        "separate_late_arrival_from_network",
        "stopped_job_nccl_tests",
        "stopped_job_superbench_or_eud",
    }
    for recommendation in stopped:
        assert recommendation["required_authority"] == "operator_stopped_job"
        assert recommendation["live_safe"] is False
        assert "explicit operator approval" in recommendation["preconditions"]


def test_recommendations_are_advisory_and_do_not_use_execution_wording():
    recommendations = recommend_probes(_summary_for_all_probe_families())
    forbidden_terms = {
        "execute",
        "executed",
        "launch",
        "launched",
        "run",
        "ran",
        "start",
        "started",
        "reroute",
        "fence",
        "rollback",
        "evict",
        "retry",
    }

    for recommendation in recommendations:
        text = " ".join(
            str(value)
            for field in (
                "kind",
                "information_target",
                "required_authority",
                "preconditions",
                "expected_artifacts",
                "source_evidence",
            )
            for value in (
                recommendation[field]
                if isinstance(recommendation[field], list)
                else [recommendation[field]]
            )
        ).lower()
        words = {word.strip("`.,:;()[]{}") for word in text.split()}
        assert forbidden_terms.isdisjoint(words)


def test_checkpoint_lineage_validation_uses_transaction_advisories():
    summary = {
        "peer_skew_findings": [],
        "observability_warnings": [],
        "fault_hypotheses": [],
        "transactions": {
            "risk_advisories": [
                {
                    "kind": "checkpoint_lineage_incomplete",
                    "missing_fields": ["checkpoint_parent"],
                    "source_evidence": ["obs-checkpoint-save"],
                },
                {
                    "kind": "checkpoint_restore_validation_absent",
                    "missing_fields": ["restore_validation_status"],
                    "source_evidence": ["obs-checkpoint-load"],
                },
            ]
        },
    }

    recommendation = _find(recommend_probes(summary), "checkpoint_lineage_validation")

    assert recommendation["competing_hypotheses"] == [
        "checkpoint_lineage_gap",
        "restore_validation_gap",
    ]
    assert recommendation["source_evidence"] == [
        "obs-checkpoint-load",
        "obs-checkpoint-save",
    ]


def _find(recommendations: list[dict[str, object]], kind: str) -> dict[str, object]:
    return next(
        recommendation
        for recommendation in recommendations
        if recommendation["kind"] == kind
    )


def _ranking_fields(
    recommendation: dict[str, object]
) -> tuple[int, int, int, str, str]:
    return (
        {"high": 0, "medium": 1, "low": 2}[
            str(recommendation["expected_information_gain_class"])
        ],
        {"low": 0, "medium": 1, "high": 2}[str(recommendation["risk_class"])],
        {"low": 0, "medium": 1, "high": 2}[str(recommendation["cost_class"])],
        "|".join(str(item) for item in recommendation["source_evidence"]),
        str(recommendation["kind"]),
    )


def _summary_for_all_probe_families() -> dict[str, object]:
    return {
        "peer_skew_findings": [
            {
                "phase": "collective/all_reduce",
                "step": 7,
                "latest_entity": {"kind": "process", "id": "rank-3"},
                "root_cause": "ambiguous",
                "candidate_modes": [
                    "host_or_data_stall",
                    "compute_degradation",
                    "network_degradation",
                    "normal_workload_skew",
                ],
            }
        ],
        "observability_warnings": [
            {
                "kind": "ambiguous_fault_mode",
                "candidate_modes": [
                    "network_degradation",
                    "host_data_stall",
                    "compute_degradation",
                ],
                "evidence": ["residual:collective:rank-3"],
            },
            {
                "kind": "fault hypotheses without independent sensor or probe evidence",
                "candidate_modes": ["memory_fault", "control_plane_fault"],
                "evidence": ["sensor-gap:rank-3"],
            },
        ],
        "fault_hypotheses": [
            {
                "mode": "compute_degradation",
                "score": 0.91,
                "affected_scope": {"entities": ["rank-3", "gpu-3"]},
                "evidence_chain": [{"observation_id": "compute-residual:rank-3"}],
            },
            {
                "mode": "network_degradation",
                "score": 0.9,
                "affected_scope": {"entities": ["rank-3", "nic-0"]},
                "evidence_chain": [{"observation_id": "network-residual:rank-3"}],
            },
            {
                "mode": "host_data_stall",
                "score": 0.89,
                "affected_scope": {"entities": ["rank-3", "host-a"]},
                "evidence_chain": [{"observation_id": "host-residual:rank-3"}],
            },
            {
                "mode": "numerical_corruption",
                "score": 0.85,
                "affected_scope": {"entities": ["rank-3", "gpu-3"]},
                "evidence_chain": [{"observation_id": "nan:rank-3"}],
            },
            {
                "mode": "storage_fault",
                "score": 0.78,
                "affected_scope": {"entities": ["checkpoint-12"]},
                "evidence_chain": [{"observation_id": "checkpoint-gap"}],
            },
            {
                "mode": "control_plane_fault",
                "score": 0.74,
                "affected_scope": {"entities": ["trainer-controller"]},
                "evidence_chain": [{"observation_id": "controller-gap"}],
            },
        ],
        "transactions": {
            "risk_advisories": [
                {
                    "kind": "checkpoint_lineage_incomplete",
                    "missing_fields": [
                        "checkpoint_parent",
                        "checkpoint_shard_inventory",
                    ],
                    "source_evidence": ["checkpoint-save:12"],
                }
            ]
        },
    }
