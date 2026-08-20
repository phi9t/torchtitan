# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

import pytest

from torchtitan.observability.state_estimator.evaluation import (
    compute_calibration_metrics,
    CpuFakeBackendEvaluationCase,
    evaluate_case,
    evaluate_manifest,
    EvaluationCase,
    EvaluationManifest,
    EvaluationMetricSummary,
    InjectedFaultEvaluationCase,
    RealSmallRunEvaluationCase,
    SyntheticEvaluationCase,
    write_calibration_report,
    write_evaluation_report,
)


def test_evaluation_detects_expected_fault_mode_and_probe():
    case = EvaluationCase(
        id="case-1",
        attempt_path="fixture",
        expected_mode="peer_relative_skew",
        expected_probe="separate_late_arrival_from_network",
        nominal=False,
    )
    summary = {
        "fault_hypotheses": [{"mode": "peer_relative_skew"}],
        "probe_recommendations": [{"kind": "separate_late_arrival_from_network"}],
        "observability_warnings": [],
    }

    result = evaluate_case(case, summary).to_json()

    assert result["detected"] is True
    assert result["expected_mode_found"] is True
    assert result["recommended_probe_found"] is True
    assert result["false_alert"] is False


def test_nominal_case_counts_unexpected_hypothesis_as_false_alert():
    case = EvaluationCase(
        id="nominal",
        attempt_path="fixture",
        expected_mode=None,
        expected_probe=None,
        nominal=True,
    )
    summary = {
        "fault_hypotheses": [{"mode": "peer_relative_skew"}],
        "probe_recommendations": [],
        "observability_warnings": [{"kind": "missing_optional_artifact"}],
    }

    result = evaluate_case(case, summary).to_json()

    assert result["detected"] is True
    assert result["false_alert"] is True
    assert result["observability_warning_count"] == 1


def test_manifest_supports_all_case_classes_and_separate_metric_sections(tmp_path):
    manifest = EvaluationManifest(
        id="issue-19-fixture",
        cases=[
            SyntheticEvaluationCase(
                id="synthetic-network",
                attempt_path="attempts/synthetic-network",
                expected_mode="network_degradation",
                expected_probe="isolated_collective",
                nominal=False,
                onset_time_ns=1_000,
                affected_scope={"processes": ["rank0", "rank1"], "steps": [7]},
                expected_root_cause="network_degradation",
                expected_warning_kinds=["ambiguous_fault_mode"],
                expected_artifact_bytes=2048,
            ),
            CpuFakeBackendEvaluationCase(
                id="cpu-nominal",
                attempt_path="attempts/cpu-nominal",
                expected_mode=None,
                expected_probe=None,
                nominal=True,
            ),
            RealSmallRunEvaluationCase(
                id="real-compute",
                attempt_path="attempts/real-compute",
                expected_mode="compute_degradation",
                expected_probe="profiler_window",
                nominal=False,
                real_training_labels=True,
                overhead_metrics={"runtime_overhead_pct": 0.7},
            ),
            InjectedFaultEvaluationCase(
                id="injected-storage",
                attempt_path="attempts/injected-storage",
                expected_mode="storage_fault",
                expected_probe="checkpoint_lineage_validation",
                nominal=False,
                injected_fault="checkpoint_corruption",
            ),
        ],
        score_label_families={"mode_scores": True},
    )
    summaries = {
        "synthetic-network": _summary(
            detected_time_ns=1_500,
            modes=[{"mode": "network_degradation", "score": 0.8}],
            probes=["isolated_collective"],
            root_causes=[
                {
                    "candidate": "network_degradation",
                    "affected_scope": {
                        "processes": ["rank0", "rank1"],
                        "steps": [7],
                    },
                }
            ],
            warnings=["ambiguous_fault_mode"],
            artifact_bytes=2048,
        ),
        "cpu-nominal": _summary(
            modes=[],
            probes=[],
            root_causes=[],
            warnings=[],
            artifact_bytes=128,
        ),
        "real-compute": _summary(
            modes=[{"mode": "compute_degradation", "score": 0.65}],
            probes=["profiler_window"],
            root_causes=[
                {
                    "candidate": "compute_degradation",
                    "affected_scope": {"processes": ["rank2"], "steps": [3]},
                }
            ],
            artifact_bytes=4096,
            overhead_metrics={"runtime_overhead_pct": 0.7},
        ),
        "injected-storage": _summary(
            modes=[{"mode": "storage_fault", "score": 0.9}],
            probes=["checkpoint_lineage_validation"],
            root_causes=[
                {
                    "candidate": "storage_fault",
                    "affected_scope": {"processes": ["rank0"], "steps": [10]},
                }
            ],
            artifact_bytes=1024,
        ),
    }

    aggregate = evaluate_manifest(manifest, summaries)
    written = write_evaluation_report(manifest.cases, summaries, tmp_path)

    assert [case["case_class"] for case in aggregate["cases"]] == [
        "synthetic",
        "cpu_fake_backend",
        "real_small_run",
        "injected_fault",
    ]
    assert aggregate["detection"] == {
        "case_count": 4,
        "detected_count": 3,
        "false_alert_count": 0,
        "expected_mode_found_count": 4,
        "detection_latency_ns": {
            "count": 1,
            "min": 500,
            "max": 500,
            "mean": 500.0,
        },
    }
    assert aggregate["localization"]["top_1_root_cause_count"] == 3
    assert aggregate["localization"]["top_3_root_cause_count"] == 3
    assert aggregate["localization"]["failure_domain_scope_score"] == 1.0
    assert aggregate["probe_action"]["expected_probe_found_count"] == 4
    assert aggregate["observability"]["warning_correct_count"] == 4
    assert aggregate["resource"]["artifact_bytes"] == {
        "total": 7296,
        "mean_per_case": 1824.0,
    }
    assert aggregate["resource"]["overhead_metrics"] == {
        "runtime_overhead_pct": {
            "count": 1,
            "mean": 0.7,
            "max": 0.7,
        }
    }
    assert aggregate["calibration"]["mode_scores"]["state"] == "calibrated"
    assert written["localization"]["top_1_root_cause_count"] == 3

    summary_json = json.loads((tmp_path / "evaluation_summary.json").read_text())
    assert set(summary_json) >= {
        "detection",
        "localization",
        "probe_action",
        "observability",
        "resource",
        "calibration",
    }
    report = (tmp_path / "evaluation_report.md").read_text()
    assert "Detection Metrics" in report
    assert "Localization Metrics" in report
    assert "Probe And Action Metrics" in report
    assert "Calibration Metrics" in report
    assert "Detection counts do not imply localization accuracy." in report
    assert "Calibration is reported only for score families with labels." in report


def test_calibration_refuses_to_mark_unlabeled_score_family_calibrated():
    manifest = EvaluationManifest(
        id="unlabeled-calibration",
        cases=[
            SyntheticEvaluationCase(
                id="case-1",
                attempt_path="attempts/case-1",
                expected_mode="network_degradation",
                expected_probe=None,
                nominal=False,
            )
        ],
        score_label_families={},
    )

    aggregate = evaluate_manifest(
        manifest,
        {
            "case-1": _summary(
                modes=[{"mode": "network_degradation", "score": 0.9}],
                probes=[],
                root_causes=[],
            )
        },
    )

    assert aggregate["calibration"]["mode_scores"] == {
        "state": "uncalibrated",
        "reason": "manifest lacks labels for score family mode_scores",
    }


def test_synthetic_only_calibration_is_not_reported_as_generic_calibrated():
    manifest = EvaluationManifest(
        id="synthetic-only-calibration",
        cases=[
            SyntheticEvaluationCase(
                id="case-1",
                attempt_path="attempts/case-1",
                expected_mode="network_degradation",
                expected_probe=None,
                nominal=False,
            )
        ],
        score_label_families={"mode_scores": True},
    )

    aggregate = evaluate_manifest(
        manifest,
        {
            "case-1": _summary(
                modes=[{"mode": "network_degradation", "score": 0.9}],
                probes=[],
                root_causes=[],
            )
        },
    )

    calibration = aggregate["calibration"]["mode_scores"]
    assert calibration["state"] == "synthetic_calibrated"
    assert calibration["count"] == 1
    assert calibration["brier_score"] == pytest.approx(0.01)
    assert calibration["expected_calibration_error"] == pytest.approx(0.1)


def test_calibration_metrics_use_labeled_probabilities():
    metrics = compute_calibration_metrics(
        [
            {"score": 0.9, "label": True},
            {"score": 0.8, "label": True},
            {"score": 0.2, "label": False},
            {"score": 0.1, "label": False},
        ],
        num_bins=2,
    )

    assert metrics == {
        "state": "calibrated",
        "count": 4,
        "brier_score": pytest.approx(0.025),
        "expected_calibration_error": pytest.approx(0.15),
    }


def test_metric_summary_and_calibration_report_write_labeled_metrics(tmp_path):
    metric_summary = EvaluationMetricSummary(
        name="mode_scores",
        metrics={"state": "calibrated", "brier_score": 0.1},
    )

    output = write_calibration_report(
        {"mode_scores": metric_summary.to_json()},
        tmp_path,
    )

    assert output == {"mode_scores": {"state": "calibrated", "brier_score": 0.1}}
    assert json.loads((tmp_path / "calibration_summary.json").read_text()) == output
    assert "mode_scores" in (tmp_path / "calibration_report.md").read_text()


def test_top_k_root_cause_and_scope_score_allow_partial_localization_credit():
    case = SyntheticEvaluationCase(
        id="case-1",
        attempt_path="attempts/case-1",
        expected_mode="network_degradation",
        expected_probe=None,
        nominal=False,
        affected_scope={"processes": ["rank0", "rank1"], "steps": [7]},
        expected_root_cause="network_degradation",
    )

    result = evaluate_case(
        case,
        _summary(
            modes=[{"mode": "network_degradation", "score": 0.7}],
            probes=[],
            root_causes=[
                {
                    "candidate": "compute_degradation",
                    "affected_scope": {"processes": ["rank2"], "steps": [7]},
                },
                {
                    "candidate": "network_degradation",
                    "affected_scope": {
                        "processes": ["rank0", "rank2"],
                        "steps": [7],
                    },
                },
            ],
        ),
    ).to_json()

    assert result["top_1_root_cause_found"] is False
    assert result["top_3_root_cause_found"] is True
    assert result["failure_domain_scope_score"] == pytest.approx(0.5)


def test_expected_probe_warning_artifacts_and_overhead_are_case_metrics():
    case = RealSmallRunEvaluationCase(
        id="case-1",
        attempt_path="attempts/case-1",
        expected_mode="compute_degradation",
        expected_probe="profiler_window",
        nominal=False,
        expected_warning_kinds=["missing_optional_artifact"],
        expected_artifact_bytes=512,
        overhead_metrics={"tier0_overhead_pct": 0.4},
    )

    result = evaluate_case(
        case,
        _summary(
            modes=[{"mode": "compute_degradation", "score": 0.7}],
            probes=["profiler_window"],
            warnings=["missing_optional_artifact"],
            artifact_bytes=512,
            overhead_metrics={"tier0_overhead_pct": 0.4},
        ),
    ).to_json()

    assert result["recommended_probe_found"] is True
    assert result["observability_warning_correct"] is True
    assert result["artifact_bytes"] == 512
    assert result["overhead_metrics"] == {"tier0_overhead_pct": 0.4}


def _summary(
    *,
    modes,
    probes,
    root_causes=None,
    warnings=None,
    detected_time_ns=None,
    artifact_bytes=None,
    overhead_metrics=None,
):
    return {
        "fault_hypotheses": modes,
        "inference": {"mode_scores": modes},
        "probe_recommendations": [{"kind": probe} for probe in probes],
        "root_cause_candidates": root_causes or [],
        "observability_warnings": [{"kind": warning} for warning in (warnings or [])],
        "detected_time_ns": detected_time_ns,
        "artifact_bytes": artifact_bytes,
        "overhead_metrics": overhead_metrics or {},
    }
