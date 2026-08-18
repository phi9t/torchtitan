# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchtitan.observability.state_estimator.inference import (
    MODE_LABELS,
    score_fault_modes,
)


def _summary_with_residuals(residuals, *, quality=None, symptoms=None):
    return {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "claim_calibration": "heuristic",
        "residuals": residuals,
        "collective_symptoms": symptoms or [],
        "quality": quality or [],
    }


def _residual(
    value,
    *,
    kind="phase_duration_residual",
    phase="forward",
    step=1,
    process_id="rank0",
    evidence="obs0",
):
    return {
        "kind": kind,
        "status": "advisory",
        "quality": "present",
        "calibration": "heuristic",
        "phase": phase,
        "step": step,
        "process": {"kind": "process", "id": process_id},
        "normalized_residual": value,
        "source_evidence": [evidence],
    }


def test_scores_all_modes_with_uncalibrated_calibration_state():
    inference = score_fault_modes(
        _summary_with_residuals([_residual(0.0, evidence="nominal")])
    )

    assert [score["mode"] for score in inference["mode_scores"]] == list(MODE_LABELS)
    assert {
        score["calibration"] for score in inference["mode_scores"]
    } == {"heuristic_uncalibrated"}
    assert all(score["advisory"] for score in inference["mode_scores"])
    assert inference["mode_scores"][0]["mode"] == "nominal"


def test_single_outlier_is_capped_by_robust_residual_scoring():
    baseline = score_fault_modes(
        _summary_with_residuals(
            [_residual(value, evidence=f"weak-{idx}") for idx, value in enumerate([1.0, 1.1, 0.9])]
        )
    )
    with_outlier = score_fault_modes(
        _summary_with_residuals(
            [
                _residual(value, evidence=f"weak-{idx}")
                for idx, value in enumerate([1.0, 1.1, 0.9, 100.0])
            ]
        )
    )

    assert with_outlier["metadata"]["robust_residual_scoring"]["method"] == "winsorized_abs_residual_mean"
    assert with_outlier["metadata"]["robust_residual_scoring"]["cap"] == 6.0
    compute_baseline = _mode_score(baseline, "compute_degradation")
    compute_outlier = _mode_score(with_outlier, "compute_degradation")
    assert compute_outlier - compute_baseline < 0.35


def test_label_specific_mode_scoring_is_robust_to_one_extreme_residual():
    weak_memory = [
        _residual(
            value,
            kind="memory_hbm_residual",
            process_id=f"rank{idx}",
            evidence=f"weak-memory-{idx}",
        )
        for idx, value in enumerate([1.0, 1.1, 0.9])
    ]
    baseline = score_fault_modes(_summary_with_residuals(weak_memory))
    with_outlier = score_fault_modes(
        _summary_with_residuals(
            [
                *weak_memory,
                _residual(
                    100.0,
                    kind="memory_hbm_residual",
                    process_id="rank3",
                    evidence="extreme-memory",
                ),
                _residual(
                    100.0,
                    kind="nan_loss_residual",
                    process_id="rank4",
                    evidence="extreme-numerical",
                ),
            ]
        )
    )

    assert _mode_score(baseline, "memory_fault") > 0.0
    assert _mode_score(with_outlier, "memory_fault") - _mode_score(
        baseline, "memory_fault"
    ) < 0.35
    assert _mode_score(with_outlier, "numerical_corruption") < 0.35


def test_repeated_weak_residuals_accumulate_cusum_score():
    one_weak = score_fault_modes(_summary_with_residuals([_residual(1.2, step=1)]))
    repeated_weak = score_fault_modes(
        _summary_with_residuals([_residual(1.2, step=step) for step in range(1, 7)])
    )

    assert repeated_weak["persistent_degradation"]["method"] == "positive_cusum"
    assert repeated_weak["persistent_degradation"]["score"] > one_weak["persistent_degradation"]["score"]
    assert _mode_score(repeated_weak, "compute_degradation") > _mode_score(
        one_weak, "compute_degradation"
    )


def test_sensor_health_states_include_healthy_delayed_biased_stuck_and_missing():
    inference = score_fault_modes(
        _summary_with_residuals(
            [
                _residual(0.1, process_id="rank0", evidence="healthy"),
                _residual(3.5, process_id="rank1", evidence="biased-a"),
                _residual(3.7, process_id="rank1", evidence="biased-b"),
                _residual(0.0, process_id="rank2", evidence="stuck-a"),
            ],
            quality=[
                {
                    "kind": "clock_quality",
                    "severity": "warning",
                    "evidence_state": "delayed",
                    "entity": {"kind": "sensor", "id": "sensor:delayed"},
                    "metadata": {"sensor_id": "sensor:delayed"},
                },
                {
                    "kind": "required_signal_missing",
                    "severity": "warning",
                    "evidence_state": "missing",
                    "entity": {"kind": "sensor", "id": "sensor:missing"},
                    "metadata": {"sensor_id": "sensor:missing"},
                },
                {
                    "kind": "sensor_quality",
                    "severity": "warning",
                    "evidence_state": "present",
                    "message": "sensor value appears stuck",
                    "entity": {"kind": "sensor", "id": "sensor:rank2"},
                    "metadata": {"sensor_id": "rank2"},
                },
            ],
        )
    )

    states = {
        item["sensor_id"]: item["state"] for item in inference["sensor_health"]
    }
    assert set(states.values()) >= {"healthy", "delayed", "biased", "stuck", "missing"}
    assert states["rank0"] == "healthy"
    assert states["rank1"] == "biased"
    assert states["rank2"] == "stuck"


def test_repeated_near_zero_residuals_do_not_imply_stuck_sensor_health():
    inference = score_fault_modes(
        _summary_with_residuals(
            [
                _residual(0.0, process_id="rank0", evidence="near-zero-a"),
                _residual(0.0, process_id="rank0", evidence="near-zero-b"),
                _residual(0.01, process_id="rank0", evidence="near-zero-c"),
            ]
        )
    )

    states = {
        item["sensor_id"]: item["state"] for item in inference["sensor_health"]
    }
    assert states["rank0"] == "healthy"


def test_explicit_quality_evidence_marks_sensor_stuck():
    inference = score_fault_modes(
        _summary_with_residuals(
            [_residual(0.0, process_id="rank0", evidence="near-zero")],
            quality=[
                {
                    "kind": "sensor_quality",
                    "severity": "warning",
                    "evidence_state": "present",
                    "message": "sensor value appears stuck",
                    "entity": {"kind": "sensor", "id": "sensor:rank0"},
                    "metadata": {"sensor_id": "rank0"},
                }
            ],
        )
    )

    states = {
        item["sensor_id"]: item["state"] for item in inference["sensor_health"]
    }
    assert states["rank0"] == "stuck"


def test_root_cause_candidates_include_scope_and_evidence_chain():
    inference = score_fault_modes(
        _summary_with_residuals(
            [
                _residual(
                    4.0,
                    kind="collective_duration_residual",
                    phase="collective/all_reduce",
                    step=9,
                    process_id="rank3",
                    evidence="collective-residual",
                )
            ],
            symptoms=[
                {
                    "phase": "collective/all_reduce",
                    "step": 9,
                    "arrival_skew_ns": 10,
                    "network_progress_ns": 900_000_000,
                    "source_evidence": ["collective-symptom"],
                }
            ],
        )
    )

    candidate = inference["root_cause_candidates"][0]
    assert candidate["candidate"] == "network_degradation"
    assert candidate["affected_scope"] == {
        "phase": "collective/all_reduce",
        "processes": ["rank3"],
        "steps": [9],
    }
    assert candidate["evidence_chain"] == [
        {
            "kind": "collective_duration_residual",
            "source_evidence": ["collective-residual"],
            "summary": "normalized_residual=4.0",
        },
        {
            "kind": "collective_symptom",
            "source_evidence": ["collective-symptom"],
            "summary": "network_progress_ns=900000000 arrival_skew_ns=10",
        },
    ]
    assert candidate["calibration"] == "heuristic_uncalibrated"


def test_ambiguous_residuals_warn_and_keep_candidate_modes():
    inference = score_fault_modes(
        _summary_with_residuals(
            [
                _residual(3.0, phase="dataloader", process_id="rank0", evidence="data"),
                _residual(
                    3.1,
                    kind="collective_duration_residual",
                    phase="collective/all_reduce",
                    process_id="rank1",
                    evidence="collective",
                ),
            ]
        )
    )

    warning = inference["observability_warnings"][0]
    assert warning["kind"] == "ambiguous_fault_mode"
    assert warning["candidate_modes"] == [
        "network_degradation",
        "host_data_stall",
        "compute_degradation",
    ]
    assert "cannot distinguish" in warning["message"]
    assert inference["root_cause_candidates"][0]["candidate"] == "ambiguous"


def test_ambiguity_warnings_cover_required_non_nominal_modes():
    inference = score_fault_modes(
        _summary_with_residuals(
            [
                _residual(
                    4.0,
                    kind="memory_hbm_residual",
                    process_id="rank0",
                    evidence="memory",
                ),
                _residual(
                    4.1,
                    kind="nan_loss_residual",
                    process_id="rank1",
                    evidence="numerical",
                ),
            ]
        )
    )

    warning = inference["observability_warnings"][0]
    assert warning["kind"] == "ambiguous_fault_mode"
    assert warning["candidate_modes"] == [
        "memory_fault",
        "numerical_corruption",
    ]
    assert "cannot distinguish" in warning["message"]
    assert inference["root_cause_candidates"][0]["candidate"] == "ambiguous"


def _mode_score(inference, mode):
    return next(score["score"] for score in inference["mode_scores"] if score["mode"] == mode)
