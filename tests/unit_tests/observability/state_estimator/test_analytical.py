# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchtitan.observability.state_estimator.analytical import (
    build_analytical_summary,
    predict_ring_all_reduce_duration,
)


def test_phase_summaries_include_process_rank_phase_step_and_predecessor():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [
            _event(
                "rank0:forward",
                process_id="rank0",
                phase="forward",
                step=7,
                global_rank=0,
                start_time_ns=1_000,
                end_time_ns=1_800,
            ),
            _event(
                "rank0:backward",
                process_id="rank0",
                phase="backward",
                step=7,
                global_rank=0,
                start_time_ns=1_900,
                end_time_ns=2_700,
                predecessor_phase="forward",
                predecessor_step=7,
            ),
        ],
    }

    summary = build_analytical_summary(graph)

    assert summary["phase_durations"] == [
        {
            "process": {"kind": "process", "id": "rank0"},
            "global_rank": 0,
            "phase": "backward",
            "step": 7,
            "duration_ns": 800,
            "clock": "time_ns",
            "start_time_ns": 1_900,
            "end_time_ns": 2_700,
            "source_evidence": ["rank0:backward"],
            "calibration": "heuristic",
            "quality": "present",
        },
        {
            "process": {"kind": "process", "id": "rank0"},
            "global_rank": 0,
            "phase": "forward",
            "step": 7,
            "duration_ns": 800,
            "clock": "time_ns",
            "start_time_ns": 1_000,
            "end_time_ns": 1_800,
            "source_evidence": ["rank0:forward"],
            "calibration": "heuristic",
            "quality": "present",
        },
    ]
    assert summary["max_plus_predecessors"] == [
        {
            "process": {"kind": "process", "id": "rank0"},
            "global_rank": 0,
            "phase": "backward",
            "step": 7,
            "predecessor_phase": "forward",
            "predecessor_step": 7,
            "source_evidence": ["rank0:backward"],
            "calibration": "heuristic",
            "quality": "present",
        }
    ]


def test_ring_all_reduce_prediction_uses_workload_metadata():
    prediction = predict_ring_all_reduce_duration(
        message_bytes=8_000,
        group_size=4,
        latency_ns=100,
        effective_bandwidth_bytes_per_ns=10,
    )

    assert prediction == {
        "model": "ring_all_reduce",
        "message_bytes": 8_000,
        "group_size": 4,
        "latency_ns": 100,
        "effective_bandwidth_bytes_per_ns": 10,
        "predicted_duration_ns": 1_800,
        "calibration": "heuristic",
        "quality": "present",
    }


def test_missing_all_reduce_workload_metadata_emits_explicit_fallback():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [
            _collective(
                "rank0:collective",
                process_id="rank0",
                step=2,
                global_rank=0,
                launch_time_ns=1_000,
                completion_time_ns=1_900,
                group_size=4,
            )
        ],
    }

    summary = build_analytical_summary(graph)

    assert summary["collective_predictions"] == []
    assert summary["residuals"] == [
        {
            "kind": "collective_prediction_unavailable",
            "status": "metadata_missing",
            "quality": "uncollected",
            "calibration": "heuristic",
            "advisory": (
                "No analytical all-reduce duration is predicted because "
                "workload metadata is incomplete."
            ),
            "source_evidence": ["rank0:collective"],
            "metadata": {
                "missing_fields": [
                    "message_bytes",
                    "latency_ns",
                    "effective_bandwidth_bytes_per_ns",
                ],
                "phase": "collective/all_reduce",
                "step": 2,
            },
        }
    ]


def test_collective_symptom_split_and_residual_do_not_claim_root_cause():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [
            _collective(
                "rank0:collective",
                process_id="rank0",
                step=2,
                global_rank=0,
                launch_time_ns=1_000,
                completion_time_ns=2_100,
                message_bytes=8_000,
                group_size=4,
                latency_ns=100,
                effective_bandwidth_bytes_per_ns=10,
            ),
            _collective(
                "rank1:collective",
                process_id="rank1",
                step=2,
                global_rank=1,
                launch_time_ns=1_300,
                completion_time_ns=2_800,
                message_bytes=8_000,
                group_size=4,
                latency_ns=100,
                effective_bandwidth_bytes_per_ns=10,
            ),
        ],
    }

    summary = build_analytical_summary(graph)

    assert summary["collective_symptoms"] == [
        {
            "phase": "collective/all_reduce",
            "step": 2,
            "group_size": 4,
            "arrival_skew_ns": 300,
            "network_progress_ns": 1_500,
            "launch_clock": "time_ns",
            "completion_clock": "time_ns",
            "source_evidence": ["rank0:collective", "rank1:collective"],
            "calibration": "heuristic",
            "quality": "present",
        }
    ]
    residual = summary["residuals"][0]
    assert residual["kind"] == "collective_duration_residual"
    assert residual["observed_duration_ns"] == 1_100
    assert residual["predicted_duration_ns"] == 1_800
    assert residual["expected_sigma_ns"] == 1_800
    assert residual["sigma_calibration"] == "heuristic_predicted_duration_scale"
    assert residual["normalized_residual"] == -0.3888888888888889
    assert residual["calibration"] == "heuristic"
    assert residual["advisory"] == (
        "Residual is advisory and indicates model mismatch or missing factors; "
        "it does not identify a root cause by itself."
    )
    assert "root_cause" not in residual


def test_checkpoint_stage_save_load_summaries_from_artifact_rows():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [
            {
                "id": "artifact:stage",
                "kind": "artifact",
                "entity": {"kind": "process", "id": "rank0"},
                "event_time_ns": 1_000,
                "source_path": "indexes/artifact_rows/0",
                "payload": {
                    "artifact_id": "checkpoint-step-7",
                    "kind": "checkpoint",
                    "phase": "checkpoint/stage",
                    "step": 7,
                    "global_rank": 0,
                    "start_time_ns": 1_000,
                    "end_time_ns": 1_200,
                },
            },
            {
                "id": "artifact:save",
                "kind": "artifact",
                "entity": {"kind": "process", "id": "rank0"},
                "event_time_ns": 1_300,
                "source_path": "indexes/artifact_rows/1",
                "payload": {
                    "artifact_id": "checkpoint-step-7",
                    "kind": "checkpoint",
                    "phase": "checkpoint/save",
                    "step": 7,
                    "global_rank": 0,
                    "start_time_ns": 1_300,
                    "end_time_ns": 1_900,
                },
            },
            {
                "id": "artifact:load",
                "kind": "artifact",
                "entity": {"kind": "process", "id": "rank0"},
                "event_time_ns": 2_000,
                "source_path": "indexes/artifact_rows/2",
                "payload": {
                    "artifact_id": "checkpoint-step-7",
                    "kind": "checkpoint",
                    "phase": "checkpoint/load",
                    "step": 7,
                    "global_rank": 0,
                    "duration_ns": 400,
                },
            },
        ],
    }

    summary = build_analytical_summary(graph)

    assert [row["checkpoint_operation"] for row in summary["checkpoint_durations"]] == [
        "load",
        "save",
        "stage",
    ]
    assert [row["duration_ns"] for row in summary["checkpoint_durations"]] == [
        400,
        600,
        200,
    ]
    assert all(
        row["calibration"] == "heuristic"
        for row in summary["checkpoint_durations"]
    )


def _event(
    observation_id: str,
    *,
    process_id: str,
    phase: str,
    step: int,
    global_rank: int,
    start_time_ns: int,
    end_time_ns: int,
    predecessor_phase: str | None = None,
    predecessor_step: int | None = None,
) -> dict[str, object]:
    payload = {
        "phase": phase,
        "step": step,
        "global_rank": global_rank,
        "start_time_ns": start_time_ns,
        "end_time_ns": end_time_ns,
    }
    if predecessor_phase is not None:
        payload["predecessor_phase"] = predecessor_phase
    if predecessor_step is not None:
        payload["predecessor_step"] = predecessor_step
    return {
        "id": observation_id,
        "kind": "structured_event",
        "entity": {"kind": "process", "id": process_id},
        "event_time_ns": start_time_ns,
        "source_path": "structured_logs",
        "payload": payload,
    }


def _collective(
    observation_id: str,
    *,
    process_id: str,
    step: int,
    global_rank: int,
    launch_time_ns: int,
    completion_time_ns: int,
    message_bytes: int | None = None,
    group_size: int | None = None,
    latency_ns: int | None = None,
    effective_bandwidth_bytes_per_ns: int | None = None,
) -> dict[str, object]:
    payload = {
        "phase": "collective/all_reduce",
        "step": step,
        "global_rank": global_rank,
        "launch_time_ns": launch_time_ns,
        "completion_time_ns": completion_time_ns,
    }
    for field, value in {
        "message_bytes": message_bytes,
        "group_size": group_size,
        "latency_ns": latency_ns,
        "effective_bandwidth_bytes_per_ns": effective_bandwidth_bytes_per_ns,
    }.items():
        if value is not None:
            payload[field] = value
    return {
        "id": observation_id,
        "kind": "structured_event",
        "entity": {"kind": "process", "id": process_id},
        "event_time_ns": launch_time_ns,
        "source_path": "structured_logs",
        "payload": payload,
    }
