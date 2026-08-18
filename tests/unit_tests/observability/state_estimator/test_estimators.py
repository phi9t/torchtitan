# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

from torchtitan.observability.state_estimator.estimators import (
    estimate_belief,
    write_belief_summary,
)
from torchtitan.observability.state_estimator.fixtures import (
    ProcessFixture,
    build_minimal_evidence_bundle,
    write_artifact_index,
)


def test_estimator_marks_failed_process_and_heuristic_calibration():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [
            {"kind": "process", "id": "process:rank0", "attrs": {"global_rank": 0}},
            {"kind": "outcome", "id": "outcome:rank0", "attrs": {"outcome": "failed"}},
        ],
        "edges": [
            {
                "kind": "process_outcome",
                "source": "process:rank0",
                "target": "outcome:rank0",
                "attrs": {},
            }
        ],
        "observations": [],
        "quality": [],
    }

    summary = estimate_belief(graph)

    assert summary["claim_calibration"] == "heuristic"
    assert summary["process_liveness"]["process:rank0"]["outcome"] == "failed"
    assert summary["fault_hypotheses"][0]["mode"] == "process_failure"
    assert summary["fault_hypotheses"][0]["score"] == 1.0


def test_failed_process_outcome_feeds_advisory_failed_mode_score():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [
            {"kind": "process", "id": "process:rank0", "attrs": {"global_rank": 0}},
            {"kind": "outcome", "id": "outcome:rank0", "attrs": {"outcome": "failed"}},
        ],
        "edges": [
            {
                "kind": "process_outcome",
                "source": "process:rank0",
                "target": "outcome:rank0",
                "attrs": {},
            }
        ],
        "observations": [],
        "quality": [],
    }

    summary = estimate_belief(graph)

    failed_score = next(
        score
        for score in summary["inference"]["mode_scores"]
        if score["mode"] == "failed"
    )
    assert failed_score["score"] > 0.0
    assert failed_score["calibration"] == "heuristic_uncalibrated"
    assert failed_score["advisory"]


def test_estimator_reports_peer_skew_without_root_cause_overclaim():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [
            {
                "id": "rank0:0",
                "kind": "structured_event",
                "entity": {"kind": "process", "id": "rank0"},
                "event_time_ns": 1_000,
                "payload": {"phase": "train/step", "step": 1, "global_rank": 0},
            },
            {
                "id": "rank1:0",
                "kind": "structured_event",
                "entity": {"kind": "process", "id": "rank1"},
                "event_time_ns": 900_001_000,
                "payload": {"phase": "train/step", "step": 1, "global_rank": 1},
            },
        ],
    }

    summary = estimate_belief(graph, skew_threshold_ns=500_000_000)

    assert summary["peer_skew_findings"][0]["phase"] == "train/step"
    assert summary["peer_skew_findings"][0]["step"] == 1
    assert summary["peer_skew_findings"][0]["skew_ns"] == 900_000_000
    assert summary["peer_skew_findings"][0]["root_cause"] == "ambiguous"


def test_estimator_exposes_advisory_inference_summary():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [
            {
                "id": "rank0:collective",
                "kind": "structured_event",
                "entity": {"kind": "process", "id": "rank0"},
                "event_time_ns": 1_000,
                "payload": {
                    "phase": "collective/all_reduce",
                    "step": 1,
                    "global_rank": 0,
                    "collective": "all_reduce",
                    "message_bytes": 1024,
                    "group_size": 2,
                    "latency_ns": 10,
                    "effective_bandwidth_bytes_per_ns": 1024,
                    "duration_ns": 100,
                    "expected_sigma_ns": 10,
                },
            },
        ],
    }

    summary = estimate_belief(graph)

    assert summary["inference"]["claim_calibration"] == "heuristic_uncalibrated"
    assert summary["inference"]["mode_scores"][0]["mode"] == "nominal"
    assert {
        score["calibration"] for score in summary["inference"]["mode_scores"]
    } == {"heuristic_uncalibrated"}
    assert summary["analytical"]["residuals"][0]["normalized_residual"] == 7.9
    assert summary["process_liveness"] == {}


def test_estimator_summarizes_phase_duration_from_start_end_events():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [
            {
                "id": "rank0:0",
                "kind": "structured_event",
                "entity": {"kind": "process", "id": "rank0"},
                "event_time_ns": 1_000,
                "payload": {
                    "phase": "forward",
                    "step": 3,
                    "global_rank": 0,
                    "event": "start",
                    "wall_time_ns": 1_000,
                    "monotonic_ns": 10_000,
                },
            },
            {
                "id": "rank0:1",
                "kind": "structured_event",
                "entity": {"kind": "process", "id": "rank0"},
                "event_time_ns": 1_700,
                "payload": {
                    "phase": "forward",
                    "step": 3,
                    "global_rank": 0,
                    "event": "end",
                    "wall_time_ns": 1_700,
                    "monotonic_ns": 10_700,
                },
            },
        ],
    }

    summary = estimate_belief(graph)

    assert summary["phase_durations"] == [
        {
            "phase": "forward",
            "step": 3,
            "process": {"kind": "process", "id": "rank0"},
            "global_rank": 0,
            "duration_ns": 700,
            "clock": "monotonic_ns",
            "start_time_ns": 10_000,
            "end_time_ns": 10_700,
            "calibration": "heuristic",
        }
    ]
    assert not [
        warning
        for warning in summary["observability_warnings"]
        if warning["kind"] == "missing_phase_duration_fields"
    ]


def test_estimator_summarizes_phase_duration_from_paired_time_fields():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [
            {
                "id": "rank0:0",
                "kind": "structured_event",
                "entity": {"kind": "process", "id": "rank0"},
                "event_time_ns": 1_000,
                "payload": {
                    "phase": "backward",
                    "step": 4,
                    "global_rank": 0,
                    "start_time_ns": 2_000,
                    "end_time_ns": 2_900,
                    "clock": "monotonic_ns",
                },
            },
        ],
    }

    summary = estimate_belief(graph)

    assert summary["phase_durations"] == [
        {
            "phase": "backward",
            "step": 4,
            "process": {"kind": "process", "id": "rank0"},
            "global_rank": 0,
            "duration_ns": 900,
            "clock": "monotonic_ns",
            "start_time_ns": 2_000,
            "end_time_ns": 2_900,
            "calibration": "heuristic",
        }
    ]
    assert summary["analytical"]["phase_durations"] == [
        {
            "process": {"kind": "process", "id": "rank0"},
            "global_rank": 0,
            "phase": "backward",
            "step": 4,
            "duration_ns": 900,
            "clock": "monotonic_ns",
            "start_time_ns": 2_000,
            "end_time_ns": 2_900,
            "source_evidence": ["rank0:0"],
            "calibration": "heuristic",
            "quality": "present",
        }
    ]


def test_estimator_reports_missing_phase_duration_fields_when_unpaired():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [
            {
                "id": "rank0:0",
                "kind": "structured_event",
                "entity": {"kind": "process", "id": "rank0"},
                "event_time_ns": 1_000,
                "payload": {"phase": "forward", "step": 3, "global_rank": 0},
            },
        ],
    }

    summary = estimate_belief(graph)

    assert summary["phase_durations"] == []
    missing = [
        warning
        for warning in summary["observability_warnings"]
        if warning["kind"] == "missing_phase_duration_fields"
    ]
    assert missing == [
        {
            "schema_version": 1,
            "kind": "missing_phase_duration_fields",
            "severity": "info",
            "evidence_state": "missing",
            "message": (
                "phase duration for phase=forward step=3 process=rank0 requires "
                "paired start/end events or explicit duration fields"
            ),
            "source_path": "structured_logs",
            "entity": {"kind": "process", "id": "rank0"},
            "metadata": {
                "phase": "forward",
                "step": 3,
                "missing_fields": ["duration_ns", "phase_state", "monotonic_ns_pair"],
            },
        }
    ]


def test_estimator_reports_missing_phase_duration_identity_fields():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [
            {
                "id": "missing-phase",
                "kind": "structured_event",
                "entity": {"kind": "process", "id": "rank0"},
                "event_time_ns": 1_000,
                "payload": {"step": 3, "global_rank": 0},
            },
            {
                "id": "missing-step",
                "kind": "structured_event",
                "entity": {"kind": "process", "id": "rank0"},
                "event_time_ns": 1_100,
                "payload": {"phase": "forward", "global_rank": 0},
            },
            {
                "id": "missing-process",
                "kind": "structured_event",
                "entity": None,
                "event_time_ns": 1_200,
                "payload": {"phase": "forward", "step": 3, "global_rank": 0},
            },
        ],
    }

    summary = estimate_belief(graph)

    assert summary["phase_durations"] == []
    missing = [
        warning
        for warning in summary["observability_warnings"]
        if warning["kind"] == "missing_phase_duration_fields"
    ]
    assert [warning["metadata"]["observation_id"] for warning in missing] == [
        "missing-phase",
        "missing-step",
        "missing-process",
    ]
    assert [warning["metadata"]["missing_fields"] for warning in missing] == [
        ["phase", "duration_fields"],
        ["step", "duration_fields"],
        ["process", "duration_fields"],
    ]
    assert missing[0]["entity"] == {"kind": "process", "id": "rank0"}
    assert missing[2]["entity"] is None


def test_write_belief_summary_uses_failed_attempt_timeline_center(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
                outcome="failed",
                phases=[("train/step", 1, 1_000, 2_000)],
            )
        ],
    )

    paths = write_belief_summary(attempt_path)

    summary = json.loads(paths.belief_summary.read_text())
    assert [row["id"] for row in summary["incident_timeline"]] == [
        "trainer.core.global_rank_000000:0"
    ]
    assert summary["incident_timeline"][0]["relative_time_ns"] == 0


def test_write_belief_summary_reaches_checkpoint_artifact_durations(tmp_path):
    process_id = "trainer.core.global_rank_000000"
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id=process_id,
                global_rank=0,
                local_rank=0,
                world_size=1,
                outcome="succeeded",
                phases=[("train/step", 1, 1_000, 2_000)],
            )
        ],
    )
    write_artifact_index(
        attempt_path,
        process_id,
        [
            {
                "schema_version": 1,
                "evidence_schema_version": 1,
                "record_type": "artifact",
                "artifact_id": "checkpoint-step-7",
                "producer": "checkpoint_manager",
                "kind": "checkpoint",
                "relation": "output",
                "state": "complete",
                "phase": "checkpoint/save",
                "step": 7,
                "global_rank": 0,
                "start_time_ns": 3_000,
                "end_time_ns": 3_600,
                "wall_time_ns": 3_600,
                "monotonic_ns": 3_600,
                "artifact_seq": 1,
                "metadata": {},
                "run_id": "fixture-run",
                "attempt_id": "attempt-001",
                "process_id": process_id,
                "local_rank": 0,
                "world_size": 1,
                "role": "trainer",
            }
        ],
    )

    paths = write_belief_summary(attempt_path)

    summary = json.loads(paths.belief_summary.read_text())
    assert summary["analytical"]["checkpoint_durations"] == [
        {
            "process": {"kind": "process", "id": process_id},
            "global_rank": 0,
            "phase": "checkpoint/save",
            "step": 7,
            "duration_ns": 600,
            "clock": "time_ns",
            "start_time_ns": 3_000,
            "end_time_ns": 3_600,
            "source_evidence": [f"{process_id}:artifact:1"],
            "calibration": "heuristic",
            "quality": "present",
            "checkpoint_operation": "save",
            "checkpoint_id": "checkpoint-step-7",
        }
    ]
