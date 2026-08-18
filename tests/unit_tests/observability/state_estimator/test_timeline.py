# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchtitan.observability.state_estimator.observation import (
    ClockQuality,
    NormalizedObservation,
    ObservationEnvelope,
    ObservationKind,
)
from torchtitan.observability.state_estimator.timeline import (
    TimelineWindow,
    build_incident_timeline,
    build_semantic_timeline,
    select_failure_center_time_ns,
    summarize_incident_timeline,
)


def test_timeline_orders_by_event_time_then_entity():
    graph = {
        "observations": [
            {
                "id": "b",
                "event_time_ns": 20,
                "entity": {"kind": "process", "id": "rank1"},
                "payload": {"phase": "backward"},
            },
            {
                "id": "a",
                "event_time_ns": 10,
                "entity": {"kind": "process", "id": "rank0"},
                "payload": {"phase": "forward"},
            },
        ],
        "entities": [],
    }

    timeline = build_incident_timeline(
        graph,
        center_time_ns=15,
        window_before_ns=10,
        window_after_ns=10,
    )

    assert [row["id"] for row in timeline] == ["a", "b"]
    assert timeline[0]["relative_time_ns"] == -5
    assert timeline[1]["relative_time_ns"] == 5


def test_timeline_uses_failed_attempt_center_when_center_not_explicit():
    graph = {
        "entities": [
            {"kind": "outcome", "id": "outcome:rank0", "attrs": {"outcome": "failed"}},
        ],
        "observations": [
            {
                "id": "early",
                "event_time_ns": 10,
                "entity": {"kind": "process", "id": "rank0"},
                "payload": {},
            },
            {
                "id": "failure",
                "event_time_ns": 30,
                "entity": {"kind": "process", "id": "rank0"},
                "payload": {},
            },
        ],
    }

    timeline = build_incident_timeline(
        graph,
        center_time_ns=None,
        window_before_ns=25,
        window_after_ns=0,
    )

    assert [row["id"] for row in timeline] == ["early", "failure"]
    assert timeline[-1]["relative_time_ns"] == 0


def test_timeline_preserves_process_local_ordering_fields():
    graph = {
        "entities": [],
        "observations": [
            {
                "id": "rank0:2",
                "event_time_ns": 10,
                "ingestion_time_ns": 12,
                "entity": {"kind": "process", "id": "rank0"},
                "payload": {
                    "monotonic_ns": 300,
                    "event_seq": 2,
                    "artifact_seq": 9,
                },
            },
        ],
    }

    timeline = build_incident_timeline(
        graph,
        center_time_ns=10,
        window_before_ns=0,
        window_after_ns=0,
    )

    assert timeline == [
        {
            "id": "rank0:2",
            "event_time_ns": 10,
            "ingestion_time_ns": 12,
            "entity": {"kind": "process", "id": "rank0"},
            "payload": {
                "monotonic_ns": 300,
                "event_seq": 2,
                "artifact_seq": 9,
            },
            "monotonic_ns": 300,
            "event_seq": 2,
            "artifact_seq": 9,
            "relative_time_ns": 0,
        }
    ]


def test_failure_center_uses_latest_observation_when_process_failed():
    graph = {
        "entities": [
            {"kind": "outcome", "id": "outcome:rank0", "attrs": {"outcome": "failed"}},
        ],
        "observations": [
            {
                "id": "a",
                "event_time_ns": 10,
                "entity": {"kind": "process", "id": "rank0"},
                "payload": {},
            },
            {
                "id": "b",
                "event_time_ns": 30,
                "entity": {"kind": "process", "id": "rank0"},
                "payload": {},
            },
        ],
    }

    assert select_failure_center_time_ns(graph) == 30


def _observation(
    observation_id: str,
    *,
    process_id: str = "rank0",
    source_path: str | None = None,
    raw_record_id: str | None = None,
    event_time_ns: int | None = None,
    ingestion_time_ns: int | None = None,
    monotonic_ns: int | None = None,
    event_seq: int | None = None,
    artifact_seq: int | None = None,
    step: int | None = None,
    phase: str | None = None,
    topology_epoch: int | None = None,
) -> NormalizedObservation:
    raw_record = {
        key: value
        for key, value in {
            "topology_epoch": topology_epoch,
            "event_time_ns": event_time_ns,
            "ingestion_time_ns": ingestion_time_ns,
            "monotonic_ns": monotonic_ns,
            "event_seq": event_seq,
            "artifact_seq": artifact_seq,
            "step": step,
            "phase": phase,
        }.items()
        if value is not None
    }
    return NormalizedObservation(
        id=observation_id,
        kind=ObservationKind.STRUCTURED_EVENT,
        envelope=ObservationEnvelope(
            run_id="run0",
            attempt_id="attempt0",
            schema_version=1,
            record_type="structured_event",
            source_path=source_path or f"structured/{observation_id}.jsonl",
            raw_record_id=raw_record_id or observation_id,
            process_id=process_id,
            event_time_ns=event_time_ns,
            ingestion_time_ns=ingestion_time_ns,
            monotonic_ns=monotonic_ns,
            event_seq=event_seq,
            artifact_seq=artifact_seq,
            step=step,
            phase=phase,
            clock_quality=(
                ClockQuality.EVENT_AND_MONOTONIC
                if event_time_ns is not None and monotonic_ns is not None
                else ClockQuality.MONOTONIC_ONLY
                if monotonic_ns is not None
                else ClockQuality.EVENT_ONLY
                if event_time_ns is not None
                else ClockQuality.INGESTION_ONLY
                if ingestion_time_ns is not None
                else ClockQuality.MISSING
            ),
        ),
        payload=raw_record,
        raw_record=raw_record,
    )


def test_semantic_timeline_inserts_delayed_observations_by_event_time():
    result = build_semantic_timeline(
        [
            _observation(
                "ingested-first",
                event_time_ns=200,
                ingestion_time_ns=210,
                monotonic_ns=20,
                event_seq=2,
                step=2,
                phase="backward",
            ),
            _observation(
                "delayed",
                event_time_ns=100,
                ingestion_time_ns=500,
                monotonic_ns=10,
                event_seq=1,
                step=1,
                phase="forward",
            ),
        ],
        window=TimelineWindow.explicit_time(
            start_time_ns=0,
            end_time_ns=300,
            center_time_ns=150,
        ),
    )

    assert [entry.observation_id for entry in result.entries] == [
        "delayed",
        "ingested-first",
    ]
    assert [entry.relative_time_ns for entry in result.entries] == [-50, 50]


def test_semantic_timeline_warns_on_missing_clocks_and_uses_local_fallback_order():
    result = build_semantic_timeline(
        [
            _observation("missing", event_seq=3, step=7, phase="loss"),
            _observation(
                "second",
                monotonic_ns=20,
                event_seq=2,
                step=7,
                phase="backward",
            ),
            _observation(
                "first",
                monotonic_ns=10,
                event_seq=1,
                step=7,
                phase="forward",
            ),
        ],
        window=TimelineWindow.explicit_step(start_step=7, end_step=7),
    )

    assert [entry.observation_id for entry in result.entries] == [
        "first",
        "second",
        "missing",
    ]
    assert result.entries[0].relative_time_ns == 0
    assert result.entries[1].relative_time_ns == 10
    assert {finding.kind for finding in result.clock_quality_findings} >= {
        "missing_clocks",
        "weak_cross_process_ordering",
    }


def test_semantic_timeline_warns_when_local_sequence_conflicts_with_event_time():
    result = build_semantic_timeline(
        [
            _observation(
                "seq2-before-seq1",
                event_time_ns=100,
                monotonic_ns=20,
                event_seq=2,
            ),
            _observation(
                "seq1-after-seq2",
                event_time_ns=200,
                monotonic_ns=10,
                event_seq=1,
            ),
        ],
        window=TimelineWindow.explicit_time(start_time_ns=0, end_time_ns=300),
    )

    assert [entry.observation_id for entry in result.entries] == [
        "seq2-before-seq1",
        "seq1-after-seq2",
    ]
    assert any(
        finding.kind == "conflicting_sequence_time_order"
        for finding in result.clock_quality_findings
    )


def test_semantic_timeline_supports_incident_step_and_last_n_step_windows():
    observations = [
        _observation("step1", event_time_ns=10, step=1, phase="forward"),
        _observation("step2", event_time_ns=20, step=2, phase="backward"),
        _observation("step3", event_time_ns=30, step=3, phase="optimizer"),
        _observation("step4", event_time_ns=40, step=4, phase="checkpoint"),
    ]

    incident = build_semantic_timeline(
        observations,
        window=TimelineWindow.incident_centered(
            center_time_ns=25,
            window_before_ns=10,
            window_after_ns=10,
        ),
    )
    explicit_step = build_semantic_timeline(
        observations,
        window=TimelineWindow.explicit_step(start_step=2, end_step=3),
    )
    last_steps = build_semantic_timeline(
        observations,
        window=TimelineWindow.last_n_steps(num_steps=2),
    )

    assert [entry.observation_id for entry in incident.entries] == ["step2", "step3"]
    assert [entry.relative_time_ns for entry in incident.entries] == [-5, 5]
    assert [entry.observation_id for entry in explicit_step.entries] == [
        "step2",
        "step3",
    ]
    assert [entry.observation_id for entry in last_steps.entries] == [
        "step3",
        "step4",
    ]


def test_semantic_entries_include_identity_source_topology_and_summary_fields():
    result = build_semantic_timeline(
        [
            _observation(
                "rank0-step5",
                process_id="rank0",
                event_time_ns=1000,
                monotonic_ns=900,
                event_seq=5,
                artifact_seq=8,
                step=5,
                phase="checkpoint",
                topology_epoch=2,
                source_path="structured/events.jsonl",
                raw_record_id="events:5",
            ),
        ],
        window=TimelineWindow.explicit_time(
            start_time_ns=900,
            end_time_ns=1100,
            center_time_ns=1000,
        ),
    )

    assert result.entries[0].to_json() == {
        "observation_id": "rank0-step5",
        "record_type": "structured_event",
        "phase": "checkpoint",
        "step": 5,
        "entity": {"kind": "process", "id": "rank0"},
        "event_time_ns": 1000,
        "ingestion_time_ns": None,
        "monotonic_ns": 900,
        "event_seq": 5,
        "artifact_seq": 8,
        "source_path": "structured/events.jsonl",
        "source_record": "events:5",
        "topology_epoch": 2,
        "relative_time_ns": 0,
        "ordering_basis": "event_time",
    }
    assert summarize_incident_timeline(result, max_entries=3) == "\n".join(
        [
            "- +0ns step=5 phase=checkpoint entity=process:rank0 "
            "source=structured/events.jsonl#events:5",
        ]
    )
