# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

from torchtitan.observability.state_estimator.fixtures import (
    ProcessFixture,
    build_minimal_evidence_bundle,
    write_structured_events,
)
from torchtitan.observability.state_estimator.graph import (
    build_evidence_graph,
    build_multiplex_evidence_graph,
    write_evidence_graph,
)


def test_graph_contains_run_process_event_and_outcome_entities(tmp_path):
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
                phases=[("train/step", 1, 1_000, 2_000)],
            )
        ],
    )

    graph = build_evidence_graph(attempt_path)

    entity_ids = {entity["id"] for entity in graph["entities"]}
    assert "run:fixture-run" in entity_ids
    assert "attempt:fixture-run/attempt-001" in entity_ids
    assert "process:trainer.core.global_rank_000000" in entity_ids
    assert "outcome:trainer.core.global_rank_000000" in entity_ids
    assert graph["quality"] == []

    process = next(
        entity
        for entity in graph["entities"]
        if entity["id"] == "process:trainer.core.global_rank_000000"
    )
    assert process["attrs"] == {
        "actor_id": "core",
        "device_index": None,
        "global_rank": 0,
        "host_name": "fixture-host",
        "local_rank": 0,
        "role": "trainer",
        "world_size": 1,
    }


def test_graph_adds_identity_entities_from_v1_evidence_rows(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000003",
                global_rank=3,
                local_rank=1,
                world_size=8,
                role="trainer",
                actor_id="policy",
                host_name="host-a",
                phases=[("forward", 7, 1_000, 2_000)],
            )
        ],
    )

    graph = build_evidence_graph(attempt_path)

    entity_ids = {entity["id"] for entity in graph["entities"]}
    assert {
        "host:host-a",
        "rank:3",
        "role:trainer",
        "actor:policy",
        "phase:forward",
        "step:7",
    } <= entity_ids


def test_graph_preserves_v1_device_fields_on_process_and_device_entities(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000003",
                global_rank=3,
                local_rank=1,
                world_size=8,
                phases=[],
            )
        ],
    )
    write_structured_events(
        attempt_path,
        "trainer.core.global_rank_000003",
        [
            {
                "evidence_schema_version": 1,
                "event_seq": 0,
                "wall_time_ns": 1_000,
                "monotonic_ns": 1_000,
                "phase": "forward",
                "step": 7,
                "message": "phase_marker",
                "run_id": "fixture-run",
                "attempt_id": "attempt-001",
                "process_id": "trainer.core.global_rank_000003",
                "role": "trainer",
                "actor_id": "policy",
                "host_name": "host-a",
                "pid": 1000,
                "global_rank": 3,
                "local_rank": 1,
                "world_size": 8,
                "device_index": 1,
                "device_type": "cuda",
                "device_uuid": "GPU-fixture-uuid",
            }
        ],
    )

    graph = build_evidence_graph(attempt_path)

    process = next(
        entity
        for entity in graph["entities"]
        if entity["id"] == "process:trainer.core.global_rank_000003"
    )
    assert process["attrs"]["device_index"] == 1
    assert process["attrs"]["device_type"] == "cuda"
    assert process["attrs"]["device_uuid"] == "GPU-fixture-uuid"

    device = next(
        entity for entity in graph["entities"] if entity["id"] == "device:1"
    )
    assert device["attrs"] == {
        "device_index": 1,
        "device_type": "cuda",
        "device_uuid": "GPU-fixture-uuid",
    }


def test_write_graph_outputs_derived_artifacts(tmp_path):
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
            )
        ],
    )

    paths = write_evidence_graph(attempt_path)

    assert json.loads(paths.evidence_graph.read_text())["schema_version"] == 1
    assert json.loads(paths.entities.read_text())["schema_version"] == 1
    assert json.loads(paths.quality.read_text())["schema_version"] == 1
    assert paths.timeline.exists()


def test_multiplex_graph_preserves_v1_graph_keys_and_adds_layers(tmp_path):
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
                phases=[("train/step", 1, 1_000, 2_000)],
            )
        ],
    )

    legacy = build_evidence_graph(attempt_path)
    multiplex = build_multiplex_evidence_graph(attempt_path)

    assert multiplex["entities"] == legacy["entities"]
    assert multiplex["edges"] == legacy["edges"]
    assert multiplex["observations"] == legacy["observations"]
    assert multiplex["quality"][: len(legacy["quality"])] == legacy["quality"]
    assert [layer["name"] for layer in multiplex["layers"]] == [
        "physical",
        "logical",
        "control",
    ]
    assert multiplex["topology_epochs"][0]["id"] == "topology_epoch:0"


def test_multiplex_graph_deduplicates_loader_quality_findings(tmp_path):
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
                phases=[("train/step", 1, 1_000, 2_000)],
            )
        ],
    )
    malformed_path = attempt_path / "structured_logs" / "malformed.jsonl"
    malformed_path.write_text('{"malformed": true}\nnot-json\n')

    legacy = build_evidence_graph(attempt_path)
    multiplex = build_multiplex_evidence_graph(attempt_path)

    legacy_malformed_jsonl = [
        finding for finding in legacy["quality"] if finding["kind"] == "malformed_jsonl"
    ]
    multiplex_malformed_jsonl = [
        finding
        for finding in multiplex["quality"]
        if finding["kind"] == "malformed_jsonl"
    ]
    assert len(legacy_malformed_jsonl) == 1
    assert multiplex_malformed_jsonl == legacy_malformed_jsonl
