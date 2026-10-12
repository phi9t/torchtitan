# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import replace

from torchtitan.observability.state_estimator.bundle import load_bundle
from torchtitan.observability.state_estimator.fixtures import (
    build_minimal_evidence_bundle,
    ProcessFixture,
    write_artifact_index,
    write_structured_events,
)
from torchtitan.observability.state_estimator.observation import (
    normalize_bundle_observations,
)
from torchtitan.observability.state_estimator.topology import (
    build_topology_snapshot,
    GraphLayer,
)


def _topology_bundle(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="run-topology",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=2,
                host_name="host-a",
                phases=[],
            )
        ],
    )
    write_structured_events(
        attempt_path,
        "trainer.core.global_rank_000000",
        [
            {
                "evidence_schema_version": 1,
                "event_seq": 1,
                "wall_time_ns": 20,
                "monotonic_ns": 20,
                "message": "topology",
                "run_id": "run-topology",
                "attempt_id": "attempt-001",
                "process_id": "trainer.core.global_rank_000000",
                "role": "trainer",
                "actor_id": "core",
                "host_name": "host-a",
                "pid": 1000,
                "global_rank": 0,
                "local_rank": 0,
                "world_size": 2,
                "device_index": 0,
                "device_uuid": "GPU-0",
                "sensor_id": "structured-events/rank-0",
                "topology_epoch": 3,
                "mesh_axis": "dp",
                "process_group_id": "pg-dp",
                "process_group_backend": "nccl",
                "incident_id": "incident-timeout",
            }
        ],
    )
    write_artifact_index(
        attempt_path,
        "trainer.core.global_rank_000000",
        [
            {
                "schema_version": 1,
                "evidence_schema_version": 1,
                "record_type": "artifact",
                "artifact_id": "artifact-events-rank-0",
                "producer": "structured_logger",
                "kind": "torchtitan.structured_events",
                "relation": "output",
                "state": "complete",
                "path": "structured_logs/trainer.core.global_rank_000000.jsonl",
                "wall_time_ns": 1,
                "monotonic_ns": 1,
                "artifact_seq": 0,
                "sensor_id": "artifact-index/rank-0",
                "run_id": "run-topology",
                "attempt_id": "attempt-001",
                "process_id": "trainer.core.global_rank_000000",
                "role": "trainer",
                "actor_id": "core",
                "host_name": "host-a",
                "pid": 1000,
                "global_rank": 0,
                "local_rank": 0,
                "world_size": 2,
                "device_index": 0,
                "device_uuid": "GPU-0",
            },
            {
                "schema_version": 1,
                "evidence_schema_version": 1,
                "record_type": "artifact",
                "artifact_id": "checkpoint-step-7",
                "checkpoint_id": "checkpoint-step-7",
                "producer": "checkpoint_manager",
                "kind": "checkpoint",
                "relation": "output",
                "state": "complete",
                "path": "checkpoints/step-7",
                "wall_time_ns": 2,
                "monotonic_ns": 2,
                "artifact_seq": 1,
                "sensor_id": "checkpoint-writer/rank-0",
                "run_id": "run-topology",
                "attempt_id": "attempt-001",
                "process_id": "trainer.core.global_rank_000000",
                "role": "trainer",
                "actor_id": "core",
                "host_name": "host-a",
                "pid": 1000,
                "global_rank": 0,
                "local_rank": 0,
                "world_size": 2,
                "device_index": 0,
                "device_uuid": "GPU-0",
            },
        ],
    )
    return load_bundle(attempt_path)


def test_topology_snapshot_separates_physical_logical_and_control_layers(tmp_path):
    snapshot = build_topology_snapshot(_topology_bundle(tmp_path)).to_json()

    assert [layer["name"] for layer in snapshot["layers"]] == [
        GraphLayer.PHYSICAL.value,
        GraphLayer.LOGICAL.value,
        GraphLayer.CONTROL.value,
    ]
    layer_entities = {
        layer["name"]: {entity["id"] for entity in layer["entities"]}
        for layer in snapshot["layers"]
    }
    assert {
        "host:host-a",
        "process:trainer.core.global_rank_000000",
        "device:GPU-0",
    } <= layer_entities[GraphLayer.PHYSICAL.value]
    assert {
        "rank:0",
        "mesh_axis:dp",
        "process_group:pg-dp",
    } <= layer_entities[GraphLayer.LOGICAL.value]
    assert {
        "artifact:artifact-events-rank-0",
        "checkpoint:checkpoint-step-7",
        "incident:incident-timeout",
        "sensor:structured-events/rank-0",
    } <= layer_entities[GraphLayer.CONTROL.value]


def test_topology_snapshot_extracts_known_relationships_without_guessing(tmp_path):
    snapshot = build_topology_snapshot(_topology_bundle(tmp_path)).to_json()
    layer_edges = {
        layer["name"]: {
            (edge["kind"], edge["source"], edge["target"]) for edge in layer["edges"]
        }
        for layer in snapshot["layers"]
    }

    assert (
        "process_on_host",
        "process:trainer.core.global_rank_000000",
        "host:host-a",
    ) in layer_edges[GraphLayer.PHYSICAL.value]
    assert (
        "process_uses_device",
        "process:trainer.core.global_rank_000000",
        "device:GPU-0",
    ) in layer_edges[GraphLayer.PHYSICAL.value]
    assert (
        "rank_in_process_group",
        "rank:0",
        "process_group:pg-dp",
    ) in layer_edges[GraphLayer.LOGICAL.value]
    assert (
        "process_observed_by_sensor",
        "sensor:structured-events/rank-0",
        "process:trainer.core.global_rank_000000",
    ) in layer_edges[GraphLayer.CONTROL.value]

    edge_targets = {
        (edge["source"], edge["target"])
        for layer in snapshot["layers"]
        for edge in layer["edges"]
    }
    assert ("rank:0", "rank:1") not in edge_targets


def test_topology_snapshot_reports_missing_topology_quality(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="run-missing-topology",
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

    snapshot = build_topology_snapshot(load_bundle(attempt_path)).to_json()

    findings = snapshot["quality"]
    assert any(
        finding["kind"] == "missing_device_topology"
        and finding["evidence_state"] == "unknown"
        and "device-localization" in finding["metadata"]["blocked_diagnoses"]
        for finding in findings
    )
    assert any(
        finding["kind"] == "missing_process_group_topology"
        and finding["evidence_state"] == "uncollected"
        and "collective-hang-localization" in finding["metadata"]["blocked_diagnoses"]
        for finding in findings
    )


def test_topology_snapshot_orders_entities_edges_and_epochs_deterministically(tmp_path):
    bundle = _topology_bundle(tmp_path)

    first = build_topology_snapshot(bundle).to_json()
    second = build_topology_snapshot(bundle).to_json()

    assert first == second
    for layer in first["layers"]:
        assert [entity["id"] for entity in layer["entities"]] == sorted(
            entity["id"] for entity in layer["entities"]
        )
        assert [
            (edge["kind"], edge["source"], edge["target"]) for edge in layer["edges"]
        ] == sorted(
            (edge["kind"], edge["source"], edge["target"]) for edge in layer["edges"]
        )
    assert first["topology_epochs"] == [
        {
            "schema_version": 1,
            "id": "topology_epoch:0",
            "epoch": 0,
            "start_time_ns": 1,
            "end_time_ns": 2,
            "source_record_ids": [
                bundle.row_record_ids[id(bundle.artifact_rows[0])],
                bundle.row_record_ids[id(bundle.artifact_rows[1])],
                bundle.row_record_ids[id(bundle.outcome_rows[0])],
            ],
        },
        {
            "schema_version": 1,
            "id": "topology_epoch:3",
            "epoch": 3,
            "start_time_ns": 20,
            "end_time_ns": 20,
            "source_record_ids": [bundle.row_record_ids[id(bundle.event_rows[0])]],
        },
    ]


def test_topology_snapshot_keeps_distinct_observations_from_same_jsonl_file(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="run-jsonl-observations",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
                host_name="host-a",
                phases=[],
            )
        ],
    )
    write_structured_events(
        attempt_path,
        "trainer.core.global_rank_000000",
        [
            {
                "evidence_schema_version": 1,
                "event_seq": 7,
                "wall_time_ns": 10,
                "monotonic_ns": 10,
                "message": "first",
                "run_id": "run-jsonl-observations",
                "attempt_id": "attempt-001",
                "process_id": "trainer.core.global_rank_000000",
                "host_name": "host-a",
                "global_rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "sensor_id": "structured-events/rank-0",
            },
            {
                "evidence_schema_version": 1,
                "event_seq": 7,
                "wall_time_ns": 11,
                "monotonic_ns": 11,
                "message": "second",
                "run_id": "run-jsonl-observations",
                "attempt_id": "attempt-001",
                "process_id": "trainer.core.global_rank_000000",
                "host_name": "host-a",
                "global_rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "sensor_id": "structured-events/rank-0",
            },
        ],
    )

    bundle = load_bundle(attempt_path)
    normalized = normalize_bundle_observations(bundle)
    snapshot = build_topology_snapshot(bundle).to_json()

    assert normalized.observations[0].id == normalized.observations[1].id
    control_layer = next(
        layer for layer in snapshot["layers"] if layer["name"] == "control"
    )
    event_record_ids = {
        bundle.row_record_ids[id(bundle.event_rows[0])],
        bundle.row_record_ids[id(bundle.event_rows[1])],
    }
    observation_entities = [
        entity
        for entity in control_layer["entities"]
        if entity["kind"] == "observation"
        and entity["attrs"]["raw_record_id"] in event_record_ids
    ]
    assert len(observation_entities) == 2
    sensor_edges = [
        edge
        for edge in control_layer["edges"]
        if edge["kind"] == "sensor_observed_record"
        and edge["source"] == "sensor:structured-events/rank-0"
    ]
    assert len(sensor_edges) == 2
    assert len({edge["target"] for edge in sensor_edges}) == 2


def test_topology_snapshot_emits_all_distinct_explicit_epochs(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="run-multi-epoch",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
                host_name="host-a",
                phases=[],
            )
        ],
    )
    write_structured_events(
        attempt_path,
        "trainer.core.global_rank_000000",
        [
            {
                "evidence_schema_version": 1,
                "event_seq": 1,
                "wall_time_ns": 30,
                "monotonic_ns": 30,
                "message": "epoch-3",
                "run_id": "run-multi-epoch",
                "attempt_id": "attempt-001",
                "process_id": "trainer.core.global_rank_000000",
                "host_name": "host-a",
                "global_rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "topology_epoch": 3,
            },
            {
                "evidence_schema_version": 1,
                "event_seq": 2,
                "wall_time_ns": 40,
                "monotonic_ns": 40,
                "message": "epoch-4",
                "run_id": "run-multi-epoch",
                "attempt_id": "attempt-001",
                "process_id": "trainer.core.global_rank_000000",
                "host_name": "host-a",
                "global_rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "topology_epoch": 4,
            },
        ],
    )

    bundle = load_bundle(attempt_path)
    snapshot = build_topology_snapshot(bundle).to_json()

    assert snapshot["topology_epochs"] == [
        {
            "schema_version": 1,
            "id": "topology_epoch:0",
            "epoch": 0,
            "start_time_ns": 1,
            "end_time_ns": 1,
            "source_record_ids": [
                bundle.row_record_ids[id(bundle.artifact_rows[0])],
                bundle.row_record_ids[id(bundle.outcome_rows[0])],
            ],
        },
        {
            "schema_version": 1,
            "id": "topology_epoch:3",
            "epoch": 3,
            "start_time_ns": 30,
            "end_time_ns": 30,
            "source_record_ids": [bundle.row_record_ids[id(bundle.event_rows[0])]],
        },
        {
            "schema_version": 1,
            "id": "topology_epoch:4",
            "epoch": 4,
            "start_time_ns": 40,
            "end_time_ns": 40,
            "source_record_ids": [bundle.row_record_ids[id(bundle.event_rows[1])]],
        },
    ]


def test_topology_snapshot_uses_raw_record_keys_when_observation_ids_collide(tmp_path):
    bundle = _topology_bundle(tmp_path)
    first = dict(bundle.event_rows[0])
    second = dict(bundle.event_rows[0])
    first.update({"process_id": 0, "event_seq": 7, "message": "first"})
    second.update({"process_id": 0, "event_seq": 7, "message": "second"})
    bundle = replace(bundle, event_rows=[first, second])
    normalized = normalize_bundle_observations(bundle)

    snapshot = build_topology_snapshot(normalized.observations).to_json()

    observation_ids = [observation.id for observation in normalized.observations]
    assert observation_ids[0] == observation_ids[1]
    sensor_edges = [
        edge
        for layer in snapshot["layers"]
        for edge in layer["edges"]
        if edge["kind"] == "sensor_observed_record"
        and edge["source"] == "sensor:structured-events/rank-0"
    ]
    assert len(sensor_edges) == 2
    assert {edge["target"] for edge in sensor_edges} == {
        "observation:event_rows:0",
        "observation:event_rows:1",
    }
