# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build deterministic derived evidence graphs from run-evidence bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torchtitan.observability.state_estimator.bundle import load_bundle
from torchtitan.observability.state_estimator.data_collection import (
    build_data_collection_plan,
)
from torchtitan.observability.state_estimator.schema import (
    DerivedPaths,
    SCHEMA_VERSION,
    write_json_atomic,
)
from torchtitan.observability.state_estimator.topology import build_topology_snapshot


def _entity(kind: str, id: str, **attrs: Any) -> dict[str, Any]:
    return {"kind": kind, "id": f"{kind}:{id}", "attrs": attrs}


def _edge(kind: str, source: str, target: str, **attrs: Any) -> dict[str, Any]:
    return {"kind": kind, "source": source, "target": target, "attrs": attrs}


def _process_attrs(row: dict[str, Any]) -> dict[str, Any]:
    attrs = {
        "actor_id": row.get("actor_id"),
        "device_index": row.get("device_index"),
        "global_rank": row.get("global_rank"),
        "host_name": row.get("host_name"),
        "local_rank": row.get("local_rank"),
        "role": row.get("role"),
        "world_size": row.get("world_size"),
    }
    for field in ("device_type", "device_uuid"):
        if row.get(field) is not None:
            attrs[field] = row[field]
    return attrs


def _device_attrs(row: dict[str, Any]) -> dict[str, Any]:
    attrs = {"device_index": row["device_index"]}
    for field in ("device_type", "device_uuid"):
        if row.get(field) is not None:
            attrs[field] = row[field]
    return attrs


def build_evidence_graph(attempt_path: Path) -> dict[str, Any]:
    bundle = load_bundle(attempt_path)
    run_id = bundle.manifest["run_id"]
    attempt_id = bundle.manifest["attempt_id"]
    entities: list[dict[str, Any]] = [
        _entity("run", run_id),
        _entity(
            "attempt",
            f"{run_id}/{attempt_id}",
            run_id=run_id,
            attempt_id=attempt_id,
        ),
    ]
    edges: list[dict[str, Any]] = [
        _edge("has_attempt", f"run:{run_id}", f"attempt:{run_id}/{attempt_id}")
    ]
    seen_entities: set[str] = {entity["id"] for entity in entities}
    observations: list[dict[str, Any]] = []

    def add_entity(entity: dict[str, Any]) -> None:
        if entity["id"] in seen_entities:
            for existing in entities:
                if existing["id"] == entity["id"]:
                    existing["attrs"].update(
                        {
                            key: value
                            for key, value in entity.get("attrs", {}).items()
                            if value is not None
                        }
                    )
                    return
            return
        seen_entities.add(entity["id"])
        entities.append(entity)

    def add_identity_entities(row: dict[str, Any]) -> None:
        host_name = row.get("host_name")
        if isinstance(host_name, str):
            add_entity(_entity("host", host_name))
        global_rank = row.get("global_rank")
        if isinstance(global_rank, int):
            add_entity(
                _entity(
                    "rank",
                    str(global_rank),
                    global_rank=global_rank,
                    local_rank=row.get("local_rank"),
                    world_size=row.get("world_size"),
                )
            )
        role = row.get("role")
        if isinstance(role, str):
            add_entity(_entity("role", role))
        actor_id = row.get("actor_id")
        if isinstance(actor_id, str):
            add_entity(_entity("actor", actor_id, role=role))
        device_index = row.get("device_index")
        if isinstance(device_index, int):
            add_entity(_entity("device", str(device_index), **_device_attrs(row)))
        phase = row.get("phase")
        if isinstance(phase, str):
            add_entity(_entity("phase", phase))
        step = row.get("step")
        if isinstance(step, int):
            add_entity(_entity("step", str(step), step=step))

    for row in bundle.artifact_rows:
        add_identity_entities(row)
        process_id = row.get("process_id")
        if isinstance(process_id, str):
            add_entity(_entity("process", process_id, **_process_attrs(row)))
            edges.append(
                _edge(
                    "attempt_process",
                    f"attempt:{run_id}/{attempt_id}",
                    f"process:{process_id}",
                )
            )
        artifact_id = str(row.get("artifact_id"))
        add_entity(
            _entity(
                "artifact",
                artifact_id,
                producer=row.get("producer"),
                artifact_kind=row.get("kind"),
                state=row.get("state"),
                path=row.get("path"),
            )
        )
        if isinstance(process_id, str):
            edges.append(
                _edge(
                    "process_artifact",
                    f"process:{process_id}",
                    f"artifact:{artifact_id}",
                )
            )
        artifact_seq = row.get("artifact_seq", len(observations))
        artifact_observation_id = (
            f"{process_id}:artifact:{artifact_seq}"
            if isinstance(process_id, str)
            else f"artifact:{artifact_seq}"
        )
        observation = {
            "schema_version": SCHEMA_VERSION,
            "id": artifact_observation_id,
            "kind": "artifact",
            "entity": (
                {"kind": "process", "id": process_id}
                if isinstance(process_id, str)
                else None
            ),
            "event_time_ns": row.get("wall_time_ns"),
            "ingestion_time_ns": None,
            "source_path": "indexes/artifact_rows",
            "quality": "present",
            "payload": row,
        }
        for field in ("monotonic_ns", "event_seq", "artifact_seq"):
            if field in row:
                observation[field] = row[field]
        observations.append(observation)

    for row in bundle.outcome_rows:
        add_identity_entities(row)
        process_id = row.get("process_id")
        if isinstance(process_id, str):
            add_entity(_entity("process", process_id, **_process_attrs(row)))
            add_entity(_entity("outcome", process_id, outcome=row.get("outcome")))
            edges.append(
                _edge(
                    "process_outcome",
                    f"process:{process_id}",
                    f"outcome:{process_id}",
                )
            )

    for row in bundle.event_rows:
        add_identity_entities(row)
        process_id = str(row.get("process_id", "unknown"))
        add_entity(_entity("process", process_id, **_process_attrs(row)))
        event_id = f"{process_id}:{row.get('event_seq', len(observations))}"
        observation = {
            "schema_version": SCHEMA_VERSION,
            "id": event_id,
            "kind": "structured_event",
            "entity": {"kind": "process", "id": process_id},
            "event_time_ns": row.get("wall_time_ns"),
            "ingestion_time_ns": None,
            "source_path": "structured_logs",
            "quality": "present",
            "payload": row,
        }
        for field in ("monotonic_ns", "event_seq", "artifact_seq"):
            if field in row:
                observation[field] = row[field]
        observations.append(observation)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "entities": entities,
        "edges": edges,
        "observations": sorted(
            observations,
            key=lambda item: (
                item["event_time_ns"] is None,
                item["event_time_ns"] or 0,
                item["id"],
            ),
        ),
        "quality": [finding.to_json() for finding in bundle.quality_findings],
    }


def build_multiplex_evidence_graph(
    attempt_path_or_bundle: Path | Any,
) -> dict[str, Any]:
    if isinstance(attempt_path_or_bundle, Path):
        attempt_path = attempt_path_or_bundle
        bundle = load_bundle(attempt_path)
        graph = build_evidence_graph(attempt_path)
    else:
        bundle = attempt_path_or_bundle
        graph = build_evidence_graph(bundle.attempt_path)

    topology = build_topology_snapshot(bundle).to_json()
    multiplex = dict(graph)
    multiplex["layers"] = topology["layers"]
    multiplex["topology_epochs"] = topology["topology_epochs"]
    multiplex["quality"] = _deduplicate_quality_findings(
        graph["quality"] + topology["quality"]
    )
    return multiplex


def _deduplicate_quality_findings(
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for finding in findings:
        key = json.dumps(finding, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(finding)
    return deduplicated


def write_evidence_graph(attempt_path: Path) -> DerivedPaths:
    paths = DerivedPaths.from_attempt_path(attempt_path)
    bundle = load_bundle(attempt_path)
    collection_plan = build_data_collection_plan(bundle).to_json()
    graph = build_evidence_graph(attempt_path)
    write_json_atomic(paths.evidence_graph, graph)
    write_json_atomic(
        paths.entities,
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": graph["run_id"],
            "attempt_id": graph["attempt_id"],
            "entities": graph["entities"],
        },
    )
    write_json_atomic(
        paths.quality,
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": graph["run_id"],
            "attempt_id": graph["attempt_id"],
            "findings": graph["quality"],
            "data_collection": collection_plan,
        },
    )
    paths.timeline.parent.mkdir(parents=True, exist_ok=True)
    paths.timeline.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in graph["observations"]
        )
    )
    return paths
