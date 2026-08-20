# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Topology snapshots and multiplex graph layers for state-estimator evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from torchtitan.observability.state_estimator.bundle import RunEvidenceBundle
from torchtitan.observability.state_estimator.observation import (
    normalize_bundle_observations,
    NormalizedObservation,
    ObservationKind,
)
from torchtitan.observability.state_estimator.schema import (
    QualityFinding,
    SCHEMA_VERSION,
)


class GraphLayer(StrEnum):
    PHYSICAL = "physical"
    LOGICAL = "logical"
    CONTROL = "control"


@dataclass(frozen=True, slots=True)
class TopologyEpoch:
    epoch: int
    start_time_ns: int | None = None
    end_time_ns: int | None = None
    source_record_ids: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": f"topology_epoch:{self.epoch}",
            "epoch": self.epoch,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "source_record_ids": list(self.source_record_ids),
        }


@dataclass(frozen=True, slots=True)
class LayerGraph:
    name: GraphLayer
    entities: tuple[dict[str, Any], ...] = ()
    edges: tuple[dict[str, Any], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name.value,
            "entities": [dict(entity) for entity in self.entities],
            "edges": [dict(edge) for edge in self.edges],
        }


@dataclass(frozen=True, slots=True)
class TopologySnapshot:
    run_id: str | None
    attempt_id: str | None
    layers: tuple[LayerGraph, ...]
    topology_epochs: tuple[TopologyEpoch, ...]
    quality_findings: tuple[QualityFinding, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "layers": [layer.to_json() for layer in self.layers],
            "topology_epochs": [
                topology_epoch.to_json() for topology_epoch in self.topology_epochs
            ],
            "quality": [finding.to_json() for finding in self.quality_findings],
        }


@dataclass(slots=True)
class _LayerBuilder:
    entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)

    def add_entity(self, kind: str, entity_id: str, **attrs: Any) -> str:
        full_id = f"{kind}:{entity_id}"
        entity = self.entities.setdefault(
            full_id, {"kind": kind, "id": full_id, "attrs": {}}
        )
        entity["attrs"].update(
            {key: value for key, value in attrs.items() if value is not None}
        )
        return full_id

    def add_edge(self, kind: str, source: str, target: str, **attrs: Any) -> None:
        edge_key = (kind, source, target)
        edge = self.edges.setdefault(
            edge_key,
            {"kind": kind, "source": source, "target": target, "attrs": {}},
        )
        edge["attrs"].update(
            {key: value for key, value in attrs.items() if value is not None}
        )

    def build(self, name: GraphLayer) -> LayerGraph:
        return LayerGraph(
            name=name,
            entities=tuple(
                self.entities[entity_id] for entity_id in sorted(self.entities)
            ),
            edges=tuple(
                self.edges[key]
                for key in sorted(
                    self.edges,
                    key=lambda item: (item[0], item[1], item[2]),
                )
            ),
        )


def build_topology_snapshot(
    bundle_or_observations: RunEvidenceBundle | Sequence[NormalizedObservation],
) -> TopologySnapshot:
    if isinstance(bundle_or_observations, RunEvidenceBundle):
        normalization = normalize_bundle_observations(bundle_or_observations)
        observations = normalization.observations
        quality_findings = list(normalization.quality_findings)
    else:
        observations = list(bundle_or_observations)
        quality_findings = [
            finding
            for observation in observations
            for finding in observation.quality_findings
        ]

    run_id = _first_present(observation.envelope.run_id for observation in observations)
    attempt_id = _first_present(
        observation.envelope.attempt_id for observation in observations
    )

    physical = _LayerBuilder()
    logical = _LayerBuilder()
    control = _LayerBuilder()
    epoch_sources: dict[int, set[str]] = {}
    epoch_times: dict[int, list[int]] = {}

    for observation in observations:
        contributed = _add_observation_topology(
            observation,
            physical=physical,
            logical=logical,
            control=control,
        )
        if contributed:
            epoch = _read_int(observation.raw_record, "topology_epoch")
            epoch_value = epoch if epoch is not None else 0
            epoch_sources.setdefault(epoch_value, set()).add(
                observation.envelope.raw_record_id
            )
            if observation.envelope.event_time_ns is not None:
                epoch_times.setdefault(epoch_value, []).append(
                    observation.envelope.event_time_ns
                )

    layers = (
        physical.build(GraphLayer.PHYSICAL),
        logical.build(GraphLayer.LOGICAL),
        control.build(GraphLayer.CONTROL),
    )
    quality_findings.extend(_missing_topology_findings(layers))
    topology_epochs = tuple(
        TopologyEpoch(
            epoch=epoch,
            start_time_ns=min(epoch_times.get(epoch, ()))
            if epoch_times.get(epoch)
            else None,
            end_time_ns=max(epoch_times.get(epoch, ()))
            if epoch_times.get(epoch)
            else None,
            source_record_ids=tuple(sorted(epoch_sources[epoch])),
        )
        for epoch in sorted(epoch_sources)
    )
    return TopologySnapshot(
        run_id=run_id,
        attempt_id=attempt_id,
        layers=layers,
        topology_epochs=topology_epochs,
        quality_findings=tuple(quality_findings),
    )


def _add_observation_topology(
    observation: NormalizedObservation,
    *,
    physical: _LayerBuilder,
    logical: _LayerBuilder,
    control: _LayerBuilder,
) -> bool:
    envelope = observation.envelope
    raw_record = observation.raw_record
    contributed = False

    host_id = None
    if envelope.host_name is not None:
        host_id = physical.add_entity("host", envelope.host_name)
        contributed = True

    process_id = None
    if envelope.process_id is not None:
        process_id = physical.add_entity(
            "process",
            envelope.process_id,
            role=envelope.role,
            actor_id=envelope.actor_id,
            pid=envelope.pid,
            global_rank=envelope.global_rank,
            local_rank=envelope.local_rank,
            world_size=envelope.world_size,
        )
        contributed = True
        if host_id is not None:
            physical.add_edge("process_on_host", process_id, host_id)

    device_id = None
    device_identity = (
        envelope.device_uuid
        if envelope.device_uuid is not None
        else str(envelope.device_index)
        if envelope.device_index is not None
        else None
    )
    if device_identity is not None:
        device_id = physical.add_entity(
            "device",
            device_identity,
            device_index=envelope.device_index,
            device_uuid=envelope.device_uuid,
            device_type=_read_string(raw_record, "device_type"),
        )
        contributed = True
        if process_id is not None:
            physical.add_edge("process_uses_device", process_id, device_id)

    rank_id = None
    if envelope.global_rank is not None:
        rank_id = logical.add_entity(
            "rank",
            str(envelope.global_rank),
            global_rank=envelope.global_rank,
            local_rank=envelope.local_rank,
            world_size=envelope.world_size,
            role=envelope.role,
            actor_id=envelope.actor_id,
        )
        contributed = True
        if process_id is not None:
            logical.add_edge("process_has_rank", process_id, rank_id)

    mesh_axis_id = None
    for mesh_axis in _read_string_values(raw_record, "mesh_axis", "mesh_axes"):
        mesh_axis_id = logical.add_entity("mesh_axis", mesh_axis)
        contributed = True
        if rank_id is not None:
            logical.add_edge("rank_on_mesh_axis", rank_id, mesh_axis_id)

    process_group_id = None
    process_group_name = _read_string(raw_record, "process_group_id")
    if process_group_name is not None:
        process_group_id = logical.add_entity(
            "process_group",
            process_group_name,
            backend=_read_string(raw_record, "process_group_backend"),
            topology_epoch=_read_int(raw_record, "topology_epoch"),
        )
        contributed = True
        if rank_id is not None:
            logical.add_edge("rank_in_process_group", rank_id, process_group_id)
        if mesh_axis_id is not None:
            logical.add_edge(
                "process_group_on_mesh_axis", process_group_id, mesh_axis_id
            )

    observation_id = control.add_entity(
        "observation",
        _source_record_key(observation),
        observation_id=observation.id,
        observation_kind=observation.kind.value,
        record_type=envelope.record_type,
        source_path=envelope.source_path,
        raw_record_id=envelope.raw_record_id,
    )
    contributed = True

    sensor_id = None
    if envelope.sensor_id is not None:
        sensor_id = control.add_entity("sensor", envelope.sensor_id)
        control.add_edge("sensor_observed_record", sensor_id, observation_id)
        if process_id is not None:
            control.add_edge("process_observed_by_sensor", sensor_id, process_id)
        contributed = True

    if observation.kind == ObservationKind.ARTIFACT:
        artifact_name = _read_string(raw_record, "artifact_id")
        if artifact_name is not None:
            artifact_id = control.add_entity(
                "artifact",
                artifact_name,
                producer=_read_string(raw_record, "producer"),
                artifact_kind=_read_string(raw_record, "kind"),
                state=_read_string(raw_record, "state"),
                path=_read_string(raw_record, "path"),
            )
            contributed = True
            control.add_edge("record_describes_artifact", observation_id, artifact_id)
            if process_id is not None:
                control.add_edge("process_produced_artifact", process_id, artifact_id)

            checkpoint_name = _checkpoint_name(raw_record)
            if checkpoint_name is not None:
                checkpoint_id = control.add_entity(
                    "checkpoint",
                    checkpoint_name,
                    state=_read_string(raw_record, "state"),
                    path=_read_string(raw_record, "path"),
                )
                control.add_edge(
                    "artifact_records_checkpoint", artifact_id, checkpoint_id
                )

    incident_name = _read_string(raw_record, "incident_id")
    if incident_name is not None:
        incident_id = control.add_entity(
            "incident",
            incident_name,
            phase=envelope.phase,
            step=envelope.step,
        )
        control.add_edge("record_describes_incident", observation_id, incident_id)
        if process_id is not None:
            control.add_edge("incident_observed_on_process", incident_id, process_id)
        contributed = True

    return contributed


def _missing_topology_findings(
    layers: Iterable[LayerGraph],
) -> list[QualityFinding]:
    entity_ids = {entity["id"] for layer in layers for entity in layer.entities}
    findings: list[QualityFinding] = []
    if not any(entity_id.startswith("device:") for entity_id in entity_ids):
        findings.append(
            QualityFinding(
                kind="missing_device_topology",
                severity="warning",
                evidence_state="unknown",
                message=(
                    "No explicit device topology evidence was present; "
                    "device-localization diagnoses are blocked."
                ),
                metadata={
                    "blocked_diagnoses": [
                        "device-localization",
                        "host-device-contention",
                    ],
                },
            )
        )
    if not any(entity_id.startswith("process_group:") for entity_id in entity_ids):
        findings.append(
            QualityFinding(
                kind="missing_process_group_topology",
                severity="warning",
                evidence_state="uncollected",
                message=(
                    "No explicit process-group topology evidence was present; "
                    "collective-hang-localization diagnoses are blocked."
                ),
                metadata={
                    "blocked_diagnoses": [
                        "collective-hang-localization",
                        "process-group-membership",
                    ],
                },
            )
        )
    if not any(entity_id.startswith("mesh_axis:") for entity_id in entity_ids):
        findings.append(
            QualityFinding(
                kind="missing_mesh_axis_topology",
                severity="warning",
                evidence_state="uncollected",
                message=(
                    "No explicit mesh-axis topology evidence was present; "
                    "mesh-axis localization diagnoses are blocked."
                ),
                metadata={
                    "blocked_diagnoses": [
                        "mesh-axis-localization",
                        "parallelism-axis-attribution",
                    ],
                },
            )
        )
    return findings


def _checkpoint_name(row: Mapping[str, Any]) -> str | None:
    checkpoint_id = _read_string(row, "checkpoint_id")
    if checkpoint_id is not None:
        return checkpoint_id
    artifact_kind = _read_string(row, "kind")
    artifact_id = _read_string(row, "artifact_id")
    if artifact_kind is not None and "checkpoint" in artifact_kind and artifact_id:
        return artifact_id
    return None


def _source_record_key(observation: NormalizedObservation) -> str:
    return observation.envelope.raw_record_id


def _read_string(row: Mapping[str, Any], field_name: str) -> str | None:
    value = row.get(field_name)
    return value if isinstance(value, str) else None


def _read_int(row: Mapping[str, Any], field_name: str) -> int | None:
    value = row.get(field_name)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _read_string_values(
    row: Mapping[str, Any],
    single_field: str,
    repeated_field: str,
) -> tuple[str, ...]:
    values: list[str] = []
    single = _read_string(row, single_field)
    if single is not None:
        values.append(single)
    repeated = row.get(repeated_field)
    if isinstance(repeated, list):
        values.extend(value for value in repeated if isinstance(value, str))
    return tuple(dict.fromkeys(values))


def _first_present(values: Iterable[str | None]) -> str | None:
    for value in values:
        if value is not None:
            return value
    return None


def _first_int(values: Iterable[int | None]) -> int | None:
    for value in values:
        if value is not None:
            return value
    return None
