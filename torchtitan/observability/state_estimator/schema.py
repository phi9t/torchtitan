# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared schemas for offline training state-estimator artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal, Mapping


SCHEMA_VERSION = 1

EvidenceState = Literal[
    "present",
    "missing",
    "unknown",
    "uncollected",
    "malformed",
    "contradictory",
]
Severity = Literal["info", "warning", "error"]
CalibrationState = Literal["heuristic", "calibrated", "not_applicable"]


@dataclass(frozen=True, slots=True)
class EntityRef:
    kind: str
    id: str

    def to_json(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.id}


@dataclass(frozen=True, slots=True)
class QualityFinding:
    kind: str
    severity: Severity
    evidence_state: EvidenceState
    message: str
    source_path: str | None = None
    entity: EntityRef | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "severity": self.severity,
            "evidence_state": self.evidence_state,
            "message": self.message,
            "source_path": self.source_path,
            "entity": self.entity.to_json() if self.entity else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Observation:
    id: str
    kind: str
    entity: EntityRef
    event_time_ns: int | None
    ingestion_time_ns: int | None
    source_path: str
    quality: EvidenceState
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "kind": self.kind,
            "entity": self.entity.to_json(),
            "event_time_ns": self.event_time_ns,
            "ingestion_time_ns": self.ingestion_time_ns,
            "source_path": self.source_path,
            "quality": self.quality,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class DerivedPaths:
    root: Path
    evidence_graph: Path
    entities: Path
    timeline: Path
    quality: Path
    belief_summary: Path
    diagnosis: Path
    evaluation_summary: Path
    evaluation_report: Path

    @classmethod
    def from_attempt_path(cls, attempt_path: Path) -> "DerivedPaths":
        root = attempt_path / "derived" / "state_estimator"
        return cls(
            root=root,
            evidence_graph=root / "evidence_graph.json",
            entities=root / "entities.json",
            timeline=root / "timeline.jsonl",
            quality=root / "quality.json",
            belief_summary=root / "belief_summary.json",
            diagnosis=root / "diagnosis.md",
            evaluation_summary=root / "evaluation_summary.json",
            evaluation_report=root / "evaluation_report.md",
        )


def canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(canonical_json(data) + "\n")
    tmp.replace(path)


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value
