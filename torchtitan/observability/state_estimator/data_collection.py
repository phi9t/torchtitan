# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Describe available and missing signals for offline state estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchtitan.observability.state_estimator.bundle import RunEvidenceBundle
from torchtitan.observability.state_estimator.schema import EvidenceState, SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class DataSignal:
    name: str
    tier: str
    state: EvidenceState
    required: bool
    producer: str | None = None
    kind: str | None = None
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "state": self.state,
            "required": self.required,
            "producer": self.producer,
            "kind": self.kind,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class DataCollectionPlan:
    run_id: str
    attempt_id: str
    signals: tuple[DataSignal, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "signals": [signal.to_json() for signal in self.signals],
        }


def _has_artifact(
    bundle: RunEvidenceBundle, *, producer: str | None = None, kind: str | None = None
) -> bool:
    for row in bundle.artifact_rows:
        if producer is not None and row.get("producer") != producer:
            continue
        if kind is not None and row.get("kind") != kind:
            continue
        return True
    return False


def build_data_collection_plan(bundle: RunEvidenceBundle) -> DataCollectionPlan:
    signals = [
        DataSignal("manifest", "required_raw", "present", True),
        DataSignal(
            "process_outcome",
            "required_raw",
            "present" if bundle.outcome_rows else "missing",
            True,
        ),
        DataSignal(
            "artifact_index",
            "required_raw",
            "present" if bundle.artifact_rows else "missing",
            True,
        ),
        DataSignal(
            "structured_events",
            "tier0_semantic",
            "present"
            if _has_artifact(bundle, kind="torchtitan.structured_events")
            else "uncollected",
            False,
            producer="structured_logger",
            kind="torchtitan.structured_events",
        ),
        DataSignal(
            "dcgm",
            "tier0_hardware",
            "present" if _has_artifact(bundle, producer="dcgm") else "uncollected",
            False,
            producer="dcgm",
        ),
        DataSignal(
            "profiler",
            "tier1_trace",
            "present"
            if _has_artifact(bundle, kind="pytorch.profiler.trace")
            else "uncollected",
            False,
            kind="pytorch.profiler.trace",
        ),
        DataSignal(
            "flight_recorder",
            "tier1_trace",
            "present"
            if _has_artifact(bundle, kind="pytorch.flight_recorder.dump")
            else "uncollected",
            False,
            kind="pytorch.flight_recorder.dump",
        ),
        DataSignal(
            "incident",
            "tier2_incident",
            "present"
            if _has_artifact(bundle, producer="incident_recorder")
            else "uncollected",
            False,
            producer="incident_recorder",
        ),
    ]
    return DataCollectionPlan(
        run_id=str(bundle.manifest["run_id"]),
        attempt_id=str(bundle.manifest["attempt_id"]),
        signals=tuple(signals),
    )
