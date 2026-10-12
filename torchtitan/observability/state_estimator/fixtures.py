# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small host-side run-evidence fixtures for state-estimator tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torchtitan.observability.state_estimator.schema import canonical_json


@dataclass(frozen=True, slots=True)
class ProcessFixture:
    process_id: str
    global_rank: int
    local_rank: int
    world_size: int
    role: str = "trainer"
    actor_id: str = "core"
    host_name: str = "fixture-host"
    pid: int = 1000
    outcome: str = "succeeded"
    phases: Sequence[tuple[str, int, int, int]] = field(default_factory=tuple)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(data) + "\n")


def _process_context(
    *, run_id: str, attempt_id: str, process: ProcessFixture
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "process_id": process.process_id,
        "role": process.role,
        "actor_id": process.actor_id,
        "host_name": process.host_name,
        "pid": process.pid,
        "global_rank": process.global_rank,
        "local_rank": process.local_rank,
        "world_size": process.world_size,
    }


def write_structured_events(
    attempt_path: Path,
    process_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path = attempt_path / "structured_logs" / f"{process_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(dict(row)) + "\n" for row in rows))
    return path


def write_artifact_index(
    attempt_path: Path,
    process_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path = attempt_path / "indexes" / f"artifacts.{process_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(dict(row)) + "\n" for row in rows))
    return path


def build_minimal_evidence_bundle(
    root: Path,
    *,
    run_id: str,
    attempt_id: str,
    processes: Sequence[ProcessFixture],
) -> Path:
    attempt_path = root / "run_evidence" / run_id / attempt_id
    attempt_path.mkdir(parents=True, exist_ok=True)
    _write_json(
        attempt_path / "manifest.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "config": {
                "normalized": {"training": {"steps": 1}},
                "sha256": "fixture",
            },
            "source": {"revision": "fixture", "dirty": False},
            "command": "fixture",
            "runtime": {"python": "fixture", "torch": "fixture"},
        },
    )
    for process in processes:
        context = _process_context(
            run_id=run_id, attempt_id=attempt_id, process=process
        )
        rows = []
        for event_seq, (phase, step, wall_time_ns, monotonic_ns) in enumerate(
            process.phases
        ):
            rows.append(
                {
                    "evidence_schema_version": 1,
                    "event_seq": event_seq,
                    "wall_time_ns": wall_time_ns,
                    "monotonic_ns": monotonic_ns,
                    "phase": phase,
                    "step": step,
                    "message": "phase_marker",
                    **context,
                }
            )
        event_path = write_structured_events(attempt_path, process.process_id, rows)
        write_artifact_index(
            attempt_path,
            process.process_id,
            [
                {
                    "schema_version": 1,
                    "evidence_schema_version": 1,
                    "record_type": "artifact",
                    "artifact_id": f"artifact-{process.process_id}",
                    "producer": "structured_logger",
                    "kind": "torchtitan.structured_events",
                    "relation": "output",
                    "state": "complete",
                    "path": str(event_path.relative_to(root)),
                    "path_type": "external_absolute",
                    "wall_time_ns": 1,
                    "monotonic_ns": 1,
                    "artifact_seq": 0,
                    "metadata": {},
                    **context,
                }
            ],
        )
        outcome = {
            "schema_version": 1,
            "evidence_schema_version": 1,
            "record_type": "process_outcome",
            "outcome": process.outcome,
            "elapsed_monotonic_ns": 1000,
            **context,
        }
        if process.outcome == "failed":
            outcome["exception_type"] = "RuntimeError"
            outcome["exception_message"] = "fixture failure"
        _write_json(
            attempt_path / "processes" / process.process_id / "outcome.json",
            outcome,
        )
    return attempt_path
