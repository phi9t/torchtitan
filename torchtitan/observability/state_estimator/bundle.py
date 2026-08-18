# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read raw run-evidence bundles for offline state estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from torchtitan.observability.state_estimator.schema import (
    EntityRef,
    QualityFinding,
    read_json_object,
)


@dataclass(slots=True)
class RunEvidenceBundle:
    attempt_path: Path
    manifest: dict[str, Any]
    artifact_rows: list[dict[str, Any]] = field(default_factory=list)
    event_rows: list[dict[str, Any]] = field(default_factory=list)
    outcome_rows: list[dict[str, Any]] = field(default_factory=list)
    quality_findings: list[QualityFinding] = field(default_factory=list)
    row_source_paths: dict[int, str] = field(default_factory=dict)
    row_record_ids: dict[int, str] = field(default_factory=dict)


def _read_jsonl(
    path: Path,
    *,
    quality: list[QualityFinding],
    row_source_paths: dict[int, str],
    row_record_ids: dict[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            quality.append(
                QualityFinding(
                    kind="malformed_jsonl",
                    severity="warning",
                    evidence_state="malformed",
                    message=f"{path}:{line_number}: {exc}",
                    source_path=str(path),
                )
            )
            continue
        if isinstance(row, dict):
            row_source_paths[id(row)] = str(path)
            row_record_ids[id(row)] = f"{path}:{line_number}"
            rows.append(row)
        else:
            quality.append(
                QualityFinding(
                    kind="non_object_jsonl",
                    severity="warning",
                    evidence_state="malformed",
                    message=f"{path}:{line_number}: row is not an object",
                    source_path=str(path),
                )
            )
    return rows


def load_bundle(attempt_path: Path) -> RunEvidenceBundle:
    manifest_path = attempt_path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"{manifest_path}: manifest.json is required")
    manifest = read_json_object(manifest_path)
    quality: list[QualityFinding] = []
    artifact_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    row_source_paths: dict[int, str] = {}
    row_record_ids: dict[int, str] = {}

    indexes = attempt_path / "indexes"
    if indexes.exists():
        for path in sorted(indexes.glob("artifacts.*.jsonl")):
            artifact_rows.extend(
                _read_jsonl(
                    path,
                    quality=quality,
                    row_source_paths=row_source_paths,
                    row_record_ids=row_record_ids,
                )
            )
    else:
        quality.append(
            QualityFinding(
                kind="missing_indexes_dir",
                severity="warning",
                evidence_state="missing",
                message="indexes directory is missing",
                source_path=str(indexes),
            )
        )

    structured_logs = attempt_path / "structured_logs"
    if structured_logs.exists():
        for path in sorted(structured_logs.glob("*.jsonl")):
            event_rows.extend(
                _read_jsonl(
                    path,
                    quality=quality,
                    row_source_paths=row_source_paths,
                    row_record_ids=row_record_ids,
                )
            )
    else:
        quality.append(
            QualityFinding(
                kind="missing_structured_logs_dir",
                severity="warning",
                evidence_state="uncollected",
                message="structured_logs directory is missing",
                source_path=str(structured_logs),
            )
        )

    processes = attempt_path / "processes"
    if processes.exists():
        for path in sorted(processes.glob("*/outcome.json")):
            try:
                outcome = read_json_object(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                quality.append(
                    QualityFinding(
                        kind="malformed_outcome",
                        severity="warning",
                        evidence_state="malformed",
                        message=str(exc),
                        source_path=str(path),
                        entity=EntityRef(kind="outcome", id=path.parent.name),
                    )
                )
            else:
                row_source_paths[id(outcome)] = str(path)
                row_record_ids[id(outcome)] = str(path)
                outcome_rows.append(outcome)
    else:
        quality.append(
            QualityFinding(
                kind="missing_processes_dir",
                severity="warning",
                evidence_state="missing",
                message="processes directory is missing",
                source_path=str(processes),
            )
        )

    return RunEvidenceBundle(
        attempt_path=attempt_path,
        manifest=manifest,
        artifact_rows=artifact_rows,
        event_rows=event_rows,
        outcome_rows=outcome_rows,
        quality_findings=quality,
        row_source_paths=row_source_paths,
        row_record_ids=row_record_ids,
    )
