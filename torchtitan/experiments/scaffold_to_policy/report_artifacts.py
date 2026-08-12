# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Run-artifact provenance helpers for scaffold-to-policy reports."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def describe_artifact(
    path: Path,
    *,
    run_id: str,
    payload: object | None = None,
) -> dict[str, object]:
    path_contains_run_id = run_id in str(path)
    payload_contains_run_id = _payload_contains_run_id(payload, run_id)
    record: dict[str, object] = {
        "path": str(path),
        "exists": path.is_file(),
        "run_binding": {
            "run_id": run_id,
            "path_contains_run_id": path_contains_run_id,
            "payload_contains_run_id": payload_contains_run_id,
            "status": (
                "fresh"
                if path_contains_run_id or payload_contains_run_id
                else "reused_or_unscoped"
            ),
        },
    }
    if not path.is_file():
        return record

    stat = path.stat()
    record.update(
        {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256(path),
        }
    )
    return record


def describe_artifacts(
    paths: dict[str, Path],
    *,
    run_id: str,
    payloads: dict[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    payloads = payloads or {}
    return {
        name: describe_artifact(
            path,
            run_id=run_id,
            payload=payloads.get(name),
        )
        for name, path in paths.items()
    }


def summarize_artifact_freshness(
    artifacts: dict[str, Any],
) -> dict[str, object]:
    flat = list(_iter_artifact_records(artifacts))
    statuses: dict[str, int] = {}
    for record in flat:
        run_binding = record.get("run_binding", {})
        if isinstance(run_binding, dict):
            status = str(run_binding.get("status", "unknown"))
        else:
            status = "unknown"
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "num_artifacts": len(flat),
        "all_labeled": all("run_binding" in record for record in flat),
        "status_counts": statuses,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_contains_run_id(payload: object | None, run_id: str) -> bool:
    if payload is None:
        return False
    if isinstance(payload, str):
        return payload == run_id or run_id in payload
    if isinstance(payload, dict):
        return any(_payload_contains_run_id(value, run_id) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(_payload_contains_run_id(value, run_id) for value in payload)
    return False


def _iter_artifact_records(value: object):
    if isinstance(value, dict):
        if "path" in value and "exists" in value:
            yield value
            return
        for child in value.values():
            yield from _iter_artifact_records(child)
