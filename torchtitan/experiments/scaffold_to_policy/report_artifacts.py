# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Run-artifact provenance helpers for scaffold-to-policy reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def build_report_input(
    *,
    data_root: Path,
    results_root: Path,
    run_id: str,
    task: str,
    lane: str,
    scaffold: dict[str, object],
    split_registry: Path,
    summary_paths: dict[str, Path],
    verifier: dict[str, object],
    preflight_paths: dict[str, Path] | None = None,
    preflight_check_name: str | None = None,
    runtime_path: Path | None = None,
    summary_to_registry_split: dict[str, str] | None = None,
    extra_artifacts: dict[str, object] | None = None,
    extra_checks: dict[str, bool] | None = None,
    extra_sections: dict[str, object] | None = None,
) -> dict[str, object]:
    summaries = load_json_files(summary_paths)
    preflights = load_json_files(preflight_paths or {})
    runtime = load_json(runtime_path) if runtime_path is not None else None
    registry = load_json(split_registry)
    registry_splits = registry["splits"]
    artifact_details = {
        "split_registry": describe_artifact(
            split_registry,
            run_id=run_id,
            payload=registry,
        ),
        "summaries": describe_artifacts(
            summary_paths,
            run_id=run_id,
            payloads=summaries,
        ),
        "preflights": describe_artifacts(
            preflight_paths or {},
            run_id=run_id,
            payloads=preflights,
        ),
    }
    if runtime_path is not None:
        artifact_details["runtime"] = describe_artifact(
            runtime_path,
            run_id=run_id,
            payload=runtime,
        )
    freshness = summarize_artifact_freshness(artifact_details)
    checks = {
        "split_registry_selected": bool(registry.get("selected", False)),
        "summaries_present": all(path.is_file() for path in summary_paths.values()),
        "summary_split_counts_match": all(
            summaries[summary_name]["num_problems"]
            == registry_splits[
                _registry_split_name(
                    summary_name,
                    registry_splits,
                    summary_to_registry_split or {},
                )
            ]["num_problems"]
            for summary_name in summary_paths
        ),
        "preflights_present": all(
            path.is_file() for path in (preflight_paths or {}).values()
        ),
        "runtime_metadata_present": runtime_path is None or runtime_path.is_file(),
        "preflight_split_counts_match": all(
            preflights[split]["num_problems"] == registry_splits[split]["num_problems"]
            for split in preflights
        ),
        "artifact_provenance_labeled": bool(freshness["all_labeled"]),
    }
    if preflight_check_name is not None:
        checks[preflight_check_name] = all(
            bool(preflight.get("selected", False)) for preflight in preflights.values()
        )
    checks.update(extra_checks or {})

    report = {
        "schema_version": 1,
        "run": {
            "run_id": run_id,
            "task": task,
            "lane": lane,
            "scaffold": scaffold,
        },
        "artifacts": {
            "data_root": str(data_root),
            "results_root": str(results_root),
            "split_registry": str(split_registry),
            "summaries": {split: str(path) for split, path in summary_paths.items()},
            "preflights": {
                split: str(path) for split, path in (preflight_paths or {}).items()
            },
            "runtime": None if runtime_path is None else str(runtime_path),
            "details": artifact_details,
            "freshness": freshness,
            **(extra_artifacts or {}),
        },
        "verifier": verifier,
        "checks": checks,
        "metrics": {"splits": summaries},
    }
    if preflights:
        report["preflight"] = {"splits": preflights}
    if runtime is not None:
        report["runtime"] = runtime
    report.update(extra_sections or {})
    return report


def load_json(path: Path) -> object:
    return json.loads(path.read_text())


def validate_report_identity(
    payload: dict[str, object],
    *,
    seen: dict[str, str],
) -> str:
    """Reject reusing a run_id with a different declaration digest.

    A run_id identifies one immutable scientific declaration (roadmap 3.1).
    Reusing it with a different normalized declaration is an error, because it
    would let a later, differently configured run silently masquerade as the
    same result. ``seen`` maps observed run_ids to their declaration digest and
    is updated in place. Returns the declaration digest.
    """

    run = payload.get("run")
    if not isinstance(run, dict):
        raise ValueError("report payload has no run object")
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("report payload has no run.run_id")
    declaration = payload.get("declaration")
    if declaration is None:
        # Fall back to the run object when no separate declaration is present,
        # so historical reports without an explicit declaration still bind to a
        # stable digest rather than silently skipping the check.
        declaration = run
    digest = _declaration_digest(declaration)
    prior = seen.get(run_id)
    if prior is not None and prior != digest:
        raise ValueError(
            f"run_id {run_id!r} reused with a different declaration digest: "
            f"{prior} != {digest}"
        )
    seen[run_id] = digest
    return digest


def validate_pass_at_k_budget(
    summary: dict[str, object],
    *,
    num_rollouts: int,
) -> list[int]:
    """Reject pass@k claims whose k exceeds the actual rollout budget.

    A pass@k value is only measurable when at least k rollouts were sampled per
    problem (roadmap 19, Wave F0). A summary that reports pass@32 from an
    8-rollout budget is overstating evidence. Returns the sorted list of valid
    k values.
    """

    if num_rollouts < 1:
        raise ValueError(f"num_rollouts must be >= 1, got {num_rollouts}")
    pass_at_k = summary.get("pass_at_k")
    if not isinstance(pass_at_k, dict):
        raise ValueError("summary has no pass_at_k object")
    ks = []
    over_budget = []
    for key in pass_at_k:
        k = int(key)
        if k > num_rollouts:
            over_budget.append(k)
        else:
            ks.append(k)
    if over_budget:
        raise ValueError(
            f"pass@k claims exceed the rollout budget of {num_rollouts}: "
            f"{sorted(over_budget)}"
        )
    return sorted(ks)


def _declaration_digest(declaration: object) -> str:
    normalized = json.dumps(declaration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_json_files(paths: dict[str, Path]) -> dict[str, object]:
    return {name: load_json(path) for name, path in paths.items()}


def build_latest_report_index(
    *,
    manifests_dir: Path,
    pattern: str = "report_input_*.json",
    task: str | None = None,
) -> dict[str, object]:
    candidates = []
    for path in sorted(manifests_dir.glob(pattern)):
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"report input at {path} is not a JSON object")
        run = payload.get("run")
        if not isinstance(run, dict):
            raise ValueError(f"report input at {path} has no run object")
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"report input at {path} has no run.run_id")
        if task is not None and run.get("task") != task:
            continue
        checks = payload.get("checks", {})
        if not isinstance(checks, dict):
            raise ValueError(f"report input at {path} has non-object checks")
        candidates.append(
            {
                "run_id": run_id,
                "path": str(path),
                "task": run.get("task"),
                "lane": run.get("lane"),
                "checks_passed": all(bool(value) for value in checks.values()),
                "artifact": describe_artifact(path, run_id=run_id, payload=payload),
            }
        )
    candidates.sort(
        key=lambda candidate: (str(candidate["run_id"]), str(candidate["path"]))
    )
    latest_attempt = candidates[-1] if candidates else None
    valid_candidates = [c for c in candidates if c["checks_passed"]]
    latest_valid = valid_candidates[-1] if valid_candidates else None
    return {
        "schema_version": 1,
        "kind": "latest_report_input_index",
        "selected": latest_attempt is not None,
        "manifests_dir": str(manifests_dir),
        "pattern": pattern,
        "task": task,
        # latest_attempt is the most recent terminal report input regardless of
        # its checks; latest_valid is the most recent one whose checks all pass.
        # Reporting both keeps a recent failure from silently replacing or hiding
        # the last valid evidence (roadmap 7.3). ``latest`` aliases
        # ``latest_attempt`` for backward compatibility.
        "latest_attempt": latest_attempt,
        "latest_valid": latest_valid,
        "latest": latest_attempt,
        "num_candidates": len(candidates),
        "num_valid": len(valid_candidates),
        "candidates": candidates,
    }


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
        return any(
            _payload_contains_run_id(value, run_id) for value in payload.values()
        )
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


def _registry_split_name(
    summary_name: str,
    registry_splits: dict[str, object],
    summary_to_registry_split: dict[str, str],
) -> str:
    if summary_name in summary_to_registry_split:
        return summary_to_registry_split[summary_name]
    if summary_name in registry_splits:
        return summary_name
    for split in registry_splits:
        if summary_name.endswith(f"_{split}"):
            return split
    raise KeyError(f"summary {summary_name!r} does not match a registry split")
