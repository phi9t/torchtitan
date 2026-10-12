# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Versioned evidence contracts and legacy normalization for Falcon.

Native Falcon runs write one immutable bundle per launch attempt. Campaign B
predates that contract, so the legacy importer only references source artifacts
by path and digest. It never copies checkpoints or rewrites historical output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


FALCON_EVIDENCE_SCHEMA_VERSION = 1
FALCON_PRE_B09_CATALOG_ID = "falcon_pre_b09_v1"
_NATIVE_REQUIRED_FIELDS = (
    "run_id",
    "attempt_id",
    "lane",
    "mode",
    "arm",
    "logical_run",
    "claim_label",
    "evidence_tier",
    "environment_class",
    "processes",
    "mesh",
    "device",
    "clocks",
    "steps",
    "phases",
    "data_lineage",
    "checkpoint_lineage",
    "artifacts",
    "outcome",
)
_PROCESS_REQUIRED_FIELDS = (
    "process_id",
    "role",
    "global_rank",
    "local_rank",
    "world_size",
    "host_name",
)
_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class EvidenceContractError(ValueError):
    """The supplied evidence does not meet the versioned contract."""


class EvidenceConflictError(RuntimeError):
    """An immutable bundle or ledger conflicts with an existing artifact."""


@dataclass(frozen=True)
class LegacyFixture:
    """One normalized view of a legacy source artifact.

    ``facts`` are directly declared or mechanically selected from ``source``.
    ``unavailable`` names every native-required fact the legacy artifact cannot
    prove, rather than inventing a value during normalization.
    """

    key: str
    source_path: str
    artifact_role: str
    record_status: str
    facts: dict[str, Any]
    unavailable: dict[str, str]
    source_selector: str | None = None
    inventory_scope: str = "campaign_b_recap"
    related_source_paths: tuple[str, ...] = ()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _json_safe(value: Any) -> Any:
    """Preserve legacy facts while converting non-JSON numeric sentinels to null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    """Load the complete structured artifact index without guessing omitted rows."""
    records = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise EvidenceContractError(f"JSONL artifact row must be an object: {path}")
        records.append(value)
    return records


def _digest_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json(_json_safe(value))).hexdigest()


def _select_source_value(repo_root: Path, source_path: str, selector: str) -> Any:
    """Resolve one deterministic legacy selector to its source-declared value."""
    source, _ = _relative_source_path(repo_root, source_path)
    if selector.startswith("lines[") and selector.endswith("]"):
        match = re.fullmatch(r"lines\[(\d+):(\d+)\]", selector)
        if match is None:
            raise EvidenceContractError("document selector must be a line range")
        start, end = (int(value) for value in match.groups())
        lines = source.read_text().splitlines()
        if start < 1 or end < start or end > len(lines):
            raise EvidenceContractError("document selector is outside its source")
        return {"start_line": start, "end_line": end, "lines": lines[start - 1 : end]}
    try:
        value = json.loads(source.read_text())
    except json.JSONDecodeError as exc:
        raise EvidenceContractError(
            "non-document source requires a line selector"
        ) from exc
    if selector in {"summary", "hero_outcome"}:
        if not isinstance(value, dict):
            raise EvidenceContractError("object selector does not resolve")
        return value
    match = re.fullmatch(r"arms\[([^,]+),(\d+)\]", selector)
    if match and isinstance(value, dict):
        arm, seed = match.groups()
        for row in value.get("arms", []):
            if row.get("arm") == arm and row.get("seed") == int(seed):
                return row
    match = re.fullmatch(r"addition\[([^]]+)\]", selector)
    if (
        match
        and isinstance(value, dict)
        and match.group(1) in value.get("addition", {})
    ):
        return value["addition"][match.group(1)]
    if selector == "processes[trainer.core.global_rank_000000].outcome" and isinstance(
        value, dict
    ):
        return value
    raise EvidenceContractError("source selector does not resolve")


def _source_binding(
    repo_root: Path,
    *,
    source_path: str,
    source_selector: str,
    facts: dict[str, Any],
    inferred_linkage: Any,
    source_declared_ids: dict[str, Any],
    unavailable: dict[str, str],
) -> dict[str, str]:
    """Bind normalized claims to selected source bytes without inventing provenance."""
    selected = _select_source_value(repo_root, source_path, source_selector)
    normalized = {
        "facts": _json_safe(facts),
        "inferred_linkage": _json_safe(inferred_linkage),
        "source_declared_ids": source_declared_ids,
        "unavailable": unavailable,
    }
    return {
        "algorithm": "sha256",
        "selected_source_digest": _digest_value(selected),
        "normalized_record_digest": _digest_value(normalized),
    }


def _ledger_binding(payload: dict[str, Any]) -> dict[str, str]:
    """Bind all ledger claims and classifications into one immutable payload hash."""
    fields = (
        "records",
        "sources",
        "artifact_index",
        "excluded_derived_artifacts",
        "associated_artifacts",
        "inventory_subtotals",
    )
    return {
        "algorithm": "sha256",
        "digest": _digest_value({field: payload.get(field) for field in fields}),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_source_path(repo_root: Path, source_path: str) -> tuple[Path, str]:
    candidate = repo_root / source_path
    resolved_root = repo_root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        relative = resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise EvidenceContractError(
            f"source_path escapes repository: {source_path}"
        ) from exc
    if not resolved_candidate.is_file():
        raise EvidenceContractError(f"legacy source is missing: {source_path}")
    return resolved_candidate, relative.as_posix()


def _require_nonempty_string(record: dict[str, Any], field: str) -> None:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise EvidenceContractError(f"{field} must be a nonempty string")


def _require_safe_segment(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not _SAFE_SEGMENT_RE.fullmatch(value)
        or value in {".", ".."}
    ):
        raise EvidenceContractError(f"{field} must be a safe path segment")
    return value


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceContractError(f"{field} must be an integer >= {minimum}")
    return value


def canonical_native_run_id(logical_run: dict[str, Any]) -> str:
    """Return the single native run identity for one workload/arm/seed cell."""
    if not isinstance(logical_run, dict):
        raise EvidenceContractError("logical_run must be an object")
    workload = _require_safe_segment(
        logical_run.get("workload"), "logical_run.workload"
    )
    arm = _require_safe_segment(logical_run.get("arm"), "logical_run.arm")
    seed = _require_int(logical_run.get("seed"), "logical_run.seed")
    return f"falcon-{workload}-{arm}-seed{seed}"


def _validate_native_attempt(record: dict[str, Any]) -> None:
    missing = [field for field in _NATIVE_REQUIRED_FIELDS if field not in record]
    if missing:
        raise EvidenceContractError(
            f"native Falcon evidence missing required fields: {', '.join(missing)}"
        )
    for field in (
        "run_id",
        "attempt_id",
        "lane",
        "mode",
        "arm",
        "claim_label",
        "evidence_tier",
        "environment_class",
    ):
        _require_nonempty_string(record, field)
    logical_run = record["logical_run"]
    if not isinstance(logical_run, dict) or logical_run.get("arm") != record["arm"]:
        raise EvidenceContractError("logical_run must identify the declared arm")
    canonical_run_id = canonical_native_run_id(logical_run)
    if record["run_id"] != canonical_run_id:
        raise EvidenceContractError(
            "run_id must equal the canonical logical-run identity"
        )
    _require_safe_segment(record["run_id"], "run_id")
    _require_safe_segment(record["attempt_id"], "attempt_id")
    processes = record["processes"]
    if not isinstance(processes, list) or not processes:
        raise EvidenceContractError("processes must be a nonempty list")
    for process in processes:
        if not isinstance(process, dict):
            raise EvidenceContractError("each process must be an object")
        missing_process = [
            field for field in _PROCESS_REQUIRED_FIELDS if field not in process
        ]
        if missing_process:
            raise EvidenceContractError(
                f"process identity missing required fields: {', '.join(missing_process)}"
            )
        for field in ("process_id", "role", "host_name"):
            if not isinstance(process[field], str) or not process[field].strip():
                raise EvidenceContractError(
                    f"process {field} must be a nonempty string"
                )
        global_rank = _require_int(process["global_rank"], "process global_rank")
        local_rank = _require_int(process["local_rank"], "process local_rank")
        world_size = _require_int(
            process["world_size"], "process world_size", minimum=1
        )
        if global_rank >= world_size or local_rank >= world_size:
            raise EvidenceContractError("process ranks must be smaller than world_size")
    for field in (
        "mesh",
        "device",
        "clocks",
        "steps",
        "data_lineage",
        "checkpoint_lineage",
        "outcome",
    ):
        if not isinstance(record[field], dict):
            raise EvidenceContractError(f"{field} must be an object")
    for field in ("phases", "artifacts"):
        if not isinstance(record[field], list) or not record[field]:
            raise EvidenceContractError(f"{field} must be a nonempty list")
    mesh_axes = record["mesh"].get("axes")
    if not isinstance(mesh_axes, dict) or not mesh_axes:
        raise EvidenceContractError("mesh.axes must be a nonempty object")
    mesh_world_size = 1
    for axis, size in mesh_axes.items():
        _require_safe_segment(axis, "mesh axis")
        mesh_world_size *= _require_int(size, f"mesh.axes.{axis}", minimum=1)
    if any(process["world_size"] != mesh_world_size for process in processes):
        raise EvidenceContractError("process world_size must equal the mesh size")
    if len({process["global_rank"] for process in processes}) != len(processes):
        raise EvidenceContractError("process global ranks must be unique")
    if len({process["process_id"] for process in processes}) != len(processes):
        raise EvidenceContractError("process IDs must be unique")
    for field in ("type", "uuid"):
        if (
            not isinstance(record["device"].get(field), str)
            or not record["device"][field]
        ):
            raise EvidenceContractError(f"device.{field} must be a nonempty string")
    _require_int(record["device"].get("index"), "device.index")
    parsed_clocks = []
    for field in ("started_utc", "ended_utc"):
        if (
            not isinstance(record["clocks"].get(field), str)
            or not record["clocks"][field]
        ):
            raise EvidenceContractError(f"clocks.{field} must be a nonempty string")
        try:
            parsed = datetime.fromisoformat(
                record["clocks"][field].replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise EvidenceContractError(f"clocks.{field} must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise EvidenceContractError(f"clocks.{field} must include a timezone")
        parsed_clocks.append(parsed)
    if parsed_clocks[1] < parsed_clocks[0]:
        raise EvidenceContractError("clocks.ended_utc must not precede started_utc")
    for field in ("requested", "completed"):
        _require_int(record["steps"].get(field), f"steps.{field}")
    if record["steps"]["completed"] > record["steps"]["requested"]:
        raise EvidenceContractError("steps.completed must not exceed steps.requested")
    if not record["data_lineage"] or not record["checkpoint_lineage"]:
        raise EvidenceContractError(
            "data_lineage and checkpoint_lineage must be nonempty"
        )
    for name, lineage in (
        ("data_lineage", record["data_lineage"]),
        ("checkpoint_lineage", record["checkpoint_lineage"]),
    ):
        if any(
            not isinstance(key, str)
            or not key
            or (value is not None and not isinstance(value, str))
            for key, value in lineage.items()
        ):
            raise EvidenceContractError(
                f"{name} must map nonempty names to strings or null"
            )
        if not any(isinstance(value, str) and value for value in lineage.values()):
            raise EvidenceContractError(
                f"{name} must contain at least one nonempty lineage reference"
            )
    if (
        not isinstance(record["outcome"].get("status"), str)
        or not record["outcome"]["status"]
    ):
        raise EvidenceContractError("outcome.status must be a nonempty string")
    phase_names = set()
    for phase in record["phases"]:
        if (
            not isinstance(phase, dict)
            or not isinstance(phase.get("name"), str)
            or not isinstance(phase.get("outcome"), str)
            or not phase["name"]
            or not phase["outcome"]
        ):
            raise EvidenceContractError("each phase requires nonempty name and outcome")
        if phase["name"] in phase_names:
            raise EvidenceContractError("phase names must be unique")
        phase_names.add(phase["name"])
    artifact_paths = set()
    for artifact in record["artifacts"]:
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("path"), str)
            or not isinstance(artifact.get("kind"), str)
            or not artifact["path"]
            or not artifact["kind"]
        ):
            raise EvidenceContractError("each artifact requires nonempty path and kind")
        artifact_path = Path(artifact["path"])
        if (
            artifact_path.is_absolute()
            or ".." in artifact_path.parts
            or "." in artifact_path.parts
        ):
            raise EvidenceContractError("artifact path must be repository-relative")
        if artifact["path"] in artifact_paths:
            raise EvidenceContractError("artifact paths must be unique")
        artifact_paths.add(artifact["path"])


def _write_bundle_file(path: Path, contents: bytes) -> None:
    path.write_bytes(contents)


def write_native_attempt_bundle(results_root: Path, record: dict[str, Any]) -> Path:
    """Write or verify an immutable native Falcon run-attempt bundle."""
    _validate_native_attempt(record)
    runs_root = (results_root / "runs").resolve()
    run_path = runs_root / record["run_id"]
    bundle_path = run_path / record["attempt_id"]
    try:
        bundle_path.resolve().relative_to(runs_root)
    except ValueError as exc:
        raise EvidenceContractError("native bundle path escapes results root") from exc
    manifest = {
        "schema_version": FALCON_EVIDENCE_SCHEMA_VERSION,
        "record_type": "falcon_native_attempt",
        **{
            field: record[field]
            for field in _NATIVE_REQUIRED_FIELDS
            if field not in {"artifacts", "outcome"}
        },
    }
    expected = {
        "manifest.json": _canonical_json(manifest),
        "artifact_index.json": _canonical_json(record["artifacts"]),
        "outcome.json": _canonical_json(record["outcome"]),
    }
    if bundle_path.exists():
        actual = {
            name: path.read_bytes() if path.is_file() else None
            for name, path in ((name, bundle_path / name) for name in expected)
        }
        if actual != expected:
            raise EvidenceConflictError(
                f"immutable Falcon attempt bundle already exists: {bundle_path}"
            )
        return bundle_path
    run_path.mkdir(parents=True, exist_ok=True)
    for existing_manifest in run_path.glob("*/manifest.json"):
        try:
            existing = json.loads(existing_manifest.read_text())
        except json.JSONDecodeError as exc:
            raise EvidenceConflictError(
                f"existing native manifest is invalid: {existing_manifest}"
            ) from exc
        if existing.get("logical_run") != record["logical_run"]:
            raise EvidenceConflictError(
                "canonical run directory belongs to a different logical run"
            )
    stage_path = run_path / f".{record['attempt_id']}.tmp-{uuid.uuid4().hex}"
    try:
        stage_path.mkdir()
        for name, contents in expected.items():
            _write_bundle_file(stage_path / name, contents)
        os.replace(stage_path, bundle_path)
    except BaseException:
        shutil.rmtree(stage_path, ignore_errors=True)
        if run_path.exists() and not any(run_path.iterdir()):
            run_path.rmdir()
        raise
    return bundle_path


def _legacy_record(repo_root: Path, fixture: LegacyFixture) -> dict[str, Any]:
    if fixture.record_status not in {
        "valid",
        "smoke",
        "invalid",
        "incomplete",
        "omitted",
        "learnability_smoke",
    }:
        raise EvidenceContractError(
            f"unsupported legacy record status: {fixture.record_status}"
        )
    if not fixture.key or not fixture.artifact_role:
        raise EvidenceContractError(
            "legacy fixture key and artifact_role must be nonempty"
        )
    source, source_path = _relative_source_path(repo_root, fixture.source_path)
    source_digest = _sha256(source)
    selector = fixture.source_selector or fixture.key
    if not selector:
        raise EvidenceContractError("legacy fixture source_selector must be nonempty")
    related_sources = []
    for related_path in fixture.related_source_paths:
        related, normalized_path = _relative_source_path(repo_root, related_path)
        if normalized_path == source_path:
            raise EvidenceContractError(
                "legacy related source duplicates primary source"
            )
        related_sources.append(
            {
                "path": normalized_path,
                "digest_algorithm": "sha256",
                "digest": _sha256(related),
            }
        )
    absent = dict(fixture.unavailable)
    for field in _NATIVE_REQUIRED_FIELDS:
        if field not in fixture.facts and field not in absent:
            absent[field] = "legacy source does not declare this native evidence field"
    identity = {
        "schema_version": FALCON_EVIDENCE_SCHEMA_VERSION,
        "source_path": source_path,
        "source_digest": source_digest,
        "source_selector": selector,
        "artifact_role": fixture.artifact_role,
    }
    import_id = "legacy-" + hashlib.sha256(_canonical_json(identity)).hexdigest()[:24]
    source_declared_ids = {
        "run_id": fixture.facts.get("run_id"),
        "attempt_id": fixture.facts.get("attempt_id"),
    }
    return {
        "schema_version": FALCON_EVIDENCE_SCHEMA_VERSION,
        "record_type": "falcon_legacy_import",
        "import_id": import_id,
        "record_status": fixture.record_status,
        "inventory_scope": fixture.inventory_scope,
        "artifact_role": fixture.artifact_role,
        "source_selector": selector,
        "source": {
            "path": source_path,
            "digest_algorithm": "sha256",
            "digest": source_digest,
        },
        "related_sources": related_sources,
        "source_declared_ids": source_declared_ids,
        "inferred_linkage": fixture.facts.get("inferred_linkage"),
        "facts": _json_safe(fixture.facts),
        "unavailable": absent,
        "source_binding": _source_binding(
            repo_root,
            source_path=source_path,
            source_selector=selector,
            facts=fixture.facts,
            inferred_linkage=fixture.facts.get("inferred_linkage"),
            source_declared_ids=source_declared_ids,
            unavailable=absent,
        ),
    }


def _ledger_payload(
    repo_root: Path,
    fixtures: list[LegacyFixture],
    *,
    include_known_artifacts: bool = False,
    catalog_id: str | None = None,
) -> dict[str, Any]:
    records = [_legacy_record(repo_root, fixture) for fixture in fixtures]
    records = _deduplicate_semantic_records(records)
    records.sort(key=lambda record: record["import_id"])
    ids = [record["import_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise EvidenceContractError("legacy fixtures produce duplicate import IDs")
    source_digests: dict[str, str] = {}
    for record in records:
        for source in [record["source"], *record["related_sources"]]:
            existing = source_digests.setdefault(source["path"], source["digest"])
            if existing != source["digest"]:
                raise EvidenceContractError(f"source digest conflict: {source['path']}")
    sources = sorted(source_digests.items())
    campaign_b_records = [
        record for record in records if record["inventory_scope"] == "campaign_b_recap"
    ]
    artifact_index = _artifact_index(
        records, include_known_artifacts=include_known_artifacts
    )
    payload = {
        "schema_version": FALCON_EVIDENCE_SCHEMA_VERSION,
        "record_type": "falcon_legacy_ledger",
        "records": records,
        "sources": [
            {"path": path, "digest_algorithm": "sha256", "digest": digest}
            for path, digest in sources
        ],
        "artifact_index": artifact_index,
        "excluded_derived_artifacts": [
            artifact["path"]
            for artifact in artifact_index
            if artifact["classification"] == "derived_artifact"
        ],
        "associated_artifacts": [
            artifact
            for artifact in artifact_index
            if artifact["classification"].startswith("associated_")
            or artifact["classification"].startswith("tooling_only_")
        ],
        "inventory_subtotals": {
            "campaign_b_recap": {
                "record_count": len(campaign_b_records),
                "status_counts": {
                    status: sum(
                        record["record_status"] == status
                        for record in campaign_b_records
                    )
                    for status in ("valid", "smoke", "invalid", "incomplete", "omitted")
                },
            },
            "all_pre_b09_falcon": {"record_count": len(records)},
        },
    }
    if catalog_id is not None:
        payload["catalog_id"] = catalog_id
    payload["ledger_binding"] = _ledger_binding(payload)
    return payload


def _semantic_key(record: dict[str, Any]) -> tuple[Any, ...]:
    facts = record["facts"]
    linkage = record.get("inferred_linkage") or {}
    return (
        facts.get("campaign_step", linkage.get("campaign_step")),
        linkage.get("kind", linkage.get("workload", facts.get("mode"))),
        facts.get("arm"),
        facts.get("seed"),
        facts.get("requested_steps"),
        facts.get("source_tag", record["artifact_role"]),
    )


def _deduplicate_semantic_records(
    records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Collapse identical derived evidence and reject conflicting claims per cell."""
    retained: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in sorted(records, key=lambda item: item["import_id"]):
        key = _semantic_key(record)
        existing = retained.get(key)
        if existing is None:
            retained[key] = record
            continue
        compared_fields = ("record_status", "facts", "inferred_linkage", "unavailable")
        if any(existing[field] != record[field] for field in compared_fields):
            raise EvidenceConflictError(f"semantic evidence conflict: {key}")
        existing.setdefault("deduplicated_sources", []).append(record["source"])
    return list(retained.values())


def _artifact_index(
    records: list[dict[str, Any]], *, include_known_artifacts: bool
) -> list[dict[str, str]]:
    """Inventory raw, copied, derived, and associated retained artifacts."""
    artifacts: dict[str, dict[str, str]] = {}

    def add(path: str, classification: str, parent: str, hash_policy: str) -> None:
        existing = artifacts.get(path)
        entry = {
            "path": path,
            "classification": classification,
            "parent": parent,
            "hash_policy": hash_policy,
        }
        if existing is not None and existing != entry:
            raw_classes = {"raw_evidence_source", "raw_related_source"}
            if (
                existing["classification"] in raw_classes
                and classification in raw_classes
                and existing["parent"] == parent
                and existing["hash_policy"] == hash_policy
            ):
                entry["classification"] = "raw_evidence_source"
            else:
                raise EvidenceContractError(f"artifact index conflict: {path}")
        artifacts[path] = entry

    for record in records:
        add(
            record["source"]["path"],
            "raw_evidence_source",
            "legacy_ledger_records",
            "sha256_in_sources",
        )
        for source in record["related_sources"]:
            add(
                source["path"],
                "raw_related_source",
                "legacy_ledger_records",
                "sha256_in_sources",
            )

    if not include_known_artifacts:
        return [artifacts[path] for path in sorted(artifacts)]

    for path, parent in (
        (
            "experiments/falcon/results/ablations/combined/table.json",
            "step4_full_matrix",
        ),
        ("experiments/falcon/results/ablations/combined/table.md", "step4_full_matrix"),
        ("experiments/falcon/results/ablations/dry_run/table.md", "step4_dry_matrix"),
        (
            "experiments/falcon/results/ablations/full_2seed_8000/table.md",
            "step4_full_matrix",
        ),
        (
            "experiments/falcon/results/ablations/full_A1_2seed_8000/table.md",
            "step4_a1_matrix",
        ),
    ):
        add(path, "derived_artifact", parent, "sha256_when_verified")
    for arm in ("combined", "dry_run", "full_2seed_8000", "full_A1_2seed_8000"):
        canonical = "experiments/falcon/results/ablations/" f"{arm}/table.json"
        for suffix in ("table.json", "table.md"):
            add(
                ".scratch/falcon-fast-weight-attention/evidence/ablations/"
                f"{arm}/{suffix}",
                "copied_artifact",
                canonical,
                "sha256_when_verified",
            )
    for path, parent in (
        (
            ".scratch/falcon-fast-weight-attention/evidence/ebpf-nosudo-REPORT.md",
            "experiments/falcon/results/ebpf_nosudo_20260906T050950Z/REPORT.md",
        ),
        (
            ".scratch/falcon-fast-weight-attention/evidence/rootfs-all-tools-REPORT.md",
            "experiments/falcon/results/tooling/rootfs_all_tools/REPORT.md",
        ),
        (
            ".scratch/falcon-fast-weight-attention/evidence/rootfs-privileged-REPORT.md",
            "experiments/falcon/results/tooling/rootfs_privileged/REPORT.md",
        ),
        (
            ".scratch/falcon-fast-weight-attention/evidence/tooling-REPORT.md",
            "experiments/falcon/results/tooling/REPORT.md",
        ),
    ):
        add(path, "copied_artifact", parent, "sha256_when_verified")
    for tier, steps in (("hero_20k", (19000, 19500, 20000)), ("hero_dry", (4,))):
        for arm in ("A0", "A1", "A5"):
            parent = f"{tier}-{arm}-seed0"
            base = f"experiments/falcon/results/hero/{tier}/{arm}_seed0"
            add(
                f"{base}/hero_status.json",
                "associated_hero_status",
                parent,
                "sha256_when_verified",
            )
            if tier == "hero_20k":
                add(
                    f"{base}/hero.host.pid",
                    "associated_process_marker",
                    parent,
                    "sha256_when_verified",
                )
                add(
                    f"{base}/hero.log",
                    "associated_hero_log",
                    parent,
                    "sha256_when_verified",
                )
            for step in steps:
                add(
                    f"{base}/checkpoint/step-{step}",
                    "associated_checkpoint",
                    parent,
                    "directory_contents_not_hashed",
                )
    for path, classification, parent in (
        (
            "experiments/falcon/results/ebpf_nosudo_20260906T050950Z",
            "associated_matrix_diagnostic",
            "step4_full_matrix",
        ),
        (
            "experiments/falcon/results/tooling/attach_v1",
            "associated_profiler_artifact",
            "profiler-v1-a5",
        ),
        (
            "experiments/falcon/results/tooling/attach_v3",
            "associated_profiler_artifact",
            "tiny-overfit-30k-profiler",
        ),
        (
            "experiments/falcon/results/tooling/tiny_inject_v3/console.log",
            "associated_profiler_artifact",
            "tiny-overfit-30k-profiler",
        ),
        (
            "experiments/falcon/results/tooling/nsys_diag",
            "tooling_only_diagnostic",
            "none",
        ),
        (
            "experiments/falcon/results/tooling/py_profilers",
            "tooling_only_diagnostic",
            "none",
        ),
        (
            "experiments/falcon/results/tooling/rootfs_all_tools",
            "tooling_only_environment",
            "none",
        ),
        (
            "experiments/falcon/results/tooling/rootfs_privileged",
            "tooling_only_environment",
            "none",
        ),
        (
            "experiments/falcon/results/tooling/attach_privileged",
            "tooling_only_environment",
            "none",
        ),
    ):
        add(path, classification, parent, "directory_contents_not_hashed")
    return [artifacts[path] for path in sorted(artifacts)]


def _write_or_verify_ledger(
    repo_root: Path, ledger_path: Path, payload: dict[str, Any]
) -> Path:
    expected = _canonical_json(payload)
    if ledger_path.exists():
        try:
            existing = json.loads(ledger_path.read_text())
        except json.JSONDecodeError as exc:
            raise EvidenceConflictError(
                f"legacy ledger is not valid JSON: {ledger_path}"
            ) from exc
        for source in existing.get("sources", []):
            source_path, relative = _relative_source_path(repo_root, source["path"])
            if relative != source["path"] or _sha256(source_path) != source["digest"]:
                raise EvidenceConflictError(
                    f"legacy source digest changed: {source['path']}"
                )
        if ledger_path.read_bytes() != expected:
            raise EvidenceConflictError(
                f"immutable legacy ledger conflicts with regenerated evidence: {ledger_path}"
            )
        return ledger_path
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    temporary.write_bytes(expected)
    temporary.replace(ledger_path)
    return ledger_path


def import_legacy_fixtures(
    repo_root: Path,
    ledger_path: Path,
    fixtures: list[LegacyFixture],
    *,
    include_known_artifacts: bool = False,
    catalog_id: str | None = None,
) -> Path:
    """Normalize explicit legacy fixtures into a stable, immutable ledger."""
    return _write_or_verify_ledger(
        repo_root.resolve(),
        ledger_path,
        _ledger_payload(
            repo_root.resolve(),
            fixtures,
            include_known_artifacts=include_known_artifacts,
            catalog_id=catalog_id,
        ),
    )


def _read_legacy_ledger(ledger_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(ledger_path.read_text())
    except json.JSONDecodeError as exc:
        raise EvidenceContractError(
            f"legacy ledger is not valid JSON: {ledger_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise EvidenceContractError("legacy ledger must be a JSON object")
    return payload


def verify_legacy_ledger(repo_root: Path, ledger_path: Path) -> int:
    """Structurally verify an ad hoc ledger and its immutable source digests."""
    payload = _read_legacy_ledger(ledger_path)
    if "catalog_id" in payload:
        raise EvidenceContractError(
            "catalog-bearing Falcon evidence requires "
            "verify_known_pre_b09_ledger; generic verification is limited to "
            "ad hoc ledgers without catalog_id"
        )
    return _verify_legacy_payload(repo_root.resolve(), payload)


def verify_known_pre_b09_ledger(repo_root: Path, ledger_path: Path) -> int:
    """Verify the code-owned pre-B09 catalog against independently rebuilt facts."""
    payload = _read_legacy_ledger(ledger_path)
    catalog_id = payload.get("catalog_id")
    if catalog_id != FALCON_PRE_B09_CATALOG_ID:
        raise EvidenceContractError(
            "known pre-B09 ledger catalog_id must equal "
            f"{FALCON_PRE_B09_CATALOG_ID!r}; got {catalog_id!r}"
        )
    repo_root = repo_root.resolve()
    expected_payload = _ledger_payload(
        repo_root,
        _known_legacy_fixtures(repo_root),
        include_known_artifacts=True,
        catalog_id=FALCON_PRE_B09_CATALOG_ID,
    )
    if _canonical_json(payload) != _canonical_json(expected_payload):
        raise EvidenceConflictError(
            "canonical catalog does not match regenerated Falcon evidence"
        )
    return _verify_legacy_payload(repo_root, payload)


def _verify_legacy_payload(repo_root: Path, payload: dict[str, Any]) -> int:
    """Verify structure, selectors, bindings, and immutable source digests."""
    if payload.get("schema_version") != FALCON_EVIDENCE_SCHEMA_VERSION:
        raise EvidenceContractError("unsupported Falcon legacy ledger schema version")
    if payload.get("ledger_binding") != _ledger_binding(payload):
        raise EvidenceContractError("legacy ledger binding does not match its payload")
    sources = {}
    for source in payload.get("sources", []):
        if (
            not isinstance(source, dict)
            or source.get("digest_algorithm") != "sha256"
            or not isinstance(source.get("path"), str)
            or not isinstance(source.get("digest"), str)
        ):
            raise EvidenceContractError("legacy ledger source entry is invalid")
        if source["path"] in sources:
            raise EvidenceContractError("legacy ledger has duplicate source paths")
        sources[source["path"]] = source["digest"]
    for path, digest in sources.items():
        source, relative = _relative_source_path(repo_root.resolve(), path)
        if relative != path or _sha256(source) != digest:
            raise EvidenceConflictError(f"legacy source digest changed: {path}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise EvidenceContractError("legacy ledger must contain records")
    allowed_statuses = {
        "valid",
        "smoke",
        "learnability_smoke",
        "invalid",
        "incomplete",
        "omitted",
    }
    seen_import_ids = set()
    semantic_keys = set()
    record_source_paths = set()
    for record in records:
        if (
            record.get("schema_version") != FALCON_EVIDENCE_SCHEMA_VERSION
            or record.get("record_type") != "falcon_legacy_import"
            or not isinstance(record.get("artifact_role"), str)
            or not isinstance(record.get("inventory_scope"), str)
        ):
            raise EvidenceContractError("legacy record has an invalid envelope")
        if record.get("record_status") not in allowed_statuses:
            raise EvidenceContractError("legacy record has an invalid status")
        if record.get("import_id") in seen_import_ids:
            raise EvidenceContractError("legacy ledger has duplicate import IDs")
        seen_import_ids.add(record.get("import_id"))
        identity = {
            "schema_version": FALCON_EVIDENCE_SCHEMA_VERSION,
            "source_path": record.get("source", {}).get("path"),
            "source_digest": record.get("source", {}).get("digest"),
            "source_selector": record.get("source_selector"),
            "artifact_role": record.get("artifact_role"),
        }
        expected_id = (
            "legacy-" + hashlib.sha256(_canonical_json(identity)).hexdigest()[:24]
        )
        if record.get("import_id") != expected_id:
            raise EvidenceContractError(
                "legacy record import ID does not match its source identity"
            )
        if (
            not isinstance(record.get("source_selector"), str)
            or not record["source_selector"]
        ):
            raise EvidenceContractError("legacy record requires a source selector")
        if not isinstance(record.get("facts"), dict) or not isinstance(
            record.get("unavailable"), dict
        ):
            raise EvidenceContractError(
                "legacy record facts and unavailable must be objects"
            )
        if any(
            not isinstance(reason, str) or not reason
            for reason in record["unavailable"].values()
        ):
            raise EvidenceContractError(
                "legacy unavailable fields require nonempty reasons"
            )
        source_declared_ids = record.get("source_declared_ids")
        if not isinstance(source_declared_ids, dict):
            raise EvidenceContractError("source-declared IDs must be an object")
        for value in source_declared_ids.values():
            if value is not None and not isinstance(value, str):
                raise EvidenceContractError(
                    "source-declared IDs must be strings or null"
                )
        source_record = record.get("source")
        if (
            not isinstance(source_record, dict)
            or source_record.get("digest_algorithm") != "sha256"
            or not isinstance(source_record.get("path"), str)
            or not isinstance(source_record.get("digest"), str)
        ):
            raise EvidenceContractError("legacy record has an invalid primary source")
        expected_binding = _source_binding(
            repo_root.resolve(),
            source_path=source_record["path"],
            source_selector=record["source_selector"],
            facts=record["facts"],
            inferred_linkage=record.get("inferred_linkage"),
            source_declared_ids=source_declared_ids,
            unavailable=record["unavailable"],
        )
        if record.get("source_binding") != expected_binding:
            raise EvidenceContractError("legacy source binding does not match record")
        semantic_key = _semantic_key(record)
        if semantic_key in semantic_keys:
            raise EvidenceContractError("legacy ledger has duplicate semantic evidence")
        semantic_keys.add(semantic_key)
        for source in [record.get("source", {}), *record.get("related_sources", [])]:
            if source.get("digest_algorithm") != "sha256":
                raise EvidenceContractError(
                    "legacy record source digest algorithm is invalid"
                )
            if source.get("path") not in sources or sources[
                source["path"]
            ] != source.get("digest"):
                raise EvidenceContractError(
                    f"legacy record does not resolve to a ledger source: {record.get('import_id')}"
                )
            record_source_paths.add(source["path"])
    if record_source_paths != set(sources):
        raise EvidenceContractError(
            "ledger sources do not exactly match record sources"
        )
    expected_subtotal = [
        record
        for record in records
        if record.get("inventory_scope") == "campaign_b_recap"
    ]
    subtotal = payload.get("inventory_subtotals", {}).get("campaign_b_recap", {})
    if subtotal.get("record_count") != len(expected_subtotal):
        raise EvidenceContractError("Campaign-B subtotal does not match records")
    expected_status_counts = {
        status: sum(record["record_status"] == status for record in expected_subtotal)
        for status in ("valid", "smoke", "invalid", "incomplete", "omitted")
    }
    if subtotal.get("status_counts") != expected_status_counts:
        raise EvidenceContractError("Campaign-B status subtotal does not match records")
    if payload.get("inventory_subtotals", {}).get("all_pre_b09_falcon", {}).get(
        "record_count"
    ) != len(records):
        raise EvidenceContractError("pre-B09 subtotal does not match records")
    _verify_artifact_index(repo_root.resolve(), payload.get("artifact_index"))
    artifact_index = payload["artifact_index"]
    derived_paths = [
        artifact["path"]
        for artifact in artifact_index
        if artifact["classification"] == "derived_artifact"
    ]
    if payload.get("excluded_derived_artifacts") != derived_paths:
        raise EvidenceContractError("derived artifact exclusion list is inconsistent")
    expected_associated = [
        artifact
        for artifact in artifact_index
        if artifact["classification"].startswith("associated_")
        or artifact["classification"].startswith("tooling_only_")
    ]
    if payload.get("associated_artifacts") != expected_associated:
        raise EvidenceContractError("associated artifact index is inconsistent")
    return len(records)


def _verify_artifact_index(repo_root: Path, artifacts: Any) -> None:
    if not isinstance(artifacts, list) or not artifacts:
        raise EvidenceContractError("artifact index must be a nonempty list")
    seen_paths = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not all(
            isinstance(artifact.get(field), str) and artifact[field]
            for field in ("path", "classification", "parent", "hash_policy")
        ):
            raise EvidenceContractError("artifact index entry is incomplete")
        path = artifact["path"]
        if path in seen_paths:
            raise EvidenceContractError("artifact index has duplicate paths")
        seen_paths.add(path)
        candidate = (repo_root / path).resolve()
        try:
            candidate.relative_to(repo_root)
        except ValueError as exc:
            raise EvidenceContractError(
                "artifact index path escapes repository"
            ) from exc
        if not candidate.exists():
            raise EvidenceContractError(f"artifact index path is missing: {path}")


def _verify_selector(repo_root: Path, source_path: str, selector: str) -> None:
    _select_source_value(repo_root, source_path, selector)


def _table_fixtures(
    source: Path, source_path: str, *, dry_run: bool
) -> list[LegacyFixture]:
    table = json.loads(source.read_text())
    # This helper only extracts facts. The caller replaces the absolute input
    # path with the repository-relative source path before hashing it.
    status = "smoke" if dry_run else "valid"
    fixtures: list[LegacyFixture] = []
    for row in table["arms"]:
        key = f"{'dry' if dry_run else 'full'}-lm-{row['arm']}-seed{row['seed']}"
        facts = {
            **row,
            "campaign_step": "step4",
            "source_tag": "dry_run" if dry_run else source_path,
            "lane": "science",
            "mode": "train",
            "claim_label": table["claim_label"],
            "inferred_linkage": {"arm": row["arm"], "seed": row["seed"], "kind": "lm"},
        }
        facts["source_declared_claim_label"] = facts["claim_label"]
        fixtures.append(
            LegacyFixture(
                key,
                source_path,
                "raw_ablation_table",
                status,
                facts,
                {},
                source_selector=f"arms[{row['arm']},{row['seed']}]",
            )
        )
    for key, row in table["addition"].items():
        facts = {
            **row,
            "campaign_step": "step4",
            "source_tag": "dry_run" if dry_run else source_path,
            "requested_steps": row["steps"],
            "lane": "science",
            "mode": "addition_train",
            "claim_label": table["claim_label"],
            "inferred_linkage": {
                "arm": row["arm"],
                "seed": row["seed"],
                "kind": "addition",
            },
        }
        facts["training_bank_exact_suffix_accuracy"] = facts.pop("id_acc")
        facts["ood_exact_suffix_accuracy"] = facts.pop("ood_acc")
        facts[
            "legacy_metric_correction"
        ] = "ID examples overlap the historical training bank; this is not held-out ID validation."
        facts["source_declared_claim_label"] = facts["claim_label"]
        if not dry_run:
            facts["claim_label"] = "representative_small"
        fixtures.append(
            LegacyFixture(
                f"{'dry' if dry_run else 'full'}-addition-{key}",
                source_path,
                "raw_ablation_table",
                status,
                facts,
                {},
                source_selector=f"addition[{key}]",
            )
        )
    return fixtures


def _known_legacy_fixtures(repo_root: Path) -> list[LegacyFixture]:
    full_path = "experiments/falcon/results/ablations/full_2seed_8000/table.json"
    a1_path = "experiments/falcon/results/ablations/full_A1_2seed_8000/table.json"
    dry_path = "experiments/falcon/results/ablations/dry_run/table.json"
    fixtures = _table_fixtures(repo_root / full_path, full_path, dry_run=False)
    fixtures += _table_fixtures(repo_root / a1_path, a1_path, dry_run=False)
    fixtures += _table_fixtures(repo_root / dry_path, dry_path, dry_run=True)
    for tier, status in (("hero_20k", "valid"), ("hero_dry", "smoke")):
        for arm in ("A0", "A1", "A5"):
            source_path = (
                f"experiments/falcon/results/hero/{tier}/{arm}_seed0/hero_outcome.json"
            )
            outcome = json.loads((repo_root / source_path).read_text())
            checkpoint_path = (
                f"experiments/falcon/results/hero/{tier}/{arm}_seed0/checkpoint/"
                f"step-{outcome['completed_steps']}"
            )
            fixtures.append(
                LegacyFixture(
                    key=f"{tier}-{arm}-seed0",
                    source_path=source_path,
                    artifact_role="raw_hero_outcome",
                    record_status=status,
                    facts={
                        **outcome,
                        "lane": "science",
                        "mode": "train",
                        "inferred_linkage": {"arm": arm, "seed": 0, "kind": "hero"},
                        "checkpoint_lineage": {"output_checkpoint": checkpoint_path},
                    },
                    unavailable={},
                    source_selector="hero_outcome",
                )
            )
    task07_report = ".scratch/falcon-fast-weight-attention/sdd/task-07-report.md"
    issue07 = ".scratch/falcon-fast-weight-attention/issues/07-step4-ablations.md"
    failed_tiny_base = (
        "experiments/falcon/results/tooling/tiny_inject_v3/run_evidence/"
        "f87875d8-73a1-4a71-be68-6a575d008bd4/"
        "5fc6fa7c-8b1f-4d17-9677-20b95b1830fb"
    )
    failed_tiny_outcome = json.loads(
        (
            repo_root
            / failed_tiny_base
            / "processes/trainer.core.global_rank_000000/outcome.json"
        ).read_text()
    )
    fixtures += [
        LegacyFixture(
            "invalid-collapsed-ablation-table",
            task07_report,
            "audit_record",
            "invalid",
            {
                "lane": "science",
                "mode": "train",
                "arm": "unknown",
                "inferred_linkage": {
                    "campaign": "B",
                    "description": "initial collapsed ablation table",
                },
            },
            {"claim_label": "audit says the table was invalidated"},
            source_selector="lines[35:49]",
        ),
        LegacyFixture(
            "incomplete-a3-8000",
            task07_report,
            "audit_record",
            "incomplete",
            {
                "lane": "science",
                "mode": "train",
                "arm": "A3",
                "seed": 0,
                "inferred_linkage": {
                    "arm": "A3",
                    "seed": 0,
                    "requested_steps": 8000,
                },
            },
            {"claim_label": "no completed outcome artifact"},
            source_selector="lines[161:168]",
        ),
        LegacyFixture(
            "incomplete-a1-seed1",
            task07_report,
            "audit_record",
            "incomplete",
            {
                "lane": "science",
                "mode": "train",
                "arm": "A1",
                "seed": 1,
                "inferred_linkage": {"arm": "A1", "seed": 1},
            },
            {"claim_label": "launch described without an outcome artifact"},
            source_selector="lines[156:160]",
        ),
        LegacyFixture(
            "failed-tiny-profiler-scheduler",
            f"{failed_tiny_base}/processes/trainer.core.global_rank_000000/outcome.json",
            "native_process_outcome",
            "invalid",
            {
                "lane": "tooling",
                "mode": "profile",
                "arm": "tiny_overfit",
                "run_id": "f87875d8-73a1-4a71-be68-6a575d008bd4",
                "attempt_id": "5fc6fa7c-8b1f-4d17-9677-20b95b1830fb",
                "source_manifest": json.loads(
                    (repo_root / failed_tiny_base / "manifest.json").read_text()
                ),
                "source_artifact_index": _read_json_lines(
                    repo_root
                    / failed_tiny_base
                    / "indexes/artifacts.trainer.core.global_rank_000000.jsonl"
                ),
                "source_process_outcome": failed_tiny_outcome,
                "processes": [failed_tiny_outcome],
                "outcome": failed_tiny_outcome["outcome"],
                "inferred_linkage": {
                    "workload": "tiny_overfit_30k_profiler",
                    "arm": "tiny_overfit",
                    "seed": 0,
                    "requested_steps": 30000,
                },
            },
            {"claim_label": "first profiler launch failed due to invalid scheduler"},
            source_selector="processes[trainer.core.global_rank_000000].outcome",
            related_source_paths=(
                f"{failed_tiny_base}/manifest.json",
                f"{failed_tiny_base}/indexes/artifacts.trainer.core.global_rank_000000.jsonl",
            ),
        ),
    ]
    for key, arm, seed, kind, selector in (
        ("omitted-a3-seed1-lm", "A3", 1, "lm", "lines[68:69]"),
        ("omitted-a1-seed0-addition", "A1", 0, "addition", "lines[70:71]"),
        ("omitted-a1-seed1-addition", "A1", 1, "addition", "lines[71:72]"),
        ("omitted-a3-seed0-addition", "A3", 0, "addition", "lines[72:73]"),
        ("omitted-a3-seed1-addition", "A3", 1, "addition", "lines[73:75]"),
    ):
        fixtures.append(
            LegacyFixture(
                key,
                issue07,
                "audit_record",
                "omitted",
                {
                    "lane": "science",
                    "mode": "train" if kind == "lm" else "addition_train",
                    "arm": arm,
                    "seed": seed,
                    "inferred_linkage": {"arm": arm, "seed": seed, "kind": kind},
                },
                {"claim_label": "audit records no completed outcome artifact"},
                source_selector=selector,
                related_source_paths=(task07_report,),
            )
        )
    learnability_source = (
        ".scratch/falcon-fast-weight-attention/issues/02-tiny-overfit.md"
    )
    for key, arm, variant, steps, outcome, selector in (
        ("step3-falcon1a-seed0", "A2", "falcon1a", 80, "passed", "lines[23:23]"),
        ("step3-falcon1-seed0", "A3", "falcon1", 80, "passed", "lines[24:24]"),
        (
            "step3-falcon1a-lr0-negative-control",
            "A2",
            "falcon1a",
            16,
            "expected_failure",
            "lines[25:25]",
        ),
    ):
        fixtures.append(
            LegacyFixture(
                key,
                learnability_source,
                "documented_learnability_outcome",
                "learnability_smoke",
                {
                    "lane": "learnability",
                    "mode": "overfit",
                    "arm": arm,
                    "variant": variant,
                    "seed": 0,
                    "requested_steps": steps,
                    "completed_steps": steps,
                    "claim_label": "learnability_smoke",
                    "inferred_linkage": {"workload": key, "outcome": outcome},
                },
                {},
                source_selector=selector,
                inventory_scope="pre_campaign_step3",
            )
        )
    for key, source_path, selector, facts in (
        (
            "task04-five-step-trainer-smoke",
            "experiments/falcon/results/task04_smoke_trainer.log",
            "lines[6:6]",
            {
                "lane": "science",
                "mode": "train",
                "arm": "falcon_science",
                "requested_steps": 5,
                "completed_steps": 5,
                "claim_label": "smoke",
            },
        ),
        (
            "task06-three-step-tiny-science-smoke",
            ".scratch/falcon-fast-weight-attention/sdd/task-06-report.md",
            "lines[91:106]",
            {
                "lane": "science",
                "mode": "train",
                "arm": "falcon_science_tiny",
                "requested_steps": 3,
                "completed_steps": 3,
                "claim_label": "smoke",
            },
        ),
        (
            "task06-two-step-full-shape-science-smoke",
            ".scratch/falcon-fast-weight-attention/sdd/task-06-report.md",
            "lines[107:109]",
            {
                "lane": "science",
                "mode": "train",
                "arm": "falcon_science_full_shape",
                "requested_steps": 2,
                "completed_steps": 2,
                "claim_label": "smoke",
            },
        ),
    ):
        fixtures.append(
            LegacyFixture(
                key,
                source_path,
                "documented_outcome",
                "smoke",
                {**facts, "inferred_linkage": {"workload": key}},
                {},
                source_selector=selector,
            )
        )

    profiler_sources = (
        ("profiler-v1-a5", "perf_base_A5_v1", 20000, 1140.13, ("console.log",)),
        ("profiler-v2-a5", "nsys_v2", 200, 14.56, ("console.log", "stats.txt")),
        ("profiler-v3-a5", "nsys_in_rootfs", 20, 4.59, ("console.log",)),
    )
    for key, directory, steps, elapsed_sec, text_artifacts in profiler_sources:
        base = f"experiments/falcon/results/tooling/{directory}"
        summary = json.loads((repo_root / base / "summary.json").read_text())
        manifest = json.loads((repo_root / base / "manifest.json").read_text())
        related_artifacts = {
            name: (repo_root / base / name).read_text() for name in text_artifacts
        }
        fixtures.append(
            LegacyFixture(
                key,
                f"{base}/summary.json",
                "profiler_summary",
                "smoke",
                {
                    **summary,
                    "lane": "tooling",
                    "mode": "profile",
                    "arm": "A5",
                    "requested_steps": summary.get("steps", steps),
                    "completed_steps": summary.get("completed_steps", steps),
                    "elapsed_sec": summary.get("elapsed_sec", elapsed_sec),
                    "source_declared_claim_label": summary.get("claim_label"),
                    "claim_label": "smoke",
                    "configuration": manifest,
                    "related_artifacts": related_artifacts,
                    "environment": {
                        "cuda_visible_devices": summary.get("cuda_visible_devices"),
                        "dtype": summary.get("dtype"),
                    },
                    "inferred_linkage": {"workload": key},
                },
                {},
                source_selector="summary",
                related_source_paths=(
                    f"{base}/manifest.json",
                    *(f"{base}/{name}" for name in text_artifacts),
                ),
            )
        )

    native_sources = (
        (
            "tiny-overfit-native-smoke",
            "experiments/falcon/results/tiny_overfit/run_evidence/"
            "b7c9f7cc-646c-4675-8374-0be587092cfa/"
            "2fba701e-f51e-4fa0-ab87-e35db9a5f25a",
            "b7c9f7cc-646c-4675-8374-0be587092cfa",
            "2fba701e-f51e-4fa0-ab87-e35db9a5f25a",
            1,
        ),
        (
            "tiny-overfit-30k-profiler",
            "experiments/falcon/results/tooling/tiny_inject_v3/run_evidence/"
            "2fcb43e2-b2f4-43b7-9595-267b30a04511/"
            "b0782e28-858b-4a3b-9085-99a20036b099",
            "2fcb43e2-b2f4-43b7-9595-267b30a04511",
            "b0782e28-858b-4a3b-9085-99a20036b099",
            30000,
        ),
    )
    for key, base, run_id, attempt_id, steps in native_sources:
        outcome = json.loads(
            (
                repo_root
                / base
                / "processes/trainer.core.global_rank_000000/outcome.json"
            ).read_text()
        )
        manifest = json.loads((repo_root / base / "manifest.json").read_text())
        artifact_index_path = (
            repo_root / base / "indexes/artifacts.trainer.core.global_rank_000000.jsonl"
        )
        artifact_index = _read_json_lines(artifact_index_path)
        linkage = {"workload": key}
        if key == "tiny-overfit-30k-profiler":
            linkage = {
                "workload": "tiny_overfit_30k_profiler",
                "arm": "tiny_overfit",
                "seed": 0,
                "requested_steps": 30000,
            }
        fixtures.append(
            LegacyFixture(
                key,
                f"{base}/processes/trainer.core.global_rank_000000/outcome.json",
                "native_process_outcome",
                "smoke",
                {
                    "source_manifest": manifest,
                    "source_process_outcome": outcome,
                    "source_artifact_index": artifact_index,
                    "lane": "tooling",
                    "mode": "train",
                    "arm": "tiny_overfit",
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "requested_steps": steps,
                    "completed_steps": steps,
                    "claim_label": "smoke",
                    "processes": [outcome],
                    "outcome": outcome["outcome"],
                    "runtime": manifest.get("runtime"),
                    "configuration": manifest.get("config", {}).get("normalized"),
                    "source_revision": manifest.get("source"),
                    "inferred_linkage": linkage,
                },
                {},
                source_selector="processes[trainer.core.global_rank_000000].outcome",
                related_source_paths=(
                    f"{base}/manifest.json",
                    f"{base}/indexes/artifacts.trainer.core.global_rank_000000.jsonl",
                ),
            )
        )
    return fixtures


def import_known_legacy_falcon_evidence(repo_root: Path, ledger_path: Path) -> Path:
    """Import the audited Campaign B fixture inventory into the canonical ledger."""
    repo_root = repo_root.resolve()
    return import_legacy_fixtures(
        repo_root,
        ledger_path,
        _known_legacy_fixtures(repo_root),
        include_known_artifacts=True,
        catalog_id=FALCON_PRE_B09_CATALOG_ID,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Falcon evidence ledger utility")
    parser.add_argument("command", choices=("import-legacy", "verify-ledger"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/falcon/results/evidence/legacy-ledger-v1.json"),
    )
    args = parser.parse_args()
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise SystemExit(
            "Falcon evidence commands must run inside scripts/rootfs/enter_rootfs.sh"
        )
    repo_root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else repo_root / args.output
    if args.command == "import-legacy":
        print(import_known_legacy_falcon_evidence(repo_root, output))
    else:
        print(verify_known_pre_b09_ledger(repo_root, output))


if __name__ == "__main__":
    _main()
