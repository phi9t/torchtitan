# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""External harness smoke metadata and ingestion helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HarnessPin:
    name: str
    repo: str
    revision: str
    package_module: str
    package_name: str
    package_version: str
    role: str

    def to_json(self) -> dict[str, object]:
        installed = importlib.util.find_spec(self.package_module) is not None
        installed_version = None
        if installed:
            try:
                installed_version = importlib.metadata.version(self.package_name)
            except importlib.metadata.PackageNotFoundError:
                installed_version = None
        return {
            "name": self.name,
            "repo": self.repo,
            "revision": self.revision,
            "package_module": self.package_module,
            "package_name": self.package_name,
            "package_version": self.package_version,
            "role": self.role,
            "installed": installed,
            "installed_version": installed_version,
        }


def default_harbor_terminal_pins() -> list[HarnessPin]:
    return [
        HarnessPin(
            name="harbor",
            repo="https://github.com/harbor-framework/harbor.git",
            revision="b7e2f71b4563618af3a42279740f5f412dcf7046",
            package_module="harbor",
            package_name="harbor",
            package_version="0.21.0",
            role="agent harness",
        ),
        HarnessPin(
            name="terminal-bench-2-1",
            repo="https://github.com/harbor-framework/terminal-bench-2-1.git",
            revision="7131e4375048a0e408a8fb404b5f499d726b695b",
            package_module="terminal_bench",
            package_name="terminal-bench",
            package_version="0.2.18",
            role="benchmark tasks and scoring",
        ),
    ]


def default_tau2_pins() -> list[HarnessPin]:
    return [
        HarnessPin(
            name="tau2-bench",
            repo="https://github.com/sierra-research/tau2-bench.git",
            revision="668d3bcd135c02aa3438f987ef45735b7c163ee3",
            package_module="tau2",
            package_name="tau2",
            package_version="2.3.3",
            role="benchmark tasks and scoring",
        )
    ]


def write_harness_smoke(
    *,
    output: Path,
    run_id: str,
    harness_family: str,
    pins: Sequence[HarnessPin],
    dry_run: bool,
    task_subset: str,
) -> dict[str, object]:
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "harness_family": harness_family,
        "mode": "dry_run" if dry_run else "installed_harness",
        "task_subset": task_subset,
        "rootfs": {
            "in_rootfs": os.environ.get("TORCHTITAN_IN_ROOTFS") == "1",
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "tools": {
            "git": _tool_version("git"),
            "docker": _tool_version("docker"),
            "bwrap": _tool_version("bwrap"),
        },
        "pins": [pin.to_json() for pin in pins],
        "raw_result": _raw_dry_run_result(harness_family, task_subset, pins),
    }
    write_json(output, record)
    return record


def write_installed_preflight(
    *,
    output: Path,
    run_id: str,
    harness_family: str,
    pins: Sequence[HarnessPin],
    task_subset: str,
    cli_names: Sequence[str],
) -> dict[str, object]:
    pin_rows = [pin.to_json() for pin in pins]
    cli_rows = [_cli_probe(name) for name in cli_names]
    all_imported = all(row["installed"] for row in pin_rows)
    all_versions_match = all(
        row["installed_version"] == row["package_version"] for row in pin_rows
    )
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "harness_family": harness_family,
        "mode": "installed_preflight",
        "task_subset": task_subset,
        "rootfs": {
            "in_rootfs": os.environ.get("TORCHTITAN_IN_ROOTFS") == "1",
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "tools": {
            "git": _tool_version("git"),
            "docker": _tool_version("docker"),
            "bwrap": _tool_version("bwrap"),
        },
        "pins": pin_rows,
        "cli": cli_rows,
        "raw_result": {
            "metric_name": "installed_preflight",
            "score": 1.0 if all_imported and all_versions_match else 0.0,
            "num_tasks": 0,
            "score_source": "package import and version preflight",
            "task_metadata": {
                "task_subset": task_subset,
                "pin_hash": _hash_lines([pin.revision for pin in pins]),
            },
            "trajectory": [
                {
                    "step": 0,
                    "actor": "torchtitan",
                    "event": "create_isolated_rootfs_virtualenv",
                    "harness_family": harness_family,
                },
                {
                    "step": 1,
                    "actor": "external_harness",
                    "event": "import_and_version_preflight",
                    "task_subset": task_subset,
                },
            ],
        },
    }
    write_json(output, record)
    return record


def ingest_harness_smoke(
    *,
    raw_result: Path,
    output: Path,
    results_root: Path,
) -> dict[str, object]:
    raw = json.loads(raw_result.read_text())
    checks = {
        "rootfs_selected": bool(raw.get("rootfs", {}).get("in_rootfs", False)),
        "pins_present": bool(raw.get("pins")),
        "raw_result_present": bool(raw.get("raw_result")),
        "dry_run_labeled": raw.get("mode") == "dry_run",
        "installed_preflight_labeled": raw.get("mode") == "installed_preflight",
        "all_imports_available": all(
            bool(pin.get("installed")) for pin in raw.get("pins", [])
        ),
        "all_versions_match": all(
            pin.get("installed_version") == pin.get("package_version")
            for pin in raw.get("pins", [])
        ),
    }
    score = raw["raw_result"]["score"]
    ingested = {
        "schema_version": 1,
        "run_id": raw["run_id"],
        "harness_family": raw["harness_family"],
        "mode": raw["mode"],
        "task_subset": raw["task_subset"],
        "artifacts": {
            "results_root": str(results_root),
            "raw_result": str(raw_result),
        },
        "pins": raw["pins"],
        "environment": {
            "rootfs": raw["rootfs"],
            "tools": raw["tools"],
        },
        "metric": {
            "name": raw["raw_result"]["metric_name"],
            "score": score,
            "num_tasks": raw["raw_result"]["num_tasks"],
            "score_source": raw["raw_result"]["score_source"],
        },
        "cli": raw.get("cli", []),
        "trajectory": raw["raw_result"]["trajectory"],
        "checks": checks,
        "limitations": _limitations_for_mode(raw.get("mode")),
    }
    write_json(output, ingested)
    return ingested


def build_report_input(
    *,
    results_root: Path,
    run_id: str,
    ingested_paths: dict[str, Path],
) -> dict[str, object]:
    ingested = {
        name: json.loads(path.read_text()) for name, path in ingested_paths.items()
    }
    installed_preflight_values = [
        value
        for value in ingested.values()
        if value["checks"]["installed_preflight_labeled"]
    ]
    checks = {
        "ingested_present": all(path.is_file() for path in ingested_paths.values()),
        "all_rootfs_selected": all(
            value["checks"]["rootfs_selected"] for value in ingested.values()
        ),
        "all_modes_labeled": all(
            value["checks"]["dry_run_labeled"]
            or value["checks"]["installed_preflight_labeled"]
            for value in ingested.values()
        ),
        "all_pins_present": all(
            value["checks"]["pins_present"] for value in ingested.values()
        ),
        "installed_preflight_imports_available": all(
            value["checks"]["all_imports_available"]
            for value in installed_preflight_values
        ),
        "installed_preflight_versions_match": all(
            value["checks"]["all_versions_match"] for value in installed_preflight_values
        ),
    }
    mode_counts: dict[str, int] = {}
    for value in ingested.values():
        mode = value["mode"]
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    scaffold_type = (
        "installed_preflight"
        if mode_counts.get("installed_preflight", 0) == len(ingested)
        else "dry_run_ingestion"
    )
    return {
        "schema_version": 1,
        "run": {
            "run_id": run_id,
            "task": "external_harness_smoke",
            "lane": "agentic_harness",
            "scaffold": {
                "type": scaffold_type,
                "budget": 0,
            },
        },
        "artifacts": {
            "results_root": str(results_root),
            "ingested": {name: str(path) for name, path in ingested_paths.items()},
        },
        "harnesses": ingested,
        "mode_counts": mode_counts,
        "checks": checks,
        "limitations": [
            "compatibility and artifact-ingestion evidence only",
            "no external benchmark task execution",
            "no model, adapter, or harness capability claim",
        ],
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _raw_dry_run_result(
    harness_family: str,
    task_subset: str,
    pins: Sequence[HarnessPin],
) -> dict[str, object]:
    trajectory = [
        {
            "step": 0,
            "actor": "torchtitan",
            "event": "declare_external_harness_boundary",
            "harness_family": harness_family,
        },
        {
            "step": 1,
            "actor": "external_harness",
            "event": "dry_run_score_fixture",
            "task_subset": task_subset,
        },
    ]
    return {
        "metric_name": "dry_run_compatibility",
        "score": 1.0,
        "num_tasks": 1,
        "score_source": "synthetic dry-run fixture",
        "task_metadata": {
            "task_subset": task_subset,
            "pin_hash": _hash_lines([pin.revision for pin in pins]),
        },
        "trajectory": trajectory,
    }


def _tool_version(name: str) -> str | None:
    path = shutil.which(name)
    if path is None:
        return None
    try:
        completed = subprocess.run(
            [path, "--version"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return path
    first_line = (completed.stdout or completed.stderr).splitlines()
    if not first_line:
        return path
    return first_line[0]


def _cli_probe(name: str) -> dict[str, object]:
    path = shutil.which(name)
    row: dict[str, object] = {"name": name, "path": path}
    if path is None:
        row["help_returncode"] = None
        row["help_first_line"] = None
        return row
    try:
        completed = subprocess.run(
            [path, "--help"],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:
        row["help_returncode"] = None
        row["help_first_line"] = str(exc)
        return row
    lines = (completed.stdout or completed.stderr).splitlines()
    row["help_returncode"] = completed.returncode
    row["help_first_line"] = lines[0] if lines else ""
    return row


def _limitations_for_mode(mode: object) -> list[str]:
    if mode == "installed_preflight":
        return [
            "package import and version preflight only",
            "external benchmark tasks were not executed",
            "not a model capability or benchmark score",
        ]
    return [
        "dry-run artifact only",
        "external harness packages were not executed",
        "not a model capability or benchmark score",
    ]


def _hash_lines(lines: Iterable[str]) -> str:
    hasher = hashlib.sha256()
    for line in lines:
        hasher.update(line.encode())
        hasher.update(b"\n")
    return hasher.hexdigest()
