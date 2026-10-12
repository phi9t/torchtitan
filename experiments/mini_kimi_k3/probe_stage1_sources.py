#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summarize first-party Mini Kimi K3 Stage 1 source resolution."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or summarize the first-party Mini Kimi K3 Stage 1 source probe."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara"),
        help="First-party Mini Kimi K3 source root containing data/probe.py.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/reports/stage1-source-resolution.json"),
        help="Stable summary report to write.",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="Do not run data/probe.py; summarize an existing data/resolved.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = probe_stage1_sources(
            source_root=args.source_root,
            report_path=args.report,
            run_probe=not args.skip_probe,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 21

    print(
        "stage1 source probe: "
        f"{report['status']} "
        f"({len(report['unresolved_benchmarks'])} unresolved benchmark(s), "
        f"{len(report['unresolved_corpora'])} unresolved corpus source(s))",
        file=sys.stderr,
    )
    return 0 if report["status"] == "ready" else 21


def probe_stage1_sources(
    *,
    source_root: Path,
    report_path: Path,
    run_probe: bool,
) -> dict[str, Any]:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise ValueError("Mini-K3 Stage 1 source probing must run inside rootfs")

    source_root = source_root.resolve()
    data_dir = source_root / "data"
    probe_py = data_dir / "probe.py"
    resolved_path = data_dir / "resolved.json"
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"first-party data directory does not exist: {data_dir}"
        )
    if run_probe:
        if not probe_py.is_file():
            raise FileNotFoundError(f"first-party probe does not exist: {probe_py}")
        proc = subprocess.run(
            [sys.executable, str(probe_py)],
            cwd=source_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        probe = {
            "ran": True,
            "return_code": proc.returncode,
            "stdout_tail": proc.stdout.splitlines()[-80:],
        }
    else:
        probe = {"ran": False, "return_code": None, "stdout_tail": []}

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"first-party resolved report does not exist: {resolved_path}"
        )
    resolved = json.loads(resolved_path.read_text())
    if not isinstance(resolved, list):
        raise ValueError("first-party resolved report must be a list")

    benchmarks: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    unresolved_benchmarks: list[str] = []
    unresolved_corpora: list[str] = []
    for row in resolved:
        if not isinstance(row, dict):
            raise ValueError("first-party resolved rows must be objects")
        kind = row.get("kind")
        name = row.get("name")
        if kind not in {"benchmark", "corpus"}:
            raise ValueError("first-party resolved rows must be benchmark or corpus")
        if not isinstance(name, str) or not name:
            raise ValueError("first-party resolved row name must be a nonempty string")
        chosen = row.get("chosen")
        item = _summarize_choice(row)
        if kind == "benchmark":
            benchmarks[name] = item
            if chosen is None:
                unresolved_benchmarks.append(name)
        else:
            sources[name] = item
            if chosen is None:
                unresolved_corpora.append(name)

    status = (
        "ready" if not unresolved_benchmarks and not unresolved_corpora else "blocked"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_stage1_source_resolution",
        "status": status,
        "source_root": str(source_root),
        "resolved_json": str(resolved_path),
        "probe": probe,
        "num_benchmarks": len(benchmarks),
        "num_corpora": len(sources),
        "unresolved_benchmarks": unresolved_benchmarks,
        "unresolved_corpora": unresolved_corpora,
        "benchmarks": benchmarks,
        "sources": sources,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _summarize_choice(row: dict[str, Any]) -> dict[str, Any]:
    chosen = row.get("chosen")
    tried = row.get("tried", [])
    if chosen is None:
        return {
            "status": "unresolved",
            "role": row.get("role"),
            "tried": tried,
        }
    if not isinstance(chosen, dict):
        raise ValueError("first-party chosen source must be an object or null")
    return {
        "status": "resolved",
        "repo": chosen.get("repo"),
        "config": chosen.get("config"),
        "split": chosen.get("split"),
        "text_field": chosen.get("text_field"),
        "path_prefix": chosen.get("path_prefix"),
        "filter_field": chosen.get("filter_field"),
        "filter_values": chosen.get("filter_values"),
        "n_files": chosen.get("n_files"),
        "role": row.get("role"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
