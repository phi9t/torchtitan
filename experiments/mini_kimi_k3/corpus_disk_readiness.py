#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check filesystem capacity for Mini Kimi K3 r1 raw token shards."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
UINT32_BYTES = 4
DEFAULT_OVERHEAD_FRACTION = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check free disk capacity before preparing Mini Kimi K3 r1 shards."
    )
    parser.add_argument(
        "--corpus-plan",
        type=Path,
        default=Path("experiments/mini_kimi_k3/reports/r1-corpus-plan.json"),
        help="Report written by plan-r1-corpus.",
    )
    parser.add_argument(
        "--target-path",
        type=Path,
        default=Path("experiments/mini_kimi_k3/data/tokens"),
        help="Directory or filesystem path that will hold raw uint32 token shards.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/reports/corpus-disk-readiness.json"),
        help="Disk readiness report to write.",
    )
    parser.add_argument(
        "--overhead-fraction",
        type=float,
        default=DEFAULT_OVERHEAD_FRACTION,
        help="Extra capacity fraction for sidecars, manifests, temp files, and slack.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = check_corpus_disk_readiness(
            corpus_plan_path=args.corpus_plan,
            target_path=args.target_path,
            report_path=args.report,
            overhead_fraction=args.overhead_fraction,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 21

    print(
        "corpus disk readiness: "
        f"{report['status']} "
        f"({report['free_bytes']}/{report['required_bytes']} bytes free, "
        f"deficit {report['deficit_bytes']})",
        file=sys.stderr,
    )
    return 0 if report["status"] == "ready" else 21


def check_corpus_disk_readiness(
    *,
    corpus_plan_path: Path,
    target_path: Path,
    report_path: Path,
    overhead_fraction: float = DEFAULT_OVERHEAD_FRACTION,
    disk_usage_fn: Callable[[Path], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise ValueError(
            "Mini-K3 corpus disk readiness must run inside the TorchTitan rootfs"
        )
    if overhead_fraction < 0 or not math.isfinite(overhead_fraction):
        raise ValueError("overhead-fraction must be a finite nonnegative number")
    if not corpus_plan_path.is_file():
        raise FileNotFoundError(f"r1 corpus plan does not exist: {corpus_plan_path}")

    plan = json.loads(corpus_plan_path.read_text())
    _validate_plan(plan)
    target_tokens = _nonnegative_int(plan, "target_tokens")
    raw_bytes = target_tokens * UINT32_BYTES
    required_bytes = int(math.ceil(raw_bytes * (1.0 + overhead_fraction)))
    measured_path = _nearest_existing_path(target_path)
    usage = disk_usage_fn(measured_path)
    free_bytes = int(usage.free)
    deficit_bytes = max(0, required_bytes - free_bytes)
    status = "ready" if deficit_bytes == 0 else "blocked"

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_corpus_disk_readiness",
        "status": status,
        "path": str(target_path),
        "measured_path": str(measured_path),
        "target_tokens": target_tokens,
        "uint32_bytes_per_token": UINT32_BYTES,
        "required_raw_bytes": raw_bytes,
        "overhead_fraction": overhead_fraction,
        "required_bytes": required_bytes,
        "free_bytes": free_bytes,
        "deficit_bytes": deficit_bytes,
        "disk_usage": {
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": free_bytes,
        },
        "corpus_plan": {
            "path": str(corpus_plan_path),
            "status": plan["status"],
            "target_tokens": target_tokens,
            "total_planned_tokens": _nonnegative_int(plan, "total_planned_tokens"),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("kind") != "mini_kimi_k3_r1_corpus_plan":
        raise ValueError("corpus plan kind must be mini_kimi_k3_r1_corpus_plan")
    if plan.get("schema_version") != 1:
        raise ValueError("corpus plan schema_version must be 1")
    if plan.get("status") != "planned":
        raise ValueError("corpus plan status must be planned")
    target_tokens = _nonnegative_int(plan, "target_tokens")
    total_planned_tokens = _nonnegative_int(plan, "total_planned_tokens")
    if target_tokens != total_planned_tokens:
        raise ValueError(
            "corpus plan target_tokens must match total_planned_tokens: "
            f"{target_tokens} != {total_planned_tokens}"
        )


def _nonnegative_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"corpus plan {key} must be a nonnegative integer")
    return value


def _nearest_existing_path(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(f"no existing filesystem parent for: {path}")
        candidate = parent
    return candidate


if __name__ == "__main__":
    raise SystemExit(main())
