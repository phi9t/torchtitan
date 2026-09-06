#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check GPU memory readiness for the Mini Kimi K3 r1 Trainer smoke."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from experiments.mini_kimi_k3.r1_training_smoke import _gpu_memory_evidence


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record whether a GPU is ready for the Mini Kimi K3 r1 smoke."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/results/r1_smoke_gpu_readiness.json"),
    )
    parser.add_argument(
        "--min-free-gpu-memory-mib",
        type=int,
        default=12_000,
        help="Required free memory on at least one visible GPU.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = check_r1_smoke_gpu_readiness(
            report_path=args.report,
            min_free_gpu_memory_mib=args.min_free_gpu_memory_mib,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 21

    print(f"r1 smoke GPU readiness: {report['status']}", file=sys.stderr)
    print(f"wrote GPU readiness report: {args.report}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 21


def check_r1_smoke_gpu_readiness(
    *,
    report_path: Path,
    min_free_gpu_memory_mib: int,
) -> dict[str, Any]:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise ValueError(
            "Mini-K3 r1 smoke GPU readiness must run inside the TorchTitan rootfs"
        )
    if min_free_gpu_memory_mib < 0:
        raise ValueError(
            "min_free_gpu_memory_mib must be >= 0, got " f"{min_free_gpu_memory_mib}"
        )
    evidence = _gpu_memory_evidence(
        min_free_gpu_memory_mib=min_free_gpu_memory_mib,
        skip_gpu_memory_precheck=False,
        gpu_memory_probe=None,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_r1_smoke_gpu_readiness",
        **evidence,
        "rootfs": {
            "marker": os.environ.get("TORCHTITAN_IN_ROOTFS"),
            "cwd": str(Path.cwd()),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
