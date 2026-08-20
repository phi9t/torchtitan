#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run a bounded MLP backend diagnostic under the TorchTitan rootfs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.modded_nanogpt_b200 import cli_guard, preflight


SCHEMA_VERSION = 1


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _base_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "backend": args.backend,
        "source": str(args.source),
        "timeout_seconds": args.timeout_seconds,
    }


def _run_triton_mlp_smoke(source: Path, timeout_seconds: int) -> dict[str, Any]:
    return preflight._run_triton_mlp_smoke(source, timeout_seconds=timeout_seconds)


def _populate_rootfs_context(report: dict[str, Any]) -> None:
    preflight.check_rootfs()
    report["rootfs"] = preflight.collect_rootfs_detail()
    report["environment"] = preflight.collect_environment_detail()


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    report = _base_report(args)
    _populate_rootfs_context(report)
    if args.backend == "torch":
        report["detail"] = preflight._run_torch_mlp_smoke(args.source)
    elif args.backend == "triton":
        report["detail"] = _run_triton_mlp_smoke(args.source, args.timeout_seconds)
    else:
        raise preflight.CheckFailure(f"unsupported MLP backend: {args.backend}")
    report["ok"] = True
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=["torch", "triton"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args()


def main(*, enforce_rootfs: bool = False) -> int:
    args = parse_args()
    if enforce_rootfs:
        guard_exit = cli_guard.guard_rootfs_cli(
            "experiments/modded_nanogpt_b200/diagnose_mlp_backend.sh"
        )
        if guard_exit is not None:
            return guard_exit
    report = _base_report(args)
    try:
        _populate_rootfs_context(report)
        if args.backend == "torch":
            report["detail"] = preflight._run_torch_mlp_smoke(args.source)
        elif args.backend == "triton":
            report["detail"] = _run_triton_mlp_smoke(args.source, args.timeout_seconds)
        else:
            raise preflight.CheckFailure(f"unsupported MLP backend: {args.backend}")
        report["ok"] = True
    except preflight.CheckFailure as exc:
        report["error"] = str(exc)
        report["failure_class"] = (
            "rootfs"
            if "TORCHTITAN_IN_ROOTFS" in str(exc) or "rootfs" in str(exc)
            else "mlp_diagnostic"
        )
        if exc.detail:
            report["detail"] = exc.detail
            if exc.detail.get("failure_class"):
                report["failure_class"] = exc.detail["failure_class"]
            if exc.detail.get("stdout"):
                report["stdout"] = exc.detail["stdout"]
        _write_json_atomic(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 21
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["failure_class"] = "runtime_error"
        _write_json_atomic(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 21
    _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(enforce_rootfs=True))
