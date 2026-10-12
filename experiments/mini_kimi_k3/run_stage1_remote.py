#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run guarded first-party Modal Stage 1 commands for Mini Kimi K3."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from torchtitan.experiments.execution.executor import (
    CommandResult,
    Executor,
    SubprocessExecutor,
)

from .env_file import build_command_env, redact_sensitive_env_values


SCHEMA_VERSION = 1
SAFE_STEPS = {
    "probe": ["modal", "run", "data/stage1.py::probe"],
    "gate_tokenizer": ["modal", "run", "data/stage1.py::gate_tokenizer"],
    "build_index": ["modal", "run", "data/stage1.py::build_index"],
    "dry_run": ["modal", "run", "data/stage1.py::dry_run"],
    "report": ["modal", "run", "data/stage1.py::report"],
}
SPENDFUL_STEPS = {
    "ingest_all": ["modal", "run", "--detach", "data/stage1.py::ingest_all"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a guarded first-party Mini Kimi K3 Stage 1 Modal step."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara"),
        help="First-party Mini Kimi K3 source root containing data/stage1.py.",
    )
    parser.add_argument(
        "--step",
        choices=tuple(SAFE_STEPS) + tuple(SPENDFUL_STEPS),
        required=True,
        help="First-party Stage 1 Modal entrypoint to run.",
    )
    parser.add_argument(
        "--readiness-report",
        type=Path,
        required=True,
        help="Ready report from stage1-remote-readiness.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/reports/stage1-remote-step.json"),
        help="Machine-readable Stage 1 step report to write.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional KEY=VALUE file to pass to the Modal command.",
    )
    parser.add_argument(
        "--allow-spend",
        action="store_true",
        help="Required for steps that can submit spendful remote work.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_stage1_remote(
            source_root=args.source_root,
            step=args.step,
            readiness_report=args.readiness_report,
            report_path=args.report,
            allow_spend=args.allow_spend,
            env_file=args.env_file,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 21

    print(
        f"Mini Kimi K3 Stage 1 remote step {args.step}: {report['status']}",
        file=sys.stderr,
    )
    print(f"wrote Stage 1 remote step report: {args.report}", file=sys.stderr)
    return 0 if report["status"] == "completed" else 21


def run_stage1_remote(
    *,
    source_root: Path,
    step: str,
    readiness_report: Path,
    report_path: Path,
    allow_spend: bool = False,
    env_file: Path | None = None,
    executor: Executor | None = None,
) -> dict[str, Any]:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise ValueError("Mini-K3 Stage 1 remote runner must run inside rootfs")
    if step in SPENDFUL_STEPS and not allow_spend:
        return _write_report(
            report_path,
            source_root=source_root,
            step=step,
            argv=[],
            status="blocked",
            detail=f"{step} requires --allow-spend",
        )
    try:
        command_env, env_file_report = build_command_env(env_file)
    except ValueError as exc:
        return _write_report(
            report_path,
            source_root=source_root,
            step=step,
            argv=[],
            status="blocked",
            detail=str(exc),
            env_file=_blocked_env_file_evidence(env_file=env_file, detail=str(exc)),
        )
    stage1_py = source_root / "data" / "stage1.py"
    if not stage1_py.is_file():
        return _write_report(
            report_path,
            source_root=source_root,
            step=step,
            argv=[],
            status="blocked",
            detail=f"missing first-party Stage 1 file: {stage1_py}",
            env_file=env_file_report,
        )
    readiness, readiness_error = _readiness_evidence(readiness_report)
    if readiness_error is not None:
        return _write_report(
            report_path,
            source_root=source_root,
            step=step,
            argv=[],
            status="blocked",
            detail=readiness_error,
            readiness=readiness,
            env_file=env_file_report,
        )
    env_file_mismatch = _readiness_env_file_mismatch_detail(
        readiness=readiness,
        env_file=env_file,
    )
    if env_file_mismatch is not None:
        return _write_report(
            report_path,
            source_root=source_root,
            step=step,
            argv=[],
            status="blocked",
            detail=env_file_mismatch,
            readiness=readiness,
            env_file=env_file_report,
        )

    modal_args = SAFE_STEPS.get(step) or SPENDFUL_STEPS.get(step)
    if modal_args is None:
        raise ValueError(f"unsupported Stage 1 remote step: {step}")
    argv = ["uv", "tool", "run", "--from", "modal", *modal_args]
    executor = executor or SubprocessExecutor()
    result = executor.run(argv, cwd=source_root, env=command_env)
    status = "completed" if result.return_code == 0 else "failed"
    return _write_report(
        report_path,
        source_root=source_root,
        step=step,
        argv=argv,
        status=status,
        detail="Stage 1 remote step finished"
        if status == "completed"
        else "Stage 1 remote step failed",
        readiness=readiness,
        env_file=env_file_report,
        command_env=command_env,
        command=result,
    )


def _readiness_evidence(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        report = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"report": str(path), "status": "unreadable"}, str(exc)

    evidence = {
        "report": str(path),
        "kind": report.get("kind"),
        "status": report.get("status"),
    }
    if report.get("kind") != "mini_kimi_k3_stage1_remote_readiness":
        return (
            evidence,
            "readiness report kind must be mini_kimi_k3_stage1_remote_readiness",
        )
    if report.get("schema_version") != 1:
        return evidence, "readiness report schema_version must be 1"
    rootfs = report.get("rootfs")
    if not isinstance(rootfs, dict) or rootfs.get("marker") != "1":
        return evidence, "readiness report rootfs marker must be 1"
    if rootfs.get("network_mode") != "networked":
        return evidence, "readiness report network_mode must be networked"
    checks = report.get("checks")
    if not isinstance(checks, dict):
        return evidence, "readiness report checks must be an object"
    rootfs_network_check = checks.get("rootfs_network")
    if not isinstance(rootfs_network_check, dict):
        return evidence, "readiness report missing rootfs_network check"
    if rootfs_network_check.get("status") != "pass":
        return (
            evidence,
            "readiness report must be ready "
            f"(rootfs_network={rootfs_network_check.get('status')})",
        )
    env_file_check = checks.get("env_file")
    if isinstance(env_file_check, dict) and env_file_check.get("status") != "pass":
        evidence["env_file"] = env_file_check
        return (
            evidence,
            f"readiness env file is blocked ({env_file_check.get('detail')})",
        )
    if isinstance(report.get("env_file"), dict):
        evidence["env_file"] = report["env_file"]
    for check_name in ("huggingface_dns", "modal_cli", "modal_auth"):
        check = checks.get(check_name)
        if not isinstance(check, dict):
            return evidence, f"readiness report missing {check_name} check"
        if check.get("status") != "pass":
            return (
                evidence,
                "readiness report must be ready "
                f"({check_name}={check.get('status')})",
            )
    if report.get("status") != "ready":
        return evidence, "readiness report must be ready before remote Stage 1"
    return evidence, None


def _readiness_env_file_mismatch_detail(
    *,
    readiness: dict[str, Any],
    env_file: Path | None,
) -> str | None:
    readiness_env_file = readiness.get("env_file")
    if not isinstance(readiness_env_file, dict):
        return None
    readiness_path = readiness_env_file.get("path")
    if readiness_path is None:
        return None
    if env_file is None:
        return (
            "readiness env file was checked but no env file was requested: "
            f"{readiness_path}"
        )
    requested_path = str(env_file)
    if readiness_path != requested_path:
        return (
            "readiness env file does not match requested env file: "
            f"{readiness_path} != {requested_path}"
        )
    return None


def _blocked_env_file_evidence(
    *,
    env_file: Path | None,
    detail: str,
) -> dict[str, Any]:
    evidence = {
        "status": "blocked",
        "detail": detail,
    }
    if env_file is not None:
        evidence["path"] = str(env_file)
    return evidence


def _write_report(
    path: Path,
    *,
    source_root: Path,
    step: str,
    argv: list[str],
    status: str,
    detail: str,
    readiness: dict[str, Any] | None = None,
    env_file: dict[str, Any] | None = None,
    command_env: dict[str, str] | None = None,
    command: CommandResult | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_stage1_remote_step",
        "status": status,
        "detail": detail,
        "step": step,
        "source_root": str(source_root),
        "argv": argv,
        "rootfs": {
            "marker": os.environ.get("TORCHTITAN_IN_ROOTFS"),
            "network_mode": os.environ.get("TORCHTITAN_ROOTFS_NETWORK", "offline"),
            "cwd": str(Path.cwd()),
        },
    }
    if readiness is not None:
        report["readiness"] = readiness
    if env_file is not None:
        report["env_file"] = env_file
    if command is not None:
        report["command"] = {
            "return_code": command.return_code,
            "stdout_tail": _tail(command.stdout, env=command_env),
            "stderr_tail": _tail(command.stderr, env=command_env),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _tail(
    text: str,
    *,
    env: dict[str, str] | None = None,
    limit: int = 20,
) -> list[str]:
    lines = text.splitlines()[-limit:]
    if env is None:
        return lines
    return [redact_sensitive_env_values(line, env) for line in lines]


if __name__ == "__main__":
    raise SystemExit(main())
