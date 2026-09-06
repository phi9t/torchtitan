#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check non-ingesting remote readiness for Mini Kimi K3 Stage 1."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from torchtitan.experiments.execution.executor import Executor, SubprocessExecutor

from .env_file import build_command_env, redact_sensitive_env_values


SCHEMA_VERSION = 1
ENV_FILE_TEMPLATE = ".scratch/mini-kimi-k3-replication/.env.example"
REQUIRED_ENV_KEYS = [
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "MODAL_PROFILE",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Mini Kimi K3 Stage 1 remote readiness without ingestion."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/reports/stage1-remote-readiness.json"),
        help="Machine-readable readiness report to write.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional KEY=VALUE file to pass to Modal readiness commands.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = check_stage1_remote_readiness(
            report_path=args.report,
            env_file=args.env_file,
        )
    except ValueError as exc:
        if args.env_file is not None:
            report = _blocked_env_file_report(
                report_path=args.report,
                env_file=args.env_file,
                detail=str(exc),
            )
            print(f"stage1 remote readiness: {report['status']}", file=sys.stderr)
            print(
                f"wrote Stage 1 remote readiness report: {args.report}", file=sys.stderr
            )
            return 21
        print(str(exc), file=sys.stderr)
        return 21

    print(f"stage1 remote readiness: {report['status']}", file=sys.stderr)
    print(f"wrote Stage 1 remote readiness report: {args.report}", file=sys.stderr)
    return 0 if report["status"] == "ready" else 21


def check_stage1_remote_readiness(
    *,
    report_path: Path,
    env_file: Path | None = None,
    executor: Executor | None = None,
) -> dict[str, Any]:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise ValueError("Mini-K3 Stage 1 remote readiness must run inside rootfs")
    executor = executor or SubprocessExecutor()
    command_env, env_file_report = build_command_env(env_file)
    rootfs = _rootfs_context()
    if env_file_report is not None:
        env_file_error = _env_file_error(command_env)
        if env_file_error is not None:
            return _blocked_env_file_report(
                report_path=report_path,
                env_file=env_file,
                detail=env_file_error,
            )
    checks = {
        "rootfs_network": _check_rootfs_network(rootfs),
        "huggingface_dns": _check_huggingface_dns(executor, env=command_env),
        "modal_cli": _check_modal_cli(executor, env=command_env),
        "modal_auth": _check_modal_auth(executor, env=command_env),
    }
    status = (
        "ready"
        if all(check["status"] == "pass" for check in checks.values())
        else "blocked"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_stage1_remote_readiness",
        "status": status,
        "checks": checks,
        "rootfs": rootfs,
    }
    if env_file_report is not None:
        report["env_file"] = env_file_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _env_file_error(env: dict[str, str]) -> str | None:
    missing = [key for key in REQUIRED_ENV_KEYS if not env.get(key)]
    placeholder = [
        key for key in REQUIRED_ENV_KEYS if env.get(key, "").startswith("replace-with-")
    ]
    details = []
    if missing:
        details.append("env file is missing required values: " + ", ".join(missing))
    if placeholder:
        details.append(
            "env file still contains placeholder values: " + ", ".join(placeholder)
        )
    return "; ".join(details) if details else None


def _blocked_env_file_report(
    *,
    report_path: Path,
    env_file: Path,
    detail: str,
) -> dict[str, Any]:
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_stage1_remote_readiness",
        "status": "blocked",
        "checks": {
            "env_file": {
                "status": "blocked",
                "path": str(env_file),
                "detail": detail,
                "template": ENV_FILE_TEMPLATE,
                "required_keys": REQUIRED_ENV_KEYS,
            }
        },
        "rootfs": _rootfs_context(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _rootfs_context() -> dict[str, str | None]:
    return {
        "marker": os.environ.get("TORCHTITAN_IN_ROOTFS"),
        "network_mode": os.environ.get("TORCHTITAN_ROOTFS_NETWORK", "offline"),
        "cwd": str(Path.cwd()),
    }


def _check_rootfs_network(rootfs: dict[str, str | None]) -> dict[str, str | None]:
    network_mode = rootfs.get("network_mode")
    if rootfs.get("marker") == "1" and network_mode == "networked":
        return {
            "status": "pass",
            "detail": "rootfs network mode is enabled",
            "network_mode": network_mode,
        }
    return {
        "status": "blocked",
        "detail": "Stage 1 remote readiness requires networked rootfs",
        "network_mode": network_mode,
    }


def _check_huggingface_dns(
    executor: Executor, *, env: dict[str, str]
) -> dict[str, Any]:
    result = executor.run(["getent", "hosts", "huggingface.co"], env=env)
    if result.return_code == 0 and result.stdout.strip():
        return {"status": "pass", "detail": "huggingface.co resolves"}
    return {
        "status": "blocked",
        "detail": "huggingface.co does not resolve inside rootfs",
        "return_code": result.return_code,
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
    }


def _check_modal_cli(executor: Executor, *, env: dict[str, str]) -> dict[str, Any]:
    result = executor.run(
        ["uv", "tool", "run", "--from", "modal", "modal", "--version"],
        env=env,
    )
    text = "\n".join(part for part in (result.stdout, result.stderr) if part)
    version_match = re.search(r"modal client version:\s*([^\s]+)", text)
    if result.return_code == 0 and version_match is not None:
        return {
            "status": "pass",
            "detail": "modal CLI is available through uv tool",
            "version": version_match.group(1),
        }
    return {
        "status": "blocked",
        "detail": "modal CLI is not available through uv tool",
        "return_code": result.return_code,
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
    }


def _check_modal_auth(executor: Executor, *, env: dict[str, str]) -> dict[str, Any]:
    result = executor.run(
        ["uv", "tool", "run", "--from", "modal", "modal", "token", "info"],
        env=env,
    )
    text = "\n".join(part for part in (result.stdout, result.stderr) if part)
    auth_context = _modal_auth_context(env)
    if result.return_code == 0:
        return {
            "status": "pass",
            "detail": "modal token is configured",
            **auth_context,
            "stdout_tail": _redacted_tail(result.stdout, env=env),
            "stderr_tail": _redacted_tail(result.stderr, env=env),
        }
    if "Token missing" in text or "Could not authenticate client" in text:
        detail = "Token missing"
    else:
        detail = "modal token info failed"
    return {
        "status": "blocked",
        "detail": detail,
        "return_code": result.return_code,
        **auth_context,
        "stdout_tail": _redacted_tail(result.stdout, env=env),
        "stderr_tail": _redacted_tail(result.stderr, env=env),
    }


def _modal_auth_context(env: dict[str, str] | None = None) -> dict[str, Any]:
    env = env or os.environ
    home = Path.home()
    xdg_config_home = Path(env.get("XDG_CONFIG_HOME", home / ".config"))
    config_locations = []
    modal_config_path = env.get("MODAL_CONFIG_PATH")
    if modal_config_path:
        config_locations.append(
            {
                "path": modal_config_path,
                "exists": Path(modal_config_path).is_file(),
                "source": "MODAL_CONFIG_PATH",
            }
        )
    config_locations.extend(
        {"path": str(path), "exists": path.is_file()}
        for path in (
            home / ".modal.toml",
            home / ".modal" / "config.toml",
            xdg_config_home / "modal.toml",
            xdg_config_home / "modal" / "config.toml",
        )
    )
    return {
        "env": {
            name: "set" if env.get(name) else "unset"
            for name in (
                "MODAL_CONFIG_PATH",
                "MODAL_PROFILE",
                "MODAL_TOKEN_ID",
                "MODAL_TOKEN_SECRET",
            )
        },
        "config_locations": config_locations,
    }


def _tail(text: str, *, limit: int = 20) -> list[str]:
    return text.splitlines()[-limit:]


def _redacted_tail(
    text: str,
    *,
    env: dict[str, str] | None = None,
    limit: int = 20,
) -> list[str]:
    redacted = []
    for line in _tail(text, limit=limit):
        if env is not None:
            line = redact_sensitive_env_values(line, env)
        if re.search(r"(token|secret|key)\s*[:=]", line, flags=re.IGNORECASE):
            redacted.append(
                re.sub(
                    r"(?i)((?:token|secret|key)\s*[:=]\s*)\S+",
                    r"\1<redacted>",
                    line,
                )
            )
        else:
            redacted.append(line)
    return redacted


if __name__ == "__main__":
    raise SystemExit(main())
