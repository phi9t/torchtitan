# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attempt-local runtime verification for the modded-nanogpt B200 program."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.modded_nanogpt_b200.runtime import schema_validation


SCHEMA_VERSION = 1
EXPECTED_REQUIRED_ENV = {
    "TORCHTITAN_ROOTFS_PROJECT": (
        "/workspace/torchtitan",
        "rootfs_project",
        "command environment has unexpected rootfs project",
    ),
    "TORCHTITAN_ROOTFS_NETWORK": (
        "offline",
        "rootfs_network",
        "command environment has unexpected rootfs network mode",
    ),
    "PYTHON": (
        "/project/venvs/b200-runtime/bin/python",
        "python_path",
        "command environment has unexpected Python path",
    ),
}


def verify_attempt_runtime(
    *,
    command_env_path: Path,
    report_path: Path,
    require_training_launch_allowed: bool,
) -> dict[str, Any]:
    """Write and return the attempt-local runtime verification report."""

    blockers: list[dict[str, str]] = []
    try:
        command_env_record = _load_command_env(command_env_path)
    except (
        json.JSONDecodeError,
        OSError,
        schema_validation.SchemaValidationError,
    ) as exc:
        command_env_record = _load_partial_command_env(command_env_path)
        blockers.append(
            _blocker(
                phase="command_env",
                message=str(exc),
                expected="schema-valid command_env",
                actual="invalid",
                artifact_path=command_env_path,
            )
        )

    environment = command_env_record.get("environment", {})
    if not isinstance(environment, dict):
        environment = {}
    _check_rootfs_sentinel(
        environment=environment,
        command_env_path=command_env_path,
        blockers=blockers,
    )
    _check_required_env_invariants(
        environment=environment,
        command_env_path=command_env_path,
        blockers=blockers,
    )

    training_launch_allowed = not blockers and require_training_launch_allowed
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "runtime_verification",
        "run_id": str(command_env_record.get("run_id", "unknown")),
        "attempt_id": str(command_env_record.get("attempt_id", "unknown")),
        "ok": not blockers,
        "training_launch_allowed": training_launch_allowed,
        "command_env_path": str(command_env_path),
        "command_env_digest": command_env_record.get(
            "environment_digest", {"algorithm": "sha256", "sha256": "unknown"}
        ),
        "blockers": blockers,
    }
    schema_validation.validate_and_write(report_path, "runtime_verification", report)
    return report


def _load_command_env(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise schema_validation.SchemaValidationError("command_env must be an object")
    return schema_validation.validate("command_env", payload)


def _load_partial_command_env(path: Path) -> dict[str, Any]:
    fallback = {
        "run_id": "unknown",
        "attempt_id": "unknown",
        "environment_digest": {"algorithm": "sha256", "sha256": "unknown"},
    }
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return fallback
    if not isinstance(payload, dict):
        return fallback
    return {**fallback, **payload}


def _check_rootfs_sentinel(
    *,
    environment: dict[str, Any],
    command_env_path: Path,
    blockers: list[dict[str, str]],
) -> None:
    actual = environment.get("TORCHTITAN_IN_ROOTFS")
    if actual is None:
        blockers.append(
            _blocker(
                phase="rootfs_sentinel",
                message="command environment is missing the rootfs sentinel",
                expected="1",
                actual="missing",
                artifact_path=command_env_path,
            )
        )
    elif actual != "1":
        blockers.append(
            _blocker(
                phase="rootfs_sentinel",
                message="command environment is not marked as running inside rootfs",
                expected="1",
                actual=str(actual),
                artifact_path=command_env_path,
            )
        )


def _check_required_env_invariants(
    *,
    environment: dict[str, Any],
    command_env_path: Path,
    blockers: list[dict[str, str]],
) -> None:
    for key, (expected, phase, message) in EXPECTED_REQUIRED_ENV.items():
        actual = environment.get(key)
        if actual is None:
            blockers.append(
                _blocker(
                    phase=phase,
                    message=f"command environment is missing {key}",
                    expected=expected,
                    actual="missing",
                    artifact_path=command_env_path,
                )
            )
        elif actual != expected:
            blockers.append(
                _blocker(
                    phase=phase,
                    message=message,
                    expected=expected,
                    actual=str(actual),
                    artifact_path=command_env_path,
                )
            )


def _blocker(
    *,
    phase: str,
    message: str,
    expected: str,
    actual: str,
    artifact_path: Path,
) -> dict[str, str]:
    return {
        "phase": phase,
        "message": message,
        "expected": expected,
        "actual": actual,
        "artifact_path": str(artifact_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a modded-nanogpt B200 attempt runtime."
    )
    parser.add_argument("--command-env", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--require-training-launch-allowed",
        action="store_true",
        help="Require the report to certify a training launch path.",
    )
    args = parser.parse_args(argv)

    report = verify_attempt_runtime(
        command_env_path=args.command_env,
        report_path=args.report,
        require_training_launch_allowed=args.require_training_launch_allowed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 21


if __name__ == "__main__":
    raise SystemExit(main())
