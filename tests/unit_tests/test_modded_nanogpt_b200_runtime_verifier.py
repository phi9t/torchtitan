# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path

from experiments.modded_nanogpt_b200.runtime import verify_runtime


def _command_env_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "command_environment",
        "run_id": "run",
        "attempt_id": "run_attempt_001",
        "environment": {
            "TORCHTITAN_IN_ROOTFS": "1",
            "TORCHTITAN_ROOTFS_PROJECT": "/workspace/torchtitan",
            "TORCHTITAN_ROOTFS_NETWORK": "offline",
            "PYTHON": "/project/venvs/b200-runtime/bin/python",
        },
        "environment_digest": {"algorithm": "sha256", "sha256": "0" * 64},
    }


def test_verify_runtime_writes_ok_report_with_matching_command_env_digest(
    tmp_path: Path,
):
    command_env_path = tmp_path / "command.env.json"
    command_env_path.write_text(json.dumps(_command_env_record()) + "\n")
    report_path = tmp_path / "runtime" / "runtime_verification.json"

    report = verify_runtime.verify_attempt_runtime(
        command_env_path=command_env_path,
        report_path=report_path,
        require_training_launch_allowed=False,
    )

    assert report["ok"] is True
    assert report["run_id"] == "run"
    assert report["attempt_id"] == "run_attempt_001"
    assert report["training_launch_allowed"] is False
    assert report["command_env_digest"] == {"algorithm": "sha256", "sha256": "0" * 64}
    assert json.loads(report_path.read_text()) == report


def test_verify_runtime_rejects_missing_rootfs_sentinel(tmp_path: Path):
    command_env_path = tmp_path / "command.env.json"
    command_env_path.write_text(
        json.dumps(
            {
                **_command_env_record(),
                "environment": {
                    "TORCHTITAN_ROOTFS_PROJECT": "/workspace/torchtitan",
                    "TORCHTITAN_ROOTFS_NETWORK": "offline",
                    "PYTHON": "/project/venvs/b200-runtime/bin/python",
                },
            }
        )
        + "\n"
    )
    report_path = tmp_path / "runtime" / "runtime_verification.json"

    report = verify_runtime.verify_attempt_runtime(
        command_env_path=command_env_path,
        report_path=report_path,
        require_training_launch_allowed=True,
    )

    assert report["ok"] is False
    assert report["training_launch_allowed"] is False
    assert report["run_id"] == "run"
    assert report["attempt_id"] == "run_attempt_001"
    assert report["blockers"][0]["phase"] == "command_env"
    assert (
        "missing required field TORCHTITAN_IN_ROOTFS"
        in report["blockers"][0]["message"]
    )
    assert report["blockers"][1:] == [
        {
            "phase": "rootfs_sentinel",
            "message": "command environment is missing the rootfs sentinel",
            "expected": "1",
            "actual": "missing",
            "artifact_path": str(command_env_path),
        }
    ]


def test_verify_runtime_preserves_partial_failure_report(tmp_path: Path):
    command_env_path = tmp_path / "command.env.json"
    command_env_path.write_text(
        json.dumps(
            {
                **_command_env_record(),
                "environment": {"TORCHTITAN_IN_ROOTFS": "0"},
            }
        )
        + "\n"
    )
    report_path = tmp_path / "runtime" / "runtime_verification.json"

    report = verify_runtime.verify_attempt_runtime(
        command_env_path=command_env_path,
        report_path=report_path,
        require_training_launch_allowed=False,
    )

    assert report["ok"] is False
    assert report["run_id"] == "run"
    assert report["attempt_id"] == "run_attempt_001"
    assert report["command_env_digest"] == {"algorithm": "sha256", "sha256": "0" * 64}
    assert report["blockers"][0]["phase"] == "command_env"
    assert report["blockers"][0]["message"].startswith("command_env: ")
    assert report["blockers"][1:] == [
        {
            "phase": "rootfs_sentinel",
            "message": "command environment is not marked as running inside rootfs",
            "expected": "1",
            "actual": "0",
            "artifact_path": str(command_env_path),
        },
        {
            "phase": "rootfs_project",
            "message": "command environment is missing TORCHTITAN_ROOTFS_PROJECT",
            "expected": "/workspace/torchtitan",
            "actual": "missing",
            "artifact_path": str(command_env_path),
        },
        {
            "phase": "rootfs_network",
            "message": "command environment is missing TORCHTITAN_ROOTFS_NETWORK",
            "expected": "offline",
            "actual": "missing",
            "artifact_path": str(command_env_path),
        },
        {
            "phase": "python_path",
            "message": "command environment is missing PYTHON",
            "expected": "/project/venvs/b200-runtime/bin/python",
            "actual": "missing",
            "artifact_path": str(command_env_path),
        },
    ]
    assert json.loads(report_path.read_text()) == report


def test_verify_runtime_rejects_noncanonical_project_path(tmp_path: Path):
    command_env_path = tmp_path / "command.env.json"
    command_env_path.write_text(
        json.dumps(
            {
                **_command_env_record(),
                "environment": {
                    "TORCHTITAN_IN_ROOTFS": "1",
                    "TORCHTITAN_ROOTFS_PROJECT": "/tmp/torchtitan",
                    "TORCHTITAN_ROOTFS_NETWORK": "offline",
                    "PYTHON": "/project/venvs/b200-runtime/bin/python",
                },
            }
        )
        + "\n"
    )
    report_path = tmp_path / "runtime" / "runtime_verification.json"

    report = verify_runtime.verify_attempt_runtime(
        command_env_path=command_env_path,
        report_path=report_path,
        require_training_launch_allowed=True,
    )

    assert report["ok"] is False
    assert report["training_launch_allowed"] is False
    assert report["run_id"] == "run"
    assert report["attempt_id"] == "run_attempt_001"
    assert report["blockers"][0]["phase"] == "command_env"
    assert "TORCHTITAN_ROOTFS_PROJECT expected" in report["blockers"][0]["message"]
    assert report["blockers"][1:] == [
        {
            "phase": "rootfs_project",
            "message": "command environment has unexpected rootfs project",
            "expected": "/workspace/torchtitan",
            "actual": "/tmp/torchtitan",
            "artifact_path": str(command_env_path),
        }
    ]
