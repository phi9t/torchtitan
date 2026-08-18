# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host-side tests for TorchTitan bwrap plan emission and verification.

These tests do not enter bwrap. They pin the pre-launch contract artifact that
`scripts/rootfs/enter_rootfs.sh` must emit before a payload can be trusted.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.rootfs.verify_runtime_env import (
    BwrapPlanError,
    validate_bwrap_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ENTER_ROOTFS = REPO_ROOT / "scripts" / "rootfs" / "enter_rootfs.sh"


def _minimal_rootfs(tmp_path: Path) -> Path:
    rootfs = tmp_path / "rootfs"
    bash = rootfs / "bin" / "bash"
    bash.parent.mkdir(parents=True)
    bash.write_text("#!/usr/bin/env bash\n")
    bash.chmod(0o755)
    return rootfs


def test_enter_rootfs_emits_plan_without_running_bwrap(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    output = tmp_path / "bwrap_plan.json"

    proc = subprocess.run(
        [
            "bash",
            str(ENTER_ROOTFS),
            "--rootfs",
            str(rootfs),
            "--",
            "python",
            "-c",
            "print('should not run')",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
            "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(output),
            "CUDA_VISIBLE_DEVICES": "0,1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    assert "should not run" not in proc.stdout
    plan = json.loads(output.read_text())
    assert plan["schema_version"] == 1
    assert plan["entrypoint"].endswith("scripts/rootfs/enter_rootfs.sh")
    assert plan["rootfs"]["path"] == str(rootfs)
    assert plan["cwd"] == "/workspace/torchtitan"
    assert plan["network_mode"] == "shared"
    assert plan["inner_argv"] == ["python", "-c", "print('should not run')"]
    assert plan["environment"]["TORCHTITAN_IN_ROOTFS"] == "1"
    assert plan["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert any(
        mount["target"] == "/workspace/torchtitan"
        and mount["source"] == str(REPO_ROOT)
        and mount["kind"] == "bind"
        for mount in plan["mounts"]
    )

    validated = validate_bwrap_plan(plan)
    assert validated["ok"] is True
    assert validated["blockers"] == []


def test_enter_rootfs_emit_plan_stdout_when_no_output_path(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    proc = subprocess.run(
        [
            "bash",
            str(ENTER_ROOTFS),
            "--rootfs",
            str(rootfs),
            "--",
            "/bin/true",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    plan = json.loads(proc.stdout)
    assert plan["inner_argv"] == ["/bin/true"]


def test_enter_rootfs_emit_plan_does_not_require_bwrap_on_path(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    python_link = fake_bin / "python3"
    python_link.symlink_to("/usr/bin/python3")
    dirname_link = fake_bin / "dirname"
    dirname_link.symlink_to("/usr/bin/dirname")
    pwd_link = fake_bin / "pwd"
    pwd_link.symlink_to("/usr/bin/pwd")
    bash_link = fake_bin / "bash"
    bash_link.symlink_to("/usr/bin/bash")
    cat_link = fake_bin / "cat"
    cat_link.symlink_to("/usr/bin/cat")
    mkdir_link = fake_bin / "mkdir"
    mkdir_link.symlink_to("/usr/bin/mkdir")

    proc = subprocess.run(
        [
            str(bash_link),
            str(ENTER_ROOTFS),
            "--rootfs",
            str(rootfs),
            "--",
            "/bin/true",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": str(fake_bin),
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    assert "bwrap not found" not in proc.stdout
    assert json.loads(proc.stdout)["inner_argv"] == ["/bin/true"]


def test_bwrap_plan_verifier_rejects_missing_workspace_mount(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    plan = {
        "schema_version": 1,
        "entrypoint": str(ENTER_ROOTFS),
        "rootfs": {"path": str(rootfs), "explicit": True},
        "cwd": "/workspace/torchtitan",
        "network_mode": "shared",
        "mounts": [
            {"kind": "bind", "source": str(rootfs), "target": "/", "writable": True}
        ],
        "devices": [],
        "driver_libraries": [],
        "environment": {
            "PATH": "/opt/cuda-synth/bin:/usr/local/bin:/usr/bin:/bin",
            "CUDA_HOME": "/opt/cuda-synth",
            "CUDA_PATH": "/opt/cuda-synth",
            "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:/opt/cuda-synth/lib64",
            "TORCHTITAN_IN_ROOTFS": "1",
            "HOME": "/root",
            "NVIDIA_VISIBLE_DEVICES": "all",
        },
        "inner_argv": ["/bin/true"],
        "bwrap_argv": [],
    }

    with pytest.raises(BwrapPlanError, match="/workspace/torchtitan"):
        validate_bwrap_plan(plan)


def test_bwrap_plan_verifier_rejects_rootfs_marker_drift(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    plan = {
        "schema_version": 1,
        "entrypoint": str(ENTER_ROOTFS),
        "rootfs": {"path": str(rootfs), "explicit": True},
        "cwd": "/workspace/torchtitan",
        "network_mode": "shared",
        "mounts": [
            {"kind": "bind", "source": str(rootfs), "target": "/", "writable": True},
            {"kind": "proc", "source": "procfs", "target": "/proc", "writable": False},
            {"kind": "tmpfs", "source": "tmpfs", "target": "/tmp", "writable": True},
            {"kind": "dev", "source": "devfs", "target": "/dev", "writable": True},
            {
                "kind": "bind",
                "source": str(REPO_ROOT),
                "target": "/workspace/torchtitan",
                "writable": True,
            },
        ],
        "devices": [],
        "driver_libraries": [],
        "environment": {
            "PATH": "/opt/cuda-synth/bin:/usr/local/bin:/usr/bin:/bin",
            "CUDA_HOME": "/opt/cuda-synth",
            "CUDA_PATH": "/opt/cuda-synth",
            "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:/opt/cuda-synth/lib64",
            "TORCHTITAN_IN_ROOTFS": "0",
            "HOME": "/root",
            "NVIDIA_VISIBLE_DEVICES": "all",
        },
        "inner_argv": ["/bin/true"],
        "bwrap_argv": [],
    }

    with pytest.raises(BwrapPlanError, match="TORCHTITAN_IN_ROOTFS"):
        validate_bwrap_plan(plan)
