# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shell-level tests for the canonical rootfs runtime environment helper."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ENV = REPO_ROOT / "scripts" / "rootfs" / "runtime_env.sh"


def _run(
    body: str, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    script = f"""
set -euo pipefail
source "{RUNTIME_ENV}"
{body}
"""
    run_env = {"PATH": "/usr/bin:/bin"}
    if env:
        run_env.update(env)
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=run_env,
        capture_output=True,
        text=True,
    )


def test_runtime_env_exports_canonical_json(tmp_path: Path):
    host_state = REPO_ROOT / ".cache" / "pytest-rootfs-state"
    result = _run(
        "rootfs_runtime_init && rootfs_runtime_env_json",
        env={"TORCHTITAN_ROOTFS_HOST_STATE": str(host_state)},
    )

    assert result.returncode == 0, result.stderr
    environment = json.loads(result.stdout)
    assert environment["TORCHTITAN_ROOTFS_ENV"] == "modded_nanogpt_b200"
    assert environment["TORCHTITAN_ROOTFS_STORE_ID"] == "legacy-rootfs"
    assert environment["TORCHTITAN_ROOTFS_HOST_STATE"] == str(host_state)
    assert environment["HOME"] == "/project/home"
    assert environment["TMPDIR"] == "/project/tmp"
    assert environment["UV_CACHE_DIR"] == "/project/uv-cache"
    assert environment["MISE_DATA_DIR"] == "/project/mise/data"
    assert environment["HF_HUB_CACHE"] == "/workspace/torchtitan/.cache/huggingface/hub"
    assert environment["PYTHON"] == "/project/venvs/b200-runtime/bin/python"
    assert environment["PATH"].split(":")[:2] == [
        "/project/venvs/b200-runtime/bin",
        "/project/mise/data/shims",
    ]


def test_runtime_env_rejects_host_state_outside_repo(tmp_path: Path):
    result = _run(
        "rootfs_runtime_init",
        env={"TORCHTITAN_ROOTFS_HOST_STATE": str(tmp_path / "outside")},
    )

    assert result.returncode != 0
    assert "runtime state root" in (result.stderr + result.stdout).lower()
