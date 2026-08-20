# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shell-level tests for managed rootfs store build activation.

These tests use the builder's test-only staged-rootfs hook, so they do not call
Docker and do not mutate the repository's real rootfs. They pin the observable
store contract owned by scripts/rootfs/build_rootfs.sh: staged content is
published under content/<store_id>, selected.json is updated atomically, and the
resolved selected directory is launchable by enter_rootfs.sh plan emission.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_ROOTFS = REPO_ROOT / "scripts" / "rootfs" / "build_rootfs.sh"
ENTER_ROOTFS = REPO_ROOT / "scripts" / "rootfs" / "enter_rootfs.sh"


def test_build_rootfs_can_publish_and_select_managed_store(tmp_path: Path):
    store = tmp_path / "managed-store"
    plan = tmp_path / "bwrap_plan.json"

    proc = subprocess.run(
        [
            "bash",
            str(BUILD_ROOTFS),
            "--store",
            str(store),
            "--test-staged-rootfs",
        ],
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    selection = json.loads((store / "selected.json").read_text())
    store_id = selection["store_id"]
    rootfs = store / "content" / store_id
    assert rootfs.is_dir()
    assert (rootfs / "bin" / "bash").is_file()
    assert (rootfs / ".torchtitan-rootfs-owner").read_text().strip() == "torchtitan-rootfs"
    manifest = json.loads((rootfs / "build_manifest.json").read_text())
    assert manifest["store_id"] == store_id
    assert manifest["mutable_rootfs_allowed"] is False

    plan_proc = subprocess.run(
        [
            "bash",
            str(ENTER_ROOTFS),
            "--rootfs-store",
            str(store),
            "--",
            "/bin/true",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
            "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(plan),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert plan_proc.returncode == 0, plan_proc.stdout
    emitted = json.loads(plan.read_text())
    assert emitted["rootfs"]["path"] == str(rootfs.resolve())
    assert emitted["rootfs"]["store_root"] == str(store.resolve())
    assert emitted["rootfs"]["store_id"] == store_id


def test_build_rootfs_rejects_store_destination_under_existing_file(tmp_path: Path):
    store = tmp_path / "not-a-store"
    store.write_text("not a directory\n")

    proc = subprocess.run(
        [
            "bash",
            str(BUILD_ROOTFS),
            "--store",
            str(store),
            "--test-staged-rootfs",
        ],
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode != 0
    assert "store" in proc.stdout.lower()
