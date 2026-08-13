# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wave F3 shell resolver: selection-record resolution in rootfs_target.sh.

The launcher resolves a store root's selection record to a canonical content
directory before bwrap (roadmap Section 8.2). These tests exercise the shell
resolver directly through a subprocess so the fail-closed behavior is verified
in the same interpreter the launcher uses, without Docker or bwrap.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOTFS_TARGET = REPO_ROOT / "scripts" / "rootfs" / "rootfs_target.sh"

MARKER_NAME = ".torchtitan-rootfs-owner"
MARKER_VALUE = "torchtitan-rootfs"


def _run(body: str, cwd: Path) -> subprocess.CompletedProcess:
    script = f"""
set -euo pipefail
source "{ROOTFS_TARGET}"
{body}
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def _make_store(root: Path, store_id: str, *, with_marker: bool = True) -> Path:
    """Create a minimal store with one published entry and a selection record."""

    entry = root / "content" / store_id
    entry.mkdir(parents=True)
    if with_marker:
        (entry / MARKER_NAME).write_text(MARKER_VALUE + "\n")
    (root / "selected.json").write_text(
        '{\n  "kind": "rootfs_selection",\n' f'  "store_id": "{store_id}"\n}}\n'
    )
    return entry


def test_resolves_selected_entry(tmp_path):
    store = tmp_path / "store"
    entry = _make_store(store, "rootfs-abc123")
    result = _run(f'rootfs_resolve_selected_dir "{store}"', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(entry.resolve())


def test_missing_selection_record_fails(tmp_path):
    store = tmp_path / "store"
    (store / "content").mkdir(parents=True)
    result = _run(f'rootfs_resolve_selected_dir "{store}"', cwd=tmp_path)
    assert result.returncode != 0
    assert "selection record" in (result.stderr + result.stdout).lower()


def test_entry_without_marker_fails(tmp_path):
    store = tmp_path / "store"
    _make_store(store, "rootfs-abc123", with_marker=False)
    result = _run(f'rootfs_resolve_selected_dir "{store}"', cwd=tmp_path)
    assert result.returncode != 0
    assert "ownership marker" in (result.stderr + result.stdout).lower()


def test_traversal_store_id_is_refused(tmp_path):
    store = tmp_path / "store"
    (store / "content").mkdir(parents=True)
    # A record whose store_id tries to escape content/ must be refused before
    # any directory is resolved.
    (store / "selected.json").write_text('{"store_id": "../../etc"}\n')
    result = _run(f'rootfs_resolve_selected_dir "{store}"', cwd=tmp_path)
    assert result.returncode != 0
    assert "invalid store_id" in (result.stderr + result.stdout).lower()


def test_symlinked_entry_is_refused(tmp_path):
    store = tmp_path / "store"
    (store / "content").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / MARKER_NAME).write_text(MARKER_VALUE + "\n")
    link = store / "content" / "rootfs-evil"
    link.symlink_to(outside)
    (store / "selected.json").write_text('{"store_id": "rootfs-evil"}\n')
    result = _run(f'rootfs_resolve_selected_dir "{store}"', cwd=tmp_path)
    assert result.returncode != 0
    assert "not a directory" in (result.stderr + result.stdout).lower()


def test_symlinked_store_root_is_refused(tmp_path):
    real = tmp_path / "real_store"
    _make_store(real, "rootfs-abc123")
    link = tmp_path / "link_store"
    link.symlink_to(real)
    result = _run(f'rootfs_resolve_selected_dir "{link}"', cwd=tmp_path)
    assert result.returncode != 0
    assert "real directory" in (result.stderr + result.stdout).lower()
