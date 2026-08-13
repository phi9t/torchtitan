# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wave F3 done-criteria: rootfs identity, store, and atomic selection.

Host-only tests (roadmap Section 17.1): no Docker, bwrap, CUDA, or a live
rootfs. They pin the identity model (two builds from the same locked inputs get
the same store ID; any changed input changes it), staged-tree acceptance and
quarantine, atomic selection with retained prior selection, rollback on a failed
post-activation launch check, and launcher-side revalidation.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from torchtitan.experiments.execution.rootfs import (
    manifest as manifest_mod,
    store as store_mod,
)


def _inputs(**overrides) -> manifest_mod.LockedBuildInputs:
    base = {
        "base_image_digest": "ubuntu@sha256:" + "a" * 64,
        "uv_image_digest": "ghcr.io/astral-sh/uv@sha256:" + "b" * 64,
        "apt_lock_digest": "sha256:" + "c" * 64,
        "python_lock_digest": "sha256:" + "d" * 64,
        "cuda_package_versions": ("cuda-toolkit-13-2=13.2.0", "libcudnn9=9.1.0"),
        "dockerfile_digest": "sha256:" + "e" * 64,
    }
    base.update(overrides)
    return manifest_mod.LockedBuildInputs(**base)


def _manifest(inputs: manifest_mod.LockedBuildInputs) -> dict[str, object]:
    return manifest_mod.build_manifest(
        inputs=inputs,
        provenance=manifest_mod.BuildProvenance(
            build_time="20260813T000000Z",
            host_tool_versions={"docker": "27.0", "uv": "0.5.0"},
            source_revision="abc123",
            source_tree_clean=True,
        ),
        image_id="sha256:" + "f" * 64,
        exported_rootfs_digest="sha256:" + "0" * 64,
    )


def _write_staged(root: Path, manifest: dict[str, object]) -> Path:
    staged = root / "staging"
    (staged / "bin").mkdir(parents=True)
    bash = staged / "bin" / "bash"
    bash.write_text("#!/bin/sh\n")
    bash.chmod(bash.stat().st_mode | stat.S_IXUSR)
    (staged / store_mod.OWNERSHIP_MARKER_NAME).write_text(
        store_mod.OWNERSHIP_MARKER_VALUE + "\n"
    )
    (staged / store_mod.MANIFEST_FILENAME).write_text(json.dumps(manifest))
    return staged


# --- identity model --------------------------------------------------------


def test_same_inputs_same_store_id():
    a = manifest_mod.compute_store_id(_inputs())
    b = manifest_mod.compute_store_id(_inputs())
    assert a == b
    assert a.startswith("rootfs-")


def test_cuda_package_order_does_not_change_identity():
    a = manifest_mod.compute_store_id(_inputs(cuda_package_versions=("x=1", "y=2")))
    b = manifest_mod.compute_store_id(_inputs(cuda_package_versions=("y=2", "x=1")))
    assert a == b


def test_changing_a_locked_input_changes_identity():
    base = manifest_mod.compute_store_id(_inputs())
    changed = manifest_mod.compute_store_id(
        _inputs(dockerfile_digest="sha256:" + "9" * 64)
    )
    assert base != changed


def test_provenance_does_not_change_identity():
    # A rebuild on a different day at the same inputs is the same environment.
    m1 = _manifest(_inputs())
    m2 = manifest_mod.build_manifest(
        inputs=_inputs(),
        provenance=manifest_mod.BuildProvenance(build_time="20991231T235959Z"),
        image_id="sha256:" + "1" * 64,
        exported_rootfs_digest="sha256:" + "2" * 64,
    )
    assert m1["store_id"] == m2["store_id"]


def test_unpinned_base_image_is_rejected():
    with pytest.raises(ValueError, match="digest-pinned"):
        _inputs(base_image_digest="ubuntu:latest")


def test_validate_manifest_rejects_tampered_store_id():
    manifest = _manifest(_inputs())
    manifest["store_id"] = "rootfs-" + "0" * 64
    with pytest.raises(ValueError, match="does not match recomputed"):
        manifest_mod.validate_manifest(manifest)


# --- staged-tree validation and quarantine ---------------------------------


def test_publish_accepts_valid_staged_tree(tmp_path):
    store = store_mod.RootfsStore(tmp_path / "store")
    inputs = _inputs()
    staged = _write_staged(tmp_path, _manifest(inputs))
    published = store.publish(staged)
    assert published == store.content_dir / manifest_mod.compute_store_id(inputs)
    assert (published / "bin" / "bash").is_file()
    assert not staged.exists()


def test_staged_tree_without_marker_is_rejected_and_quarantined(tmp_path):
    store = store_mod.RootfsStore(tmp_path / "store")
    staged = _write_staged(tmp_path, _manifest(_inputs()))
    (staged / store_mod.OWNERSHIP_MARKER_NAME).unlink()
    with pytest.raises(ValueError, match="ownership marker"):
        store.validate_staged_tree(staged)
    quarantined = store.quarantine(staged, reason="missing marker")
    assert quarantined.parent == store.quarantine_dir
    assert (quarantined / ".quarantine-reason").read_text().strip() == "missing marker"
    assert not staged.exists()


def test_staged_tree_with_nonexecutable_bash_is_rejected(tmp_path):
    store = store_mod.RootfsStore(tmp_path / "store")
    staged = _write_staged(tmp_path, _manifest(_inputs()))
    bash = staged / "bin" / "bash"
    bash.chmod(bash.stat().st_mode & ~0o111)
    with pytest.raises(ValueError, match="bin/bash is not executable"):
        store.validate_staged_tree(staged)


def test_republish_same_id_is_noop_and_drops_redundant_tree(tmp_path):
    store = store_mod.RootfsStore(tmp_path / "store")
    inputs = _inputs()
    first = store.publish(_write_staged(tmp_path / "a", _manifest(inputs)))
    second = store.publish(_write_staged(tmp_path / "b", _manifest(inputs)))
    assert first == second
    assert not (tmp_path / "b" / "staging").exists()


# --- selection, rollback, resolution ---------------------------------------


def _publish(store, tmp_path, name, inputs):
    return store.publish(_write_staged(tmp_path / name, _manifest(inputs)))


def test_activate_selects_and_resolves(tmp_path):
    store = store_mod.RootfsStore(tmp_path / "store")
    inputs = _inputs()
    _publish(store, tmp_path, "a", inputs)
    store_id = manifest_mod.compute_store_id(inputs)
    record = store_mod.activate(store, store_id, launch_check=lambda d: True)
    assert record.store_id == store_id
    resolved = store.resolve_selected_dir()
    assert resolved == (store.content_dir / store_id).resolve()


def test_failed_launch_check_rolls_back_to_prior_selection(tmp_path):
    store = store_mod.RootfsStore(tmp_path / "store")
    good = _inputs()
    bad = _inputs(dockerfile_digest="sha256:" + "7" * 64)
    _publish(store, tmp_path, "good", good)
    _publish(store, tmp_path, "bad", bad)
    good_id = manifest_mod.compute_store_id(good)
    bad_id = manifest_mod.compute_store_id(bad)

    store_mod.activate(store, good_id, launch_check=lambda d: True)
    with pytest.raises(ValueError, match="launch check failed"):
        store_mod.activate(store, bad_id, launch_check=lambda d: False)

    # The prior good selection survives a failed activation.
    assert store.read_selection().store_id == good_id


def test_failed_first_activation_leaves_no_selection(tmp_path):
    store = store_mod.RootfsStore(tmp_path / "store")
    inputs = _inputs()
    _publish(store, tmp_path, "a", inputs)
    store_id = manifest_mod.compute_store_id(inputs)
    with pytest.raises(ValueError, match="launch check failed"):
        store_mod.activate(store, store_id, launch_check=lambda d: False)
    assert store.read_selection() is None


def test_resolve_rejects_swapped_manifest_digest(tmp_path):
    store = store_mod.RootfsStore(tmp_path / "store")
    inputs = _inputs()
    published = _publish(store, tmp_path, "a", inputs)
    store_id = manifest_mod.compute_store_id(inputs)
    store_mod.activate(store, store_id, launch_check=lambda d: True)
    # Corrupt the published manifest so the recorded selection digest no longer
    # matches: the launcher must refuse to launch it.
    other = _inputs(dockerfile_digest="sha256:" + "5" * 64)
    (published / store_mod.MANIFEST_FILENAME).write_text(json.dumps(_manifest(other)))
    with pytest.raises(ValueError, match="does not match the selection record"):
        store.resolve_selected_dir()


def test_select_refuses_unpublished_entry(tmp_path):
    store = store_mod.RootfsStore(tmp_path / "store")
    store.content_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="unpublished store entry"):
        store.select("rootfs-" + "0" * 64)
