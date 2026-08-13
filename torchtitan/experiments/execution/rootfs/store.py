# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Content-addressed rootfs store and atomic selection (roadmap Section 8.2).

Build and activation are separate transactions. A build exports into an
invocation-scoped staging directory; only after its ownership marker, manifest
digest, expected structure, and executable ``bin/bash`` validate does it become
publishable as an immutable content-addressed directory named by its store ID.

Activation never recursively replaces the active directory. It atomically
replaces a small fsynced selection record that names the validated directory,
and keeps the previously selected store ID until a post-activation launch check
passes. A failed check or interrupted activation restores the prior selection
record; an invalid staged tree is quarantined for inspected cleanup.

This module is pure filesystem logic over a store root, host-testable without
Docker or bwrap. The store root, publication, selection, rollback, and
quarantine are modeled here; the launcher/builder shell scripts call an
equivalent resolver and pass the resolved directory to bwrap.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from torchtitan.experiments.execution.rootfs import manifest as manifest_mod

# Layout under the store root. Content entries live under content/<store_id>;
# the current selection is a single small record; quarantine holds rejected
# staged trees for inspection.
CONTENT_DIRNAME = "content"
QUARANTINE_DIRNAME = "quarantine"
SELECTION_FILENAME = "selected.json"
MANIFEST_FILENAME = "build_manifest.json"
OWNERSHIP_MARKER_NAME = ".torchtitan-rootfs-owner"
OWNERSHIP_MARKER_VALUE = "torchtitan-rootfs"

# Structure every published rootfs must contain to be selectable. bin/bash must
# additionally be executable; the launcher relies on it.
REQUIRED_ENTRIES = ("bin/bash",)


@dataclass(frozen=True)
class SelectionRecord:
    """The single small record naming the currently selected store entry."""

    store_id: str
    manifest_digest: str

    def to_payload(self) -> dict[str, object]:
        """Return the JSON payload persisted as the selection record."""

        return {
            "schema_version": 1,
            "kind": "rootfs_selection",
            "store_id": self.store_id,
            "manifest_digest": self.manifest_digest,
        }


class RootfsStore:
    """A content-addressed store of published rootfs directories."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.content_dir = self.root / CONTENT_DIRNAME
        self.quarantine_dir = self.root / QUARANTINE_DIRNAME
        self.selection_path = self.root / SELECTION_FILENAME

    # -- staging validation and publication ---------------------------------

    def validate_staged_tree(self, staged: Path) -> str:
        """Validate an exported staging tree and return its store ID.

        A staged tree is selectable only when it carries the builder ownership
        marker, a manifest whose recorded store ID is internally consistent, the
        required structure, and an executable ``bin/bash``. Any failure raises;
        callers quarantine the tree rather than publishing it.
        """

        staged = Path(staged)
        if not staged.is_dir() or staged.is_symlink():
            raise ValueError(f"staged tree is not a real directory: {staged}")

        marker = staged / OWNERSHIP_MARKER_NAME
        if not marker.is_file() or marker.read_text().strip() != OWNERSHIP_MARKER_VALUE:
            raise ValueError(f"staged tree lacks a valid ownership marker: {staged}")

        manifest_path = staged / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ValueError(f"staged tree has no {MANIFEST_FILENAME}: {staged}")
        manifest = json.loads(manifest_path.read_text())
        store_id = manifest_mod.validate_manifest(manifest)

        for rel in REQUIRED_ENTRIES:
            entry = staged / rel
            if not entry.is_file():
                raise ValueError(f"staged tree missing required entry {rel}: {staged}")
        bash = staged / "bin" / "bash"
        if not os.access(bash, os.X_OK):
            raise ValueError(f"staged tree bin/bash is not executable: {staged}")
        return store_id

    def publish(self, staged: Path) -> Path:
        """Publish a validated staging tree as an immutable content entry.

        Publication is atomic: the staged directory is renamed into place under
        content/<store_id>. Re-publishing an existing store ID is a no-op that
        removes the redundant staging tree, because content is immutable and the
        already-published entry is authoritative.
        """

        store_id = self.validate_staged_tree(staged)
        self.content_dir.mkdir(parents=True, exist_ok=True)
        target = self.content_dir / store_id
        if target.exists():
            # Identical content already published; drop the redundant staging
            # tree rather than replacing immutable content.
            _remove_tree(Path(staged))
            return target
        # Same-filesystem atomic publication. If the platform cannot rename
        # atomically here, os.rename raises rather than falling back to a
        # non-atomic copy, satisfying the roadmap's fail-rather-than-degrade rule.
        os.rename(Path(staged), target)
        return target

    def quarantine(self, staged: Path, *, reason: str) -> Path:
        """Move an invalid staged tree into quarantine for inspection."""

        staged = Path(staged)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        dest = Path(tempfile.mkdtemp(prefix="rejected-", dir=self.quarantine_dir))
        # mkdtemp created an empty dir; replace it with the staged tree.
        dest.rmdir()
        os.rename(staged, dest)
        (dest / ".quarantine-reason").write_text(reason + "\n")
        return dest

    # -- selection, rollback, resolution ------------------------------------

    def read_selection(self) -> SelectionRecord | None:
        """Return the current selection record, or None when none is selected."""

        if not self.selection_path.is_file():
            return None
        payload = json.loads(self.selection_path.read_text())
        return SelectionRecord(
            store_id=payload["store_id"],
            manifest_digest=payload["manifest_digest"],
        )

    def select(self, store_id: str) -> SelectionRecord:
        """Atomically select a published store entry, retaining the prior one.

        The prior selection is returned so a caller can restore it if a
        post-activation launch check fails. Selection only rewrites the small
        record; it never touches the content directories, so an already-live
        job bound to the prior directory is unaffected.
        """

        entry = self.content_dir / store_id
        if not entry.is_dir() or entry.is_symlink():
            raise ValueError(f"cannot select unpublished store entry: {store_id}")
        manifest = json.loads((entry / MANIFEST_FILENAME).read_text())
        digest = manifest_mod.validate_manifest(manifest)
        record = SelectionRecord(store_id=store_id, manifest_digest=digest)
        _atomic_write(
            self.selection_path,
            json.dumps(record.to_payload(), sort_keys=True, indent=2) + "\n",
        )
        return record

    def restore_selection(self, prior: SelectionRecord | None) -> None:
        """Restore a prior selection after a failed or interrupted activation.

        Restoring to no prior selection removes the record entirely, so a failed
        first activation does not leave a dangling selection pointing at an
        unverified entry.
        """

        if prior is None:
            # os.replace cannot target a nonexistent name, so unlink directly.
            self.selection_path.unlink(missing_ok=True)
            return
        _atomic_write(
            self.selection_path,
            json.dumps(prior.to_payload(), sort_keys=True, indent=2) + "\n",
        )

    def resolve_selected_dir(self) -> Path:
        """Resolve the selection to a revalidated canonical content directory.

        The launcher calls this immediately before bwrap: it re-reads the small
        record, resolves it to a real directory, and revalidates the ownership
        marker and manifest digest so a corrupted or swapped entry cannot be
        launched.
        """

        record = self.read_selection()
        if record is None:
            raise ValueError("no rootfs is selected")
        entry = self.content_dir / record.store_id
        if not entry.is_dir() or entry.is_symlink():
            raise ValueError(
                f"selected store entry is missing or not a directory: {record.store_id}"
            )
        canonical = entry.resolve(strict=True)
        marker = canonical / OWNERSHIP_MARKER_NAME
        if not marker.is_file() or marker.read_text().strip() != OWNERSHIP_MARKER_VALUE:
            raise ValueError(
                f"selected entry lacks a valid ownership marker: {record.store_id}"
            )
        manifest = json.loads((canonical / MANIFEST_FILENAME).read_text())
        digest = manifest_mod.validate_manifest(manifest)
        if digest != record.manifest_digest:
            raise ValueError(
                f"selected entry manifest digest {digest!r} does not match the "
                f"selection record {record.manifest_digest!r}"
            )
        return canonical


def activate(
    store: RootfsStore,
    store_id: str,
    *,
    launch_check: Callable[[Path], bool],
) -> SelectionRecord:
    """Select store_id, run a launch check, and roll back on failure.

    Models the roadmap's two-phase activation: the small selection record is
    replaced atomically, then a post-activation launch check runs against the
    resolved directory. If the check fails (or raises), the prior selection is
    restored so a bad activation never leaves the store pointing at an entry
    that cannot launch. Returns the new selection record on success.
    """

    prior = store.read_selection()
    store.select(store_id)
    try:
        resolved = store.resolve_selected_dir()
        ok = launch_check(resolved)
    except BaseException:
        store.restore_selection(prior)
        raise
    if not ok:
        store.restore_selection(prior)
        raise ValueError(
            f"post-activation launch check failed for {store_id}; "
            "restored prior selection"
        )
    selected = store.read_selection()
    assert selected is not None  # select() just wrote it
    return selected


def _atomic_write(path: Path, text: str) -> None:
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _remove_tree(path: Path) -> None:
    # Shallow-safe recursive removal for a redundant staging tree we own.
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_dir() and not child.is_symlink():
            child.rmdir()
        else:
            child.unlink()
    path.rmdir()
