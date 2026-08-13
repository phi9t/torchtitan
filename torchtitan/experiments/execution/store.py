# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Atomic and append-only persistence for the typed lifecycle (roadmap 3.3).

Terminal files are immutable and written atomically (write to a temp file in
the same directory, fsync, then rename). Raw process event streams and artifact
indexes are append-only JSONL. Derived reports and views may be regenerated,
but terminal files must never be silently overwritten.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def write_terminal_json(path: Path, payload: object) -> None:
    """Atomically write an immutable terminal JSON file.

    Refuses to overwrite an existing terminal file: a terminal outcome or
    manifest is immutable evidence, so a second write is a programming error
    rather than an update.
    """

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"terminal file already exists (immutable): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")


def append_jsonl(path: Path, row: object) -> None:
    """Append one JSON row to an append-only stream, creating it if absent."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[object]:
    """Read a JSONL stream, returning an empty list when it does not exist."""

    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _atomic_write(path: Path, text: str) -> None:
    directory = path.parent
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
