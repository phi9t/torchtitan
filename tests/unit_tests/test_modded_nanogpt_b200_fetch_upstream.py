# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from experiments.modded_nanogpt_b200 import fetch_upstream


def _make_repo(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "train_gpt.py").write_text("print('fixture')\n")
    (path / "triton_kernels.py").write_text("# fixture\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=path,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def test_existing_clean_source_writes_source_json(tmp_path: Path):
    source = tmp_path / "sources" / "modded-nanogpt"
    commit = _make_repo(source)
    result_dir = tmp_path / "results" / "attempt"

    record = fetch_upstream.prepare_source(
        source=source,
        result_dir=result_dir,
        pinned_commit=commit,
    )

    written = json.loads((result_dir / "source.json").read_text())
    assert written == record
    assert record["path"] == str(source)
    assert record["commit"] == commit
    assert record["status"] == ""
    assert not (result_dir / "source.json.tmp").exists()


def test_existing_dirty_source_is_refused(tmp_path: Path):
    source = tmp_path / "sources" / "modded-nanogpt"
    commit = _make_repo(source)
    (source / "train_gpt.py").write_text("dirty\n")

    with pytest.raises(ValueError, match="refusing dirty source"):
        fetch_upstream.prepare_source(
            source=source,
            result_dir=tmp_path / "results" / "attempt",
            pinned_commit=commit,
        )


def test_missing_source_clones_from_local_remote(tmp_path: Path):
    remote = tmp_path / "remote"
    commit = _make_repo(remote)
    source = tmp_path / "sources" / "modded-nanogpt"

    record = fetch_upstream.prepare_source(
        source=source,
        result_dir=tmp_path / "results" / "attempt",
        pinned_commit=commit,
        repository=str(remote),
    )

    assert source.exists()
    assert record["commit"] == commit
    assert (
        subprocess.check_output(
            ["git", "-C", str(source), "status", "--short"], text=True
        )
        == ""
    )
