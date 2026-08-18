# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.modded_nanogpt_b200 import prepare_data


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def test_write_manifest_records_shards_environment_and_freshness(tmp_path: Path):
    data_dir = tmp_path / "data" / "fineweb10B"
    data_dir.mkdir(parents=True)
    shard0 = data_dir / "train_000000.bin"
    shard1 = data_dir / "train_000001.bin"
    shard0.write_bytes(b"abc")
    shard1.write_bytes(b"defg")
    output = tmp_path / "results" / "data_manifest.json"

    manifest = prepare_data.write_manifest(
        output=output,
        data_dir=data_dir,
        source_path=tmp_path / "source",
        source_commit="ecbb586296d3dac36fd206211f25d63bad4a6b35",
        command=["python", "data/cached_fineweb10B.py", "1"],
        token_budget="smoke",
        freshness="reused",
        environment={"python": "3.12.3", "torch": "2.13.0+cu132"},
    )

    written = json.loads(output.read_text())
    assert written == manifest
    assert manifest["schema_version"] == 1
    assert manifest["dataset"] == "fineweb10B"
    assert manifest["token_budget"] == "smoke"
    assert manifest["source"]["commit"] == "ecbb586296d3dac36fd206211f25d63bad4a6b35"
    assert manifest["command"] == ["python", "data/cached_fineweb10B.py", "1"]
    assert manifest["freshness"] == "reused"
    assert manifest["environment"]["torch"] == "2.13.0+cu132"
    assert manifest["num_files"] == 2
    assert manifest["total_bytes"] == 7
    assert manifest["verified_sha"] is True
    assert manifest["files"] == [
        {"path": str(shard0), "bytes": 3, "sha256": _sha256(shard0)},
        {"path": str(shard1), "bytes": 4, "sha256": _sha256(shard1)},
    ]
    assert not (tmp_path / "results" / "data_manifest.json.tmp").exists()


def test_manifest_requires_bin_shards(tmp_path: Path):
    data_dir = tmp_path / "data" / "fineweb10B"
    data_dir.mkdir(parents=True)

    try:
        prepare_data.build_manifest(
            data_dir=data_dir,
            source_path=tmp_path / "source",
            source_commit="commit",
            command=["python", "data/cached_fineweb10B.py", "1"],
            token_budget="smoke",
            freshness="fresh",
            environment={},
        )
    except ValueError as exc:
        assert "no .bin shards" in str(exc)
    else:
        raise AssertionError("expected missing-shards failure")


def test_full_manifest_validates_expected_shape(tmp_path: Path):
    data_dir = tmp_path / "data" / "fineweb10B"
    data_dir.mkdir(parents=True)
    for idx in range(2):
        (data_dir / f"train_{idx:06d}.bin").write_bytes(b"x")

    try:
        prepare_data.build_manifest(
            data_dir=data_dir,
            source_path=tmp_path / "source",
            source_commit="ecbb586296d3dac36fd206211f25d63bad4a6b35",
            command=["python", "data/cached_fineweb10B.py", "9"],
            token_budget="900M",
            freshness="fresh",
            environment={},
        )
    except ValueError as exc:
        assert "full manifest requires exactly 10 .bin shards" in str(exc)
    else:
        raise AssertionError("expected full-shape validation failure")
