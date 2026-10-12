# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from torchtitan.experiments.mini_kimi_k3.token_shards import (
    load_manifest,
    TokenShardLoader,
)


def _write_shard(path: Path, tokens: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(tokens, dtype="<u4").tofile(path)


def _write_manifest(path: Path, sources: dict[str, list[str]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "sources": {
                    name: {"shards": shard_paths}
                    for name, shard_paths in sources.items()
                }
            },
            sort_keys=True,
        )
    )
    return path


def test_reads_little_endian_uint32_sequences_across_shards(tmp_path: Path):
    _write_shard(tmp_path / "alpha" / "a.bin", [1, 2, 3])
    _write_shard(tmp_path / "alpha" / "b.bin", [4, 5, 6])
    manifest = _write_manifest(
        tmp_path / "manifest.json", {"alpha": ["a.bin", "b.bin"]}
    )

    loader = TokenShardLoader(
        manifest=load_manifest(manifest),
        tokens_dir=tmp_path,
        seq_len=4,
    )

    assert loader.next_sequence().tolist() == [1, 2, 3, 4]
    assert loader.state_dict()["tokens_emitted"] == 4
    assert loader.tokens_by_source == {"alpha": 4}


def test_rank_assignment_uses_round_robin_shards_per_source(tmp_path: Path):
    for idx in range(4):
        _write_shard(tmp_path / "alpha" / f"s{idx}.bin", [idx * 10, idx * 10 + 1])
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {"alpha": [f"s{idx}.bin" for idx in range(4)]},
    )

    rank0 = TokenShardLoader(
        load_manifest(manifest), tmp_path, seq_len=2, rank=0, world_size=2
    )
    rank1 = TokenShardLoader(
        load_manifest(manifest), tmp_path, seq_len=2, rank=1, world_size=2
    )

    assert [path.name for path in rank0.assigned_shards["alpha"]] == [
        "s0.bin",
        "s2.bin",
    ]
    assert [path.name for path in rank1.assigned_shards["alpha"]] == [
        "s1.bin",
        "s3.bin",
    ]
    assert rank0.assigned_token_count == 4
    assert rank1.assigned_token_count == 4
    assert rank0.next_sequence().tolist() == [0, 1]
    assert rank1.next_sequence().tolist() == [10, 11]


def test_state_round_trip_resumes_at_next_sequence(tmp_path: Path):
    _write_shard(tmp_path / "alpha" / "a.bin", [1, 2, 3, 4, 5, 6])
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})

    loader = TokenShardLoader(load_manifest(manifest), tmp_path, seq_len=2)
    assert loader.next_sequence().tolist() == [1, 2]
    state = loader.state_dict()
    expected = loader.next_sequence().tolist()

    resumed = TokenShardLoader(load_manifest(manifest), tmp_path, seq_len=2)
    resumed.load_state_dict(state)

    assert resumed.next_sequence().tolist() == expected
    assert resumed.state_dict()["tokens_emitted"] == 4


@pytest.mark.parametrize(
    ("manifest_data", "kwargs", "message"),
    [
        ({"sources": {"alpha": {"shards": ["a.bin"]}}}, {"seq_len": 0}, "seq_len"),
        (
            {"sources": {"alpha": {"shards": ["a.bin"]}}},
            {"seq_len": 2, "rank": 1, "world_size": 1},
            "rank",
        ),
        ({"sources": {}}, {"seq_len": 2}, "at least one source"),
        ({"sources": {"alpha": {"shards": []}}}, {"seq_len": 2}, "has no shards"),
    ],
)
def test_invalid_config_fails_loudly(
    tmp_path: Path,
    manifest_data: dict,
    kwargs: dict,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        TokenShardLoader(manifest_data, tmp_path, **kwargs)


def test_missing_and_short_shards_fail_loudly(tmp_path: Path):
    missing_manifest = {"sources": {"alpha": {"shards": ["missing.bin"]}}}
    with pytest.raises(FileNotFoundError, match="missing.bin"):
        TokenShardLoader(missing_manifest, tmp_path, seq_len=2)

    _write_shard(tmp_path / "alpha" / "short.bin", [1])
    short_manifest = {"sources": {"alpha": {"shards": ["short.bin"]}}}
    with pytest.raises(ValueError, match="fewer tokens than seq_len"):
        TokenShardLoader(short_manifest, tmp_path, seq_len=2)
