# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken

from torchtitan.experiments.falcon.bin_reader import read_nanogpt_bin

from torchtitan.experiments.glm53_arch_speedrun.data_prep import fineweb_edu
from torchtitan.experiments.glm53_arch_speedrun.data_prep.fineweb_edu import (
    EOT_TOKEN_ID,
    main,
    read_nanogpt_header,
)


class FakeEncoding:
    def encode_ordinary(self, text: str) -> list[int]:
        return [(ord(char) % 251) + 1 for char in text]

    def encode_ordinary_batch(
        self, texts: list[str], num_threads: int = 1
    ) -> list[list[int]]:
        del num_threads
        return [self.encode_ordinary(text) for text in texts]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def _write_fixture_parquet(path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": texts})
    pq.write_table(table, path)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_tokenize_writes_s01_compatible_shards(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    fake_encoding = FakeEncoding()
    monkeypatch.setattr(
        fineweb_edu.tiktoken, "get_encoding", lambda name: fake_encoding
    )
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: fake_encoding)
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    revision = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    raw_dir.mkdir(parents=True)
    (raw_dir / "DATASET_REVISION.txt").write_text(revision + "\n")
    texts = ["hello world", "small data prep test", "another document"]
    _write_fixture_parquet(raw_dir / "sample" / "10BT" / "000_00000.parquet", texts)

    assert (
        main(
            [
                "tokenize",
                "--raw-dir",
                str(raw_dir),
                "--output-dir",
                str(out_dir),
                "--shard-size",
                "8",
                "--batch-rows",
                "2",
                "--num-processes",
                "1",
            ]
        )
        == 0
    )

    manifest = _read_jsonl(out_dir / "MANIFEST.jsonl")
    assert manifest[0]["file"] == "fineweb_edu_val_000000.bin"
    assert manifest[1]["file"] == "fineweb_edu_train_000001.bin"
    assert manifest[0]["num_tokens"] == 8

    first_shard = out_dir / manifest[0]["file"]
    header = read_nanogpt_header(first_shard)
    assert header["num_tokens"] == 8
    tokens = read_nanogpt_bin(first_shard).tolist()
    encoder = tiktoken.get_encoding("gpt2")
    expected_prefix = [EOT_TOKEN_ID] + encoder.encode_ordinary(texts[0])
    assert tokens == expected_prefix[: len(tokens)]

    assert main(["verify", "--data-dir", str(out_dir), "--spot-tokens", "8"]) == 0
    verify = json.loads((out_dir / "VERIFY.json").read_text())
    assert verify["spot_check"]["starts_with_eot"]
    assert verify["num_shards"] == len(manifest)
