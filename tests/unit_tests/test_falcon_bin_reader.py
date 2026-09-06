# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch

from torchtitan.experiments.falcon.bin_reader import (
    NanoGptBinDataLoader,
    read_nanogpt_bin,
    write_nanogpt_bin,
)


def _write_raw_bin(
    path: Path, tokens: list[int], *, magic: int = 20240520, version: int = 1
) -> None:
    """Write a nanogpt .bin exactly per the on-disk contract (no helper)."""
    header = [0] * 256
    header[0] = magic
    header[1] = version
    header[2] = len(tokens)
    with path.open("wb") as f:
        f.write(struct.pack("<256i", *header))
        f.write(struct.pack(f"<{len(tokens)}H", *tokens))


def test_read_nanogpt_bin_roundtrips_tokens(tmp_path):
    path = tmp_path / "tiny.bin"
    tokens = [1, 2, 3, 65535, 0, 50256]
    _write_raw_bin(path, tokens)

    read = read_nanogpt_bin(path)

    assert read.dtype == torch.int64
    assert read.tolist() == tokens


def test_write_then_read_roundtrips(tmp_path):
    path = tmp_path / "wr.bin"
    tokens = torch.tensor([7, 8, 9, 10, 11], dtype=torch.int64)
    write_nanogpt_bin(path, tokens)
    read = read_nanogpt_bin(path)
    assert read.tolist() == tokens.tolist()


def test_read_rejects_bad_magic(tmp_path):
    path = tmp_path / "bad_magic.bin"
    _write_raw_bin(path, [1, 2, 3], magic=12345)
    with pytest.raises(ValueError, match="magic"):
        read_nanogpt_bin(path)


def test_read_rejects_bad_version(tmp_path):
    path = tmp_path / "bad_version.bin"
    _write_raw_bin(path, [1, 2, 3], version=2)
    with pytest.raises(ValueError, match="version"):
        read_nanogpt_bin(path)


def test_read_rejects_truncated_token_stream(tmp_path):
    path = tmp_path / "trunc.bin"
    header = [0] * 256
    header[0] = 20240520
    header[1] = 1
    header[2] = 100  # claim 100 tokens but only write 3
    with path.open("wb") as f:
        f.write(struct.pack("<256i", *header))
        f.write(struct.pack("<3H", 1, 2, 3))
    with pytest.raises(ValueError, match="token"):
        read_nanogpt_bin(path)


def test_dataloader_yields_shifted_next_token_labels(tmp_path):
    path = tmp_path / "shard.bin"
    # 21 tokens -> with seq_len 4, local_batch 2, one step needs 2*4+1 = 9.
    tokens = list(range(21))
    _write_raw_bin(path, tokens)

    loader = NanoGptBinDataLoader(
        [path], seq_len=4, local_batch_size=2, vocab_size=50257
    )
    batch, labels = next(iter(loader))

    assert batch["input"].shape == (2, 4)
    assert labels.shape == (2, 4)
    # Labels are the next-token shift of a contiguous window, not a roll.
    flat_in = batch["input"].reshape(-1).tolist()
    flat_lbl = labels.reshape(-1).tolist()
    # First window: inputs 0..3, labels 1..4; second window: 4..7, labels 5..8.
    assert flat_in == [0, 1, 2, 3, 4, 5, 6, 7]
    assert flat_lbl == [1, 2, 3, 4, 5, 6, 7, 8]


def test_dataloader_advances_and_wraps_across_shards(tmp_path):
    path = tmp_path / "small.bin"
    tokens = list(range(11))  # only enough for one step of 2*4+1=9, then wrap
    _write_raw_bin(path, tokens)

    loader = NanoGptBinDataLoader(
        [path], seq_len=4, local_batch_size=2, vocab_size=50257
    )
    it = iter(loader)
    first, _ = next(it)
    second, _ = next(it)
    # After exhausting the single shard the loader wraps to the start.
    assert first["input"].reshape(-1).tolist()[0] == 0
    assert second["input"].reshape(-1).tolist()[0] == 0


def test_dataloader_rejects_token_over_vocab(tmp_path):
    path = tmp_path / "overflow.bin"
    # 60000 fits in uint16 but exceeds a small vocab, so the loader must reject
    # it as an out-of-range token id.
    _write_raw_bin(path, [0, 1, 2, 60000, 4, 5, 6, 7, 8, 9])
    loader = NanoGptBinDataLoader(
        [path], seq_len=4, local_batch_size=2, vocab_size=50257
    )
    with pytest.raises(ValueError, match="vocab"):
        next(iter(loader))


def test_dataloader_config_builds_from_registry(tmp_path):
    path = tmp_path / "cfg.bin"
    _write_raw_bin(path, list(range(64)))
    config = NanoGptBinDataLoader.Config(
        bin_glob=str(path),
        vocab_size=50257,
    )
    loader = config.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=None,
        seq_len=4,
        local_batch_size=2,
    )
    batch, labels = next(iter(loader))
    assert batch["input"].shape == (2, 4)
