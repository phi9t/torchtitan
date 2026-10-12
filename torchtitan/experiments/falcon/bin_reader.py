# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanogpt ``.bin`` reader and dataloader for the Falcon FineWeb path.

The nanogpt binary format is a 256-int32 little-endian header followed by a
flat uint16 token stream:

- ``header[0]`` magic number ``20240520``,
- ``header[1]`` version ``1``,
- ``header[2]`` claimed token count,
- ``header[3:]`` reserved (zero).

This layout is confirmed against the on-host FineWeb 10B GPT-2 shards under
``experiments/modded_nanogpt_b200/.../data/fineweb10B/`` (see the ticket 05
report). GPT-2 vocab is 50257, so the Falcon science model uses
``vocab_size=50257`` on this path.

The dataloader is single-rank (dp_world_size == 1) for the Step 4 science
smoke; it packs contiguous windows into ``(local_batch_size, seq_len)`` inputs
with next-token-shift labels, cycling shards forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torchtitan.components.dataloader import BaseDataLoader

_NANOGPT_MAGIC = 20240520
_NANOGPT_VERSION = 1
_HEADER_INTS = 256


def _resolve_bins(pattern: str) -> list[Path]:
    """Resolve a glob or literal path (relative or absolute) to shard paths.

    ``Path.glob`` rejects absolute patterns, so route absolute inputs through
    the anchored-root glob or treat them as a literal file.
    """
    literal = Path(pattern)
    if literal.is_file():
        return [literal]
    if literal.is_absolute():
        anchor = literal.anchor
        rel = str(literal.relative_to(anchor))
        return sorted(Path(anchor).glob(rel))
    return sorted(Path().glob(pattern))


def read_nanogpt_bin(path: str | Path) -> torch.Tensor:
    """Read a nanogpt ``.bin`` shard and return its tokens as int64.

    Raises ``ValueError`` if the magic number, version, or token count do not
    match the on-disk contract. Fails loud rather than silently truncating.
    """
    path = Path(path)
    header = np.fromfile(path, dtype=np.int32, count=_HEADER_INTS)
    if header.shape[0] != _HEADER_INTS:
        raise ValueError(f"nanogpt bin {path} is smaller than a 256-int32 header")
    if int(header[0]) != _NANOGPT_MAGIC:
        raise ValueError(
            f"nanogpt bin {path} magic number mismatch: got {int(header[0])}, "
            f"expected {_NANOGPT_MAGIC}"
        )
    if int(header[1]) != _NANOGPT_VERSION:
        raise ValueError(
            f"nanogpt bin {path} unsupported version {int(header[1])}, "
            f"expected {_NANOGPT_VERSION}"
        )
    num_tokens = int(header[2])
    tokens = np.fromfile(
        path, dtype=np.uint16, count=num_tokens, offset=_HEADER_INTS * 4
    )
    if tokens.shape[0] != num_tokens:
        raise ValueError(
            f"nanogpt bin {path} token count mismatch: header claims "
            f"{num_tokens}, read {tokens.shape[0]}"
        )
    return torch.from_numpy(tokens.astype(np.int64))


def write_nanogpt_bin(path: str | Path, tokens: torch.Tensor) -> None:
    """Write ``tokens`` to a nanogpt ``.bin`` shard (test/support helper)."""
    path = Path(path)
    flat = tokens.reshape(-1).to(torch.int64)
    if flat.min().item() < 0 or flat.max().item() > 0xFFFF:
        raise ValueError("nanogpt bin tokens must fit in uint16")
    header = np.zeros(_HEADER_INTS, dtype=np.int32)
    header[0] = _NANOGPT_MAGIC
    header[1] = _NANOGPT_VERSION
    header[2] = flat.numel()
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(flat.numpy().astype(np.uint16).tobytes())


class NanoGptBinDataLoader(BaseDataLoader):
    """Single-rank contiguous-window loader over nanogpt ``.bin`` shards."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        bin_glob: str = ""
        vocab_size: int = 50257

        def build(
            self,
            *,
            dp_world_size: int,
            dp_rank: int,
            tokenizer,
            seq_len: int,
            local_batch_size: int,
            snapshot_every_n_steps: int | None = 1,
            **kwargs,
        ) -> "NanoGptBinDataLoader":
            del tokenizer, snapshot_every_n_steps, kwargs
            if dp_world_size != 1 or dp_rank != 0:
                raise ValueError("Falcon nanogpt bin loader is single-rank only")
            files = _resolve_bins(self.bin_glob)
            if not files:
                raise ValueError(f"no nanogpt bins matched glob {self.bin_glob!r}")
            return NanoGptBinDataLoader(
                files,
                seq_len=seq_len,
                local_batch_size=local_batch_size,
                vocab_size=self.vocab_size,
            )

    def __init__(
        self,
        shard_paths: list[Path],
        *,
        seq_len: int,
        local_batch_size: int,
        vocab_size: int,
    ) -> None:
        if not shard_paths:
            raise ValueError("shard_paths must be non-empty")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if local_batch_size < 1:
            raise ValueError("local_batch_size must be >= 1")
        self.shard_paths = [Path(p) for p in shard_paths]
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.vocab_size = vocab_size
        # tokens needed to build one input/label window batch (+1 for the shift)
        self._window = local_batch_size * seq_len + 1

    def _iter_tokens(self):
        """Yield one shard's tokens at a time, cycling forever."""
        while True:
            for path in self.shard_paths:
                yield read_nanogpt_bin(path)

    def __iter__(self):
        shard_iter = self._iter_tokens()
        tokens = next(shard_iter)
        pos = 0
        positions = torch.arange(self.seq_len).expand(self.local_batch_size, -1)
        while True:
            if pos + self._window > tokens.numel():
                tokens = next(shard_iter)
                pos = 0
                if self._window > tokens.numel():
                    raise ValueError(
                        f"shard has {tokens.numel()} tokens, too few for one "
                        f"window of {self._window}"
                    )
            buf = tokens[pos : pos + self._window]
            if buf.max().item() >= self.vocab_size or buf.min().item() < 0:
                raise ValueError(
                    "token id outside vocab range "
                    f"[0, {self.vocab_size}) in nanogpt shard"
                )
            body = buf[:-1].view(self.local_batch_size, self.seq_len)
            labels = buf[1:].view(self.local_batch_size, self.seq_len)
            pos += self.local_batch_size * self.seq_len
            batch = {"input": body.clone(), "positions": positions}
            yield batch, labels.clone()

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict
