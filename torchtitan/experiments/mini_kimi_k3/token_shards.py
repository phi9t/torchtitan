# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pretokenized uint32 shard loading for the Mini Kimi K3 experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class _SourceSpec:
    name: str
    shards: list[Path]


class _SourceCursor:
    def __init__(self, name: str, shards: list[Path]) -> None:
        if not shards:
            raise ValueError(f"source {name!r} has no shards")
        self.name = name
        self.shards = shards
        self.shard_index = 0
        self.offset = 0
        self.epochs = 0
        self._array: np.memmap | None = None
        self._array_index: int | None = None

    def take(self, num_tokens: int) -> np.ndarray:
        out = np.empty(num_tokens, dtype=np.uint32)
        filled = 0
        while filled < num_tokens:
            array = self._current_array()
            available = int(array.size) - self.offset
            if available <= 0:
                self._advance_shard()
                continue
            take = min(available, num_tokens - filled)
            out[filled : filled + take] = array[self.offset : self.offset + take]
            self.offset += take
            filled += take
            if self.offset >= int(array.size):
                self._advance_shard()
        return out

    def state_dict(self) -> dict[str, int]:
        return {
            "shard_index": self.shard_index,
            "offset": self.offset,
            "epochs": self.epochs,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        shard_index = int(state["shard_index"])
        offset = int(state["offset"])
        epochs = int(state["epochs"])
        if shard_index < 0 or shard_index >= len(self.shards):
            raise ValueError(
                f"source {self.name!r} checkpoint shard_index {shard_index} "
                f"is outside [0, {len(self.shards)})"
            )
        shard_size = _uint32_token_count(self.shards[shard_index])
        if offset < 0 or offset > shard_size:
            raise ValueError(
                f"source {self.name!r} checkpoint offset {offset} is outside "
                f"[0, {shard_size}]"
            )
        if epochs < 0:
            raise ValueError(f"source {self.name!r} checkpoint epochs must be >= 0")
        self.shard_index = shard_index
        self.offset = offset
        self.epochs = epochs
        self._array_index = None
        self._array = None

    def _current_array(self) -> np.memmap:
        if self._array_index != self.shard_index:
            self._array = np.memmap(
                self.shards[self.shard_index],
                dtype="<u4",
                mode="r",
            )
            self._array_index = self.shard_index
        return self._array

    def _advance_shard(self) -> None:
        self.shard_index += 1
        self.offset = 0
        self._array_index = None
        self._array = None
        if self.shard_index >= len(self.shards):
            self.shard_index = 0
            self.epochs += 1


class TokenShardLoader:
    """Read fixed-length sequences from raw little-endian uint32 shards.

    This is the first Mini-K3 data gate: it validates the on-disk shard contract
    and checkpoint semantics without coupling the experiment to core datasets.
    Sources are consumed in manifest order for now; weighted source sampling is
    a later training-loader concern.
    """

    def __init__(
        self,
        manifest: dict[str, Any],
        tokens_dir: str | Path,
        *,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        if world_size <= 0:
            raise ValueError(f"world_size must be > 0, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must satisfy 0 <= rank < world_size, got {rank}")

        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.tokens_dir = Path(tokens_dir)

        specs = _parse_manifest(manifest, self.tokens_dir)
        self.assigned_shards: dict[str, list[Path]] = {}
        self.assigned_token_counts: dict[str, int] = {}
        self._source_names: list[str] = []
        self._cursors: dict[str, _SourceCursor] = {}
        for spec in specs:
            assigned = spec.shards[rank::world_size]
            if not assigned:
                raise ValueError(
                    f"source {spec.name!r} has {len(spec.shards)} shards but "
                    f"world_size is {world_size}; rank {rank} would get no data"
                )
            assigned_token_count = sum(_uint32_token_count(path) for path in assigned)
            if assigned_token_count < seq_len:
                raise ValueError(
                    f"source {spec.name!r} rank {rank} assignment has fewer tokens "
                    f"than seq_len ({assigned_token_count} < {seq_len})"
                )
            self.assigned_shards[spec.name] = assigned
            self.assigned_token_counts[spec.name] = assigned_token_count
            self._source_names.append(spec.name)
            self._cursors[spec.name] = _SourceCursor(spec.name, assigned)

        self._source_index = 0
        self.tokens_emitted = 0
        self.tokens_by_source = {name: 0 for name in self._source_names}

    def next_sequence(self) -> np.ndarray:
        name = self._source_names[self._source_index]
        self._source_index = (self._source_index + 1) % len(self._source_names)
        tokens = self._cursors[name].take(self.seq_len)
        self.tokens_emitted += self.seq_len
        self.tokens_by_source[name] += self.seq_len
        return tokens

    @property
    def assigned_token_count(self) -> int:
        return sum(self.assigned_token_counts.values())

    def state_dict(self) -> dict[str, Any]:
        return {
            "seq_len": self.seq_len,
            "rank": self.rank,
            "world_size": self.world_size,
            "source_index": self._source_index,
            "tokens_emitted": self.tokens_emitted,
            "tokens_by_source": dict(self.tokens_by_source),
            "sources": {
                name: self._cursors[name].state_dict() for name in self._source_names
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["seq_len"]) != self.seq_len:
            raise ValueError("checkpoint seq_len does not match loader seq_len")
        if (
            int(state["rank"]) != self.rank
            or int(state["world_size"]) != self.world_size
        ):
            raise ValueError("checkpoint rank/world_size does not match loader")
        source_index = int(state["source_index"])
        if source_index < 0 or source_index >= len(self._source_names):
            raise ValueError("checkpoint source_index is out of range")

        checkpoint_sources = set(state["sources"])
        current_sources = set(self._source_names)
        if checkpoint_sources != current_sources:
            raise ValueError(
                "checkpoint sources do not match loader sources: "
                f"{sorted(checkpoint_sources)} != {sorted(current_sources)}"
            )

        self._source_index = source_index
        self.tokens_emitted = int(state["tokens_emitted"])
        self.tokens_by_source = {
            name: int(value) for name, value in state["tokens_by_source"].items()
        }
        for name, cursor_state in state["sources"].items():
            self._cursors[name].load_state_dict(cursor_state)


def _parse_manifest(manifest: dict[str, Any], tokens_dir: Path) -> list[_SourceSpec]:
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("manifest must define at least one source")

    specs: list[_SourceSpec] = []
    for name, source_data in sources.items():
        if not isinstance(source_data, dict):
            raise ValueError(f"source {name!r} must be an object")
        shard_names = source_data.get("shards")
        if not isinstance(shard_names, list) or not shard_names:
            raise ValueError(f"source {name!r} has no shards")
        shards = []
        for shard_name in shard_names:
            if not isinstance(shard_name, str) or not shard_name:
                raise ValueError(f"source {name!r} has an invalid shard entry")
            path = tokens_dir / name / shard_name
            if not path.is_file():
                raise FileNotFoundError(f"manifest shard does not exist: {path}")
            shards.append(path)
        specs.append(_SourceSpec(name=name, shards=shards))
    return specs


def _uint32_token_count(path: Path) -> int:
    size = path.stat().st_size
    if size % np.dtype("<u4").itemsize != 0:
        raise ValueError(f"shard {path} size is not divisible by uint32 width")
    return size // np.dtype("<u4").itemsize
