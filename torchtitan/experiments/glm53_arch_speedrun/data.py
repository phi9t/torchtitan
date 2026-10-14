# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FineWeb GPT-2 nanogpt-bin dataloader for the GLM-5.3 speedrun harness."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.experiments.falcon.bin_reader import read_nanogpt_bin


def resolve_bin_paths(pattern: str) -> list[Path]:
    """Resolve a literal path or glob, including absolute globs."""
    literal = Path(pattern)
    if literal.is_file():
        return [literal]
    if literal.is_absolute():
        anchor = literal.anchor
        rel = str(literal.relative_to(anchor))
        return sorted(Path(anchor).glob(rel))
    return sorted(Path().glob(pattern))


class FineWebBinDataLoader(BaseDataLoader):
    """DP-aware contiguous-window loader over nanogpt ``.bin`` shards.

    Each data-parallel rank consumes a disjoint contiguous window from the same
    shard cursor. Labels are the next-token shift of the input body; they never
    wrap around the end of the local row or shard.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        bin_glob: str
        vocab_size: int = 50257
        repeat: bool = True

        def build(
            self,
            *,
            dp_world_size: int,
            dp_rank: int,
            tokenizer,
            max_context_length: int,
            num_tokens_per_microbatch: int,
            **kwargs: Any,
        ) -> "FineWebBinDataLoader":
            del tokenizer, kwargs
            if num_tokens_per_microbatch % max_context_length != 0:
                raise ValueError(
                    "num_tokens_per_microbatch must be divisible by "
                    "max_context_length for FineWeb bin batches"
                )
            local_batch_size = num_tokens_per_microbatch // max_context_length
            files = resolve_bin_paths(self.bin_glob)
            if not files:
                raise ValueError(f"no nanogpt bins matched glob {self.bin_glob!r}")
            return FineWebBinDataLoader(
                files,
                seq_len=max_context_length,
                local_batch_size=local_batch_size,
                vocab_size=self.vocab_size,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                repeat=self.repeat,
            )

    def __init__(
        self,
        shard_paths: list[Path],
        *,
        seq_len: int,
        local_batch_size: int,
        vocab_size: int,
        dp_world_size: int,
        dp_rank: int,
        repeat: bool,
    ) -> None:
        if not shard_paths:
            raise ValueError("shard_paths must be non-empty")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if local_batch_size < 1:
            raise ValueError("local_batch_size must be >= 1")
        if dp_world_size < 1:
            raise ValueError("dp_world_size must be >= 1")
        if not 0 <= dp_rank < dp_world_size:
            raise ValueError(f"dp_rank must be in [0, {dp_world_size}), got {dp_rank}")
        self.shard_paths = [Path(path) for path in shard_paths]
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.vocab_size = vocab_size
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.repeat = repeat
        self._rank_stride = local_batch_size * seq_len
        self._rank_window = self._rank_stride + 1
        self._global_stride = self._rank_stride * dp_world_size
        self._global_window = self._global_stride + 1

    def _iter_shards(self) -> Iterator[torch.Tensor]:
        while True:
            for path in self.shard_paths:
                yield read_nanogpt_bin(path)
            if not self.repeat:
                return

    def __iter__(self) -> Iterator[TokenizedTrainingMicrobatch]:
        shard_iter = self._iter_shards()
        try:
            tokens = next(shard_iter)
        except StopIteration:
            return
        pos = 0
        positions = torch.arange(self.seq_len).expand(self.local_batch_size, -1)
        while True:
            if pos + self._global_window > tokens.numel():
                try:
                    tokens = next(shard_iter)
                except StopIteration:
                    return
                pos = 0
                if self._global_window > tokens.numel():
                    raise ValueError(
                        f"shard has {tokens.numel()} tokens, too few for one "
                        f"DP window of {self._global_window}"
                    )

            rank_start = pos + self.dp_rank * self._rank_stride
            buf = tokens[rank_start : rank_start + self._rank_window]
            if buf.max().item() >= self.vocab_size or buf.min().item() < 0:
                raise ValueError(
                    "token id outside vocab range "
                    f"[0, {self.vocab_size}) in nanogpt shard"
                )
            body = buf[:-1].view(self.local_batch_size, self.seq_len)
            labels = buf[1:].view(self.local_batch_size, self.seq_len)
            pos += self._global_stride
            input_ids = body.clone()
            labels = labels.clone()
            yield TokenizedTrainingMicrobatch(
                input=input_ids,
                labels=labels,
                positions=positions,
                padding_mask=torch.zeros_like(input_ids, dtype=torch.bool),
                loss_token_counts=torch.tensor(labels.numel(), dtype=torch.long),
                routing_token_counts=torch.tensor(
                    labels.numel(), dtype=torch.long
                ).view(1),
            )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict
