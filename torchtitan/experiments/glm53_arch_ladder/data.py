# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FineWeb GPT-2 data views for the architecture ladder.

The ladder reuses the S01/S05 speedrun shard location and nanogpt-bin resolver,
but decoder-family models consume packed token streams while the GLM experiment
model consumes ``[B, L]`` batches.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.experiments.falcon.bin_reader import read_nanogpt_bin
from torchtitan.experiments.glm53_arch_speedrun.data import resolve_bin_paths


def _validate_window(
    tokens: torch.Tensor,
    *,
    vocab_size: int,
    required_tokens: int,
    source: str,
) -> torch.Tensor:
    if tokens.numel() < required_tokens:
        raise ValueError(
            f"{source} has {tokens.numel()} tokens, too few for "
            f"{required_tokens} required tokens"
        )
    bank = tokens[:required_tokens]
    if bank.max().item() >= vocab_size or bank.min().item() < 0:
        raise ValueError(
            "token id outside vocab range " f"[0, {vocab_size}) in nanogpt shard"
        )
    return bank


def _clone_microbatch(
    batch: TokenizedTrainingMicrobatch,
) -> TokenizedTrainingMicrobatch:
    return TokenizedTrainingMicrobatch(
        input=batch.input.clone(),
        labels=batch.labels.clone(),
        positions=batch.positions.clone(),
        padding_mask=batch.padding_mask.clone(),
        loss_token_counts=batch.loss_token_counts.clone(),
        routing_token_counts=batch.routing_token_counts.clone(),
    )


class PackedFineWebBinDataLoader(BaseDataLoader):
    """DP-aware contiguous-window loader returning packed decoder tensors."""

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
        ) -> "PackedFineWebBinDataLoader":
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
            return PackedFineWebBinDataLoader(
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
        positions = torch.arange(self.seq_len).repeat(self.local_batch_size)
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
            body = buf[:-1].clone()
            labels = buf[1:].clone()
            pos += self._global_stride
            yield TokenizedTrainingMicrobatch(
                input=body,
                labels=labels,
                positions=positions,
                padding_mask=torch.zeros_like(body, dtype=torch.bool),
                loss_token_counts=torch.tensor(labels.numel(), dtype=torch.long),
                routing_token_counts=torch.tensor(
                    labels.numel(), dtype=torch.long
                ).view(1),
            )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict


class FixedBankPackedFineWebBinDataLoader(BaseDataLoader):
    """Repeated fixed FineWeb bank returning packed decoder tensors."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        bin_glob: str
        vocab_size: int = 50257
        bank_microbatches: int = 1
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
        ) -> "FixedBankPackedFineWebBinDataLoader":
            del tokenizer, kwargs
            if not self.repeat:
                raise ValueError("fixed-bank loader requires repeat=True")
            if num_tokens_per_microbatch % max_context_length != 0:
                raise ValueError(
                    "num_tokens_per_microbatch must be divisible by "
                    "max_context_length for FineWeb bin batches"
                )
            local_batch_size = num_tokens_per_microbatch // max_context_length
            files = resolve_bin_paths(self.bin_glob)
            if not files:
                raise ValueError(f"no nanogpt bins matched glob {self.bin_glob!r}")
            return FixedBankPackedFineWebBinDataLoader(
                files,
                seq_len=max_context_length,
                local_batch_size=local_batch_size,
                vocab_size=self.vocab_size,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                bank_microbatches=self.bank_microbatches,
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
        bank_microbatches: int,
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
        if bank_microbatches < 1:
            raise ValueError("bank_microbatches must be >= 1")

        self.shard_paths = [Path(path) for path in shard_paths]
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.vocab_size = vocab_size
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.bank_microbatches = bank_microbatches
        rank_stride = local_batch_size * seq_len
        global_stride = rank_stride * dp_world_size
        required_tokens = bank_microbatches * global_stride + 1
        tokens = _validate_window(
            read_nanogpt_bin(self.shard_paths[0]),
            vocab_size=vocab_size,
            required_tokens=required_tokens,
            source=str(self.shard_paths[0]),
        )
        positions = torch.arange(seq_len).repeat(local_batch_size)
        self._batches: list[TokenizedTrainingMicrobatch] = []
        for bank_idx in range(bank_microbatches):
            rank_start = bank_idx * global_stride + dp_rank * rank_stride
            buf = tokens[rank_start : rank_start + rank_stride + 1]
            body = buf[:-1].clone()
            labels = buf[1:].clone()
            self._batches.append(
                TokenizedTrainingMicrobatch(
                    input=body,
                    labels=labels,
                    positions=positions,
                    padding_mask=torch.zeros_like(body, dtype=torch.bool),
                    loss_token_counts=torch.tensor(labels.numel(), dtype=torch.long),
                    routing_token_counts=torch.tensor(
                        labels.numel(), dtype=torch.long
                    ).view(1),
                )
            )

    def __iter__(self) -> Iterator[TokenizedTrainingMicrobatch]:
        while True:
            for batch in self._batches:
                yield _clone_microbatch(batch)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict


class FixedBankFineWebBinDataLoader(BaseDataLoader):
    """Repeated fixed FineWeb bank returning ``[B, L]`` GLM tensors."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        bin_glob: str
        vocab_size: int = 50257
        bank_microbatches: int = 1
        repeat: bool = True
        num_mtp_layers: int = 0

        def build(
            self,
            *,
            dp_world_size: int,
            dp_rank: int,
            tokenizer,
            max_context_length: int,
            num_tokens_per_microbatch: int,
            **kwargs: Any,
        ) -> "FixedBankFineWebBinDataLoader":
            del tokenizer, kwargs
            if not self.repeat:
                raise ValueError("fixed-bank loader requires repeat=True")
            if num_tokens_per_microbatch % max_context_length != 0:
                raise ValueError(
                    "num_tokens_per_microbatch must be divisible by "
                    "max_context_length for FineWeb bin batches"
                )
            if self.num_mtp_layers != 0:
                raise ValueError("fixed-bank GLM loader does not support MTP labels")
            local_batch_size = num_tokens_per_microbatch // max_context_length
            files = resolve_bin_paths(self.bin_glob)
            if not files:
                raise ValueError(f"no nanogpt bins matched glob {self.bin_glob!r}")
            return FixedBankFineWebBinDataLoader(
                files,
                seq_len=max_context_length,
                local_batch_size=local_batch_size,
                vocab_size=self.vocab_size,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                bank_microbatches=self.bank_microbatches,
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
        bank_microbatches: int,
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
        if bank_microbatches < 1:
            raise ValueError("bank_microbatches must be >= 1")

        self.shard_paths = [Path(path) for path in shard_paths]
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.vocab_size = vocab_size
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.bank_microbatches = bank_microbatches
        rank_stride = local_batch_size * seq_len
        global_stride = rank_stride * dp_world_size
        required_tokens = bank_microbatches * global_stride + 1
        tokens = _validate_window(
            read_nanogpt_bin(self.shard_paths[0]),
            vocab_size=vocab_size,
            required_tokens=required_tokens,
            source=str(self.shard_paths[0]),
        )
        positions = torch.arange(seq_len).expand(local_batch_size, -1).clone()
        self._batches: list[TokenizedTrainingMicrobatch] = []
        for bank_idx in range(bank_microbatches):
            rank_start = bank_idx * global_stride + dp_rank * rank_stride
            buf = tokens[rank_start : rank_start + rank_stride + 1]
            body = buf[:-1].view(local_batch_size, seq_len).clone()
            labels = buf[1:].view(local_batch_size, seq_len).clone()
            self._batches.append(
                TokenizedTrainingMicrobatch(
                    input=body,
                    labels=labels,
                    positions=positions,
                    padding_mask=torch.zeros_like(body, dtype=torch.bool),
                    loss_token_counts=torch.tensor(labels.numel(), dtype=torch.long),
                    routing_token_counts=torch.tensor(
                        labels.numel(), dtype=torch.long
                    ).view(1),
                )
            )

    def __iter__(self) -> Iterator[TokenizedTrainingMicrobatch]:
        while True:
            for batch in self._batches:
                yield _clone_microbatch(batch)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict
