# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixed repeating sequence bank for the Falcon overfit gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.components.dataloader import BaseDataLoader


def make_overfit_bank(
    *,
    num_sequences: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> torch.Tensor:
    """Build ``num_sequences`` distinct token rows of length ``seq_len + 1``.

    The extra token is the next-token label for the last input position.
    """
    if num_sequences <= 0:
        raise ValueError("num_sequences must be > 0")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if vocab_size < 2:
        raise ValueError("vocab_size must be >= 2")

    generator = torch.Generator().manual_seed(seed)
    rows: list[torch.Tensor] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    while len(rows) < num_sequences:
        attempts += 1
        if attempts > num_sequences * 64:
            raise RuntimeError("failed to sample a distinct overfit bank")
        row = torch.randint(
            0,
            vocab_size,
            (seq_len + 1,),
            generator=generator,
        )
        key = tuple(row.tolist())
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return torch.stack(rows, dim=0)


class FalconRepeatDataLoader(BaseDataLoader):
    """Cycle one fixed token bank so every optimizer step is one epoch."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        num_sequences: int = 8
        vocab_size: int = 32
        seed: int = 0

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
        ) -> "FalconRepeatDataLoader":
            del tokenizer, snapshot_every_n_steps, kwargs
            if dp_world_size != 1 or dp_rank != 0:
                raise ValueError("Falcon overfit dataloader is single-rank only")
            bank = make_overfit_bank(
                num_sequences=self.num_sequences,
                seq_len=seq_len,
                vocab_size=self.vocab_size,
                seed=self.seed,
            )
            return FalconRepeatDataLoader(bank, local_batch_size=local_batch_size)

    def __init__(self, bank: torch.Tensor, *, local_batch_size: int) -> None:
        if bank.ndim != 2 or bank.shape[0] == 0 or bank.shape[1] < 2:
            raise ValueError("bank must have shape (num_sequences, seq_len+1)")
        if local_batch_size != bank.shape[0]:
            raise ValueError(
                "local_batch_size must equal the overfit bank size so every "
                "step is one epoch over the fixed set"
            )
        self.bank = bank.long()
        self.local_batch_size = local_batch_size
        self.seq_len = int(bank.shape[1] - 1)

    def __iter__(self):
        input_ids = self.bank[:, :-1]
        labels = self.bank[:, 1:]
        positions = torch.arange(self.seq_len).expand(self.local_batch_size, -1)
        batch = {"input": input_ids, "positions": positions}
        while True:
            yield batch, labels

    def state_dict(self) -> dict[str, Any]:
        return {"bank": self.bank.clone()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.bank = state_dict["bank"].long()
