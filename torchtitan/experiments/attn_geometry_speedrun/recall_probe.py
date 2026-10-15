# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import Tensor


BOS_ID = 50256


@dataclass(frozen=True)
class RecallExample:
    input_ids: Tensor
    target_id: int
    retrieval_distance: int


@dataclass(frozen=True)
class RecallDistanceResult:
    distance: int
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


def make_recall_example(
    *,
    distance: int,
    vocab_size: int,
    generator: torch.Generator,
    sequence_multiple: int = 16,
    bos_id: int = BOS_ID,
) -> RecallExample:
    """Create one fresh key->value recall prompt.

    The target value appears once after the key, then the same key is queried
    after exactly `distance` intervening token positions from the value token.
    A small prefix pads the total length to the attention kernel multiple
    without changing the retrieval distance.
    """

    if distance < 1:
        raise ValueError("distance must be positive")
    if vocab_size <= bos_id + 2:
        raise ValueError("vocab_size must include normal GPT-2 token ids")
    prefix_len = (-(distance + 3)) % sequence_multiple
    sample = torch.randint(0, vocab_size - 1, (distance + 1,), generator=generator)
    key = int(sample[0])
    value = int(sample[1])
    filler = sample[2:].clone()
    filler[filler == key] = (key + 1) % (vocab_size - 1)
    filler[filler == value] = (value + 1) % (vocab_size - 1)
    prefix = torch.full((prefix_len,), bos_id, dtype=torch.long)
    input_ids = torch.cat(
        (
            prefix,
            torch.tensor([bos_id, key, value], dtype=torch.long),
            filler.to(dtype=torch.long),
            torch.tensor([key], dtype=torch.long),
        )
    )
    actual_distance = input_ids.numel() - 1 - (prefix_len + 2)
    if actual_distance != distance:
        raise AssertionError(
            f"constructed distance {actual_distance}, expected {distance}"
        )
    if input_ids.numel() % sequence_multiple != 0:
        raise AssertionError("recall prompt length is not kernel-aligned")
    return RecallExample(
        input_ids=input_ids,
        target_id=value,
        retrieval_distance=distance,
    )


def evaluate_recall_distances(
    *,
    logits_fn: Callable[[Tensor], Tensor],
    distances: Iterable[int],
    samples_per_distance: int,
    vocab_size: int,
    seed: int,
) -> list[RecallDistanceResult]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    results: list[RecallDistanceResult] = []
    for distance in distances:
        correct = 0
        for _ in range(samples_per_distance):
            example = make_recall_example(
                distance=distance,
                vocab_size=vocab_size,
                generator=generator,
            )
            logits = logits_fn(example.input_ids)
            prediction = int(logits[-1].argmax().item())
            correct += int(prediction == example.target_id)
        results.append(
            RecallDistanceResult(
                distance=distance,
                correct=correct,
                total=samples_per_distance,
            )
        )
    return results
