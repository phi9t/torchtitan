# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Variable-digit addition task for the Falcon transfer gate (paper section 5.2).

Protocol:

- Prompt ``digits_n(a) + digits_n(b) =`` where ``a`` and ``b`` are ``n``-digit
  operands.
- Target: the reversed ``(n + 1)``-digit sum, least-significant digit first.
  Reversing makes generation left-to-right causal in carry order, which is the
  paper's setup.
- Loss is computed on the answer suffix only; the prompt is context.

A character-level vocabulary is used deliberately: GPT-2 BPE fragments digit
runs unpredictably, so a char encoder keeps digit positions aligned. Train
widths are uniform in ``{1..N}`` (in-distribution) and OOD is ``{N+1..M}``.
This ticket locks ``N=16, M=24`` and declares that as the OOD gap.

Legend for tensors below: B batch, T sequence (prompt + answer).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

# Fixed character alphabet: digits, plus, and equals. Index 0 is a pad symbol
# so collated batches have a distinguishable padding id.
_PAD = "\x00"
_ALPHABET = _PAD + "0123456789+="


class AdditionVocab:
    """Character-level vocabulary for the addition task."""

    def __init__(self) -> None:
        self._stoi = {ch: i for i, ch in enumerate(_ALPHABET)}
        self._itos = {i: ch for ch, i in self._stoi.items()}

    @property
    def pad_id(self) -> int:
        return self._stoi[_PAD]

    @property
    def size(self) -> int:
        return len(self._stoi)

    def encode(self, text: str) -> list[int]:
        return [self._stoi[ch] for ch in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self._itos[int(i)] for i in ids)


def reversed_sum_digits(a: int, b: int, *, width: int) -> list[int]:
    """Return the ``a + b`` digits, least-significant first, over ``width+1``.

    ``width`` is the operand digit count ``n``; the sum needs at most ``n + 1``
    digits, so the answer is zero-padded on the most-significant end to exactly
    ``width + 1`` digits before reversing.
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    total = a + b
    text = str(total).rjust(width + 1, "0")
    # LSD first == reverse of the MSD-first string.
    return [int(ch) for ch in text[::-1]]


@dataclass(frozen=True)
class AdditionExample:
    a: int
    b: int
    width: int
    tokens: list[int]
    loss_mask: list[int]


def encode_addition(
    a: int, b: int, *, width: int, vocab: AdditionVocab
) -> AdditionExample:
    """Encode one addition example with a suffix-only loss mask."""
    prompt = f"{a}+{b}="
    answer_digits = reversed_sum_digits(a, b, width=width)
    answer = "".join(str(d) for d in answer_digits)
    tokens = vocab.encode(prompt + answer)
    # Mask marks the answer positions only; the prompt is context.
    loss_mask = [0] * len(vocab.encode(prompt)) + [1] * len(answer)
    return AdditionExample(a=a, b=b, width=width, tokens=tokens, loss_mask=loss_mask)


def _sample_operand(width: int, rng: random.Random) -> int:
    """Sample an exactly ``width``-digit operand (no leading zero for width>1)."""
    if width == 1:
        return rng.randint(0, 9)
    low = 10 ** (width - 1)
    high = 10**width - 1
    return rng.randint(low, high)


@dataclass
class AdditionDataset:
    n: int
    m: int
    id_examples: list[AdditionExample]
    ood_examples: list[AdditionExample]

    def collate(
        self, examples: list[AdditionExample], *, vocab: AdditionVocab
    ) -> dict[str, torch.Tensor]:
        """Right-pad examples into input/labels/loss_mask tensors.

        Labels are the next-token shift; loss_mask marks answer positions in the
        label frame. Padding positions are masked out.
        """
        max_len = max(len(ex.tokens) for ex in examples)
        pad = vocab.pad_id
        inputs = []
        labels = []
        masks = []
        for ex in examples:
            toks = ex.tokens
            # input predicts labels[t] from input[t]; standard shift.
            inp = toks[:-1]
            lbl = toks[1:]
            # loss_mask in the label frame: mask entry i is on if label i is an
            # answer token (loss_mask[i+1] in the token frame).
            m = ex.loss_mask[1:]
            pad_len = max_len - 1 - len(inp)
            inputs.append(inp + [pad] * pad_len)
            labels.append(lbl + [pad] * pad_len)
            masks.append([bool(x) for x in m] + [False] * pad_len)
        return {
            "input": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_mask": torch.tensor(masks, dtype=torch.bool),
        }


def build_addition_dataset(
    *,
    n: int,
    m: int,
    num_id: int,
    num_ood: int,
    seed: int,
) -> AdditionDataset:
    """Build an ID/OOD addition split. ID widths ``{1..n}``, OOD ``{n+1..m}``."""
    if not (1 <= n < m):
        raise ValueError("require 1 <= n < m for a declared OOD gap")
    rng = random.Random(seed)
    vocab = AdditionVocab()

    def draw(widths: range, count: int) -> list[AdditionExample]:
        out: list[AdditionExample] = []
        widths = list(widths)
        for _ in range(count):
            width = rng.choice(widths)
            a = _sample_operand(width, rng)
            b = _sample_operand(width, rng)
            out.append(encode_addition(a, b, width=width, vocab=vocab))
        return out

    id_examples = draw(range(1, n + 1), num_id)
    ood_examples = draw(range(n + 1, m + 1), num_ood)
    return AdditionDataset(n=n, m=m, id_examples=id_examples, ood_examples=ood_examples)
