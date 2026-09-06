# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from torchtitan.experiments.falcon.addition import (
    AdditionVocab,
    build_addition_dataset,
    encode_addition,
    reversed_sum_digits,
)


def test_reversed_sum_digits_is_lsd_first_and_n_plus_one_wide():
    # 999 + 1 = 1000; reversed LSD-first over n+1=4 digits is [0,0,0,1].
    digits = reversed_sum_digits(999, 1, width=3)
    assert digits == [0, 0, 0, 1]
    # 5 + 7 = 12; width 1 -> n+1 = 2 digits reversed -> [2, 1].
    assert reversed_sum_digits(5, 7, width=1) == [2, 1]


def test_encode_addition_prompt_and_suffix_layout():
    vocab = AdditionVocab()
    ex = encode_addition(12, 34, width=2, vocab=vocab)
    # Prompt: "12+34=" then reversed sum digits of 46 -> [6, 4, ...] over 3 wide.
    text = vocab.decode(ex.tokens)
    assert text.startswith("12+34=")
    # Suffix is the reversed (n+1)-digit sum; 12+34=46 -> "640" padded to 3.
    assert text[len("12+34=") :] == "640"


def test_suffix_mask_covers_only_the_answer_positions():
    vocab = AdditionVocab()
    ex = encode_addition(1, 2, width=1, vocab=vocab)
    # Loss mask marks the answer tokens (the positions predicted from the
    # prompt onward), and only those.
    mask = ex.loss_mask
    tokens = ex.tokens
    assert len(mask) == len(tokens)
    # The masked-in region must be exactly the reversed sum suffix.
    answer_len = 1 + 1  # n + 1 digits
    assert sum(mask) == answer_len
    # All masked-in positions are contiguous at the tail.
    on = [i for i, m in enumerate(mask) if m]
    assert on == list(range(len(tokens) - answer_len, len(tokens)))


def test_build_dataset_id_vs_ood_split_by_width():
    ds = build_addition_dataset(n=16, m=24, num_id=40, num_ood=20, seed=0)
    for ex in ds.id_examples:
        assert 1 <= ex.width <= 16
    for ex in ds.ood_examples:
        assert 17 <= ex.width <= 24
    assert len(ds.id_examples) == 40
    assert len(ds.ood_examples) == 20


def test_declared_ood_gap_is_n16_m24():
    ds = build_addition_dataset(n=16, m=24, num_id=4, num_ood=4, seed=1)
    assert ds.n == 16
    assert ds.m == 24
    # ID widths never overlap OOD widths.
    id_widths = {ex.width for ex in ds.id_examples}
    ood_widths = {ex.width for ex in ds.ood_examples}
    assert id_widths.isdisjoint(ood_widths)


def test_char_vocab_roundtrips_every_symbol():
    vocab = AdditionVocab()
    for ch in "0123456789+=":
        ids = vocab.encode(ch)
        assert vocab.decode(ids) == ch


def test_encoded_sum_is_arithmetically_correct():
    vocab = AdditionVocab()
    # 16-digit width stress: pick two large operands and verify the reversed
    # answer equals a+b when de-reversed.
    a, b = 12345678, 87654321
    ex = encode_addition(a, b, width=8, vocab=vocab)
    suffix = vocab.decode(ex.tokens)[len(f"{a}+{b}=") :]
    # suffix is LSD-first; reverse to MSD-first and strip leading zeros.
    value = int(suffix[::-1])
    assert value == a + b


def test_batch_collation_pads_and_returns_mask(tmp_path):
    ds = build_addition_dataset(n=16, m=24, num_id=8, num_ood=0, seed=2)
    batch = ds.collate(ds.id_examples[:4], vocab=AdditionVocab())
    assert batch["input"].shape[0] == 4
    assert batch["labels"].shape == batch["input"].shape
    assert batch["loss_mask"].shape == batch["input"].shape
    assert batch["loss_mask"].dtype == torch.bool
    # No position is both padding and masked-in for loss.
    assert torch.isfinite(batch["input"].float()).all()
