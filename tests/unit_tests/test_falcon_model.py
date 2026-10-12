# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from torchtitan.experiments.falcon.model import FalconForCausalLM, tiny_falcon_config
from torchtitan.experiments.falcon.repeat_data import (
    FalconRepeatDataLoader,
    make_overfit_bank,
)


def test_tiny_falcon_forward_returns_vocab_logits():
    config = tiny_falcon_config()
    model = FalconForCausalLM(config)
    tokens = torch.arange(config.seq_len).view(1, -1) % config.vocab_size

    logits = model(tokens)

    assert logits.shape == (1, config.seq_len, config.vocab_size)
    assert torch.isfinite(logits).all()


def test_overfit_bank_is_distinct_and_in_vocab():
    bank = make_overfit_bank(num_sequences=8, seq_len=16, vocab_size=32, seed=0)

    assert bank.shape == (8, 17)
    assert bank.min() >= 0
    assert bank.max() < 32
    assert len({tuple(row.tolist()) for row in bank}) == 8


def test_repeat_dataloader_cycles_the_same_bank_and_shifts_labels():
    bank = make_overfit_bank(num_sequences=4, seq_len=8, vocab_size=16, seed=1)
    loader = FalconRepeatDataLoader(bank, local_batch_size=4)
    first = next(iter(loader))
    second = next(iter(loader))

    batch, labels = first
    batch_again, labels_again = second
    assert torch.equal(batch["input"], bank[:, :-1])
    assert torch.equal(labels, bank[:, 1:])
    assert torch.equal(batch["input"], batch_again["input"])
    assert torch.equal(labels, labels_again)


def test_repeat_dataloader_labels_are_a_shift_not_a_circular_roll():
    # Labels must be bank[:, 1:] (next-token shift), never torch.roll, which
    # would wrap the first token in as the label for the last position.
    bank = make_overfit_bank(num_sequences=3, seq_len=6, vocab_size=16, seed=2)
    loader = FalconRepeatDataLoader(bank, local_batch_size=3)
    batch, labels = next(iter(loader))

    assert torch.equal(labels, bank[:, 1:])
    # A circular roll of the inputs would place input[:, 0] at labels[:, -1].
    rolled = torch.roll(batch["input"], shifts=-1, dims=1)
    assert not torch.equal(labels, rolled)
    # The last label is the appended next-token target, not the wrapped head.
    assert torch.equal(labels[:, -1], bank[:, -1])


def test_falcon_config_alignment_defaults_to_delayed():
    config = tiny_falcon_config()
    assert config.alignment == "delayed"


def test_falcon_mixer_threads_same_step_alignment_to_the_kernel():
    # Same-step alignment must change the model output relative to delayed;
    # if the mixer silently ignored the switch the two would be identical.
    torch.manual_seed(0)
    delayed = tiny_falcon_config()
    same_step = tiny_falcon_config()
    same_step.alignment = "same_step"

    torch.manual_seed(0)
    delayed_model = FalconForCausalLM(delayed)
    torch.manual_seed(0)
    same_step_model = FalconForCausalLM(same_step)

    tokens = torch.arange(delayed.seq_len).view(1, -1) % delayed.vocab_size
    delayed_logits = delayed_model(tokens)
    same_step_logits = same_step_model(tokens)

    assert not torch.allclose(delayed_logits, same_step_logits)
