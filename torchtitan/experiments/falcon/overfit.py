# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mercor-style overfit gate for the tiny Falcon LM."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.experiments.falcon.falcon import FalconVariant
from torchtitan.experiments.falcon.model import FalconForCausalLM, tiny_falcon_config
from torchtitan.experiments.falcon.repeat_data import (
    FalconRepeatDataLoader,
    make_overfit_bank,
)


def _initialize_weights(model: nn.Module, *, std: float = 0.02) -> None:
    from torchtitan.experiments.falcon.model import FalconRMSNorm

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, FalconRMSNorm):
            # RMSNorm gains are set to ones in __init__, but the Trainer builds
            # on ``meta`` then ``to_empty`` (uninitialized/zeroed) before calling
            # init. Without re-setting the gain here it stays at the garbage
            # value (zeros on this host), which zeros every norm output and
            # collapses the whole mixer path to a constant loss. Restore ones.
            nn.init.ones_(module.weight)


def run_overfit(
    *,
    variant: FalconVariant = "falcon1a",
    steps: int = 80,
    lr: float = 3.0e-2,
    seed: int = 0,
    num_sequences: int = 8,
    seq_len: int = 16,
    vocab_size: int = 32,
) -> dict[str, Any]:
    """Train on one repeating bank. Pass if CE collapses.

    Every step sees the full bank (one epoch). This is the LM analog of
    Mercor's 32-task synchronous overfit run: if loss does not drop, the
    mixer or harness is broken.
    """
    if steps < 0:
        raise ValueError("steps must be >= 0")
    torch.manual_seed(seed)
    config = tiny_falcon_config(variant=variant)
    config.vocab_size = vocab_size
    config.seq_len = seq_len
    model = FalconForCausalLM(config)
    _initialize_weights(model)
    bank = make_overfit_bank(
        num_sequences=num_sequences,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=seed,
    )
    loader = FalconRepeatDataLoader(bank, local_batch_size=num_sequences)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses: list[float] = []
    batch_iter = iter(loader)
    model.train()
    for _ in range(max(steps, 1)):
        batch, labels = next(batch_iter)
        logits = model(batch["input"])
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            labels.reshape(-1),
        )
        losses.append(float(loss.detach()))
        if steps == 0:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if lr == 0.0:
            # Keep a second evaluation so the report has initial vs final.
            continue

    if steps > 0:
        batch, labels = next(batch_iter)
        with torch.no_grad():
            logits = model(batch["input"])
            final = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                labels.reshape(-1),
            )
        losses.append(float(final))

    initial_loss = losses[0]
    final_loss = losses[-1]
    passed = final_loss < 0.5 * initial_loss and final_loss < 1.0
    return {
        "status": "pass" if passed else "fail",
        "variant": variant,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "losses": losses,
        "steps": steps,
        "num_sequences": num_sequences,
        "lr": lr,
    }
