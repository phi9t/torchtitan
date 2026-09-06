# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for the Trainer meta-init path of the Falcon model.

The core Trainer builds the model on ``torch.device("meta")`` and then calls
``to_empty()`` before ``init_states``/``init_weights``. ``to_empty`` leaves
parameter storage uninitialized (zeroed on this host), so any parameter whose
value is set in ``__init__`` (rather than re-set by the init pass) is silently
lost. ``FalconRMSNorm.weight`` is initialized to ones in ``__init__``; if the
init pass does not restore it, every norm output is zeroed and the whole mixer
path collapses to a constant loss that is identical across arms. This module
guards against that regression.
"""

from __future__ import annotations

import torch
from torch import nn

from torchtitan.experiments.falcon.model import (
    FalconForCausalLM,
    FalconRMSNorm,
    tiny_falcon_config,
)
from torchtitan.experiments.falcon.overfit import _initialize_weights


def _to_empty_like_trainer(model: nn.Module) -> None:
    """Simulate the Trainer's ``to_empty`` on an uninitialized (zeroed) buffer.

    ``to_empty`` reallocates storage without copying values; here we make the
    lost-value case explicit and deterministic by zeroing every parameter,
    which is what this host observes after ``to_empty`` from meta.
    """
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()


def test_initialize_weights_restores_rmsnorm_gain_after_to_empty():
    config = tiny_falcon_config()
    model = FalconForCausalLM(config)
    _to_empty_like_trainer(model)

    # After to_empty every RMSNorm gain is zeroed; without a fix the init pass
    # leaves them at zero, which zeros every norm output.
    for module in model.modules():
        if isinstance(module, FalconRMSNorm):
            assert bool((module.weight == 0).all())

    _initialize_weights(model)

    for module in model.modules():
        if isinstance(module, FalconRMSNorm):
            assert torch.equal(module.weight, torch.ones_like(module.weight))


def test_falcon_forward_is_nonconstant_after_to_empty_then_init():
    # The failure symptom: zeroed norm gains make the mixer output identically
    # zero, so logits depend only on the (untrained) lm_head and the loss is a
    # constant that cannot distinguish arms. After the fix the mixer must inject
    # a token-dependent, nonzero contribution.
    config = tiny_falcon_config()
    model = FalconForCausalLM(config)
    _to_empty_like_trainer(model)
    _initialize_weights(model)

    tokens = torch.arange(config.seq_len).view(1, -1) % config.vocab_size
    with torch.no_grad():
        mixer_in = model.layers[0].input_norm(model.embed_tokens(tokens))
        assert float(mixer_in.abs().mean()) > 0.0
        mixer_out = model.layers[0].mixer(mixer_in)
        assert float(mixer_out.abs().mean()) > 0.0
