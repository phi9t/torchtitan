# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch

from torchtitan.experiments.falcon.model import FalconForCausalLM, tiny_falcon_config


def _forward(config):
    torch.manual_seed(0)
    model = FalconForCausalLM(config)
    tokens = torch.arange(config.seq_len).view(1, -1) % config.vocab_size
    return model(tokens)


def test_mixer_defaults_to_falcon():
    config = tiny_falcon_config()
    assert config.mixer == "falcon"


def test_falcon_mixer_shape_and_finiteness():
    config = tiny_falcon_config()
    config.mixer = "falcon"
    logits = _forward(config)
    assert logits.shape == (1, config.seq_len, config.vocab_size)
    assert torch.isfinite(logits).all()


def test_gdn_mixer_shape_and_finiteness():
    config = tiny_falcon_config()
    config.mixer = "gdn"
    logits = _forward(config)
    assert logits.shape == (1, config.seq_len, config.vocab_size)
    assert torch.isfinite(logits).all()


def test_softmax_mixer_shape_and_finiteness():
    config = tiny_falcon_config()
    config.mixer = "softmax"
    logits = _forward(config)
    assert logits.shape == (1, config.seq_len, config.vocab_size)
    assert torch.isfinite(logits).all()


def test_mixers_produce_distinct_outputs():
    # The three mixers are genuinely different math; if any two collapsed to
    # the same values the switch would be a silent no-op for that arm.
    falcon = tiny_falcon_config()
    falcon.mixer = "falcon"
    gdn = tiny_falcon_config()
    gdn.mixer = "gdn"
    softmax = tiny_falcon_config()
    softmax.mixer = "softmax"

    out_falcon = _forward(falcon)
    out_gdn = _forward(gdn)
    out_softmax = _forward(softmax)

    assert not torch.allclose(out_falcon, out_gdn)
    assert not torch.allclose(out_falcon, out_softmax)
    assert not torch.allclose(out_gdn, out_softmax)


def test_unknown_mixer_raises():
    config = tiny_falcon_config()
    config.mixer = "linear"
    with pytest.raises(ValueError, match="mixer"):
        FalconForCausalLM(config)


def test_gdn_mixer_is_causal():
    # A future token must not change an earlier position's output. This guards
    # the same-step GDN write against leaking information backward in time.
    config = tiny_falcon_config()
    config.mixer = "gdn"
    torch.manual_seed(0)
    model = FalconForCausalLM(config)
    model.eval()

    tokens = torch.arange(config.seq_len).view(1, -1) % config.vocab_size
    with torch.no_grad():
        base = model(tokens)
        perturbed = tokens.clone()
        perturbed[0, -1] = (perturbed[0, -1] + 1) % config.vocab_size
        changed = model(perturbed)

    # Positions before the last must be identical; only the final position sees
    # the changed token.
    torch.testing.assert_close(base[:, :-1], changed[:, :-1])


def test_softmax_mixer_is_causal():
    config = tiny_falcon_config()
    config.mixer = "softmax"
    torch.manual_seed(0)
    model = FalconForCausalLM(config)
    model.eval()

    tokens = torch.arange(config.seq_len).view(1, -1) % config.vocab_size
    with torch.no_grad():
        base = model(tokens)
        perturbed = tokens.clone()
        perturbed[0, -1] = (perturbed[0, -1] + 1) % config.vocab_size
        changed = model(perturbed)

    torch.testing.assert_close(base[:, :-1], changed[:, :-1])


def test_falcon_mixer_still_threads_alignment():
    # The Falcon mixer must keep honoring the alignment switch after the mixer
    # field is introduced (guards against a refactor regression).
    delayed = tiny_falcon_config()
    delayed.mixer = "falcon"
    delayed.alignment = "delayed"
    same_step = tiny_falcon_config()
    same_step.mixer = "falcon"
    same_step.alignment = "same_step"

    assert not torch.allclose(_forward(delayed), _forward(same_step))
