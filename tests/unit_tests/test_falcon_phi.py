# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Falcon ``phi`` (QK-normalization) knob (arm A5).

``phi`` selects the query/key feature map used before the fast-weight write:

- ``rms``: QK-RMSNorm (paper default, arms A2/A3/A4).
- ``l2``: QK-L2 normalization (paper 1A.1 vs 1A.3, arm A5).

The recurrent kernel already accepts ``normalize_qk``; these tests pin the
``phi`` selection threaded from ``FalconConfig`` through ``FalconMixer`` into
the kernel, and the L2 branch of the kernel itself.
"""

from __future__ import annotations

import torch

from torchtitan.experiments.falcon.falcon import falcon_recurrent_forward
from torchtitan.experiments.falcon.model import FalconForCausalLM, tiny_falcon_config


def _forward(config):
    torch.manual_seed(0)
    model = FalconForCausalLM(config)
    tokens = torch.arange(config.seq_len).view(1, -1) % config.vocab_size
    return model(tokens)


def test_falcon_config_phi_defaults_to_rms():
    config = tiny_falcon_config()
    assert config.phi == "rms"


def test_kernel_phi_l2_normalizes_keys_to_unit_norm():
    # With phi="l2" the per-head write feature phi(k) must have unit L2 norm.
    # Use same_step so phi(k_0) is written at t=0, then read it straight back
    # with a matching query to recover the value scaled by <q_hat, k_hat> = 1.
    q_BLNK = torch.tensor([[[3.0, 4.0]]]).view(1, 1, 1, 2)
    k_BLNK = torch.tensor([[[3.0, 4.0]]]).view(1, 1, 1, 2)
    v_BLNV = torch.tensor([[[2.0, 5.0]]]).view(1, 1, 1, 2)
    beta_BLN = torch.ones(1, 1, 1)
    lambda_BLN = torch.zeros(1, 1, 1)

    out_BLNV, _ = falcon_recurrent_forward(
        q_BLNK,
        k_BLNK,
        v_BLNV,
        beta_BLN,
        lambda_BLN,
        variant="falcon1a",
        alignment="same_step",
        phi="l2",
        eps=0.0,
    )

    # phi(k) = k / ||k|| = (0.6, 0.8); q_hat . k_hat = 1, so eta = beta/1 = 1
    # and o_0 = <q_hat, k_hat> * v_0 = v_0.
    torch.testing.assert_close(out_BLNV[:, 0], v_BLNV[:, 0])


def test_kernel_phi_rms_and_l2_differ():
    torch.manual_seed(3)
    q_BLNK = torch.randn(2, 5, 3, 4)
    k_BLNK = torch.randn(2, 5, 3, 4)
    v_BLNV = torch.randn(2, 5, 3, 4)
    beta_BLN = 2.0 * torch.sigmoid(torch.randn(2, 5, 3))
    lambda_BLN = torch.nn.functional.softplus(torch.randn(2, 5, 3))

    out_rms, _ = falcon_recurrent_forward(
        q_BLNK, k_BLNK, v_BLNV, beta_BLN, lambda_BLN, phi="rms"
    )
    out_l2, _ = falcon_recurrent_forward(
        q_BLNK, k_BLNK, v_BLNV, beta_BLN, lambda_BLN, phi="l2"
    )
    assert not torch.allclose(out_rms, out_l2)


def test_kernel_normalize_qk_false_maps_to_no_phi():
    # normalize_qk=False must still behave as the identity feature map, so the
    # kernel identity tests that pass normalize_qk=False keep their meaning.
    q_BLNK = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]).view(1, 2, 1, 2)
    k_BLNK = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]).view(1, 2, 1, 2)
    v_BLNV = torch.tensor([[[9.0, 9.0], [2.0, 3.0]]]).view(1, 2, 1, 2)
    beta_BLN = torch.ones(1, 2, 1)
    lambda_BLN = torch.zeros(1, 2, 1)

    out_flag, _ = falcon_recurrent_forward(
        q_BLNK,
        k_BLNK,
        v_BLNV,
        beta_BLN,
        lambda_BLN,
        normalize_qk=False,
        eps=0.0,
    )
    out_none, _ = falcon_recurrent_forward(
        q_BLNK,
        k_BLNK,
        v_BLNV,
        beta_BLN,
        lambda_BLN,
        phi="none",
        eps=0.0,
    )
    torch.testing.assert_close(out_flag, out_none)


def test_falcon_mixer_threads_phi_to_the_kernel():
    # phi="l2" must change the model output relative to the rms default; a
    # silently ignored knob would make A5 identical to A2.
    torch.manual_seed(0)
    rms = tiny_falcon_config()
    l2 = tiny_falcon_config()
    l2.phi = "l2"

    torch.manual_seed(0)
    rms_model = FalconForCausalLM(rms)
    torch.manual_seed(0)
    l2_model = FalconForCausalLM(l2)

    tokens = torch.arange(rms.seq_len).view(1, -1) % rms.vocab_size
    assert not torch.allclose(rms_model(tokens), l2_model(tokens))


def test_falcon_science_default_phi_is_rms():
    from torchtitan.experiments.falcon.config_registry import falcon_science

    config = falcon_science()
    assert config.model_spec.model.config.phi == "rms"
