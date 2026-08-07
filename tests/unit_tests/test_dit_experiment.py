# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.experiments.dit.model import DiTBlock, DiTModel, patchify, unpatchify


def test_dit_patchify_unpatchify_round_trip():
    x_BCHW = torch.randn(2, 4, 16, 16)
    patches_BTD = patchify(x_BCHW, patch_size=2)

    assert patches_BTD.shape == (2, 64, 16)
    torch.testing.assert_close(
        unpatchify(
            patches_BTD,
            latent_channels=4,
            latent_height=16,
            latent_width=16,
            patch_size=2,
        ),
        x_BCHW,
    )


def test_dit_block_zero_gates_are_initially_residual_safe():
    block = DiTBlock.Config(hidden_size=32, num_heads=4).build()
    block.init_states()

    x_BTD = torch.randn(2, 5, 32)
    c_BD = torch.randn(2, 32)

    torch.testing.assert_close(block(x_BTD, c_BD), x_BTD)


def test_dit_model_initial_output_shape_and_zero_output():
    model = DiTModel.Config(
        latent_channels=4,
        latent_height=16,
        latent_width=16,
        patch_size=2,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_classes=10,
    ).build()
    model.init_states()

    x_BCHW = torch.randn(2, 4, 16, 16)
    t_B = torch.tensor([0, 999], dtype=torch.long)
    y_B = torch.tensor([1, 3], dtype=torch.long)

    out_BCHW = model(x_BCHW, t_B, y_B)

    assert out_BCHW.shape == x_BCHW.shape
    torch.testing.assert_close(out_BCHW, torch.zeros_like(out_BCHW))
