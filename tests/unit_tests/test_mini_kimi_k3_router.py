# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch

from torchtitan.experiments.mini_kimi_k3.router import NoAuxTcRouter


def test_noaux_tc_router_uses_bias_for_selection_and_unbiased_scores_for_weights():
    router = NoAuxTcRouter(
        hidden_size=3,
        num_experts=4,
        num_experts_per_token=2,
        bias_update_rate=0.1,
    )
    with torch.no_grad():
        router.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                ]
            )
        )
        router.e_score_correction_bias.copy_(torch.tensor([0.0, 0.3, 0.0, 0.95]))

    hidden_states = torch.tensor([[[4.0, 0.0, 0.0]]])

    topk_idx, topk_weight = router(hidden_states)

    assert topk_idx.shape == (1, 2)
    assert set(topk_idx[0].tolist()) == {0, 3}
    unbiased_scores = torch.sigmoid(hidden_states.reshape(1, 3) @ router.weight.T)
    expected = unbiased_scores.gather(1, topk_idx)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(topk_weight, expected)


def test_noaux_tc_router_records_training_load_and_updates_non_grad_bias():
    router = NoAuxTcRouter(
        hidden_size=2,
        num_experts=4,
        num_experts_per_token=1,
        bias_update_rate=0.25,
    )
    with torch.no_grad():
        router.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                ]
            )
        )

    assert router.e_score_correction_bias.requires_grad is False
    persistent_buffers = dict(router.named_buffers())
    assert "expert_load" in persistent_buffers
    assert "tokens_seen" in persistent_buffers
    assert "expert_load" not in router.state_dict()
    assert "tokens_seen" not in router.state_dict()

    router.train()
    router(torch.tensor([[[2.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 5.0]]]))
    torch.testing.assert_close(router.expert_load, torch.tensor([2.0, 2.0, 0.0, 0.0]))
    assert router.tokens_seen.item() == 4
    assert router.last_entropy is not None

    stats = router.balancer_step()

    assert stats["updated"] is True
    assert stats["dead_expert_frac"] == 0.5
    torch.testing.assert_close(
        router.e_score_correction_bias,
        torch.tensor([-0.25, -0.25, 0.25, 0.25]),
    )
    torch.testing.assert_close(router.expert_load, torch.zeros(4))
    assert router.tokens_seen.item() == 0


def test_noaux_tc_router_group_limited_selection_masks_unselected_groups():
    router = NoAuxTcRouter(
        hidden_size=4,
        num_experts=4,
        num_experts_per_token=1,
        num_expert_group=2,
        topk_group=1,
    )
    with torch.no_grad():
        router.weight.copy_(
            torch.tensor(
                [
                    [8.0, 0.0, 0.0, 0.0],
                    [7.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
        router.e_score_correction_bias.copy_(torch.tensor([0.0, 0.0, 5.0, 5.0]))

    topk_idx, _ = router(torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]))

    assert topk_idx.item() in {2, 3}


def test_noaux_tc_router_rejects_unsupported_or_invalid_config():
    with pytest.raises(ValueError, match="num_experts must be divisible"):
        NoAuxTcRouter(
            hidden_size=2,
            num_experts=5,
            num_experts_per_token=1,
            num_expert_group=2,
            topk_group=1,
        )

    with pytest.raises(ValueError, match="activation"):
        NoAuxTcRouter(
            hidden_size=2,
            num_experts=4,
            num_experts_per_token=1,
            activation="gelu",
        )
