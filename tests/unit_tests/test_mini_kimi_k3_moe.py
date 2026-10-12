# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import nn

from torchtitan.experiments.mini_kimi_k3.moe import (
    MiniK3BlockSparseMLP,
    MiniK3DenseMLP,
    MiniK3SparseMoeBlock,
)
from torchtitan.experiments.mini_kimi_k3.router import NoAuxTcRouter


class _FixedRouter(nn.Module):
    num_experts = 1

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_count = hidden_states.numel() // hidden_states.shape[-1]
        return (
            torch.zeros(token_count, 1, dtype=torch.long, device=hidden_states.device),
            torch.ones(
                token_count, 1, dtype=hidden_states.dtype, device=hidden_states.device
            ),
        )


def _diagonal_linear(scale: float) -> nn.Linear:
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(2) * scale)
    return layer


def test_sparse_moe_dispatches_tokens_to_selected_experts_and_adds_shared_expert():
    router = NoAuxTcRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_token=1,
    )
    with torch.no_grad():
        router.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    block = MiniK3SparseMoeBlock(
        router=router,
        experts=[_diagonal_linear(2.0), _diagonal_linear(3.0)],
        shared_experts=_diagonal_linear(5.0),
    )
    x = torch.tensor([[[4.0, 0.0], [0.0, 6.0]]])

    actual = block(x)

    topk_idx, topk_weight = router(x)
    assert topk_idx.tolist() == [[0], [1]]
    expected = torch.tensor(
        [
            [
                [8.0 * topk_weight[0, 0].item() + 20.0, 0.0],
                [0.0, 18.0 * topk_weight[1, 0].item() + 30.0],
            ]
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_sparse_moe_uses_weighted_sum_for_multi_expert_routing():
    router = NoAuxTcRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_token=2,
    )
    with torch.no_grad():
        router.weight.zero_()

    block = MiniK3SparseMoeBlock(
        router=router,
        experts=[_diagonal_linear(2.0), _diagonal_linear(4.0)],
    )
    x = torch.tensor([[[3.0, 5.0]]])

    actual = block(x)

    expected = torch.tensor([[[9.0, 15.0]]])
    torch.testing.assert_close(actual, expected)


def test_sparse_moe_keeps_expert_dispatch_differentiable():
    router = NoAuxTcRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_token=1,
    )
    with torch.no_grad():
        router.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    selected = _diagonal_linear(2.0)
    unselected = _diagonal_linear(3.0)
    block = MiniK3SparseMoeBlock(router=router, experts=[selected, unselected])
    x = torch.tensor([[[4.0, 0.0]]], requires_grad=True)

    block(x).sum().backward()

    assert x.grad is not None
    assert selected.weight.grad is not None
    assert selected.weight.grad.abs().sum() > 0
    assert unselected.weight.grad is None


def test_block_sparse_mlp_uses_situ_gate_up_projection_shape():
    mlp = MiniK3BlockSparseMLP(hidden_size=2, intermediate_size=3)
    x = torch.randn(2, 4, 2)

    out = mlp(x)

    assert out.shape == x.shape


def test_dense_mlp_uses_first_party_projection_names():
    mlp = MiniK3DenseMLP(hidden_size=2, intermediate_size=3)
    x = torch.randn(2, 4, 2)

    out = mlp(x)

    assert out.shape == x.shape
    assert mlp.gate_proj.weight.shape == (3, 2)
    assert mlp.up_proj.weight.shape == (3, 2)
    assert mlp.down_proj.weight.shape == (2, 3)


def test_sparse_moe_supports_latent_routed_expert_width():
    router = NoAuxTcRouter(
        hidden_size=4,
        num_experts=2,
        num_experts_per_token=1,
    )
    with torch.no_grad():
        router.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            )
        )

    block = MiniK3SparseMoeBlock(
        router=router,
        experts=[
            MiniK3BlockSparseMLP(hidden_size=2, intermediate_size=3),
            MiniK3BlockSparseMLP(hidden_size=2, intermediate_size=3),
        ],
        hidden_size=4,
        routed_expert_hidden_size=2,
        latent_moe_use_norm=True,
    )
    x = torch.randn(2, 3, 4)

    out = block(x)
    out.sum().backward()

    assert out.shape == x.shape
    assert block.routed_expert_down_proj.weight.shape == (2, 4)
    assert block.routed_expert_up_proj.weight.shape == (4, 2)
    assert block.routed_expert_norm is not None


def test_sparse_moe_applies_latent_norm_after_expert_dispatch():
    expert = _diagonal_linear(3.0)
    block = MiniK3SparseMoeBlock(
        router=_FixedRouter(),
        experts=[expert],
        hidden_size=4,
        routed_expert_hidden_size=2,
        latent_moe_use_norm=True,
    )
    with torch.no_grad():
        assert block.routed_expert_down_proj is not None
        assert block.routed_expert_up_proj is not None
        block.routed_expert_down_proj.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        )
        block.routed_expert_up_proj.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        )

    x = torch.tensor([[[3.0, 4.0, 100.0, 200.0]]])

    actual = block(x)

    routed_input = block.routed_expert_down_proj(x.reshape(-1, 4))
    expert_out = expert(routed_input)
    assert block.routed_expert_norm is not None
    expected = block.routed_expert_up_proj(block.routed_expert_norm(expert_out))
    expected = expected.view_as(x)
    torch.testing.assert_close(actual, expected)


def test_sparse_moe_casts_latent_routed_output_to_up_projection_dtype():
    router = NoAuxTcRouter(
        hidden_size=4,
        num_experts=1,
        num_experts_per_token=1,
    )
    block = MiniK3SparseMoeBlock(
        router=router,
        experts=[MiniK3BlockSparseMLP(hidden_size=2, intermediate_size=3)],
        hidden_size=4,
        routed_expert_hidden_size=2,
        latent_moe_use_norm=True,
    ).to(dtype=torch.bfloat16)
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)

    out = block(x)

    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape
