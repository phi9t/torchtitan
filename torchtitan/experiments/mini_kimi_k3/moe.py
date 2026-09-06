# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mini Kimi K3 sparse MoE primitives."""

from __future__ import annotations

import torch
from torch import nn

from torchtitan.experiments.mini_kimi_k3.activation import SituAndMul
from torchtitan.experiments.mini_kimi_k3.mla import MiniK3RMSNorm
from torchtitan.experiments.mini_kimi_k3.router import NoAuxTcRouter


class MiniK3BlockSparseMLP(nn.Module):
    """Gate/up/down expert MLP with Mini-K3's SITU activation."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        activation_beta: float = 4.0,
        activation_linear_beta: float | None = 25.0,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = SituAndMul(
            beta=activation_beta,
            linear_beta=activation_linear_beta,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat([self.w1(hidden_states), self.w3(hidden_states)], dim=-1)
        return self.w2(self.act_fn(gate_up))


class MiniK3DenseMLP(nn.Module):
    """Dense/shared Mini-K3 MLP with first-party projection names."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        activation_beta: float = 4.0,
        activation_linear_beta: float | None = 25.0,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SituAndMul(
            beta=activation_beta,
            linear_beta=activation_linear_beta,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat(
            [self.gate_proj(hidden_states), self.up_proj(hidden_states)], dim=-1
        )
        return self.down_proj(self.act_fn(gate_up))


class MiniK3SparseMoeBlock(nn.Module):
    """Differentiable sparse expert dispatch using the noaux_tc router."""

    def __init__(
        self,
        *,
        router: NoAuxTcRouter,
        experts: list[nn.Module],
        shared_experts: nn.Module | None = None,
        hidden_size: int | None = None,
        routed_expert_hidden_size: int | None = None,
        latent_moe_use_norm: bool = False,
    ) -> None:
        super().__init__()
        if len(experts) != router.num_experts:
            raise ValueError(
                "number of experts must match router.num_experts "
                f"({len(experts)} != {router.num_experts})"
            )
        self.router = router
        self.experts = nn.ModuleList(experts)
        self.shared_experts = shared_experts
        self.routed_expert_down_proj: nn.Linear | None = None
        self.routed_expert_up_proj: nn.Linear | None = None
        self.routed_expert_norm: MiniK3RMSNorm | None = None
        if routed_expert_hidden_size is not None:
            if hidden_size is None:
                raise ValueError("hidden_size is required for latent MoE")
            if hidden_size <= 0:
                raise ValueError("hidden_size must be > 0")
            if routed_expert_hidden_size <= 0:
                raise ValueError("routed_expert_hidden_size must be > 0")
            if routed_expert_hidden_size != hidden_size:
                self.routed_expert_down_proj = nn.Linear(
                    hidden_size,
                    routed_expert_hidden_size,
                    bias=False,
                )
                self.routed_expert_up_proj = nn.Linear(
                    routed_expert_hidden_size,
                    hidden_size,
                    bias=False,
                )
                self.routed_expert_norm = (
                    MiniK3RMSNorm(routed_expert_hidden_size)
                    if latent_moe_use_norm
                    else None
                )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        original_shape = hidden_states.shape
        topk_idx, topk_weight = self.router(hidden_states)
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        if self.routed_expert_down_proj is not None:
            flat_hidden = self.routed_expert_down_proj(flat_hidden)

        num_tokens, top_k = topk_idx.shape
        flat_expert_idx = topk_idx.reshape(-1)
        order = flat_expert_idx.argsort()
        sorted_tokens = flat_hidden[order // top_k]
        counts = torch.bincount(flat_expert_idx, minlength=len(self.experts))

        chunks: list[torch.Tensor] = []
        start = 0
        for expert_idx, count in enumerate(counts.tolist()):
            if count == 0:
                continue
            end = start + count
            chunks.append(self.experts[expert_idx](sorted_tokens[start:end]))
            start = end
        if not chunks:
            raise RuntimeError("MoE router selected no experts")

        sorted_outputs = torch.cat(chunks, dim=0)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        routed = sorted_outputs[inverse]

        routed = (
            routed.view(num_tokens, top_k, -1).type(topk_weight.dtype)
            * topk_weight.unsqueeze(-1)
        ).sum(dim=1)
        if self.routed_expert_up_proj is not None:
            if self.routed_expert_norm is not None:
                routed = self.routed_expert_norm(routed)
            routed = routed.to(self.routed_expert_up_proj.weight.dtype)
            routed = self.routed_expert_up_proj(routed)
        routed = routed.type(hidden_states.dtype).view(*original_shape)
        if self.shared_experts is not None:
            routed = routed + self.shared_experts(identity)
        return routed
