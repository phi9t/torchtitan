# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mini Kimi K3 router primitives."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


DEFAULT_BIAS_UPDATE_RATE = 1.0e-2


class NoAuxTcRouter(nn.Module):
    """Training-capable noaux_tc top-k router.

    The correction bias affects expert choice only. Expert weights are gathered
    from the original, unbiased scores, matching the first-party Mini-K3 train
    path.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int,
        routed_scaling_factor: float = 1.0,
        activation: str = "sigmoid",
        num_expert_group: int = 1,
        topk_group: int = 1,
        moe_renormalize: bool = True,
        bias_update_rate: float = DEFAULT_BIAS_UPDATE_RATE,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if num_experts_per_token <= 0:
            raise ValueError("num_experts_per_token must be > 0")
        if num_experts_per_token > num_experts:
            raise ValueError("num_experts_per_token must be <= num_experts")
        if activation not in {"sigmoid", "softmax"}:
            raise ValueError("noaux_tc router activation must be sigmoid or softmax")
        if num_expert_group <= 0:
            raise ValueError("num_expert_group must be > 0")
        if topk_group <= 0:
            raise ValueError("topk_group must be > 0")
        if topk_group > num_expert_group:
            raise ValueError("topk_group must be <= num_expert_group")
        if num_experts % num_expert_group != 0:
            raise ValueError("num_experts must be divisible by num_expert_group")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.routed_scaling_factor = routed_scaling_factor
        self.activation = activation
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.moe_renormalize = moe_renormalize
        self.bias_update_rate = bias_update_rate

        self.weight = nn.Parameter(torch.empty((num_experts, hidden_size)))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts))
        self.e_score_correction_bias.requires_grad_(False)
        self.register_buffer(
            "expert_load",
            torch.zeros(num_experts),
            persistent=False,
        )
        self.register_buffer(
            "tokens_seen",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.last_entropy: float | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.e_score_correction_bias.zero_()

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states last dimension must match router hidden_size "
                f"({hidden_states.shape[-1]} != {self.hidden_size})"
            )

        flat_hidden = hidden_states.reshape(-1, self.hidden_size)
        logits = F.linear(
            flat_hidden.to(torch.float32),
            self.weight.to(torch.float32),
            None,
        )
        if self.activation == "sigmoid":
            scores = logits.sigmoid()
        else:
            scores = logits.softmax(dim=1)

        biased_scores = scores + self.e_score_correction_bias.unsqueeze(0)
        choice_scores = self._apply_group_limit(biased_scores)
        _, topk_idx = torch.topk(
            choice_scores,
            k=self.num_experts_per_token,
            dim=-1,
            sorted=False,
        )
        topk_weight = scores.gather(1, topk_idx)
        if self.num_experts_per_token > 1 and self.moe_renormalize:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * self.routed_scaling_factor

        if self.training:
            self._record_load(topk_idx, scores)
        return topk_idx, topk_weight

    def _apply_group_limit(self, scores: torch.Tensor) -> torch.Tensor:
        if self.num_expert_group <= 1 or self.num_expert_group <= self.topk_group:
            return scores

        batch = scores.shape[0]
        grouped = scores.view(batch, self.num_expert_group, -1)
        group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(
            group_scores,
            k=self.topk_group,
            dim=-1,
            sorted=False,
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        expert_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                batch, self.num_expert_group, self.num_experts // self.num_expert_group
            )
            .reshape(batch, self.num_experts)
            .bool()
        )
        return scores.masked_fill(~expert_mask, float("-inf"))

    @torch.no_grad()
    def _record_load(self, topk_idx: torch.Tensor, scores: torch.Tensor) -> None:
        counts = torch.bincount(topk_idx.reshape(-1), minlength=self.num_experts)
        self.expert_load += counts.to(self.expert_load.dtype)
        self.tokens_seen += topk_idx.shape[0]

        p = scores.float().mean(dim=0)
        p = p / (p.sum() + 1e-20)
        entropy = -(p * (p + 1e-20).log()).sum()
        self.last_entropy = float(entropy / math.log(self.num_experts))

    @torch.no_grad()
    def balancer_step(self) -> dict[str, float | bool]:
        load = self.expert_load
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(load, op=torch.distributed.ReduceOp.SUM)

        total = load.sum()
        if total <= 0:
            return {"updated": False}

        mean = load.mean()
        update = self.bias_update_rate * torch.sign(mean - load)
        self.e_score_correction_bias.add_(update.to(self.e_score_correction_bias.dtype))

        frac = load / total
        dead_expert_frac = float((load == 0).float().mean())
        load_imbalance = float(load.max() / (mean + 1e-20))
        max_expert_frac = float(frac.max())
        router_entropy = self.last_entropy

        self.expert_load.zero_()
        self.tokens_seen.zero_()
        return {
            "updated": True,
            "dead_expert_frac": dead_expert_frac,
            "load_imbalance": load_imbalance,
            "max_expert_frac": max_expert_frac,
            "router_entropy": router_entropy,
        }
