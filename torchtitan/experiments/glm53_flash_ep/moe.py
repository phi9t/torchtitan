# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-compatible MoE adapter backed by TorchTitan common EP dispatch.

Shape suffixes:
    B = batch, L = sequence length, T = flattened tokens, D = model dimension,
    E = experts, F = expert hidden dimension, K = top-k experts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.experiments.glm53_flash.flavors import Glm53FlashConfig
from torchtitan.experiments.glm53_flash.model import GlmMLP
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher


class GlmEPRouter(nn.Module):
    """GLM sigmoid top-k router with persistent correction bias."""

    def __init__(self, config: Glm53FlashConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "tokens_per_expert_E",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=False,
        )

    def reset_tokens_per_expert(self, *, device: torch.device | None = None) -> None:
        if device is None:
            device = self.tokens_per_expert_E.device
        self.tokens_per_expert_E = torch.zeros(
            self.num_experts,
            dtype=torch.float32,
            device=device,
        )

    def forward(
        self,
        x_TD: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_TD.ndim != 2 or x_TD.shape[-1] != self.hidden_dim:
            raise ValueError(
                "GLM EP router expects [tokens, hidden] input, got "
                f"{tuple(x_TD.shape)}."
            )
        if padding_mask_T is not None:
            if padding_mask_T.dtype != torch.bool:
                raise ValueError(
                    "padding_mask_T must have dtype bool, "
                    f"got {padding_mask_T.dtype}."
                )
            if padding_mask_T.shape != (x_TD.shape[0],):
                raise ValueError(
                    "padding_mask_T must have shape [tokens], got "
                    f"{tuple(padding_mask_T.shape)} for {tuple(x_TD.shape)}."
                )

        scores_TE = F.linear(x_TD.float(), self.weight.float()).sigmoid()
        scores_for_choice_TE = scores_TE + self.e_score_correction_bias
        topk_expert_ids_TK = torch.topk(
            scores_for_choice_TE, k=self.top_k, dim=-1, sorted=False
        ).indices
        topk_scores_TK = scores_TE.gather(1, topk_expert_ids_TK)
        if self.norm_topk_prob:
            topk_scores_TK = topk_scores_TK / (
                topk_scores_TK.sum(dim=-1, keepdim=True) + 1.0e-20
            )
        topk_scores_TK = topk_scores_TK * self.routed_scaling_factor

        routing_map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
            -1,
            topk_expert_ids_TK,
            True,
        )
        if self.training:
            masked_routing_map_TE = (
                routing_map_TE
                if padding_mask_T is None
                else routing_map_TE & ~padding_mask_T.unsqueeze(-1)
            )
            with torch.no_grad():
                self.tokens_per_expert_E.add_(masked_routing_map_TE.sum(dim=0))
        return topk_scores_TK, topk_expert_ids_TK, routing_map_TE


class GlmClampedSwiGLU:
    """GLM SwiGLU clamp semantics for common routed experts."""

    @dataclass(kw_only=True, slots=True)
    class Config:
        swiglu_limit: float

        def build(self) -> "GlmClampedSwiGLU":
            return GlmClampedSwiGLU(self)

    def __init__(self, config: Config) -> None:
        self.swiglu_limit = config.swiglu_limit

    def __call__(
        self,
        gate_up_R2F: torch.Tensor,
        *,
        offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del offsets
        gate_RF, up_RF = gate_up_R2F.unbind(-2)
        return F.silu(gate_RF.clamp(max=self.swiglu_limit)) * up_RF.clamp(
            min=-self.swiglu_limit,
            max=self.swiglu_limit,
        )


class GlmEPExperts(RoutedExperts):
    """Common routed experts with GLM dimensions and clamp behavior."""

    def __init__(self, config: Glm53FlashConfig) -> None:
        routed_config = RoutedExperts.Config(
            w13=GroupedLinear.Config(
                group_size=config.num_experts,
                in_features=config.hidden_size,
                out_features=config.moe_intermediate_size,
                num_linears=2,
            ),
            w2=GroupedLinear.Config(
                group_size=config.num_experts,
                in_features=config.moe_intermediate_size,
                out_features=config.hidden_size,
            ),
            token_dispatcher=AllToAllTokenDispatcher.Config(
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
            ),
            activation_fn=GlmClampedSwiGLU.Config(
                swiglu_limit=config.swiglu_limit,
            ),
        )
        super().__init__(routed_config)


class GlmEPMoE(nn.Module):
    """GLM MoE wrapper using common token dispatch for routed experts."""

    def __init__(self, config: Glm53FlashConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.experts = GlmEPExperts(config)
        self.gate = GlmEPRouter(config)
        self.shared_experts = GlmMLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            prefix="shared_experts",
        )

    @property
    def routed_experts(self) -> GlmEPExperts:
        return self.experts

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        original_shape = x_BLD.shape
        x_TD = x_BLD.view(-1, original_shape[-1])
        topk_scores_TK, topk_expert_ids_TK, routing_map_TE = self.gate(x_TD)
        num_local_tokens_per_expert_E = routing_map_TE.sum(dim=0)
        routed_TD = self.experts(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        return routed_TD.view(original_shape) + self.shared_experts(x_BLD)
