# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small integration helpers for the S01 speedrun baseline model."""

from __future__ import annotations

import torch
from torch import nn

from .attention import TopKSparseCausalAttention
from .config import SpeedrunElementsConfig
from .kda import KDALayer
from .mhc import MHCResidualBlock
from .mtp import MTPHead


def build_attention(layer_idx: int, config: SpeedrunElementsConfig) -> nn.Module:
    """Build the attention element for one speedrun layer."""

    config.validate()
    if config.attention_kind(layer_idx) == "kda":
        return KDALayer(config)
    return TopKSparseCausalAttention(config)


def maybe_wrap_mhc(
    branch: nn.Module,
    config: SpeedrunElementsConfig,
) -> nn.Module:
    if not config.mhc_enabled:
        return branch
    return MHCResidualBlock(
        config.hidden_size,
        branch,
        num_streams=config.mhc_streams,
        sinkhorn_iters=config.mhc_sinkhorn_iters,
        eps=config.mhc_eps,
        identity_at_init=config.mhc_init == "gated_identity",
    )


def maybe_build_mtp(config: SpeedrunElementsConfig) -> MTPHead | None:
    if not config.mtp_enabled:
        return None
    return MTPHead(
        config.hidden_size,
        config.vocab_size,
        aux_loss_weight=config.mtp_aux_loss_weight,
    )


def collect_element_aux_losses(module: nn.Module) -> torch.Tensor | None:
    """Sum cached element auxiliary losses exposed by element adapters."""

    losses = [
        loss
        for child in module.modules()
        for loss in (getattr(child, "last_indexer_aux_loss", None),)
        if loss is not None
    ]
    if not losses:
        return None
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss.to(device=total.device, dtype=total.dtype)
    return total
