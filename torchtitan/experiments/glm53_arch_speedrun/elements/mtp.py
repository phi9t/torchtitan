# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One-layer multi-token-prediction head for arm A6."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MTPHead(nn.Module):
    """Predict token t + 2 from hidden state t with a weighted auxiliary CE."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        aux_loss_weight: float,
    ) -> None:
        super().__init__()
        if aux_loss_weight < 0.0:
            raise ValueError("aux_loss_weight must be non-negative.")
        self.aux_loss_weight = aux_loss_weight
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_BLD: torch.Tensor) -> torch.Tensor:
        return self.lm_head(F.silu(self.proj(self.norm(hidden_BLD))))

    def aux_loss(
        self,
        hidden_BLD: torch.Tensor,
        labels_BL: torch.Tensor,
        *,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        logits_BLV = self(hidden_BLD[:, :-2])
        targets_BL = labels_BL[:, 2:]
        if targets_BL.numel() == 0:
            return hidden_BLD.new_zeros(())
        loss = F.cross_entropy(
            logits_BLV.reshape(-1, logits_BLV.shape[-1]).float(),
            targets_BL.reshape(-1),
            ignore_index=ignore_index,
        )
        return loss * self.aux_loss_weight
