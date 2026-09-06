# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mini Kimi K3 activation modules."""

from __future__ import annotations

import torch
from torch import nn


class SituAndMul(nn.Module):
    """SITU gate/up activation from the first-party Mini-K3 reference."""

    def __init__(self, beta: float = 1.0, linear_beta: float | None = None) -> None:
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                "SITU input last dimension must split evenly into gate and up halves"
            )

        split_dim = x.shape[-1] // 2
        gate = x[..., :split_dim].to(torch.float32)
        up = x[..., split_dim:].to(torch.float32)
        situ_gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_gate * up).to(x.dtype)
