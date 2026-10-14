# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Manifold hyper-connection residual modules for arm A1.

Shape suffixes:
    B = batch, L = sequence length, N = mHC stream, D = model dimension,
    W = HC mix width.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.distributed.local_compile import local_compile


def expand_streams(x_BLD: torch.Tensor, *, num_streams: int = 4) -> torch.Tensor:
    return x_BLD.unsqueeze(-2).expand(*x_BLD.shape[:-1], num_streams, x_BLD.shape[-1])


def collapse_streams(x_BLND: torch.Tensor) -> torch.Tensor:
    return x_BLND.mean(dim=-2)


def _unweighted_rms_norm(x: torch.Tensor, *, eps: float) -> torch.Tensor:
    x_float = x.float()
    out = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    return out.to(x.dtype)


@local_compile("glm53_speedrun_mhc_pre", batch_invariant=True)
def mhc_pre_tensor_program(
    x_BLND: torch.Tensor,
    hc_fn_WD: torch.Tensor,
    hc_base_W: torch.Tensor,
    hc_scale_3: torch.Tensor,
    *,
    num_streams: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_dtype = x_BLND.dtype
    flat_BLD = _unweighted_rms_norm(
        x_BLND.flatten(-2),
        eps=norm_eps,
    )
    mixes_BLW = F.linear(flat_BLD.float(), hc_fn_WD.float())
    n = num_streams
    pre_BLN, post_BLN, comb_BLNN = mixes_BLW.split([n, n, n * n], dim=-1)
    comb_BLNN = comb_BLNN.unflatten(-1, (n, n))
    pre_base_N, post_base_N, comb_base_NN = hc_base_W.float().split([n, n, n * n])
    comb_base_NN = comb_base_NN.view(n, n)

    pre_BLN = torch.sigmoid(pre_BLN * hc_scale_3.float()[0] + pre_base_N) + eps
    post_BLN = 2.0 * torch.sigmoid(post_BLN * hc_scale_3.float()[1] + post_base_N)
    comb_BLNN = (
        torch.softmax(
            comb_BLNN * hc_scale_3.float()[2] + comb_base_NN,
            dim=-1,
        )
        + eps
    )
    comb_BLNN = comb_BLNN / (comb_BLNN.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb_BLNN = comb_BLNN / (comb_BLNN.sum(dim=-1, keepdim=True) + eps)
        comb_BLNN = comb_BLNN / (comb_BLNN.sum(dim=-2, keepdim=True) + eps)
    collapsed_BLD = torch.sum(pre_BLN.unsqueeze(-1) * x_BLND, dim=-2)
    return collapsed_BLD.to(input_dtype), post_BLN, comb_BLNN


@local_compile("glm53_speedrun_mhc_post", batch_invariant=True)
def mhc_post_tensor_program(
    update_BLD: torch.Tensor,
    residual_BLND: torch.Tensor,
    post_BLN: torch.Tensor,
    comb_BLNN: torch.Tensor,
) -> torch.Tensor:
    mixed_BLND = torch.matmul(
        comb_BLNN.to(residual_BLND.dtype).transpose(-1, -2),
        residual_BLND,
    )
    return post_BLN.to(update_BLD.dtype).unsqueeze(-1) * update_BLD.unsqueeze(
        -2
    ) + mixed_BLND.to(update_BLD.dtype)


class HcPre(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        num_streams: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1.0e-6,
        norm_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.norm_eps = norm_eps
        mix_width = (2 + num_streams) * num_streams
        self.hc_fn = nn.Parameter(torch.empty(mix_width, num_streams * hidden_size))
        self.hc_base = nn.Parameter(torch.empty(mix_width))
        self.hc_scale = nn.Parameter(torch.empty(3))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.hc_fn, mean=0.0, std=0.02)
        nn.init.zeros_(self.hc_base)
        nn.init.ones_(self.hc_scale)

    def forward(
        self, x_BLND: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_BLND.shape[-2] != self.num_streams:
            raise ValueError(
                f"expected {self.num_streams} mHC streams, got {x_BLND.shape[-2]}."
            )
        return mhc_pre_tensor_program(
            x_BLND,
            self.hc_fn,
            self.hc_base,
            self.hc_scale,
            num_streams=self.num_streams,
            sinkhorn_iters=self.sinkhorn_iters,
            eps=self.eps,
            norm_eps=self.norm_eps,
        )

    def _split_sinkhorn(
        self,
        mixes_BLW: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.num_streams
        pre_BLN, post_BLN, comb_BLNN = mixes_BLW.split([n, n, n * n], dim=-1)
        comb_BLNN = comb_BLNN.unflatten(-1, (n, n))
        pre_base_N, post_base_N, comb_base_NN = self.hc_base.float().split(
            [n, n, n * n]
        )
        comb_base_NN = comb_base_NN.view(n, n)

        pre_BLN = (
            torch.sigmoid(pre_BLN * self.hc_scale.float()[0] + pre_base_N) + self.eps
        )
        post_BLN = 2.0 * torch.sigmoid(
            post_BLN * self.hc_scale.float()[1] + post_base_N
        )
        comb_BLNN = (
            torch.softmax(
                comb_BLNN * self.hc_scale.float()[2] + comb_base_NN,
                dim=-1,
            )
            + self.eps
        )
        comb_BLNN = comb_BLNN / (comb_BLNN.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb_BLNN = comb_BLNN / (comb_BLNN.sum(dim=-1, keepdim=True) + self.eps)
            comb_BLNN = comb_BLNN / (comb_BLNN.sum(dim=-2, keepdim=True) + self.eps)
        return pre_BLN, post_BLN, comb_BLNN


class HcPost(nn.Module):
    def forward(
        self,
        update_BLD: torch.Tensor,
        residual_BLND: torch.Tensor,
        post_BLN: torch.Tensor,
        comb_BLNN: torch.Tensor,
    ) -> torch.Tensor:
        return mhc_post_tensor_program(update_BLD, residual_BLND, post_BLN, comb_BLNN)


class MHCResidualBlock(nn.Module):
    """Wrap a residual branch with GLM mHC and an exact identity-at-init mode."""

    def __init__(
        self,
        hidden_size: int,
        branch: nn.Module,
        *,
        num_streams: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1.0e-6,
        norm_eps: float = 1.0e-5,
        identity_at_init: bool = True,
    ) -> None:
        super().__init__()
        self.branch = branch
        self.pre = HcPre(
            hidden_size,
            num_streams=num_streams,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
            norm_eps=norm_eps,
        )
        self.post = HcPost()
        self.identity_at_init = identity_at_init
        if identity_at_init:
            self.identity_mix = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("identity_mix", None)

    def forward(
        self,
        streams_BLND: torch.Tensor,
        branch_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        branch = self.branch if branch_fn is None else branch_fn
        collapsed_BLD, post_BLN, comb_BLNN = self.pre(streams_BLND)
        learned_BLND = self.post(
            branch(collapsed_BLD), streams_BLND, post_BLN, comb_BLNN
        )
        if self.identity_mix is None:
            return learned_BLND
        plain_BLD = collapse_streams(streams_BLND)
        plain_BLND = streams_BLND + branch(plain_BLD).unsqueeze(-2)
        return plain_BLND + self.identity_mix * (learned_BLND - plain_BLND)
