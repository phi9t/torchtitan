# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Switchable GLM-5.3-Flash architecture elements for speedrun ablations."""

from .adapter import build_attention, maybe_build_mtp, maybe_wrap_mhc
from .attention import (
    causal_token_indices,
    dense_causal_attention,
    IndexPool,
    LightningIndexer,
    sparse_causal_attention,
    TopKSparseCausalAttention,
)
from .config import arm_config, b0_config, compose_arms, SpeedrunElementsConfig
from .kda import kda_chunked_reference, kda_recurrent, KDALayer
from .mhc import collapse_streams, expand_streams, HcPost, HcPre, MHCResidualBlock
from .mtp import MTPHead
from .positional import apply_rope, maybe_apply_rope

__all__ = [
    "HcPost",
    "HcPre",
    "IndexPool",
    "KDALayer",
    "LightningIndexer",
    "MHCResidualBlock",
    "MTPHead",
    "SpeedrunElementsConfig",
    "TopKSparseCausalAttention",
    "apply_rope",
    "arm_config",
    "b0_config",
    "build_attention",
    "causal_token_indices",
    "collapse_streams",
    "compose_arms",
    "dense_causal_attention",
    "expand_streams",
    "kda_chunked_reference",
    "kda_recurrent",
    "maybe_build_mtp",
    "maybe_apply_rope",
    "maybe_wrap_mhc",
    "sparse_causal_attention",
]
