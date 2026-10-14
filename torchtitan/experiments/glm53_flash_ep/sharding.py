# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelism validation and EP state sharding for the GLM EP prototype."""

from __future__ import annotations

import spmd_types as spmd

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.experiments.glm53_flash.flavors import Glm53FlashConfig
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.moe_sharding import expert_param_placement_sparse

from .moe import GlmEPExperts


def validate_glm53_flash_ep_parallelism(
    config: Glm53FlashConfig,
    parallelism: ParallelismConfig,
) -> None:
    """Validate the isolated GLM EP prototype mesh contract."""

    if parallelism.tensor_parallel_degree != 1:
        raise ValueError(
            "GLM-5.3-Flash EP prototype TP is not implemented yet; use "
            "tensor_parallel_degree=1."
        )
    if parallelism.context_parallel_degree != 1:
        raise ValueError(
            "GLM-5.3-Flash EP prototype CP is not implemented yet; use "
            "context_parallel_degree=1."
        )
    if parallelism.pipeline_parallel_degree != 1:
        raise ValueError(
            "GLM-5.3-Flash EP prototype does not support pipeline parallelism."
        )
    if parallelism.expert_parallel_degree < 1:
        raise ValueError("expert_parallel_degree must be >= 1.")
    if config.num_experts % parallelism.expert_parallel_degree != 0:
        raise ValueError(
            "num_experts must be divisible by expert_parallel_degree: "
            f"{config.num_experts} vs {parallelism.expert_parallel_degree}."
        )


def set_glm53_flash_ep_expert_sharding(
    experts: GlmEPExperts,
    *,
    enable_ep: bool,
) -> None:
    """Attach state sharding to the common grouped expert weights."""

    if enable_ep:
        expert_layout = expert_param_placement_sparse()
    else:
        expert_layout = dense_param_placement(tp=spmd.R)
    experts.w13._sharding_config = ShardingConfig(
        state_shardings={"weight": expert_layout}
    )
    experts.w2._sharding_config = ShardingConfig(
        state_shardings={"weight": expert_layout}
    )


def parallelize_glm53_flash_ep_experts(
    model,
    parallelism_context: ParallelismContext,
) -> None:
    """Apply EP-axis model sharding to every GLM EP routed expert module."""

    enable_ep = parallelism_context.ep > 1
    for module in model.modules():
        if isinstance(module, GlmEPExperts):
            set_glm53_flash_ep_expert_sharding(module, enable_ep=enable_ep)
            module._parallelize(parallelism_context)
