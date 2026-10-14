# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelism validation for the GLM-5.3-Flash experiment seam."""

from __future__ import annotations

from torchtitan.config.parallelism import ParallelismConfig

from .flavors import Glm53FlashConfig


def validate_glm53_flash_parallelism(
    config: Glm53FlashConfig,
    parallelism: ParallelismConfig,
) -> None:
    """Validate mesh axes supported by this experiment-owned model.

    The model applies TorchTitan FSDP in ``Glm53FlashForCausalLM._apply_fsdp``.
    Expert parameters can be sharded by the sparse FSDP mesh when EP is enabled,
    but this first experiment seam does not implement token-dispatch expert
    parallel execution, TP, or CP for sparse MLA.
    """

    if parallelism.tensor_parallel_degree != 1:
        raise ValueError(
            "GLM-5.3-Flash experiment TP is not implemented yet; use "
            "tensor_parallel_degree=1."
        )
    if parallelism.context_parallel_degree != 1:
        raise ValueError(
            "GLM-5.3-Flash sparse MLA CP is not implemented yet; use "
            "context_parallel_degree=1."
        )
    if parallelism.pipeline_parallel_degree != 1:
        raise ValueError(
            "GLM-5.3-Flash experiment model does not support pipeline parallelism."
        )
    if parallelism.expert_parallel_degree < 1:
        raise ValueError("expert_parallel_degree must be >= 1.")
    if config.num_experts % parallelism.expert_parallel_degree != 0:
        raise ValueError(
            "num_experts must be divisible by expert_parallel_degree: "
            f"{config.num_experts} vs {parallelism.expert_parallel_degree}."
        )
