# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lean FSDP wrapping policy for GLM-5.3-Flash bf16 4-GPU probes."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Shard

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    enable_fsdp_symm_mem,
    get_fsdp_reshard_after_forward_policy,
    linear_param_shard_placements,
    resolve_fsdp_mesh,
)
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.experiments.glm53_flash.model import (
    Glm53FlashDecoderLayer,
    Glm53FlashForCausalLM,
    GlmMoE,
)


logger = logging.getLogger(__name__)

FSDPWrapPolicy = Literal["layer", "split"]


@dataclass(frozen=True, slots=True)
class FSDPUnit:
    """One bottom-up FSDP group used by the lean wrapper."""

    label: str
    module: nn.Module
    shard_placement_fn: Callable[[nn.Parameter], Shard | None] | None = None


class LeanGlm53FlashForCausalLM(Glm53FlashForCausalLM):
    """GLM model with experiment-owned FSDP grouping knobs.

    The forward path, parameters, state-dict names, and initialization are
    inherited from ``Glm53FlashForCausalLM``. Only the FSDP unit boundaries are
    changed when ``fsdp_wrap_policy='split'``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Glm53FlashForCausalLM.Config):
        fsdp_wrap_policy: FSDPWrapPolicy = "split"
        clear_explicit_prefetch: bool = True

    config: Config

    def _apply_fsdp(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        if self.config.fsdp_wrap_policy == "layer":
            super()._apply_fsdp(
                parallelism_context=parallelism_context,
                training=training,
                parallelism=parallelism,
            )
            if self.config.clear_explicit_prefetch:
                clear_explicit_fsdp_prefetch(self)
            return

        if self.config.fsdp_wrap_policy != "split":
            raise ValueError(
                "fsdp_wrap_policy must be 'layer' or 'split', got "
                f"{self.config.fsdp_wrap_policy!r}"
            )

        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
        reshard_after_forward = get_fsdp_reshard_after_forward_policy(
            parallelism.fsdp_reshard_after_forward,
            pp_enabled=False,
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            cast_forward_inputs=False,
        )
        fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
        if dp_mesh_dims is not None and (
            parallelism.tensor_parallel_degree != 1
            or parallelism.context_parallel_degree != 1
            or parallelism.data_parallel_replicate_degree != 1
        ):
            fsdp_config["dp_mesh_dims"] = dp_mesh_dims
        if training.enable_cpu_offload:
            fsdp_config["offload_policy"] = CPUOffloadPolicy()

        if self.enable_weight_tying:
            modules = [
                module
                for module in (self.tok_embeddings, self.norm, self.lm_head)
                if module is not None
            ]
            fully_shard(
                modules,
                **fsdp_config,
                reshard_after_forward=parallelism.fsdp_reshard_after_forward
                == "always",
            )
        else:
            if self.tok_embeddings is not None:
                fully_shard(
                    self.tok_embeddings,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )
            if self.norm is not None and self.lm_head is not None:
                fully_shard(
                    [self.norm, self.lm_head],
                    **fsdp_config,
                    reshard_after_forward=parallelism.fsdp_reshard_after_forward
                    == "always",
                )

        for layer_id, layer in self.layers.items():
            assert isinstance(layer, Glm53FlashDecoderLayer)
            for unit in iter_split_fsdp_units(layer):
                fully_shard(
                    unit.module,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                    shard_placement_fn=unit.shard_placement_fn,
                )
                logger.debug("Applied GLM lean FSDP unit %s.%s", layer_id, unit.label)
            fully_shard(
                layer,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

        fully_shard(self, **fsdp_config)
        enable_fsdp_symm_mem(self, parallelism.fsdp_symm_mem_scope)
        disable_fsdp_gradient_division(self)
        if self.config.clear_explicit_prefetch:
            clear_explicit_fsdp_prefetch(self)
        logger.info(
            "Applied GLM lean split FSDP: reshard_after_forward=%s, "
            "clear_explicit_prefetch=%s",
            reshard_after_forward,
            self.config.clear_explicit_prefetch,
        )


def iter_split_fsdp_units(layer: Glm53FlashDecoderLayer) -> Iterator[FSDPUnit]:
    """Yield safe bottom-up FSDP groups inside one GLM decoder layer."""

    yield FSDPUnit("attn_hc", layer.attn_hc)
    yield FSDPUnit(
        "self_attn",
        layer.self_attn,
        shard_placement_fn=linear_param_shard_placements(layer.self_attn).get,
    )
    yield FSDPUnit("ffn_hc", layer.ffn_hc)
    if isinstance(layer.mlp, GlmMoE):
        yield FSDPUnit("mlp.gate", layer.mlp.gate)
        yield FSDPUnit(
            "mlp.shared_experts",
            layer.mlp.shared_experts,
            shard_placement_fn=linear_param_shard_placements(
                layer.mlp.shared_experts
            ).get,
        )
        expert_placements = {
            param: Shard(0) for param in layer.mlp.experts.parameters()
        }
        yield FSDPUnit(
            "mlp.experts",
            layer.mlp.experts,
            shard_placement_fn=expert_placements.get,
        )
    else:
        yield FSDPUnit(
            "mlp",
            layer.mlp,
            shard_placement_fn=linear_param_shard_placements(layer.mlp).get,
        )


def clear_explicit_fsdp_prefetch(model: nn.Module) -> None:
    """Clear explicit FSDP prefetch lists where the installed torch exposes them."""

    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        set_forward = getattr(module, "set_modules_to_forward_prefetch", None)
        if set_forward is not None:
            set_forward([])
        set_backward = getattr(module, "set_modules_to_backward_prefetch", None)
        if set_backward is not None:
            set_backward([])

