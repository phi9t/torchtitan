# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Isolated GLM-5.3-Flash model variant with common-MoE EP routing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import annotate_replicated_parameters
from torchtitan.experiments.glm53_flash.flavors import Glm53FlashConfig
from torchtitan.experiments.glm53_flash.model import (
    apply_lora_trainable_policy,
    Glm53FlashDecoderLayer,
    Glm53FlashForCausalLM,
    GlmExperts,
    GlmGatedRMSNorm,
    GlmHyperConnection,
    GlmIndexPool,
    GlmKDAAttention,
    GlmRMSNorm,
    GlmTopKRouter,
    LoRALinear,
)
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher

from .flavors import debugmodel_ep
from .moe import GlmEPExperts, GlmEPMoE, GlmEPRouter
from .sharding import (
    parallelize_glm53_flash_ep_experts,
    validate_glm53_flash_ep_parallelism,
)

logger = logging.getLogger(__name__)


class Glm53FlashEPForCausalLM(Glm53FlashForCausalLM):
    """GLM model variant replacing only sparse MoE layers with the EP adapter."""

    @dataclass(kw_only=True, slots=True)
    class Config(Glm53FlashForCausalLM.Config):
        glm_config: Glm53FlashConfig = field(default_factory=debugmodel_ep)
        experiment_name: str = "glm53_flash_ep"
        experiment_flavor: str = "debugmodel_ep"
        token_dispatcher: LocalTokenDispatcher.Config | None = None

        def set_sharding_(self, parallelism: ParallelismConfig) -> None:
            validate_glm53_flash_ep_parallelism(self.glm_config, parallelism)

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        cfg = config.glm_config
        for layer_idx, layer in self.layers.items():
            layer_id = int(layer_idx)
            if cfg.mlp_layer_types[layer_id] != "sparse":
                continue
            assert isinstance(layer, Glm53FlashDecoderLayer)
            layer.mlp = GlmEPMoE(cfg, token_dispatcher=config.token_dispatcher)
            layer.moe_enabled = True
        if cfg.lora_rank is not None:
            apply_lora_trainable_policy(self)

    def parallelize(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
        local_compile_regions: list[str],
        ac_config: ActivationCheckpointingConfig | None,
        dump_folder: str,
    ) -> "Glm53FlashEPForCausalLM":
        apply_local_compile(local_compile_regions)
        with parallelism_context.activate_spmd():
            annotate_replicated_parameters(self, parallelism_context)
            parallelize_glm53_flash_ep_experts(self, parallelism_context)
            if ac_config is not None:
                self._apply_activation_checkpoint(ac_config, dump_folder=dump_folder)
            self._apply_fsdp(
                parallelism_context=parallelism_context,
                training=training,
                parallelism=parallelism,
            )
        return self

    def _apply_fsdp(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        from torchtitan.distributed.fsdp import (
            apply_fsdp_to_decoder,
            resolve_fsdp_mesh,
            resolve_sparse_fsdp_mesh,
        )

        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallelism_context)
        apply_fsdp_to_decoder(
            self,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=False,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallelism_context.ep,
            edp_mesh=edp_mesh,
            dp_mesh_dims=dp_mesh_dims,
            edp_mesh_dims=edp_mesh_dims,
            symm_mem_scope=parallelism.fsdp_symm_mem_scope,
        )

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        if self.enable_weight_tying:
            self.lm_head.weight = self.tok_embeddings.weight
        _initialize_glm53_flash_ep_weights(self, self.glm_config)
        _initialize_glm53_flash_ep_buffers(self, buffer_device=buffer_device)


def _initialize_glm53_flash_ep_weights(
    model: nn.Module,
    config: Glm53FlashConfig,
) -> None:
    """Initialize EP models in the same RNG order as the plain GLM model.

    LoRA B starts at zero, so adapter weights do not affect the first forward
    loss. The first LoRA B gradient still depends on LoRA A. Keep sparse-layer
    expert/router draws before shared-expert LoRA draws, matching GlmMoE.
    """

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if module.rank is not None:
                    assert module.lora_a is not None and module.lora_b is not None
                    nn.init.normal_(module.lora_a, mean=0.0, std=0.01)
                    nn.init.zeros_(module.lora_b)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (GlmRMSNorm, GlmGatedRMSNorm)):
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, GlmHyperConnection):
                nn.init.normal_(module.fn, mean=0.0, std=0.02)
                nn.init.zeros_(module.base)
                nn.init.ones_(module.scale)
            elif isinstance(module, GlmEPExperts):
                nn.init.normal_(
                    module.w13.weight,
                    mean=0.0,
                    std=config.initializer_range,
                )
                nn.init.normal_(
                    module.w2.weight,
                    mean=0.0,
                    std=config.initializer_range,
                )
            elif isinstance(module, GlmExperts):
                nn.init.normal_(
                    module.gate_up_proj, mean=0.0, std=config.initializer_range
                )
                nn.init.normal_(
                    module.down_proj, mean=0.0, std=config.initializer_range
                )
            elif isinstance(module, GlmEPRouter):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=config.initializer_range,
                )
                module.e_score_correction_bias.zero_()
                module.reset_tokens_per_expert(device=module.weight.device)
            elif isinstance(module, GlmTopKRouter):
                nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
                module.e_score_correction_bias.zero_()
            elif isinstance(module, GlmIndexPool):
                nn.init.zeros_(module.index_kpool_compress_ape)
                nn.init.ones_(module.index_kpool_compress_gate)
            elif isinstance(module, GlmKDAAttention):
                nn.init.zeros_(module.A_log)
                dt = torch.empty_like(module.dt_bias).uniform_(1.0e-3, 1.0e-1)
                module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))


def _initialize_glm53_flash_ep_buffers(
    model: nn.Module,
    *,
    buffer_device: torch.device | None,
) -> None:
    """Initialize EP communication buffers after custom GLM weight init."""

    for module in model.modules():
        if isinstance(module, GlmEPExperts):
            module._init_self_buffers(buffer_device=buffer_device)


from .state_dict_adapter import Glm53FlashEPStateDictAdapter


Glm53FlashEPForCausalLM.state_dict_adapter_cls = Glm53FlashEPStateDictAdapter
