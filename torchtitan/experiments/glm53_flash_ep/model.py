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

from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import annotate_replicated_parameters
from torchtitan.experiments.glm53_flash.flavors import Glm53FlashConfig
from torchtitan.experiments.glm53_flash.model import (
    _initialize_glm53_flash_weights,
    apply_lora_trainable_policy,
    Glm53FlashDecoderLayer,
    Glm53FlashForCausalLM,
    LoRALinear,
)

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
            layer.mlp = GlmEPMoE(cfg)
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

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        del buffer_device
        if self.enable_weight_tying:
            self.lm_head.weight = self.tok_embeddings.weight
        _initialize_glm53_flash_weights(self, self.glm_config)
        _initialize_glm53_flash_ep_weights(self, self.glm_config)


def _initialize_glm53_flash_ep_weights(
    model: nn.Module,
    config: Glm53FlashConfig,
) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, GlmEPExperts):
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
            elif isinstance(module, GlmEPRouter):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=config.initializer_range,
                )
                module.e_score_correction_bias.zero_()
                module.reset_tokens_per_expert(device=module.weight.device)
            elif isinstance(module, LoRALinear):
                # The base GLM initializer already handled LoRA modules.
                continue


from .state_dict_adapter import Glm53FlashEPStateDictAdapter


Glm53FlashEPForCausalLM.state_dict_adapter_cls = Glm53FlashEPStateDictAdapter
