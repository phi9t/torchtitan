# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP8 frozen-base GLM-5.3-Flash model wrapper."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.experiments.glm53_flash.flavors import debugmodel, Glm53FlashConfig
from torchtitan.experiments.glm53_flash.model import (
    apply_lora_trainable_policy,
    Glm53FlashForCausalLM,
    GlmGatedRMSNorm,
    GlmHyperConnection,
    GlmIndexPool,
    GlmKDAAttention,
    GlmRMSNorm,
    GlmTopKRouter,
    LoRALinear,
)

from .fp8 import (
    FP8FrozenExperts,
    FP8FrozenLoRALinear,
    fp8_param_dtype_override,
    local_tensor,
)


logger = logging.getLogger(__name__)

_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")


@dataclass(frozen=True, slots=True)
class FP8ConversionStats:
    """Counts of modules converted to FP8 frozen storage."""

    linear_modules: int = 0
    expert_modules: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "linear_modules": self.linear_modules,
            "expert_modules": self.expert_modules,
        }


def _should_quantize_lora_linear(name: str, config: Glm53FlashConfig) -> bool:
    """Match the official GLM FP8 exclusion contract for text modules."""
    match = _LAYER_RE.match(name)
    if match is None:
        return False
    layer_idx = int(match.group(1))
    suffix = match.group(2)

    if ".indexer." in suffix:
        return False
    if suffix == "mlp.gate":
        return False
    if config.layer_types[layer_idx] == "linear_attention" and suffix.startswith(
        "self_attn."
    ):
        return False
    if suffix in {"self_attn.kv_b_proj"}:
        return False
    return True


def _replace_child(parent: nn.Module, child_name: str, child: nn.Module) -> None:
    if isinstance(parent, nn.ModuleDict):
        parent[child_name] = child
    else:
        setattr(parent, child_name, child)


def apply_fp8_frozen_base(model: nn.Module, config: Glm53FlashConfig) -> FP8ConversionStats:
    """Replace selected frozen base modules with FP8-storage variants."""
    linear_modules = 0
    expert_modules = 0

    def visit(parent: nn.Module, prefix: str = "") -> None:
        nonlocal linear_modules, expert_modules
        for child_name, child in list(parent.named_children()):
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LoRALinear):
                if _should_quantize_lora_linear(child_prefix, config):
                    _replace_child(
                        parent,
                        child_name,
                        FP8FrozenLoRALinear.from_lora_linear(child),
                    )
                    linear_modules += 1
                continue
            if child.__class__.__name__ == "GlmExperts":
                _replace_child(parent, child_name, FP8FrozenExperts.from_experts(child))
                expert_modules += 1
                continue
            visit(child, child_prefix)

    visit(model)
    if config.lora_rank is not None:
        apply_lora_trainable_policy(model)
    return FP8ConversionStats(
        linear_modules=linear_modules,
        expert_modules=expert_modules,
    )


class Glm53FlashFP8ForCausalLM(Glm53FlashForCausalLM):
    """GLM-5.3-Flash with FP8 persistent storage for frozen base weights."""

    @dataclass(kw_only=True, slots=True)
    class Config(Glm53FlashForCausalLM.Config):
        glm_config: Glm53FlashConfig = field(default_factory=debugmodel)
        experiment_name: str = "glm53_flash_4gpu_fp8"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.fp8_conversion_stats = apply_fp8_frozen_base(self, self.glm_config)
        logger.info("Applied GLM FP8 frozen-base conversion: %s", self.fp8_conversion_stats)

    def parallelize(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
        local_compile_regions: list[str],
        ac_config: ActivationCheckpointingConfig | None,
        dump_folder: str,
    ) -> "Glm53FlashFP8ForCausalLM":
        apply_local_compile(local_compile_regions)
        with parallelism_context.activate_spmd():
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

        dp_mesh, _ = resolve_fsdp_mesh(parallelism_context)
        edp_mesh, _ = resolve_sparse_fsdp_mesh(parallelism_context)
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
            symm_mem_scope=parallelism.fsdp_symm_mem_scope,
            param_dtype_override_fn=fp8_param_dtype_override,
        )

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        del buffer_device
        if self.enable_weight_tying:
            self.lm_head.weight = self.tok_embeddings.weight
        _initialize_glm53_flash_fp8_weights(self, self.glm_config)


def _initialize_glm53_flash_fp8_weights(
    model: nn.Module,
    config: Glm53FlashConfig,
) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, FP8FrozenLoRALinear):
                weight = local_tensor(module.weight)
                source = torch.empty(
                    tuple(weight.shape),
                    device=weight.device,
                    dtype=torch.bfloat16,
                )
                nn.init.normal_(source, mean=0.0, std=config.initializer_range)
                module.load_base_from_bf16_(source)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if module.rank is not None:
                    assert module.lora_a is not None and module.lora_b is not None
                    nn.init.normal_(module.lora_a, mean=0.0, std=0.01)
                    nn.init.zeros_(module.lora_b)
            elif isinstance(module, LoRALinear):
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
            elif isinstance(module, FP8FrozenExperts):
                gate_up = local_tensor(module.gate_up_proj)
                gate_up_source = torch.empty(
                    tuple(gate_up.shape),
                    device=gate_up.device,
                    dtype=torch.bfloat16,
                )
                nn.init.normal_(
                    gate_up_source, mean=0.0, std=config.initializer_range
                )
                module.load_gate_up_from_bf16_(gate_up_source)
                down = local_tensor(module.down_proj)
                down_source = torch.empty(
                    tuple(down.shape),
                    device=down.device,
                    dtype=torch.bfloat16,
                )
                nn.init.normal_(down_source, mean=0.0, std=config.initializer_range)
                module.load_down_from_bf16_(down_source)
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


from torchtitan.experiments.glm53_flash.state_dict_adapter import (  # noqa: E402
    Glm53FlashStateDictAdapter,
)


Glm53FlashFP8ForCausalLM.state_dict_adapter_cls = Glm53FlashStateDictAdapter

