# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stream-specific GLM model wrapper for FSDP CPU-offload probes."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.experiments.glm53_flash.flavors import debugmodel, Glm53FlashConfig
from torchtitan.experiments.glm53_flash.model import (
    _initialize_glm53_flash_weights,
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


class StreamGlm53FlashForCausalLM(Glm53FlashForCausalLM):
    """Honor ``buffer_device`` while reusing the GLM implementation.

    TorchTitan allocates parameters on CPU and buffers on CUDA when FSDP2 CPU
    offload is enabled. The base GLM experiment currently discards the
    ``buffer_device`` argument, leaving persistent router buffers on CPU and
    causing device mismatches during routed MoE forward. This wrapper keeps
    that compatibility fix local to T09a.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Glm53FlashForCausalLM.Config):
        glm_config: Glm53FlashConfig = field(default_factory=debugmodel)

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        if self.enable_weight_tying:
            self.lm_head.weight = self.tok_embeddings.weight
        if buffer_device is None:
            _initialize_glm53_flash_weights(self, self.glm_config)
        else:
            _initialize_stream_glm53_flash_weights(
                self,
                self.glm_config,
                rng_device=buffer_device,
            )
        if buffer_device is not None:
            for module in self.modules():
                for name, buffer in module.named_buffers(recurse=False):
                    if buffer is not None:
                        setattr(module, name, buffer.to(device=buffer_device))


def _initialize_stream_glm53_flash_weights(
    model: nn.Module,
    config: Glm53FlashConfig,
    *,
    rng_device: torch.device,
) -> None:
    """Initialize CPU-offloaded shards with the plain path's CUDA RNG stream."""

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                _normal_like_cuda_(
                    module.weight,
                    mean=0.0,
                    std=config.initializer_range,
                    rng_device=rng_device,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if module.rank is not None:
                    assert module.lora_a is not None and module.lora_b is not None
                    _normal_like_cuda_(
                        module.lora_a,
                        mean=0.0,
                        std=0.01,
                        rng_device=rng_device,
                    )
                    nn.init.zeros_(module.lora_b)
            elif isinstance(module, nn.Linear):
                _normal_like_cuda_(
                    module.weight,
                    mean=0.0,
                    std=config.initializer_range,
                    rng_device=rng_device,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                _normal_like_cuda_(
                    module.weight,
                    mean=0.0,
                    std=config.initializer_range,
                    rng_device=rng_device,
                )
            elif isinstance(module, nn.Conv1d):
                _normal_like_cuda_(
                    module.weight,
                    mean=0.0,
                    std=config.initializer_range,
                    rng_device=rng_device,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (GlmRMSNorm, GlmGatedRMSNorm)):
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, GlmHyperConnection):
                _normal_like_cuda_(
                    module.fn,
                    mean=0.0,
                    std=0.02,
                    rng_device=rng_device,
                )
                nn.init.zeros_(module.base)
                nn.init.ones_(module.scale)
            elif isinstance(module, GlmExperts):
                _normal_like_cuda_(
                    module.gate_up_proj,
                    mean=0.0,
                    std=config.initializer_range,
                    rng_device=rng_device,
                )
                _normal_like_cuda_(
                    module.down_proj,
                    mean=0.0,
                    std=config.initializer_range,
                    rng_device=rng_device,
                )
            elif isinstance(module, GlmTopKRouter):
                _normal_like_cuda_(
                    module.weight,
                    mean=0.0,
                    std=config.initializer_range,
                    rng_device=rng_device,
                )
                module.e_score_correction_bias.zero_()
            elif isinstance(module, GlmIndexPool):
                nn.init.zeros_(module.index_kpool_compress_ape)
                nn.init.ones_(module.index_kpool_compress_gate)
            elif isinstance(module, GlmKDAAttention):
                nn.init.zeros_(module.A_log)
                _dt_bias_like_cuda_(module.dt_bias, rng_device=rng_device)


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor._local_tensor if isinstance(tensor, DTensor) else tensor


def _normal_like_cuda_(
    tensor: torch.Tensor,
    *,
    mean: float,
    std: float,
    rng_device: torch.device,
) -> None:
    local_tensor = _local_tensor(tensor)
    if isinstance(tensor, DTensor):
        tmp = torch.empty_like(tensor, device=rng_device)
        nn.init.normal_(tmp, mean=mean, std=std)
        local_tensor.copy_(tmp._local_tensor.to(device=local_tensor.device))
        return

    tmp = torch.empty_like(local_tensor, device=rng_device)
    nn.init.normal_(tmp, mean=mean, std=std)
    local_tensor.copy_(tmp.to(device=local_tensor.device))


def _dt_bias_like_cuda_(tensor: torch.Tensor, *, rng_device: torch.device) -> None:
    local_tensor = _local_tensor(tensor)
    if isinstance(tensor, DTensor):
        dt = torch.empty_like(tensor, device=rng_device).uniform_(1.0e-3, 1.0e-1)
        value = dt + torch.log(-torch.expm1(-dt))
        local_tensor.copy_(value._local_tensor.to(device=local_tensor.device))
        return

    dt = torch.empty_like(local_tensor, device=rng_device).uniform_(1.0e-3, 1.0e-1)
    local_tensor.copy_(
        (dt + torch.log(-torch.expm1(-dt))).to(device=local_tensor.device)
    )
