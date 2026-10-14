# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stream-specific GLM model wrapper for FSDP CPU-offload probes."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from torchtitan.experiments.glm53_flash.flavors import debugmodel, Glm53FlashConfig
from torchtitan.experiments.glm53_flash.model import (
    _initialize_glm53_flash_weights,
    Glm53FlashForCausalLM,
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
        _initialize_glm53_flash_weights(self, self.glm_config)
        if buffer_device is not None:
            for module in self.modules():
                for name, buffer in module.named_buffers(recurse=False):
                    if buffer is not None:
                        setattr(module, name, buffer.to(device=buffer_device))
