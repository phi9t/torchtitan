# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Isolated GLM-5.3-Flash expert-parallel prototype."""

from .flavors import build_model_config, MODEL_FLAVORS
from .model import Glm53FlashEPForCausalLM, GlmEPExperts, GlmEPMoE, GlmEPRouter

__all__ = [
    "Glm53FlashEPForCausalLM",
    "GlmEPExperts",
    "GlmEPMoE",
    "GlmEPRouter",
    "MODEL_FLAVORS",
    "build_model_config",
]
