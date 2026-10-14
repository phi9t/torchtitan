# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5.3-Flash experiment seam."""

from .flavors import build_model_config, GLM53_FLASH_SPECIAL_TOKENS, MODEL_FLAVORS
from .model import Glm53FlashForCausalLM

__all__ = [
    "GLM53_FLASH_SPECIAL_TOKENS",
    "Glm53FlashForCausalLM",
    "MODEL_FLAVORS",
    "build_model_config",
]
