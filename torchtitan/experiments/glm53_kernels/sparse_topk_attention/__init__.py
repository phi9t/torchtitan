# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton top-k sparse causal attention for GLM-style index selections."""

from .op import (
    sparse_topk_attention_bhld,
    sparse_topk_attention_blhd,
)

__all__ = [
    "sparse_topk_attention_bhld",
    "sparse_topk_attention_blhd",
]

