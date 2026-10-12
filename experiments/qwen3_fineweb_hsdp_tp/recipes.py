# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recipes for the local Qwen3 FineWeb HSDP + TP experiment."""

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan_recipes.tests.models.qwen3 import qwen3_debugmodel

from torchtitan.trainer import Trainer


def qwen3_debugmodel_fineweb() -> Trainer.Config:
    config = qwen3_debugmodel(seq_len=2048)
    config.dataloader.dataset.dataset = DATASETS["fineweb_test"]
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        last_save_model_only=False,
    )
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    return config
