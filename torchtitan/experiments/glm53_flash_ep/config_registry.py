# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan config factories for the isolated GLM EP prototype."""

from __future__ import annotations

import os
from dataclasses import replace

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC

from torchtitan.experiments.glm53_flash.config_registry import (
    _countdown_data_loader_config,
    _countdown_hf_assets_path,
    _env_bool,
    _lora_optim_config,
    _nonnegative_env_float,
    _nonnegative_env_int,
    _optim_config,
    _positive_env_int,
    Glm53PretokenizedTokenizer,
    Glm53SyntheticDataLoader,
)
from torchtitan.experiments.glm53_flash.flavors import with_lora
from torchtitan.experiments.glm53_flash_4gpu_lean.trainer import LeanMemoryTrainer
from torchtitan.models.base.model import BaseModel
from torchtitan.observability.metrics import MetricsProcessor

from .checkpoint import Glm53FlashEPLoRACheckpointManager
from .flavors import build_model_config
from .model import Glm53FlashEPForCausalLM


def model_registry(flavor: str) -> BaseModel.Config:
    return Glm53FlashEPForCausalLM.Config(
        glm_config=build_model_config(flavor),
        experiment_flavor=flavor,
    )


def glm53_flash_ep_debugmodel() -> "Trainer.Config":
    from torchtitan.trainer import Trainer

    cfg = build_model_config("debugmodel_ep")
    seq_len = _positive_env_int("GLM53_FLASH_EP_DEBUG_SEQ_LEN", 32)
    local_batch_size = _positive_env_int("GLM53_FLASH_EP_DEBUG_LOCAL_BATCH_SIZE", 1)
    steps = _positive_env_int("GLM53_FLASH_EP_DEBUG_STEPS", 1)
    return Trainer.Config(
        dump_folder=os.environ.get(
            "GLM53_FLASH_EP_DUMP_FOLDER",
            "./outputs/glm53_flash_ep_debugmodel",
        ),
        hf_assets_path=os.environ.get(
            "GLM53_FLASH_EP_HF_ASSETS_PATH", "./tests/assets/tokenizer"
        ),
        tokenizer=Glm53PretokenizedTokenizer.Config(vocab_size=cfg.vocab_size),
        model=Glm53FlashEPForCausalLM.Config(
            glm_config=cfg,
            experiment_flavor="debugmodel_ep",
        ),
        dataloader=Glm53SyntheticDataLoader.Config(vocab_size=cfg.vocab_size),
        metrics=MetricsProcessor.Config(log_freq=1),
        optim=_optim_config(steps=steps, lr=1.0e-3),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=local_batch_size * seq_len,
            max_context_length=seq_len,
            steps=steps,
            dtype="bfloat16",
            disable_cuda_graphs=True,
        ),
        parallelism=_ep_parallelism_config(),
        debug=DebugConfig(
            seed=42,
            deterministic=False,
            enable_structured_logging=False,
        ),
        checkpointer=None,
        activation_checkpoint=FullAC.Config(),
        loss=CrossEntropyLoss.Config(global_vocab_size=cfg.vocab_size),
    )


def glm53_flash_ep_debugmodel_lora() -> "Trainer.Config":
    config = glm53_flash_ep_debugmodel()
    assert isinstance(config.model, Glm53FlashEPForCausalLM.Config)
    config.model.glm_config = with_lora(
        build_model_config("debugmodel_ep"),
        rank=8,
        alpha=32.0,
    )
    config.model.experiment_flavor = "debugmodel_ep_lora"
    config.optim = _lora_optim_config(steps=config.training.steps, lr=1.0e-3)
    config.checkpointer = Glm53FlashEPLoRACheckpointManager.Config(
        interval=1,
        initial_load_model_only=False,
        initial_load_in_hf=False,
        last_save_model_only=False,
        keep_latest_k=2,
        load_only=False,
        verify_frozen_base_load=False,
    )
    config.dump_folder = os.environ.get(
        "GLM53_FLASH_EP_LORA_DUMP_FOLDER",
        "./outputs/glm53_flash_ep_debugmodel_lora",
    )
    return config


def glm53_flash_ep_mini() -> "Trainer.Config":
    config = glm53_flash_ep_debugmodel()
    cfg = build_model_config("mini_ep")
    assert isinstance(config.model, Glm53FlashEPForCausalLM.Config)
    config.model = Glm53FlashEPForCausalLM.Config(
        glm_config=cfg,
        experiment_flavor="mini_ep",
    )
    config.tokenizer = Glm53PretokenizedTokenizer.Config(vocab_size=cfg.vocab_size)
    config.dataloader = Glm53SyntheticDataLoader.Config(vocab_size=cfg.vocab_size)
    config.loss = CrossEntropyLoss.Config(global_vocab_size=cfg.vocab_size)
    config.training.max_context_length = _positive_env_int(
        "GLM53_FLASH_EP_MINI_SEQ_LEN", 16
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = (
        _positive_env_int("GLM53_FLASH_EP_MINI_LOCAL_BATCH_SIZE", 1)
        * config.training.max_context_length
    )
    config.training.steps = _positive_env_int("GLM53_FLASH_EP_MINI_STEPS", 1)
    config.dump_folder = os.environ.get(
        "GLM53_FLASH_EP_MINI_DUMP_FOLDER",
        "./outputs/glm53_flash_ep_mini",
    )
    return config


def glm53_flash_ep_countdown_lora_truncated() -> "LeanMemoryTrainer.Config":
    """Full-width, first-8-layer GLM EP Countdown LoRA parity config."""

    num_layers = _positive_env_int("GLM53_FLASH_EP_TRUNCATED_NUM_LAYERS", 8)
    layer_offset = _nonnegative_env_int("GLM53_FLASH_EP_TRUNCATED_LAYER_OFFSET", 0)
    if layer_offset != 0:
        raise ValueError(
            "GLM53_FLASH_EP_TRUNCATED_LAYER_OFFSET must stay 0 for this parity "
            "config so layers 0-7 are loaded from the released checkpoint."
        )

    cfg = with_lora(
        replace(
            build_model_config("glm53_flash_ep"),
            num_hidden_layers=num_layers,
            layer_offset=layer_offset,
        ),
        rank=16,
        alpha=64.0,
    )
    cfg.attn_impl = os.environ.get(
        "GLM53_FLASH_EP_ATTN_IMPL",
        os.environ.get("GLM53_FLASH_ATTN_IMPL", "fast"),
    )
    cfg.sparse_attention_mode = os.environ.get(
        "GLM53_FLASH_EP_SPARSE_ATTENTION_MODE",
        os.environ.get("GLM53_FLASH_SPARSE_ATTENTION_MODE", "indexer"),
    )
    cfg.validate()

    steps = _positive_env_int("GLM53_FLASH_FULL_STEPS", 3)
    seq_len = _positive_env_int("GLM53_FLASH_FULL_SEQ_LEN", 512)
    local_batch_size = _positive_env_int("GLM53_FLASH_FULL_LOCAL_BATCH_SIZE", 1)
    tokens_per_step = _positive_env_int(
        "GLM53_FLASH_FULL_TOKENS_PER_STEP",
        64 * 512,
    )
    lr = _nonnegative_env_float("GLM53_FLASH_FULL_LR", 1.0e-4)

    return LeanMemoryTrainer.Config(
        dump_folder=os.environ.get(
            "GLM53_FLASH_EP_FULL_DUMP_FOLDER",
            os.environ.get(
                "GLM53_FLASH_FULL_DUMP_FOLDER",
                "./outputs/glm53_flash_ep_countdown_lora_truncated",
            ),
        ),
        hf_assets_path=_countdown_hf_assets_path(),
        tokenizer=HuggingFaceTokenizer.Config(),
        model=Glm53FlashEPForCausalLM.Config(
            glm_config=cfg,
            experiment_flavor=f"glm53_flash_ep_first_{num_layers}",
        ),
        dataloader=_countdown_data_loader_config(
            seed=_positive_env_int("GLM53_FLASH_COUNTDOWN_SEED", 42)
        ),
        metrics=MetricsProcessor.Config(log_freq=1, enable_tensorboard=True),
        optim=_lora_optim_config(steps=steps, lr=lr),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=local_batch_size * seq_len,
            num_tokens_per_train_step=tokens_per_step,
            max_context_length=seq_len,
            steps=steps,
            dtype="bfloat16",
            disable_cuda_graphs=True,
            enable_cpu_offload=False,
        ),
        parallelism=_ep_parallelism_config(),
        debug=DebugConfig(
            seed=_nonnegative_env_int("GLM53_FLASH_SEED", 42),
            deterministic=_env_bool("GLM53_FLASH_DETERMINISTIC", True),
            enable_structured_logging=False,
        ),
        checkpointer=Glm53FlashEPLoRACheckpointManager.Config(
            interval=_positive_env_int("GLM53_FLASH_FULL_CHECKPOINT_INTERVAL", 99),
            initial_load_model_only=False,
            initial_load_in_hf=False,
            last_save_model_only=False,
            keep_latest_k=_nonnegative_env_int("GLM53_FLASH_FULL_KEEP_LATEST_K", 2),
            load_only=_env_bool("GLM53_FLASH_FULL_CHECKPOINT_LOAD_ONLY", False),
            verify_frozen_base_load=_env_bool(
                "GLM53_FLASH_EP_VERIFY_FROZEN_BASE_LOAD",
                True,
            ),
        ),
        activation_checkpoint=FullAC.Config(),
        loss=ChunkedLossWrapper.Config(
            num_chunks=_positive_env_int("GLM53_FLASH_LOSS_CHUNKS", 8),
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=cfg.vocab_size),
        ),
    )


def _ep_parallelism_config() -> ParallelismConfig:
    return ParallelismConfig(
        data_parallel_shard_degree=-1,
        expert_parallel_degree=_positive_env_int("GLM53_FLASH_EP_DEGREE", 1),
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        enable_sequence_parallel=False,
    )
