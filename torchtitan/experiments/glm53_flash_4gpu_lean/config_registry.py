# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config factories for T09c GLM-5.3-Flash 4-GPU bf16 lean probes."""

from __future__ import annotations

import os
from dataclasses import fields, replace

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.experiments.glm53_flash.config_registry import (
    _countdown_data_loader_config,
    _countdown_hf_assets_path,
    _env_bool,
    _nonnegative_env_float,
    _nonnegative_env_int,
    _positive_env_int,
    glm53_flash_countdown_lora,
)
from torchtitan.experiments.glm53_flash.flavors import build_model_config, with_lora
from torchtitan.experiments.glm53_flash.model import Glm53FlashForCausalLM

from .model import FSDPWrapPolicy, LeanGlm53FlashForCausalLM
from .trainer import LeanMemoryTrainer


def glm53_flash_countdown_lora_lean() -> "LeanMemoryTrainer.Config":
    """Full-depth GLM LoRA SFT config with T09c FSDP/memory knobs."""

    config = _lean_trainer_config_from_base(glm53_flash_countdown_lora())
    full_cfg = with_lora(build_model_config("glm53_flash"), rank=16, alpha=64.0)
    full_cfg.attn_impl = _attention_impl()
    full_cfg.sparse_attention_mode = os.environ.get(
        "GLM53_LEAN_SPARSE_ATTENTION_MODE",
        full_cfg.sparse_attention_mode,
    )
    full_cfg.validate()
    config.model = LeanGlm53FlashForCausalLM.Config(
        glm_config=full_cfg,
        experiment_flavor="glm53_flash",
        fsdp_wrap_policy=_fsdp_wrap_policy(),
        clear_explicit_prefetch=_env_bool(
            "GLM53_LEAN_CLEAR_EXPLICIT_PREFETCH", True
        ),
    )
    _apply_lean_runtime_knobs(config, vocab_size=full_cfg.vocab_size)
    return config


def glm53_flash_countdown_lora_truncated_lean() -> "LeanMemoryTrainer.Config":
    """Full-width, truncated-depth T09c config for 4-GPU memory probes."""

    num_layers = _positive_env_int("GLM53_LEAN_NUM_LAYERS", 8)
    layer_offset = _nonnegative_env_int("GLM53_LEAN_LAYER_OFFSET", 0)
    if layer_offset != 0:
        raise ValueError(
            "GLM53_LEAN_LAYER_OFFSET must stay 0 for rank-local HF loading in "
            "this ticket package. Nonzero offsets need a layer-remapping loader."
        )
    base = _lean_trainer_config_from_base(glm53_flash_countdown_lora())
    cfg = with_lora(
        replace(
            build_model_config("glm53_flash"),
            num_hidden_layers=num_layers,
            layer_offset=layer_offset,
        ),
        rank=16,
        alpha=64.0,
    )
    cfg.attn_impl = _attention_impl()
    cfg.sparse_attention_mode = os.environ.get(
        "GLM53_LEAN_SPARSE_ATTENTION_MODE",
        cfg.sparse_attention_mode,
    )
    cfg.validate()
    base.model = LeanGlm53FlashForCausalLM.Config(
        glm_config=cfg,
        experiment_flavor=f"glm53_flash_first_{num_layers}",
        fsdp_wrap_policy=_fsdp_wrap_policy(),
        clear_explicit_prefetch=_env_bool(
            "GLM53_LEAN_CLEAR_EXPLICIT_PREFETCH", True
        ),
    )
    base.training.steps = _positive_env_int("GLM53_LEAN_STEPS", 3)
    _apply_lean_runtime_knobs(base, vocab_size=cfg.vocab_size)
    return base


def glm53_flash_countdown_lora_truncated_plain() -> "LeanMemoryTrainer.Config":
    """Plain GLM wrapper for truncated-depth parity against the lean path."""

    num_layers = _positive_env_int("GLM53_LEAN_NUM_LAYERS", 8)
    layer_offset = _nonnegative_env_int("GLM53_LEAN_LAYER_OFFSET", 0)
    if layer_offset != 0:
        raise ValueError(
            "GLM53_LEAN_LAYER_OFFSET must stay 0 for rank-local HF loading in "
            "this ticket package. Nonzero offsets need a layer-remapping loader."
        )
    config = _lean_trainer_config_from_base(glm53_flash_countdown_lora())
    cfg = with_lora(
        replace(
            build_model_config("glm53_flash"),
            num_hidden_layers=num_layers,
            layer_offset=layer_offset,
        ),
        rank=16,
        alpha=64.0,
    )
    cfg.attn_impl = _attention_impl()
    cfg.sparse_attention_mode = os.environ.get(
        "GLM53_LEAN_SPARSE_ATTENTION_MODE",
        cfg.sparse_attention_mode,
    )
    cfg.validate()
    config.model = Glm53FlashForCausalLM.Config(
        glm_config=cfg,
        experiment_flavor=f"glm53_flash_first_{num_layers}_plain",
    )
    config.training.steps = _positive_env_int("GLM53_LEAN_STEPS", 3)
    _apply_lean_runtime_knobs(config, vocab_size=cfg.vocab_size)
    return config


def _lean_trainer_config_from_base(config) -> "LeanMemoryTrainer.Config":
    kwargs = {
        field.name: getattr(config, field.name)
        for field in fields(LeanMemoryTrainer.Config)
        if hasattr(config, field.name)
    }
    return LeanMemoryTrainer.Config(**kwargs)


def _apply_lean_runtime_knobs(
    config: "LeanMemoryTrainer.Config", *, vocab_size: int
) -> None:
    config.hf_assets_path = _countdown_hf_assets_path()
    config.dataloader = _countdown_data_loader_config(
        seed=_positive_env_int("GLM53_LEAN_COUNTDOWN_SEED", 42)
    )
    config.dump_folder = os.environ.get(
        "GLM53_LEAN_DUMP_FOLDER",
        os.environ.get(
            "GLM53_FLASH_FULL_DUMP_FOLDER",
            ".scratch/glm53-flash-sft-ladder/evidence/task-09c/lean_run",
        ),
    )
    config.training.max_context_length = _positive_env_int(
        "GLM53_LEAN_SEQ_LEN",
        _positive_env_int("GLM53_FLASH_FULL_SEQ_LEN", 512),
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = (
        _positive_env_int(
            "GLM53_LEAN_LOCAL_BATCH_SIZE",
            _positive_env_int("GLM53_FLASH_FULL_LOCAL_BATCH_SIZE", 1),
        )
        * config.training.max_context_length
    )
    config.training.num_tokens_per_train_step = _positive_env_int(
        "GLM53_LEAN_TOKENS_PER_STEP",
        _positive_env_int("GLM53_FLASH_FULL_TOKENS_PER_STEP", 64 * 512),
    )
    config.training.steps = _positive_env_int(
        "GLM53_LEAN_STEPS",
        _positive_env_int("GLM53_FLASH_FULL_STEPS", config.training.steps),
    )
    config.parallelism.fsdp_reshard_after_forward = os.environ.get(
        "GLM53_LEAN_RESHARD_AFTER_FORWARD",
        config.parallelism.fsdp_reshard_after_forward,
    )
    config.debug.seed = _nonnegative_env_int("GLM53_LEAN_SEED", config.debug.seed)
    config.debug.deterministic = _env_bool(
        "GLM53_LEAN_DETERMINISTIC",
        config.debug.deterministic,
    )
    config.loss = ChunkedLossWrapper.Config(
        num_chunks=_positive_env_int(
            "GLM53_LEAN_LOSS_CHUNKS",
            _positive_env_int("GLM53_FLASH_LOSS_CHUNKS", 8),
        ),
        loss_fn=CrossEntropyLoss.Config(global_vocab_size=vocab_size),
    )
    if config.checkpointer is not None:
        config.checkpointer.keep_latest_k = _nonnegative_env_int(
            "GLM53_LEAN_KEEP_LATEST_K",
            config.checkpointer.keep_latest_k,
        )
        config.checkpointer.interval = _positive_env_int(
            "GLM53_LEAN_CHECKPOINT_INTERVAL",
            config.checkpointer.interval,
        )
    config.optim.lr_scheduler.warmup_steps = max(1, config.training.steps // 10)
    config.optim.lr_scheduler.total_steps = config.training.steps
    for optimizer_config in config.optim.optimizer.optimizers:
        optimizer_config.lr = _nonnegative_env_float(
            "GLM53_LEAN_LR",
            optimizer_config.lr,
        )


def _fsdp_wrap_policy() -> FSDPWrapPolicy:
    value = os.environ.get("GLM53_LEAN_FSDP_WRAP_POLICY", "split")
    if value not in {"layer", "split"}:
        raise ValueError(
            "GLM53_LEAN_FSDP_WRAP_POLICY must be 'layer' or 'split', got "
            f"{value!r}"
        )
    return value


def _attention_impl() -> str:
    return os.environ.get(
        "GLM53_LEAN_ATTN_IMPL",
        os.environ.get("GLM53_FLASH_ATTN_IMPL", "fast"),
    )

