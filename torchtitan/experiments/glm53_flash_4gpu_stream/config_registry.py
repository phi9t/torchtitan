# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config factories for T09a GLM-5.3-Flash 4-GPU CPU-offload streaming."""

from __future__ import annotations

import os
from dataclasses import replace

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.glm53_flash.checkpoint import (
    Glm53FlashLoRACheckpointManager,
)
from torchtitan.experiments.glm53_flash.config_registry import (
    _countdown_data_loader_config,
    _countdown_hf_assets_path,
    _lora_optim_config,
)
from torchtitan.experiments.glm53_flash.flavors import build_model_config, with_lora
from torchtitan.experiments.glm53_flash.model import Glm53FlashForCausalLM
from torchtitan.experiments.glm53_flash_4gpu_stream.model import (
    StreamGlm53FlashForCausalLM,
)
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer


_DEFAULT_DUMP_ROOT = ".scratch/glm53-flash-sft-ladder/evidence/task-09a"


def glm53_flash_countdown_lora_stream() -> Trainer.Config:
    """Full-depth GLM LoRA SFT with FSDP2 CPU-offloaded frozen-base shards."""

    return _streaming_config(
        config_name="glm53_flash_countdown_lora_stream",
        num_layers=None,
        cpu_offload=True,
        default_steps=12,
    )


def glm53_flash_countdown_lora_truncated_stream() -> Trainer.Config:
    """Full-width, first-N-layer CPU-offload config for parity and smoke probes."""

    return _streaming_config(
        config_name="glm53_flash_countdown_lora_truncated_stream",
        num_layers=_env_int("GLM53_STREAM_NUM_LAYERS", 8),
        cpu_offload=True,
        default_steps=3,
    )


def glm53_flash_countdown_lora_truncated_plain() -> Trainer.Config:
    """Full-width, first-N-layer GPU-resident baseline for parity probes."""

    return _streaming_config(
        config_name="glm53_flash_countdown_lora_truncated_plain",
        num_layers=_env_int("GLM53_STREAM_NUM_LAYERS", 8),
        cpu_offload=False,
        default_steps=3,
    )


def _streaming_config(
    *,
    config_name: str,
    num_layers: int | None,
    cpu_offload: bool,
    default_steps: int,
) -> Trainer.Config:
    seq_len = _env_int(
        "GLM53_STREAM_SEQ_LEN",
        _env_int("GLM53_FLASH_FULL_SEQ_LEN", 512),
    )
    tokens_per_step = _env_int(
        "GLM53_STREAM_TOKENS_PER_STEP",
        _env_int("GLM53_FLASH_FULL_TOKENS_PER_STEP", 64 * 512),
    )
    local_batch_size = _env_int(
        "GLM53_STREAM_LOCAL_BATCH_SIZE",
        _env_int("GLM53_FLASH_FULL_LOCAL_BATCH_SIZE", 1),
    )
    expected_world_size = _env_int("GLM53_STREAM_EXPECTED_WORLD_SIZE", 4)
    _validate_batch_math(
        seq_len=seq_len,
        local_batch_size=local_batch_size,
        tokens_per_step=tokens_per_step,
        expected_world_size=expected_world_size,
    )

    glm_config = build_model_config("glm53_flash")
    if num_layers is not None:
        if num_layers <= 0:
            raise ValueError("GLM53_STREAM_NUM_LAYERS must be positive")
        layer_offset = _env_int("GLM53_STREAM_LAYER_OFFSET", 0)
        if layer_offset != 0:
            raise ValueError(
                "GLM53_STREAM_LAYER_OFFSET must stay 0: the inherited rank-local "
                "HF loader maps native layer ids directly to HF layer ids."
            )
        glm_config = replace(
            glm_config,
            num_hidden_layers=num_layers,
            layer_offset=layer_offset,
        )
    glm_config = with_lora(glm_config, rank=16, alpha=64.0)
    _apply_attention_env(glm_config)

    steps = _env_int(
        "GLM53_STREAM_STEPS",
        _env_int("GLM53_FLASH_FULL_STEPS", default_steps),
    )
    config = Trainer.Config(
        dump_folder=_dump_folder(config_name),
        hf_assets_path=_countdown_hf_assets_path(),
        tokenizer=HuggingFaceTokenizer.Config(),
        model=_model_config(
            glm_config=glm_config, num_layers=num_layers, stream=cpu_offload
        ),
        dataloader=_countdown_data_loader_config(
            seed=_env_int("GLM53_STREAM_COUNTDOWN_SEED", 42)
        ),
        metrics=MetricsProcessor.Config(log_freq=1, enable_tensorboard=True),
        optim=_lora_optim_config(
            steps=steps,
            lr=_env_float("GLM53_STREAM_LR", _env_float("GLM53_FLASH_FULL_LR", 1e-4)),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=local_batch_size * seq_len,
            num_tokens_per_train_step=tokens_per_step,
            max_context_length=seq_len,
            steps=steps,
            enable_cpu_offload=cpu_offload,
            dtype="bfloat16",
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            expert_parallel_degree=1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            enable_sequence_parallel=False,
            fsdp_reshard_after_forward=os.environ.get(
                "GLM53_STREAM_RESHARD_AFTER_FORWARD", "default"
            ),
        ),
        debug=DebugConfig(
            seed=_env_int("GLM53_STREAM_SEED", _env_int("GLM53_FLASH_SEED", 42)),
            deterministic=_env_bool(
                "GLM53_STREAM_DETERMINISTIC",
                _env_bool("GLM53_FLASH_DETERMINISTIC", False),
            ),
            distinct_seed_mesh_axes=["dp_shard"],
            enable_structured_logging=False,
        ),
        checkpointer=Glm53FlashLoRACheckpointManager.Config(
            interval=_env_int("GLM53_STREAM_CHECKPOINT_INTERVAL", 31),
            initial_load_model_only=False,
            initial_load_in_hf=False,
            last_save_model_only=False,
            keep_latest_k=_env_int("GLM53_STREAM_KEEP_LATEST_K", 10),
            load_only=_env_bool("GLM53_STREAM_CHECKPOINT_LOAD_ONLY", False),
            load_frozen_base_from_hf=True,
            trainable_model_only=True,
            verify_frozen_base_load=_env_bool(
                "GLM53_STREAM_VERIFY_FROZEN_BASE_LOAD", True
            ),
        ),
        activation_checkpoint=_activation_checkpoint_config(),
        loss=ChunkedLossWrapper.Config(
            num_chunks=_env_int(
                "GLM53_STREAM_LOSS_CHUNKS",
                _env_int("GLM53_FLASH_LOSS_CHUNKS", 8),
            ),
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=glm_config.vocab_size),
        ),
    )
    return config


def _dump_folder(config_name: str) -> str:
    default = f"{_DEFAULT_DUMP_ROOT}/{config_name}"
    return os.environ.get(
        "GLM53_STREAM_DUMP_FOLDER",
        os.environ.get("GLM53_FLASH_FULL_DUMP_FOLDER", default),
    )


def _model_config(
    *,
    glm_config,
    num_layers: int | None,
    stream: bool,
) -> Glm53FlashForCausalLM.Config:
    model_cls = StreamGlm53FlashForCausalLM if stream else Glm53FlashForCausalLM
    return model_cls.Config(
        glm_config=glm_config,
        experiment_flavor=(
            "glm53_flash" if num_layers is None else f"glm53_flash_first_{num_layers}"
        ),
    )


def _activation_checkpoint_config() -> FullAC.Config | None:
    mode = (
        os.environ.get(
            "GLM53_STREAM_ACTIVATION_CHECKPOINT",
            os.environ.get("GLM53_STREAM_AC", "full"),
        )
        .strip()
        .lower()
    )
    if mode in {"0", "false", "none", "off", "no"}:
        return None
    if mode in {"1", "true", "full", "on", "yes"}:
        return FullAC.Config()
    raise ValueError(
        "GLM53_STREAM_ACTIVATION_CHECKPOINT must be 'full' or 'none', " f"got {mode!r}"
    )


def _apply_attention_env(cfg) -> None:
    cfg.attn_impl = os.environ.get(
        "GLM53_STREAM_ATTN_IMPL",
        os.environ.get("GLM53_FLASH_ATTN_IMPL", "fast"),
    )
    cfg.sparse_attention_mode = os.environ.get(
        "GLM53_STREAM_SPARSE_ATTENTION_MODE",
        os.environ.get("GLM53_FLASH_SPARSE_ATTENTION_MODE", "indexer"),
    )
    cfg.validate()


def _validate_batch_math(
    *,
    seq_len: int,
    local_batch_size: int,
    tokens_per_step: int,
    expected_world_size: int,
) -> None:
    if seq_len != 512:
        raise ValueError("T09a keeps seq length fixed at 512")
    if tokens_per_step != 64 * 512:
        raise ValueError("T09a keeps tokens per optimizer step fixed at 32768")
    if expected_world_size <= 0:
        raise ValueError("GLM53_STREAM_EXPECTED_WORLD_SIZE must be positive")
    tokens_per_accumulation = expected_world_size * local_batch_size * seq_len
    if tokens_per_step % tokens_per_accumulation != 0:
        raise ValueError(
            "GLM53_STREAM_LOCAL_BATCH_SIZE must divide the fixed 32768-token "
            "step at the expected world size"
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
