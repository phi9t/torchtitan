# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan configs for the GLM-5.3 architecture ladder."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import LRSchedulersContainer, Optim
from torchtitan.components.validator import Validator
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.glm53_arch_speedrun.config_registry import (
    _train_glob,
    _val_glob,
    DEFAULT_DATA_DIR,
    DEFAULT_TOKEN_BUDGET,
    DEFAULT_TOKENS_PER_STEP,
    FALLBACK_DATA_DIR,
)
from torchtitan.experiments.glm53_arch_speedrun.data import FineWebBinDataLoader
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

from .data import (
    FixedBankFineWebBinDataLoader,
    FixedBankPackedFineWebBinDataLoader,
    PackedFineWebBinDataLoader,
)
from .flavors import build_ladder_model_config, GPT2_VOCAB_SIZE
from .params import ladder_optimizer_config


RUNG_CONFIGS = {
    "L1": "glm53_arch_ladder_l1_qwen3_dense",
    "L2": "glm53_arch_ladder_l2_qwen3_moe",
    "L3": "glm53_arch_ladder_l3_kimi_k3",
    "L4": "glm53_arch_ladder_l4_glm53_flash",
}


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


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    value = default if raw is None else int(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    value = default if raw is None else float(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _data_dir() -> str:
    configured = os.environ.get("GLM53_ARCH_LADDER_DATA_DIR")
    if configured is not None:
        return configured
    default_path = Path(DEFAULT_DATA_DIR)
    if default_path.exists():
        return DEFAULT_DATA_DIR
    return FALLBACK_DATA_DIR


def _training_steps(*, token_budget: int, tokens_per_step: int) -> int:
    explicit = os.environ.get("GLM53_ARCH_LADDER_STEPS")
    if explicit is not None:
        return _env_int("GLM53_ARCH_LADDER_STEPS", 1, minimum=1)
    return max(1, math.ceil(token_budget / tokens_per_step))


def _default_local_batch_size(rung: str, seq_len: int) -> int:
    microbatch_tokens = 8192 if rung.upper() in {"L3", "L4"} else 16384
    return max(1, microbatch_tokens // seq_len)


def _set_cuda_memory_cap() -> None:
    cap_gib = _env_float("GLM53_ARCH_LADDER_GPU_MEMORY_CAP_GIB", 24.0, minimum=0.0)
    if cap_gib <= 0.0 or not torch.cuda.is_available():
        return
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gib / total_gib))


def _base_trainer_config(rung: str) -> Trainer.Config:
    canonical = rung.upper()
    _set_cuda_memory_cap()
    seq_len = _env_int("GLM53_ARCH_LADDER_SEQ_LEN", 4096, minimum=1)
    local_batch_size = _env_int(
        "GLM53_ARCH_LADDER_LOCAL_BATCH_SIZE",
        _default_local_batch_size(canonical, seq_len),
        minimum=1,
    )
    microbatch_tokens = local_batch_size * seq_len
    tokens_per_step = _env_int(
        "GLM53_ARCH_LADDER_TOKENS_PER_STEP",
        DEFAULT_TOKENS_PER_STEP,
        minimum=microbatch_tokens,
    )
    if tokens_per_step % microbatch_tokens != 0:
        raise ValueError(
            "GLM53_ARCH_LADDER_TOKENS_PER_STEP must be divisible by "
            "GLM53_ARCH_LADDER_LOCAL_BATCH_SIZE * GLM53_ARCH_LADDER_SEQ_LEN"
        )
    token_budget = _env_int(
        "GLM53_ARCH_LADDER_TOKEN_BUDGET",
        DEFAULT_TOKEN_BUDGET,
        minimum=1,
    )
    steps = _training_steps(token_budget=token_budget, tokens_per_step=tokens_per_step)
    val_tokens = _env_int(
        "GLM53_ARCH_LADDER_VAL_TOKENS",
        10_485_760,
        minimum=microbatch_tokens,
    )
    val_steps = max(1, math.ceil(val_tokens / microbatch_tokens))
    model = build_ladder_model_config(
        canonical,
        seq_len=seq_len,
        attn_backend=os.environ.get("GLM53_ARCH_LADDER_ATTN_BACKEND", "varlen"),
        glm_attn_impl=os.environ.get("GLM53_ARCH_LADDER_GLM_ATTN_IMPL", "fast"),
        glm_sparse_attention_mode=os.environ.get(
            "GLM53_ARCH_LADDER_GLM_SPARSE_ATTENTION_MODE",
            "indexer",
        ),
    )
    with torch.device("meta"):
        built_model = model.build()
    data_dir = _data_dir()
    is_glm = canonical == "L4"
    data_mode = os.environ.get("GLM53_ARCH_LADDER_DATA_MODE", "fineweb")
    if data_mode not in {"fineweb", "fixed_bank"}:
        raise ValueError(
            "GLM53_ARCH_LADDER_DATA_MODE must be 'fineweb' or 'fixed_bank', "
            f"got {data_mode!r}"
        )
    fixed_bank = data_mode == "fixed_bank"
    if fixed_bank:
        loader_cls = (
            FixedBankFineWebBinDataLoader
            if is_glm
            else FixedBankPackedFineWebBinDataLoader
        )
    else:
        loader_cls = FineWebBinDataLoader if is_glm else PackedFineWebBinDataLoader
    loader_kwargs = {
        "bin_glob": _train_glob(data_dir),
        "vocab_size": GPT2_VOCAB_SIZE,
    }
    val_loader_kwargs = {
        "bin_glob": _train_glob(data_dir) if fixed_bank else _val_glob(data_dir),
        "vocab_size": GPT2_VOCAB_SIZE,
    }
    if fixed_bank:
        bank_microbatches = _env_int(
            "GLM53_ARCH_LADDER_FIXED_BANK_MICROBATCHES",
            1,
            minimum=1,
        )
        loader_kwargs["bank_microbatches"] = bank_microbatches
        val_loader_kwargs["bank_microbatches"] = bank_microbatches
    else:
        loader_kwargs["repeat"] = True
        val_loader_kwargs["repeat"] = True
    warmup_steps = _env_int(
        "GLM53_ARCH_LADDER_WARMUP_STEPS",
        max(1, min(200, steps // 10)),
        minimum=0,
    )
    ac_enabled = _env_bool("GLM53_ARCH_LADDER_ACTIVATION_CHECKPOINT", False)

    return Trainer.Config(
        dump_folder=os.environ.get(
            "GLM53_ARCH_LADDER_DUMP_FOLDER",
            f"experiments/glm53_arch_ladder/results/{canonical.lower()}",
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(
            log_freq=_env_int("GLM53_ARCH_LADDER_LOG_FREQ", 10, minimum=1),
            enable_tensorboard=_env_bool("GLM53_ARCH_LADDER_TENSORBOARD", True),
        ),
        model=model,
        dataloader=loader_cls.Config(**loader_kwargs),
        validator=Validator.Config(
            freq=_env_int("GLM53_ARCH_LADDER_VAL_FREQ", 100, minimum=1),
            steps=val_steps,
            dataloader=loader_cls.Config(**val_loader_kwargs),
        ),
        optim=Optim.Config(
            optimizer=ladder_optimizer_config(
                built_model,
                muon_lr=_env_float("GLM53_ARCH_LADDER_MUON_LR", 0.02, minimum=0.0),
                adamw_lr=_env_float(
                    "GLM53_ARCH_LADDER_ADAMW_LR",
                    8.0e-4,
                    minimum=0.0,
                ),
                weight_decay=_env_float(
                    "GLM53_ARCH_LADDER_WEIGHT_DECAY",
                    0.1,
                    minimum=0.0,
                ),
                momentum=_env_float("GLM53_ARCH_LADDER_MUON_MOMENTUM", 0.95),
                moe_matrix_batches_on_muon=_env_bool(
                    "GLM53_ARCH_LADDER_MOE_MATRIX_BATCHES_ON_MUON",
                    True,
                ),
                moe_router_on_muon=_env_bool(
                    "GLM53_ARCH_LADDER_MOE_ROUTER_ON_MUON",
                    True,
                ),
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=warmup_steps,
                total_steps=steps,
                decay_ratio=_env_float("GLM53_ARCH_LADDER_DECAY_RATIO", 0.2),
                decay_type="linear",
                min_lr_factor=_env_float("GLM53_ARCH_LADDER_MIN_LR_FACTOR", 0.0),
            ),
            max_norm=_env_float("GLM53_ARCH_LADDER_MAX_GRAD_NORM", 1.0, minimum=0.0),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=microbatch_tokens,
            num_tokens_per_train_step=tokens_per_step,
            max_context_length=seq_len,
            steps=steps,
            dtype=os.environ.get("GLM53_ARCH_LADDER_DTYPE", "bfloat16"),
            mixed_precision_param=os.environ.get(
                "GLM53_ARCH_LADDER_MIXED_PRECISION_PARAM",
                "bfloat16",
            ),
            mixed_precision_reduce=os.environ.get(
                "GLM53_ARCH_LADDER_MIXED_PRECISION_REDUCE",
                "float32",
            ),
            disable_cuda_graphs=_env_bool(
                "GLM53_ARCH_LADDER_DISABLE_CUDA_GRAPHS",
                True,
            ),
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            expert_parallel_degree=1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            enable_sequence_parallel=False,
        ),
        debug=DebugConfig(
            seed=_env_int("GLM53_ARCH_LADDER_SEED", 0, minimum=0),
            deterministic=_env_bool("GLM53_ARCH_LADDER_DETERMINISTIC", False),
            enable_structured_logging=_env_bool(
                "GLM53_ARCH_LADDER_STRUCTURED_LOGGING",
                True,
            ),
            save_config_file="config.json",
        ),
        checkpointer=None,
        activation_checkpoint=FullAC.Config() if ac_enabled else None,
        profiler=Profiler.Config(
            enable_profiling=_env_bool(
                "GLM53_ARCH_LADDER_ENABLE_PROFILING",
                False,
            ),
            profile_freq=_env_int(
                "GLM53_ARCH_LADDER_PROFILE_FREQ",
                1,
                minimum=1,
            ),
            profiler_warmup=_env_int(
                "GLM53_ARCH_LADDER_PROFILE_WARMUP",
                0,
                minimum=0,
            ),
            profiler_active=_env_int(
                "GLM53_ARCH_LADDER_PROFILE_ACTIVE",
                1,
                minimum=1,
            ),
            profiler_repeat=_env_int(
                "GLM53_ARCH_LADDER_PROFILE_REPEAT",
                1,
                minimum=1,
            ),
        ),
        loss=ChunkedLossWrapper.Config(
            num_chunks=_env_int("GLM53_ARCH_LADDER_LOSS_CHUNKS", 8, minimum=1),
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=GPT2_VOCAB_SIZE),
        ),
    )


def glm53_arch_ladder_l1_qwen3_dense() -> Trainer.Config:
    return _base_trainer_config("L1")


def glm53_arch_ladder_l2_qwen3_moe() -> Trainer.Config:
    return _base_trainer_config("L2")


def glm53_arch_ladder_l3_kimi_k3() -> Trainer.Config:
    return _base_trainer_config("L3")


def glm53_arch_ladder_l4_glm53_flash() -> Trainer.Config:
    return _base_trainer_config("L4")
