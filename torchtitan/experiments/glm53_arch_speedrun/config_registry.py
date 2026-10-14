# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config factories for the GLM-5.3 architecture speedrun B0 harness."""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    DistMuon,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.validator import Validator
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.distributed.flex_shard import BucketConfig, ComputeLayout, Owned
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from .data import FineWebBinDataLoader
from .model import b0_elements_config, Glm53ArchSpeedrunModel


DEFAULT_DATA_DIR = (
    "experiments/glm53_arch_speedrun/results/data/"
    "fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1"
)
FALLBACK_DATA_DIR = (
    "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa/"
    "data/fineweb10B"
)
GPT2_VOCAB_SIZE = 50257


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    value = int(os.environ.get(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    value = float(os.environ.get(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _data_dir() -> str:
    configured = os.environ.get("GLM53_SPEEDRUN_DATA_DIR")
    if configured is not None:
        return configured
    default_path = Path(DEFAULT_DATA_DIR)
    if default_path.exists():
        return DEFAULT_DATA_DIR
    return FALLBACK_DATA_DIR


def _train_glob(data_dir: str) -> str:
    return str(Path(data_dir) / "fineweb_train_*.bin")


def _val_glob(data_dir: str) -> str:
    return str(Path(data_dir) / "fineweb_val_000000.bin")


def _default_local_batch_size(seq_len: int) -> int:
    return 1 if seq_len >= 4096 else 4


def _speedrun_model_config(seq_len: int) -> Glm53ArchSpeedrunModel.Config:
    compile_enabled = _env_bool("GLM53_SPEEDRUN_COMPILE", True)
    return Glm53ArchSpeedrunModel.Config(
        elements=b0_elements_config(seq_len=seq_len, vocab_size=GPT2_VOCAB_SIZE),
        experiment_flavor=os.environ.get("GLM53_SPEEDRUN_ARM", "B0"),
        initializer_std=_env_float("GLM53_SPEEDRUN_INIT_STD", 0.02, minimum=0.0),
        logit_softcap=_env_float("GLM53_SPEEDRUN_LOGIT_SOFTCAP", 30.0, minimum=0.0),
        gpu_memory_cap_gib=_env_float(
            "GLM53_SPEEDRUN_GPU_MEMORY_CAP_GIB",
            24.0,
            minimum=0.0,
        ),
        local_compile_regions=(
            ["glm53_speedrun_attention", "glm53_speedrun_mlp"]
            if compile_enabled
            else []
        ),
    )


def _owned_layout() -> ComputeLayout:
    return ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: Owned(),
        },
    )


def _muon_fqns(model_config: Glm53ArchSpeedrunModel.Config) -> list[str]:
    num_layers = model_config.elements.num_layers
    projections = (
        "attention.q_proj.weight",
        "attention.k_proj.weight",
        "attention.v_proj.weight",
        "attention.out_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    return [
        f"blocks.{layer_idx}.{projection}"
        for layer_idx in range(num_layers)
        for projection in projections
    ]


def speedrun_optimizer_config(
    model_config: Glm53ArchSpeedrunModel.Config,
) -> OptimizersContainer.Config:
    muon_fqns = _muon_fqns(model_config)
    compute_layouts = {fqn: _owned_layout() for fqn in muon_fqns}
    bucket_configs = [
        BucketConfig(
            name=f"blocks.{layer_idx}",
            patterns=tuple(
                fqn for fqn in muon_fqns if fqn.startswith(f"blocks.{layer_idx}.")
            ),
        )
        for layer_idx in range(model_config.elements.num_layers)
    ]
    return OptimizersContainer.Config(
        optimizers=[
            DistMuon.Config(
                pattern=(
                    r"^blocks\.[0-9]+\."
                    r"(?:attention\.(?:q_proj|k_proj|v_proj|out_proj)|"
                    r"mlp\.(?:up_proj|down_proj))\.weight$"
                ),
                compute_sharding_by_fqn=compute_layouts,
                bucket_configs=tuple(bucket_configs),
                lr=_env_float("GLM53_SPEEDRUN_MUON_LR", 0.02, minimum=0.0),
                weight_decay=_env_float(
                    "GLM53_SPEEDRUN_MUON_WEIGHT_DECAY",
                    0.1,
                    minimum=0.0,
                ),
                momentum=_env_float("GLM53_SPEEDRUN_MUON_MOMENTUM", 0.95),
                adjust_lr_fn="match_rms_adamw",
            ),
            AdamW.Config(
                pattern=r".*",
                lr=_env_float("GLM53_SPEEDRUN_ADAMW_LR", 8.0e-4, minimum=0.0),
                betas=(0.9, 0.95),
                eps=1.0e-8,
                weight_decay=_env_float(
                    "GLM53_SPEEDRUN_ADAMW_WEIGHT_DECAY",
                    0.1,
                    minimum=0.0,
                ),
                foreach=True,
                fused=False,
            ),
        ],
    )


def _training_steps(*, token_budget: int, tokens_per_step: int) -> int:
    explicit = os.environ.get("GLM53_SPEEDRUN_STEPS")
    if explicit is not None:
        return _env_int("GLM53_SPEEDRUN_STEPS", 1, minimum=1)
    return max(1, math.ceil(token_budget / tokens_per_step))


def glm53_arch_speedrun_b0() -> Trainer.Config:
    """B0 speedrun config over FineWeb10B GPT-2 nanogpt-bin shards."""

    seq_len = _env_int("GLM53_SPEEDRUN_SEQ_LEN", 1024, minimum=1)
    local_batch_size = _env_int(
        "GLM53_SPEEDRUN_LOCAL_BATCH_SIZE",
        _default_local_batch_size(seq_len),
        minimum=1,
    )
    microbatch_tokens = local_batch_size * seq_len
    tokens_per_step = _env_int(
        "GLM53_SPEEDRUN_TOKENS_PER_STEP",
        512 * 1024,
        minimum=microbatch_tokens,
    )
    if tokens_per_step % microbatch_tokens != 0:
        raise ValueError(
            "GLM53_SPEEDRUN_TOKENS_PER_STEP must be divisible by "
            "GLM53_SPEEDRUN_LOCAL_BATCH_SIZE * GLM53_SPEEDRUN_SEQ_LEN"
        )
    token_budget = _env_int(
        "GLM53_SPEEDRUN_TOKEN_BUDGET",
        500_000_000,
        minimum=1,
    )
    steps = _training_steps(token_budget=token_budget, tokens_per_step=tokens_per_step)
    val_tokens = _env_int(
        "GLM53_SPEEDRUN_VAL_TOKENS",
        10_485_760,
        minimum=microbatch_tokens,
    )
    val_steps = max(1, math.ceil(val_tokens / microbatch_tokens))
    data_dir = _data_dir()
    model = _speedrun_model_config(seq_len)
    warmup_steps = _env_int(
        "GLM53_SPEEDRUN_WARMUP_STEPS",
        max(1, min(200, steps // 10)),
        minimum=0,
    )

    return Trainer.Config(
        dump_folder=os.environ.get(
            "GLM53_SPEEDRUN_DUMP_FOLDER",
            f"experiments/glm53_arch_speedrun/results/{model.experiment_flavor.lower()}",
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(
            log_freq=_env_int("GLM53_SPEEDRUN_LOG_FREQ", 10, minimum=1),
            enable_tensorboard=_env_bool("GLM53_SPEEDRUN_TENSORBOARD", True),
        ),
        model=model,
        dataloader=FineWebBinDataLoader.Config(
            bin_glob=_train_glob(data_dir),
            vocab_size=GPT2_VOCAB_SIZE,
            repeat=True,
        ),
        validator=Validator.Config(
            freq=_env_int("GLM53_SPEEDRUN_VAL_FREQ", 100, minimum=1),
            steps=val_steps,
            dataloader=FineWebBinDataLoader.Config(
                bin_glob=_val_glob(data_dir),
                vocab_size=GPT2_VOCAB_SIZE,
                repeat=True,
            ),
        ),
        optim=Optim.Config(
            optimizer=speedrun_optimizer_config(model),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=warmup_steps,
                total_steps=steps,
                decay_ratio=_env_float("GLM53_SPEEDRUN_DECAY_RATIO", 0.2),
                decay_type="linear",
                min_lr_factor=_env_float("GLM53_SPEEDRUN_MIN_LR_FACTOR", 0.0),
            ),
            max_norm=_env_float("GLM53_SPEEDRUN_MAX_GRAD_NORM", 1.0, minimum=0.0),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=microbatch_tokens,
            num_tokens_per_train_step=tokens_per_step,
            max_context_length=seq_len,
            steps=steps,
            dtype=os.environ.get("GLM53_SPEEDRUN_DTYPE", "bfloat16"),
            mixed_precision_param=os.environ.get(
                "GLM53_SPEEDRUN_MIXED_PRECISION_PARAM",
                "bfloat16",
            ),
            mixed_precision_reduce=os.environ.get(
                "GLM53_SPEEDRUN_MIXED_PRECISION_REDUCE",
                "float32",
            ),
            disable_cuda_graphs=_env_bool("GLM53_SPEEDRUN_DISABLE_CUDA_GRAPHS", True),
        ),
        debug=DebugConfig(
            seed=_env_int("GLM53_SPEEDRUN_SEED", 0, minimum=0),
            deterministic=_env_bool("GLM53_SPEEDRUN_DETERMINISTIC", False),
            enable_structured_logging=_env_bool(
                "GLM53_SPEEDRUN_STRUCTURED_LOGGING",
                True,
            ),
            save_config_file="config.json",
        ),
        checkpointer=None,
        loss=ChunkedLossWrapper.Config(
            num_chunks=_env_int("GLM53_SPEEDRUN_LOSS_CHUNKS", 8, minimum=1),
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=GPT2_VOCAB_SIZE),
        ),
    )
