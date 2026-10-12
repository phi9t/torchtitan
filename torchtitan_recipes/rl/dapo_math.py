# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified single-node Qwen3-4B-Base DAPO-Math recipes."""

from __future__ import annotations

import dataclasses
import os

from renderers import Qwen3RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.renderer import from_renderers
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    BatchInvariantFlexConverter,
    LMHeadFP32OutputConverter,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3 import build_model_config
from torchtitan.rl.components.data import IterableRLDataLoader
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.examples.dapo_math.data import AIME2025Dataset, DapoMathDataset
from torchtitan.rl.examples.dapo_math.env import DapoMathEnv
from torchtitan.rl.examples.dapo_math.rubric import RewardMathVerify
from torchtitan.rl.generator import (
    SamplingConfig,
    VLLMCudaGraphConfig,
    VLLMGenerator,
    VLLMKernelConfig,
)
from torchtitan.rl.losses import DAPOLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rollout.rubric import Rubric
from torchtitan.rl.rollout.token_env import TokenEnv
from torchtitan.rl.trainer import Trainer

_REFERENCE_SEED = 42

# TODO: Enable CUDA graphs for RL trainers after eager/graph numerics parity is
# verified.


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def _dapo_math_rollouter_config(
    *,
    validation_dataset: AIME2025Dataset.Config,
    token_env: TokenEnv.Config,
) -> Rollouter.Config:
    return Rollouter.Config(
        training_dataloader=IterableRLDataLoader.Config(
            dataset=DapoMathDataset.Config()
        ),
        validation_dataset=validation_dataset,
        worker=RolloutWorker.Config(
            rubric=Rubric.Config(
                reward_fns=[RewardMathVerify.Config(weight=1.0)],
                error_reward=0.0,
            ),
            message_env=DapoMathEnv.Config(),
            token_env=token_env,
            advantage=AdvantageEstimator.Config(should_std_normalize=False),
        ),
    )


def _qwen3_4b_dapo_math_config(
    *,
    max_response_tokens: int,
    max_total_tokens: int,
    dump_folder: str,
) -> Controller.Config:
    """Build the shared Qwen3-4B DAPO-Math configuration."""
    num_validation_samples = 30
    validation_dataset = AIME2025Dataset.Config()
    model_config = build_model_config(
        "4B",
        seq_len=max_total_tokens,
        attn_backend="varlen",
        # Compute vocabulary logits in fp32; the rest of the forward uses bf16.
        converters=[LMHeadFP32OutputConverter.Config()],
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="./assets/hf/Qwen3-4B-Base",
        dump_folder=dump_folder,
        async_loop=AsyncLoopConfig(
            num_training_steps=150,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=16,
            target_offpolicy_steps=4,
            validation=ValidationConfig(
                steps=num_validation_samples,
            ),
        ),
        rollouter=_dapo_math_rollouter_config(
            validation_dataset=validation_dataset,
            token_env=TokenEnv.Config(
                max_rollout_tokens=max_total_tokens,
                max_num_turns=1,
            ),
        ),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=True)),
        num_generators=6,
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
            console_log_keys_validation=[
                "validation_reward/_mean",
                "validation_reward/_max",
                "validation/response_length/mean",
                "timing/validate",
            ],
        ),
        trainer=Trainer.Config(
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[
                        AdamW.Config(
                            pattern=r".*",
                            lr=1e-6,
                            betas=(0.9, 0.98),
                            weight_decay=0.1,
                        )
                    ]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=0,
                    min_lr_factor=1.0,
                ),
            ),
            # A minimum factor of 1 keeps the learning rate constant.
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=max_total_tokens,
                max_context_length=max_total_tokens,
            ),
            parallelism=ParallelismConfig(
                data_parallel_replicate_degree=1,
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=100,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                    global_vocab_size=decoder_vocab_size(model_config),
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=1,
            ),
            cuda_graph=VLLMCudaGraphConfig(mode="FULL"),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_response_tokens,
            ),
        ),
    )


def rl_dapo_qwen3_4b_math_8k() -> Controller.Config:
    """Run 8K responses on one node: one TP=2 trainer and six TP=1 generators."""
    return _qwen3_4b_dapo_math_config(
        max_response_tokens=8192,
        max_total_tokens=10240,
        dump_folder="outputs/rl/qwen3_4b_dapo_math_8k",
    )


def rl_dapo_qwen3_4b_math_32k() -> Controller.Config:
    """Run 32K responses on one node: one TP=2 trainer and six TP=1 generators."""
    return _qwen3_4b_dapo_math_config(
        max_response_tokens=32768,
        max_total_tokens=34816,
        dump_folder="outputs/rl/qwen3_4b_dapo_math_32k",
    )


def _rl_batch_invariance_reference_base(
    *,
    dump_folder: str,
    max_response_tokens: int,
    max_total_tokens: int,
    num_training_steps: int,
    validation_steps: int,
    num_prompts_per_train_step: int,
    num_samples_per_prompt: int,
    target_offpolicy_steps: int,
) -> Controller.Config:
    """Shared 8K DAPO-Math reference settings for BI comparison runs."""
    config = _qwen3_4b_dapo_math_config(
        max_response_tokens=max_response_tokens,
        max_total_tokens=max_total_tokens,
        dump_folder=dump_folder,
    )
    config.metrics = MetricsProcessor.Config(
        enable_wandb=False,
        enable_tensorboard=True,
    )
    config.async_loop = dataclasses.replace(
        config.async_loop,
        num_training_steps=num_training_steps,
        num_prompts_per_train_step=num_prompts_per_train_step,
        num_samples_per_prompt=num_samples_per_prompt,
        target_offpolicy_steps=target_offpolicy_steps,
        validation=dataclasses.replace(
            config.async_loop.validation,
            steps=validation_steps,
        ),
    )
    config.trainer = dataclasses.replace(
        config.trainer,
        debug=DebugConfig(seed=_REFERENCE_SEED),
    )
    config.generator = dataclasses.replace(
        config.generator,
        cuda_graph=VLLMCudaGraphConfig(mode="NONE"),
        debug=DebugConfig(seed=_REFERENCE_SEED),
        gpu_memory_limit=0.75,
        kernel=VLLMKernelConfig(
            enable_flashinfer_autotune=False,
            enable_cutedsl_warmup=False,
            enable_jit_warmup=False,
        ),
    )
    return config


def _rl_batch_invariance_reference_from_env(
    *,
    dump_folder: str,
    max_response_tokens: int = 8192,
    max_total_tokens: int = 10240,
    num_training_steps: int = 150,
    validation_steps: int = 30,
    num_prompts_per_train_step: int = 8,
    num_samples_per_prompt: int = 16,
    target_offpolicy_steps: int = 4,
) -> Controller.Config:
    """Build a reference recipe, with script-owned env overrides."""
    return _rl_batch_invariance_reference_base(
        dump_folder=dump_folder,
        max_response_tokens=_env_int(
            "TT_RL_REFERENCE_MAX_TOKENS", max_response_tokens
        ),
        max_total_tokens=_env_int(
            "TT_RL_REFERENCE_MAX_CONTEXT_LENGTH", max_total_tokens
        ),
        num_training_steps=_env_int(
            "TT_RL_REFERENCE_STEPS", num_training_steps
        ),
        validation_steps=_env_int(
            "TT_RL_REFERENCE_VALIDATION_STEPS", validation_steps
        ),
        num_prompts_per_train_step=_env_int(
            "TT_RL_REFERENCE_NUM_PROMPTS", num_prompts_per_train_step
        ),
        num_samples_per_prompt=_env_int(
            "TT_RL_REFERENCE_NUM_SAMPLES", num_samples_per_prompt
        ),
        target_offpolicy_steps=_env_int(
            "TT_RL_REFERENCE_TARGET_OFFPOLICY_STEPS", target_offpolicy_steps
        ),
    )


def _apply_batch_invariant_reference(config: Controller.Config) -> Controller.Config:
    """Make a reference config suitable for bitwise trainer/generator parity."""
    assert config.model is not None
    max_total_tokens = config.trainer.training.max_context_length
    config.model = build_model_config(
        "4B",
        seq_len=max_total_tokens,
        attn_backend="flex",
        converters=[
            BatchInvariantFlexConverter.Config(),
            LMHeadFP32OutputConverter.Config(),
        ],
    )
    # Local compile regions do not support batch-invariant mode.
    config.model.local_compile_regions = []
    block_size = config.model.layers[0].attention.inner_attention.block_size
    config.async_loop = dataclasses.replace(
        config.async_loop,
        target_offpolicy_steps=0,
        windowed_fifo_batches=1,
        batcher=dataclasses.replace(
            config.async_loop.batcher,
            per_sample_pad_multiple=block_size,
        ),
    )
    batch_invariant_debug = DebugConfig(
        seed=_REFERENCE_SEED,
        batch_invariant=True,
        deterministic=True,
    )
    config.num_generators = 3
    config.trainer = dataclasses.replace(
        config.trainer,
        debug=batch_invariant_debug,
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            enable_sequence_parallel=False,
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        debug=batch_invariant_debug,
        parallelism=dataclasses.replace(
            config.generator.parallelism,
            tensor_parallel_degree=2,
        ),
        reset_kv_cache_on_weight_sync=True,
    )
    return config


def rl_dapo_qwen3_4b_math_8k_reference_no_bi() -> Controller.Config:
    """150-step DAPO-Math baseline: TP=2 trainer and six TP=1 generators."""
    return _rl_batch_invariance_reference_from_env(
        dump_folder="outputs/rl_batch_invariance_async/no_bi",
    )


def rl_dapo_qwen3_4b_math_8k_reference_bi() -> Controller.Config:
    """150-step DAPO-Math BI parity run: TP=2 trainer and three TP=2 generators."""
    return _apply_batch_invariant_reference(
        _rl_batch_invariance_reference_from_env(
            dump_folder="outputs/rl_batch_invariance_async/bi",
            target_offpolicy_steps=0,
        )
    )


def rl_dapo_qwen3_4b_math_8k_reference_no_bi_preflight() -> Controller.Config:
    """Bounded startup check for the no-BI reference arm."""
    return _rl_batch_invariance_reference_from_env(
        dump_folder="outputs/rl_batch_invariance_async/preflight_no_bi",
        max_response_tokens=512,
        max_total_tokens=2048,
        num_training_steps=2,
        validation_steps=0,
        num_prompts_per_train_step=1,
        num_samples_per_prompt=1,
        target_offpolicy_steps=0,
    )


def rl_dapo_qwen3_4b_math_8k_reference_bi_preflight() -> Controller.Config:
    """Bounded startup and parity check for the BI reference arm."""
    return _apply_batch_invariant_reference(
        _rl_batch_invariance_reference_from_env(
            dump_folder="outputs/rl_batch_invariance_async/preflight_bi",
            max_response_tokens=512,
            max_total_tokens=2048,
            num_training_steps=2,
            validation_steps=0,
            num_prompts_per_train_step=1,
            num_samples_per_prompt=1,
            target_offpolicy_steps=0,
        )
    )
