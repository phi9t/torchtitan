# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-node Qwen3-4B-Base DAPO-Math recipes."""

from __future__ import annotations

import dataclasses

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
    VLLMKernelConfig,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batch_invariance import BatchInvariantFlexConverter
from torchtitan.experiments.rl.components.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.examples.dapo_math.data import AIME2025Dataset
from torchtitan.experiments.rl.examples.dapo_math.rollouter import DapoMathRollouter
from torchtitan.experiments.rl.losses import DAPOLoss
from torchtitan.experiments.rl.models.cast_linear import LMHeadCastConverter
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.routing.inter_generator_router import (
    InterGeneratorRouter,
)
from torchtitan.experiments.rl.routing.strategies import LeastLoadedRoutingStrategy
from torchtitan.models.qwen3 import model_registry


_REFERENCE_SEED = 42


def _qwen3_4b_dapo_math_config(
    *,
    max_response_tokens: int,
    max_total_tokens: int,
    dump_folder: str,
) -> Controller.Config:
    """Build the shared Qwen3-4B DAPO-Math configuration."""
    num_validation_samples = 30
    validation_dataset = AIME2025Dataset.Config(
        num_samples=num_validation_samples,
    )
    return Controller.Config(
        model_spec=model_registry(
            "4B",
            attn_backend="varlen",
            # Compute vocabulary logits in fp32; the rest of the forward uses bf16.
            converters=[LMHeadCastConverter.Config()],
        ),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-4B-Base",
        dump_folder=dump_folder,
        async_loop=AsyncLoopConfig(
            num_training_steps=150,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=16,
            target_offpolicy_steps=4,
            validation=ValidationConfig(
                num_samples=num_validation_samples,
            ),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=1, seq_len=max_total_tokens),
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=DapoMathRollouter.Config(
            validation_dataset=validation_dataset,
            token_env=TokenEnv.Config(
                max_rollout_tokens=max_total_tokens,
                max_num_turns=1,
            ),
        ),
        renderer=RendererConfig(name="qwen3", enable_thinking=True),
        num_generators=6,
        generator_router=InterGeneratorRouter.Config(
            strategy=LeastLoadedRoutingStrategy.Config()
        ),
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
            console_log_keys_validation=[
                "validation_reward/_mean",
                "validation_reward/_max",
                "validation/response_length/mean",
                "timing/validate",
            ],
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(
                lr=1e-6,
                betas=(0.9, 0.98),
                weight_decay=0.1,
            ),
            # A minimum factor of 1 keeps the learning rate constant.
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0,
                min_lr_factor=1.0,
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_replicate_degree=1,
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
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
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=1,
            ),
            cudagraph=VLLMCudagraphConfig(enable=True),
            checkpoint=CheckpointManager.Config(enable=False),
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


def _rl_batch_invariance_reference_base(*, dump_folder: str) -> Controller.Config:
    """Shared 8K DAPO-Math reference settings for BI comparison runs."""
    config = rl_dapo_qwen3_4b_math_8k()
    config.dump_folder = dump_folder
    config.metrics = MetricsProcessor.Config(
        enable_wandb=False,
        enable_tensorboard=True,
    )
    config.trainer = dataclasses.replace(
        config.trainer,
        debug=DebugConfig(seed=_REFERENCE_SEED),
    )
    config.generator = dataclasses.replace(
        config.generator,
        cudagraph=VLLMCudagraphConfig(enable=False),
        debug=DebugConfig(seed=_REFERENCE_SEED),
        gpu_memory_limit=0.75,
        kernel=VLLMKernelConfig(
            enable_flashinfer_autotune=False,
            enable_cutedsl_warmup=False,
            enable_jit_warmup=False,
        ),
    )
    return config


def rl_dapo_qwen3_4b_math_8k_reference_no_bi() -> Controller.Config:
    """150-step DAPO-Math baseline: TP=2 trainer and six TP=1 generators."""
    return _rl_batch_invariance_reference_base(
        dump_folder="outputs/rl_batch_invariance_async/no_bi",
    )


def rl_dapo_qwen3_4b_math_8k_reference_bi() -> Controller.Config:
    """150-step DAPO-Math BI parity run: TP=2 trainer and three TP=2 generators.

    Current batch-invariant parity support requires trainer/generator tensor
    parallelism to match, and strict on-policy generation keeps logprobs
    bitwise-identical at every train step.
    """
    config = _rl_batch_invariance_reference_base(
        dump_folder="outputs/rl_batch_invariance_async/bi",
    )
    config.model_spec = model_registry(
        "4B",
        attn_backend="flex",
        converters=[
            BatchInvariantFlexConverter.Config(),
            LMHeadCastConverter.Config(),
        ],
    )
    block_size = config.model_spec.model.layers[0].attention.inner_attention.block_size
    config.async_loop.batcher = dataclasses.replace(
        config.async_loop.batcher,
        per_sample_pad_multiple=block_size,
    )
    config.async_loop.target_offpolicy_steps = 0
    config.async_loop.window_fraction = None
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
    )
    return config
