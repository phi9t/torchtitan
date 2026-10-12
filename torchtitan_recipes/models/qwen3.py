# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified Qwen3 model recipes."""

import os
from pathlib import Path

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.text_datasets import ChatProcessor, DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms, LoRATransform
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer


def _countdown_sft_process_sample(sample):
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


def _scaffold_to_policy_sft_process_sample(sample):
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


def _qwen3_countdown_lora_sft(
    *,
    data_file: str,
    dump_folder: str,
    steps: int = 94,
    lora_rank: int = 32,
    lora_alpha: float = 64.0,
    model_flavor: str = "1.7B",
    hf_assets_path: str = "./assets/hf/Qwen3-1.7B",
    initial_load_in_hf: bool = True,
) -> Trainer.Config:
    model_config = build_model_config(model_flavor, seq_len=512, attn_backend="varlen")
    config = Trainer.Config(
        dump_folder=dump_folder,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path=hf_assets_path,
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=1e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=max(1, steps // 20),
                decay_ratio=0.9,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=512,
            num_tokens_per_train_step=64 * 512,
            max_context_length=512,
            steps=steps,
            dtype="bfloat16",
        ),
        dataloader=GrainDataLoader.Config(
            dataset=FirstFitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path="json",
                        split="train",
                        load_dataset_kwargs={"data_files": data_file},
                    ),
                    processor=ChatProcessor.Config(
                        messages_fn=_countdown_sft_process_sample
                    ),
                    post_filters=(lambda sample: sample is not None,),
                ),
            ),
        ),
        checkpointer=CheckpointManager.Config(
            interval=max(1, steps // 3),
            initial_load_in_hf=initial_load_in_hf,
            last_save_model_only=False,
            export_dtype="bfloat16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )
    return apply_transforms(
        config,
        [
            LoRATransform(
                rank=lora_rank,
                alpha=lora_alpha,
                target_modules=["wqkv", "wo", "w13", "w2"],
            )
        ],
    )


def _qwen3_scaffold_to_policy_lora_sft(
    *,
    data_file: str,
    dump_folder: str,
    steps: int,
    lora_rank: int,
    lora_alpha: float,
    model_flavor: str = "1.7B",
    hf_assets_path: str = "./assets/hf/Qwen3-1.7B",
    initial_load_in_hf: bool = True,
) -> Trainer.Config:
    model_config = build_model_config(model_flavor, seq_len=512, attn_backend="varlen")
    config = Trainer.Config(
        dump_folder=dump_folder,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path=hf_assets_path,
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=1e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=max(1, steps // 20),
                decay_ratio=0.9,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=512,
            num_tokens_per_train_step=64 * 512,
            max_context_length=512,
            steps=steps,
            dtype="bfloat16",
        ),
        dataloader=GrainDataLoader.Config(
            dataset=FirstFitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path="json",
                        split="train",
                        load_dataset_kwargs={"data_files": data_file},
                    ),
                    processor=ChatProcessor.Config(
                        messages_fn=_scaffold_to_policy_sft_process_sample
                    ),
                    post_filters=(lambda sample: sample is not None,),
                ),
            ),
        ),
        checkpointer=CheckpointManager.Config(
            interval=max(1, steps // 3),
            initial_load_in_hf=initial_load_in_hf,
            last_save_model_only=False,
            export_dtype="bfloat16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )
    return apply_transforms(
        config,
        [
            LoRATransform(
                rank=lora_rank,
                alpha=lora_alpha,
                target_modules=["wqkv", "wo", "w13", "w2"],
            )
        ],
    )


def _scaffold_to_policy_data_file(name: str) -> str:
    data_root = Path(
        os.environ.get(
            "TORCHTITAN_SCAFFOLD_TO_POLICY_DATA_ROOT",
            "./experiments/scaffold_to_policy/data",
        )
    )
    return str(data_root / "train" / f"{name}.jsonl")


def _scaffold_to_policy_dump_folder(name: str) -> str:
    results_root = Path(
        os.environ.get(
            "TORCHTITAN_SCAFFOLD_TO_POLICY_RESULTS_ROOT",
            "./experiments/scaffold_to_policy/results",
        )
    )
    return str(results_root / "train" / name)


def _scaffold_to_policy_lora_rank() -> int:
    return int(os.environ.get("TORCHTITAN_SCAFFOLD_TO_POLICY_LORA_RANK", "16"))


def _scaffold_to_policy_lora_alpha() -> float:
    return float(os.environ.get("TORCHTITAN_SCAFFOLD_TO_POLICY_LORA_ALPHA", "32.0"))


def _scaffold_to_policy_steps() -> int:
    return int(os.environ.get("TORCHTITAN_SCAFFOLD_TO_POLICY_STEPS", "24"))


def _countdown_data_file(arm: str) -> str:
    data_root = Path(
        os.environ.get(
            "TORCHTITAN_COUNTDOWN_DATA_ROOT",
            "./experiments/countdown_search_distill/data",
        )
    )
    return str(data_root / "train" / f"{arm}.jsonl")


def _countdown_dump_folder(arm: str) -> str:
    results_root = Path(
        os.environ.get(
            "TORCHTITAN_COUNTDOWN_RESULTS_ROOT",
            "./experiments/countdown_search_distill/results",
        )
    )
    return str(results_root / "train" / arm)


def _countdown_lora_rank() -> int:
    return int(os.environ.get("TORCHTITAN_COUNTDOWN_LORA_RANK", "32"))


def _countdown_lora_alpha() -> float:
    return float(os.environ.get("TORCHTITAN_COUNTDOWN_LORA_ALPHA", "64.0"))


def _qwen3_countdown_lora_arm(arm: str) -> Trainer.Config:
    return _qwen3_countdown_lora_sft(
        data_file=_countdown_data_file(arm),
        dump_folder=_countdown_dump_folder(arm),
        lora_rank=_countdown_lora_rank(),
        lora_alpha=_countdown_lora_alpha(),
    )


def qwen3_1_7b_countdown_lora_raw() -> Trainer.Config:
    return _qwen3_countdown_lora_arm("raw")


def qwen3_1_7b_countdown_lora_clean() -> Trainer.Config:
    return _qwen3_countdown_lora_arm("clean")


def qwen3_1_7b_countdown_lora_formatting() -> Trainer.Config:
    return _qwen3_countdown_lora_arm("formatting")


def qwen3_1_7b_countdown_lora_hindsight() -> Trainer.Config:
    return _qwen3_countdown_lora_arm("hindsight")


def qwen3_1_7b_countdown_lora_curriculum() -> Trainer.Config:
    return _qwen3_countdown_lora_arm("curriculum")


def qwen3_debugmodel_countdown_lora_smoke() -> Trainer.Config:
    return _qwen3_countdown_lora_sft(
        data_file=_countdown_data_file("raw"),
        dump_folder=_countdown_dump_folder("debug_smoke"),
        steps=2,
        lora_rank=_countdown_lora_rank(),
        lora_alpha=_countdown_lora_alpha(),
        model_flavor="debugmodel",
        hf_assets_path="./tests/assets/tokenizer",
        initial_load_in_hf=False,
    )


def qwen3_1_7b_modular_sequences_lora_raw() -> Trainer.Config:
    return _qwen3_scaffold_to_policy_lora_sft(
        data_file=_scaffold_to_policy_data_file("modular_sequences_raw"),
        dump_folder=_scaffold_to_policy_dump_folder("modular_sequences_raw"),
        steps=_scaffold_to_policy_steps(),
        lora_rank=_scaffold_to_policy_lora_rank(),
        lora_alpha=_scaffold_to_policy_lora_alpha(),
    )


def qwen3_debugmodel_modular_sequences_lora_smoke() -> Trainer.Config:
    return _qwen3_scaffold_to_policy_lora_sft(
        data_file=_scaffold_to_policy_data_file("modular_sequences_raw"),
        dump_folder=_scaffold_to_policy_dump_folder("modular_sequences_debug_smoke"),
        steps=2,
        lora_rank=_scaffold_to_policy_lora_rank(),
        lora_alpha=_scaffold_to_policy_lora_alpha(),
        model_flavor="debugmodel",
        hf_assets_path="./tests/assets/tokenizer",
        initial_load_in_hf=False,
    )


def qwen3_14b(seq_len: int | None = None) -> Trainer.Config:
    model_config = build_model_config("14B", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3-14B",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=3000,
        ),
        parallelism=ParallelismConfig(),
        checkpointer=None,
        activation_checkpoint=FullAC.Config(),
    )
