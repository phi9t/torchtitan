# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import cast

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import (
    default_adamw,
    OptimizersContainer,
    ParamGroupConfig,
)
from torchtitan.components.quantization import NVFP4LinearConverter
from torchtitan.components.quantization.nvfp4 import nvfp4_bf16_tail_fqns
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.text_datasets import (
    ChatDataLoader,
    HuggingFaceTextDataLoader,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import model_registry
from .model import Qwen3Model


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
    from torchtitan.components.lora import LoRAConverter

    model_spec = model_registry(
        model_flavor,
        attn_backend="varlen",
        converters=[LoRAConverter.Config(rank=lora_rank, alpha=lora_alpha)],
    )
    return Trainer.Config(
        dump_folder=dump_folder,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path=hf_assets_path,
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        optimizer=default_adamw(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=max(1, steps // 20),
            decay_ratio=0.9,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            global_batch_size=64,
            local_batch_size=1,
            seq_len=512,
            steps=steps,
            dtype="bfloat16",
        ),
        dataloader=ChatDataLoader.Config(
            dataset_path="json",
            load_dataset_kwargs={"data_files": data_file, "split": "train"},
            sample_processor=_countdown_sft_process_sample,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=max(1, steps // 3),
            initial_load_in_hf=initial_load_in_hf,
            last_save_model_only=False,
            export_dtype="bfloat16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


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
    from torchtitan.components.lora import LoRAConverter

    model_spec = model_registry(
        model_flavor,
        attn_backend="varlen",
        converters=[LoRAConverter.Config(rank=lora_rank, alpha=lora_alpha)],
    )
    return Trainer.Config(
        dump_folder=dump_folder,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path=hf_assets_path,
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        optimizer=default_adamw(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=max(1, steps // 20),
            decay_ratio=0.9,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            global_batch_size=64,
            local_batch_size=1,
            seq_len=512,
            steps=steps,
            dtype="bfloat16",
        ),
        dataloader=ChatDataLoader.Config(
            dataset_path="json",
            load_dataset_kwargs={"data_files": data_file, "split": "train"},
            sample_processor=_scaffold_to_policy_sft_process_sample,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=max(1, steps // 3),
            initial_load_in_hf=initial_load_in_hf,
            last_save_model_only=False,
            export_dtype="bfloat16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
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


def qwen3_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen3_debugmodel_fineweb() -> Trainer.Config:
    # Same debug model as qwen3_debugmodel() but reading from the locally
    # prefetched FineWeb-edu subset (see experiments/qwen3_fineweb_hsdp_tp/).
    # Parallelism is left at defaults so the launcher picks the mesh.
    config = qwen3_debugmodel()
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset="fineweb_test")
    # qwen3_debugmodel() leaves checkpointing disabled; enable it so the
    # recorded experiment writes a checkpoint at the final step (interval=10).
    config.checkpoint.enable = True
    return config


def qwen3_debugmodel_nvfp4() -> Trainer.Config:
    config = qwen3_debugmodel()
    config.parallelism.spmd_backend = "spmd_types"
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    # Convert every decoder-layer Linear while leaving the lm_head in bf16.
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            NVFP4LinearConverter.Config(
                fqns=["layers"],
                model_compile_enabled=model_compile_enabled,
            ),
        ],
    )
    return config


def qwen3_debugmodel_first_85_pct_layers_nvfp4() -> Trainer.Config:
    config = qwen3_debugmodel()
    config.parallelism.spmd_backend = "spmd_types"
    assert config.model_spec is not None
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    # Keep the last 15% of decoder layers and the lm_head in bf16.
    num_layers = len(cast(Qwen3Model.Config, config.model_spec.model).layers)
    _NVFP4_BF16_TAIL_FRACTION = 0.15
    fqns = nvfp4_bf16_tail_fqns(
        num_layers,
        _NVFP4_BF16_TAIL_FRACTION,
    )
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            NVFP4LinearConverter.Config(
                fqns=fqns,
                model_compile_enabled=model_compile_enabled,
            ),
        ],
    )
    return config


def qwen3_debugmodel_moe_param_groups() -> Trainer.Config:
    config = qwen3_moe_debug()
    config.optimizer = OptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=r"(?:tok_embeddings|output)\.",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": 8e-4,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            ),
            ParamGroupConfig(
                pattern=r"\.router\.gate\.",
                optimizer_name="Adam",
                optimizer_kwargs={"lr": 1e-4, "betas": (0.9, 0.95), "eps": 1e-8},
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": 8e-4,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                },
            ),
        ],
    )
    return config


def qwen3_debugmodel_flex_flash() -> Trainer.Config:
    model_spec = model_registry("debugmodel", attn_backend="flex_flash")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen3_0_6b() -> Trainer.Config:
    model_spec = model_registry("0.6B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3-0.6B",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen3_1_7b() -> Trainer.Config:
    model_spec = model_registry("1.7B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3-1.7B",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=100,
        ),
        checkpoint=CheckpointManager.Config(
            interval=50,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
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
        lora_rank=_countdown_lora_rank(),
        lora_alpha=_countdown_lora_alpha(),
        steps=2,
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


def qwen3_8b_first_85_pct_layers_nvfp4() -> Trainer.Config:
    config = sft_qwen3_8b_math()
    config.parallelism.spmd_backend = "spmd_types"
    assert config.model_spec is not None
    config.compile = CompileConfig(enable=True, components=["model"])
    # Keep the last 15% of decoder layers and the lm_head in bf16.
    num_layers = len(cast(Qwen3Model.Config, config.model_spec.model).layers)
    _NVFP4_BF16_TAIL_FRACTION = 0.15
    fqns = nvfp4_bf16_tail_fqns(
        num_layers,
        _NVFP4_BF16_TAIL_FRACTION,
    )
    config.model_spec = model_registry(
        "8B",
        attn_backend="varlen",
        converters=[
            NVFP4LinearConverter.Config(
                fqns=fqns,
                model_compile_enabled=True,
            ),
        ],
    )
    return config


def qwen3_14b() -> Trainer.Config:
    model_spec = model_registry("14B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3-14B",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=3000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen3_30b_a3b() -> Trainer.Config:
    model_spec = model_registry("30B-A3B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3-30B-A3B",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=3000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen3_32b() -> Trainer.Config:
    model_spec = model_registry("32B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3-32B",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=3000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen3_debugmodel_non_fused_qkv() -> Trainer.Config:
    # Reverse test: exercise the separate wq/wk/wv path now that fused QKV is
    # the debugmodel default.
    config = qwen3_debugmodel()
    config.model_spec = model_registry("debugmodel_non_fused_qkv")
    return config


def qwen3_moe_debug() -> Trainer.Config:
    model_spec = model_registry("debugmodel_moe")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen3_moe_deepep() -> Trainer.Config:
    """Qwen3 debug MoE pretraining with the DeepEP v2 backend (compact training path), EP=4.

    The MoE expert dispatch uses the DeepEP v2 ElasticBuffer all-to-all; under autograd it
    takes the compact, host-synced, backward-able path. EP=4 (4 GPUs) so the dispatch is
    actually exercised (EP=1 falls back to local); the compact path auto-sizes its buffer from
    the per-rank token count. Numerics match the standard all-to-all backend (step-1 bitwise,
    reduction-order drift thereafter). Needs deep_ep v2 (ElasticBuffer) in the env.

    Local devgpu (no RDMA NIC) needs these env vars so the ElasticBuffer inits NVLink-only:
      - EP_DISABLE_GIN=1            skip the NCCL GIN / RDMA requirement (no RDMA NIC)
      - EP_REUSE_NCCL_COMM=0        avoid the ElasticBuffer null-device-comm segfault
      - NVSHMEM_REMOTE_TRANSPORT=none + NVSHMEM_DISABLE_MNNVL=1   intra-node NVLink only
      - LD_LIBRARY_PATH must include the deep_ep wheels' nvshmem + nccl lib dirs
    Then launch with NGPU=4 ./run_train.sh (none of this is needed on RDMA/RoCE hosts).
    """
    model_spec = model_registry("debugmodel_moe", moe_comm_backend="deepep")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(local_batch_size=2, seq_len=512, steps=10),
        parallelism=ParallelismConfig(expert_parallel_degree=4),
        checkpoint=CheckpointManager.Config(
            interval=1000, last_save_model_only=False, export_dtype="float16"
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def sft_qwen3_8b_math() -> Trainer.Config:
    """Qwen3-8B SFT on GSM8K math dataset."""

    def process_sample(sample):
        answer = sample["answer"]
        reasoning, final_answer = answer.rsplit("####", 1)
        return [
            {"role": "user", "content": sample["question"]},
            {
                "role": "assistant",
                "reasoning_content": reasoning.strip(),
                "content": final_answer.strip(),
            },
        ]

    model_spec = model_registry("8B", attn_backend="varlen")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3-8B",
        model_spec=model_spec,
        optimizer=default_adamw(lr=2e-5),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=15,
            decay_ratio=0.9,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=2048,
            steps=180,
        ),
        dataloader=ChatDataLoader.Config(
            dataset_path="openai/gsm8k",
            load_dataset_kwargs={"name": "main", "split": "train"},
            sample_processor=process_sample,
        ),
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )
