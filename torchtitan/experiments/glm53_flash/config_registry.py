# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan config factories for the GLM-5.3-Flash experiment."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.text_datasets import ChatProcessor
from torchtitan.components.data.types import (
    TokenizedTrainingMicrobatch,
    TrainingMicrobatch,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.base.model import BaseModel
from torchtitan.observability.metrics import MetricsProcessor

from .checkpoint import Glm53FlashLoRACheckpointManager
from .flavors import build_model_config, with_lora
from .model import Glm53FlashForCausalLM


class Glm53PretokenizedTokenizer(BaseTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        vocab_size: int = 154_880

    def __init__(self, config: Config | None = None, *, tokenizer_path: str) -> None:
        super().__init__()
        del tokenizer_path
        config = config or self.Config()
        self.vocab_size = config.vocab_size
        self.eos_id = 154820 if self.vocab_size > 154820 else self.vocab_size - 1

    def encode(self, *args: Any, **kwargs: Any) -> list[int]:
        raise NotImplementedError("GLM smoke configs use pretokenized ids.")

    def decode(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("GLM smoke configs use pretokenized ids.")

    def get_vocab_size(self) -> int:
        return self.vocab_size


class Glm53SyntheticDataLoader(BaseDataLoader):
    """Deterministic next-token synthetic data for launch and smoke tests."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        vocab_size: int = 512
        seed: int = 2718
        countdown_tokens_path: str | None = None

        def build(
            self,
            *,
            dp_world_size: int,
            dp_rank: int,
            tokenizer,
            max_context_length: int,
            num_tokens_per_microbatch: int,
            **kwargs: Any,
        ) -> "Glm53SyntheticDataLoader":
            del tokenizer, kwargs
            return Glm53SyntheticDataLoader(
                self,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                max_context_length=max_context_length,
                num_tokens_per_microbatch=num_tokens_per_microbatch,
            )

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        max_context_length: int,
        num_tokens_per_microbatch: int,
    ) -> None:
        if num_tokens_per_microbatch % max_context_length != 0:
            raise ValueError(
                "num_tokens_per_microbatch must be divisible by max_context_length"
            )
        self.config = config
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.max_context_length = max_context_length
        self.local_batch_size = num_tokens_per_microbatch // max_context_length
        self.sample_offset = 0

    def __iter__(self) -> Iterator[TrainingMicrobatch]:
        while True:
            ids_B = (
                torch.arange(self.local_batch_size, dtype=torch.long)
                + self.sample_offset * self.local_batch_size
            )
            ids_B = ids_B * self.dp_world_size + self.dp_rank
            base_BL = torch.arange(self.max_context_length + 1, dtype=torch.long)
            batch = (
                base_BL.unsqueeze(0) + ids_B.unsqueeze(1) * 17 + self.config.seed
            ) % self.config.vocab_size
            self.sample_offset += 1
            input_ids_BL = batch[:, :-1].contiguous()
            labels_BL = batch[:, 1:].contiguous()
            positions_BL = torch.arange(self.max_context_length).expand_as(input_ids_BL)
            yield TokenizedTrainingMicrobatch(
                input=input_ids_BL,
                labels=labels_BL,
                positions=positions_BL,
                padding_mask=torch.zeros_like(input_ids_BL, dtype=torch.bool),
                loss_token_counts=torch.tensor(labels_BL.numel(), dtype=torch.long),
                routing_token_counts=torch.tensor(
                    labels_BL.numel(), dtype=torch.long
                ).view(1),
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "sample_offset": self.sample_offset,
            "dp_world_size": self.dp_world_size,
            "dp_rank": self.dp_rank,
            "max_context_length": self.max_context_length,
            "local_batch_size": self.local_batch_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if state_dict["dp_world_size"] != self.dp_world_size:
            raise ValueError("cannot resume GLM synthetic data with changed DP size")
        if state_dict["dp_rank"] != self.dp_rank:
            raise ValueError("GLM synthetic data checkpoint rank mismatch")
        self.sample_offset = state_dict["sample_offset"]


def _countdown_sft_process_sample(sample: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


def _countdown_data_file() -> str:
    data_root = Path(
        os.environ.get(
            "GLM53_FLASH_COUNTDOWN_DATA_ROOT",
            "/project/formal-cache/datasets/countdown-sft/v1",
        )
    )
    return os.environ.get(
        "GLM53_FLASH_COUNTDOWN_DATA_FILE",
        str(data_root / "train" / "formatting.jsonl"),
    )


def _countdown_hf_assets_path() -> str:
    return os.environ.get(
        "GLM53_FLASH_COUNTDOWN_HF_ASSETS_PATH",
        os.environ.get(
            "GLM53_FLASH_HF_ASSETS_PATH",
            "/project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16",
        ),
    )


def _countdown_data_loader_config(*, seed: int = 42) -> GrainDataLoader.Config:
    return GrainDataLoader.Config(
        max_num_documents=_positive_env_int("GLM53_FLASH_COUNTDOWN_MAX_DOCS", 64),
        dataset=FirstFitPackingConfig(
            dataset=SingleDatasetConfig(
                source=HuggingFaceRandomAccessSource.Config(
                    path="json",
                    split=os.environ.get("GLM53_FLASH_COUNTDOWN_SPLIT", "train[:64]"),
                    load_dataset_kwargs={"data_files": _countdown_data_file()},
                ),
                processor=ChatProcessor.Config(
                    messages_fn=_countdown_sft_process_sample
                ),
                post_filters=(lambda sample: sample is not None,),
            ),
        ),
        seed=seed,
        shuffle=_env_bool("GLM53_FLASH_COUNTDOWN_SHUFFLE", False),
        repeat=True,
    )


def _apply_attention_env(cfg) -> None:
    cfg.attn_impl = os.environ.get("GLM53_FLASH_ATTN_IMPL", "fast")
    cfg.sparse_attention_mode = os.environ.get(
        "GLM53_FLASH_SPARSE_ATTENTION_MODE", "indexer"
    )
    cfg.validate()


def _mini_countdown_model_config(*, use_lora: bool):
    cfg = build_model_config("mini")
    cfg.vocab_size = _positive_env_int("GLM53_FLASH_COUNTDOWN_VOCAB_SIZE", 151_936)
    _apply_attention_env(cfg)
    if use_lora:
        return with_lora(cfg, rank=16, alpha=64.0)
    return cfg


def _mini_countdown_trainer_config(
    cfg,
    *,
    dump_folder: str,
    steps: int,
    lr: float,
) -> "Trainer.Config":
    from torchtitan.trainer import Trainer

    seq_len = _positive_env_int("GLM53_FLASH_MINI_SEQ_LEN", 512)
    local_batch_size = _positive_env_int("GLM53_FLASH_MINI_LOCAL_BATCH_SIZE", 1)
    tokens_per_step = _positive_env_int("GLM53_FLASH_MINI_TOKENS_PER_STEP", 64 * 512)
    return Trainer.Config(
        dump_folder=dump_folder,
        hf_assets_path=_countdown_hf_assets_path(),
        tokenizer=HuggingFaceTokenizer.Config(),
        model=Glm53FlashForCausalLM.Config(
            glm_config=cfg,
            experiment_flavor="mini",
        ),
        dataloader=_countdown_data_loader_config(
            seed=_positive_env_int("GLM53_FLASH_COUNTDOWN_SEED", 42)
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        optim=_optim_config(steps=steps, lr=lr),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=local_batch_size * seq_len,
            num_tokens_per_train_step=tokens_per_step,
            max_context_length=seq_len,
            steps=steps,
            dtype="bfloat16",
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            expert_parallel_degree=1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            enable_sequence_parallel=False,
        ),
        debug=DebugConfig(
            seed=42, deterministic=False, enable_structured_logging=False
        ),
        checkpointer=None,
        activation_checkpoint=FullAC.Config(),
        loss=ChunkedLossWrapper.Config(
            num_chunks=_positive_env_int("GLM53_FLASH_LOSS_CHUNKS", 8),
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=cfg.vocab_size),
        ),
    )


def model_registry(flavor: str) -> BaseModel.Config:
    return Glm53FlashForCausalLM.Config(
        glm_config=build_model_config(flavor),
        experiment_flavor=flavor,
    )


def glm53_flash_debugmodel() -> "Trainer.Config":
    from torchtitan.trainer import Trainer

    cfg = build_model_config("debugmodel")
    _apply_attention_env(cfg)
    seq_len = _positive_env_int("GLM53_FLASH_DEBUG_SEQ_LEN", 128)
    local_batch_size = _positive_env_int("GLM53_FLASH_DEBUG_LOCAL_BATCH_SIZE", 1)
    steps = _positive_env_int("GLM53_FLASH_DEBUG_STEPS", 20)
    return Trainer.Config(
        dump_folder=os.environ.get(
            "GLM53_FLASH_DUMP_FOLDER",
            "./outputs/glm53_flash_debugmodel",
        ),
        hf_assets_path=os.environ.get(
            "GLM53_FLASH_HF_ASSETS_PATH", "./tests/assets/tokenizer"
        ),
        tokenizer=Glm53PretokenizedTokenizer.Config(vocab_size=cfg.vocab_size),
        model=Glm53FlashForCausalLM.Config(
            glm_config=cfg,
            experiment_flavor="debugmodel",
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
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            expert_parallel_degree=1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            enable_sequence_parallel=False,
        ),
        debug=DebugConfig(
            seed=42, deterministic=False, enable_structured_logging=False
        ),
        checkpointer=None,
        activation_checkpoint=FullAC.Config(),
        loss=CrossEntropyLoss.Config(global_vocab_size=cfg.vocab_size),
    )


def glm53_flash_debugmodel_lora() -> "Trainer.Config":
    config = glm53_flash_debugmodel()
    assert isinstance(config.model, Glm53FlashForCausalLM.Config)
    config.model.glm_config = with_lora(config.model.glm_config, rank=8, alpha=32.0)
    config.optim = _lora_optim_config(steps=config.training.steps, lr=1.0e-3)
    config.dump_folder = os.environ.get(
        "GLM53_FLASH_LORA_DUMP_FOLDER",
        "./outputs/glm53_flash_debugmodel_lora",
    )
    return config


def glm53_flash_mini_countdown_lora() -> "Trainer.Config":
    cfg = _mini_countdown_model_config(use_lora=True)
    config = _mini_countdown_trainer_config(
        cfg,
        dump_folder=os.environ.get(
            "GLM53_FLASH_MINI_DUMP_FOLDER",
            "./outputs/glm53_flash_mini_countdown_lora",
        ),
        steps=_positive_env_int("GLM53_FLASH_MINI_STEPS", 20),
        lr=_nonnegative_env_float("GLM53_FLASH_MINI_LR", 2.0e-4),
    )
    config.optim = _lora_optim_config(
        steps=config.training.steps,
        lr=_nonnegative_env_float("GLM53_FLASH_MINI_LR", 2.0e-4),
    )
    return config


def glm53_flash_mini_countdown_full() -> "Trainer.Config":
    cfg = _mini_countdown_model_config(use_lora=False)
    return _mini_countdown_trainer_config(
        cfg,
        dump_folder=os.environ.get(
            "GLM53_FLASH_MINI_FULL_DUMP_FOLDER",
            "./outputs/glm53_flash_mini_countdown_full",
        ),
        steps=_positive_env_int("GLM53_FLASH_MINI_STEPS", 300),
        lr=_nonnegative_env_float("GLM53_FLASH_MINI_LR", 1.0e-3),
    )


def glm53_flash_countdown_lora() -> "Trainer.Config":
    config = glm53_flash_mini_countdown_lora()
    assert isinstance(config.model, Glm53FlashForCausalLM.Config)
    full_cfg = with_lora(build_model_config("glm53_flash"), rank=16, alpha=64.0)
    _apply_attention_env(full_cfg)
    config.model = Glm53FlashForCausalLM.Config(
        glm_config=full_cfg,
        experiment_flavor="glm53_flash",
    )
    config.tokenizer = HuggingFaceTokenizer.Config()
    config.hf_assets_path = _countdown_hf_assets_path()
    config.dataloader = _countdown_data_loader_config()
    config.loss = ChunkedLossWrapper.Config(
        num_chunks=_positive_env_int("GLM53_FLASH_LOSS_CHUNKS", 8),
        loss_fn=CrossEntropyLoss.Config(global_vocab_size=full_cfg.vocab_size),
    )
    config.dump_folder = os.environ.get(
        "GLM53_FLASH_FULL_DUMP_FOLDER",
        "./outputs/glm53_flash_countdown_lora",
    )
    config.training.max_context_length = _positive_env_int(
        "GLM53_FLASH_FULL_SEQ_LEN", 512
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = (
        _positive_env_int("GLM53_FLASH_FULL_LOCAL_BATCH_SIZE", 1)
        * config.training.max_context_length
    )
    config.training.num_tokens_per_train_step = _positive_env_int(
        "GLM53_FLASH_FULL_TOKENS_PER_STEP", 64 * 512
    )
    config.training.steps = _positive_env_int("GLM53_FLASH_FULL_STEPS", 94)
    if _env_bool("GLM53_FLASH_FULL_CPU_OFFLOAD", False):
        raise ValueError(
            "GLM53_FLASH_FULL_CPU_OFFLOAD is disabled for the full GLM rung. "
            "Use mode A: meta init, FSDP-sharded GPU params, and rank-local "
            "HF/DCP shard loading."
        )
    config.training.enable_cpu_offload = False
    config.checkpointer = Glm53FlashLoRACheckpointManager.Config(
        interval=_positive_env_int("GLM53_FLASH_FULL_CHECKPOINT_INTERVAL", 31),
        initial_load_model_only=False,
        initial_load_in_hf=False,
        last_save_model_only=False,
        keep_latest_k=_nonnegative_env_int("GLM53_FLASH_FULL_KEEP_LATEST_K", 10),
        load_only=_env_bool("GLM53_FLASH_FULL_CHECKPOINT_LOAD_ONLY", False),
    )
    config.optim = _lora_optim_config(
        steps=config.training.steps,
        lr=_nonnegative_env_float("GLM53_FLASH_FULL_LR", 1.0e-4),
    )
    return config


def _lora_optim_config(*, steps: int, lr: float) -> Optim.Config:
    return Optim.Config(
        optimizer=OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(
                    pattern=r".*\.lora_[ab]$",
                    lr=lr,
                    betas=(0.9, 0.95),
                    eps=1.0e-8,
                    weight_decay=0.0,
                ),
            ],
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=max(1, steps // 10),
            total_steps=steps,
            decay_ratio=0.0,
            decay_type="linear",
            min_lr_factor=1.0,
        ),
    )


def _optim_config(*, steps: int, lr: float) -> Optim.Config:
    return Optim.Config(
        optimizer=OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(
                    pattern=r".*(?:A_log|dt_bias|e_score_correction_bias)$",
                    lr=lr,
                    betas=(0.9, 0.95),
                    eps=1.0e-8,
                    weight_decay=0.0,
                ),
                AdamW.Config(
                    pattern=r".*",
                    lr=lr,
                    betas=(0.9, 0.95),
                    eps=1.0e-8,
                    weight_decay=0.1,
                ),
            ],
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=max(1, steps // 10),
            total_steps=steps,
            decay_ratio=0.0,
            decay_type="linear",
            min_lr_factor=1.0,
        ),
    )


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _nonnegative_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _nonnegative_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


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
