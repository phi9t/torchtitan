# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan config surface for Mini Kimi K3 experiment plumbing."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.base import BaseModel
from torchtitan.observability.metrics import MetricsProcessor

from .kda import initialize_dt_bias_
from .model import MiniK3ForCausalLM
from .model_contract import mini_k3_r1_config, MiniK3Config
from .recipe import mini_k3_r1_launch_recipe
from .token_shards import load_manifest, TokenShardLoader


def _initialize_mini_k3_states(model: torch.nn.Module, *, std: float) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "dt_bias" in name:
                initialize_dt_bias_(param)
            elif "e_score_correction_bias" in name:
                param.zero_()


def _tiny_training_config() -> MiniK3Config:
    return MiniK3Config(
        name="tiny",
        hidden_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=4,
        routed_expert_hidden_size=4,
        linear_attn_num_heads=2,
        full_attn_layers=(2,),
        kda_layers=(1, 3),
        vocab_size=32,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=3,
        kv_lora_rank=4,
        q_lora_rank=4,
        num_shared_experts=1,
        first_k_dense_replace=1,
        intermediate_size=16,
        max_position_embeddings=16,
        linear_attn_head_dim=4,
        short_conv_kernel_size=1,
    )


class MiniK3PretokenizedTokenizer(BaseTokenizer):
    """Tokenizer stub for Mini-K3 training over already-tokenized uint32 shards."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        vocab_size: int = 163_840

    def __init__(
        self,
        config: Config | None = None,
        *,
        tokenizer_path: str,
    ) -> None:
        super().__init__()
        config = config or self.Config()
        self.tokenizer_path = tokenizer_path
        self.vocab_size = config.vocab_size
        tokenizer_config_path = Path(tokenizer_path) / "tokenizer_config.json"
        if tokenizer_config_path.is_file():
            tokenizer_config = json.loads(tokenizer_config_path.read_text())
            self.bos_token = _token_content(tokenizer_config.get("bos_token"))
            self.eos_token = _token_content(tokenizer_config.get("eos_token"))
        else:
            self.bos_token = None
            self.eos_token = None

    def encode(self, *args, **kwargs) -> list[int]:
        raise NotImplementedError(
            "Mini-K3 training uses pretokenized uint32 shards; text encoding is "
            "not available through MiniK3PretokenizedTokenizer."
        )

    def decode(self, *args, **kwargs) -> str:
        raise NotImplementedError(
            "Mini-K3 training uses pretokenized uint32 shards; text decoding is "
            "not available through MiniK3PretokenizedTokenizer."
        )

    def get_vocab_size(self) -> int:
        return self.vocab_size


def _token_content(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return None


def mini_kimi_k3_tiny_plumbing() -> "Trainer.Config":
    from torchtitan.trainer import Trainer

    return Trainer.Config(
        dump_folder="./experiments/mini_kimi_k3/results/tiny_trainer",
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_registry("tiny_plumbing"),
        dataloader=MiniK3TokenDataLoader.Config(),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[
                    AdamW.Config(
                        pattern=r".*",
                        lr=1.0e-3,
                        betas=(0.9, 0.95),
                        eps=1.0e-8,
                        weight_decay=0.1,
                    ),
                ],
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=1,
                total_steps=1,
                decay_ratio=0.0,
                decay_type="linear",
                min_lr_factor=1.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * 16,
            max_context_length=16,
            steps=1,
            dtype="bfloat16",
        ),
        debug=DebugConfig(enable_structured_logging=False),
        checkpointer=None,
        loss=CrossEntropyLoss.Config(global_vocab_size=32),
    )


def _mini_kimi_r1_optim_config() -> Optim.Config:
    recipe = mini_k3_r1_launch_recipe()
    return Optim.Config(
        optimizer=OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(
                    pattern=r".*(?:A_log|dt_bias)$",
                    lr=recipe.peak_lr,
                    betas=recipe.adam_betas,
                    eps=recipe.adam_eps,
                    weight_decay=0.0,
                ),
                AdamW.Config(
                    pattern=r".*(?:norm|layernorm)\.weight$",
                    lr=recipe.peak_lr,
                    betas=recipe.adam_betas,
                    eps=recipe.adam_eps,
                    weight_decay=0.0,
                ),
                AdamW.Config(
                    pattern=r".*",
                    lr=recipe.peak_lr,
                    betas=recipe.adam_betas,
                    eps=recipe.adam_eps,
                    weight_decay=recipe.weight_decay,
                ),
            ],
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=max(1, round(recipe.optimizer_steps * recipe.warmup_fraction)),
            total_steps=recipe.optimizer_steps,
            decay_ratio=recipe.decay_fraction,
            decay_type="linear",
            min_lr_factor=recipe.min_lr_fraction,
        ),
    )


def mini_kimi_k3_r1_contract() -> "Trainer.Config":
    """Launchable r1 Trainer config gated by experiment preflight evidence."""

    from torchtitan.trainer import Trainer

    recipe = mini_k3_r1_launch_recipe()
    model_config = mini_k3_r1_config()
    return Trainer.Config(
        dump_folder="./experiments/mini_kimi_k3/results/r1_contract",
        hf_assets_path=os.environ.get(
            "MINI_KIMI_K3_HF_ASSETS_PATH",
            "./experiments/mini_kimi_k3/assets/kimi-k3-tokenizer",
        ),
        tokenizer=MiniK3PretokenizedTokenizer.Config(
            vocab_size=model_config.vocab_size
        ),
        metrics=MetricsProcessor.Config(log_freq=10),
        model=model_registry("r1_contract"),
        dataloader=_token_data_config(),
        optim=_mini_kimi_r1_optim_config(),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=recipe.local_batch_size
            * recipe.seq_len,
            num_tokens_per_train_step=recipe.local_batch_size
            * recipe.gradient_accumulation_steps
            * recipe.world_size
            * recipe.seq_len,
            max_context_length=recipe.seq_len,
            steps=recipe.optimizer_steps,
            dtype=recipe.dtype,
        ),
        debug=DebugConfig(enable_structured_logging=False),
        checkpointer=CheckpointManager.Config(
            interval=recipe.checkpoint_interval,
            last_save_model_only=False,
            keep_latest_k=_positive_env_int(
                "MINI_KIMI_K3_CHECKPOINT_KEEP_LATEST_K", 10
            ),
        ),
        activation_checkpoint=FullAC.Config(),
        loss=CrossEntropyLoss.Config(global_vocab_size=model_config.vocab_size),
    )


def mini_kimi_k3_r1_training_smoke() -> "Trainer.Config":
    """One-step Mini-K3 r1 Trainer recipe used by the experiment smoke runner."""

    from torchtitan.trainer import Trainer

    recipe = mini_k3_r1_launch_recipe()
    model_config = mini_k3_r1_config()
    seq_len = _positive_env_int("MINI_KIMI_K3_SMOKE_SEQ_LEN", recipe.seq_len)
    local_batch_size = _positive_env_int(
        "MINI_KIMI_K3_SMOKE_LOCAL_BATCH_SIZE", 1
    )
    global_batch_size = _positive_env_int(
        "MINI_KIMI_K3_SMOKE_GLOBAL_BATCH_SIZE", local_batch_size
    )
    steps = _positive_env_int("MINI_KIMI_K3_SMOKE_STEPS", 1)
    if global_batch_size % local_batch_size != 0:
        raise ValueError(
            "MINI_KIMI_K3_SMOKE_GLOBAL_BATCH_SIZE must be divisible by "
            "MINI_KIMI_K3_SMOKE_LOCAL_BATCH_SIZE"
        )
    hf_assets_path = os.environ.get(
        "MINI_KIMI_K3_HF_ASSETS_PATH",
        "./tests/assets/tokenizer",
    )
    return Trainer.Config(
        dump_folder="./experiments/mini_kimi_k3/results/r1_training_smoke",
        hf_assets_path=hf_assets_path,
        tokenizer=MiniK3PretokenizedTokenizer.Config(
            vocab_size=model_config.vocab_size
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_registry("r1_contract"),
        dataloader=_token_data_config(),
        optim=_mini_kimi_r1_optim_config(),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=local_batch_size * seq_len,
            num_tokens_per_train_step=global_batch_size * seq_len,
            max_context_length=seq_len,
            steps=steps,
            dtype=recipe.dtype,
        ),
        debug=DebugConfig(enable_structured_logging=False),
        checkpointer=None,
        activation_checkpoint=FullAC.Config(),
        loss=CrossEntropyLoss.Config(global_vocab_size=model_config.vocab_size),
    )


def _token_data_config() -> "MiniK3TokenDataLoader.Config":
    return MiniK3TokenDataLoader.Config(
        token_manifest=os.environ.get(
            "MINI_KIMI_K3_TOKEN_MANIFEST",
            "experiments/mini_kimi_k3/data/manifest.json",
        ),
        tokens_dir=os.environ.get(
            "MINI_KIMI_K3_TOKENS_DIR",
            "experiments/mini_kimi_k3/data/tokens",
        ),
    )


def _positive_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def model_registry(flavor: str) -> BaseModel.Config:
    if flavor == "tiny_plumbing":
        return MiniK3TinyModel.Config(
            config=_tiny_training_config(),
            experiment_name="mini_kimi_k3",
            experiment_flavor=flavor,
        )
    if flavor == "r1_contract":
        return MiniK3R1ContractModel.Config(
            config=mini_k3_r1_config(),
            experiment_name="mini_kimi_k3",
            experiment_flavor=flavor,
        )
    else:
        raise ValueError(f"unsupported Mini-K3 flavor: {flavor}")


class MiniK3TinyModel(BaseModel):
    """Trainer-compatible wrapper for the CPU-reference Mini-K3 tiny model."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        config: MiniK3Config = field(default_factory=_tiny_training_config)
        experiment_name: str = "mini_kimi_k3"
        experiment_flavor: str = "tiny_plumbing"

        def update_from_config(self, *, config, **kwargs) -> None:
            del kwargs
            seq_len = config.training.max_context_length
            if seq_len > self.config.max_position_embeddings:
                raise ValueError(
                    "training.max_context_length "
                    f"({seq_len}) exceeds Mini-K3 tiny max position embeddings "
                    f"({self.config.max_position_embeddings})"
                )

        def get_nparams_and_flops(
            self, model: torch.nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams = sum(param.numel() for param in model.parameters())
            # MetricsProcessor requires a positive value for MFU logging. This
            # tiny CPU-reference adapter does not make a performance claim.
            return nparams, max(1, 2 * nparams)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.inner = MiniK3ForCausalLM(config.config)

    @property
    def lm_head(self) -> torch.nn.Module:
        return self.inner.lm_head

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: object | None = None,
    ) -> torch.Tensor:
        del positions, attention_masks
        logits_BLV = self.inner(tokens)
        return logits_BLV.reshape(-1, logits_BLV.shape[-1])

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        del kwargs
        labels = input_dict.pop("labels")
        tokens = input_dict.pop("input")
        input_dict.pop("padding_mask", None)
        return tokens, labels.reshape(-1), input_dict

    def _apply_fsdp(self, **kwargs) -> None:
        del kwargs
        return

    def verify_module_protocol(self) -> None:
        # This adapter deliberately wraps plain PyTorch reference modules. The
        # production Mini-K3 model must replace this with native Module pieces.
        return

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        del buffer_device
        _initialize_mini_k3_states(
            self,
            std=self.config.config.initializer_range,
        )


class MiniK3R1ContractModel(BaseModel):
    """Trainer-compatible wrapper for the source-derived r1 model contract."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        config: MiniK3Config = field(default_factory=mini_k3_r1_config)
        weight_decay_policy: str = "exclude_router_parameters_and_1d_parameters"
        experiment_name: str = "mini_kimi_k3"
        experiment_flavor: str = "r1_contract"

        def update_from_config(self, *, config, **kwargs) -> None:
            del kwargs
            seq_len = config.training.max_context_length
            if seq_len > self.config.max_position_embeddings:
                raise ValueError(
                    "training.max_context_length "
                    f"({seq_len}) exceeds Mini-K3 r1 max position embeddings "
                    f"({self.config.max_position_embeddings})"
                )

        def get_nparams_and_flops(
            self, model: torch.nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams = sum(param.numel() for param in model.parameters())
            return nparams, max(1, 2 * nparams)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.inner = MiniK3ForCausalLM(config.config)

    @property
    def lm_head(self) -> torch.nn.Module:
        return self.inner.lm_head

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: object | None = None,
    ) -> torch.Tensor:
        del attention_masks
        logits_BLV = self.inner(tokens, positions=positions)
        return logits_BLV.reshape(-1, logits_BLV.shape[-1])

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        del kwargs
        labels = input_dict.pop("labels")
        tokens = input_dict.pop("input")
        input_dict.pop("padding_mask", None)
        return tokens, labels.reshape(-1), input_dict

    def _apply_fsdp(self, **kwargs) -> None:
        del kwargs
        return

    def verify_module_protocol(self) -> None:
        # Full launch readiness is decided by the experiment preflight, which
        # requires first-party oracle and launch-backend evidence.
        return

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        del buffer_device
        _initialize_mini_k3_states(
            self,
            std=self.config.config.initializer_range,
        )


class MiniK3TokenDataLoader(BaseDataLoader):
    """Trainer-shaped dataloader over Mini-K3 raw uint32 token shards."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        token_manifest: str = "experiments/mini_kimi_k3/data/manifest.json"
        tokens_dir: str = "experiments/mini_kimi_k3/data/tokens"

        def build(
            self,
            *,
            dp_world_size: int,
            dp_rank: int,
            tokenizer,
            max_context_length: int,
            num_tokens_per_microbatch: int,
            **kwargs,
        ) -> "MiniK3TokenDataLoader":
            del tokenizer, kwargs
            if num_tokens_per_microbatch % max_context_length != 0:
                raise ValueError(
                    "num_tokens_per_microbatch must be divisible by "
                    "max_context_length for Mini-K3 token batches"
                )
            local_batch_size = num_tokens_per_microbatch // max_context_length
            return MiniK3TokenDataLoader(
                config=self,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                seq_len=max_context_length,
                local_batch_size=local_batch_size,
            )

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        seq_len: int,
        local_batch_size: int,
    ) -> None:
        self.config = config
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.loader = TokenShardLoader(
            load_manifest(config.token_manifest),
            Path(config.tokens_dir),
            seq_len=seq_len + 1,
            rank=dp_rank,
            world_size=dp_world_size,
        )

    def __iter__(self):
        while True:
            sequences = [
                self.loader.next_sequence().astype(np.int64, copy=False)
                for _ in range(self.local_batch_size)
            ]
            batch = torch.from_numpy(np.stack(sequences)).long()
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]
            positions = torch.arange(self.seq_len).expand(self.local_batch_size, -1)
            yield TokenizedTrainingMicrobatch(
                input=input_ids,
                labels=labels,
                positions=positions,
                padding_mask=torch.zeros_like(input_ids, dtype=torch.bool),
                loss_token_counts=torch.tensor(labels.numel(), dtype=torch.long),
                routing_token_counts=torch.tensor(
                    labels.numel(), dtype=torch.long
                ).view(1),
            )

    def state_dict(self) -> dict[str, Any]:
        return self.loader.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.loader.load_state_dict(state_dict)
