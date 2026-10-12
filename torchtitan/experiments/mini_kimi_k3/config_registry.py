# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan config surface for Mini Kimi K3 experiment plumbing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ModelSpec

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

    model_spec = model_registry("tiny_plumbing")
    return Trainer.Config(
        dump_folder="./experiments/mini_kimi_k3/results/tiny_trainer",
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=MiniK3TokenDataLoader.Config(),
        optimizer=OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={
                        "lr": 1.0e-3,
                        "betas": (0.9, 0.95),
                        "eps": 1.0e-8,
                        "weight_decay": 0.1,
                    },
                )
            ],
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1,
            total_steps=1,
            decay_ratio=0.0,
            decay_type="linear",
            min_lr_factor=1.0,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=16,
            steps=1,
            dtype="bfloat16",
        ),
        debug=DebugConfig(enable_structured_logging=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=1,
            last_save_model_only=False,
        ),
        loss=CrossEntropyLoss.Config(global_vocab_size=32),
    )


def mini_kimi_k3_r1_contract() -> "Trainer.Config":
    """Launchable r1 Trainer config gated by experiment preflight evidence."""

    from torchtitan.trainer import Trainer

    recipe = mini_k3_r1_launch_recipe()
    model_config = mini_k3_r1_config()
    model_spec = model_registry("r1_contract")
    return Trainer.Config(
        dump_folder="./experiments/mini_kimi_k3/results/r1_contract",
        hf_assets_path="./experiments/mini_kimi_k3/assets/kimi-k3-tokenizer",
        tokenizer=MiniK3PretokenizedTokenizer.Config(
            vocab_size=model_config.vocab_size
        ),
        metrics=MetricsProcessor.Config(log_freq=10),
        model_spec=model_spec,
        dataloader=MiniK3TokenDataLoader.Config(),
        optimizer=OptimizersContainer.Config(
            implementation="fused",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*(?:A_log|dt_bias)$",
                    optimizer_name=recipe.optimizer,
                    optimizer_kwargs={
                        "lr": recipe.peak_lr,
                        "betas": recipe.adam_betas,
                        "eps": recipe.adam_eps,
                        "weight_decay": 0.0,
                    },
                ),
                ParamGroupConfig(
                    pattern=r".*(?:norm|layernorm)\.weight$",
                    optimizer_name=recipe.optimizer,
                    optimizer_kwargs={
                        "lr": recipe.peak_lr,
                        "betas": recipe.adam_betas,
                        "eps": recipe.adam_eps,
                        "weight_decay": 0.0,
                    },
                ),
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name=recipe.optimizer,
                    optimizer_kwargs={
                        "lr": recipe.peak_lr,
                        "betas": recipe.adam_betas,
                        "eps": recipe.adam_eps,
                        "weight_decay": recipe.weight_decay,
                    },
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
        training=TrainingConfig(
            local_batch_size=recipe.local_batch_size,
            global_batch_size=recipe.local_batch_size
            * recipe.gradient_accumulation_steps
            * recipe.world_size,
            seq_len=recipe.seq_len,
            steps=recipe.optimizer_steps,
            dtype=recipe.dtype,
        ),
        debug=DebugConfig(enable_structured_logging=False),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=recipe.checkpoint_interval,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
        loss=CrossEntropyLoss.Config(global_vocab_size=model_config.vocab_size),
    )


def model_registry(flavor: str) -> ModelSpec:
    if flavor == "tiny_plumbing":
        return ModelSpec(
            name="mini_kimi_k3",
            flavor=flavor,
            model=MiniK3TinyModel.Config(config=_tiny_training_config()),
            parallelize_fn=_identity_parallelize,
            pipelining_fn=None,
            post_optimizer_build_fn=None,
            state_dict_adapter=None,
        )
    if flavor == "r1_contract":
        return ModelSpec(
            name="mini_kimi_k3",
            flavor=flavor,
            model=MiniK3R1ContractModel.Config(config=mini_k3_r1_config()),
            parallelize_fn=_identity_parallelize,
            pipelining_fn=None,
            post_optimizer_build_fn=None,
            state_dict_adapter=None,
        )
    else:
        raise ValueError(f"unsupported Mini-K3 flavor: {flavor}")


class MiniK3TinyModel(BaseModel):
    """Trainer-compatible wrapper for the CPU-reference Mini-K3 tiny model."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        config: MiniK3Config = field(default_factory=_tiny_training_config)

        def update_from_config(self, *, config, **kwargs) -> None:
            seq_len = config.training.seq_len
            if seq_len > self.config.max_position_embeddings:
                raise ValueError(
                    f"training.seq_len ({seq_len}) exceeds Mini-K3 tiny max "
                    f"position embeddings ({self.config.max_position_embeddings})"
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
        return self.inner(tokens)

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

        def update_from_config(self, *, config, **kwargs) -> None:
            del kwargs
            seq_len = config.training.seq_len
            if seq_len > self.config.max_position_embeddings:
                raise ValueError(
                    f"training.seq_len ({seq_len}) exceeds Mini-K3 r1 max "
                    f"position embeddings ({self.config.max_position_embeddings})"
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
        return self.inner(tokens, positions=positions)

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
            seq_len: int,
            local_batch_size: int,
            snapshot_every_n_steps: int | None = 1,
            **kwargs,
        ) -> "MiniK3TokenDataLoader":
            del tokenizer, snapshot_every_n_steps, kwargs
            return MiniK3TokenDataLoader(
                config=self,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                seq_len=seq_len,
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
            yield {"input": input_ids, "positions": positions}, labels

    def state_dict(self) -> dict[str, Any]:
        return self.loader.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.loader.load_state_dict(state_dict)


def _identity_parallelize(model: torch.nn.Module, **kwargs) -> torch.nn.Module:
    del kwargs
    return model
