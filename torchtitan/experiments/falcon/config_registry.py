# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan config surface for the Falcon overfit experiment."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.experiments.falcon.bin_reader import NanoGptBinDataLoader
from torchtitan.experiments.falcon.model import FalconConfig, FalconForCausalLM
from torchtitan.experiments.falcon.overfit import _initialize_weights
from torchtitan.experiments.falcon.repeat_data import FalconRepeatDataLoader
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ModelSpec


def _identity_parallelize(model: torch.nn.Module, **kwargs) -> torch.nn.Module:
    del kwargs
    return model


def _tiny_overfit_model_config() -> FalconConfig:
    return FalconConfig(variant="falcon1a")


def _science_model_config() -> FalconConfig:
    # Science shape locked by ticket 05/06: 4 layers, hidden 256, 8 heads,
    # head_dim 32, GPT-2 vocab for the FineWeb .bin path. Delayed Falcon
    # default mixer (the arm-agnostic baseline; ablations flip mixer/alignment).
    return FalconConfig(
        vocab_size=50257,
        hidden_size=256,
        num_hidden_layers=4,
        num_heads=8,
        head_dim=32,
        intermediate_size=1024,
        seq_len=512,
        variant="falcon1a",
        alignment="delayed",
        mixer="falcon",
        phi="rms",
    )


# FineWeb 10B GPT-2 shards confirmed on host (ticket 05 report): magic
# 20240520, version 1, uint16 tokens, vocab 50257.
_FINEWEB10B_TRAIN_GLOB = (
    "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa/"
    "data/fineweb10B/fineweb_train_00000[1-9].bin"
)


class FalconTinyModel(BaseModel):
    """Trainer-compatible wrapper over the tiny Falcon causal LM."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        config: FalconConfig = field(default_factory=_tiny_overfit_model_config)

        def update_from_config(self, *, config, **kwargs) -> None:
            del kwargs
            seq_len = config.training.seq_len
            if seq_len > self.config.seq_len:
                raise ValueError(
                    f"training.seq_len ({seq_len}) exceeds Falcon tiny seq_len "
                    f"({self.config.seq_len})"
                )
            self.config.seq_len = seq_len

        def get_nparams_and_flops(
            self, model: torch.nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams = sum(param.numel() for param in model.parameters())
            return nparams, max(1, 2 * nparams)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.inner = FalconForCausalLM(config.config)

    @property
    def lm_head(self) -> torch.nn.Module:
        return self.inner.lm_head

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: object | None = None,
    ) -> torch.Tensor:
        return self.inner(
            tokens,
            positions=positions,
            attention_masks=attention_masks,
        )

    def verify_module_protocol(self) -> None:
        return

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        del buffer_device
        _initialize_weights(self)


def model_registry(flavor: str) -> ModelSpec:
    if flavor == "tiny_overfit":
        model_config = _tiny_overfit_model_config()
    elif flavor == "science":
        model_config = _science_model_config()
    else:
        raise ValueError(f"unsupported Falcon flavor: {flavor}")
    return ModelSpec(
        name="falcon",
        flavor=flavor,
        model=FalconTinyModel.Config(config=model_config),
        parallelize_fn=_identity_parallelize,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


# Step 4 ablation arms A0-A5 (ticket 07). Each arm flips exactly one hypothesized
# knob on the shared science shell; everything else (data, GBS, tokens/step,
# steps, seed policy) is matched by falcon_science(). See spec.md Step 4.
_ARM_KNOBS: dict[str, dict[str, str]] = {
    "A0": {"mixer": "softmax"},
    "A1": {"mixer": "gdn"},
    "A2": {
        "mixer": "falcon",
        "variant": "falcon1a",
        "alignment": "delayed",
        "phi": "rms",
    },
    "A3": {
        "mixer": "falcon",
        "variant": "falcon1",
        "alignment": "delayed",
        "phi": "rms",
    },
    "A4": {
        "mixer": "falcon",
        "variant": "falcon1a",
        "alignment": "same_step",
        "phi": "rms",
    },
    "A5": {
        "mixer": "falcon",
        "variant": "falcon1a",
        "alignment": "delayed",
        "phi": "l2",
    },
}


def apply_arm(config: "Trainer.Config", arm_id: str) -> "Trainer.Config":
    """Overlay arm ``arm_id`` knobs onto a science Trainer config in place.

    The mixer/variant/alignment/phi knobs live on the model's ``FalconConfig``;
    this is the single configuration path (no second knob source).
    """
    if arm_id not in _ARM_KNOBS:
        raise ValueError(
            f"unknown Falcon arm {arm_id!r}; expected one of {list(_ARM_KNOBS)}"
        )
    inner = config.model_spec.model.config
    for key, value in _ARM_KNOBS[arm_id].items():
        setattr(inner, key, value)
    return config


def falcon_tiny_overfit() -> "Trainer.Config":
    from torchtitan.trainer import Trainer

    model_spec = model_registry("tiny_overfit")
    return Trainer.Config(
        dump_folder="./experiments/falcon/results/tiny_overfit",
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=FalconRepeatDataLoader.Config(
            num_sequences=8,
            vocab_size=32,
            seed=0,
        ),
        optimizer=default_adamw(lr=3.0e-2, weight_decay=0.0),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1,
            total_steps=80,
            decay_ratio=0.0,
            decay_type="linear",
            min_lr_factor=1.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=16,
            steps=80,
            dtype="float32",
        ),
        debug=DebugConfig(enable_structured_logging=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=80,
            last_save_model_only=False,
        ),
        loss=CrossEntropyLoss.Config(global_vocab_size=32),
    )


def falcon_science() -> "Trainer.Config":
    """Step 4 science-scale Falcon Trainer config over the FineWeb .bin path.

    The full 8000-step campaign config is registered here (see ticket 05/06),
    but ticket 06 is setup only: the smoke driver runs a few steps with a tiny
    override, never the 8000-step ablation. Claim label: ``smoke`` until ticket
    07 runs the ablations.
    """
    from torchtitan.trainer import Trainer

    model_spec = model_registry("science")
    return Trainer.Config(
        dump_folder="./experiments/falcon/results/science",
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=10),
        model_spec=model_spec,
        dataloader=NanoGptBinDataLoader.Config(
            bin_glob=_FINEWEB10B_TRAIN_GLOB,
            vocab_size=50257,
        ),
        # AdamW betas (0.9, 0.95) per the paper-like science setup.
        optimizer=default_adamw(lr=1.0e-3, weight_decay=0.1),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            total_steps=8000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=32,
            seq_len=512,
            steps=8000,
            dtype="float32",
        ),
        debug=DebugConfig(enable_structured_logging=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=1000,
            last_save_model_only=False,
        ),
        loss=CrossEntropyLoss.Config(global_vocab_size=50257),
    )
