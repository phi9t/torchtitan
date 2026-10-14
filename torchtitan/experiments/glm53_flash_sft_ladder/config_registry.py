# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan configs for GLM-5.3 Flash ladder Countdown LoRA SFT."""

from __future__ import annotations

import os
from pathlib import Path

from torchtitan_recipes.models.qwen3 import _qwen3_countdown_lora_sft

from torchtitan.config.transform import apply_transforms, LoRATransform

from torchtitan.trainer import Trainer


_CACHE_ROOT = Path("/data05") / os.environ.get("USER", "user") / "glm53-ladder-cache"
_DEFAULT_DATA_FILE = _CACHE_ROOT / "datasets/countdown-sft/v1/train/formatting.jsonl"
_DEFAULT_RESULTS_ROOT = _CACHE_ROOT / "results/glm53-ladder/rung0"
_DEFAULT_MODEL_ROOT = _CACHE_ROOT / "model-weights/Qwen"


def _env_path(name: str, default: Path) -> str:
    return os.environ.get(name, str(default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _seed() -> int | None:
    value = os.environ.get("GLM53_LADDER_SEED")
    return None if value is None or value == "" else int(value)


def _data_file() -> str:
    return _env_path("GLM53_LADDER_DATA_FILE", _DEFAULT_DATA_FILE)


def _hf_assets_path(model_name: str) -> str:
    env_name = (
        f"GLM53_LADDER_HF_ASSETS_PATH_"
        f"{model_name.upper().replace('.', '_').replace('-', '_')}"
    )
    return _env_path(env_name, _DEFAULT_MODEL_ROOT / f"Qwen3-{model_name}")


def _dump_folder(config_name: str) -> str:
    base = Path(_env_path("GLM53_LADDER_DUMP_ROOT", _DEFAULT_RESULTS_ROOT))
    return os.environ.get("GLM53_LADDER_DUMP_FOLDER", str(base / config_name))


def _finish_config(config: Trainer.Config) -> Trainer.Config:
    config.debug.seed = _seed()
    config.debug.deterministic = _env_bool("GLM53_LADDER_DETERMINISTIC", False)
    config.metrics.enable_tensorboard = True
    return config


def _lora_arm() -> str:
    return os.environ.get("GLM53_LADDER_LORA_ARM", "A").upper()


def _countdown_config(
    *,
    config_name: str,
    model_flavor: str,
    use_lora: bool,
    lr: float,
) -> Trainer.Config:
    config = _qwen3_countdown_lora_sft(
        data_file=_data_file(),
        dump_folder=_dump_folder(config_name),
        steps=_env_int("GLM53_LADDER_STEPS", 94),
        checkpoint_interval=_env_int("GLM53_LADDER_CHECKPOINT_INTERVAL", 31),
        lora_rank=16,
        lora_alpha=64.0,
        lr=lr,
        use_lora=use_lora,
        model_flavor=model_flavor,
        hf_assets_path=_hf_assets_path(model_flavor),
        initial_load_in_hf=True,
    )
    config.checkpointer.keep_latest_k = _env_int("GLM53_LADDER_KEEP_LATEST_K", 10)
    return _finish_config(config)


def _qwen3_30b_a3b_countdown_config() -> Trainer.Config:
    arm = _lora_arm()
    if arm == "A":
        target_modules = ["wqkv", "wo"]
    elif arm == "B":
        target_modules = [
            "wqkv",
            "wo",
            "routed_experts.w13",
            "routed_experts.w2",
        ]
    else:
        raise ValueError(f"GLM53_LADDER_LORA_ARM must be A or B, got {arm!r}.")

    config = _qwen3_countdown_lora_sft(
        data_file=_data_file(),
        dump_folder=_dump_folder("qwen3_30b_a3b_countdown_lora"),
        steps=_env_int("GLM53_LADDER_STEPS", 94),
        checkpoint_interval=_env_int("GLM53_LADDER_CHECKPOINT_INTERVAL", 31),
        lora_rank=16,
        lora_alpha=64.0,
        lr=1e-4,
        use_lora=False,
        model_flavor="30B-A3B",
        hf_assets_path=_hf_assets_path("30B-A3B"),
        initial_load_in_hf=True,
    )
    config.parallelism.expert_parallel_degree = _env_int("GLM53_LADDER_EP_DEGREE", 4)
    config.parallelism.save_parallelism_folder = "parallelism"
    config.training.disable_cuda_graphs = True
    config.training.enable_cpu_offload = _env_bool("GLM53_LADDER_CPU_OFFLOAD", False)
    config.checkpointer.keep_latest_k = _env_int("GLM53_LADDER_KEEP_LATEST_K", 10)
    config = apply_transforms(
        config,
        [
            LoRATransform(
                rank=16,
                alpha=64.0,
                target_modules=target_modules,
            )
        ],
    )
    return _finish_config(config)


def qwen3_0_6b_countdown_lora() -> Trainer.Config:
    return _countdown_config(
        config_name="qwen3_0_6b_countdown_lora",
        model_flavor="0.6B",
        use_lora=True,
        lr=1e-4,
    )


def qwen3_1_7b_countdown_lora() -> Trainer.Config:
    return _countdown_config(
        config_name="qwen3_1_7b_countdown_lora",
        model_flavor="1.7B",
        use_lora=True,
        lr=1e-4,
    )


def qwen3_8b_countdown_lora() -> Trainer.Config:
    return _countdown_config(
        config_name="qwen3_8b_countdown_lora",
        model_flavor="8B",
        use_lora=True,
        lr=1e-4,
    )


def qwen3_30b_a3b_countdown_lora() -> Trainer.Config:
    return _qwen3_30b_a3b_countdown_config()


def qwen3_0_6b_countdown_full() -> Trainer.Config:
    return _countdown_config(
        config_name="qwen3_0_6b_countdown_full",
        model_flavor="0.6B",
        use_lora=False,
        lr=1e-5,
    )
