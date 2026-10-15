# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export GLM-5.3-Flash TorchTitan LoRA checkpoints to PEFT-style files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader, load

from torchtitan.experiments.glm53_flash.flavors import build_model_config, with_lora
from torchtitan.experiments.glm53_flash.model import Glm53FlashForCausalLM
from torchtitan.experiments.glm53_flash.state_dict_adapter import (
    Glm53FlashStateDictAdapter,
)


@dataclass(frozen=True)
class GlmLoRAExportConfig:
    rank: int = 16
    alpha: int = 64
    base_model_name_or_path: str = (
        "/project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16"
    )
    torch_dtype: str = "bfloat16"


def _read_torchtitan_lora_tensors(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    reader = FileSystemReader(checkpoint_dir)
    metadata = reader.read_metadata()
    state: dict[str, torch.Tensor] = {}
    for key, value in metadata.state_dict_metadata.items():
        if not key.endswith((".lora_a", ".lora_b")):
            continue
        size = getattr(value, "size", None)
        properties = getattr(value, "properties", None)
        dtype = getattr(properties, "dtype", torch.bfloat16)
        if size is None:
            continue
        state[key] = torch.empty(tuple(size), dtype=dtype)

    if not state:
        raise ValueError(f"no GLM LoRA tensors found in {checkpoint_dir}")

    load(state, checkpoint_id=str(checkpoint_dir), no_dist=True)
    return state


def _lora_metadata_counts(checkpoint_dir: Path) -> dict[str, int]:
    reader = FileSystemReader(checkpoint_dir)
    metadata = reader.read_metadata()
    exact_lora_keys = 0
    skipped_lora_state_keys = 0
    for key in metadata.state_dict_metadata:
        has_lora_name = ".lora_a" in key or ".lora_b" in key
        if key.endswith((".lora_a", ".lora_b")):
            exact_lora_keys += 1
        elif has_lora_name:
            skipped_lora_state_keys += 1
    return {
        "source_lora_key_count": exact_lora_keys,
        "skipped_lora_state_key_count": skipped_lora_state_keys,
    }


def _state_dict_adapter(config: GlmLoRAExportConfig) -> Glm53FlashStateDictAdapter:
    glm_config = with_lora(
        build_model_config("glm53_flash"),
        rank=config.rank,
        alpha=float(config.alpha),
    )
    model_config = Glm53FlashForCausalLM.Config(
        glm_config=glm_config,
        experiment_flavor="glm53_flash",
    )
    return Glm53FlashStateDictAdapter(
        model_config,
        config.base_model_name_or_path,
    )


def _native_lora_to_peft_key(
    native_key: str,
    adapter: Glm53FlashStateDictAdapter,
) -> str:
    if native_key.endswith(".lora_a"):
        base_native_key = f"{native_key.removesuffix('.lora_a')}.weight"
        side = "lora_A"
    elif native_key.endswith(".lora_b"):
        base_native_key = f"{native_key.removesuffix('.lora_b')}.weight"
        side = "lora_B"
    else:
        raise ValueError(f"not a GLM LoRA tensor key: {native_key}")

    hf_weight_key = adapter._map_native_key(base_native_key)
    if hf_weight_key is None:
        raise ValueError(
            f"cannot map TorchTitan GLM LoRA base key to HF: {base_native_key}"
        )
    if not hf_weight_key.endswith(".weight"):
        raise ValueError(f"expected HF weight key, got: {hf_weight_key}")
    hf_module_key = hf_weight_key.removesuffix(".weight")
    return f"base_model.model.{hf_module_key}.{side}.weight"


def peft_adapter_config(config: GlmLoRAExportConfig) -> dict[str, object]:
    glm_config = with_lora(
        build_model_config("glm53_flash"),
        rank=config.rank,
        alpha=float(config.alpha),
    )
    return {
        "base_model_name_or_path": config.base_model_name_or_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": config.alpha,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": config.rank,
        "target_modules": list(glm_config.lora_target_modules),
        "task_type": "CAUSAL_LM",
        "torch_dtype": config.torch_dtype,
    }


def _adapter_key_shape_check(
    tensors: dict[str, torch.Tensor],
    *,
    rank: int,
) -> dict[str, object]:
    modules: dict[str, set[str]] = {}
    shape_errors: list[str] = []
    for key, tensor in tensors.items():
        if key.endswith(".lora_A.weight"):
            module_key = key.removesuffix(".lora_A.weight")
            side = "A"
            if tensor.ndim < 2 or tensor.shape[0] != rank:
                shape_errors.append(f"{key}: expected lora_A dim0 == {rank}")
        elif key.endswith(".lora_B.weight"):
            module_key = key.removesuffix(".lora_B.weight")
            side = "B"
            if tensor.ndim < 2 or tensor.shape[-1] != rank:
                shape_errors.append(f"{key}: expected lora_B last dim == {rank}")
        else:
            shape_errors.append(f"{key}: unexpected PEFT LoRA key suffix")
            continue
        modules.setdefault(module_key, set()).add(side)

    missing_pairs = [
        module_key for module_key, sides in modules.items() if sides != {"A", "B"}
    ]
    return {
        "module_pair_count": len(modules),
        "missing_pair_count": len(missing_pairs),
        "missing_pair_sample": missing_pairs[:20],
        "shape_error_count": len(shape_errors),
        "shape_error_sample": shape_errors[:20],
    }


def export_glm_lora_adapter(
    checkpoint_dir: Path,
    output_dir: Path,
    config: GlmLoRAExportConfig,
) -> dict[str, object]:
    adapter = _state_dict_adapter(config)
    tensors = _read_torchtitan_lora_tensors(checkpoint_dir)
    metadata_counts = _lora_metadata_counts(checkpoint_dir)
    exported = {
        _native_lora_to_peft_key(key, adapter): tensor.contiguous()
        for key, tensor in tensors.items()
    }
    exported = dict(sorted(exported.items()))
    key_shape_check = _adapter_key_shape_check(exported, rank=config.rank)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(exported, output_dir / "adapter_model.safetensors")
    (output_dir / "adapter_config.json").write_text(
        json.dumps(peft_adapter_config(config), indent=2, sort_keys=True) + "\n"
    )

    hf_target_modules = sorted(
        {
            key.removeprefix("base_model.model.")
            .removesuffix(".lora_A.weight")
            .removesuffix(".lora_B.weight")
            for key in exported
        }
    )
    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "source_tensor_count": len(tensors),
        "exported_tensor_count": len(exported),
        **metadata_counts,
        "rank": config.rank,
        "alpha": config.alpha,
        "base_model_name_or_path": config.base_model_name_or_path,
        "target_modules": peft_adapter_config(config)["target_modules"],
        "hf_target_modules": hf_target_modules,
        "key_shape_check": key_shape_check,
        "sample_exported_keys": list(exported)[:20],
    }
    (output_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument(
        "--base-model-name-or-path",
        default="/project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16",
    )
    parser.add_argument("--torch-dtype", default="bfloat16")
    args = parser.parse_args()

    summary = export_glm_lora_adapter(
        args.checkpoint,
        args.output,
        GlmLoRAExportConfig(
            rank=args.rank,
            alpha=args.alpha,
            base_model_name_or_path=args.base_model_name_or_path,
            torch_dtype=args.torch_dtype,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
