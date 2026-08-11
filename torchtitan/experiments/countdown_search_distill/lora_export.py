# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export Countdown TorchTitan LoRA checkpoints to PEFT/vLLM format."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Qwen3LoRAExportConfig:
    rank: int = 32
    alpha: int = 64
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    base_model_name_or_path: str = "./assets/hf/Qwen3-1.7B"
    torch_dtype: str = "bfloat16"


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
]

_LAYER_LORA_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\."
    r"(?P<module>attention\.qkv_linear\.wqkv|attention\.wo|feed_forward\.w1|feed_forward\.w2|feed_forward\.w3)"
    r"\.(?P<side>lora_[ab])\.weight$"
)


def peft_adapter_config(config: Qwen3LoRAExportConfig) -> dict[str, object]:
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
        "target_modules": TARGET_MODULES,
        "task_type": "CAUSAL_LM",
        "torch_dtype": config.torch_dtype,
    }


def split_qwen3_fused_qkv_lora_b(
    tensor: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heads_per_key_value = num_attention_heads // num_key_value_heads
    expected_out_features = (
        num_key_value_heads * (heads_per_key_value + 2) * head_dim
    )
    if tensor.ndim != 2:
        raise ValueError(f"expected a 2D fused QKV LoRA B tensor, got {tensor.shape}")
    if tensor.shape[0] != expected_out_features:
        raise ValueError(
            "fused QKV LoRA B output dimension mismatch: "
            f"got {tensor.shape[0]}, expected {expected_out_features}"
        )

    rank = tensor.shape[1]
    grouped = tensor.reshape(
        num_key_value_heads,
        heads_per_key_value + 2,
        head_dim,
        rank,
    )
    q = grouped[:, :heads_per_key_value].reshape(
        num_attention_heads * head_dim,
        rank,
    )
    k = grouped[:, heads_per_key_value : heads_per_key_value + 1].reshape(
        num_key_value_heads * head_dim,
        rank,
    )
    v = grouped[:, heads_per_key_value + 1 :].reshape(
        num_key_value_heads * head_dim,
        rank,
    )
    return q.contiguous(), k.contiguous(), v.contiguous()


def convert_torchtitan_lora_tensors(
    tensors: dict[str, torch.Tensor],
    config: Qwen3LoRAExportConfig,
) -> dict[str, torch.Tensor]:
    exported: dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if key == "lm_head.lora_a.weight":
            exported["base_model.model.lm_head.lora_A.weight"] = tensor.contiguous()
            continue
        if key == "lm_head.lora_b.weight":
            exported["base_model.model.lm_head.lora_B.weight"] = tensor.contiguous()
            continue

        match = _LAYER_LORA_RE.match(key)
        if match is None:
            raise ValueError(f"unexpected TorchTitan LoRA tensor key: {key}")

        layer = match.group("layer")
        module = match.group("module")
        side = match.group("side")
        prefix = f"base_model.model.model.layers.{layer}"

        if module == "attention.qkv_linear.wqkv":
            if side == "lora_a":
                for projection in ("q_proj", "k_proj", "v_proj"):
                    exported[
                        f"{prefix}.self_attn.{projection}.lora_A.weight"
                    ] = tensor.contiguous().clone()
            else:
                q, k, v = split_qwen3_fused_qkv_lora_b(
                    tensor,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                )
                exported[f"{prefix}.self_attn.q_proj.lora_B.weight"] = q
                exported[f"{prefix}.self_attn.k_proj.lora_B.weight"] = k
                exported[f"{prefix}.self_attn.v_proj.lora_B.weight"] = v
            continue

        peft_module = {
            "attention.wo": "self_attn.o_proj",
            "feed_forward.w1": "mlp.gate_proj",
            "feed_forward.w2": "mlp.down_proj",
            "feed_forward.w3": "mlp.up_proj",
        }[module]
        peft_side = "lora_A" if side == "lora_a" else "lora_B"
        exported[f"{prefix}.{peft_module}.{peft_side}.weight"] = tensor.contiguous()

    return dict(sorted(exported.items()))


def _read_torchtitan_lora_tensors(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    from torch.distributed.checkpoint import FileSystemReader, load

    reader = FileSystemReader(checkpoint_dir)
    metadata = reader.read_metadata()
    state: dict[str, torch.Tensor] = {}
    for key, value in metadata.state_dict_metadata.items():
        if key.startswith("optimizer.") or "lora_" not in key:
            continue
        size = getattr(value, "size", None)
        properties = getattr(value, "properties", None)
        dtype = getattr(properties, "dtype", torch.bfloat16)
        if size is None:
            continue
        state[key] = torch.empty(tuple(size), dtype=dtype)

    if not state:
        raise ValueError(f"no model LoRA tensors found in {checkpoint_dir}")

    load(state, checkpoint_id=str(checkpoint_dir), no_dist=True)
    return state


def export_lora_adapter(
    checkpoint_dir: Path,
    output_dir: Path,
    config: Qwen3LoRAExportConfig,
) -> dict[str, object]:
    from safetensors.torch import save_file

    tensors = _read_torchtitan_lora_tensors(checkpoint_dir)
    exported = convert_torchtitan_lora_tensors(tensors, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(exported, output_dir / "adapter_model.safetensors")
    (output_dir / "adapter_config.json").write_text(
        json.dumps(peft_adapter_config(config), indent=2, sort_keys=True) + "\n"
    )

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "source_tensor_count": len(tensors),
        "exported_tensor_count": len(exported),
        "rank": config.rank,
        "alpha": config.alpha,
        "target_modules": TARGET_MODULES,
    }
    (output_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
