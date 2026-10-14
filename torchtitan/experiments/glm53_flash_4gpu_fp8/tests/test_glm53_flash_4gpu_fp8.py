# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from torchtitan.experiments.glm53_flash.flavors import debugmodel, with_lora
from torchtitan.experiments.glm53_flash.model import Glm53FlashForCausalLM
from torchtitan.experiments.glm53_flash_4gpu_fp8.checkpoint import (
    Glm53FlashFP8LoRACheckpointManager,
)
from torchtitan.experiments.glm53_flash_4gpu_fp8.config_registry import (
    glm53_flash_countdown_lora,
    glm53_flash_countdown_lora_truncated_fp8,
    glm53_flash_countdown_lora_truncated_plain,
    glm53_flash_debugmodel_lora,
)
from torchtitan.experiments.glm53_flash_4gpu_fp8.fp8 import (
    dequantize_blockwise_128,
    FP8FrozenLoRALinear,
    quantize_blockwise_128,
)
from torchtitan.experiments.glm53_flash_4gpu_fp8.model import (
    Glm53FlashFP8ForCausalLM,
)
from torchtitan.experiments.glm53_flash_4gpu_fp8.sharded_hf_load import (
    load_rank_local_fp8_from_bf16_safetensors,
)


def test_blockwise_fp8_roundtrip_is_bounded():
    torch.manual_seed(1)
    source = torch.randn(2, 128, 256, dtype=torch.bfloat16) * 0.2

    qdata, scale = quantize_blockwise_128(source)
    restored = dequantize_blockwise_128(qdata, scale, dtype=torch.float32)
    diff = (restored - source.float()).abs()

    assert qdata.dtype == torch.float8_e4m3fn
    assert scale.shape == (2, 1, 2)
    assert diff.max().item() < 0.04
    assert diff.mean().item() < 0.005


def test_fp8_lora_linear_keeps_lora_and_activation_grads():
    torch.manual_seed(2)
    layer = FP8FrozenLoRALinear(128, 256, rank=4, alpha=16.0)
    base = torch.randn(256, 128, dtype=torch.bfloat16) * 0.1
    layer.load_base_from_bf16_(base)
    torch.nn.init.normal_(layer.lora_a, mean=0.0, std=0.01)
    torch.nn.init.normal_(layer.lora_b, mean=0.0, std=0.01)
    x = torch.randn(3, 128, requires_grad=True)

    out = layer(x)
    out.float().square().mean().backward()

    assert layer.weight.grad is None
    assert layer.weight_scale.grad is None
    assert layer.lora_a.grad is not None
    assert layer.lora_b.grad is not None
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_debug_model_uses_fp8_storage_for_selected_frozen_base():
    config = glm53_flash_debugmodel_lora()
    with torch.device("meta"):
        model = config.model.build()

    assert isinstance(model, Glm53FlashFP8ForCausalLM)
    assert model.layers["0"].self_attn.q_proj.weight.dtype != torch.float8_e4m3fn
    assert model.layers["0"].mlp.gate_proj.weight.dtype == torch.float8_e4m3fn
    assert model.layers["3"].self_attn.q_a_proj.weight.dtype == torch.float8_e4m3fn
    assert model.layers["3"].self_attn.kv_b_proj.weight.dtype != torch.float8_e4m3fn
    assert model.layers["3"].self_attn.indexer.wk.weight.dtype != torch.float8_e4m3fn
    assert model.layers["3"].mlp.experts.gate_up_proj.dtype == torch.float8_e4m3fn
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable
    assert all(name.endswith((".lora_a", ".lora_b")) for name in trainable)


def test_fp8_debug_model_forward_backward_is_finite():
    torch.manual_seed(3)
    cfg = with_lora(debugmodel(), rank=4, alpha=16.0)
    cfg.attn_impl = "reference"
    model = Glm53FlashFP8ForCausalLM.Config(glm_config=cfg).build()
    model.to_empty(device="cpu")
    model.init_weights()
    tokens = torch.arange(8).view(1, 8) % cfg.vocab_size

    logits = model(tokens)
    loss = logits.float().square().mean()
    loss.backward()

    assert torch.isfinite(logits).all()
    num_lora_grads = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            continue
        num_lora_grads += int(name.endswith((".lora_a", ".lora_b")))
        assert torch.isfinite(param.grad).all(), name
    assert num_lora_grads > 0


class _TinyAdapter:
    glm_config = SimpleNamespace(moe_intermediate_size=128)

    @staticmethod
    def _map_native_key(native_key: str) -> str | None:
        return {
            "layers.0.self_attn.q_proj.weight": (
                "model.language_model.layers.0.self_attn.q_proj.weight"
            ),
            "layers.0.mlp.gate_proj.weight": (
                "model.language_model.layers.0.mlp.gate_proj.weight"
            ),
        }.get(native_key)


class _TinySelfAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(128, 128, bias=False)


class _TinyExperts(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torchtitan.experiments.glm53_flash_4gpu_fp8.fp8 import FP8FrozenExperts

        fp8 = FP8FrozenExperts(
            num_experts=2,
            hidden_dim=128,
            intermediate_dim=128,
            swiglu_limit=10.0,
        )
        self.gate_up_proj = fp8.gate_up_proj
        self.gate_up_proj_scale = fp8.gate_up_proj_scale
        self.down_proj = fp8.down_proj
        self.down_proj_scale = fp8.down_proj_scale


class _TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = FP8FrozenLoRALinear(128, 128, rank=2, alpha=8.0)
        self.experts = _TinyExperts()


class _TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _TinySelfAttention()
        self.mlp = _TinyMLP()


class _TinyFp8HfLoadModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleDict({"0": _TinyLayer()})


def _write_tiny_checkpoint(root: Path) -> None:
    tensors = {
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.full(
            (128, 128),
            0.25,
            dtype=torch.bfloat16,
        ),
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.linspace(
            -1.0, 1.0, 128 * 128, dtype=torch.bfloat16
        ).view(128, 128),
    }
    for expert in range(2):
        tensors[
            f"model.language_model.layers.0.mlp.experts.{expert}.gate_proj.weight"
        ] = torch.full((128, 128), float(expert + 1), dtype=torch.bfloat16)
        tensors[
            f"model.language_model.layers.0.mlp.experts.{expert}.up_proj.weight"
        ] = torch.full((128, 128), float(expert + 11), dtype=torch.bfloat16)
        tensors[
            f"model.language_model.layers.0.mlp.experts.{expert}.down_proj.weight"
        ] = torch.full((128, 128), float(expert + 21), dtype=torch.bfloat16)
    shard = root / "model-00001-of-00001.safetensors"
    save_file(tensors, shard)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {key: shard.name for key in tensors}})
    )


def test_fp8_rank_local_loader_quantizes_tiny_model_and_keeps_lora(tmp_path):
    _write_tiny_checkpoint(tmp_path)
    model = _TinyFp8HfLoadModel()
    torch.nn.init.normal_(model.layers["0"].mlp.gate_proj.lora_a)
    lora_before = model.layers["0"].mlp.gate_proj.lora_a.detach().clone()

    stats = load_rank_local_fp8_from_bf16_safetensors(
        model,
        _TinyAdapter(),
        tmp_path,
        verify=False,
    )

    gate = model.layers["0"].mlp.gate_proj
    restored = dequantize_blockwise_128(
        gate.weight,
        gate.weight_scale,
        dtype=torch.float32,
    )
    expected = torch.linspace(-1.0, 1.0, 128 * 128).view(128, 128)
    diff = (restored - expected).abs()
    assert diff.max().item() < 0.04
    assert diff.mean().item() < 0.02
    torch.testing.assert_close(gate.lora_a, lora_before)
    assert stats.tensors_quantized_fp8 == 3
    assert stats.tensors_skipped_lora == 2
    assert stats.tensors_skipped_fp8_scale == 3
    assert stats.fp8_qdata_bytes_written == (
        gate.weight.numel()
        + model.layers["0"].mlp.experts.gate_up_proj.numel()
        + model.layers["0"].mlp.experts.down_proj.numel()
    )


def test_full_countdown_config_uses_fp8_model_and_checkpoint(monkeypatch, tmp_path):
    data_file = tmp_path / "formatting.jsonl"
    data_file.write_text(
        '{"question":"Numbers: 1, 2\\nTarget: 3","answer":"1 + 2 = 3\\nFINAL: 3"}\n'
    )
    monkeypatch.setenv("GLM53_FLASH_COUNTDOWN_DATA_FILE", str(data_file))
    monkeypatch.setenv("GLM53_FLASH_COUNTDOWN_HF_ASSETS_PATH", str(tmp_path / "glm53"))

    config = glm53_flash_countdown_lora()

    assert isinstance(config.model, Glm53FlashFP8ForCausalLM.Config)
    assert config.model.experiment_name == "glm53_flash_4gpu_fp8"
    assert isinstance(config.checkpointer, Glm53FlashFP8LoRACheckpointManager.Config)
    assert config.checkpointer.load_frozen_base_from_hf
    assert config.checkpointer.trainable_model_only
    assert config.training.steps == 94
    assert config.training.num_tokens_per_train_step == 64 * 512


def test_truncated_countdown_configs_keep_full_width(monkeypatch, tmp_path):
    data_file = tmp_path / "formatting.jsonl"
    data_file.write_text(
        '{"question":"Numbers: 1, 2\\nTarget: 3","answer":"1 + 2 = 3\\nFINAL: 3"}\n'
    )
    monkeypatch.setenv("GLM53_FLASH_COUNTDOWN_DATA_FILE", str(data_file))
    monkeypatch.setenv("GLM53_FLASH_COUNTDOWN_HF_ASSETS_PATH", str(tmp_path / "glm53"))
    monkeypatch.setenv("GLM53_FP8_NUM_LAYERS", "8")
    monkeypatch.setenv("GLM53_FP8_STEPS", "3")

    fp8_config = glm53_flash_countdown_lora_truncated_fp8()
    plain_config = glm53_flash_countdown_lora_truncated_plain()

    assert isinstance(fp8_config.model, Glm53FlashFP8ForCausalLM.Config)
    assert isinstance(plain_config.model, Glm53FlashForCausalLM.Config)
    assert not isinstance(plain_config.model, Glm53FlashFP8ForCausalLM.Config)
    assert fp8_config.model.glm_config.num_hidden_layers == 8
    assert plain_config.model.glm_config.num_hidden_layers == 8
    assert fp8_config.model.glm_config.hidden_size == 4096
    assert plain_config.model.glm_config.hidden_size == 4096
    assert fp8_config.training.steps == 3
    assert plain_config.training.steps == 3

