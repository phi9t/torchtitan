# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

import torch

from torchtitan.experiments.glm53_flash.flavors import debugmodel, with_lora
from torchtitan.experiments.glm53_flash.model import (
    _initialize_glm53_flash_weights,
    Glm53FlashForCausalLM,
)
from torchtitan.experiments.glm53_flash_4gpu_lean.config_registry import (
    glm53_flash_countdown_lora_truncated_lean,
    glm53_flash_countdown_lora_truncated_plain,
)
from torchtitan.experiments.glm53_flash_4gpu_lean.memory import (
    cuda_memory_record,
    write_memory_evidence,
)
from torchtitan.experiments.glm53_flash_4gpu_lean.model import (
    iter_split_fsdp_units,
    LeanGlm53FlashForCausalLM,
)


def test_truncated_lean_config_uses_full_width_first_eight_layers(monkeypatch):
    monkeypatch.setenv("GLM53_LEAN_NUM_LAYERS", "8")
    monkeypatch.setenv("GLM53_LEAN_STEPS", "3")
    config = glm53_flash_countdown_lora_truncated_lean()

    assert isinstance(config.model, LeanGlm53FlashForCausalLM.Config)
    assert config.model.glm_config.hidden_size == 4096
    assert config.model.glm_config.num_hidden_layers == 8
    assert config.model.glm_config.sparse_layers == (3, 7)
    assert config.model.glm_config.mlp_layer_types[:4] == (
        "dense",
        "dense",
        "dense",
        "sparse",
    )
    assert config.model.glm_config.lora_rank == 16
    assert config.model.fsdp_wrap_policy == "split"
    assert config.training.steps == 3


def test_truncated_plain_config_keeps_plain_model(monkeypatch):
    monkeypatch.setenv("GLM53_LEAN_NUM_LAYERS", "8")
    monkeypatch.setenv("GLM53_LEAN_STEPS", "3")
    config = glm53_flash_countdown_lora_truncated_plain()

    assert isinstance(config.model, Glm53FlashForCausalLM.Config)
    assert not isinstance(config.model, LeanGlm53FlashForCausalLM.Config)
    assert config.model.glm_config.num_hidden_layers == 8


def test_lean_forward_matches_plain_debug_model():
    glm_config = with_lora(debugmodel(), rank=4, alpha=16.0)
    plain = Glm53FlashForCausalLM(
        Glm53FlashForCausalLM.Config(glm_config=glm_config)
    )
    lean = LeanGlm53FlashForCausalLM(
        LeanGlm53FlashForCausalLM.Config(glm_config=glm_config)
    )
    _initialize_glm53_flash_weights(plain, glm_config)
    lean.load_state_dict(plain.state_dict())

    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    with torch.no_grad():
        torch.testing.assert_close(lean(tokens), plain(tokens), rtol=0, atol=0)


def test_split_fsdp_units_separate_attention_and_experts():
    glm_config = with_lora(debugmodel(), rank=4, alpha=16.0)
    model = LeanGlm53FlashForCausalLM(
        LeanGlm53FlashForCausalLM.Config(glm_config=glm_config)
    )
    sparse_layer = model.layers["3"]
    labels = [unit.label for unit in iter_split_fsdp_units(sparse_layer)]

    assert labels == [
        "attn_hc",
        "self_attn",
        "ffn_hc",
        "mlp.gate",
        "mlp.shared_experts",
        "mlp.experts",
    ]


def test_memory_evidence_writes_rank_local_json(tmp_path):
    record = write_memory_evidence(
        tmp_path,
        label="unit_test",
        include_summary=False,
        include_snapshot=False,
    )
    path = tmp_path / "lean_memory" / "rank00000_unit_test.json"

    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["label"] == "unit_test"
    assert loaded["pid"] == record["pid"]
    assert "rss_gib" in loaded
    assert cuda_memory_record("now")["label"] == "now"

