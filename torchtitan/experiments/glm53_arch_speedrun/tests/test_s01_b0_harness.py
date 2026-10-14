# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from torchtitan.components.optim import AdamW, DistMuon
from torchtitan.experiments.falcon.bin_reader import write_nanogpt_bin
from torchtitan.experiments.glm53_arch_speedrun.config_registry import (
    glm53_arch_speedrun_b0,
    speedrun_optimizer_config,
)
from torchtitan.experiments.glm53_arch_speedrun.data import FineWebBinDataLoader
from torchtitan.experiments.glm53_arch_speedrun.elements.config import (
    SpeedrunElementsConfig,
)
from torchtitan.experiments.glm53_arch_speedrun.model import (
    adamw_parameter_names,
    Glm53ArchSpeedrunModel,
    muon_parameter_names,
)


def _tiny_elements(seq_len: int = 16, vocab_size: int = 64) -> SpeedrunElementsConfig:
    cfg = SpeedrunElementsConfig(
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        head_dim=8,
        mlp_hidden_size=64,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        indexer_head_dim=8,
    )
    cfg.validate()
    return cfg


def test_fineweb_bin_loader_labels_are_shift_not_roll(tmp_path):
    shard = tmp_path / "tiny.bin"
    write_nanogpt_bin(shard, torch.arange(20, dtype=torch.long))
    loader = FineWebBinDataLoader.Config(bin_glob=str(shard), vocab_size=64).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=None,
        max_context_length=4,
        num_tokens_per_microbatch=8,
    )

    batch = next(iter(loader))

    torch.testing.assert_close(
        batch.input,
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
    )
    torch.testing.assert_close(
        batch.labels,
        torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
    )
    assert batch.labels[-1, -1].item() == 8


def test_muon_and_adamw_partition_covers_each_parameter_once():
    model_config = Glm53ArchSpeedrunModel.Config(elements=_tiny_elements())
    model = Glm53ArchSpeedrunModel(model_config)

    muon_names = muon_parameter_names(model)
    adamw_names = adamw_parameter_names(model)
    trainable_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }

    assert muon_names
    assert adamw_names
    assert muon_names.isdisjoint(adamw_names)
    assert muon_names | adamw_names == trainable_names
    assert "token_embeddings.weight" in adamw_names
    assert "head.proj.weight" in adamw_names
    assert all("norm" not in name for name in muon_names)


def test_optimizer_config_matches_parameter_partition():
    model_config = Glm53ArchSpeedrunModel.Config(elements=_tiny_elements())
    optimizer_config = speedrun_optimizer_config(model_config)

    assert isinstance(optimizer_config.optimizers[0], DistMuon.Config)
    assert isinstance(optimizer_config.optimizers[1], AdamW.Config)
    muon_config = optimizer_config.optimizers[0]
    assert set(muon_config.compute_sharding_by_fqn) == {
        f"blocks.{layer}.{projection}"
        for layer in range(2)
        for projection in (
            "attention.q_proj.weight",
            "attention.k_proj.weight",
            "attention.v_proj.weight",
            "attention.out_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        )
    }


def test_to_empty_then_init_leaves_norm_gains_nonzero():
    model_config = Glm53ArchSpeedrunModel.Config(elements=_tiny_elements())
    with torch.device("meta"):
        model = Glm53ArchSpeedrunModel(model_config)
    model.to_empty(device="cpu")
    model.init_states()

    norm_weights = {
        name: param
        for name, param in model.named_parameters()
        if "norm" in name and name.endswith("weight")
    }
    assert norm_weights
    for name, weight in norm_weights.items():
        assert torch.linalg.vector_norm(weight).item() > 0.0, name
        torch.testing.assert_close(weight, torch.ones_like(weight))


def test_config_contract_uses_single_env_path(monkeypatch):
    monkeypatch.setenv("GLM53_SPEEDRUN_SEQ_LEN", "128")
    monkeypatch.setenv("GLM53_SPEEDRUN_LOCAL_BATCH_SIZE", "2")
    monkeypatch.setenv("GLM53_SPEEDRUN_TOKENS_PER_STEP", "1024")
    monkeypatch.setenv("GLM53_SPEEDRUN_TOKEN_BUDGET", "2048")
    monkeypatch.setenv("GLM53_SPEEDRUN_VAL_TOKENS", "1024")
    monkeypatch.setenv("GLM53_SPEEDRUN_VAL_FREQ", "1")
    monkeypatch.setenv("GLM53_SPEEDRUN_COMPILE", "0")

    config = glm53_arch_speedrun_b0()

    assert config.training.max_context_length == 128
    assert config.model.elements.max_seq_len == 128
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 256
    assert config.training.num_tokens_per_train_step == 1024
    assert config.training.steps == 2
    assert config.validator is not None
    assert config.validator.steps == 4
    assert config.model.local_compile_regions == []
