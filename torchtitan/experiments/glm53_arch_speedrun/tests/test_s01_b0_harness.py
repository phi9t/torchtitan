# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

from torchtitan.components.optim import AdamW, DistMuon
from torchtitan.experiments.falcon.bin_reader import write_nanogpt_bin
from torchtitan.experiments.glm53_arch_speedrun.config_registry import (
    glm53_arch_speedrun_b0,
    speedrun_optimizer_config,
)
from torchtitan.experiments.glm53_arch_speedrun.data import FineWebBinDataLoader
from torchtitan.experiments.glm53_arch_speedrun.elements.config import (
    compose_arms,
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
        num_layers=4,
        num_attention_heads=4,
        head_dim=8,
        mlp_hidden_size=64,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        indexer_head_dim=8,
    )
    cfg.validate()
    return cfg


def _tiny_arm_elements(*arms: str) -> SpeedrunElementsConfig:
    cfg = compose_arms(*arms, base=_tiny_elements())
    cfg.validate()
    return cfg


def _trainable_names(model: torch.nn.Module) -> set[str]:
    return {name for name, param in model.named_parameters() if param.requires_grad}


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


def test_muon_and_adamw_partition_covers_each_parameter_once_for_every_arm():
    arm_configs = [
        _tiny_elements(),
        _tiny_arm_elements("A1"),
        _tiny_arm_elements("A2"),
        _tiny_arm_elements("A3"),
        _tiny_arm_elements("A4"),
        _tiny_arm_elements("A5"),
        _tiny_arm_elements("A6"),
        _tiny_arm_elements("A1", "A2", "A3", "A5", "A6"),
    ]
    for elements in arm_configs:
        model_config = Glm53ArchSpeedrunModel.Config(elements=elements)
        model = Glm53ArchSpeedrunModel(model_config)

        muon_names = muon_parameter_names(model)
        adamw_names = adamw_parameter_names(model)
        trainable_names = _trainable_names(model)

        assert muon_names, elements
        assert adamw_names, elements
        assert muon_names.isdisjoint(adamw_names)
        assert muon_names | adamw_names == trainable_names
        assert "token_embeddings.weight" in adamw_names
        assert "head.proj.weight" in adamw_names
        assert all("norm" not in name for name in muon_names)


def test_full_f1_f5_factorial_builds_and_runs_two_toy_steps():
    sparse_arms = {
        "dense": (),
        "indexer_topk": ("A4",),
        "indexer_topk_indexpool": ("A5",),
    }
    for mhc, nope, kda, sparse, mtp in itertools.product(
        (False, True),
        (False, True),
        (False, True),
        tuple(sparse_arms),
        (False, True),
    ):
        arms = []
        if mhc:
            arms.append("A1")
        if nope:
            arms.append("A2")
        if kda:
            arms.append("A3")
        arms.extend(sparse_arms[sparse])
        if mtp:
            arms.append("A6")
        elements = _tiny_arm_elements(*arms)
        model_config = Glm53ArchSpeedrunModel.Config(elements=elements)
        model = Glm53ArchSpeedrunModel(model_config)
        model.init_states()
        model.train()
        tokens = torch.arange(elements.max_seq_len).view(1, -1) % elements.vocab_size
        labels = (tokens + 1) % elements.vocab_size
        positions = torch.arange(elements.max_seq_len).view(1, -1)
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            logits = model(
                tokens,
                positions=positions,
                mtp_labels=labels,
                aux_loss_denominators=torch.tensor([labels.numel()]),
            )
            loss = F.cross_entropy(logits.float(), labels.reshape(-1))
            loss.backward()
            assert torch.isfinite(loss)
            for name, param in model.named_parameters():
                assert torch.isfinite(param).all(), name


def test_optimizer_config_matches_partition_for_kda_sparse_mtp_combination():
    model_config = Glm53ArchSpeedrunModel.Config(
        elements=_tiny_arm_elements("A3", "A5", "A6")
    )
    model = Glm53ArchSpeedrunModel(model_config)
    optimizer_config = speedrun_optimizer_config(model_config)
    muon_config = optimizer_config.optimizers[0]

    assert isinstance(muon_config, DistMuon.Config)
    assert set(muon_config.compute_sharding_by_fqn) == muon_parameter_names(model)


def test_optimizer_config_matches_parameter_partition():
    model_config = Glm53ArchSpeedrunModel.Config(elements=_tiny_elements())
    optimizer_config = speedrun_optimizer_config(model_config)

    assert isinstance(optimizer_config.optimizers[0], DistMuon.Config)
    assert isinstance(optimizer_config.optimizers[1], AdamW.Config)
    muon_config = optimizer_config.optimizers[0]
    assert set(muon_config.compute_sharding_by_fqn) == {
        f"blocks.{layer}.{projection}"
        for layer in range(model_config.elements.num_layers)
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


def test_config_contract_applies_arm_and_element_env(monkeypatch):
    monkeypatch.setenv("GLM53_SPEEDRUN_SEQ_LEN", "128")
    monkeypatch.setenv("GLM53_SPEEDRUN_LOCAL_BATCH_SIZE", "2")
    monkeypatch.setenv("GLM53_SPEEDRUN_TOKENS_PER_STEP", "1024")
    monkeypatch.setenv("GLM53_SPEEDRUN_TOKEN_BUDGET", "2048")
    monkeypatch.setenv("GLM53_SPEEDRUN_VAL_TOKENS", "1024")
    monkeypatch.setenv("GLM53_SPEEDRUN_VAL_FREQ", "1")
    monkeypatch.setenv("GLM53_SPEEDRUN_ARM", "A5")
    monkeypatch.setenv("GLM53_SPEEDRUN_SPARSE_TOP_K", "16")
    monkeypatch.setenv("GLM53_SPEEDRUN_INDEXER_AUX_LOSS_WEIGHT", "0.25")

    config = glm53_arch_speedrun_b0()

    assert config.model.elements.sparse_attention_enabled
    assert config.model.elements.index_pool_size == 4
    assert config.model.elements.sparse_top_k == 16
    assert config.model.elements.indexer_aux_loss_weight == 0.25
    assert config.training.num_tokens_per_train_step == 1024
    assert config.model.local_compile_regions == [
        "glm53_speedrun_attention",
        "glm53_speedrun_mlp",
        "glm53_speedrun_mhc_pre",
        "glm53_speedrun_mhc_post",
        "loss",
    ]


def test_default_microbatch_policy_keeps_expensive_arms_under_cap(monkeypatch):
    monkeypatch.setenv("GLM53_SPEEDRUN_SEQ_LEN", "4096")
    monkeypatch.setenv("GLM53_SPEEDRUN_VAL_TOKENS", "32768")
    monkeypatch.setenv("GLM53_SPEEDRUN_COMPILE", "0")

    monkeypatch.setenv("GLM53_SPEEDRUN_ARM", "B0")
    b0_config = glm53_arch_speedrun_b0()

    monkeypatch.setenv("GLM53_SPEEDRUN_ARM", "A3")
    kda_config = glm53_arch_speedrun_b0()

    monkeypatch.setenv("GLM53_SPEEDRUN_ARM", "A4")
    sparse_config = glm53_arch_speedrun_b0()

    monkeypatch.setenv("GLM53_SPEEDRUN_ARM", "G")
    combined_config = glm53_arch_speedrun_b0()

    assert b0_config.training.num_tokens_per_microbatch_per_dp_rank == 32_768
    assert kda_config.training.num_tokens_per_microbatch_per_dp_rank == 16_384
    assert sparse_config.training.num_tokens_per_microbatch_per_dp_rank == 8_192
    assert combined_config.training.num_tokens_per_microbatch_per_dp_rank == 8_192
    assert b0_config.training.num_tokens_per_train_step == 262_144
    assert kda_config.training.num_tokens_per_train_step == 262_144
    assert sparse_config.training.num_tokens_per_train_step == 262_144
    assert combined_config.training.num_tokens_per_train_step == 262_144
