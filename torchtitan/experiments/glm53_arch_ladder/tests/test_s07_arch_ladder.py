# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc

import pytest
import torch
import torch.nn.functional as F

from torchtitan.components.optim import AdamW, DistMuon
from torchtitan.experiments.falcon.bin_reader import write_nanogpt_bin
from torchtitan.experiments.glm53_arch_ladder.config_registry import (
    glm53_arch_ladder_l1_qwen3_dense,
    glm53_arch_ladder_l2_qwen3_moe,
    glm53_arch_ladder_l4_glm53_flash,
)
from torchtitan.experiments.glm53_arch_ladder.data import (
    FixedBankFineWebBinDataLoader,
    FixedBankPackedFineWebBinDataLoader,
    PackedFineWebBinDataLoader,
)
from torchtitan.experiments.glm53_arch_ladder.flavors import (
    build_ladder_model_config,
    GPT2_VOCAB_SIZE,
    TARGET_ACTIVE_NON_EMBEDDING_PARAMS,
)
from torchtitan.experiments.glm53_arch_ladder.params import (
    adamw_parameter_names,
    ladder_optimizer_config,
    muon_parameter_names,
    parameter_counts,
)
from torchtitan.experiments.glm53_flash.model import Glm53FlashForCausalLM
from torchtitan.models.common.decoder import Decoder


RUNGS = ("L1", "L2", "L3", "L4")


def _trainable_names(model: torch.nn.Module) -> set[str]:
    return {name for name, param in model.named_parameters() if param.requires_grad}


def test_packed_fineweb_loader_preserves_shifted_labels(tmp_path):
    shard = tmp_path / "tiny.bin"
    write_nanogpt_bin(shard, torch.arange(40, dtype=torch.long))
    loader = PackedFineWebBinDataLoader.Config(
        bin_glob=str(shard),
        vocab_size=64,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=None,
        max_context_length=4,
        num_tokens_per_microbatch=8,
    )

    batch = next(iter(loader))

    torch.testing.assert_close(batch.input, torch.arange(8))
    torch.testing.assert_close(batch.labels, torch.arange(1, 9))
    torch.testing.assert_close(batch.positions, torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
    assert batch.labels[-1].item() == 8


def test_fixed_bank_loaders_repeat_small_bank(tmp_path):
    shard = tmp_path / "tiny.bin"
    write_nanogpt_bin(shard, torch.arange(40, dtype=torch.long))
    packed_loader = FixedBankPackedFineWebBinDataLoader.Config(
        bin_glob=str(shard),
        vocab_size=64,
        bank_microbatches=2,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=None,
        max_context_length=4,
        num_tokens_per_microbatch=4,
    )
    batched_loader = FixedBankFineWebBinDataLoader.Config(
        bin_glob=str(shard),
        vocab_size=64,
        bank_microbatches=2,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=None,
        max_context_length=4,
        num_tokens_per_microbatch=4,
    )

    packed_iter = iter(packed_loader)
    first = next(packed_iter)
    second = next(packed_iter)
    third = next(packed_iter)
    torch.testing.assert_close(first.input, torch.arange(4))
    torch.testing.assert_close(second.input, torch.arange(4, 8))
    torch.testing.assert_close(third.input, torch.arange(4))
    torch.testing.assert_close(first.labels, torch.arange(1, 5))

    batched_iter = iter(batched_loader)
    batched_first = next(batched_iter)
    batched_second = next(batched_iter)
    torch.testing.assert_close(batched_first.input, torch.arange(4).view(1, 4))
    torch.testing.assert_close(batched_second.labels, torch.arange(5, 9).view(1, 4))


def test_ladder_param_counts_hit_declared_active_non_embedding_match_rule():
    lower = int(0.9 * TARGET_ACTIVE_NON_EMBEDDING_PARAMS)
    upper = int(1.1 * TARGET_ACTIVE_NON_EMBEDDING_PARAMS)
    with torch.device("meta"):
        for rung in RUNGS:
            model = build_ladder_model_config(rung, seq_len=128).build()
            counts = parameter_counts(model)
            assert lower <= counts.active_non_embedding <= upper, (rung, counts)
            assert counts.total >= counts.non_embedding >= counts.active_non_embedding


def test_muon_adamw_partition_covers_each_rung_once():
    with torch.device("meta"):
        for rung in RUNGS:
            model = build_ladder_model_config(rung, seq_len=128).build()
            muon_names = muon_parameter_names(model)
            adamw_names = adamw_parameter_names(model)
            trainable_names = _trainable_names(model)
            optim_config = ladder_optimizer_config(
                model,
                muon_lr=0.02,
                adamw_lr=8.0e-4,
                weight_decay=0.1,
                momentum=0.95,
            )

            assert muon_names, rung
            assert adamw_names, rung
            assert muon_names.isdisjoint(adamw_names), rung
            assert muon_names | adamw_names == trainable_names, rung
            assert isinstance(optim_config.optimizers[0], DistMuon.Config)
            assert isinstance(optim_config.optimizers[1], AdamW.Config)
            assert set(optim_config.optimizers[0].compute_sharding_by_fqn) == muon_names


def test_l2_qwen3_moe_uses_muon_for_router_and_routed_experts(monkeypatch):
    monkeypatch.setenv("GLM53_ARCH_LADDER_SEQ_LEN", "128")
    monkeypatch.setenv("GLM53_ARCH_LADDER_LOCAL_BATCH_SIZE", "1")
    monkeypatch.setenv("GLM53_ARCH_LADDER_TOKENS_PER_STEP", "128")
    monkeypatch.setenv("GLM53_ARCH_LADDER_TOKEN_BUDGET", "128")
    monkeypatch.setenv("GLM53_ARCH_LADDER_VAL_TOKENS", "128")
    monkeypatch.setenv("GLM53_ARCH_LADDER_TENSORBOARD", "0")
    monkeypatch.setenv("GLM53_ARCH_LADDER_GPU_MEMORY_CAP_GIB", "0")
    with torch.device("meta"):
        config = glm53_arch_ladder_l2_qwen3_moe()
        muon_config = next(
            optimizer
            for optimizer in config.optim.optimizer.optimizers
            if isinstance(optimizer, DistMuon.Config)
        )
        adamw_config = next(
            optimizer
            for optimizer in config.optim.optimizer.optimizers
            if isinstance(optimizer, AdamW.Config)
        )

    assert "layers.0.moe.router.gate.weight" in muon_config.compute_sharding_by_fqn
    assert "layers.0.moe.routed_experts.w13.weight" in (
        muon_config.compute_sharding_by_fqn
    )
    assert "layers.0.moe.routed_experts.w2.weight" in (
        muon_config.compute_sharding_by_fqn
    )
    assert "layers.0.moe.expert_bias_E" not in muon_config.compute_sharding_by_fqn
    assert adamw_config.pattern == r".*"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Kimi K3 KDA requires CUDA")
def test_ladder_rungs_forward_backward_on_tiny_batch_cuda():
    device = torch.device("cuda")
    seq_len = 16
    for rung in RUNGS:
        torch.manual_seed(7)
        model_config = build_ladder_model_config(
            rung,
            seq_len=seq_len,
            glm_attn_impl="reference",
            glm_sparse_attention_mode="dense_causal",
        )
        model = model_config.build()
        model.init_states()
        model.to(device)
        model.train()
        tokens = (
            torch.arange(seq_len, device=device, dtype=torch.long) % GPT2_VOCAB_SIZE
        )
        labels = (tokens + 1) % GPT2_VOCAB_SIZE

        if isinstance(model, Glm53FlashForCausalLM):
            inputs = tokens.unsqueeze(0)
            forward_kwargs = {}
        else:
            assert isinstance(model, Decoder)
            inputs = tokens
            positions = torch.arange(seq_len, device=device)
            forward_kwargs = {
                "positions": positions,
                "attention_metadata": model._get_attention_metadata(positions),
                "padding_mask": torch.zeros(seq_len, device=device, dtype=torch.bool),
                "aux_loss_denominators": torch.tensor([seq_len], device=device),
            }

        logits = model(inputs, **forward_kwargs)
        loss = F.cross_entropy(logits.float(), labels)
        assert torch.isfinite(loss), rung
        loss.backward()
        assert any(
            param.grad is not None and torch.isfinite(param.grad).all()
            for param in model.parameters()
            if param.requires_grad
        ), rung
        del model, logits, loss
        gc.collect()
        torch.cuda.empty_cache()


def test_config_registry_uses_ladder_env_and_shared_fineweb_paths(monkeypatch):
    monkeypatch.setenv("GLM53_ARCH_LADDER_SEQ_LEN", "128")
    monkeypatch.setenv("GLM53_ARCH_LADDER_LOCAL_BATCH_SIZE", "2")
    monkeypatch.setenv("GLM53_ARCH_LADDER_TOKENS_PER_STEP", "1024")
    monkeypatch.setenv("GLM53_ARCH_LADDER_TOKEN_BUDGET", "2048")
    monkeypatch.setenv("GLM53_ARCH_LADDER_VAL_TOKENS", "1024")
    monkeypatch.setenv("GLM53_ARCH_LADDER_VAL_FREQ", "1")
    monkeypatch.setenv("GLM53_ARCH_LADDER_TENSORBOARD", "0")
    monkeypatch.setenv("GLM53_ARCH_LADDER_GPU_MEMORY_CAP_GIB", "0")
    monkeypatch.setenv("GLM53_ARCH_LADDER_DATA_MODE", "fineweb")

    l1_config = glm53_arch_ladder_l1_qwen3_dense()
    l4_config = glm53_arch_ladder_l4_glm53_flash()

    assert l1_config.training.max_context_length == 128
    assert l1_config.training.num_tokens_per_microbatch_per_dp_rank == 256
    assert l1_config.training.num_tokens_per_train_step == 1024
    assert l1_config.training.steps == 2
    assert l1_config.validator is not None
    assert l1_config.validator.steps == 4
    assert isinstance(l1_config.dataloader, PackedFineWebBinDataLoader.Config)
    assert "glm53_arch_speedrun/results/data" in l1_config.dataloader.bin_glob
    assert isinstance(l4_config.model, Glm53FlashForCausalLM.Config)
    assert l4_config.model.glm_config.vocab_size == GPT2_VOCAB_SIZE


def test_config_registry_can_select_fixed_bank_loaders(monkeypatch):
    monkeypatch.setenv("GLM53_ARCH_LADDER_SEQ_LEN", "128")
    monkeypatch.setenv("GLM53_ARCH_LADDER_LOCAL_BATCH_SIZE", "1")
    monkeypatch.setenv("GLM53_ARCH_LADDER_TOKENS_PER_STEP", "128")
    monkeypatch.setenv("GLM53_ARCH_LADDER_TOKEN_BUDGET", "128")
    monkeypatch.setenv("GLM53_ARCH_LADDER_VAL_TOKENS", "128")
    monkeypatch.setenv("GLM53_ARCH_LADDER_VAL_FREQ", "1")
    monkeypatch.setenv("GLM53_ARCH_LADDER_TENSORBOARD", "0")
    monkeypatch.setenv("GLM53_ARCH_LADDER_GPU_MEMORY_CAP_GIB", "0")
    monkeypatch.setenv("GLM53_ARCH_LADDER_DATA_MODE", "fixed_bank")
    monkeypatch.setenv("GLM53_ARCH_LADDER_FIXED_BANK_MICROBATCHES", "2")

    l1_config = glm53_arch_ladder_l1_qwen3_dense()
    l4_config = glm53_arch_ladder_l4_glm53_flash()

    assert isinstance(l1_config.dataloader, FixedBankPackedFineWebBinDataLoader.Config)
    assert l1_config.dataloader.bank_microbatches == 2
    assert l1_config.validator is not None
    assert isinstance(
        l1_config.validator.dataloader,
        FixedBankPackedFineWebBinDataLoader.Config,
    )
    assert isinstance(l4_config.dataloader, FixedBankFineWebBinDataLoader.Config)
