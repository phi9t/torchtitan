# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from torchtitan.config import ConfigManager
from torchtitan.experiments.mini_kimi_k3.config_registry import (
    mini_kimi_k3_tiny_plumbing,
    MiniK3PretokenizedTokenizer,
    MiniK3R1ContractModel,
    MiniK3TokenDataLoader,
)


def _write_shard(path: Path, tokens: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(tokens, dtype="<u4").tofile(path)


def _write_manifest(path: Path, sources: dict[str, list[str]]) -> Path:
    path.write_text(
        json.dumps(
            {"sources": {name: {"shards": shards} for name, shards in sources.items()}},
            sort_keys=True,
        )
    )
    return path


def test_mini_kimi_k3_tiny_config_loads_through_config_manager(tmp_path: Path):
    config = ConfigManager().parse_args(
        [
            "--module",
            "mini_kimi_k3",
            "--config",
            "mini_kimi_k3_tiny_plumbing",
            "--dump_folder",
            str(tmp_path / "dump"),
        ]
    )

    assert config.model_spec.name == "mini_kimi_k3"
    assert config.model_spec.flavor == "tiny_plumbing"
    assert isinstance(config.dataloader, MiniK3TokenDataLoader.Config)
    assert config.training.seq_len == 16
    assert config.training.steps == 1
    assert config.checkpoint.enable is False
    assert config.debug.enable_structured_logging is False


def test_mini_kimi_k3_r1_contract_config_loads_and_builds_inspection_model(
    tmp_path: Path,
):
    config = ConfigManager().parse_args(
        [
            "--module",
            "mini_kimi_k3",
            "--config",
            "mini_kimi_k3_r1_contract",
            "--dump_folder",
            str(tmp_path / "dump"),
        ]
    )

    assert config.model_spec.name == "mini_kimi_k3"
    assert config.model_spec.flavor == "r1_contract"
    assert isinstance(config.dataloader, MiniK3TokenDataLoader.Config)
    assert config.training.seq_len == 4096
    assert config.training.local_batch_size == 4
    assert config.training.global_batch_size == 32
    assert config.training.steps == 38_147
    assert config.training.dtype == "bfloat16"
    assert config.checkpoint.enable is True
    assert config.checkpoint.interval == 100
    assert config.loss.global_vocab_size == 163_840

    optimizer_group = config.optimizer.param_groups[-1]
    assert optimizer_group.pattern == r".*"
    assert optimizer_group.optimizer_name == "AdamW"
    assert optimizer_group.optimizer_kwargs["lr"] == 6.0e-4
    assert optimizer_group.optimizer_kwargs["betas"] == (0.9, 0.95)
    assert optimizer_group.optimizer_kwargs["eps"] == 1.0e-8
    assert optimizer_group.optimizer_kwargs["weight_decay"] == 0.1
    assert config.model_spec.model.weight_decay_policy == (
        "exclude_router_parameters_and_1d_parameters"
    )

    model = config.model_spec.model.build()
    model.verify_module_protocol()

    assert isinstance(model, MiniK3R1ContractModel)
    assert model.lm_head.weight.shape == (163_840, 512)
    nparams, flops_per_token = config.model_spec.model.get_nparams_and_flops(
        model,
        config.training.seq_len,
    )
    assert nparams == 1_023_206_628
    assert flops_per_token > 0


def test_mini_kimi_k3_r1_contract_uses_pretokenized_shard_tokenizer() -> None:
    config = ConfigManager().parse_args(
        [
            "--module",
            "mini_kimi_k3",
            "--config",
            "mini_kimi_k3_r1_contract",
        ]
    )

    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)

    assert isinstance(tokenizer, MiniK3PretokenizedTokenizer)
    assert tokenizer.get_vocab_size() == 163_840
    with pytest.raises(NotImplementedError, match="pretokenized uint32 shards"):
        tokenizer.encode("hello")
    with pytest.raises(NotImplementedError, match="pretokenized uint32 shards"):
        tokenizer.decode([1, 2, 3])


def test_mini_kimi_k3_r1_optimizer_groups_exclude_1d_parameters_from_decay(
    tmp_path: Path,
):
    config = ConfigManager().parse_args(
        [
            "--module",
            "mini_kimi_k3",
            "--config",
            "mini_kimi_k3_r1_contract",
            "--dump_folder",
            str(tmp_path / "dump"),
        ]
    )
    groups = config.optimizer.param_groups

    assert groups[-1].pattern == r".*"
    assert groups[-1].optimizer_kwargs["weight_decay"] == 0.1
    assert all(group.optimizer_name == "AdamW" for group in groups)
    assert all(group.optimizer_kwargs["weight_decay"] == 0.0 for group in groups[:-1])

    model = config.model_spec.model.build()
    model.init_states()
    trainable_params = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }
    assigned_groups: dict[str, int] = {}
    for group_index, group in enumerate(groups):
        pattern = re.compile(group.pattern)
        for name in trainable_params:
            if name not in assigned_groups and pattern.search(name):
                assigned_groups[name] = group_index

    assert set(assigned_groups) == set(trainable_params)
    no_decay_names = {
        name
        for name, group_index in assigned_groups.items()
        if group_index != len(groups) - 1
    }
    assert no_decay_names
    assert no_decay_names == {
        name for name, param in trainable_params.items() if param.ndim == 1
    }
    assert all("e_score_correction_bias" not in name for name in assigned_groups)


def test_mini_kimi_k3_r1_init_states_preserves_special_parameter_initializers():
    config = ConfigManager().parse_args(
        ["--module", "mini_kimi_k3", "--config", "mini_kimi_k3_r1_contract"]
    )
    torch.manual_seed(1234)
    model = config.model_spec.model.build()

    model.init_states()

    first_kda = model.inner.layers[0].self_attn
    a = first_kda.A_log.detach().exp()
    assert torch.all(a >= 1.0)
    assert torch.all(a < 16.0)
    assert a.std() > 0

    first_router = model.inner.layers[1].block_sparse_moe.router
    assert torch.count_nonzero(first_router.e_score_correction_bias) == 0
    assert first_router.e_score_correction_bias.requires_grad is False


def test_mini_kimi_k3_r1_init_states_preserves_kda_conv_initializers():
    config = ConfigManager().parse_args(
        ["--module", "mini_kimi_k3", "--config", "mini_kimi_k3_r1_contract"]
    )
    torch.manual_seed(1234)
    model = config.model_spec.model.build()

    model.init_states()

    first_kda = model.inner.layers[0].self_attn
    assert float(first_kda.q_proj.weight.detach().std()) == pytest.approx(
        0.02, rel=0.05
    )
    assert float(first_kda.q_conv1d.weight.detach().std()) > 0.1
    assert float(first_kda.k_conv1d.weight.detach().std()) > 0.1
    assert float(first_kda.v_conv1d.weight.detach().std()) > 0.1


def test_mini_kimi_k3_tiny_model_config_builds_and_runs_forward():
    config = mini_kimi_k3_tiny_plumbing()
    model = config.model_spec.model.build()
    model.verify_module_protocol()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    logits = model(input_ids, positions=torch.arange(4).unsqueeze(0))

    assert logits.shape == (1, 4, 32)
    nparams, flops_per_token = config.model_spec.model.get_nparams_and_flops(model, 4)
    assert nparams == sum(param.numel() for param in model.parameters())
    assert flops_per_token > 0


def test_mini_kimi_k3_token_dataloader_yields_trainer_batch(tmp_path: Path):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", list(range(12)))
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    dataloader = MiniK3TokenDataLoader.Config(
        token_manifest=str(manifest),
        tokens_dir=str(tmp_path / "tokens"),
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=object(),
        seq_len=4,
        local_batch_size=2,
    )

    input_dict, labels = next(iter(dataloader))

    assert set(input_dict) == {"input", "positions"}
    assert input_dict["input"].shape == (2, 4)
    assert input_dict["positions"].tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert labels.shape == (2, 4)
    assert labels.tolist() == [[1, 2, 3, 4], [6, 7, 8, 9]]
