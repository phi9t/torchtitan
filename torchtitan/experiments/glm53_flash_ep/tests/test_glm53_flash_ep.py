# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path

import pytest
import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import (
    DistributedTopology,
    ParallelismContext,
)
from torchtitan.distributed.utils import init_fake_mode
from torchtitan.experiments.glm53_flash.checkpoint import TrainableModelWrapper
from torchtitan.experiments.glm53_flash.flavors import with_lora
from torchtitan.experiments.glm53_flash.model import (
    Glm53FlashForCausalLM,
    GlmMoE,
    GlmTopKRouter,
)

from torchtitan.experiments.glm53_flash_ep.config_registry import (
    glm53_flash_ep_countdown_lora_truncated,
    glm53_flash_ep_debugmodel,
    glm53_flash_ep_debugmodel_lora,
    glm53_flash_ep_mini,
)
from torchtitan.experiments.glm53_flash_ep.flavors import tiny_ep
from torchtitan.experiments.glm53_flash_ep.grad_probe import (
    compare_snapshots,
    offline_grad_norm,
)
from torchtitan.experiments.glm53_flash_ep.model import Glm53FlashEPForCausalLM
from torchtitan.experiments.glm53_flash_ep.moe import GlmEPMoE, GlmEPRouter
from torchtitan.experiments.glm53_flash_ep.parity_report import (
    compare_metric_steps,
    compare_tensor_maps,
    GRAD_NORM_TAG,
    LOSS_TAG,
    read_memory_summary,
    STEP_TIME_TAG,
    summarize_scalar_deltas,
    THROUGHPUT_TAG,
)
from torchtitan.experiments.glm53_flash_ep.sharded_hf_load import (
    load_rank_local_ep_hf_safetensors,
)
from torchtitan.experiments.glm53_flash_ep.sharding import (
    set_glm53_flash_ep_expert_sharding,
    validate_glm53_flash_ep_parallelism,
)
from torchtitan.experiments.glm53_flash_ep.state_dict_adapter import (
    Glm53FlashEPStateDictAdapter,
)
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.moe_sharding import expert_param_placement_sparse


def _copy_glm_moe_to_ep(ref: GlmMoE, ep: GlmEPMoE) -> None:
    with torch.no_grad():
        ep.gate.weight.copy_(ref.gate.weight)
        ep.gate.e_score_correction_bias.copy_(ref.gate.e_score_correction_bias)
        ep.shared_experts.load_state_dict(ref.shared_experts.state_dict())
        ep.experts.w13.weight.copy_(
            ref.experts.gate_up_proj.view_as(ep.experts.w13.weight)
        )
        ep.experts.w2.weight.copy_(ref.experts.down_proj)


def _init_model_with_seed(model: nn.Module, seed: int) -> nn.Module:
    model.to_empty(device="cpu")
    torch.manual_seed(seed)
    with torch.no_grad():
        model.init_states()
    return model


def test_ep_config_registry_uses_requested_expert_parallel_degree(monkeypatch):
    monkeypatch.setenv("GLM53_FLASH_EP_DEGREE", "4")

    debug_config = glm53_flash_ep_debugmodel()
    mini_config = glm53_flash_ep_mini()

    assert isinstance(debug_config.model, Glm53FlashEPForCausalLM.Config)
    assert isinstance(mini_config.model, Glm53FlashEPForCausalLM.Config)
    assert debug_config.parallelism.expert_parallel_degree == 4
    assert mini_config.parallelism.expert_parallel_degree == 4
    assert debug_config.parallelism.tensor_parallel_degree == 1
    assert debug_config.parallelism.context_parallel_degree == 1


def test_ep_init_preserves_plain_lora_rng_order_after_sparse_layers():
    seed = 1234
    plain_cfg = with_lora(tiny_ep(), rank=4, alpha=16.0)
    ep_cfg = with_lora(tiny_ep(), rank=4, alpha=16.0)

    with torch.device("meta"):
        plain = Glm53FlashForCausalLM(
            Glm53FlashForCausalLM.Config(glm_config=plain_cfg)
        )
        ep = Glm53FlashEPForCausalLM(Glm53FlashEPForCausalLM.Config(glm_config=ep_cfg))

    _init_model_with_seed(plain, seed)
    _init_model_with_seed(ep, seed)

    plain_state = plain.state_dict()
    ep_state = ep.state_dict()
    lora_keys = sorted(key for key in plain_state if key.endswith(".lora_a"))
    assert lora_keys
    for key in lora_keys:
        torch.testing.assert_close(ep_state[key], plain_state[key], rtol=0, atol=0)

    torch.testing.assert_close(
        ep.layers["1"].mlp.experts.w13.weight.view_as(
            plain.layers["1"].mlp.experts.gate_up_proj
        ),
        plain.layers["1"].mlp.experts.gate_up_proj,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        ep.layers["1"].mlp.gate.weight,
        plain.layers["1"].mlp.gate.weight,
        rtol=0,
        atol=0,
    )


def test_truncated_countdown_lora_config_matches_t13_contract(monkeypatch):
    monkeypatch.setenv("GLM53_FLASH_EP_DEGREE", "4")
    monkeypatch.setenv("GLM53_FLASH_FULL_SEQ_LEN", "512")
    monkeypatch.setenv("GLM53_FLASH_FULL_LOCAL_BATCH_SIZE", "1")
    monkeypatch.setenv("GLM53_FLASH_FULL_TOKENS_PER_STEP", "32768")
    monkeypatch.setenv("GLM53_FLASH_FULL_STEPS", "3")
    monkeypatch.setenv("GLM53_FLASH_SEED", "42")
    monkeypatch.setenv("GLM53_FLASH_DETERMINISTIC", "1")

    config = glm53_flash_ep_countdown_lora_truncated()

    assert isinstance(config.model, Glm53FlashEPForCausalLM.Config)
    glm_config = config.model.glm_config
    assert glm_config.num_hidden_layers == 8
    assert glm_config.layer_offset == 0
    assert list(glm_config.mlp_layer_types[:8]) == [
        "dense",
        "dense",
        "dense",
        "sparse",
        "sparse",
        "sparse",
        "sparse",
        "sparse",
    ]
    assert glm_config.lora_rank == 16
    assert config.parallelism.expert_parallel_degree == 4
    assert config.training.max_context_length == 512
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 512
    assert config.training.num_tokens_per_train_step == 32768
    assert config.training.steps == 3
    assert config.training.dtype == "bfloat16"
    assert config.debug.seed == 42
    assert config.debug.deterministic
    assert config.metrics.enable_tensorboard
    assert config.checkpointer is not None


@pytest.mark.parametrize("world_size,valid_eps", [(4, (1, 2, 4)), (8, (1, 2, 4, 8))])
def test_ep_mesh_ladder_validates_current_glm_boundary(world_size, valid_eps):
    for ep in valid_eps:
        parallelism = ParallelismConfig(
            data_parallel_shard_degree=-1,
            expert_parallel_degree=ep,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
        )
        context = ParallelismContext.from_config(
            parallelism,
            DistributedTopology(world_size=world_size),
            dump_folder=".",
        )
        assert context.ep == ep
        assert context.dp_shard == world_size

    invalid = ParallelismConfig(
        data_parallel_shard_degree=-1,
        expert_parallel_degree=8,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
    )
    if world_size == 4:
        with pytest.raises(ValueError, match="expert_parallel_degree"):
            ParallelismContext.from_config(
                invalid,
                DistributedTopology(world_size=world_size),
                dump_folder=".",
            )


def test_glm_ep_parallelism_rejects_unsupported_axes():
    cfg = tiny_ep()
    with pytest.raises(ValueError, match="TP is not implemented"):
        validate_glm53_flash_ep_parallelism(
            cfg,
            ParallelismConfig(
                expert_parallel_degree=1,
                tensor_parallel_degree=2,
            ),
        )
    with pytest.raises(ValueError, match="num_experts must be divisible"):
        validate_glm53_flash_ep_parallelism(
            cfg,
            ParallelismConfig(expert_parallel_degree=3),
        )


def test_ep_expert_sharding_uses_sparse_expert_axis_when_enabled():
    experts = GlmEPMoE(tiny_ep()).experts

    set_glm53_flash_ep_expert_sharding(experts, enable_ep=True)
    assert (
        experts.w13._sharding_config.state_shardings["weight"]
        == expert_param_placement_sparse()
    )
    assert (
        experts.w2._sharding_config.state_shardings["weight"]
        == expert_param_placement_sparse()
    )

    set_glm53_flash_ep_expert_sharding(experts, enable_ep=False)
    assert experts.w13._sharding_config.state_shardings[
        "weight"
    ] == dense_param_placement(tp=spmd.R)


def test_ep_grouped_linears_have_distinct_fallback_remat_names():
    experts = GlmEPMoE(tiny_ep()).experts

    assert experts.w13.remat_region_name("grouped_mm") == "w13.grouped_mm"
    assert experts.w2.remat_region_name("grouped_mm") == "w2.grouped_mm"


def test_glm_ep_router_matches_current_glm_router():
    torch.manual_seed(0)
    cfg = tiny_ep()
    ref = GlmTopKRouter(cfg).eval()
    ep = GlmEPRouter(cfg).eval()
    with torch.no_grad():
        ref.weight.normal_(0, 0.02)
        ref.e_score_correction_bias.copy_(torch.linspace(-0.2, 0.2, cfg.num_experts))
        ep.weight.copy_(ref.weight)
        ep.e_score_correction_bias.copy_(ref.e_score_correction_bias)

    x_BLD = torch.randn(2, 3, cfg.hidden_size)
    ref_ids, ref_scores = ref(x_BLD)
    ep_scores, ep_ids, routing_map = ep(x_BLD.view(-1, cfg.hidden_size))

    torch.testing.assert_close(ep_ids, ref_ids)
    torch.testing.assert_close(ep_scores, ref_scores)
    assert routing_map.shape == (x_BLD.numel() // cfg.hidden_size, cfg.num_experts)
    assert (
        routing_map.sum().item()
        == x_BLD.shape[0] * x_BLD.shape[1] * cfg.num_experts_per_tok
    )


def test_glm_ep_moe_matches_current_glm_moe_with_bounded_grouped_delta():
    torch.manual_seed(1)
    cfg = tiny_ep()
    ref = GlmMoE(cfg).float()
    ep = GlmEPMoE(cfg).float()
    with torch.no_grad():
        for param in ref.parameters():
            param.normal_(0, 0.02)
    _copy_glm_moe_to_ep(ref, ep)

    x_ref = torch.randn(2, 3, cfg.hidden_size, requires_grad=True)
    x_ep = x_ref.detach().clone().requires_grad_(True)
    ref_out = ref(x_ref)
    ep_out = ep(x_ep)
    ref_out.square().mean().backward()
    ep_out.square().mean().backward()

    assert (ref_out - ep_out).abs().max().item() < 3.0e-5
    assert (x_ref.grad - x_ep.grad).abs().max().item() < 1.0e-8
    assert (ref.gate.weight.grad - ep.gate.weight.grad).abs().max().item() < 1.0e-8
    assert (
        ref.experts.gate_up_proj.grad.view_as(ep.experts.w13.weight.grad)
        - ep.experts.w13.weight.grad
    ).abs().max().item() < 5.0e-8
    assert (
        ref.experts.down_proj.grad - ep.experts.w2.weight.grad
    ).abs().max().item() < 5.0e-8


def test_ep_state_dict_adapter_maps_common_expert_keys_to_glm_hf_keys():
    cfg = tiny_ep()
    model_config = Glm53FlashEPForCausalLM.Config(glm_config=cfg)
    adapter = Glm53FlashEPStateDictAdapter(model_config, None)
    w13 = torch.arange(cfg.num_experts * 2 * 2 * 3, dtype=torch.float32).view(
        cfg.num_experts, 2, 2, 3
    )
    w2 = torch.arange(cfg.num_experts * 3 * 2, dtype=torch.float32).view(
        cfg.num_experts, 3, 2
    )

    hf = adapter.to_hf(
        {
            "layers.0.mlp.experts.w13.weight": w13,
            "layers.0.mlp.experts.w2.weight": w2,
        }
    )

    assert set(hf) == {
        f"model.language_model.layers.0.mlp.experts.{expert}.{proj}_proj.weight"
        for expert in range(cfg.num_experts)
        for proj in ("gate", "up", "down")
    }
    restored = adapter.from_hf(hf)
    torch.testing.assert_close(restored["layers.0.mlp.experts.w13.weight"], w13)
    torch.testing.assert_close(restored["layers.0.mlp.experts.w2.weight"], w2)


class _TinyEPExperts(nn.Module):
    def __init__(self, mesh) -> None:
        super().__init__()
        self.w13 = nn.Module()
        self.w13.weight = nn.Parameter(
            DTensor.from_local(
                torch.empty(1, 2, 2, 3),
                mesh,
                (Shard(0),),
                run_check=False,
            )
        )
        self.w2 = nn.Module()
        self.w2.weight = nn.Parameter(
            DTensor.from_local(
                torch.empty(1, 3, 2),
                mesh,
                (Shard(0),),
                run_check=False,
            )
        )


class _TinyEPMlp(nn.Module):
    def __init__(self, mesh) -> None:
        super().__init__()
        self.experts = _TinyEPExperts(mesh)


class _TinyEPLayer(nn.Module):
    def __init__(self, mesh) -> None:
        super().__init__()
        self.mlp = _TinyEPMlp(mesh)


class _TinyEPLoadModel(nn.Module):
    def __init__(self, mesh) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"0": _TinyEPLayer(mesh)})
        self.adapter = nn.Module()
        self.adapter.lora_a = nn.Parameter(torch.full((1, 3), -7.0))


def _write_tiny_ep_hf_checkpoint(root: Path) -> None:
    tensors = {}
    for expert in range(4):
        tensors[
            f"model.language_model.layers.0.mlp.experts.{expert}.gate_proj.weight"
        ] = torch.full((2, 3), float(expert + 1))
        tensors[
            f"model.language_model.layers.0.mlp.experts.{expert}.up_proj.weight"
        ] = torch.full((2, 3), float(expert + 11))
        tensors[
            f"model.language_model.layers.0.mlp.experts.{expert}.down_proj.weight"
        ] = torch.full((3, 2), float(expert + 21))
    shard = root / "model-00001-of-00001.safetensors"
    save_file(tensors, shard)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {key: shard.name for key in tensors}})
    )


def test_rank_local_ep_loader_reads_only_this_rank_experts(tmp_path):
    _write_tiny_ep_hf_checkpoint(tmp_path)
    init_fake_mode(4, rank=2)
    try:
        mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("ep",))
        model = _TinyEPLoadModel(mesh)
        model_config = Glm53FlashEPForCausalLM.Config(glm_config=tiny_ep())
        adapter = Glm53FlashEPStateDictAdapter(model_config, None)
        lora_before = model.adapter.lora_a.detach().clone()

        stats = load_rank_local_ep_hf_safetensors(
            model,
            adapter,
            tmp_path,
            verify=True,
        )

        torch.testing.assert_close(
            model.layers["0"].mlp.experts.w13.weight._local_tensor[0, 0],
            torch.full((2, 3), 3.0),
        )
        torch.testing.assert_close(
            model.layers["0"].mlp.experts.w13.weight._local_tensor[0, 1],
            torch.full((2, 3), 13.0),
        )
        torch.testing.assert_close(
            model.layers["0"].mlp.experts.w2.weight._local_tensor[0],
            torch.full((3, 2), 23.0),
        )
        torch.testing.assert_close(model.adapter.lora_a, lora_before)
        assert stats.tensors_loaded == 2
        assert stats.tensors_skipped_lora == 1
        assert stats.verified_keys == ["layers.0.mlp.experts.w13.weight"]
        assert stats.expert_keys_loaded == [
            "model.language_model.layers.0.mlp.experts.2.gate_proj.weight",
            "model.language_model.layers.0.mlp.experts.2.up_proj.weight",
            "model.language_model.layers.0.mlp.experts.2.down_proj.weight",
        ]
    finally:
        dist.destroy_process_group()


def test_trainable_only_lora_checkpoint_round_trip(tmp_path):
    torch.manual_seed(2)
    config = glm53_flash_ep_debugmodel_lora()
    assert isinstance(config.model, Glm53FlashEPForCausalLM.Config)
    with torch.device("meta"):
        model = config.model.build()
    model.to_empty(device="cpu")
    model.init_weights()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith((".lora_a", ".lora_b")):
                param.fill_(float(len(name) % 17))

    state_dict = TrainableModelWrapper(model).state_dict()
    assert state_dict
    assert all(key.endswith((".lora_a", ".lora_b")) for key in state_dict)
    assert not any(
        ".experts.w13." in key or ".experts.w2." in key for key in state_dict
    )
    checkpoint_path = tmp_path / "trainable.pt"
    torch.save(state_dict, checkpoint_path)

    with torch.device("meta"):
        restored = config.model.build()
    restored.to_empty(device="cpu")
    restored.init_weights()
    missing, unexpected = restored.load_state_dict(
        torch.load(checkpoint_path, weights_only=True),
        strict=False,
    )

    assert not unexpected
    assert set(state_dict).isdisjoint(missing)
    for key, value in state_dict.items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_parity_report_aligns_metrics_and_memory_records(tmp_path):
    plain_metrics = {
        LOSS_TAG: [{"step": 1, "value": 1.0, "wall_time": 0.0}],
        GRAD_NORM_TAG: [{"step": 1, "value": 2.0, "wall_time": 0.0}],
        THROUGHPUT_TAG: [{"step": 1, "value": 100.0, "wall_time": 0.0}],
        STEP_TIME_TAG: [{"step": 1, "value": 0.5, "wall_time": 0.0}],
    }
    ep_metrics = {
        LOSS_TAG: [{"step": 1, "value": 1.25, "wall_time": 0.0}],
        GRAD_NORM_TAG: [{"step": 1, "value": 1.5, "wall_time": 0.0}],
        THROUGHPUT_TAG: [{"step": 1, "value": 80.0, "wall_time": 0.0}],
        STEP_TIME_TAG: [{"step": 1, "value": 0.75, "wall_time": 0.0}],
    }

    rows = compare_metric_steps(plain_metrics, ep_metrics)
    assert rows == [
        {
            "step": 1,
            "plain_loss": 1.0,
            "ep_loss": 1.25,
            "loss_delta": 0.25,
            "plain_grad_norm": 2.0,
            "ep_grad_norm": 1.5,
            "grad_norm_delta": -0.5,
            "plain_tokens_per_second": 100.0,
            "ep_tokens_per_second": 80.0,
            "tokens_per_second_delta": -20.0,
            "plain_step_time_ms": 500.0,
            "ep_step_time_ms": 750.0,
            "step_time_ms_delta": 250.0,
        }
    ]
    assert summarize_scalar_deltas(rows)["max_abs_grad_norm_delta"] == 0.5

    memory_dir = tmp_path / "run" / "lean_memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "rank00000_after_step_0001.json").write_text(
        json.dumps(
            {
                "rank": 0,
                "local_rank": 0,
                "device_name": "unit-test",
                "max_reserved_gib": 3.0,
                "max_allocated_gib": 2.0,
            }
        )
    )
    (memory_dir / "rank00000_after_step_0002.json").write_text(
        json.dumps(
            {
                "rank": 0,
                "local_rank": 0,
                "device_name": "unit-test",
                "max_reserved_gib": 4.0,
                "max_allocated_gib": 2.5,
            }
        )
    )

    summary = read_memory_summary(tmp_path / "run")
    assert summary["num_records"] == 2
    assert summary["peak_by_rank"][0]["max_reserved_gib"] == 4.0
    assert summary["peak_by_rank"][0]["max_allocated_gib"] == 2.5


def test_parity_report_compares_tensor_maps():
    rows = compare_tensor_maps(
        {
            "a.lora_a": torch.tensor([1.0, 2.0]),
            "b.lora_b": torch.tensor([3.0]),
        },
        {
            "a.lora_a": torch.tensor([1.5, 2.0]),
            "c.lora_b": torch.tensor([3.0]),
        },
    )

    by_name = {row["name"]: row for row in rows}
    assert by_name["a.lora_a"]["max_abs_delta"] == 0.5
    assert by_name["b.lora_b"]["reason"] == "missing tensor"
    assert by_name["c.lora_b"]["reason"] == "missing tensor"


def test_grad_probe_offline_norm_and_snapshot_comparison():
    reference = {
        "parameters": {
            "a.lora_a": {"gradient": torch.tensor([3.0, 4.0])},
            "b.lora_b": {"gradient": torch.tensor([0.0])},
        }
    }
    candidate = {
        "parameters": {
            "a.lora_a": {"gradient": torch.tensor([3.0, 5.0])},
            "b.lora_b": {"gradient": torch.tensor([0.0])},
        }
    }

    assert offline_grad_norm(reference["parameters"]) == 5.0
    rows = compare_snapshots(reference, candidate)
    by_name = {row["name"]: row for row in rows}

    assert by_name["a.lora_a"]["max_abs_gradient_delta"] == 1.0
    assert by_name["a.lora_a"]["reference_gradient_l2_norm"] == 5.0
    assert by_name["b.lora_b"]["max_abs_gradient_delta"] == 0.0
