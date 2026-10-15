# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parameter counting and optimizer partition helpers for S07."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

import torch.nn as nn
from torch.distributed.tensor import Shard

from torchtitan.components.optim import AdamW, DistMuon, OptimizersContainer
from torchtitan.distributed.flex_shard import BucketConfig, ComputeLayout, Owned
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.experiments.glm53_flash.model import GlmMoE
from torchtitan.models.common.moe import MoE


@dataclass(frozen=True, slots=True)
class ParameterCounts:
    total: int
    non_embedding: int
    active_non_embedding: int


def _parameter_ids(modules: Iterable[nn.Module | None]) -> set[int]:
    result: set[int] = set()
    for module in modules:
        if module is None:
            continue
        result.update(id(param) for param in module.parameters())
    return result


def parameter_counts(model: nn.Module) -> ParameterCounts:
    """Count total, non-embedding, and active non-embedding parameters."""

    named_parameters = list(model.named_parameters())
    total = sum(param.numel() for _, param in named_parameters)
    embedding_or_head_ids = _parameter_ids(
        (getattr(model, "tok_embeddings", None), getattr(model, "lm_head", None))
    )
    active_weights = {id(param): Fraction(1) for _, param in named_parameters}

    for module in model.modules():
        if isinstance(module, MoE):
            ratio = Fraction(module.router.top_k, module.router.num_experts)
            for param in module.routed_experts.parameters():
                active_weights[id(param)] = ratio
        elif isinstance(module, GlmMoE):
            ratio = Fraction(module.gate.top_k, module.gate.num_experts)
            for param in module.experts.parameters():
                active_weights[id(param)] = ratio

    non_embedding = 0
    active = Fraction(0)
    for _, param in named_parameters:
        if id(param) in embedding_or_head_ids:
            continue
        non_embedding += param.numel()
        active += param.numel() * active_weights[id(param)]
    assert active.denominator == 1
    return ParameterCounts(
        total=total,
        non_embedding=non_embedding,
        active_non_embedding=active.numerator,
    )


def _owned_layout() -> ComputeLayout:
    return ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: Owned(),
        },
    )


def _matrix_batch_layout() -> ComputeLayout:
    return ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: Shard(0),
        },
    )


def _module_by_parameter_id(model: nn.Module) -> dict[int, tuple[str, nn.Module]]:
    result: dict[int, tuple[str, nn.Module]] = {}
    for module_name, module in model.named_modules():
        for param in module.parameters(recurse=False):
            result[id(param)] = (module_name, module)
    return result


def _is_embedding_or_head(name: str, param: nn.Parameter, model: nn.Module) -> bool:
    embedding_or_head_ids = _parameter_ids(
        (getattr(model, "tok_embeddings", None), getattr(model, "lm_head", None))
    )
    return id(param) in embedding_or_head_ids or name in {
        "tok_embeddings.weight",
        "token_embeddings.weight",
        "lm_head.weight",
    }


def _is_muon_candidate(
    name: str,
    param: nn.Parameter,
    model: nn.Module,
    module_by_param: dict[int, tuple[str, nn.Module]],
    *,
    moe_matrix_batches_on_muon: bool,
    moe_router_on_muon: bool,
) -> bool:
    if not param.requires_grad or param.ndim < 2:
        return False
    if _is_embedding_or_head(name, param, model):
        return False
    if param.ndim > 2:
        return moe_matrix_batches_on_muon and _is_routed_expert_matrix_batch(name)
    if not moe_router_on_muon and ".moe.router." in name:
        return False
    lowered = name.lower()
    adam_fragments = (
        "norm",
        "expert_bias",
        "e_score_correction_bias",
        "dt_bias",
        "a_log",
        "indexer",
        "res_proj",
        "res_norm",
        "output_res",
        "index_kpool",
        "index_pool",
        "attn_hc",
        "ffn_hc",
        "hyper",
    )
    if any(fragment in lowered for fragment in adam_fragments):
        return False
    _, module = module_by_param[id(param)]
    if isinstance(module, nn.Embedding):
        return False
    if module.__class__.__name__.lower().endswith("norm"):
        return False
    return True


def _is_routed_expert_matrix_batch(name: str) -> bool:
    """Return true for native expert stacks that DistMuon orthogonalizes per matrix."""

    return any(
        fragment in name
        for fragment in (
            ".routed_experts.w13.weight",
            ".routed_experts.w2.weight",
            ".mlp.experts.gate_up_proj",
            ".mlp.experts.down_proj",
        )
    )


def muon_parameter_names(
    model: nn.Module,
    *,
    moe_matrix_batches_on_muon: bool = True,
    moe_router_on_muon: bool = True,
) -> set[str]:
    """Return trainable hidden matrices and expert matrix batches owned by Muon."""

    module_by_param = _module_by_parameter_id(model)
    return {
        name
        for name, param in model.named_parameters()
        if _is_muon_candidate(
            name,
            param,
            model,
            module_by_param,
            moe_matrix_batches_on_muon=moe_matrix_batches_on_muon,
            moe_router_on_muon=moe_router_on_muon,
        )
    }


def adamw_parameter_names(
    model: nn.Module,
    *,
    moe_matrix_batches_on_muon: bool = True,
    moe_router_on_muon: bool = True,
) -> set[str]:
    trainable = {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    return trainable - muon_parameter_names(
        model,
        moe_matrix_batches_on_muon=moe_matrix_batches_on_muon,
        moe_router_on_muon=moe_router_on_muon,
    )


def _exact_regex(names: Iterable[str]) -> str:
    return r"^(?:" + "|".join(re.escape(name) for name in sorted(names)) + r")$"


def ladder_optimizer_config(
    model: nn.Module,
    *,
    muon_lr: float,
    adamw_lr: float,
    weight_decay: float,
    momentum: float,
    moe_matrix_batches_on_muon: bool = True,
    moe_router_on_muon: bool = True,
) -> OptimizersContainer.Config:
    """Build the S07 Muon/AdamW first-match optimizer partition."""

    muon_names = muon_parameter_names(
        model,
        moe_matrix_batches_on_muon=moe_matrix_batches_on_muon,
        moe_router_on_muon=moe_router_on_muon,
    )
    if not muon_names:
        raise ValueError("architecture ladder expected at least one Muon parameter")
    compute_layouts = {
        name: (
            _matrix_batch_layout()
            if _is_routed_expert_matrix_batch(name)
            else _owned_layout()
        )
        for name in muon_names
    }
    return OptimizersContainer.Config(
        optimizers=[
            DistMuon.Config(
                pattern=_exact_regex(muon_names),
                compute_sharding_by_fqn=compute_layouts,
                bucket_configs=(
                    BucketConfig(name="hidden_matrices", patterns=tuple(muon_names)),
                ),
                lr=muon_lr,
                weight_decay=weight_decay,
                momentum=momentum,
                adjust_lr_fn="match_rms_adamw",
            ),
            AdamW.Config(
                pattern=r".*",
                lr=adamw_lr,
                betas=(0.9, 0.95),
                eps=1.0e-8,
                weight_decay=weight_decay,
                foreach=True,
                fused=False,
            ),
        ],
    )
