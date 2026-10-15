# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collect L2 Qwen3-MoE diagnostics for the architecture ladder."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from torchtitan.experiments.glm53_arch_speedrun.config_registry import _train_glob
from torchtitan.models.common.moe import MoE

from .data import PackedFineWebBinDataLoader
from .flavors import build_ladder_model_config, GPT2_VOCAB_SIZE
from .params import (
    adamw_parameter_names,
    ladder_optimizer_config,
    muon_parameter_names,
    parameter_counts,
)


def _set_cuda_memory_cap(cap_gib: float) -> None:
    if cap_gib <= 0.0 or not torch.cuda.is_available():
        return
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gib / total_gib))


def _float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float()
    return {
        "mean": _float(values.mean()),
        "std": _float(values.std(unbiased=False)),
        "abs_mean": _float(values.abs().mean()),
        "abs_max": _float(values.abs().max()),
    }


def _param_group(name: str) -> str:
    if "tok_embeddings" in name or "lm_head" in name:
        return "embedding_or_head"
    if ".moe.router." in name:
        return "router"
    if ".moe.routed_experts." in name:
        return "routed_experts"
    if ".moe.shared_experts." in name:
        return "shared_experts"
    if ".attention." in name:
        return "attention"
    if "norm" in name:
        return "norm"
    return "other"


def _grad_norms(model: torch.nn.Module) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        group = _param_group(name)
        grad = param.grad.detach().float()
        sums[group] += float(torch.sum(grad * grad).cpu().item())
    return {group: math.sqrt(value) for group, value in sorted(sums.items())}


def _router_summary(moes: list[tuple[str, MoE]]) -> list[dict[str, Any]]:
    summaries = []
    for name, moe in moes:
        counts = moe.router.tokens_per_expert_E.detach().float().cpu()
        total = float(counts.sum().item())
        if total > 0.0:
            fractions = counts / total
            entropy = -float(
                (fractions.clamp_min(1.0e-12) * fractions.clamp_min(1.0e-12).log())
                .sum()
                .item()
            ) / math.log(float(counts.numel()))
            max_fraction = float(fractions.max().item())
            min_fraction = float(fractions.min().item())
        else:
            fractions = torch.zeros_like(counts)
            entropy = 0.0
            max_fraction = 0.0
            min_fraction = 0.0
        expert_bias = moe.expert_bias_E
        summaries.append(
            {
                "module": name,
                "total_assignments": total,
                "entropy_normalized": entropy,
                "min_fraction": min_fraction,
                "max_fraction": max_fraction,
                "fractions": [float(v) for v in fractions.tolist()],
                "expert_bias_abs_max": (
                    None if expert_bias is None else _float(expert_bias.abs().max())
                ),
            }
        )
    return summaries


def _apply_aux_free_bias_update(moes: list[tuple[str, MoE]]) -> list[dict[str, Any]]:
    updates = []
    with torch.no_grad():
        for name, moe in moes:
            if moe.load_balance_coeff is None or moe.expert_bias_E is None:
                continue
            counts = moe.router.tokens_per_expert_E.float()
            before = moe.expert_bias_E.detach().float().clone()
            delta = moe.load_balance_coeff * torch.sign(counts.mean() - counts)
            delta = delta - delta.mean()
            moe.expert_bias_E.add_(delta)
            moe.router.tokens_per_expert_E.zero_()
            after = moe.expert_bias_E.detach().float()
            updates.append(
                {
                    "module": name,
                    "delta_abs_max": _float((after - before).abs().max()),
                    "bias_abs_max_after": _float(after.abs().max()),
                }
            )
    return updates


def _optimizer_assignment(model: torch.nn.Module) -> dict[str, Any]:
    muon_names = muon_parameter_names(model)
    adamw_names = adamw_parameter_names(model)
    optim_config = ladder_optimizer_config(
        model,
        muon_lr=0.02,
        adamw_lr=8.0e-4,
        weight_decay=0.1,
        momentum=0.95,
    )
    moe_params = []
    for name, param in model.named_parameters():
        if ".moe." not in name:
            continue
        moe_params.append(
            {
                "name": name,
                "shape": list(param.shape),
                "numel": param.numel(),
                "optimizer": "DistMuon" if name in muon_names else "AdamW",
            }
        )
    return {
        "muon_parameter_count": len(muon_names),
        "adamw_parameter_count": len(adamw_names),
        "muon_regex": optim_config.optimizers[0].pattern,
        "moe_parameters": moe_params,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _set_cuda_memory_cap(args.gpu_memory_cap_gib)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model_config = build_ladder_model_config("L2", seq_len=args.seq_len)
    model = model_config.build()
    model.init_states()
    model.to(device=device, dtype=dtype)
    model.train()

    counts = parameter_counts(model)
    optimizer = _optimizer_assignment(model)
    init_stats = {
        name: _tensor_stats(param)
        for name, param in model.named_parameters()
        if ".moe.router.gate.weight" in name
        or ".moe.routed_experts." in name
        or ".moe.shared_experts." in name
    }

    loader = PackedFineWebBinDataLoader.Config(
        bin_glob=_train_glob(args.data_dir),
        vocab_size=GPT2_VOCAB_SIZE,
        repeat=True,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=None,
        max_context_length=args.seq_len,
        num_tokens_per_microbatch=args.local_batch_size * args.seq_len,
    )
    data_iter = iter(loader)
    moes = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, MoE)
    ]

    step_summaries = []
    for step in range(1, args.steps + 1):
        batch = next(data_iter)
        tokens = batch.input.to(device)
        labels = batch.labels.to(device)
        positions = batch.positions.to(device)
        padding_mask = batch.padding_mask.to(device)
        aux_denominators = batch.routing_token_counts.to(device)

        model.zero_grad(set_to_none=True)
        logits = model(
            tokens,
            attention_metadata=model._get_attention_metadata(positions),
            positions=positions,
            padding_mask=padding_mask,
            aux_loss_denominators=aux_denominators,
        )
        loss = F.cross_entropy(logits.float(), labels)
        loss.backward()
        router_before_update = _router_summary(moes)
        bias_updates = _apply_aux_free_bias_update(moes)
        step_summaries.append(
            {
                "step": step,
                "loss": _float(loss),
                "grad_norms": _grad_norms(model),
                "router_before_bias_update": router_before_update,
                "bias_updates": bias_updates,
            }
        )

    first_moe = moes[0][1]
    return {
        "diagnostic": "l2_qwen3_moe",
        "seq_len": args.seq_len,
        "local_batch_size": args.local_batch_size,
        "steps": args.steps,
        "seed": args.seed,
        "parameter_counts": {
            "total": counts.total,
            "non_embedding": counts.non_embedding,
            "active_non_embedding": counts.active_non_embedding,
        },
        "optimizer_assignment": optimizer,
        "moe_contract": {
            "num_layers": len(moes),
            "num_experts": first_moe.num_experts,
            "top_k": first_moe.router.top_k,
            "route_norm": first_moe.router.route_norm,
            "load_balance_coeff": first_moe.load_balance_coeff,
            "has_shared_experts": first_moe.shared_experts is not None,
            "capacity": "dropless token-choice routing",
            "dropped_tokens": 0,
        },
        "initialization_stats": init_stats,
        "steps_summary": step_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-cap-gib", type=float, default=24.0)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
