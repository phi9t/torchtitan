# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trace exact-state Mini Kimi K3 forward differences layer by layer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

from torchtitan.config import ConfigManager
from torchtitan.experiments.mini_kimi_k3.model import apply_attention_residual

from .dump_first_party_logits import (
    _install_transformers_compat_shims,
    _load_config_payload,
    _parse_input_ids,
    _seed_everything,
    _validate_source_root,
)
from .dump_torchtitan_logits import (
    _load_first_party_state_dict,
    _patch_first_party_transformers_drift,
    align_first_party_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact-state first-party and TorchTitan Mini-K3 models, then "
            "trace forward differences without keeping both full models on GPU."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Rootfs-visible first-party Mini-K3 checkout root.",
    )
    parser.add_argument(
        "--first-party-state-dict",
        type=Path,
        required=True,
        help="State payload from dump-first-party-logits --state-dict-output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON report path for the layer trace.",
    )
    parser.add_argument(
        "--input-ids",
        required=True,
        help="Comma-separated token IDs for one batch row.",
    )
    parser.add_argument(
        "--config",
        default="mini_kimi_k3_r1_contract",
        help="TorchTitan Mini-K3 config_registry entry.",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        help="Optional first-party KimiLinearConfig JSON. Defaults to train.ladder r1.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device used for one layer at a time, e.g. cuda:0 or cpu.",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = trace_forward_oracle(
            source_root=args.source_root,
            first_party_state_dict=args.first_party_state_dict,
            output_path=args.output,
            input_ids_csv=args.input_ids,
            config_name=args.config,
            config_json=args.config_json,
            device=args.device,
            seed=args.seed,
        )
    except (AttributeError, ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"Mini Kimi K3 forward trace failed: {exc}", file=sys.stderr)
        return 21
    print(f"wrote Mini-K3 forward trace: {args.output}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 21


def trace_forward_oracle(
    *,
    source_root: Path,
    first_party_state_dict: Path,
    output_path: Path,
    input_ids_csv: str,
    config_name: str,
    config_json: Path | None,
    device: str,
    seed: int,
) -> dict[str, Any]:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise RuntimeError(
            "Mini-K3 forward trace must run inside the TorchTitan rootfs"
        )
    _validate_source_root(source_root)
    input_ids = _parse_input_ids(input_ids_csv)
    _seed_everything(seed)

    state_payload = _load_first_party_state_dict(first_party_state_dict)
    state_dict = state_payload["state_dict"]
    torch_device = torch.device(device)

    first_party_model = _build_first_party_model(
        source_root=source_root,
        config_json=config_json,
        state_dict=state_dict,
    )
    config = ConfigManager().parse_args(
        ["--module", "mini_kimi_k3", "--config", config_name]
    )
    torch_titan_model = config.model_spec.model.build()
    torch_titan_model.init_states()
    alignment_report = align_first_party_state_dict(state_dict, torch_titan_model)
    first_party_model.eval()
    torch_titan_model.eval()

    report = _trace_models(
        first_party_model=first_party_model,
        torch_titan_model=torch_titan_model.inner,
        input_ids=input_ids,
        device=torch_device,
    )
    report.update(
        {
            "schema_version": 1,
            "kind": "mini_kimi_k3_forward_trace",
            "status": "pass",
            "config": config_name,
            "model_flavor": config.model_spec.flavor,
            "seed": seed,
            "device": str(torch_device),
            "source": {
                "source_root": str(source_root),
                "first_party_state_dict": str(first_party_state_dict),
                "config_json": str(config_json) if config_json is not None else None,
            },
            "alignment_report": alignment_report,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _build_first_party_model(
    *,
    source_root: Path,
    config_json: Path | None,
    state_dict: dict[str, torch.Tensor],
) -> torch.nn.Module:
    sys.path.insert(0, str(source_root))
    _install_transformers_compat_shims()
    config_payload = _load_config_payload(
        source_root=source_root,
        config_json=config_json,
    )
    try:
        from model._ref import modeling_kimi_linear
        from model._ref.configuration_kimi_k3 import KimiLinearConfig
        from model._ref.modeling_kimi_linear import KimiLinearForCausalLM
    except ImportError as exc:
        raise ImportError(
            "first-party Mini-K3 source root is visible but not importable with "
            f"the current rootfs Python environment: {exc}"
        ) from exc

    _patch_first_party_transformers_drift(modeling_kimi_linear, KimiLinearForCausalLM)
    model = KimiLinearForCausalLM(KimiLinearConfig(**config_payload))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "first-party state dict did not load cleanly: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model


def _trace_models(
    *,
    first_party_model: torch.nn.Module,
    torch_titan_model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    with torch.no_grad():
        first_party_hidden = first_party_model.model.embed_tokens(input_ids).to(device)
        torch_titan_hidden = torch_titan_model.embed_tokens(input_ids).to(device)

    batch_size, seq_len, hidden_size = first_party_hidden.shape
    first_party_block = first_party_hidden.new_zeros(
        batch_size * seq_len, 0, hidden_size
    )
    torch_titan_block = torch.zeros_like(first_party_block)
    recursive_first_party_hidden = first_party_hidden
    recursive_torch_titan_hidden = torch_titan_hidden
    recursive_first_party_block = first_party_block
    recursive_torch_titan_block = torch_titan_block

    layer_reports: list[dict[str, Any]] = []
    for layer_idx, (first_party_layer, torch_titan_layer) in enumerate(
        zip(first_party_model.model.layers, torch_titan_model.layers)
    ):
        first_party_layer.to(device)
        torch_titan_layer.to(device)
        try:
            with torch.no_grad():
                (
                    isolated_torch_titan_hidden,
                    isolated_torch_titan_block,
                ) = torch_titan_layer(
                    recursive_first_party_hidden,
                    block_residual=recursive_first_party_block,
                )
                (next_first_party_hidden, next_first_party_block,) = first_party_layer(
                    recursive_first_party_hidden,
                    block_residual=recursive_first_party_block,
                    use_cache=False,
                )
                (next_torch_titan_hidden, next_torch_titan_block,) = torch_titan_layer(
                    recursive_torch_titan_hidden,
                    block_residual=recursive_torch_titan_block,
                )

            layer_reports.append(
                {
                    "layer": layer_idx,
                    "is_linear_attention": bool(first_party_layer.is_linear_attn),
                    "isolated_output": _diff_stats(
                        next_first_party_hidden,
                        isolated_torch_titan_hidden,
                    ),
                    "isolated_block_residual": _diff_stats(
                        next_first_party_block,
                        isolated_torch_titan_block,
                    ),
                    "recursive_output": _diff_stats(
                        next_first_party_hidden,
                        next_torch_titan_hidden,
                    ),
                    "recursive_block_residual": _diff_stats(
                        next_first_party_block,
                        next_torch_titan_block,
                    ),
                }
            )
            recursive_first_party_hidden = next_first_party_hidden
            recursive_torch_titan_hidden = next_torch_titan_hidden
            recursive_first_party_block = next_first_party_block
            recursive_torch_titan_block = next_torch_titan_block
        finally:
            first_party_layer.to("cpu")
            torch_titan_layer.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    first_party_model.model.output_attn_res_norm.to(device)
    first_party_model.model.output_attn_res_proj.to(device)
    first_party_model.model.norm.to(device)
    first_party_model.lm_head.to(device)
    torch_titan_model.output_attn_res_norm.to(device)
    torch_titan_model.output_attn_res_proj.to(device)
    torch_titan_model.norm.to(device)
    torch_titan_model.lm_head.to(device)
    with torch.no_grad():
        first_party_resolved = first_party_model.model._apply_output_attn_res(
            recursive_first_party_hidden,
            recursive_first_party_block,
        )
        torch_titan_resolved = apply_attention_residual(
            recursive_torch_titan_hidden.view(-1, hidden_size),
            recursive_torch_titan_block,
            torch_titan_model.output_attn_res_proj,
            torch_titan_model.output_attn_res_norm,
        ).view_as(recursive_torch_titan_hidden)
        first_party_normed = first_party_model.model.norm(first_party_resolved)
        torch_titan_normed = torch_titan_model.norm(torch_titan_resolved)
        first_party_logits = first_party_model.lm_head(first_party_normed)
        torch_titan_logits = torch_titan_model.lm_head(torch_titan_normed)

    return {
        "input_ids": input_ids.tolist(),
        "embedding": _diff_stats(first_party_hidden, torch_titan_hidden),
        "layers": layer_reports,
        "final": {
            "output_attention_residual": _diff_stats(
                first_party_resolved,
                torch_titan_resolved,
            ),
            "norm": _diff_stats(first_party_normed, torch_titan_normed),
            "logits": _diff_stats(first_party_logits, torch_titan_logits),
        },
    }


def _diff_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    diff = (left.detach().float() - right.detach().float()).abs().cpu()
    if diff.numel() == 0:
        return {
            "shape": list(left.shape),
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
        }
    return {
        "shape": list(left.shape),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
