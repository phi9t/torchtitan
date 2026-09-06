#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dump TorchTitan Mini Kimi K3 logits for forward-oracle comparison."""

from __future__ import annotations

import argparse
import inspect
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torchtitan.config import ConfigManager

from .dump_first_party_logits import (
    _install_transformers_compat_shims,
    _load_config_payload,
    _validate_source_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Mini-K3 TorchTitan config and save logits for inputs."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Mini-K3 config_registry entry, e.g. mini_kimi_k3_tiny_plumbing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Torch .pt output path for the logits payload.",
    )
    parser.add_argument(
        "--input-ids",
        required=True,
        help="Comma-separated token IDs for one batch row.",
    )
    parser.add_argument(
        "--allow-unverified-r1",
        action="store_true",
        help=(
            "Allow dumping logits from the r1 candidate model before a reviewed "
            "launch-backend report exists. "
            "The payload is labeled unverified and must not be used as launch "
            "readiness evidence unless compare-forward-oracle passes."
        ),
    )
    parser.add_argument(
        "--align-first-party-weights",
        action="store_true",
        help=(
            "Build the first-party Mini-K3 model in-process and copy every "
            "mapped parameter into the TorchTitan candidate before dumping. "
            "This is an oracle diagnostic, not launch readiness evidence."
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Rootfs-visible first-party Mini-K3 checkout root for alignment.",
    )
    parser.add_argument(
        "--first-party-state-dict",
        type=Path,
        help=(
            "Optional first-party state-dict payload produced by "
            "dump-first-party-logits --state-dict-output. When supplied, "
            "TorchTitan aligns from this exact payload instead of building a "
            "fresh first-party model."
        ),
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        help="Optional first-party KimiLinearConfig JSON. Defaults to train.ladder r1.",
    )
    parser.add_argument(
        "--device",
        help=(
            "Torch device for aligned first-party and TorchTitan models. "
            "Defaults to cuda:0 for aligned dumps when CUDA is available."
        ),
    )
    parser.add_argument(
        "--prepare-for-training",
        action="store_true",
        help="Apply first-party from-scratch init/router patches before alignment.",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dump_torchtitan_logits(
            config_name=args.config,
            output_path=args.output,
            input_ids_csv=args.input_ids,
            allow_unverified_r1=args.allow_unverified_r1,
            align_first_party_weights_enabled=args.align_first_party_weights,
            source_root=args.source_root,
            first_party_state_dict=args.first_party_state_dict,
            config_json=args.config_json,
            device=args.device,
            prepare_for_training=args.prepare_for_training,
            seed=args.seed,
        )
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 21
    except (AttributeError, ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"Mini Kimi K3 TorchTitan logits dump failed: {exc}", file=sys.stderr)
        return 21
    print(f"wrote TorchTitan Mini-K3 logits: {args.output}", file=sys.stderr)
    return 0


def dump_torchtitan_logits(
    *,
    config_name: str,
    output_path: Path,
    input_ids_csv: str,
    allow_unverified_r1: bool,
    align_first_party_weights_enabled: bool = False,
    source_root: Path | None = None,
    first_party_state_dict: Path | None = None,
    config_json: Path | None = None,
    device: str | None = None,
    prepare_for_training: bool = False,
    seed: int,
) -> None:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise RuntimeError("Mini-K3 logits dump must run inside the TorchTitan rootfs")
    if align_first_party_weights_enabled and source_root is None:
        if first_party_state_dict is None:
            raise ValueError(
                "--source-root or --first-party-state-dict is required with "
                "--align-first-party-weights"
            )

    input_ids = _parse_input_ids(input_ids_csv)
    _seed_everything(seed)

    config = ConfigManager().parse_args(
        ["--module", "mini_kimi_k3", "--config", config_name]
    )
    if config.model_spec.flavor == "r1_contract":
        if allow_unverified_r1:
            fidelity_status = "unverified_candidate"
        else:
            raise NotImplementedError(
                "Mini-K3 r1 logits dump requires a verified forward-oracle path; "
                "pass --allow-unverified-r1 only for explicit oracle diagnostics."
            )
    else:
        fidelity_status = "tiny_plumbing"
    model = config.model_spec.model.build()
    model.init_states()
    alignment_report = None
    torch_device = _resolve_device(
        device,
        align_first_party_weights_enabled=align_first_party_weights_enabled,
    )
    if align_first_party_weights_enabled:
        if config.model_spec.flavor != "r1_contract":
            raise ValueError(
                "--align-first-party-weights is only supported for r1_contract"
            )
        if first_party_state_dict is not None:
            state_payload = _load_first_party_state_dict(first_party_state_dict)
            alignment_report = align_first_party_state_dict(
                state_payload["state_dict"],
                model,
            )
            prepare_report = state_payload.get("prepare_report")
            prepare_for_training = bool(state_payload.get("prepared_for_training"))
        else:
            assert source_root is not None
            first_party_model, prepare_report = _build_first_party_model(
                source_root=source_root,
                config_json=config_json,
                device=torch_device,
                prepare_for_training=prepare_for_training,
            )
            alignment_report = align_first_party_weights(first_party_model, model)
            del first_party_model
        fidelity_status = "aligned_unverified_candidate"
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
    model.to(torch_device)
    model.eval()

    if input_ids.numel() == 0:
        raise ValueError("input-ids must contain at least one token")

    input_ids = input_ids.to(torch_device)
    positions = torch.arange(
        input_ids.shape[1],
        dtype=torch.long,
        device=torch_device,
    ).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids, positions=positions)

    payload = {
        "config": config_name,
        "model_flavor": config.model_spec.flavor,
        "fidelity_status": fidelity_status,
        "seed": seed,
        "input_ids": input_ids.cpu(),
        "logits": logits.detach().cpu(),
    }
    if alignment_report is not None:
        payload.update(
            {
                "device": str(torch_device),
                "alignment_report": alignment_report,
                "source_root": str(source_root),
                "first_party_state_dict": (
                    str(first_party_state_dict)
                    if first_party_state_dict is not None
                    else None
                ),
                "config_json": str(config_json) if config_json is not None else None,
                "prepared_for_training": prepare_for_training,
                "prepare_report": prepare_report,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def align_first_party_weights(
    first_party_model: torch.nn.Module,
    torch_titan_model: torch.nn.Module,
) -> dict[str, Any]:
    return align_first_party_state_dict(
        dict(first_party_model.named_parameters()),
        torch_titan_model,
    )


def align_first_party_state_dict(
    first_party_state_dict: dict[str, torch.Tensor],
    torch_titan_model: torch.nn.Module,
) -> dict[str, Any]:
    first_party_params = {
        name: tensor
        for name, tensor in first_party_state_dict.items()
        if _is_parameter_like_tensor(name, tensor)
    }
    first_party_params = _drop_redundant_tied_weights(first_party_params)
    torch_titan_params = dict(torch_titan_model.named_parameters())

    mapped: list[tuple[str, str, torch.nn.Parameter]] = []
    missing: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    for first_party_name, first_party_param in first_party_params.items():
        torch_titan_name = _to_torchtitan_parameter_name(first_party_name)
        torch_titan_param = torch_titan_params.get(torch_titan_name)
        if torch_titan_param is None:
            missing.append(first_party_name)
            continue
        if tuple(torch_titan_param.shape) != tuple(first_party_param.shape):
            shape_mismatches.append(
                {
                    "first_party": first_party_name,
                    "torch_titan": torch_titan_name,
                    "first_party_shape": list(first_party_param.shape),
                    "torch_titan_shape": list(torch_titan_param.shape),
                }
            )
            continue
        mapped.append((first_party_name, torch_titan_name, first_party_param))

    expected_torch_titan = {
        _to_torchtitan_parameter_name(first_party_name)
        for first_party_name in first_party_params
    }
    extra = sorted(set(torch_titan_params) - expected_torch_titan)
    report = {
        "status": "pass",
        "first_party_parameters": len(first_party_params),
        "torch_titan_parameters": len(torch_titan_params),
        "mapped_parameters": len(mapped),
        "first_party_elements": sum(
            parameter.numel() for parameter in first_party_params.values()
        ),
        "torch_titan_elements": sum(
            parameter.numel() for parameter in torch_titan_params.values()
        ),
        "mapped_elements": sum(parameter.numel() for _, _, parameter in mapped),
        "missing": sorted(missing),
        "extra": extra,
        "shape_mismatches": shape_mismatches,
    }
    if missing or extra or shape_mismatches:
        report["status"] = "fail"
        raise RuntimeError(
            "Mini-K3 first-party weight alignment is incomplete: " f"{report}"
        )

    with torch.no_grad():
        for _, torch_titan_name, first_party_param in mapped:
            torch_titan_params[torch_titan_name].copy_(
                first_party_param.to(
                    device=torch_titan_params[torch_titan_name].device,
                    dtype=torch_titan_params[torch_titan_name].dtype,
                )
            )
    return report


def _to_torchtitan_parameter_name(first_party_name: str) -> str:
    if first_party_name.startswith("model."):
        torch_titan_name = f"inner.{first_party_name.removeprefix('model.')}"
    elif first_party_name.startswith("lm_head."):
        torch_titan_name = f"inner.{first_party_name}"
    else:
        torch_titan_name = first_party_name

    replacements = (
        (".mlp.gate_proj.", ".feed_forward.gate_proj."),
        (".mlp.down_proj.", ".feed_forward.down_proj."),
        (".mlp.up_proj.", ".feed_forward.up_proj."),
        (".block_sparse_moe.gate.", ".block_sparse_moe.router."),
    )
    for old, new in replacements:
        torch_titan_name = torch_titan_name.replace(old, new)
    return torch_titan_name


def _is_parameter_like_tensor(name: str, value: object) -> bool:
    if not isinstance(value, torch.Tensor):
        return False
    return not (
        name.endswith(".expert_load")
        or name.endswith(".tokens_seen")
        or name.endswith(".inv_freq")
    )


def _drop_redundant_tied_weights(
    first_party_params: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    params = dict(first_party_params)
    lm_head = params.get("lm_head.weight")
    embed = params.get("model.embed_tokens.weight")
    if lm_head is not None and embed is not None:
        if tuple(lm_head.shape) != tuple(embed.shape):
            raise RuntimeError(
                "Mini-K3 first-party tied weight shapes differ: "
                f"lm_head.weight {tuple(lm_head.shape)} vs "
                f"model.embed_tokens.weight {tuple(embed.shape)}"
            )
        params.pop("lm_head.weight")
    return params


def _load_first_party_state_dict(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError(f"first-party state dict payload is invalid: {path}")
        return payload
    if isinstance(payload, dict):
        return {"state_dict": payload}
    raise ValueError(f"first-party state dict payload is invalid: {path}")


def _build_first_party_model(
    *,
    source_root: Path,
    config_json: Path | None,
    device: torch.device,
    prepare_for_training: bool,
) -> tuple[torch.nn.Module, dict[str, Any] | None]:
    _validate_source_root(source_root)
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
    model = KimiLinearForCausalLM(KimiLinearConfig(**config_payload)).to(device)
    prep_report = None
    if prepare_for_training:
        try:
            from train.init_patch import prepare_for_training as prepare_model
        except ImportError as exc:
            raise ImportError(
                "first-party Mini-K3 prepare_for_training patch is not importable "
                f"from {source_root}: {exc}"
            ) from exc
        prep_report = prepare_model(model)
    model.eval()
    return model, prep_report


def _patch_first_party_transformers_drift(
    modeling_module: Any,
    causal_lm_cls: type[torch.nn.Module],
) -> None:
    tied_keys = getattr(causal_lm_cls, "_tied_weights_keys", None)
    if isinstance(tied_keys, list):
        causal_lm_cls._tied_weights_keys = {
            "lm_head.weight": "model.embed_tokens.weight",
        }

    create_causal_mask = getattr(modeling_module, "create_causal_mask", None)
    if create_causal_mask is None:
        return
    signature = inspect.signature(create_causal_mask)
    if (
        "input_embeds" in signature.parameters
        or "inputs_embeds" not in signature.parameters
    ):
        return

    def create_causal_mask_compat(*args: Any, **kwargs: Any) -> Any:
        if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        supported = set(signature.parameters)
        kwargs = {key: value for key, value in kwargs.items() if key in supported}
        return create_causal_mask(*args, **kwargs)

    modeling_module.create_causal_mask = create_causal_mask_compat


def _resolve_device(
    device: str | None,
    *,
    align_first_party_weights_enabled: bool,
) -> torch.device:
    if device is not None:
        return torch.device(device)
    if align_first_party_weights_enabled and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _parse_input_ids(value: str) -> torch.Tensor:
    try:
        ids = [int(piece) for piece in value.split(",") if piece]
    except ValueError as exc:
        raise ValueError(
            f"input-ids must be comma-separated integers: {value}"
        ) from exc
    if not ids:
        raise ValueError("input-ids must contain at least one token")
    if min(ids) < 0:
        raise ValueError("input-ids must be non-negative")
    return torch.tensor([ids], dtype=torch.long)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    raise SystemExit(main())
