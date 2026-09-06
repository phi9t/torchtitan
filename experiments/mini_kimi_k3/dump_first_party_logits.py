#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dump first-party Mini Kimi K3 logits for forward-oracle comparison."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the first-party Mini-K3 r1 model and save logits."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Rootfs-visible first-party Mini-K3 checkout root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Torch .pt output path for the logits payload.",
    )
    parser.add_argument(
        "--state-dict-output",
        type=Path,
        help=(
            "Optional Torch .pt path for the exact first-party state dict used "
            "for this logits dump."
        ),
    )
    parser.add_argument(
        "--input-ids",
        required=True,
        help="Comma-separated token IDs for one batch row.",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        help="Optional KimiLinearConfig JSON. Defaults to train.ladder r1.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for the first-party forward pass, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--prepare-for-training",
        action="store_true",
        help="Apply first-party from-scratch init/router patches before the forward.",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dump_first_party_logits(
            source_root=args.source_root,
            output_path=args.output,
            state_dict_output_path=args.state_dict_output,
            input_ids_csv=args.input_ids,
            config_json=args.config_json,
            device=args.device,
            prepare_for_training=args.prepare_for_training,
            seed=args.seed,
        )
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"Mini Kimi K3 first-party logits dump failed: {exc}", file=sys.stderr)
        return 21
    print(f"wrote first-party Mini-K3 logits: {args.output}", file=sys.stderr)
    return 0


def dump_first_party_logits(
    *,
    source_root: Path,
    output_path: Path,
    state_dict_output_path: Path | None,
    input_ids_csv: str,
    config_json: Path | None,
    device: str,
    prepare_for_training: bool,
    seed: int,
) -> None:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise RuntimeError(
            "Mini-K3 first-party logits dump must run inside the TorchTitan rootfs"
        )
    _validate_source_root(source_root)

    input_ids = _parse_input_ids(input_ids_csv)
    _seed_everything(seed)

    sys.path.insert(0, str(source_root))
    _install_transformers_compat_shims()
    config_payload = _load_config_payload(
        source_root=source_root, config_json=config_json
    )
    try:
        from model._ref.configuration_kimi_k3 import KimiLinearConfig
        from model._ref.modeling_kimi_linear import KimiLinearForCausalLM
    except ImportError as exc:
        raise ImportError(
            "first-party Mini-K3 source root is visible but not importable with "
            f"the current rootfs Python environment: {exc}"
        ) from exc

    config = KimiLinearConfig(**config_payload)
    torch_device = torch.device(device)
    model = KimiLinearForCausalLM(config).to(torch_device)
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

    if input_ids.numel() == 0:
        raise ValueError("input-ids must contain at least one token")

    forward_kwargs: dict[str, Any] = {"input_ids": input_ids.to(torch_device)}
    use_cache = False
    if "use_cache" in inspect.signature(model.forward).parameters:
        forward_kwargs["use_cache"] = False

    with torch.no_grad():
        outputs = model(**forward_kwargs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source_root": str(source_root),
            "config_json": str(config_json) if config_json is not None else None,
            "model_flavor": "r1",
            "seed": seed,
            "device": str(torch_device),
            "use_cache": use_cache,
            "prepared_for_training": prepare_for_training,
            "prepare_report": prep_report,
            "input_ids": input_ids.cpu(),
            "logits": logits.detach().cpu(),
        },
        output_path,
    )
    if state_dict_output_path is not None:
        state_dict_output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "source_root": str(source_root),
                "config_json": str(config_json) if config_json is not None else None,
                "model_flavor": "r1",
                "seed": seed,
                "device": str(torch_device),
                "use_cache": use_cache,
                "prepared_for_training": prepare_for_training,
                "prepare_report": prep_report,
                "state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
            },
            state_dict_output_path,
        )


def _validate_source_root(source_root: Path) -> None:
    if not source_root.is_dir():
        raise RuntimeError(
            "first-party Mini-K3 source root is not visible inside the TorchTitan "
            f"rootfs: {source_root}"
        )
    required_files = (
        "model/_ref/configuration_kimi_k3.py",
        "model/_ref/modeling_kimi_linear.py",
        "train/ladder.py",
    )
    missing = [path for path in required_files if not (source_root / path).is_file()]
    if missing:
        raise RuntimeError(
            "first-party Mini-K3 source root is incomplete inside the TorchTitan "
            f"rootfs: {source_root}; missing {', '.join(missing)}"
        )


class OutputRecorder:  # noqa: B903
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs


def _install_transformers_compat_shims() -> None:
    import transformers.utils.generic as generic

    if hasattr(generic, "OutputRecorder"):
        return

    generic.OutputRecorder = OutputRecorder


def _load_config_payload(
    *,
    source_root: Path,
    config_json: Path | None,
) -> dict[str, Any]:
    if config_json is not None:
        try:
            return json.loads(config_json.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"first-party config JSON is invalid: {config_json}"
            ) from exc

    try:
        from train.ladder import all_configs
    except ImportError as exc:
        raise ImportError(
            "first-party Mini-K3 ladder config is not importable from "
            f"{source_root}: {exc}"
        ) from exc
    configs = all_configs()
    r1 = configs.get("r1")
    if not isinstance(r1, dict):
        raise ValueError("first-party train.ladder.all_configs() did not return r1")
    return r1


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
