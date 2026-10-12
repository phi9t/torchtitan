#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare Mini Kimi K3 logits against a first-party forward oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saved Mini Kimi K3 logits and write an oracle report."
    )
    parser.add_argument(
        "--first-party-logits",
        type=Path,
        required=True,
        help="Torch-saved tensor or {'logits': tensor} from the first-party model.",
    )
    parser.add_argument(
        "--torchtitan-logits",
        type=Path,
        required=True,
        help="Torch-saved tensor or {'logits': tensor} from the TorchTitan model.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to write the Mini-K3 forward-oracle JSON report.",
    )
    parser.add_argument(
        "--model-flavor",
        default="r1",
        help="Mini-K3 model flavor being compared.",
    )
    parser.add_argument(
        "--max-abs-diff",
        type=float,
        default=1.0e-5,
        help="Maximum accepted absolute logit difference.",
    )
    parser.add_argument(
        "--max-rel-diff",
        type=float,
        default=1.0e-5,
        help="Maximum accepted relative logit difference.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = compare_forward_oracle(
            first_party_logits_path=args.first_party_logits,
            torchtitan_logits_path=args.torchtitan_logits,
            model_flavor=args.model_flavor,
            max_abs_diff=args.max_abs_diff,
            max_rel_diff=args.max_rel_diff,
        )
    except (FileNotFoundError, ValueError, TypeError) as exc:
        report = {
            "schema_version": 1,
            "kind": "mini_kimi_k3_forward_oracle",
            "status": "fail",
            "model_flavor": args.model_flavor,
            "source": {
                "first_party_logits": str(args.first_party_logits),
                "torchtitan_logits": str(args.torchtitan_logits),
            },
            "tolerances": {
                "max_abs_diff": args.max_abs_diff,
                "max_rel_diff": args.max_rel_diff,
            },
            "error": str(exc),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "pass":
        print("forward oracle comparison failed", file=sys.stderr)
        print(f"wrote forward oracle report: {args.report}", file=sys.stderr)
        return 21
    print("forward oracle comparison passed", file=sys.stderr)
    print(f"wrote forward oracle report: {args.report}", file=sys.stderr)
    return 0


def compare_forward_oracle(
    *,
    first_party_logits_path: Path,
    torchtitan_logits_path: Path,
    model_flavor: str,
    max_abs_diff: float,
    max_rel_diff: float,
) -> dict[str, Any]:
    first_party = _load_logits(first_party_logits_path)
    torchtitan = _load_logits(torchtitan_logits_path)
    _validate_finite(first_party, name="first-party logits")
    _validate_finite(torchtitan, name="TorchTitan logits")
    if first_party.shape != torchtitan.shape:
        raise ValueError(
            "logit shape mismatch: "
            f"{tuple(first_party.shape)} != {tuple(torchtitan.shape)}"
        )

    first_party = first_party.detach().to(torch.float64).cpu()
    torchtitan = torchtitan.detach().to(torch.float64).cpu()
    diff = (torchtitan - first_party).abs()
    denom = first_party.abs().clamp_min(1.0e-12)
    rel = diff / denom
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    max_rel = float(rel.max().item()) if rel.numel() else 0.0
    status = "pass" if max_abs <= max_abs_diff and max_rel <= max_rel_diff else "fail"
    return {
        "schema_version": 1,
        "kind": "mini_kimi_k3_forward_oracle",
        "status": status,
        "model_flavor": model_flavor,
        "source": {
            "first_party_logits": str(first_party_logits_path),
            "torchtitan_logits": str(torchtitan_logits_path),
        },
        "tolerances": {
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
        },
        "results": {
            "shape": list(first_party.shape),
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
        },
    }


def _load_logits(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"logits file not found: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, torch.Tensor):
        return loaded
    if isinstance(loaded, dict):
        logits = loaded.get("logits")
        if isinstance(logits, torch.Tensor):
            return logits
    raise TypeError(f"logits file must contain a tensor or a logits tensor: {path}")


def _validate_finite(logits: torch.Tensor, *, name: str) -> None:
    if logits.is_floating_point() and not torch.isfinite(logits).all():
        raise ValueError(f"{name} contain non-finite values")


if __name__ == "__main__":
    raise SystemExit(main())
