#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plan Mini Kimi K3 r1 corpus acquisition from the first-party mix plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from torchtitan.experiments.mini_kimi_k3.recipe import mini_k3_r1_launch_recipe


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale a first-party Mini Kimi K3 mix plan to the r1 token target."
    )
    parser.add_argument(
        "--mix-plan",
        type=Path,
        default=Path(
            "experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara/data/mix_plan.json"
        ),
        help="First-party mix_plan.json path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/reports/r1-corpus-plan.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = plan_r1_corpus(mix_plan_path=args.mix_plan, report_path=args.report)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 21

    print(
        "planned Mini Kimi K3 r1 corpus: "
        f"{report['total_planned_tokens']} tokens across "
        f"{len(report['sources'])} source(s)",
        file=sys.stderr,
    )
    return 0


def plan_r1_corpus(*, mix_plan_path: Path, report_path: Path) -> dict[str, Any]:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise ValueError(
            "Mini-K3 r1 corpus planning must run inside the TorchTitan rootfs"
        )
    if not mix_plan_path.is_file():
        raise FileNotFoundError(f"mix plan does not exist: {mix_plan_path}")

    mix_plan = json.loads(mix_plan_path.read_text())
    first_party_total = _positive_number(mix_plan, "total_tokens")
    source_tokens = _number_map(mix_plan, "source_tokens")
    tokens_per_file = _number_map(mix_plan, "tokens_per_file")
    files = _number_map(mix_plan, "files")

    target_tokens = mini_k3_r1_launch_recipe().target_tokens
    scale = target_tokens / first_party_total
    planned_source_tokens = {
        source: int(round(tokens * scale)) for source, tokens in source_tokens.items()
    }

    sources: dict[str, dict[str, Any]] = {}
    for source, tokens in planned_source_tokens.items():
        per_file = tokens_per_file.get(source)
        available_files = files.get(source)
        if per_file is None:
            raise ValueError(f"mix plan missing tokens_per_file for source {source!r}")
        if available_files is None:
            raise ValueError(f"mix plan missing files for source {source!r}")
        estimated_files = max(1, math.ceil(tokens / per_file)) if tokens > 0 else 0
        sources[source] = {
            "target_tokens": tokens,
            "source_fraction": source_tokens[source] / first_party_total,
            "tokens_per_file": per_file,
            "estimated_files": estimated_files,
            "first_party_files": int(available_files),
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_r1_corpus_plan",
        "status": "planned",
        "target_tokens": target_tokens,
        "first_party_total_tokens": first_party_total,
        "scale": scale,
        "mix_plan": {
            "path": str(mix_plan_path),
            "sha256": _sha256(mix_plan_path),
        },
        "source_tokens": planned_source_tokens,
        "total_planned_tokens": sum(planned_source_tokens.values()),
        "sources": sources,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _positive_number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"mix plan {key} must be a positive number")
    return float(value)


def _number_map(data: dict[str, Any], key: str) -> dict[str, float]:
    value = data.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"mix plan {key} must be a nonempty object")
    out = {}
    for name, item in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"mix plan {key} contains an invalid source name")
        if not isinstance(item, (int, float)) or item < 0:
            raise ValueError(f"mix plan {key}.{name} must be a nonnegative number")
        out[name] = float(item)
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
