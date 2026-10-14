# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare full and resumed rung losses from TensorBoard events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_LOSS_TAG = "loss_metrics/global_avg_loss"


def _event_dirs(run_dir: Path) -> list[Path]:
    tb_dir = run_dir / "tb"
    if not tb_dir.is_dir():
        raise FileNotFoundError(f"TensorBoard directory not found: {tb_dir}")
    event_dirs = sorted(
        (path for path in tb_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not event_dirs:
        raise FileNotFoundError(f"No TensorBoard event directories under {tb_dir}")
    return event_dirs


def _read_scalars(run_dir: Path, tag: str) -> tuple[dict[int, float], list[str]]:
    values: dict[int, float] = {}
    used_dirs = []
    for event_dir in _event_dirs(run_dir):
        accumulator = EventAccumulator(str(event_dir))
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        used_dirs.append(str(event_dir))
        for scalar in accumulator.Scalars(tag):
            values[int(scalar.step)] = float(scalar.value)
    if not values:
        raise KeyError(f"Scalar tag {tag!r} not found under {run_dir / 'tb'}")
    return dict(sorted(values.items())), used_dirs


def compare_resume_loss(
    *,
    full: Path,
    resumed: Path,
    output_json: Path,
    output_md: Path,
    tag: str = _LOSS_TAG,
    first_step: int = 63,
) -> dict[str, object]:
    full_values, full_dirs = _read_scalars(full, tag)
    resumed_values, resumed_dirs = _read_scalars(resumed, tag)
    common_steps = sorted(
        step for step in set(full_values) & set(resumed_values) if step >= first_step
    )
    if not common_steps:
        raise ValueError(
            f"No common TensorBoard steps >= {first_step} for {full} and {resumed}"
        )

    comparisons = []
    max_abs_delta = 0.0
    for step in common_steps:
        full_value = full_values[step]
        resumed_value = resumed_values[step]
        delta = abs(full_value - resumed_value)
        max_abs_delta = max(max_abs_delta, delta)
        comparisons.append(
            {
                "step": step,
                "full": full_value,
                "resumed": resumed_value,
                "delta": delta,
                "exact_equal": full_value == resumed_value,
            }
        )

    result = {
        "tag": tag,
        "full_event_dirs": full_dirs,
        "resumed_event_dirs": resumed_dirs,
        "compared_steps": [common_steps[0], common_steps[-1]],
        "num_steps": len(common_steps),
        "all_exact_equal": all(item["exact_equal"] for item in comparisons),
        "max_abs_delta": max_abs_delta,
        "comparisons": comparisons,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Resume loss comparison",
        "",
        f"Tag: `{tag}`",
        f"Steps compared: {common_steps[0]}-{common_steps[-1]} ({len(common_steps)} steps)",
        f"All exact equal: `{result['all_exact_equal']}`",
        f"Max abs delta: `{max_abs_delta}`",
        "",
        "| step | full | resumed | abs delta |",
        "|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item['step']} | {item['full']:.17g} | "
            f"{item['resumed']:.17g} | {item['delta']:.17g} |"
        )
    output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--tag", default=_LOSS_TAG)
    parser.add_argument("--first-step", type=int, default=63)
    args = parser.parse_args()
    compare_resume_loss(
        full=args.full,
        resumed=args.resumed,
        output_json=args.output_json,
        output_md=args.output_md,
        tag=args.tag,
        first_step=args.first_step,
    )


if __name__ == "__main__":
    main()
