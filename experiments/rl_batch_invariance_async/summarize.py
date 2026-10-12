#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summarize RL batch-invariance comparison outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


DEFAULT_METRICS = [
    "bit_wise/logprob_diff/max",
    "bit_wise/logprob_diff/mean",
    "bit_wise/ratio_tokens_different/mean",
    "rollout_reward/_mean",
    "validation_reward/_mean",
    "validation_reward/_max",
    "validation/reward/_mean",
    "validation/reward/_max",
    "perf/trainer/tokens_per_second_full_step",
    "perf/trainer/tokens_per_second_fwd_bwd",
    "perf/trainer/full_step_throughput",
    "perf/trainer/tokens_per_second",
    "generator/inter_token_latency_ms/mean",
    "generator/decode_time_ms/mean",
    "generator/queue_time_ms/mean",
    "timing/step/total",
]

_STEP_RE = re.compile(r"\b(?:Train|Validation) \| Step:\s*(?P<step>\d+)\b")
_METRIC_RE = re.compile(
    r"(?P<key>[A-Za-z0-9_./-]+):\s*(?P<value>[-+0-9.eE]+|nan|inf|-inf)"
)


@dataclass(frozen=True)
class Observation:
    step: int
    key: str
    value: float
    source: str


def _parse_float(raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        return math.nan


def parse_console_log(path: Path) -> list[Observation]:
    observations: list[Observation] = []
    for line in path.read_text(errors="replace").splitlines():
        step_match = _STEP_RE.search(line)
        if step_match is None:
            continue
        step = int(step_match.group("step"))
        for metric_match in _METRIC_RE.finditer(line):
            key = metric_match.group("key")
            if key.endswith("Step"):
                continue
            observations.append(
                Observation(
                    step=step,
                    key=key,
                    value=_parse_float(metric_match.group("value")),
                    source=str(path),
                )
            )
    return observations


def parse_tensorboard_events(root: Path) -> list[Observation]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception:
        return []

    observations: list[Observation] = []
    for event_file in root.rglob("events.out.tfevents.*"):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception:
            continue
        for key in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(key):
                observations.append(
                    Observation(
                        step=int(event.step),
                        key=key,
                        value=float(event.value),
                        source=str(event_file),
                    )
                )
    return observations


def dedupe_observations(observations: Iterable[Observation]) -> list[Observation]:
    by_step_key: dict[tuple[int, str], Observation] = {}
    for obs in observations:
        by_step_key.setdefault((obs.step, obs.key), obs)
    return list(by_step_key.values())


def collect_observations(arm_dir: Path) -> list[Observation]:
    observations = parse_tensorboard_events(arm_dir)
    for log_file in sorted(arm_dir.rglob("*.log")):
        observations.extend(parse_console_log(log_file))
    for log_file in sorted(arm_dir.glob("run_*.txt")):
        observations.extend(parse_console_log(log_file))
    return dedupe_observations(observations)


def candidate_arm_dirs(root: Path, arm: str) -> list[Path]:
    dirs = [root / arm, root / f"preflight_{arm}"]
    return [path for path in dirs if path.exists()]


def summarize_observations(
    observations: Iterable[Observation],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for obs in observations:
        if math.isfinite(obs.value):
            grouped[obs.key].append(obs)

    summary: dict[str, dict[str, float]] = {}
    for key, values in grouped.items():
        values = sorted(values, key=lambda obs: obs.step)
        nums = [obs.value for obs in values]
        first = values[0]
        last = values[-1]
        summary[key] = {
            "first_step": float(first.step),
            "first": first.value,
            "last_step": float(last.step),
            "last": last.value,
            "min": min(nums),
            "max": max(nums),
            "mean": sum(nums) / len(nums),
            "count": float(len(nums)),
        }
    return summary


def write_markdown(
    *,
    summaries: dict[str, dict[str, dict[str, float]]],
    metrics: list[str],
    output: Path,
) -> None:
    lines = [
        "# RL Batch-Invariance Summary",
        "",
        "| Metric | Arm | First | Last | Min | Max | Mean | Count |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in metrics:
        for arm in sorted(summaries):
            row = summaries[arm].get(metric)
            if row is None:
                lines.append(
                    f"| `{metric}` | `{arm}` | n/a | n/a | n/a | n/a | n/a | 0 |"
                )
                continue
            lines.append(
                "| `{metric}` | `{arm}` | {first:.6g} @ {first_step:.0f} | "
                "{last:.6g} @ {last_step:.0f} | {min:.6g} | {max:.6g} | "
                "{mean:.6g} | {count:.0f} |".format(metric=metric, arm=arm, **row)
            )
    output.write_text("\n".join(lines) + "\n")


def write_csv(
    *,
    summaries: dict[str, dict[str, dict[str, float]]],
    output: Path,
) -> None:
    with output.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "arm",
                "metric",
                "first_step",
                "first",
                "last_step",
                "last",
                "min",
                "max",
                "mean",
                "count",
            ]
        )
        for arm in sorted(summaries):
            for metric in sorted(summaries[arm]):
                row = summaries[arm][metric]
                writer.writerow(
                    [
                        arm,
                        metric,
                        int(row["first_step"]),
                        row["first"],
                        int(row["last_step"]),
                        row["last"],
                        row["min"],
                        row["max"],
                        row["mean"],
                        int(row["count"]),
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/rl_batch_invariance_async_rootfs"),
        help="Directory containing one subdirectory per experiment arm.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["no_bi", "bi"],
        help="Arm directory names under --root.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric key to include in the markdown table. May be repeated.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/rl_batch_invariance_async/summary.md"),
        help="Markdown summary path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("experiments/rl_batch_invariance_async/summary.csv"),
        help="CSV summary path.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("experiments/rl_batch_invariance_async/summary.json"),
        help="JSON summary path.",
    )
    args = parser.parse_args()

    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for arm in args.arms:
        observations: list[Observation] = []
        for arm_dir in candidate_arm_dirs(args.root, arm):
            observations.extend(collect_observations(arm_dir))
        summaries[arm] = summarize_observations(observations)

    metrics = args.metrics or DEFAULT_METRICS
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(summaries=summaries, metrics=metrics, output=args.out)
    write_csv(summaries=summaries, output=args.csv)
    args.json.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}, {args.csv}, and {args.json}")


if __name__ == "__main__":
    main()
