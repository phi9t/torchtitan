# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summarize T18 GLM DeepEP dispatch evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from torchtitan.experiments.glm53_flash_ep.parity_report import read_tensorboard_metrics


LOSS_TAG = "loss_metrics/global_avg_loss"
GRAD_TAG = "grad_norm"
TPS_TAG = "throughput(tps)"
STEP_TIME_TAG = "time_metrics/end_to_end(s)"


def _metric_series(run_dir: Path, tag: str) -> list[dict[str, float]]:
    return read_tensorboard_metrics(run_dir).get(tag, [])


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_run(run_dir: Path) -> dict[str, Any]:
    loss = _metric_series(run_dir, LOSS_TAG)
    grad = _metric_series(run_dir, GRAD_TAG)
    tps = _metric_series(run_dir, TPS_TAG)
    step_time = _metric_series(run_dir, STEP_TIME_TAG)
    steady_tps = [event["value"] for event in tps if event["step"] >= 2]
    steady_ms = [event["value"] * 1000.0 for event in step_time if event["step"] >= 2]
    return {
        "run_dir": str(run_dir),
        "steps": sorted({int(row["step"]) for series in (loss, grad) for row in series}),
        "final_loss": loss[-1]["value"] if loss else None,
        "final_grad_norm": grad[-1]["value"] if grad else None,
        "mean_ms_per_step_steps_2_plus": _mean(steady_ms),
        "mean_tokens_per_second_steps_2_plus": _mean(steady_tps),
        "loss_series": loss,
        "grad_norm_series": grad,
        "tokens_per_second_series": tps,
        "step_time_series": step_time,
    }


def summarize_microbench(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path)}
    data = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    for rank_record in data.get("world", []):
        for bench in rank_record.get("benchmarks", []):
            rows.append(
                {
                    "rank": rank_record.get("rank"),
                    "name": bench.get("name"),
                    "mean_dispatch_ms": bench.get("mean_dispatch_ms"),
                    "mean_combine_ms": bench.get("mean_combine_ms"),
                    "mean_roundtrip_ms": bench.get("mean_roundtrip_ms"),
                    "max_memory_reserved_gib": bench.get("max_memory_reserved_gib"),
                }
            )
    return {
        "available": True,
        "path": str(path),
        "shape": data.get("world", [{}])[0].get("shape") if data.get("world") else None,
        "rows": rows,
        "by_name": {
            name: {
                "mean_dispatch_ms": _mean(
                    [float(row["mean_dispatch_ms"]) for row in rows if row["name"] == name]
                ),
                "mean_combine_ms": _mean(
                    [float(row["mean_combine_ms"]) for row in rows if row["name"] == name]
                ),
                "mean_roundtrip_ms": _mean(
                    [float(row["mean_roundtrip_ms"]) for row in rows if row["name"] == name]
                ),
            }
            for name in sorted({str(row["name"]) for row in rows})
        },
    }


def build_summary(result_root: Path) -> dict[str, Any]:
    alltoall = summarize_run(result_root / "train" / "alltoall_first8")
    deepep = summarize_run(result_root / "train" / "deepep_first8")
    return {
        "schema_version": 1,
        "result_root": str(result_root),
        "alltoall_first8": alltoall,
        "deepep_first8": deepep,
        "deltas": {
            "final_loss": _optional_delta(alltoall["final_loss"], deepep["final_loss"]),
            "final_grad_norm": _optional_delta(
                alltoall["final_grad_norm"], deepep["final_grad_norm"]
            ),
            "mean_ms_per_step_steps_2_plus": _optional_delta(
                alltoall["mean_ms_per_step_steps_2_plus"],
                deepep["mean_ms_per_step_steps_2_plus"],
            ),
            "mean_tokens_per_second_steps_2_plus": _optional_delta(
                alltoall["mean_tokens_per_second_steps_2_plus"],
                deepep["mean_tokens_per_second_steps_2_plus"],
            ),
        },
        "microbench": summarize_microbench(
            result_root / "microbench" / "dispatch_microbench.json"
        ),
    }


def _optional_delta(reference: Any, candidate: Any) -> float | None:
    if reference is None or candidate is None:
        return None
    return float(candidate) - float(reference)


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# T18 GLM DeepEP dispatch summary",
        "",
        f"Result root: `{summary['result_root']}`",
        "",
        "| run | final loss | final grad norm | mean ms/step step>=2 | mean tokens/s step>=2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in ("alltoall_first8", "deepep_first8"):
        row = summary[key]
        lines.append(
            "| "
            f"{key} | {_fmt(row['final_loss'])} | {_fmt(row['final_grad_norm'])} | "
            f"{_fmt(row['mean_ms_per_step_steps_2_plus'])} | "
            f"{_fmt(row['mean_tokens_per_second_steps_2_plus'])} |"
        )
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            json.dumps(summary["deltas"], indent=2, sort_keys=True),
            "",
            "## Microbench",
            "",
            json.dumps(summary["microbench"].get("by_name"), indent=2, sort_keys=True),
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def _fmt(value: Any) -> str:
    if value is None:
        return "missing"
    return f"{float(value):.10g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    summary = build_summary(args.result_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(summary, args.output_md)


if __name__ == "__main__":
    main()
