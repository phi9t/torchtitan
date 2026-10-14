#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Analyze completed GLM-5.3 architecture speedrun slices.

The scheduler invokes this inside the rootfs after a slice completes. It is
written to tolerate the S01 runner's final metric filename varying: any JSON
file under the attempt directory may provide val CE and throughput fields.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


QUEUE_FILE = Path("experiments/glm53_arch_speedrun/queue/cells.jsonl")
STATE_DIR = Path("experiments/glm53_arch_speedrun/queue/state")
RESULTS_DIR = Path("experiments/glm53_arch_speedrun/results")
DOC_PATH = Path("docs/research/2026-10-14-glm-5.3-flash-arch-speedrun-ablations.md")
REPORT_PATH = Path(".scratch/glm53-arch-speedrun/sdd/s03-report.md")
ANALYSIS_DIR = Path("experiments/glm53_arch_speedrun/results/analysis")


def _load_cells() -> dict[str, dict[str, Any]]:
    cells = {}
    with QUEUE_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cell = json.loads(line)
                cells[cell["cell_id"]] = cell
    return cells


def _load_state(cell_id: str) -> dict[str, Any] | None:
    path = STATE_DIR / f"{cell_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_json(path: Path) -> list[dict[str, Any]]:
    records = []
    for json_path in sorted(path.rglob("*.json")):
        if json_path.name == "cell.json":
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payload["_source_json"] = str(json_path)
            records.append(payload)
    return records


def _first_number(
    payloads: list[dict[str, Any]], names: tuple[str, ...]
) -> float | None:
    stack: list[Any] = list(payloads)
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for name in names:
                value = item.get(name)
                if isinstance(value, int | float) and math.isfinite(float(value)):
                    return float(value)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return None


def _cell_result(cell: dict[str, Any], state: dict[str, Any] | None) -> dict[str, Any]:
    result = {
        "cell_id": cell["cell_id"],
        "slice": cell["slice"],
        "arm": cell["arm"],
        "seed": cell["seed"],
        "seq_len": cell["seq_len"],
        "status": state.get("status") if state else "missing_state",
        "run_id": state.get("run_id") if state else None,
        "attempt_id": state.get("attempt_id") if state else None,
        "val_ce": None,
        "tokens_to_target": None,
        "tokens_per_second": None,
        "max_memory_gib": None,
        "source_json": None,
    }
    if not state or state.get("status") != "done":
        return result
    attempt_dir_value = state.get("attempt_dir")
    if not attempt_dir_value:
        return result
    attempt_dir = Path(str(attempt_dir_value))
    payloads = _walk_json(attempt_dir)
    result["val_ce"] = _first_number(
        payloads,
        ("val_ce", "validation_ce", "val_cross_entropy", "final_val_ce"),
    )
    result["tokens_to_target"] = _first_number(
        payloads,
        ("tokens_to_target", "tokens_to_3_28", "tokens_to_calibrated_target"),
    )
    result["tokens_per_second"] = _first_number(
        payloads,
        ("tokens_per_second", "tokens_per_sec", "tokens_s"),
    )
    result["max_memory_gib"] = _first_number(
        payloads,
        ("max_memory_gib", "peak_memory_gib", "gpu_memory_gib"),
    )
    for payload in payloads:
        if result["val_ce"] is not None and any(
            key in payload for key in ("val_ce", "validation_ce", "final_val_ce")
        ):
            result["source_json"] = payload.get("_source_json")
            break
    return result


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _group_key(result: dict[str, Any]) -> tuple[str, int, str]:
    return result["arm"], int(result["seq_len"]), result["slice"]


def _summaries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[_group_key(result)].append(result)
    rows = []
    for (arm, seq_len, slice_name), group in sorted(grouped.items()):
        val_values = [
            float(item["val_ce"])
            for item in group
            if isinstance(item.get("val_ce"), int | float)
        ]
        tok_values = [
            float(item["tokens_per_second"])
            for item in group
            if isinstance(item.get("tokens_per_second"), int | float)
        ]
        mem_values = [
            float(item["max_memory_gib"])
            for item in group
            if isinstance(item.get("max_memory_gib"), int | float)
        ]
        val_mean, val_std = _mean_std(val_values)
        tok_mean, _ = _mean_std(tok_values)
        mem_mean, _ = _mean_std(mem_values)
        rows.append(
            {
                "slice": slice_name,
                "arm": arm,
                "seq_len": seq_len,
                "runs": len(group),
                "done_runs": sum(1 for item in group if item["status"] == "done"),
                "val_ce_mean": val_mean,
                "val_ce_std": val_std,
                "tokens_per_second_mean": tok_mean,
                "max_memory_gib_mean": mem_mean,
            }
        )
    return rows


def _bootstrap_delta_ci(
    baseline: list[float],
    treatment: list[float],
    *,
    rounds: int = 2000,
) -> tuple[float | None, float | None, float | None]:
    if not baseline or not treatment:
        return None, None, None
    rng = random.Random(5303)
    observed = statistics.fmean(treatment) - statistics.fmean(baseline)
    samples = []
    for _ in range(rounds):
        b = [baseline[rng.randrange(len(baseline))] for _ in baseline]
        t = [treatment[rng.randrange(len(treatment))] for _ in treatment]
        samples.append(statistics.fmean(t) - statistics.fmean(b))
    samples.sort()
    lo = samples[int(0.025 * (rounds - 1))]
    hi = samples[int(0.975 * (rounds - 1))]
    return observed, lo, hi


def _slice_markdown(slice_name: str, summaries: list[dict[str, Any]]) -> str:
    rows = [row for row in summaries if row["slice"] == slice_name]
    if not rows:
        return f"No completed `{slice_name}` result rows were available."
    lines = [
        f"S03 analysis generated for slice `{slice_name}`.",
        "",
        "| Arm | Seq | Done/runs | Val CE mean | Val CE std | Tokens/s | Memory GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:

        def fmt(value: float | None) -> str:
            return "NA" if value is None else f"{value:.6g}"

        lines.append(
            "| {arm} | {seq_len} | {done}/{runs} | {mean} | {std} | {tps} | {mem} |".format(
                arm=row["arm"],
                seq_len=row["seq_len"],
                done=row["done_runs"],
                runs=row["runs"],
                mean=fmt(row["val_ce_mean"]),
                std=fmt(row["val_ce_std"]),
                tps=fmt(row["tokens_per_second_mean"]),
                mem=fmt(row["max_memory_gib_mean"]),
            )
        )
    return "\n".join(lines)


def _replace_doc_todo(slice_name: str, markdown: str) -> None:
    placeholder = "TODO(S03)"
    if not DOC_PATH.exists():
        return
    text = DOC_PATH.read_text(encoding="utf-8")
    if placeholder not in text:
        return
    table_key = f"| {slice_name} "
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(table_key) and placeholder in line:
            escaped = markdown.replace("\n", "<br>")
            lines[index] = line.replace(placeholder, escaped)
            DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return


def _append_report(slice_name: str, markdown: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## Slice {slice_name} Analysis\n\n")
        handle.write(markdown)
        handle.write("\n")


def analyze(slice_name: str) -> dict[str, Any]:
    cells = _load_cells()
    slice_cells = {
        cell_id: cell for cell_id, cell in cells.items() if cell["slice"] == slice_name
    }
    results = [
        _cell_result(cell, _load_state(cell_id))
        for cell_id, cell in sorted(slice_cells.items())
    ]
    summaries = _summaries(results)
    markdown = _slice_markdown(slice_name, summaries)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "slice": slice_name,
        "results": results,
        "summaries": summaries,
        "markdown": markdown,
    }
    output_path = ANALYSIS_DIR / f"{slice_name}.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ANALYSIS_DIR / f"{slice_name}.md").write_text(markdown + "\n", encoding="utf-8")
    _replace_doc_todo(slice_name, markdown)
    _append_report(slice_name, markdown)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slice", required=True, choices=["4a", "4b", "4c", "4d", "c1"]
    )
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()

    if args.check_inputs:
        missing = [str(path) for path in (QUEUE_FILE, STATE_DIR) if not path.exists()]
        if missing:
            raise SystemExit(f"missing analysis input(s): {', '.join(missing)}")
    output = analyze(args.slice)
    print(
        json.dumps(
            {
                "slice": output["slice"],
                "rows": len(output["results"]),
                "summaries": len(output["summaries"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
