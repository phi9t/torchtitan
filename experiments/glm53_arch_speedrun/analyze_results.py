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
import os
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


QUEUE_FILE = Path(
    os.environ.get(
        "GLM53_ARCH_CELLS_FILE",
        "experiments/glm53_arch_speedrun/queue/cells.jsonl",
    )
)
STATE_DIR = Path(
    os.environ.get(
        "GLM53_ARCH_STATE_DIR",
        "experiments/glm53_arch_speedrun/queue/state",
    )
)
DOC_PATH = Path(
    os.environ.get(
        "GLM53_ARCH_DOC_PATH",
        "docs/research/2026-10-14-glm-5.3-flash-arch-speedrun-ablations.md",
    )
)
REPORT_PATH = Path(
    os.environ.get(
        "GLM53_ARCH_REPORT_PATH",
        ".scratch/glm53-arch-speedrun/sdd/s03-report.md",
    )
)
ANALYSIS_DIR = Path(
    os.environ.get(
        "GLM53_ARCH_ANALYSIS_DIR",
        "experiments/glm53_arch_speedrun/results/analysis",
    )
)
B0_EFFECT_THRESHOLD = 2 * 0.00194
VALIDATION_RE = re.compile(r"validate step:\s*([0-9]+).*loss:\s*([0-9.]+)")


def _repo_local_path(path: Path) -> Path:
    if path.exists():
        return path
    parts = path.parts
    if "experiments" in parts:
        repo_relative = Path(*parts[parts.index("experiments") :])
        candidate = Path.cwd() / repo_relative
        if candidate.exists():
            return candidate
    return path


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
        "token_budget": cell.get("token_budget")
        or cell.get("config_overrides", {}).get("training.token_budget"),
        "tokens_per_step": cell.get("config_overrides", {}).get(
            "training.tokens_per_step"
        ),
        "side_table": bool(cell.get("side_table", False)),
        "status": state.get("status") if state else "missing_state",
        "run_id": state.get("run_id") if state else None,
        "attempt_id": state.get("attempt_id") if state else None,
        "hdfs_uri": state.get("hdfs_uri") if state else None,
        "val_ce": None,
        "val_curve": [],
        "tokens_to_target": None,
        "tokens_per_second": None,
        "max_memory_gib": None,
        "behavior_reads": {},
        "source_json": None,
    }
    if not state or state.get("status") != "done":
        return result
    attempt_dir_value = state.get("attempt_dir")
    if not attempt_dir_value:
        return result
    attempt_dir = _repo_local_path(Path(str(attempt_dir_value)))
    payloads = _walk_json(attempt_dir)
    result["val_curve"] = _validation_curve(attempt_dir / "train.log")
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
    result["behavior_reads"] = _behavior_summary(attempt_dir / "behavior_reads.jsonl")
    return result


def _validation_curve(log_path: Path) -> list[dict[str, float | int]]:
    if not log_path.exists():
        return []
    curve = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        clean = re.sub(r"\x1B\[[0-9;]*[A-Za-z]", "", line)
        match = VALIDATION_RE.search(clean)
        if match:
            curve.append(
                {
                    "step": int(match.group(1)),
                    "val_ce": float(match.group(2)),
                }
            )
    return curve


def _behavior_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = str(record.get("kind", "unknown"))
            by_kind[kind].append(record)
    summary = {}
    for kind, records in by_kind.items():
        tags = sorted({str(record.get("tag", "")) for record in records})
        latest = records[-1]
        summary[kind] = {
            "records": len(records),
            "tags": tags[:8],
            "latest": latest,
        }
    return summary


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _group_key(result: dict[str, Any]) -> tuple[str, int, str, int]:
    return (
        result["arm"],
        int(result["seq_len"]),
        result["slice"],
        int(result["token_budget"] or 0),
    )


def _summaries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[_group_key(result)].append(result)
    rows = []
    for (arm, seq_len, slice_name, token_budget), group in sorted(grouped.items()):
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
                "token_budget": token_budget,
                "side_table": any(item.get("side_table") for item in group),
                "runs": len(group),
                "done_runs": sum(1 for item in group if item["status"] == "done"),
                "val_ce_mean": val_mean,
                "val_ce_std": val_std,
                "tokens_per_second_mean": tok_mean,
                "max_memory_gib_mean": mem_mean,
            }
        )
    return _with_baseline_deltas(rows, results)


def _curve_value_at_step(result: dict[str, Any], step: int) -> float | None:
    for item in result.get("val_curve", []):
        if int(item.get("step", -1)) == step:
            return float(item["val_ce"])
    return None


def _with_baseline_deltas(
    rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baselines = {
        (row["seq_len"], row["token_budget"]): row
        for row in rows
        if row["arm"] == "b0"
    }
    full_b0_results = [
        result
        for result in results
        if result["arm"] == "b0"
        and result["status"] == "done"
        and int(result.get("token_budget") or 0) == 268_435_456
    ]
    for row in rows:
        baseline = baselines.get((row["seq_len"], row["token_budget"]))
        if baseline is not None and baseline["val_ce_mean"] is not None:
            row["b0_val_ce_mean"] = baseline["val_ce_mean"]
            row["b0_val_ce_std"] = baseline["val_ce_std"]
            row["delta_vs_b0"] = (
                None
                if row["val_ce_mean"] is None
                else row["val_ce_mean"] - baseline["val_ce_mean"]
            )
        elif row["token_budget"]:
            step = row["token_budget"] // 262_144
            values = [
                value
                for value in (
                    _curve_value_at_step(result, step)
                    for result in full_b0_results
                    if int(result["seq_len"]) == row["seq_len"]
                )
                if value is not None
            ]
            mean, std = _mean_std(values)
            row["b0_val_ce_mean"] = mean
            row["b0_val_ce_std"] = std
            row["delta_vs_b0"] = (
                None
                if row["val_ce_mean"] is None or mean is None
                else row["val_ce_mean"] - mean
            )
        else:
            row["b0_val_ce_mean"] = None
            row["b0_val_ce_std"] = None
            row["delta_vs_b0"] = None
        delta = row.get("delta_vs_b0")
        row["effect_call"] = (
            "tie"
            if delta is None or abs(delta) <= B0_EFFECT_THRESHOLD
            else ("better" if delta < 0 else "worse")
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
        f"S10 analysis generated for slice `{slice_name}`.",
        "",
        f"Effect threshold: 2 x B0 seed std = {B0_EFFECT_THRESHOLD:.5f}. Negative delta means lower validation CE than B0.",
        "",
    ]

    def fmt(value: float | None) -> str:
        return "NA" if value is None else f"{value:.6g}"

    def add_table(title: str, selected_rows: list[dict[str, Any]]) -> None:
        if not selected_rows:
            return
        lines.extend(
            [
                f"### {title}",
                "",
                "| Arm | Seq | Budget tokens | Done/runs | Val CE mean | Val CE std | Delta vs B0 | Call | Tokens/s | Memory GiB |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
            ]
        )
        for row in selected_rows:
            lines.append(
                "| {arm} | {seq_len} | {budget} | {done}/{runs} | {mean} | {std} | {delta} | {call} | {tps} | {mem} |".format(
                    arm=row["arm"],
                    seq_len=row["seq_len"],
                    budget=row["token_budget"],
                    done=row["done_runs"],
                    runs=row["runs"],
                    mean=fmt(row["val_ce_mean"]),
                    std=fmt(row["val_ce_std"]),
                    delta=fmt(row["delta_vs_b0"]),
                    call=row["effect_call"],
                    tps=fmt(row["tokens_per_second_mean"]),
                    mem=fmt(row["max_memory_gib_mean"]),
                )
            )
        lines.append("")

    full_rows = [row for row in rows if not row.get("side_table")]
    side_rows = [row for row in rows if row.get("side_table")]
    add_table("Full-budget 4a rows", full_rows)
    add_table("Reduced-budget slow-arm side table", side_rows)
    if side_rows:
        lines.append(
            "Side-table deltas use B0 validation CE from the full-budget curve at the same 33,554,432-token point."
        )
        lines.append("")
    return "\n".join(lines)


def _behavior_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "### Behavior Reads",
        "",
        "| Kind | Arm | Seq | Seed | Records | Latest read |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    any_rows = False

    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.4g}"

    for result in sorted(results, key=lambda item: item["cell_id"]):
        for kind, payload in sorted(result.get("behavior_reads", {}).items()):
            latest = payload.get("latest", {})
            if kind == "mhc":
                detail = (
                    f"comb_row_sum_mean={fmt(latest.get('comb_row_sum_mean'))}, "
                    f"comb_col_sum_mean={fmt(latest.get('comb_col_sum_mean'))}"
                )
            elif kind == "indexer":
                detail = (
                    "recall_vs_dense_topk_mean="
                    f"{fmt(latest.get('recall_vs_dense_topk_mean'))}, "
                    f"path={latest.get('attention_path')}"
                )
            elif kind == "kda":
                detail = (
                    f"gate_mean={fmt(latest.get('gate_mean'))}, "
                    f"beta_mean={fmt(latest.get('beta_mean'))}, "
                    f"impl={latest.get('impl')}"
                )
            elif kind == "mtp":
                detail = f"aux_loss={fmt(latest.get('aux_loss'))}"
            else:
                detail = json.dumps(latest, sort_keys=True)[:160]
            lines.append(
                f"| {kind} | {result['arm']} | {result['seq_len']} | {result['seed']} | {payload.get('records', 0)} | {detail} |"
            )
            any_rows = True
    if not any_rows:
        lines.append(
            "| missing | all | NA | NA | 0 | No behavior_reads.jsonl files were found. |"
        )
    return "\n".join(lines)


def _replace_doc_todo(slice_name: str, markdown: str) -> None:
    placeholder = "TODO(S03)"
    if not DOC_PATH.exists():
        return
    text = DOC_PATH.read_text(encoding="utf-8")
    table_key = f"| {slice_name} "
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(table_key):
            escaped = markdown.replace("|", r"\|").replace("\n", "<br>")
            pattern = re.compile(
                rf"^(\| {re.escape(slice_name)} [^|]* \| [^|]* \| [^|]* \| ).*( \|)$"
            )
            match = pattern.match(line)
            if match is None:
                if placeholder not in line:
                    return
                lines[index] = line.replace(placeholder, escaped)
            else:
                lines[index] = f"{match.group(1)}{escaped} |"
            DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return


def _append_report(slice_name: str, markdown: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if os.environ.get("GLM53_ARCH_REPORT_MODE") == "replace" else "a"
    with REPORT_PATH.open(mode, encoding="utf-8") as handle:
        if mode == "w":
            handle.write("# S10 report: slice 4a add-one via scheduler\n\n")
        handle.write(f"## Slice {slice_name} Analysis\n\n")
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
    markdown = markdown + "\n" + _behavior_markdown(results)

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
