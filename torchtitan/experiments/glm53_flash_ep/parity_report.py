# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summarize GLM EP truncated-depth parity evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint import FileSystemReader, load


LOSS_TAG = "loss_metrics/global_avg_loss"
GRAD_NORM_TAG = "grad_norm"
THROUGHPUT_TAG = "throughput(tps)"
STEP_TIME_TAG = "time_metrics/end_to_end(s)"
REPORT_TAGS = (LOSS_TAG, GRAD_NORM_TAG, THROUGHPUT_TAG, STEP_TIME_TAG)


def build_parity_report(plain_dir: Path, ep_dir: Path) -> dict[str, Any]:
    """Build a JSON-serializable parity summary from two run directories."""

    plain_metrics = read_tensorboard_metrics(plain_dir)
    ep_metrics = read_tensorboard_metrics(ep_dir)
    plain_memory = read_memory_summary(plain_dir)
    ep_memory = read_memory_summary(ep_dir)
    checkpoint_comparison = compare_latest_lora_checkpoints(plain_dir, ep_dir)
    gradient_probe_comparison = compare_grad_probe_dirs(
        plain_dir / "grad_probe",
        ep_dir / "grad_probe",
    )
    step_rows = compare_metric_steps(plain_metrics, ep_metrics)
    scalar_deltas = summarize_scalar_deltas(step_rows)

    return {
        "schema_version": 1,
        "plain_dir": str(plain_dir),
        "ep_dir": str(ep_dir),
        "metric_tags": list(REPORT_TAGS),
        "plain": {
            "metrics": summarize_metrics(plain_metrics),
            "memory": plain_memory,
        },
        "ep": {
            "metrics": summarize_metrics(ep_metrics),
            "memory": ep_memory,
        },
        "step_comparison": step_rows,
        "scalar_deltas": scalar_deltas,
        "checkpoint_parameter_comparison": checkpoint_comparison,
        "gradient_probe_comparison": gradient_probe_comparison,
        "claim_hint": claim_hint(scalar_deltas, gradient_probe_comparison),
        "numerical_note": (
            "Bitwise identity is not expected for the EP adapter because the "
            "routed expert path uses the common grouped expert implementation "
            "and token-dispatch ordering rather than the current GLM local "
            "per-expert loop."
        ),
    }


def read_tensorboard_metrics(run_dir: Path) -> dict[str, list[dict[str, float]]]:
    """Read selected scalar metrics from TensorBoard event files."""

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:
        raise RuntimeError("tensorboard is required to read parity metrics") from exc

    event_files = sorted((run_dir / "tb").glob("**/*tfevents*"))
    metrics: dict[str, dict[int, dict[str, float]]] = {tag: {} for tag in REPORT_TAGS}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", ()))
        for tag in REPORT_TAGS:
            if tag not in available:
                continue
            for scalar in accumulator.Scalars(tag):
                previous = metrics[tag].get(scalar.step)
                if previous is None or scalar.wall_time >= previous["wall_time"]:
                    metrics[tag][scalar.step] = {
                        "step": int(scalar.step),
                        "value": float(scalar.value),
                        "wall_time": float(scalar.wall_time),
                    }
    return {
        tag: [by_step[step] for step in sorted(by_step)]
        for tag, by_step in metrics.items()
    }


def summarize_metrics(
    metrics: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    """Return compact first/final metric values for one run."""

    summary: dict[str, Any] = {}
    for tag, events in metrics.items():
        summary[tag] = {
            "num_points": len(events),
            "first": events[0] if events else None,
            "final": events[-1] if events else None,
            "series": events,
        }
    return summary


def compare_metric_steps(
    plain_metrics: dict[str, list[dict[str, float]]],
    ep_metrics: dict[str, list[dict[str, float]]],
) -> list[dict[str, Any]]:
    """Align scalar metrics by step and compute EP-minus-plain deltas."""

    steps = sorted(
        {
            int(event["step"])
            for metrics in (plain_metrics, ep_metrics)
            for events in metrics.values()
            for event in events
        }
    )
    plain_by_tag = _metrics_by_step(plain_metrics)
    ep_by_tag = _metrics_by_step(ep_metrics)
    rows: list[dict[str, Any]] = []
    for step in steps:
        row: dict[str, Any] = {"step": step}
        for tag in REPORT_TAGS:
            plain_value = plain_by_tag.get(tag, {}).get(step)
            ep_value = ep_by_tag.get(tag, {}).get(step)
            key = _metric_key(tag)
            row[f"plain_{key}"] = plain_value
            row[f"ep_{key}"] = ep_value
            row[f"{key}_delta"] = (
                None
                if plain_value is None or ep_value is None
                else ep_value - plain_value
            )
        rows.append(row)
    return rows


def summarize_scalar_deltas(step_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize max absolute scalar deltas across aligned steps."""

    output: dict[str, Any] = {}
    for key in ("loss", "grad_norm", "tokens_per_second", "step_time_ms"):
        delta_key = f"{key}_delta"
        deltas = [
            float(row[delta_key])
            for row in step_rows
            if row.get(delta_key) is not None and math.isfinite(float(row[delta_key]))
        ]
        output[f"max_abs_{key}_delta"] = (
            max(abs(delta) for delta in deltas) if deltas else None
        )
        output[f"final_{key}_delta"] = deltas[-1] if deltas else None
    return output


def read_memory_summary(run_dir: Path) -> dict[str, Any]:
    """Read rank-local lean memory evidence, when present."""

    records: list[dict[str, Any]] = []
    for path in sorted((run_dir / "lean_memory").glob("rank*.json")):
        with path.open() as handle:
            record = json.load(handle)
        record["_path"] = str(path)
        records.append(record)

    by_rank: dict[int, dict[str, Any]] = {}
    for record in records:
        rank = int(record.get("rank", -1))
        if rank < 0:
            continue
        rank_summary = by_rank.setdefault(
            rank,
            {
                "rank": rank,
                "local_rank": record.get("local_rank"),
                "device_name": record.get("device_name"),
                "max_reserved_gib": None,
                "max_allocated_gib": None,
                "max_record": None,
            },
        )
        reserved = _optional_float(record.get("max_reserved_gib"))
        allocated = _optional_float(record.get("max_allocated_gib"))
        if reserved is not None and (
            rank_summary["max_reserved_gib"] is None
            or reserved > rank_summary["max_reserved_gib"]
        ):
            rank_summary["max_reserved_gib"] = reserved
            rank_summary["max_record"] = record
        if allocated is not None and (
            rank_summary["max_allocated_gib"] is None
            or allocated > rank_summary["max_allocated_gib"]
        ):
            rank_summary["max_allocated_gib"] = allocated

    return {
        "num_records": len(records),
        "records": records,
        "peak_by_rank": [by_rank[rank] for rank in sorted(by_rank)],
    }


def compare_latest_lora_checkpoints(
    plain_dir: Path,
    ep_dir: Path,
) -> dict[str, Any]:
    """Compare LoRA tensors from the latest DCP checkpoint in each run."""

    plain_checkpoint = latest_checkpoint_dir(plain_dir)
    ep_checkpoint = latest_checkpoint_dir(ep_dir)
    if plain_checkpoint is None or ep_checkpoint is None:
        return {
            "available": False,
            "reason": "latest checkpoint missing",
            "plain_checkpoint": None
            if plain_checkpoint is None
            else str(plain_checkpoint),
            "ep_checkpoint": None if ep_checkpoint is None else str(ep_checkpoint),
        }

    plain = read_lora_tensors(plain_checkpoint)
    ep = read_lora_tensors(ep_checkpoint)
    rows = compare_tensor_maps(plain, ep)
    return {
        "available": True,
        "plain_checkpoint": str(plain_checkpoint),
        "ep_checkpoint": str(ep_checkpoint),
        "num_tensors": len(rows),
        "num_differing_tensors": sum(1 for row in rows if row["max_abs_delta"] != 0),
        "max_abs_parameter_delta": max(
            (row["max_abs_delta"] for row in rows if row["max_abs_delta"] is not None),
            default=None,
        ),
        "tensors": sorted(
            rows,
            key=lambda row: (
                -1.0 if row["max_abs_delta"] is None else -row["max_abs_delta"],
                row["name"],
            ),
        )[:40],
    }


def latest_checkpoint_dir(run_dir: Path) -> Path | None:
    """Return the highest-numbered checkpoint/step-* directory."""

    checkpoint_root = run_dir / "checkpoint"
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.glob("step-*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"step-(\d+)", path.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def read_lora_tensors(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load all LoRA tensors from a TorchTitan DCP checkpoint."""

    reader = FileSystemReader(checkpoint_dir)
    metadata = reader.read_metadata()
    state: dict[str, torch.Tensor] = {}
    for key, value in metadata.state_dict_metadata.items():
        if ".lora_a" not in key and ".lora_b" not in key:
            continue
        size = getattr(value, "size", None)
        properties = getattr(value, "properties", None)
        dtype = getattr(properties, "dtype", torch.bfloat16)
        if size is None:
            continue
        state[key] = torch.empty(tuple(size), dtype=dtype)
    if not state:
        return {}
    load(state, checkpoint_id=str(checkpoint_dir), no_dist=True)
    return {key: tensor.cpu().float() for key, tensor in state.items()}


def compare_tensor_maps(
    plain: dict[str, torch.Tensor],
    ep: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """Compare tensors by key."""

    rows: list[dict[str, Any]] = []
    for name in sorted(set(plain) | set(ep)):
        left = plain.get(name)
        right = ep.get(name)
        if left is None or right is None:
            rows.append(
                {
                    "name": name,
                    "plain_present": left is not None,
                    "ep_present": right is not None,
                    "shape": None,
                    "max_abs_delta": None,
                    "reason": "missing tensor",
                }
            )
            continue
        if tuple(left.shape) != tuple(right.shape):
            rows.append(
                {
                    "name": name,
                    "plain_present": True,
                    "ep_present": True,
                    "shape": [list(left.shape), list(right.shape)],
                    "max_abs_delta": None,
                    "reason": "shape mismatch",
                }
            )
            continue
        delta = 0.0 if left.numel() == 0 else float((left - right).abs().max().item())
        rows.append(
            {
                "name": name,
                "plain_present": True,
                "ep_present": True,
                "shape": list(left.shape),
                "max_abs_delta": delta,
                "reason": None,
            }
        )
    return rows


def compare_grad_probe_dirs(plain_dir: Path, ep_dir: Path) -> dict[str, Any]:
    """Compare raw gradient probe snapshots, if both runs produced them."""

    if not list(plain_dir.glob("rank_*.pt")) or not list(ep_dir.glob("rank_*.pt")):
        return {
            "available": False,
            "reason": "grad_probe rank snapshots not found",
            "plain_dir": str(plain_dir),
            "ep_dir": str(ep_dir),
        }

    from torchtitan.experiments.glm53_flash_ep.grad_probe import compare_probe_dirs

    comparison = compare_probe_dirs(plain_dir, ep_dir)
    comparison["available"] = True
    comparison["plain_dir"] = str(plain_dir)
    comparison["ep_dir"] = str(ep_dir)
    return comparison


def claim_hint(
    scalar_deltas: dict[str, Any],
    gradient_probe_comparison: dict[str, Any],
) -> str:
    """Return an evidence label hint for the report writer."""

    if (
        scalar_deltas.get("max_abs_loss_delta") == 0
        and scalar_deltas.get("max_abs_grad_norm_delta") == 0
        and gradient_probe_comparison.get("available")
        and gradient_probe_comparison.get("max_abs_gradient_delta") == 0
    ):
        return "deterministic_comparison"
    return "smoke_plus_bounded_numerical_delta"


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    """Write a compact Markdown rendering of the parity report."""

    lines = [
        "# GLM EP truncated-depth parity report",
        "",
        f"Plain: `{report['plain_dir']}`",
        f"EP: `{report['ep_dir']}`",
        "",
        f"- claim hint: `{report['claim_hint']}`",
        f"- max |loss delta|: {_fmt(report['scalar_deltas'].get('max_abs_loss_delta'))}",
        f"- max |grad_norm delta|: {_fmt(report['scalar_deltas'].get('max_abs_grad_norm_delta'))}",
        f"- max |tokens/s delta|: {_fmt(report['scalar_deltas'].get('max_abs_tokens_per_second_delta'))}",
        f"- max |step time delta ms|: {_fmt(report['scalar_deltas'].get('max_abs_step_time_ms_delta'))}",
        "",
        "## Steps",
        "",
        "| step | plain loss | ep loss | loss delta | plain grad | ep grad | "
        "grad delta | plain ms | ep ms | plain tok/s | ep tok/s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["step_comparison"]:
        lines.append(
            "| "
            f"{row['step']} | "
            f"{_fmt(row.get('plain_loss'))} | "
            f"{_fmt(row.get('ep_loss'))} | "
            f"{_fmt(row.get('loss_delta'))} | "
            f"{_fmt(row.get('plain_grad_norm'))} | "
            f"{_fmt(row.get('ep_grad_norm'))} | "
            f"{_fmt(row.get('grad_norm_delta'))} | "
            f"{_fmt(row.get('plain_step_time_ms'))} | "
            f"{_fmt(row.get('ep_step_time_ms'))} | "
            f"{_fmt(row.get('plain_tokens_per_second'))} | "
            f"{_fmt(row.get('ep_tokens_per_second'))} |"
        )
    lines.extend(["", "## Peak Memory", ""])
    for label in ("plain", "ep"):
        peaks = report[label]["memory"]["peak_by_rank"]
        lines.extend(
            [
                f"### {label}",
                "",
                "| rank | local rank | max reserved GiB | max allocated GiB | device |",
                "| ---: | ---: | ---: | ---: | --- |",
            ]
        )
        if not peaks:
            lines.append("| missing | missing | missing | missing | missing |")
        for peak in peaks:
            lines.append(
                "| "
                f"{peak['rank']} | "
                f"{peak.get('local_rank')} | "
                f"{_fmt(peak.get('max_reserved_gib'))} | "
                f"{_fmt(peak.get('max_allocated_gib'))} | "
                f"{peak.get('device_name')} |"
            )
        lines.append("")

    checkpoint = report["checkpoint_parameter_comparison"]
    lines.extend(
        [
            "## Parameter Deltas",
            "",
            f"- available: `{checkpoint.get('available')}`",
            f"- max |parameter delta|: {_fmt(checkpoint.get('max_abs_parameter_delta'))}",
            f"- plain checkpoint: `{checkpoint.get('plain_checkpoint')}`",
            f"- ep checkpoint: `{checkpoint.get('ep_checkpoint')}`",
            "",
        ]
    )
    gradient_probe = report["gradient_probe_comparison"]
    lines.extend(
        [
            "## Gradient Deltas",
            "",
            f"- available: `{gradient_probe.get('available')}`",
            f"- reason: `{gradient_probe.get('reason')}`",
            f"- max |gradient delta|: {_fmt(gradient_probe.get('max_abs_gradient_delta'))}",
            "",
            "## Note",
            "",
            report["numerical_note"],
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def _metrics_by_step(
    metrics: dict[str, list[dict[str, float]]],
) -> dict[str, dict[int, float]]:
    return {
        tag: {
            int(event["step"]): _metric_value(tag, event["value"]) for event in events
        }
        for tag, events in metrics.items()
    }


def _metric_value(tag: str, value: float) -> float:
    if tag == STEP_TIME_TAG:
        return value * 1000.0
    return value


def _metric_key(tag: str) -> str:
    if tag == LOSS_TAG:
        return "loss"
    if tag == GRAD_NORM_TAG:
        return "grad_norm"
    if tag == THROUGHPUT_TAG:
        return "tokens_per_second"
    if tag == STEP_TIME_TAG:
        return "step_time_ms"
    raise ValueError(f"unsupported metric tag: {tag}")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(output):
        return None
    return output


def _fmt(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return "missing"
    return f"{number:.10g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain", required=True, type=Path)
    parser.add_argument("--ep", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    report = build_parity_report(args.plain, args.ep)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report, args.output_md)


if __name__ == "__main__":
    main()
