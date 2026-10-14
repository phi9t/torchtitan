#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""S06 fixed-bank and log-summary helpers for glm53_arch_speedrun."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

from torchtitan.experiments.falcon.bin_reader import (
    read_nanogpt_bin,
    write_nanogpt_bin,
)


ANSI_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
INIT_RE = re.compile(
    r"initialized with\s+(\d+)\s+tokens per DP rank,\s+"
    r"(\d+)\s+tokens per train step,\s+gradient accumulation steps\s+(\d+),\s+"
    r"maximum context length\s+(\d+),\s+total steps\s+(\d+)"
)
METRIC_RE = re.compile(
    r"(validate step:|step:)\s*(\d+).*loss:\s*([0-9.]+).*"
    r"memory:\s*([0-9.]+)GiB.*tps:\s*([0-9,]+)"
)


def _file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "file": path.name,
        "size": len(data),
        "md5": hashlib.md5(data).hexdigest(),
    }


def prepare_fixed_bank(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    source_bins = sorted(source_dir.glob("fineweb_train_*.bin"))
    if not source_bins:
        raise ValueError(f"no fineweb_train_*.bin files found under {source_dir}")
    num_tokens = args.num_sequences * args.seq_len + 1
    tokens = read_nanogpt_bin(source_bins[0])
    if tokens.numel() < num_tokens:
        raise ValueError(
            f"{source_bins[0]} has {tokens.numel()} tokens, need {num_tokens}"
        )
    bank = tokens[:num_tokens].clone()
    if int(bank.min()) < 0 or int(bank.max()) >= args.vocab_size:
        raise ValueError("selected bank contains token ids outside the vocab")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "fineweb_train_000000.bin"
    val_path = out_dir / "fineweb_val_000000.bin"
    write_nanogpt_bin(train_path, bank)
    write_nanogpt_bin(val_path, bank)
    records = [_file_record(train_path), _file_record(val_path)]
    with (out_dir / "MANIFEST.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "source_bin": str(source_bins[0]),
        "out_dir": str(out_dir),
        "seq_len": args.seq_len,
        "num_sequences": args.num_sequences,
        "num_tokens_with_shift": num_tokens,
        "records": records,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_train_log(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "resolved_microbatch_tokens": None,
        "resolved_tokens_per_step": None,
        "gradient_accumulation_steps": None,
        "resolved_seq_len": None,
        "resolved_steps": None,
        "train_curve": [],
        "val_curve": [],
    }
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_RE.sub("", raw_line)
        init_match = INIT_RE.search(line)
        if init_match:
            result["resolved_microbatch_tokens"] = int(init_match.group(1))
            result["resolved_tokens_per_step"] = int(init_match.group(2))
            result["gradient_accumulation_steps"] = int(init_match.group(3))
            result["resolved_seq_len"] = int(init_match.group(4))
            result["resolved_steps"] = int(init_match.group(5))
            continue
        metric_match = METRIC_RE.search(line)
        if metric_match:
            curve_name = (
                "val_curve"
                if metric_match.group(1) == "validate step:"
                else "train_curve"
            )
            result[curve_name].append(
                {
                    "step": int(metric_match.group(2)),
                    "loss": float(metric_match.group(3)),
                    "memory_gib": float(metric_match.group(4)),
                    "tokens_per_second": int(metric_match.group(5).replace(",", "")),
                }
            )
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_run(run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "train.log"
    if not log_path.exists():
        raise ValueError(f"{run_dir} does not contain train.log")
    log_summary = parse_train_log(log_path)
    metadata = _read_json(run_dir / "attempt.json")
    metrics = _read_json(run_dir / "metrics.json")
    val_curve = log_summary["val_curve"]
    train_curve = log_summary["train_curve"]
    initial_val = val_curve[0]["loss"] if val_curve else None
    final_val = val_curve[-1]["loss"] if val_curve else None
    val_drop_fraction = None
    if initial_val is not None and final_val is not None and initial_val > 0:
        val_drop_fraction = (initial_val - final_val) / initial_val
    return {
        "run_dir": str(run_dir),
        "run_id": metadata.get("run_id") or metrics.get("run_id") or run_dir.name,
        "arm": metadata.get("arm") or metrics.get("arm"),
        "mode": metadata.get("mode"),
        "claim_label": metadata.get("claim_label"),
        "environment_class": metadata.get("environment_class")
        or metrics.get("environment_class"),
        "seq_len": metadata.get("seq_len") or metrics.get("seq_len"),
        "token_budget": metadata.get("token_budget") or metrics.get("token_budget"),
        "tokens_per_step": metadata.get("tokens_per_step")
        or metrics.get("tokens_per_step"),
        "resolved_local_batch_size": metrics.get("resolved_local_batch_size"),
        "resolved_microbatch_tokens": log_summary["resolved_microbatch_tokens"],
        "resolved_tokens_per_step": log_summary["resolved_tokens_per_step"],
        "gradient_accumulation_steps": log_summary["gradient_accumulation_steps"],
        "resolved_steps": log_summary["resolved_steps"],
        "initial_val_ce": initial_val,
        "final_val_ce": final_val,
        "val_drop_fraction": val_drop_fraction,
        "final_train_loss": train_curve[-1]["loss"] if train_curve else None,
        "tokens_per_second": metrics.get("tokens_per_second")
        or (train_curve[-1]["tokens_per_second"] if train_curve else None),
        "peak_memory_gib": metrics.get("peak_memory_gib")
        or (train_curve[-1]["memory_gib"] if train_curve else None),
        "train_curve": train_curve,
        "val_curve": val_curve,
    }


def summarize_runs(args: argparse.Namespace) -> None:
    rows = [summarize_run(Path(run_dir)) for run_dir in args.run_dir]
    out = {"runs": rows}
    output = json.dumps(out, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-fixed-bank")
    prepare.add_argument("--source-dir", required=True)
    prepare.add_argument("--out-dir", required=True)
    prepare.add_argument("--seq-len", type=int, default=1024)
    prepare.add_argument("--num-sequences", type=int, default=64)
    prepare.add_argument("--vocab-size", type=int, default=50257)
    prepare.set_defaults(func=prepare_fixed_bank)

    summarize = subparsers.add_parser("summarize-runs")
    summarize.add_argument("--output")
    summarize.add_argument("run_dir", nargs="+")
    summarize.set_defaults(func=summarize_runs)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
