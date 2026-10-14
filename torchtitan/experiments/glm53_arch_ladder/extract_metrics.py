# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Extract architecture-ladder metrics from a run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SCALAR_KEYS = {
    "final_train_loss": "loss_metrics/global_avg_loss",
    "tokens_per_second": "throughput(tps)",
    "peak_memory_gib": "memory/max_active(GiB)",
    "final_val_ce": "validation_metrics/loss",
    "validation_tokens_per_second": "validation_metrics/throughput(tps)",
    "validation_peak_memory_gib": "validation_metrics/memory/max_active(GiB)",
}


def _last_scalar(
    accumulator: EventAccumulator | None,
    tag: str,
) -> dict[str, Any] | None:
    if accumulator is None:
        return None
    if tag not in accumulator.Tags().get("scalars", []):
        return None
    events = accumulator.Scalars(tag)
    if not events:
        return None
    event = events[-1]
    return {"step": int(event.step), "value": float(event.value)}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _event_file(run_dir: Path) -> Path | None:
    event_files = sorted(run_dir.glob("tb/*/events.out.tfevents.*"))
    if not event_files:
        return None
    return event_files[-1]


def extract_metrics(run_dir: Path) -> dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json")
    attempt = _load_json(run_dir / "attempt.json")
    config = _load_json(run_dir / "config.json")
    event_file = _event_file(run_dir)
    accumulator = None
    if event_file is not None:
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()

    merged: dict[str, Any] = {
        "run_id": attempt.get("run_id", metrics.get("run_id", run_dir.name)),
        "attempt_id": attempt.get("attempt_id", metrics.get("attempt_id")),
        "rung": attempt.get("rung", metrics.get("rung")),
        "environment_class": attempt.get(
            "environment_class",
            metrics.get("environment_class"),
        ),
        "selected_gpu": attempt.get("selected_gpu", metrics.get("selected_gpu")),
        "seq_len": attempt.get("seq_len", metrics.get("seq_len")),
        "tokens_per_step": attempt.get(
            "tokens_per_step",
            metrics.get("tokens_per_step"),
        ),
        "token_budget": attempt.get("token_budget", metrics.get("token_budget")),
        "data_mode": attempt.get("data_mode"),
        "muon_lr": attempt.get("muon_lr"),
        "adamw_lr": attempt.get("adamw_lr"),
        "weight_decay": attempt.get("weight_decay"),
        "metrics_source": "tensorboard" if accumulator is not None else "metrics_json",
    }

    training = config.get("training", {})
    merged["resolved_seq_len"] = training.get(
        "max_context_length",
        metrics.get("resolved_seq_len"),
    )
    merged["resolved_microbatch_tokens"] = training.get(
        "num_tokens_per_microbatch_per_dp_rank",
        metrics.get("resolved_microbatch_tokens"),
    )
    merged["resolved_tokens_per_step"] = training.get(
        "num_tokens_per_train_step",
        metrics.get("resolved_tokens_per_step"),
    )
    if merged["resolved_seq_len"]:
        merged["resolved_local_batch_size"] = (
            merged["resolved_microbatch_tokens"] // merged["resolved_seq_len"]
        )
    else:
        merged["resolved_local_batch_size"] = metrics.get("resolved_local_batch_size")
    if merged["resolved_microbatch_tokens"]:
        merged["gradient_accumulation_steps"] = (
            merged["resolved_tokens_per_step"] // merged["resolved_microbatch_tokens"]
        )
    else:
        merged["gradient_accumulation_steps"] = metrics.get(
            "gradient_accumulation_steps"
        )

    train_scalar = _last_scalar(accumulator, SCALAR_KEYS["final_train_loss"])
    val_scalar = _last_scalar(accumulator, SCALAR_KEYS["final_val_ce"])
    if train_scalar is not None:
        merged["final_train_step"] = train_scalar["step"]
    else:
        merged["final_train_step"] = metrics.get("final_train_step")
    if val_scalar is not None:
        merged["final_val_step"] = val_scalar["step"]
    else:
        merged["final_val_step"] = metrics.get("final_val_step")

    for key, tag in SCALAR_KEYS.items():
        scalar = _last_scalar(accumulator, tag)
        merged[key] = scalar["value"] if scalar is not None else metrics.get(key)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    metrics = extract_metrics(args.run_dir)
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.write:
        (args.run_dir / "metrics.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
