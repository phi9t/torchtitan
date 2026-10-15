#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Validate a GLM-5.3 Flash ladder training leg artifact."""

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LOSS_TAG = "loss_metrics/global_avg_loss"


def _last_scalar_step(run_dir: Path, tag: str) -> tuple[int | None, int]:
    event_files = sorted((run_dir / "tb").glob("*/*tfevents*"))
    last_step: int | None = None
    count = 0
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = accumulator.Tags().get("scalars", [])
        if tag not in tags:
            continue
        values = accumulator.Scalars(tag)
        count += len(values)
        if values:
            step = values[-1].step
            if last_step is None or step > last_step:
                last_step = step
    return last_step, count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-step", required=True, type=int)
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--summary-json", required=True, type=Path)
    args = parser.parse_args()

    last_step, scalar_count = _last_scalar_step(args.run_dir, LOSS_TAG)
    checkpoint_dir = args.run_dir / "checkpoint" / f"step-{args.checkpoint_step}"
    summary = {
        "run_dir": str(args.run_dir),
        "loss_tag": LOSS_TAG,
        "loss_scalar_count": scalar_count,
        "last_loss_step": last_step,
        "expected_step": args.expected_step,
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_exists": checkpoint_dir.is_dir(),
    }
    ok = (
        last_step is not None
        and last_step >= args.expected_step
        and checkpoint_dir.is_dir()
    )
    summary["ok"] = ok

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
