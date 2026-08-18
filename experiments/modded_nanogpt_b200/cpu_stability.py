#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Repeat the CPU-only NanoGPT smoke and aggregate stability telemetry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.modded_nanogpt_b200 import cli_guard, cpu_smoke


SCHEMA_VERSION = 1
WRAPPER = "experiments/modded_nanogpt_b200/run_cpu_stability.sh"
DEFAULT_RESULT_ROOT = Path("experiments/modded_nanogpt_b200/results")

SmokeRunner = Callable[..., dict[str, Any]]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _artifact_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _process_telemetry(
    *,
    start_wall: float,
    start_process: float,
    end_wall: float,
    end_process: float,
) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "started_at_unix": start_wall,
        "ended_at_unix": end_wall,
        "elapsed_seconds": end_wall - start_wall,
        "process_cpu_seconds": end_process - start_process,
        "process": {
            "max_rss_kib": usage.ru_maxrss,
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
        },
    }


def _classification(run_id: str) -> dict[str, Any]:
    return {
        "lane": "CPU",
        "mode": "cpu_stability",
        "arm": "CPU0",
        "claim_label": "CPU-only harness stability",
        "evidence_tier": "cpu-stability",
        "run_id": run_id,
        "attempt_id": f"{run_id}_stability",
        "environment_class": "torchtitan-rootfs-cpu",
        "claim_eligible": False,
    }


def run_cpu_stability(
    *,
    result_dir: Path,
    run_id: str,
    repeats: int,
    steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    mlp_dim: int,
    seed: int,
    smoke_runner: SmokeRunner = cpu_smoke.run_cpu_smoke,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    result_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir = result_dir / "telemetry"
    telemetry_dir.mkdir(exist_ok=True)
    run_log = result_dir / "run.log"
    classification = _classification(run_id)
    attempts: list[dict[str, Any]] = []
    completed_repeats = 0
    failed_repeats = 0
    validation_losses: list[float] = []
    started_at = time.time()
    process_started_at = time.process_time()

    with run_log.open("w") as log:
        log.write(
            f"cpu_stability run_id={run_id} repeats={repeats} steps={steps}\n"
        )
        for attempt_index in range(1, repeats + 1):
            attempt_dir = result_dir / f"attempt_{attempt_index:03d}"
            attempt_run_id = f"{run_id}_attempt_{attempt_index:03d}"
            attempt_seed = seed + attempt_index - 1
            start_wall = time.time()
            start_process = time.process_time()
            log.write(f"cpu_stability repeat={attempt_index}/{repeats} start\n")
            try:
                result = smoke_runner(
                    result_dir=attempt_dir,
                    run_id=attempt_run_id,
                    steps=steps,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    vocab_size=vocab_size,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    mlp_dim=mlp_dim,
                    seed=attempt_seed,
                )
            except Exception as exc:
                failed_repeats += 1
                end_wall = time.time()
                end_process = time.process_time()
                failure = {
                    "schema_version": SCHEMA_VERSION,
                    "ok": False,
                    "attempt_index": attempt_index,
                    "run_id": attempt_run_id,
                    "result_dir": str(attempt_dir),
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    "telemetry": _process_telemetry(
                        start_wall=start_wall,
                        start_process=start_process,
                        end_wall=end_wall,
                        end_process=end_process,
                    ),
                }
                failure["telemetry"]["artifact_size_bytes"] = _artifact_size_bytes(
                    attempt_dir
                )
                _write_json(attempt_dir / "failure.json", failure)
                attempts.append(failure)
                log.write(
                    f"cpu_stability repeat={attempt_index}/{repeats} failed "
                    f"error_type={type(exc).__name__} message={exc}\n"
                )
                break

            end_wall = time.time()
            end_process = time.process_time()
            telemetry = _process_telemetry(
                start_wall=start_wall,
                start_process=start_process,
                end_wall=end_wall,
                end_process=end_process,
            )
            telemetry["artifact_size_bytes"] = _artifact_size_bytes(attempt_dir)
            validation_loss = result.get("validation_loss")
            if isinstance(validation_loss, int | float):
                validation_losses.append(float(validation_loss))
            attempt = {
                "schema_version": SCHEMA_VERSION,
                "ok": True,
                "attempt_index": attempt_index,
                "run_id": attempt_run_id,
                "result_dir": str(attempt_dir),
                "seed": attempt_seed,
                "validation_loss": validation_loss,
                "elapsed_seconds": result.get("elapsed_seconds"),
                "telemetry": telemetry,
            }
            attempts.append(attempt)
            completed_repeats += 1
            log.write(
                f"cpu_stability repeat={attempt_index}/{repeats} ok "
                f"validation_loss={validation_loss}\n"
            )

        ended_at = time.time()
        process_ended_at = time.process_time()
        log.write(
            f"cpu_stability completed={completed_repeats} failed={failed_repeats} "
            f"elapsed_seconds={ended_at - started_at:.6f}\n"
        )

    ok = completed_repeats == repeats and failed_repeats == 0
    aggregate_telemetry = _process_telemetry(
        start_wall=started_at,
        start_process=process_started_at,
        end_wall=ended_at,
        end_process=process_ended_at,
    )
    aggregate_telemetry["artifact_size_bytes"] = _artifact_size_bytes(result_dir)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "classification": classification,
        "training_launched": completed_repeats > 0,
        "final_validation_reached": ok,
        "included_in_baseline_stats": False,
        "repeats": repeats,
        "completed_repeats": completed_repeats,
        "failed_repeats": failed_repeats,
        "attempts": attempts,
        "validation_loss": {
            "values": validation_losses,
            "min": min(validation_losses) if validation_losses else None,
            "max": max(validation_losses) if validation_losses else None,
            "mean": (
                sum(validation_losses) / len(validation_losses)
                if validation_losses
                else None
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "rootfs": os.environ.get("TORCHTITAN_IN_ROOTFS") == "1",
        },
        "telemetry": aggregate_telemetry,
        "teardown": {"ok": True, "reason": "no persistent child processes"},
        "blocker": None if ok else "cpu stability repeat failed",
    }
    _write_json(result_dir / "stability_summary.json", summary)
    _write_json(
        telemetry_dir / "stability_status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "ok": ok,
            "run_id": run_id,
            "completed_repeats": completed_repeats,
            "failed_repeats": failed_repeats,
            "teardown": summary["teardown"],
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--mlp-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> int:
    guard = cli_guard.guard_rootfs_cli(WRAPPER)
    if guard is not None:
        return guard
    args = parse_args()
    run_id = args.run_id or time.strftime(
        "nanogpt_cpu_stability_%Y%m%dT%H%M%SZ", time.gmtime()
    )
    result_dir = args.result_dir or DEFAULT_RESULT_ROOT / run_id
    summary = run_cpu_stability(
        result_dir=result_dir,
        run_id=run_id,
        repeats=args.repeats,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_dim=args.mlp_dim,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "ok": summary["ok"],
                "summary": str(result_dir / "stability_summary.json"),
                "completed_repeats": summary["completed_repeats"],
                "failed_repeats": summary["failed_repeats"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
