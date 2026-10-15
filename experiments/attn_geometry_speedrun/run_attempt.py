#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launch one A01 attention-geometry attempt and record local artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANT = REPO_ROOT / "experiments/attn_geometry_speedrun/variant/train_gpt.py"
RESULT_ROOT = REPO_ROOT / "experiments/attn_geometry_speedrun/results"
SUPPORTED_ARMS = ("anchor", "dot", "cosine", "gauss")


@dataclass(frozen=True)
class AttemptConfig:
    arm: str
    mode: str
    claim_label: str
    evidence_tier: str
    environment_class: str
    run_id: str
    attempt_id: str
    result_dir: Path
    data_mode: str
    train_steps: int
    val_loss_every: int
    timed_seconds: int | None
    lr_scale: float
    gpu_id: str
    wait_for_gpu: bool
    min_free_mib: int
    max_gpu_util: int
    data_path: str | None
    train_files: str | None
    val_files: str | None
    recall_eval: bool

    @property
    def classification(self) -> dict[str, Any]:
        return {
            "lane": "A01",
            "mode": self.mode,
            "arm": self.arm,
            "claim_label": self.claim_label,
            "evidence_tier": self.evidence_tier,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "environment_class": self.environment_class,
            "claim_eligible": self.mode in {"dry_run", "overfit", "timed_anchor"},
        }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=SUPPORTED_ARMS, required=True)
    parser.add_argument(
        "--mode",
        choices=("dry_run", "overfit", "timed_anchor", "pilot_5min"),
        required=True,
    )
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--no-wait-for-gpu", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=120_000)
    parser.add_argument("--max-gpu-util", type=int, default=5)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--val-loss-every", type=int, default=1)
    parser.add_argument("--timed-seconds", type=int)
    parser.add_argument("--data-mode", choices=("fineweb", "fixed_bank"))
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--data-path")
    parser.add_argument("--train-files")
    parser.add_argument("--val-files")
    parser.add_argument("--recall-eval", action="store_true")
    parser.add_argument("--environment-class")
    return parser.parse_args()


def default_steps(mode: str) -> int:
    if mode == "dry_run":
        return 3
    if mode == "overfit":
        return 80
    if mode in {"timed_anchor", "pilot_5min"}:
        return 10_000
    return 10_000


def default_data_mode(mode: str) -> str:
    if mode == "overfit":
        return "fixed_bank"
    return "fineweb"


def claim_label(mode: str) -> str:
    if mode == "overfit":
        return "learnability_smoke"
    if mode == "pilot_5min":
        return "representative_small"
    return "smoke"


def build_config(args: argparse.Namespace) -> AttemptConfig:
    if args.mode == "timed_anchor" and args.arm != "anchor":
        raise ValueError("timed_anchor mode is only valid for --arm anchor")
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = (
        args.run_id or f"a01_{args.mode}_{args.arm}_{timestamp}_{uuid.uuid4().hex[:8]}"
    )
    attempt_id = args.attempt_id or f"{run_id}_attempt_001"
    environment_class = (
        args.environment_class
        or os.environ.get("ATTN_ENVIRONMENT_CLASS")
        or ("shared_gpu" if args.gpu_id == "auto" else "exclusive_gpu")
    )
    return AttemptConfig(
        arm=args.arm,
        mode=args.mode,
        claim_label=claim_label(args.mode),
        evidence_tier=args.mode,
        environment_class=environment_class,
        run_id=run_id,
        attempt_id=attempt_id,
        result_dir=(args.result_root / run_id / attempt_id).resolve(),
        data_mode=args.data_mode or default_data_mode(args.mode),
        train_steps=args.train_steps or default_steps(args.mode),
        val_loss_every=args.val_loss_every,
        timed_seconds=(
            args.timed_seconds
            if args.timed_seconds is not None
            else (300 if args.mode in {"timed_anchor", "pilot_5min"} else None)
        ),
        lr_scale=args.lr_scale,
        gpu_id=args.gpu_id,
        wait_for_gpu=not args.no_wait_for_gpu,
        min_free_mib=args.min_free_mib,
        max_gpu_util=args.max_gpu_util,
        data_path=args.data_path,
        train_files=args.train_files,
        val_files=args.val_files,
        recall_eval=args.recall_eval,
    )


def query_gpus() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"failed to query nvidia-smi: {exc}") from exc
    gpus = []
    for line in proc.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        index, name, total, used, free, util = fields
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "memory_free_mib": int(free),
                "utilization_gpu_pct": int(util),
            }
        )
    return gpus


def select_gpu(config: AttemptConfig) -> tuple[str, list[dict[str, Any]]]:
    if config.gpu_id != "auto":
        return config.gpu_id, query_gpus()
    snapshots: list[dict[str, Any]] = []
    while True:
        gpus = query_gpus()
        snapshots = gpus
        candidates = [
            gpu
            for gpu in gpus
            if gpu["memory_free_mib"] >= config.min_free_mib
            and gpu["utilization_gpu_pct"] <= config.max_gpu_util
        ]
        if candidates:
            candidates.sort(
                key=lambda gpu: (gpu["utilization_gpu_pct"], -gpu["memory_free_mib"])
            )
            return str(candidates[0]["index"]), snapshots
        if not config.wait_for_gpu:
            raise RuntimeError(
                "no GPU meets free-memory/utilization thresholds; "
                f"thresholds: free>={config.min_free_mib} MiB, "
                f"util<={config.max_gpu_util}%"
            )
        time.sleep(60)


def training_env(config: AttemptConfig, gpu_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    cc = env.get("CC", "/usr/bin/gcc")
    cxx = env.get("CXX", "/usr/bin/g++")
    env["CC"] = cc if Path(cc).exists() else "/usr/bin/gcc"
    env["CXX"] = cxx if Path(cxx).exists() else "/usr/bin/g++"
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["LOCAL_RANK"] = "0"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = env.get("MASTER_PORT", "29571")
    env["ATTN_GEOMETRY"] = config.arm
    env["ATTN_RUN_ID"] = config.run_id
    env["ATTN_DATA_MODE"] = config.data_mode
    env["ATTN_TRAIN_STEPS"] = str(config.train_steps)
    env["ATTN_VAL_LOSS_EVERY"] = str(config.val_loss_every)
    env["ATTN_LR_SCALE"] = str(config.lr_scale)
    env["ATTN_MODEL_LAYERS"] = env.get("ATTN_MODEL_LAYERS", "8")
    env["ATTN_MODEL_HEADS"] = env.get("ATTN_MODEL_HEADS", "6")
    env["ATTN_HEAD_DIM"] = env.get("ATTN_HEAD_DIM", "64")
    env["ATTN_MODEL_DIM"] = env.get("ATTN_MODEL_DIM", "384")
    env["ATTN_SEQ_LEN"] = env.get("ATTN_SEQ_LEN", "1024")
    env["ATTN_BATCH_TOKENS"] = env.get("ATTN_BATCH_TOKENS", "8192")
    env["ATTN_VAL_BATCH_SIZE"] = env.get("ATTN_VAL_BATCH_SIZE", "16384")
    env["ATTN_VAL_TOKENS"] = env.get("ATTN_VAL_TOKENS", "16384")
    env["ATTN_BIGRAM_DIM"] = env.get("ATTN_BIGRAM_DIM", "384")
    env["ATTN_CUDA_MEMORY_FRACTION"] = env.get("ATTN_CUDA_MEMORY_FRACTION", "0.35")
    env["ATTN_CE_COMPUTE_CAPABILITY"] = env.get("ATTN_CE_COMPUTE_CAPABILITY", "100")
    env["ATTN_BACKEND"] = env.get("ATTN_BACKEND", "fa2")
    if config.data_mode == "fixed_bank":
        env["ATTN_OFFLINE_PREFIX_TABLE"] = env.get("ATTN_OFFLINE_PREFIX_TABLE", "1")
    env["DISABLE_FP8"] = env.get("DISABLE_FP8", "1")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(config.result_dir / "torchinductor_cache")
    env["TRITON_CACHE_DIR"] = str(config.result_dir / "triton_cache")
    if config.timed_seconds is not None:
        env["ATTN_TIMED_SECONDS"] = str(config.timed_seconds)
    if config.data_path is not None:
        env["DATA_PATH"] = config.data_path
    if config.train_files is not None:
        env["ATTN_TRAIN_FILES"] = config.train_files
    if config.val_files is not None:
        env["ATTN_VAL_FILES"] = config.val_files
    if config.recall_eval:
        env["ATTN_RECALL_EVAL"] = "1"
    return env


def redact_env(env: dict[str, str]) -> dict[str, str]:
    keys = [
        key
        for key in env
        if key.startswith("ATTN_")
        or key
        in {
            "CUDA_VISIBLE_DEVICES",
            "CC",
            "CXX",
            "DATA_PATH",
            "DISABLE_FP8",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            "RANK",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "WORLD_SIZE",
        }
    ]
    return {key: env[key] for key in sorted(keys)}


def data_provenance(env: dict[str, str], config: AttemptConfig) -> dict[str, Any]:
    offline_prefix = env.get("ATTN_OFFLINE_PREFIX_TABLE", "0").lower()
    offline_requested = offline_prefix in {"1", "true", "yes", "on"}
    return {
        "mode": config.data_mode,
        "source_uri": env.get("ATTN_DATA_HDFS_URI"),
        "manifest_md5": env.get("ATTN_DATA_MANIFEST_MD5"),
        "host_dir": env.get("ATTN_DATA_HOST_DIR"),
        "rootfs_dir": env.get("ATTN_DATA_ROOTFS_DIR"),
        "data_path": env.get("DATA_PATH"),
        "train_files": env.get("ATTN_TRAIN_FILES"),
        "val_files": env.get("ATTN_VAL_FILES"),
        "hdfs_strict_output": env.get("ATTN_HDFS_STRICT_OUTPUT"),
        "offline_prefix_table_requested": offline_requested,
        "prefix_table_source_prelaunch": (
            "offline_forced" if offline_requested else "tiktoken_gpt2"
        ),
    }


def parse_log(log_path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "compile_warmup_time_ms": None,
        "eval_times_ms": [],
        "val_losses": [],
        "train_times_ms": [],
        "step_avgs_ms": [],
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
        "recall_probe": None,
        "prefix_table_source": None,
        "prefix_table_fallback_reason": None,
    }
    if not log_path.exists():
        return metrics
    step_re = re.compile(
        r"step:(?P<step>\d+)/(?P<total>\d+)(?: val_loss:(?P<val>[0-9.]+))? "
        r"train_time:(?P<train>[0-9.]+)ms step_avg:(?P<avg>[0-9.]+)ms"
    )
    peak_re = re.compile(
        r"peak memory allocated: (?P<allocated>\d+) MiB reserved: (?P<reserved>\d+) MiB"
    )
    for line in log_path.read_text(errors="replace").splitlines():
        step_match = step_re.search(line)
        if step_match:
            if step_match.group("val") is not None:
                metrics["val_losses"].append(
                    {
                        "step": int(step_match.group("step")),
                        "val_loss": float(step_match.group("val")),
                    }
                )
            metrics["train_times_ms"].append(float(step_match.group("train")))
            metrics["step_avgs_ms"].append(float(step_match.group("avg")))
        peak_match = peak_re.search(line)
        if peak_match:
            metrics["peak_allocated_mib"] = int(peak_match.group("allocated"))
            metrics["peak_reserved_mib"] = int(peak_match.group("reserved"))
        if line.startswith("compile_warmup_time_ms:"):
            metrics["compile_warmup_time_ms"] = float(line.split(":", 1)[1])
        if line.startswith("eval_time_ms:"):
            pieces = line.split()
            metrics["eval_times_ms"].append(
                {
                    "eval_time_ms": float(pieces[0].split(":", 1)[1]),
                    "step": int(pieces[1].split(":", 1)[1]),
                }
            )
        if line.startswith("recall_probe:"):
            metrics["recall_probe"] = json.loads(line.split(":", 1)[1])
        if line.startswith("prefix_table_source:"):
            metrics["prefix_table_source"] = line.split(":", 1)[1]
        if line.startswith("prefix_table_fallback_reason:"):
            metrics["prefix_table_fallback_reason"] = line.split(":", 1)[1]
    return metrics


def main() -> int:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        print(
            "error: run through experiments/attn_geometry_speedrun/run_arm.sh",
            file=sys.stderr,
        )
        return 2
    config: AttemptConfig | None = None
    try:
        config = build_config(parse_args())
        config.result_dir.mkdir(parents=True, exist_ok=False)
        gpu_id, gpu_snapshot = select_gpu(config)
        env = training_env(config, gpu_id)
        data = data_provenance(env, config)
        argv = [
            sys.executable,
            str(VARIANT),
        ]
        attempt = {
            "schema_version": 1,
            "classification": config.classification,
            "source": str(VARIANT),
            "command": {"argv": argv},
            "gpu_topology": {
                "num_gpus": 1,
                "gpu_id": gpu_id,
                "visible_devices": env["CUDA_VISIBLE_DEVICES"],
                "prelaunch_snapshot": gpu_snapshot,
            },
            "data": data,
            "environment": redact_env(env),
        }
        write_json(config.result_dir / "attempt.json", attempt)
        (config.result_dir / "command.argv").write_text("\n".join(argv) + "\n")
        write_json(config.result_dir / "command.env.json", redact_env(env))

        start = time.time()
        with (config.result_dir / "stdout.log").open("w") as stdout, (
            config.result_dir / "stderr.log"
        ).open("w") as stderr:
            proc = subprocess.run(
                argv,
                cwd=config.result_dir,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=(config.timed_seconds + 1800) if config.timed_seconds else None,
            )
        elapsed = time.time() - start
        log_path = config.result_dir / "logs" / f"{config.run_id}.txt"
        metrics = parse_log(log_path)
        summary = {
            "schema_version": 1,
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "elapsed_seconds": elapsed,
            "classification": config.classification,
            "data": data,
            "metrics": metrics,
            "log_path": str(log_path),
        }
        write_json(config.result_dir / "summary.json", summary)
        return proc.returncode
    except subprocess.TimeoutExpired as exc:
        print(f"error: timed out: {exc}", file=sys.stderr)
        if config is not None:
            write_json(
                config.result_dir / "summary.json",
                {
                    "schema_version": 1,
                    "ok": False,
                    "returncode": 124,
                    "classification": config.classification,
                    "blocker": {
                        "phase": "launch",
                        "message": f"timed out after {exc.timeout} seconds",
                    },
                },
            )
        return 124
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
