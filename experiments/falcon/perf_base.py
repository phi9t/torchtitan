# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke-labeled science-shape Falcon run for the perf / attach sprint.

Builds ``falcon_science``, overlays one frozen arm (default A5), and runs
the core Trainer. No held-out eval, no addition, no checkpoints. This is
not ticket 08 and not a representative_training claim.

    experiments/falcon/run.sh perf --arm A5 --steps 20000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


_RESULTS_ROOT = Path("experiments/falcon/results/tooling")


def _set_single_process_env() -> None:
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("NGPU", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29527")
    # Redirected console.log is block-buffered without this.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _build_config(arm: str, *, seed: int, steps: int, log_freq: int, dump: Path):
    from torchtitan.config import ConfigManager
    from torchtitan.experiments.falcon.config_registry import apply_arm, falcon_science

    config = falcon_science()
    apply_arm(config, arm)
    config.training.steps = steps
    config.lr_scheduler.total_steps = steps
    config.lr_scheduler.warmup_steps = min(200, max(1, steps // 40))
    config.debug.seed = seed
    config.debug.enable_structured_logging = False
    config.checkpoint.enable = False
    config.metrics.log_freq = log_freq
    config.metrics.enable_tensorboard = False
    config.dump_folder = str(dump)
    manager = ConfigManager()
    manager.config = config
    manager._validate_config()
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="A5")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    if args.steps < 20:
        raise ValueError("perf base needs enough steps for attach windows")

    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise RuntimeError(
            "perf_base must run inside the bwrap rootfs "
            "(TORCHTITAN_IN_ROOTFS=1). Launch via experiments/falcon/run.sh perf."
        )

    _set_single_process_env()
    tag = args.tag or datetime.now(timezone.utc).strftime("perf_base_%Y%m%dT%H%M%SZ")
    dump = _RESULTS_ROOT / tag
    dump.mkdir(parents=True, exist_ok=True)

    config = _build_config(
        args.arm,
        seed=args.seed,
        steps=args.steps,
        log_freq=args.log_freq,
        dump=dump,
    )
    inner = config.model_spec.model.config
    manifest = {
        "claim_label": "smoke",
        "purpose": "perf_tooling_base",
        "arm": args.arm,
        "mixer": inner.mixer,
        "variant": inner.variant,
        "alignment": inner.alignment,
        "phi": inner.phi,
        "steps": args.steps,
        "seed": args.seed,
        "seq_len": config.training.seq_len,
        "local_batch_size": config.training.local_batch_size,
        "tokens_per_step": config.training.local_batch_size * config.training.seq_len,
        "dtype": config.training.dtype,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    (dump / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)

    started = time.time()
    trainer = config.build()
    try:
        trainer.train()
        completed = trainer.step
    finally:
        trainer.close()
    elapsed = time.time() - started
    summary = {
        **manifest,
        "completed_steps": completed,
        "elapsed_sec": round(elapsed, 2),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (dump / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
