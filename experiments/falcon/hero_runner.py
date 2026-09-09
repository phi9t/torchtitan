# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Step 5 local-hero runner for the Falcon FWA campaign (ticket 08).

Trains one frozen arm (A0 softmax, A1 GDN, or A5 winner) at the ticket-05
budget with matched 16,384 tokens/step. Checkpoints are on so a job can pause
and resume from ``{dump}/checkpoint`` without replaying completed steps.

Pause: ``touch {dump}/PAUSE`` or SIGINT/SIGTERM. The current optimizer step
finishes, a full train-state checkpoint is written, then the process exits.
Resume: re-run the same command. A leftover PAUSE file is cleared on start.

Claim label: ``representative_training``. Not Table 1 / 49.2B.

    experiments/falcon/run.sh hero --arm A5 --steps 20000
    experiments/falcon/run.sh pause --arm A5
    experiments/falcon/run.sh hero --arm A5 --steps 20000   # resumes
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import time
from pathlib import Path

_HERO_ARMS = ("A0", "A1", "A5")
_RESULTS_ROOT = Path("experiments/falcon/results/hero")
_DISK_FLOOR_BYTES = 5 * 1024**3
_PAUSE_NAME = "PAUSE"
_FINEWEB_TRAIN = Path(
    "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa/"
    "data/fineweb10B"
)


def write_synthetic_bin(path: Path, *, num_tokens: int, vocab_size: int) -> None:
    import torch

    from torchtitan.experiments.falcon.bin_reader import write_nanogpt_bin

    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(
        0, vocab_size, (num_tokens,), generator=generator, dtype=torch.int64
    )
    write_nanogpt_bin(path, tokens)


def pause_path(dump_folder: Path) -> Path:
    return dump_folder / _PAUSE_NAME


def request_pause(dump_folder: Path) -> Path:
    dump_folder.mkdir(parents=True, exist_ok=True)
    path = pause_path(dump_folder)
    path.touch()
    return path


def clear_pause(dump_folder: Path) -> None:
    path = pause_path(dump_folder)
    if path.exists():
        path.unlink()


def pause_requested(dump_folder: Path) -> bool:
    return pause_path(dump_folder).exists()


def install_pause_control(trainer, dump_folder: Path) -> dict:
    """Stop after the current step and force a resumable last-step save.

    SIGINT/SIGTERM and a ``PAUSE`` file are equivalent. The trainer loop is
    not modified; this wraps ``should_continue_training`` and ``save``.
    """
    state = {"requested": False, "saved": False}

    def _trip(*_args) -> None:
        state["requested"] = True
        request_pause(dump_folder)

    signal.signal(signal.SIGTERM, _trip)
    signal.signal(signal.SIGINT, _trip)

    orig_save = trainer.checkpointer.save

    def save(curr_step, last_step=False):
        pause = state["requested"] or pause_requested(dump_folder)
        if pause:
            state["requested"] = True
        written = orig_save(curr_step, last_step=last_step or pause)
        if pause:
            trainer.config.training.steps = curr_step
            state["saved"] = True
        return written

    trainer.checkpointer.save = save  # type: ignore[method-assign]

    orig_continue = trainer.should_continue_training

    def should_continue() -> bool:
        pause = state["requested"] or pause_requested(dump_folder)
        if not pause:
            return orig_continue()
        state["requested"] = True
        if trainer.step > 0 and not state["saved"]:
            trainer.checkpointer.save(trainer.step, last_step=True)
            state["saved"] = True
        trainer.config.training.steps = trainer.step
        return False

    trainer.should_continue_training = should_continue  # type: ignore[method-assign]
    return state


def _require_disk_headroom(path: Path) -> None:
    usage = shutil.disk_usage(path)
    if usage.free < _DISK_FLOOR_BYTES:
        raise RuntimeError(
            f"free disk {usage.free} is below the 5G checkpoint floor at {path}"
        )


def _require_train_corpus() -> None:
    shards = sorted(_FINEWEB_TRAIN.glob("fineweb_train_00000[1-9].bin"))
    if len(shards) < 9:
        raise RuntimeError(
            f"FineWeb train shards missing under {_FINEWEB_TRAIN} "
            f"(found {len(shards)} of 9)"
        )


def build_hero_config(
    arm: str,
    *,
    seed: int,
    steps: int,
    dump_folder: Path,
    dry_run: bool,
    checkpoint_interval: int,
):
    """Science Trainer config for one hero arm, with resumable checkpoints."""
    from torchtitan.config import ConfigManager
    from torchtitan.experiments.falcon.config_registry import apply_arm, falcon_science

    if arm not in _HERO_ARMS:
        raise ValueError(f"hero arm must be one of {_HERO_ARMS}, got {arm!r}")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be >= 1")

    config = falcon_science()
    apply_arm(config, arm)
    config.training.steps = steps
    config.lr_scheduler.total_steps = steps
    config.lr_scheduler.warmup_steps = min(200, max(1, steps // 40))
    config.debug.seed = seed
    config.debug.enable_structured_logging = False
    config.dump_folder = str(dump_folder)
    config.checkpoint.enable = True
    config.checkpoint.interval = checkpoint_interval
    config.checkpoint.last_save_model_only = False
    config.checkpoint.keep_latest_k = 2
    config.checkpoint.load_step = -1
    config.metrics.log_freq = 1 if dry_run else 50

    if dry_run:
        inner = config.model_spec.model.config
        inner.hidden_size = 128
        inner.num_heads = 4
        inner.head_dim = 32
        inner.num_hidden_layers = 2
        inner.intermediate_size = 512
        inner.seq_len = 128
        config.training.seq_len = 128
        config.training.local_batch_size = 8

    manager = ConfigManager()
    manager.config = config
    manager._validate_config()
    return config


def _set_single_process_env(*, master_port: int) -> None:
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("NGPU", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)


def _write_status(dump_folder: Path, payload: dict) -> None:
    (dump_folder / "hero_status.json").write_text(json.dumps(payload, indent=2) + "\n")


def run_hero(
    arm: str,
    *,
    seed: int,
    steps: int,
    dump_folder: Path,
    dry_run: bool,
    checkpoint_interval: int,
    master_port: int,
    fresh: bool = False,
) -> dict:
    from experiments.falcon.ablation_runner import (
        _eval_val_ce,
        _install_finite_loss_probe,
    )

    _set_single_process_env(master_port=master_port)
    dump_folder.mkdir(parents=True, exist_ok=True)
    _require_disk_headroom(dump_folder)
    if not dry_run:
        _require_train_corpus()

    ckpt_root = dump_folder / "checkpoint"
    if fresh and ckpt_root.is_dir() and any(ckpt_root.iterdir()):
        raise RuntimeError(f"--fresh refused: checkpoint already exists at {ckpt_root}")
    # A leftover PAUSE must not immediately re-stop a resume launch.
    clear_pause(dump_folder)

    config = build_hero_config(
        arm,
        seed=seed,
        steps=steps,
        dump_folder=dump_folder,
        dry_run=dry_run,
        checkpoint_interval=checkpoint_interval,
    )
    losses, restore_probe = _install_finite_loss_probe()
    started = time.time()
    trainer = config.build()
    install_pause_control(trainer, dump_folder)
    trainer.checkpointer.load(step=config.checkpoint.load_step)
    resume_step = trainer.step
    (dump_folder / "hero.pid").write_text(f"{os.getpid()}\n")
    _write_status(
        dump_folder,
        {
            "arm": arm,
            "pid": os.getpid(),
            "requested_steps": steps,
            "resume_step": resume_step,
            "running": True,
            "paused": False,
        },
    )
    unstable = False
    completed_steps = resume_step
    paused = False
    try:
        trainer.train()
        completed_steps = trainer.step
        paused = pause_requested(dump_folder) or completed_steps < steps
    except RuntimeError as exc:
        if "not finite" not in str(exc):
            trainer.close()
            restore_probe()
            raise
        unstable = True
        completed_steps = trainer.step
    val = {"val_ce": float("nan"), "val_ppl": float("nan"), "val_tokens": 0}
    try:
        if not dry_run and not paused and not unstable:
            val = _eval_val_ce(trainer.model_parts, config, max_windows=64)
    finally:
        trainer.close()
        restore_probe()
        pid_path = dump_folder / "hero.pid"
        if pid_path.exists():
            pid_path.unlink()

    inner = config.model_spec.model.config
    result = {
        "claim_label": "smoke" if dry_run else "representative_training",
        "arm": arm,
        "seed": seed,
        "requested_steps": steps,
        "resume_step": resume_step,
        "completed_steps": completed_steps,
        "paused": paused,
        "mixer": inner.mixer,
        "variant": inner.variant,
        "alignment": inner.alignment,
        "phi": inner.phi,
        "initial_loss": losses[0] if losses else float("nan"),
        "final_loss": losses[-1] if losses else float("nan"),
        "all_finite": all(math.isfinite(x) for x in losses) if losses else False,
        "unstable": unstable,
        "val_ce": val["val_ce"],
        "val_ppl": val["val_ppl"],
        "tokens_per_step": config.training.local_batch_size * config.training.seq_len,
        "elapsed_sec": round(time.time() - started, 2),
        "dump_folder": str(dump_folder),
        "dry_run": dry_run,
        "checkpoint_interval": checkpoint_interval,
    }
    (dump_folder / "hero_outcome.json").write_text(json.dumps(result, indent=2) + "\n")
    _write_status(
        dump_folder,
        {
            "arm": arm,
            "pid": None,
            "requested_steps": steps,
            "resume_step": resume_step,
            "completed_steps": completed_steps,
            "running": False,
            "paused": paused,
        },
    )
    return result


def _dump_for(arm: str, seed: int, tag: str) -> Path:
    return _RESULTS_ROOT / tag / f"{arm}_seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=_HERO_ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--master-port", type=int, default=29580)
    parser.add_argument("--tag", default="hero_20k")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Refuse to start if {dump}/checkpoint already exists.",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Request a graceful pause of a running hero on this dump.",
    )
    args = parser.parse_args()
    dump = _dump_for(args.arm, args.seed, args.tag)
    if args.pause:
        path = request_pause(dump)
        print(json.dumps({"pause": str(path), "arm": args.arm}, indent=2))
        return
    if args.steps < 4:
        raise ValueError("hero needs steps >= 4 so pause/resume can fire")

    result = run_hero(
        args.arm,
        seed=args.seed,
        steps=args.steps,
        dump_folder=dump,
        dry_run=args.dry_run,
        checkpoint_interval=args.checkpoint_interval,
        master_port=args.master_port,
        fresh=args.fresh,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
