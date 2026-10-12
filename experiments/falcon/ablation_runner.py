# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Step 4 ablation runner for the Falcon FWA campaign (ticket 07).

Runs arms A0-A5 (see ``config_registry._ARM_KNOBS``) on the shared science
model with matched data, global batch, tokens/step, and steps. Each arm x seed
is a single sequential run on one B200; checkpoints are disabled (44G disk).

Per run this driver:

1. Builds ``falcon_science`` through the core ``Trainer`` and overlays the arm
   knobs via ``apply_arm``. A finite-loss probe records every logged train loss.
2. Trains ``--steps`` optimizer steps (default 8000; floor 4000 if a run goes
   non-finite).
3. Evaluates held-out next-token CE/PPL on a bounded window of the FineWeb val
   shard (``fineweb_val_000000.bin``).
4. Trains a separate small addition model per mixer (N=16 ID, M=24 OOD) and
   reports teacher-forced suffix accuracy ID and OOD. Addition is kept cheap
   relative to the LM arm.

Claim labels: ``smoke`` for --dry-run, ``representative_small`` +
``replicated_eval`` for the full table. This is NOT a 130M / 50B result.

Everything runs inside the bwrap rootfs; this module must not be invoked with
host-side Python. Results (JSON + markdown) land under
``experiments/falcon/results/ablations/`` which is gitignored.

Run under the rootfs (single B200):

    scripts/rootfs/enter_rootfs.sh -- bash -lc \
      'cd /workspace/torchtitan && TORCHTITAN_IN_ROOTFS=1 CUDA_VISIBLE_DEVICES=0 \
       python -m experiments.falcon.ablation_runner --dry-run --steps 3'
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

_VAL_BIN = (
    "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa/"
    "data/fineweb10B/fineweb_val_000000.bin"
)
_ARMS = ["A0", "A1", "A2", "A3", "A4", "A5"]
_RESULTS_ROOT = Path("experiments/falcon/results/ablations")


def _pristine_metrics_log():
    """The unwrapped ``MetricsProcessor.log`` captured once at import.

    Every per-run probe restores to and wraps this, so no probe ever chains
    onto another probe (which would cross-contaminate per-arm loss lists).
    """
    from torchtitan.components.metrics import MetricsProcessor

    return MetricsProcessor.log


_PRISTINE_METRICS_LOG = _pristine_metrics_log()


def _install_finite_loss_probe() -> tuple[list[float], "callable"]:
    """Record every logged train loss for finiteness / curve checks.

    Returns the per-run loss list and a restore callable that must be invoked
    once the run finishes. The probe wraps ``MetricsProcessor.log``, which is a
    class attribute; each run installs against the pristine, module-level
    original (captured once) instead of chaining onto a previously installed
    probe. Chaining would make the last-installed probe re-enter every earlier
    probe, cross-contaminating the recorded losses across arms.
    """
    from torchtitan.components.metrics import MetricsProcessor

    seen: list[float] = []
    original_log = _PRISTINE_METRICS_LOG

    def probe(self, step, global_avg_loss, global_max_loss, grad_norm, *a, **kw):
        seen.append(float(global_avg_loss))
        return original_log(
            self, step, global_avg_loss, global_max_loss, grad_norm, *a, **kw
        )

    MetricsProcessor.log = probe  # type: ignore[method-assign]

    def restore() -> None:
        MetricsProcessor.log = _PRISTINE_METRICS_LOG  # type: ignore[method-assign]

    return seen, restore


def _set_single_process_env() -> None:
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("NGPU", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")


def _build_arm_config(arm: str, *, seed: int, steps: int, dry_run: bool):
    """Build a science Trainer config for one arm x seed."""
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
    # Per-arm dump folder keeps logs separate; no checkpoints are written.
    config.dump_folder = str(_RESULTS_ROOT / f"{arm}_seed{seed}")

    if dry_run:
        # Cheap smoke shape: keep the science surface (mixer, alignment, phi,
        # GPT-2 vocab, SwiGLU) but shrink width/seq so a few steps are fast.
        inner = config.model_spec.model.config
        inner.hidden_size = 128
        inner.num_heads = 4
        inner.head_dim = 32
        inner.num_hidden_layers = 2
        inner.intermediate_size = 512
        inner.seq_len = 128
        config.training.seq_len = 128
        config.training.local_batch_size = 8
        config.metrics.log_freq = 1
    else:
        config.metrics.log_freq = 50

    manager = ConfigManager()
    manager.config = config
    manager._validate_config()
    return config


def _eval_val_ce(model_parts, config, *, max_windows: int) -> dict[str, float]:
    """Held-out next-token CE/PPL on a bounded window of the val shard.

    Uses the same nanogpt-bin loader as training but points it at the val shard
    and consumes at most ``max_windows`` batches. Returns token-weighted mean CE
    and PPL.
    """
    import torch
    import torch.nn.functional as F

    from torchtitan.experiments.falcon.bin_reader import NanoGptBinDataLoader

    inner = config.model_spec.model.config
    loader = NanoGptBinDataLoader(
        [Path(_VAL_BIN)],
        seq_len=config.training.seq_len,
        local_batch_size=config.training.local_batch_size,
        vocab_size=inner.vocab_size,
    )
    device = next(model_parts[0].parameters()).device
    model_parts[0].eval()
    total_loss = 0.0
    total_tokens = 0
    it = iter(loader)
    with torch.no_grad():
        for _ in range(max_windows):
            batch, labels = next(it)
            tokens = batch["input"].to(device)
            labels = labels.to(device)
            logits = model_parts[0](tokens)
            loss = F.cross_entropy(
                logits.reshape(-1, inner.vocab_size).float(),
                labels.reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += labels.numel()
    model_parts[0].train()
    mean_ce = total_loss / max(1, total_tokens)
    return {
        "val_ce": mean_ce,
        "val_ppl": math.exp(min(mean_ce, 20.0)),
        "val_tokens": total_tokens,
    }


def _train_one_arm(
    arm: str, *, seed: int, steps: int, dry_run: bool, val_windows: int
) -> dict:
    """Train one arm x seed and evaluate held-out CE. Returns a result dict."""
    config = _build_arm_config(arm, seed=seed, steps=steps, dry_run=dry_run)
    losses, restore_probe = _install_finite_loss_probe()

    started = time.time()
    trainer = config.build()
    unstable = False
    completed_steps = 0
    try:
        trainer.train()
        completed_steps = trainer.step
    except RuntimeError as exc:
        # The core Trainer raises RuntimeError on a non-finite loss. Treat that
        # as an unstable run rather than a crash: keep the steps reached and let
        # the caller decide (spec floor 4000 for an unstable arm).
        if "not finite" not in str(exc):
            trainer.close()
            restore_probe()
            raise
        unstable = True
        completed_steps = trainer.step
    val = {"val_ce": float("nan"), "val_ppl": float("nan"), "val_tokens": 0}
    try:
        val = _eval_val_ce(trainer.model_parts, config, max_windows=val_windows)
    finally:
        trainer.close()
        restore_probe()
    elapsed = time.time() - started

    finite = all(math.isfinite(x) for x in losses) if losses else False
    inner = config.model_spec.model.config
    return {
        "arm": arm,
        "seed": seed,
        "requested_steps": steps,
        "completed_steps": completed_steps,
        "mixer": inner.mixer,
        "variant": inner.variant,
        "alignment": inner.alignment,
        "phi": inner.phi,
        "initial_loss": losses[0] if losses else float("nan"),
        "final_loss": losses[-1] if losses else float("nan"),
        "all_finite": finite,
        "unstable": unstable,
        "val_ce": val["val_ce"],
        "val_ppl": val["val_ppl"],
        "val_tokens": val["val_tokens"],
        "elapsed_sec": round(elapsed, 2),
        "tokens_per_step": config.training.local_batch_size * config.training.seq_len,
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# Addition transfer: separate small train per mixer (paper section 5.2 setup).
# ---------------------------------------------------------------------------


@dataclass
class _AdditionResult:
    id_acc: float
    ood_acc: float
    steps: int


def _addition_model(
    mixer: str, variant: str, alignment: str, phi: str, vocab_size: int, seq_len: int
):
    from torchtitan.experiments.falcon.model import FalconConfig, FalconForCausalLM
    from torchtitan.experiments.falcon.overfit import _initialize_weights

    config = FalconConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_hidden_layers=2,
        num_heads=4,
        head_dim=32,
        intermediate_size=512,
        seq_len=seq_len,
        mixer=mixer,
        variant=variant,
        alignment=alignment,
        phi=phi,
    )
    model = FalconForCausalLM(config)
    _initialize_weights(model)
    return model, config


def _suffix_accuracy(model, batch, vocab_size, device) -> float:
    """Teacher-forced fraction of examples whose answer suffix is fully correct."""
    import torch

    tokens = batch["input"].to(device)
    labels = batch["labels"].to(device)
    mask = batch["loss_mask"].to(device)
    with torch.no_grad():
        logits = model(tokens)
        preds = logits.argmax(dim=-1)
    correct_tok = (preds == labels) | (~mask)
    # An example passes only if every masked (answer) token is correct.
    per_example = correct_tok.all(dim=1)
    return float(per_example.float().mean().item())


def _train_addition(
    mixer: str, variant: str, alignment: str, phi: str, *, seed: int, steps: int, device
) -> _AdditionResult:
    import torch

    from torchtitan.experiments.falcon.addition import (
        AdditionVocab,
        build_addition_dataset,
    )

    torch.manual_seed(seed)
    vocab = AdditionVocab()
    # Keep addition cheap vs the LM arms: small train/eval banks, short trains.
    ds = build_addition_dataset(n=16, m=24, num_id=2048, num_ood=512, seed=seed)
    train_len = max(len(ex.tokens) for ex in ds.id_examples)
    id_eval = ds.collate(ds.id_examples[:256], vocab=vocab)
    ood_eval = ds.collate(ds.ood_examples[:256], vocab=vocab)
    seq_len = max(train_len, id_eval["input"].shape[1], ood_eval["input"].shape[1])

    model, _ = _addition_model(mixer, variant, alignment, phi, vocab.size, seq_len)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=0.0)
    rng = torch.Generator().manual_seed(seed + 1)
    batch_size = 128
    model.train()
    for _ in range(steps):
        idx = torch.randint(
            0, len(ds.id_examples), (batch_size,), generator=rng
        ).tolist()
        batch = ds.collate([ds.id_examples[i] for i in idx], vocab=vocab)
        tokens = batch["input"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["loss_mask"].to(device)
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab.size),
            labels.reshape(-1),
            reduction="none",
        )
        loss = (loss * mask.reshape(-1).float()).sum() / mask.sum().clamp(min=1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    id_acc = _suffix_accuracy(model, id_eval, vocab.size, device)
    ood_acc = _suffix_accuracy(model, ood_eval, vocab.size, device)
    return _AdditionResult(id_acc=id_acc, ood_acc=ood_acc, steps=steps)


def _mixer_signature(arm: str) -> tuple[str, str, str, str]:
    from torchtitan.experiments.falcon.config_registry import _ARM_KNOBS

    knobs = _ARM_KNOBS[arm]
    return (
        knobs.get("mixer", "falcon"),
        knobs.get("variant", "falcon1a"),
        knobs.get("alignment", "delayed"),
        knobs.get("phi", "rms"),
    )


def _write_table(
    records: list[dict], addition: dict, out_dir: Path, *, claim_label: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "claim_label": claim_label,
        "arms": records,
        "addition": addition,
    }
    (out_dir / "table.json").write_text(json.dumps(payload, indent=2))

    lines = [
        f"# Falcon Step 4 ablation table (claim_label={claim_label})",
        "",
        "## LM arms (held-out val on fineweb_val_000000.bin)",
        "",
        "| arm | mixer | variant | alignment | phi | seed | steps | final_loss | val_ce | val_ppl | finite |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        lines.append(
            f"| {r['arm']} | {r['mixer']} | {r['variant']} | {r['alignment']} | "
            f"{r['phi']} | {r['seed']} | {r['completed_steps']} | "
            f"{r['final_loss']:.4f} | {r['val_ce']:.4f} | {r['val_ppl']:.2f} | "
            f"{r['all_finite']} |"
        )
    lines += [
        "",
        "## Addition transfer (separate small train per mixer, teacher-forced)",
        "",
        "| arm | mixer | variant | alignment | phi | seed | ID acc | OOD acc |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, vals in addition.items():
        lines.append(
            f"| {vals['arm']} | {vals['mixer']} | {vals['variant']} | "
            f"{vals['alignment']} | {vals['phi']} | {vals['seed']} | "
            f"{vals['id_acc']:.3f} | {vals['ood_acc']:.3f} |"
        )
    (out_dir / "table.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--arms", type=str, nargs="+", default=_ARMS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Few-step smoke on all arms with a shrunk shape.",
    )
    parser.add_argument(
        "--val-windows",
        type=int,
        default=64,
        help="Bounded held-out val batches for CE/PPL.",
    )
    parser.add_argument("--addition-steps", type=int, default=1500)
    parser.add_argument("--skip-addition", action="store_true")
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Result subdir under experiments/falcon/results/ablations/.",
    )
    args = parser.parse_args()

    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise RuntimeError(
            "ablation_runner must run inside the bwrap rootfs "
            "(TORCHTITAN_IN_ROOTFS=1). Launch via scripts/rootfs/enter_rootfs.sh."
        )

    _set_single_process_env()

    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dry_run:
        tag = args.tag or "dry_run"
        steps = min(args.steps, 4)
        seeds = args.seeds[:1]
        val_windows = min(args.val_windows, 4)
        add_steps = min(args.addition_steps, 20)
        claim = "smoke"
    else:
        tag = args.tag or "full"
        steps = args.steps
        seeds = args.seeds
        val_windows = args.val_windows
        add_steps = args.addition_steps
        claim = "representative_small + replicated_eval"

    out_dir = _RESULTS_ROOT / tag
    records: list[dict] = []
    for seed in seeds:
        for arm in args.arms:
            print(
                f"[falcon-ablation] arm={arm} seed={seed} steps={steps} "
                f"dry_run={args.dry_run}",
                flush=True,
            )
            rec = _train_one_arm(
                arm,
                seed=seed,
                steps=steps,
                dry_run=args.dry_run,
                val_windows=val_windows,
            )
            records.append(rec)
            print(
                f"[falcon-ablation] done arm={arm} seed={seed} "
                f"final_loss={rec['final_loss']:.4f} val_ce={rec['val_ce']:.4f} "
                f"val_ppl={rec['val_ppl']:.2f} finite={rec['all_finite']} "
                f"t={rec['elapsed_sec']}s",
                flush=True,
            )
            # Persist incrementally so a mid-run interruption keeps evidence.
            _write_table(records, {}, out_dir, claim_label=claim)

    addition: dict[str, dict] = {}
    if not args.skip_addition:
        for seed in seeds:
            for arm in args.arms:
                mixer, variant, alignment, phi = _mixer_signature(arm)
                res = _train_addition(
                    mixer,
                    variant,
                    alignment,
                    phi,
                    seed=seed,
                    steps=add_steps,
                    device=device,
                )
                key = f"{arm}_seed{seed}"
                addition[key] = {
                    "arm": arm,
                    "mixer": mixer,
                    "variant": variant,
                    "alignment": alignment,
                    "phi": phi,
                    "seed": seed,
                    "id_acc": res.id_acc,
                    "ood_acc": res.ood_acc,
                    "steps": res.steps,
                }
                print(
                    f"[falcon-addition] arm={arm} seed={seed} "
                    f"id_acc={res.id_acc:.3f} ood_acc={res.ood_acc:.3f}",
                    flush=True,
                )
                _write_table(records, addition, out_dir, claim_label=claim)

    _write_table(records, addition, out_dir, claim_label=claim)
    print(f"[falcon-ablation] wrote {out_dir}/table.json and table.md", flush=True)


if __name__ == "__main__":
    main()
