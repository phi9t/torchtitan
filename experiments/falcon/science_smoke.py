# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Few-step science smoke for the Falcon Step 4 config.

Builds ``falcon_science`` through the core ``Trainer`` under the fake backend
and runs N>1 optimizer steps, asserting a finite loss at every step. The full
8000-step science config is registered in ``config_registry.falcon_science``;
this driver overrides it to a tiny step count (and, by default, a smaller model
and sequence length so a CPU smoke stays cheap), then runs a handful of steps.

The smoke reads a small synthetic nanogpt ``.bin`` written into a temp file so
it never touches the 1.9 GB FineWeb corpus. Real FineWeb bins are only needed
for a science-scale run (ticket 07), not this plumbing check.

This is a ``smoke`` claim only: N>1 finite-loss training, not convergence.

Run under the rootfs:

    scripts/rootfs/enter_rootfs.sh -- bash -lc \
      'cd /workspace/torchtitan && \
       TORCHTITAN_IN_ROOTFS=1 python -m experiments.falcon.science_smoke --steps 3'
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path


def _install_finite_loss_probe() -> list[float]:
    """Record every logged loss so we can assert finiteness after training."""
    from torchtitan.components.metrics import MetricsProcessor

    seen: list[float] = []
    original_log = MetricsProcessor.log

    def probe(self, step, global_avg_loss, global_max_loss, grad_norm, *a, **kw):
        seen.append(float(global_avg_loss))
        return original_log(
            self, step, global_avg_loss, global_max_loss, grad_norm, *a, **kw
        )

    MetricsProcessor.log = probe  # type: ignore[method-assign]
    return seen


def _write_synthetic_bin(path: Path, *, num_tokens: int, vocab_size: int) -> None:
    import torch

    from torchtitan.experiments.falcon.bin_reader import write_nanogpt_bin

    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(
        0, vocab_size, (num_tokens,), generator=generator, dtype=torch.int64
    )
    write_nanogpt_bin(path, tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--full-shape",
        action="store_true",
        help="Use the registered 4x256 seq-512 science shape (heavier on CPU).",
    )
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("smoke requires steps >= 2 to prove N>1 training")

    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("NGPU", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29514")

    from torchtitan.config import ConfigManager
    from torchtitan.experiments.falcon.config_registry import falcon_science

    losses = _install_finite_loss_probe()

    config = falcon_science()
    inner = config.model_spec.model.config

    if not args.full_shape:
        # Shrink to a CPU-friendly shape while keeping the science surface:
        # same mixer, delayed alignment, GPT-2 vocab, SwiGLU shell.
        inner.hidden_size = 64
        inner.num_heads = 4
        inner.head_dim = 16
        inner.num_hidden_layers = 2
        inner.intermediate_size = 128
        inner.seq_len = 64
        config.training.seq_len = 64
        config.training.local_batch_size = 4

    config.training.steps = args.steps
    config.lr_scheduler.total_steps = args.steps
    config.lr_scheduler.warmup_steps = 1
    config.comm.mode = "fake_backend"
    config.debug.enable_structured_logging = False
    config.metrics.log_freq = 1

    with tempfile.TemporaryDirectory() as tmp:
        bin_path = Path(tmp) / "falcon_science_smoke.bin"
        window = config.training.local_batch_size * config.training.seq_len + 1
        _write_synthetic_bin(
            bin_path,
            num_tokens=window * (args.steps + 2),
            vocab_size=inner.vocab_size,
        )
        config.dataloader.bin_glob = str(bin_path)

        manager = ConfigManager()
        manager.config = config
        manager._validate_config()

        trainer = config.build()
        try:
            trainer.train()
        finally:
            trainer.close()

    if not losses:
        raise RuntimeError("no loss was logged; smoke did not run a train step")
    if len(losses) < args.steps:
        raise RuntimeError(
            f"expected {args.steps} logged losses, saw {len(losses)}: {losses}"
        )
    for step, value in enumerate(losses, start=1):
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite loss {value} at step {step}")

    print(
        f"[falcon-science-smoke] OK steps={args.steps} "
        f"shape={'full' if args.full_shape else 'tiny'} "
        f"vocab={inner.vocab_size} "
        f"initial_loss={losses[0]:.5f} final_loss={losses[-1]:.5f} "
        f"all_finite=True"
    )


if __name__ == "__main__":
    main()
