# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Few-step Trainer smoke for the tiny Falcon overfit config.

Builds ``falcon_tiny_overfit`` through the core ``Trainer`` under the fake
backend and runs N>1 optimizer steps, asserting a finite loss at every step.
``run_train.sh COMM_MODE=fake_backend`` hard-clamps ``--training.steps 1``, so
this driver constructs the same registry config directly to exercise more than
one step. It is a plumbing/smoke artifact only, not a convergence claim.

Run under the rootfs:

    scripts/rootfs/enter_rootfs.sh -- bash -lc \
      'cd /workspace/torchtitan && \
       TORCHTITAN_IN_ROOTFS=1 python -m experiments.falcon.smoke_trainer --steps 5'
"""

from __future__ import annotations

import argparse
import math
import os


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("smoke requires steps >= 2 to prove N>1 training")

    # Single-process fake backend: no torchrun, no NCCL.
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("NGPU", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29513")

    from torchtitan.config import ConfigManager
    from torchtitan.experiments.falcon.config_registry import falcon_tiny_overfit

    losses = _install_finite_loss_probe()

    config = falcon_tiny_overfit()
    config.training.steps = args.steps
    config.lr_scheduler.total_steps = args.steps
    config.comm.mode = "fake_backend"
    config.debug.enable_structured_logging = False

    # Route through ConfigManager validation so we exercise the real config
    # path, then swap in our multi-step config object.
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
        f"[falcon-smoke] OK steps={args.steps} "
        f"initial_loss={losses[0]:.5f} final_loss={losses[-1]:.5f} "
        f"all_finite=True"
    )


if __name__ == "__main__":
    main()
