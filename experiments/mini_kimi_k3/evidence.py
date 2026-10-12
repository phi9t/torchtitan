# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run-attempt evidence helpers for the Mini Kimi K3 experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torchtitan.experiments.execution import models
from torchtitan.experiments.execution.lifecycle import RunAttempt


def initialize_launch_evidence_bundle(
    *,
    results_root: Path,
    run_id: str,
    attempt_id: str,
    mode: str,
    model_variant: str = "r1",
    parent_attempt_id: str | None = None,
) -> Path:
    """Create the immutable run-attempt bundle used before launch."""

    declaration = models.RunDeclaration(
        run_id=run_id,
        family="mini_kimi_k3",
        task="pretraining",
        lane=mode,
        fields={
            "model_variant": model_variant,
            "claim": "launch_readiness",
            "status": "initialized_not_trainable",
        },
    )
    attempt = RunAttempt.create(
        declaration,
        attempt_id=attempt_id,
        results_root=results_root,
        parent_attempt_id=parent_attempt_id,
    )
    return attempt.bundle_dir


def inspect_launch_evidence_bundle(
    *,
    results_root: Path,
    run_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    """Return the initialized bundle manifest evidence for preflight."""

    manifest_path = Path(results_root) / "runs" / run_id / attempt_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return {
        "manifest": str(manifest_path),
        "run_id": manifest["run"]["run_id"],
        "attempt_id": manifest["attempt"]["attempt_id"],
        "family": manifest["run"]["family"],
        "task": manifest["run"]["task"],
        "lane": manifest["run"]["lane"],
        "model_variant": manifest["run"]["fields"]["model_variant"],
    }
