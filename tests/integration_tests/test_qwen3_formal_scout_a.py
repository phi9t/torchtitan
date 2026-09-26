# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real-GPU acceptance test for the single-rank formal-verifier scout."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from torchtitan.experiments.qwen3_formal_verifier.scout_a import (
    run_scout_a,
    SCOUT_A_ATTEMPT_ID,
    SCOUT_A_RUN_ID,
)


pytestmark = pytest.mark.skipif(
    "QFV_SCOUT_A_OUTPUT_ROOT" not in os.environ,
    reason="run through experiments/qwen3_formal_verifier/run_scout_a.sh",
)


def test_real_cuda_qwen3_step_reaches_observed_formal_bundle() -> None:
    paths = run_scout_a(
        Path(os.environ["QFV_SCOUT_A_OUTPUT_ROOT"]),
        run_id=os.environ.get("QFV_SCOUT_A_RUN_ID", SCOUT_A_RUN_ID),
        attempt_id=os.environ.get("QFV_SCOUT_A_ATTEMPT_ID", SCOUT_A_ATTEMPT_ID),
    )
    raw = json.loads(paths.raw_trace.read_text())
    normalized = json.loads(paths.normalized_trace.read_text())
    manifest = json.loads(paths.runtime_manifest.read_text())

    assert manifest["trainer_class"] == "torchtitan.trainer.Trainer"
    assert manifest["config_manager_class"] == "torchtitan.config.manager.ConfigManager"
    assert raw["profile"]["gradient_accumulation_steps"] == 1
    assert raw["profile"]["tokens_per_optimizer_step"] == 128
    assert raw["lineage"]["first_batch_sha256"]
    assert raw["lineage"]["model_init_sha256"]
    assert [event["kind"] for event in raw["events"]] == [
        "step.started",
        "batch.observed",
        "forward.started",
        "forward.completed",
        "backward.started",
        "gradient.ready",
        "backward.completed",
        "optimizer.started",
        "optimizer.mutated",
        "step.completed",
    ]
    optimizer = next(
        event for event in raw["events"] if event["kind"] == "optimizer.mutated"
    )
    assert optimizer["observation"]["mutated"] is True
    assert normalized["lineage"] == raw["lineage"]
    assert all(event["provenance"] for event in normalized["events"])
    assert not os.access(paths.raw_trace, os.W_OK)
