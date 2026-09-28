# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real-GPU acceptance test for the four-rank DPxTP formal-verifier scout."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from torchtitan.experiments.qwen3_formal_verifier.device_mesh import (
    DEVICE_MESH_ATTEMPT_ID,
    DEVICE_MESH_RUN_ID,
    run_device_mesh,
)


pytestmark = pytest.mark.skipif(
    "QFV_DEVICE_MESH_OUTPUT_ROOT" not in os.environ,
    reason="run through experiments/qwen3_formal_verifier/run_device_mesh.sh",
)


def test_real_four_rank_cuda_qwen3_step_reaches_complete_dp_tp_bundle() -> None:
    paths = run_device_mesh(
        Path(os.environ["QFV_DEVICE_MESH_OUTPUT_ROOT"]),
        run_id=os.environ.get("QFV_DEVICE_MESH_RUN_ID", DEVICE_MESH_RUN_ID),
        attempt_id=os.environ.get("QFV_DEVICE_MESH_ATTEMPT_ID", DEVICE_MESH_ATTEMPT_ID),
    )
    normalized = json.loads(paths.normalized_bundle.read_text())
    manifest = json.loads(paths.runtime_manifest.read_text())

    assert manifest["trainer_class"] == "torchtitan.trainer.Trainer"
    assert manifest["config_manager_class"] == "torchtitan.config.manager.ConfigManager"
    assert manifest["cuda_visible_devices"] == "0,1,2,3"
    assert normalized["topology"]["degrees"] == [1, 2, 1, 2, 1, 1]
    assert normalized["ordering"]["cross_rank_total_order"] is False
    assert len(normalized["rank_traces"]) == 4
    assert normalized["input_bundle"]["dp_keys"] == ["dp:0", "dp:1"]
    assert normalized["input_bundle"]["by_dp_coordinate"]["dp:0"]["ranks"] == [
        0,
        1,
    ]
    assert normalized["input_bundle"]["by_dp_coordinate"]["dp:1"]["ranks"] == [
        2,
        3,
    ]
    for rank, rank_trace in enumerate(normalized["rank_traces"]):
        assert rank_trace["process"]["rank"] == rank
        assert rank_trace["placement_summary"]["has_dp_shard"] is True
        assert rank_trace["placement_summary"]["has_tp_shard"] is True
        kinds = [event["kind"] for event in rank_trace["events"]]
        assert "collective.enqueued" in kinds
        assert "collective.started" in kinds
        assert "collective.completed" in kinds
        assert "step.completed" in kinds
        collective_events = [
            event
            for event in rank_trace["events"]
            if event["kind"].startswith("collective.")
        ]
        assert all(event["observation"]["executor"] for event in collective_events)
        assert all(event["observation"]["stream"] for event in collective_events)
        assert all(event["observation"]["producer"] for event in collective_events)
        assert not os.access(paths.rank_raw_traces[rank], os.W_OK)
    assert not os.access(paths.normalized_bundle, os.W_OK)
