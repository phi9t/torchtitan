# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchtitan.observability.state_estimator.bundle import load_bundle
from torchtitan.observability.state_estimator.data_collection import (
    build_data_collection_plan,
)
from torchtitan.observability.state_estimator.fixtures import (
    ProcessFixture,
    build_minimal_evidence_bundle,
)


def test_data_collection_plan_marks_required_raw_signals_present(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
            )
        ],
    )

    plan = build_data_collection_plan(load_bundle(attempt_path)).to_json()

    states = {signal["name"]: signal["state"] for signal in plan["signals"]}
    assert states["manifest"] == "present"
    assert states["process_outcome"] == "present"
    assert states["artifact_index"] == "present"
    assert states["structured_events"] == "present"


def test_data_collection_plan_marks_uncollected_optional_hardware(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
            )
        ],
    )

    plan = build_data_collection_plan(load_bundle(attempt_path)).to_json()

    dcgm = next(signal for signal in plan["signals"] if signal["name"] == "dcgm")
    assert dcgm["state"] == "uncollected"
    assert dcgm["required"] is False
