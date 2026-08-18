# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

from torchtitan.observability.state_estimator.fixtures import (
    ProcessFixture,
    build_minimal_evidence_bundle,
)


def test_fixture_builds_single_process_success_bundle(tmp_path):
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
                outcome="succeeded",
            )
        ],
    )

    manifest = json.loads((attempt_path / "manifest.json").read_text())
    outcome = json.loads(
        (
            attempt_path
            / "processes"
            / "trainer.core.global_rank_000000"
            / "outcome.json"
        ).read_text()
    )
    indexes = list((attempt_path / "indexes").glob("artifacts.*.jsonl"))

    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "fixture-run"
    assert manifest["attempt_id"] == "attempt-001"
    assert outcome["record_type"] == "process_outcome"
    assert outcome["outcome"] == "succeeded"
    assert len(indexes) == 1


def test_fixture_builds_two_rank_phase_events(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=2,
                phases=[("train/step", 1, 1_000, 2_000)],
            ),
            ProcessFixture(
                process_id="trainer.core.global_rank_000001",
                global_rank=1,
                local_rank=1,
                world_size=2,
                phases=[("train/step", 1, 1_500, 3_000)],
            ),
        ],
    )

    event_paths = sorted((attempt_path / "structured_logs").glob("*.jsonl"))
    assert len(event_paths) == 2
    rank_one_rows = [
        json.loads(line) for line in event_paths[1].read_text().splitlines()
    ]
    assert rank_one_rows[0]["phase"] == "train/step"
    assert rank_one_rows[0]["step"] == 1
    assert rank_one_rows[0]["global_rank"] == 1
