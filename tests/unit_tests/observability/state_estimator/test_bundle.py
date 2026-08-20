# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from torchtitan.observability.state_estimator.bundle import load_bundle
from torchtitan.observability.state_estimator.fixtures import (
    build_minimal_evidence_bundle,
    ProcessFixture,
)


def test_load_bundle_reads_manifest_artifacts_events_and_outcomes(tmp_path):
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
                phases=[("train/step", 1, 1_000, 2_000)],
            )
        ],
    )

    bundle = load_bundle(attempt_path)

    assert bundle.manifest["run_id"] == "fixture-run"
    assert len(bundle.artifact_rows) == 1
    assert len(bundle.event_rows) == 1
    assert len(bundle.outcome_rows) == 1
    assert bundle.quality_findings == []


def test_load_bundle_fails_without_manifest(tmp_path):
    with pytest.raises(ValueError, match="manifest.json is required"):
        load_bundle(tmp_path)
