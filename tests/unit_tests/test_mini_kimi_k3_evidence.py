# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

import pytest

from experiments.mini_kimi_k3.evidence import (
    initialize_launch_evidence_bundle,
    inspect_launch_evidence_bundle,
)


def test_initialize_launch_evidence_bundle_writes_attempt_manifest(tmp_path):
    bundle_dir = initialize_launch_evidence_bundle(
        results_root=tmp_path,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )

    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["kind"] == "attempt_manifest"
    assert manifest["run"]["run_id"] == "mini-k3-r1-test"
    assert manifest["run"]["family"] == "mini_kimi_k3"
    assert manifest["run"]["task"] == "pretraining"
    assert manifest["run"]["lane"] == "full"
    assert manifest["run"]["fields"]["model_variant"] == "r1"
    assert manifest["run"]["fields"]["status"] == "initialized_not_trainable"
    assert manifest["attempt"]["attempt_id"] == "attempt-001"


def test_initialize_launch_evidence_bundle_is_immutable(tmp_path):
    kwargs = {
        "results_root": tmp_path,
        "run_id": "mini-k3-r1-test",
        "attempt_id": "attempt-001",
        "mode": "full",
    }
    initialize_launch_evidence_bundle(**kwargs)

    with pytest.raises(FileExistsError):
        initialize_launch_evidence_bundle(**kwargs)


def test_inspect_launch_evidence_bundle_reports_manifest_identity(tmp_path):
    initialize_launch_evidence_bundle(
        results_root=tmp_path,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )

    evidence = inspect_launch_evidence_bundle(
        results_root=tmp_path,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
    )

    assert evidence == {
        "manifest": str(
            tmp_path / "runs" / "mini-k3-r1-test" / "attempt-001" / "manifest.json"
        ),
        "run_id": "mini-k3-r1-test",
        "attempt_id": "attempt-001",
        "family": "mini_kimi_k3",
        "task": "pretraining",
        "lane": "full",
        "model_variant": "r1",
    }
