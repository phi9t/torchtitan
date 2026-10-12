# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchtitan.observability.state_estimator.schema import (
    DerivedPaths,
    EntityRef,
    QualityFinding,
    read_json_object,
    write_json_atomic,
)


def test_quality_finding_serializes_stable_fields(tmp_path):
    finding = QualityFinding(
        kind="missing_optional_artifact",
        severity="warning",
        evidence_state="missing",
        message="structured event stream was not present",
        source_path="indexes/artifacts.trainer.core.global_rank_000000.jsonl",
        entity=EntityRef(kind="process", id="trainer.core.global_rank_000000"),
        metadata={"producer": "structured_logger"},
    )

    assert finding.to_json() == {
        "schema_version": 1,
        "kind": "missing_optional_artifact",
        "severity": "warning",
        "evidence_state": "missing",
        "message": "structured event stream was not present",
        "source_path": "indexes/artifacts.trainer.core.global_rank_000000.jsonl",
        "entity": {"kind": "process", "id": "trainer.core.global_rank_000000"},
        "metadata": {"producer": "structured_logger"},
    }


def test_atomic_json_write_uses_canonical_json(tmp_path):
    output = tmp_path / "nested" / "artifact.json"

    write_json_atomic(output, {"b": 2, "a": 1})

    assert output.read_text() == '{"a":1,"b":2}\n'
    assert read_json_object(output) == {"a": 1, "b": 2}
    assert not output.with_suffix(".json.tmp").exists()


def test_derived_paths_are_under_state_estimator_directory(tmp_path):
    paths = DerivedPaths.from_attempt_path(tmp_path)

    assert paths.root == tmp_path / "derived" / "state_estimator"
    assert paths.evidence_graph == paths.root / "evidence_graph.json"
    assert paths.entities == paths.root / "entities.json"
    assert paths.timeline == paths.root / "timeline.jsonl"
    assert paths.quality == paths.root / "quality.json"
    assert paths.belief_summary == paths.root / "belief_summary.json"
    assert paths.diagnosis == paths.root / "diagnosis.md"
