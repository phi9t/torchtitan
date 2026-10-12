# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

from torchtitan.observability.state_estimator.estimators import (
    estimate_belief,
    write_belief_summary,
)
from torchtitan.observability.state_estimator.fixtures import (
    build_minimal_evidence_bundle,
    ProcessFixture,
    write_artifact_index,
)
from torchtitan.observability.state_estimator.transactions import (
    summarize_transaction_state,
)


FORBIDDEN_CONTROL_PHRASES = (
    "safe to commit",
    "valid checkpoint",
    "rollback",
    "should_commit",
)


def _observation(
    observation_id: str,
    *,
    kind: str = "structured_event",
    event_time_ns: int = 1_000,
    payload: dict,
) -> dict:
    return {
        "schema_version": 1,
        "id": observation_id,
        "kind": kind,
        "entity": {"kind": "process", "id": "rank0"},
        "event_time_ns": event_time_ns,
        "ingestion_time_ns": None,
        "source_path": "structured_logs",
        "quality": "present",
        "payload": payload,
    }


def test_transaction_summary_extracts_available_evidence_fields():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [
            _observation(
                "txn:step",
                payload={
                    "record_type": "transaction",
                    "committed_step": 10,
                    "speculative_step": 11,
                    "process_group_epoch": 3,
                    "replica_epoch": 5,
                    "rng_state_id": "rng-step-10",
                    "dataloader_state_id": "loader-step-10",
                    "data_cursor": {
                        "dataset": "c4",
                        "shard": "train-0001",
                        "sample_index": 2048,
                    },
                    "transaction_status": "speculative",
                },
            ),
            _observation(
                "artifact:checkpoint",
                kind="artifact",
                event_time_ns=2_000,
                payload={
                    "record_type": "artifact",
                    "artifact_id": "checkpoint-step-10",
                    "checkpoint_id": "checkpoint-step-10",
                    "kind": "checkpoint",
                    "phase": "checkpoint/save",
                    "step": 10,
                    "checkpoint_parent": "checkpoint-step-9",
                    "checkpoint_shards": [
                        {"rank": 0, "path": "rank0.pt", "size_bytes": 100},
                        {"rank": 1, "path": "rank1.pt", "size_bytes": 110},
                    ],
                    "save_status": "complete",
                    "load_status": "not_attempted",
                    "staging_status": "complete",
                    "restore_validation_status": "validated",
                },
            ),
        ],
    }

    summary = summarize_transaction_state(graph)

    assert summary["state"]["committed_step"] == {
        "value": 10,
        "source_evidence": ["txn:step"],
    }
    assert summary["state"]["speculative_step"] == {
        "value": 11,
        "source_evidence": ["txn:step"],
    }
    assert summary["state"]["process_group_epoch"] == {
        "value": 3,
        "source_evidence": ["txn:step"],
    }
    assert summary["state"]["replica_epoch"] == {
        "value": 5,
        "source_evidence": ["txn:step"],
    }
    assert summary["state"]["checkpoint_parent"] == {
        "value": "checkpoint-step-9",
        "source_evidence": ["artifact:checkpoint"],
    }
    assert summary["state"]["checkpoint_shard_inventory"] == {
        "value": [
            {"rank": 0, "path": "rank0.pt", "size_bytes": 100},
            {"rank": 1, "path": "rank1.pt", "size_bytes": 110},
        ],
        "source_evidence": ["artifact:checkpoint"],
    }
    assert summary["state"]["rng_evidence"] == {
        "value": "rng-step-10",
        "source_evidence": ["txn:step"],
    }
    assert summary["state"]["dataloader_evidence"] == {
        "value": "loader-step-10",
        "source_evidence": ["txn:step"],
    }
    assert summary["state"]["data_cursor"] == {
        "value": {"dataset": "c4", "sample_index": 2048, "shard": "train-0001"},
        "source_evidence": ["txn:step"],
    }
    assert summary["state"]["save_status"] == {
        "value": "complete",
        "source_evidence": ["artifact:checkpoint"],
    }
    assert summary["state"]["load_status"] == {
        "value": "not_attempted",
        "source_evidence": ["artifact:checkpoint"],
    }
    assert summary["state"]["staging_status"] == {
        "value": "complete",
        "source_evidence": ["artifact:checkpoint"],
    }
    assert summary["state"]["restore_validation_status"] == {
        "value": "validated",
        "source_evidence": ["artifact:checkpoint"],
    }
    assert summary["risk_advisories"][0]["kind"] == "enough_evidence_to_assess_risk"
    assert summary["risk_advisories"][0]["advisory"] is True
    assert summary["risk_advisories"][0]["calibration"] == "heuristic_uncalibrated"


def test_missing_transaction_evidence_is_reported_as_advisory_gap():
    summary = summarize_transaction_state(
        {
            "schema_version": 1,
            "run_id": "fixture-run",
            "attempt_id": "attempt-001",
            "entities": [],
            "edges": [],
            "quality": [],
            "observations": [],
        }
    )

    missing = [
        advisory
        for advisory in summary["risk_advisories"]
        if advisory["kind"] == "missing_transaction_evidence"
    ]
    assert missing == [
        {
            "kind": "missing_transaction_evidence",
            "status": "evidence_missing",
            "advisory": True,
            "calibration": "heuristic_uncalibrated",
            "message": (
                "advisory transaction-risk assessment lacks committed step, "
                "speculative step, process-group epoch, replica epoch, "
                "checkpoint parent, checkpoint shard inventory, RNG evidence, "
                "dataloader evidence, data cursor, save status, load status, "
                "staging status, and restore validation status evidence."
            ),
            "missing_fields": [
                "committed_step",
                "speculative_step",
                "process_group_epoch",
                "replica_epoch",
                "checkpoint_parent",
                "checkpoint_shard_inventory",
                "rng_evidence",
                "dataloader_evidence",
                "data_cursor",
                "save_status",
                "load_status",
                "staging_status",
                "restore_validation_status",
            ],
            "source_evidence": [],
        }
    ]


def test_anomaly_before_likely_commit_boundary_is_advisory():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [
            _observation(
                "txn:step", payload={"committed_step": 5, "speculative_step": 6}
            ),
            _observation(
                "anomaly:pre",
                event_time_ns=1_500,
                payload={
                    "record_type": "incident",
                    "event": "nonfinite_loss",
                    "step": 5,
                    "severity": "error",
                },
            ),
        ],
    }

    summary = summarize_transaction_state(graph)

    assert any(
        advisory["kind"] == "anomaly_before_likely_commit_boundary"
        and advisory["source_evidence"] == ["anomaly:pre"]
        for advisory in summary["risk_advisories"]
    )


def test_anomaly_after_likely_commit_boundary_is_advisory():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [
            _observation(
                "txn:step", payload={"committed_step": 5, "speculative_step": 6}
            ),
            _observation(
                "anomaly:post",
                event_time_ns=2_500,
                payload={
                    "record_type": "incident",
                    "event": "rank_death",
                    "step": 6,
                    "severity": "error",
                },
            ),
        ],
    }

    summary = summarize_transaction_state(graph)

    assert any(
        advisory["kind"] == "anomaly_after_likely_commit_boundary"
        and advisory["source_evidence"] == ["anomaly:post"]
        for advisory in summary["risk_advisories"]
    )


def test_checkpoint_lineage_and_restore_validation_gaps_are_distinct():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [
            _observation(
                "artifact:checkpoint",
                kind="artifact",
                payload={
                    "artifact_id": "checkpoint-step-12",
                    "kind": "checkpoint",
                    "checkpoint_shards": [{"rank": 0, "path": "rank0.pt"}],
                    "save_status": "complete",
                },
            )
        ],
    }

    summary = summarize_transaction_state(graph)

    assert any(
        advisory["kind"] == "checkpoint_lineage_incomplete"
        and advisory["missing_fields"] == ["checkpoint_parent"]
        for advisory in summary["risk_advisories"]
    )
    assert any(
        advisory["kind"] == "checkpoint_restore_validation_absent"
        and advisory["missing_fields"] == ["restore_validation_status"]
        for advisory in summary["risk_advisories"]
    )


def test_transaction_report_wording_stays_advisory_without_control_decisions():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [],
    }

    summary = summarize_transaction_state(graph)
    report_text = summary["report_text"]

    assert "advisory" in report_text
    assert "transaction-safety claim" in report_text
    assert all(phrase not in report_text for phrase in FORBIDDEN_CONTROL_PHRASES)


def test_estimator_adds_transactions_additively():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [
            _observation(
                "txn:step", payload={"committed_step": 1, "speculative_step": 2}
            )
        ],
    }

    summary = estimate_belief(graph)

    assert summary["transactions"]["state"]["committed_step"]["value"] == 1
    assert summary["transactions"]["state"]["speculative_step"]["value"] == 2
    assert summary["process_liveness"] == {}
    assert "peer_skew_findings" in summary


def test_write_belief_summary_includes_advisory_transaction_diagnosis(tmp_path):
    process_id = "trainer.core.global_rank_000000"
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id=process_id,
                global_rank=0,
                local_rank=0,
                world_size=1,
                outcome="succeeded",
                phases=[("train/step", 1, 1_000, 2_000)],
            )
        ],
    )
    write_artifact_index(
        attempt_path,
        process_id,
        [
            {
                "schema_version": 1,
                "evidence_schema_version": 1,
                "record_type": "artifact",
                "artifact_id": "checkpoint-step-1",
                "producer": "checkpoint_manager",
                "kind": "checkpoint",
                "relation": "output",
                "state": "complete",
                "phase": "checkpoint/save",
                "step": 1,
                "checkpoint_shards": [{"rank": 0, "path": "rank0.pt"}],
                "save_status": "complete",
                "wall_time_ns": 3_600,
                "monotonic_ns": 3_600,
                "artifact_seq": 1,
                "metadata": {},
                "run_id": "fixture-run",
                "attempt_id": "attempt-001",
                "process_id": process_id,
                "local_rank": 0,
                "world_size": 1,
                "role": "trainer",
            }
        ],
    )

    paths = write_belief_summary(attempt_path)

    summary = json.loads(paths.belief_summary.read_text())
    diagnosis = paths.diagnosis.read_text()
    assert "transactions" in summary
    assert "## Transaction Advisory" in diagnosis
    assert "advisory" in diagnosis
    assert "transaction-safety claim" in diagnosis
    assert all(phrase not in diagnosis for phrase in FORBIDDEN_CONTROL_PHRASES)
