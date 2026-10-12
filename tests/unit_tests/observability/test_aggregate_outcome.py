# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Behavior tests for post-hoc multi-process outcome aggregation."""

import json

import pytest

from torchtitan.observability.aggregate_outcome import (
    main,
    reduce_attempt_outcome,
    write_attempt_outcome,
)
from torchtitan.observability.run_evidence import (
    EvidenceCollisionError,
    EvidenceContractError,
)


def write_manifest(attempt_dir, run_id="research-run", attempt_id="launch-17"):
    attempt_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "attempt_id": attempt_id,
    }
    (attempt_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return manifest


def write_process_outcome(
    attempt_dir,
    process_id,
    *,
    outcome,
    global_rank,
    world_size=2,
    elapsed_monotonic_ns=1000,
    exception_type=None,
):
    process_dir = attempt_dir / "processes" / process_id
    process_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "evidence_schema_version": 1,
        "record_type": "process_outcome",
        "outcome": outcome,
        "elapsed_monotonic_ns": elapsed_monotonic_ns,
        "run_id": "research-run",
        "attempt_id": "launch-17",
        "process_id": process_id,
        "role": "trainer",
        "actor_id": "core",
        "global_rank": global_rank,
        "local_rank": global_rank,
        "world_size": world_size,
    }
    if exception_type is not None:
        record["exception_type"] = exception_type
        record["exception_message"] = "boom"
    (process_dir / "outcome.json").write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    )


def test_reduce_reports_success_only_when_all_expected_processes_succeed(tmp_path):
    write_manifest(tmp_path)
    for rank in range(2):
        write_process_outcome(
            tmp_path,
            f"trainer.core.global_rank_{rank:06d}",
            outcome="succeeded",
            global_rank=rank,
        )

    aggregate = reduce_attempt_outcome(tmp_path)

    assert aggregate["record_type"] == "attempt_outcome_aggregate"
    assert aggregate["schema_version"] == 1
    assert aggregate["run_id"] == "research-run"
    assert aggregate["attempt_id"] == "launch-17"
    assert aggregate["expected_world_size"] == 2
    assert aggregate["process_outcome_count"] == 2
    assert aggregate["missing_process_count"] == 0
    assert aggregate["aggregate_outcome"] == "succeeded"
    assert aggregate["first_failure"] is None
    assert aggregate["consistent_world_size"] is True
    assert aggregate["reduced_wall_time_ns"] == 1000
    assert [entry["global_rank"] for entry in aggregate["outcomes"]] == [0, 1]


def test_reduce_reports_failure_and_lowest_failed_rank_as_first_failure(tmp_path):
    write_manifest(tmp_path)
    write_process_outcome(
        tmp_path,
        "trainer.core.global_rank_000000",
        outcome="succeeded",
        global_rank=0,
        elapsed_monotonic_ns=500,
    )
    write_process_outcome(
        tmp_path,
        "trainer.core.global_rank_000001",
        outcome="failed",
        global_rank=1,
        exception_type="RuntimeError",
        elapsed_monotonic_ns=2000,
    )

    aggregate = reduce_attempt_outcome(tmp_path)

    assert aggregate["aggregate_outcome"] == "failed"
    assert aggregate["first_failure"] == 1
    # reduced wall time is the longest observed elapsed time.
    assert aggregate["reduced_wall_time_ns"] == 2000


def test_reduce_prefers_failure_over_interruption(tmp_path):
    write_manifest(tmp_path)
    write_process_outcome(
        tmp_path,
        "trainer.core.global_rank_000000",
        outcome="interrupted",
        global_rank=0,
    )
    write_process_outcome(
        tmp_path,
        "trainer.core.global_rank_000001",
        outcome="failed",
        global_rank=1,
        exception_type="RuntimeError",
    )

    aggregate = reduce_attempt_outcome(tmp_path)

    assert aggregate["aggregate_outcome"] == "failed"
    assert aggregate["first_failure"] == 1


def test_reduce_reports_interrupted_when_any_interrupted_and_none_failed(tmp_path):
    write_manifest(tmp_path)
    write_process_outcome(
        tmp_path,
        "trainer.core.global_rank_000000",
        outcome="succeeded",
        global_rank=0,
    )
    write_process_outcome(
        tmp_path,
        "trainer.core.global_rank_000001",
        outcome="interrupted",
        global_rank=1,
    )

    aggregate = reduce_attempt_outcome(tmp_path)

    assert aggregate["aggregate_outcome"] == "interrupted"
    assert aggregate["first_failure"] is None


def test_reduce_reports_incomplete_when_a_process_outcome_is_missing(tmp_path):
    write_manifest(tmp_path)
    # Only one of two expected processes published an outcome.
    write_process_outcome(
        tmp_path,
        "trainer.core.global_rank_000000",
        outcome="succeeded",
        global_rank=0,
    )

    aggregate = reduce_attempt_outcome(tmp_path)

    # Missing is not success.
    assert aggregate["aggregate_outcome"] == "incomplete"
    assert aggregate["process_outcome_count"] == 1
    assert aggregate["missing_process_count"] == 1
    assert aggregate["first_failure"] is None


def test_reduce_flags_inconsistent_world_size(tmp_path):
    write_manifest(tmp_path)
    write_process_outcome(
        tmp_path,
        "trainer.core.global_rank_000000",
        outcome="succeeded",
        global_rank=0,
        world_size=2,
    )
    write_process_outcome(
        tmp_path,
        "trainer.core.global_rank_000001",
        outcome="succeeded",
        global_rank=1,
        world_size=3,
    )

    aggregate = reduce_attempt_outcome(tmp_path)

    assert aggregate["consistent_world_size"] is False
    # An inconsistent world size cannot certify success.
    assert aggregate["aggregate_outcome"] == "incomplete"


def test_write_attempt_outcome_is_immutable_without_force(tmp_path):
    write_manifest(tmp_path)
    write_process_outcome(
        tmp_path,
        "trainer.core.global_rank_000000",
        outcome="succeeded",
        global_rank=0,
        world_size=1,
    )

    output_path = write_attempt_outcome(tmp_path)
    assert output_path == tmp_path / "aggregate_outcome.json"
    original = output_path.read_bytes()

    with pytest.raises(EvidenceCollisionError, match="aggregate"):
        write_attempt_outcome(tmp_path)
    assert output_path.read_bytes() == original

    # --force replaces the aggregate atomically.
    replaced = write_attempt_outcome(tmp_path, force=True)
    assert replaced == output_path
    aggregate = json.loads(output_path.read_text())
    assert aggregate["aggregate_outcome"] == "succeeded"


def test_main_prints_the_written_aggregate_path(tmp_path, capsys):
    write_manifest(tmp_path)
    write_process_outcome(
        tmp_path,
        "trainer.core.global_rank_000000",
        outcome="succeeded",
        global_rank=0,
        world_size=1,
    )

    exit_code = main([str(tmp_path)])

    assert exit_code == 0
    printed = capsys.readouterr().out.strip()
    assert printed == str(tmp_path / "aggregate_outcome.json")
    assert (tmp_path / "aggregate_outcome.json").exists()


def test_reduce_reports_incomplete_when_no_process_outcome_is_present(tmp_path):
    write_manifest(tmp_path)

    aggregate = reduce_attempt_outcome(tmp_path)

    # No authoritative world size exists in the manifest, so a total loss is
    # incomplete with a None expected size rather than succeeded.
    assert aggregate["aggregate_outcome"] == "incomplete"
    assert aggregate["expected_world_size"] is None
    assert aggregate["process_outcome_count"] == 0
    assert aggregate["missing_process_count"] == 0
    assert aggregate["reduced_wall_time_ns"] == 0


def test_reduce_rejects_malformed_process_outcome(tmp_path):
    write_manifest(tmp_path)
    process_dir = tmp_path / "processes" / "trainer.core.global_rank_000000"
    process_dir.mkdir(parents=True)
    (process_dir / "outcome.json").write_text("{not valid json")

    with pytest.raises(EvidenceContractError, match="not valid JSON"):
        reduce_attempt_outcome(tmp_path)
