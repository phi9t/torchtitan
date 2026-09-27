# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for classifying a failed collective into a v1 fault class.

The assertions that matter most are the ones proving the classifier declines:
a signature table that matched everything would label a NCCL abort or an
unrelated RuntimeError as a collective hang, and a wrong class sends a reader
to the wrong subsystem.

The two multi-process tests inject the faults for real -- one rank that stays
alive and never issues, and one that exits -- because the whole difficulty of
these two classes is that a surviving rank only ever sees them indirectly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from torchtitan.observability.distributed_faults import (
    classify_collective_failure,
    detect_step_stragglers,
    record_collective_failure,
    UNCLASSIFIED,
)
from torchtitan.observability.run_evidence import FaultConfidence, IncidentClass

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)

# Verbatim first lines observed from Gloo on four ranks with a 4 s timeout.
_HANG_TEXT = (
    "[../gloo/transport/tcp/unbound_buffer.cc:78] Timed out waiting 4000ms "
    "for recv operation to complete"
)
_DEATH_TEXT = (
    "[../gloo/transport/tcp/pair.cc:553] Connection closed by peer [127.0.0.1]:4"
)


def test_a_true_hang_signature_classifies_as_collective_hang() -> None:
    incident_class, confidence, signature = classify_collective_failure(
        RuntimeError(_HANG_TEXT)
    )
    assert incident_class is IncidentClass.COLLECTIVE_HANG
    # The ambiguity is recorded rather than resolved: a timeout says a peer did
    # not arrive, not which peer or why.
    assert confidence is FaultConfidence.COLLECTIVE_TIMEOUT_INSUFFICIENT_EVIDENCE
    assert signature == "timed out waiting"


def test_a_closed_peer_signature_classifies_as_rank_death() -> None:
    incident_class, confidence, signature = classify_collective_failure(
        RuntimeError(_DEATH_TEXT)
    )
    assert incident_class is IncidentClass.RANK_DEATH
    assert confidence is FaultConfidence.SUSPECTED_PEER_FAULT
    assert signature == "connection closed by peer"


@pytest.mark.parametrize(
    "text",
    [
        "CUDA error: device-side assert triggered",
        "NCCL watchdog thread terminated with exception",
        "Socket Timeout",
        "",
    ],
)
def test_an_unrecognized_error_is_not_classified(text: str) -> None:
    """Declining is the point: a wrong class is worse than no class.

    NCCL is deliberately among these. Its failures go through a different path
    with different text, and guessing "collective hang" from a NCCL watchdog
    message would attribute a fault the signature table never measured.
    """
    incident_class, confidence, signature = classify_collective_failure(
        RuntimeError(text)
    )
    assert incident_class is None
    assert confidence is None
    assert signature == UNCLASSIFIED


def test_recording_an_unclassified_failure_records_nothing() -> None:
    assert record_collective_failure(RuntimeError("something else entirely")) is None


_FAULT_CHILD = """
import datetime, os, sys, time
import torch
import torch.distributed as dist
from torchtitan.observability.distributed_faults import record_collective_failure
from torchtitan.observability.run_evidence import RunEvidence

rank = int(os.environ["RANK"])
mode = os.environ["QFV_FAULT_MODE"]
faulty = 1

with RunEvidence(
    RunEvidence.Config(),
    dump_folder=os.environ["QFV_DUMP"],
    job_config={"training": {"steps": 1}},
    role="trainer",
    actor_id="core",
):
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=4))
    if rank == faulty:
        if mode == "hang":
            # Alive, but never issues the collective the others are waiting on.
            time.sleep(25)
        else:
            # Exits, closing its sockets: rank death, not a hang.
            os._exit(0)
        sys.stdout.write("FAULTY_DONE")
    else:
        try:
            dist.all_reduce(torch.ones(4))
            sys.stdout.write("COMPLETED")
        except BaseException as error:
            row = record_collective_failure(
                error, step=1, last_operation="all_reduce"
            )
            sys.stdout.write(
                "RECORDED " + (row["incident_class"] if row else "NOTHING")
            )
"""


def _incidents_for(dump, run_id, attempt_id, rank):
    index = os.path.join(
        str(dump),
        "run_evidence",
        run_id,
        attempt_id,
        "indexes",
        f"artifacts.trainer.core.global_rank_{rank:06d}.jsonl",
    )
    if not os.path.isfile(index):
        return []
    with open(index, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [row for row in rows if row["record_type"] == "incident"]


def _run_fault(tmp_path, mode, port):
    child = tmp_path / "child.py"
    child.write_text(_FAULT_CHILD, encoding="utf-8")
    run_id, attempt_id = "fault-run", f"attempt-{mode}"
    world_size = 4
    processes = []
    for rank in range(world_size):
        env = os.environ.copy()
        env.update(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "TORCHTITAN_RUN_ID": run_id,
                "TORCHTITAN_ATTEMPT_ID": attempt_id,
                "QFV_DUMP": str(tmp_path),
                "QFV_FAULT_MODE": mode,
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": os.pathsep.join(
                    [_REPO_ROOT, env.get("PYTHONPATH", "")]
                ).rstrip(os.pathsep),
            }
        )
        env.pop("TORCHELASTIC_RESTART_COUNT", None)
        processes.append(
            subprocess.Popen(
                [sys.executable, str(child)],
                cwd=_REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
    outcomes = {}
    for rank, process in enumerate(processes):
        try:
            stdout, stderr = process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        outcomes[rank] = (stdout, stderr)
    return outcomes, run_id, attempt_id


def test_an_injected_collective_hang_is_recorded_by_every_survivor(
    tmp_path, free_tcp_port
) -> None:
    """A rank that stays alive and never issues must be seen by the others.

    Outcome is deterministic even though latency is not: the survivors always
    reach the timeout. That distinction is what makes this injectable at all.
    """
    outcomes, run_id, attempt_id = _run_fault(tmp_path, "hang", free_tcp_port)

    for rank in (0, 2, 3):
        stdout = outcomes[rank][0]
        assert "RECORDED collective_hang" in stdout, (
            rank,
            stdout,
            outcomes[rank][1][-500:],
        )
        incidents = _incidents_for(tmp_path, run_id, attempt_id, rank)
        assert len(incidents) == 1, (rank, incidents)
        recorded = incidents[0]
        assert recorded["incident_class"] == "collective_hang"
        assert (
            recorded["attribution_confidence"]
            == "collective_timeout_insufficient_evidence"
        )
        assert recorded["metadata"]["matched_signature"] == "timed out waiting"
        assert recorded["metadata"]["backend"] == "gloo"
        assert recorded["last_operation"] == "all_reduce"


def test_an_injected_rank_death_is_recorded_as_rank_death(
    tmp_path, free_tcp_port
) -> None:
    """A rank that exits produces a different signature, and must not be a hang.

    This is the pair that makes the classifier worth having: the same
    RuntimeError type, the same collective, and two different fault classes.
    """
    outcomes, run_id, attempt_id = _run_fault(tmp_path, "death", free_tcp_port)

    classified = []
    for rank in (0, 2, 3):
        incidents = _incidents_for(tmp_path, run_id, attempt_id, rank)
        classified.extend(row["incident_class"] for row in incidents)

    assert classified, outcomes
    assert set(classified) == {"rank_death"}, (classified, outcomes)
    assert "collective_hang" not in classified


def test_a_slow_rank_is_flagged_against_the_median() -> None:
    assert detect_step_stragglers(
        [100, 104, 98, 400],
        incident_class=IncidentClass.COMPUTE_STRAGGLER,
        slowdown_threshold=2.0,
    ) == (3,)


def test_uniform_durations_flag_nothing() -> None:
    assert (
        detect_step_stragglers(
            [100, 104, 98, 101],
            incident_class=IncidentClass.COMPUTE_STRAGGLER,
            slowdown_threshold=2.0,
        )
        == ()
    )


def test_the_baseline_is_a_median_so_one_outlier_cannot_hide_itself() -> None:
    """The discriminating test: a mean-based baseline would miss this.

    With durations [100, 100, 100, 1000] the mean is 325, so the outlier's ratio
    against it is 3.1 and a 4x threshold would let it through -- the slow rank
    drags the very baseline it is being compared to. The median is 100, so the
    ratio is 10. Any implementation that swapped median for mean passes every
    other test in this file and fails this one.
    """
    assert detect_step_stragglers(
        [100, 100, 100, 1000],
        incident_class=IncidentClass.COMPUTE_STRAGGLER,
        slowdown_threshold=4.0,
    ) == (3,)


def test_fewer_than_three_ranks_is_declined_rather_than_guessed() -> None:
    """Two samples cannot separate "the slow one" from "the fast one"."""
    for durations in ([100], [100, 1000]):
        assert (
            detect_step_stragglers(
                durations,
                incident_class=IncidentClass.COMPUTE_STRAGGLER,
                slowdown_threshold=2.0,
            )
            == ()
        )


def test_a_threshold_at_or_below_one_is_refused() -> None:
    for threshold in (1.0, 0.5):
        with pytest.raises(ValueError, match="must exceed 1.0"):
            detect_step_stragglers(
                [100, 100, 100],
                incident_class=IncidentClass.COMPUTE_STRAGGLER,
                slowdown_threshold=threshold,
            )


def test_a_non_straggler_class_is_a_programmer_error() -> None:
    with pytest.raises(AssertionError, match="not a straggler class"):
        detect_step_stragglers(
            [100, 100, 400],
            incident_class=IncidentClass.NONFINITE_LOSS,
            slowdown_threshold=2.0,
        )


def test_a_dataloader_straggler_is_attributed_to_the_dataloader(
    tmp_path, monkeypatch
) -> None:
    """The two straggler classes must not share a locus.

    Reporting a dataloader straggler against the trainer rank sends a reader to
    the wrong subsystem, which is the same defect as a mislabelled collective.
    """
    from torchtitan.observability.run_evidence import RunEvidence

    run_id, attempt_id = "straggler-locus-run", "attempt-0"
    for name, value in (
        ("WORLD_SIZE", "1"),
        ("RANK", "0"),
        ("LOCAL_RANK", "0"),
        ("TORCHTITAN_RUN_ID", run_id),
        ("TORCHTITAN_ATTEMPT_ID", attempt_id),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("TORCHELASTIC_RESTART_COUNT", raising=False)

    with RunEvidence(
        RunEvidence.Config(),
        dump_folder=str(tmp_path),
        job_config={"training": {"steps": 1}},
        role="trainer",
        actor_id="core",
    ):
        assert detect_step_stragglers(
            [100, 100, 100, 500],
            incident_class=IncidentClass.DATALOADER_STRAGGLER,
            slowdown_threshold=2.0,
            step=7,
            phase="data_loading",
        ) == (3,)

    incidents = _incidents_for(tmp_path, run_id, attempt_id, 0)
    assert len(incidents) == 1, incidents
    recorded = incidents[0]
    assert recorded["incident_class"] == "dataloader_straggler"
    assert recorded["detected_locus"] == "dataloader"
    # A straggler is a performance fault: the step was right, just late.
    assert recorded["policy"] == "continue_bounded_warning"
    assert recorded["useful_work_preserved"] is True
    assert recorded["metadata"]["straggler_ranks"] == [3]
    assert recorded["metadata"]["observed_ratios"] == [5.0]
    assert recorded["step"] == 7


_STRAGGLER_CHILD = """
import datetime, os, sys, time
import torch
import torch.distributed as dist
from torchtitan.observability.distributed_faults import detect_step_stragglers
from torchtitan.observability.run_evidence import IncidentClass, RunEvidence

rank = int(os.environ["RANK"])
slow = int(os.environ["QFV_SLOW_RANK"])

with RunEvidence(
    RunEvidence.Config(),
    dump_folder=os.environ["QFV_DUMP"],
    job_config={"training": {"steps": 1}},
    role="trainer",
    actor_id="core",
):
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=60))
    started = time.monotonic_ns()
    if rank == slow:
        time.sleep(1.0)          # the injected straggler, a fixed delay
    else:
        time.sleep(0.05)
    elapsed = time.monotonic_ns() - started

    # Every rank learns every rank's duration, then decides locally.
    gathered = [torch.zeros(1, dtype=torch.int64) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, torch.tensor([elapsed], dtype=torch.int64))
    durations = [int(t.item()) for t in gathered]

    flagged = detect_step_stragglers(
        durations,
        incident_class=IncidentClass.COMPUTE_STRAGGLER,
        slowdown_threshold=4.0,
        step=1,
        phase="forward_backward",
    )
    sys.stdout.write("FLAGGED " + ",".join(str(r) for r in flagged))
    dist.destroy_process_group()
"""


def test_an_injected_straggler_is_flagged_by_every_rank(
    tmp_path, free_tcp_port
) -> None:
    """A fixed delay on one rank must be identified by all of them, as that rank.

    Deterministic by construction: 1.0 s against 0.05 s is twenty times the
    baseline, far outside scheduling noise, so the 4x threshold is not a coin
    flip. Every rank gathers the same duration vector and so must reach the same
    verdict -- disagreement between ranks would make the detection useless for
    attribution.
    """
    child = tmp_path / "child.py"
    child.write_text(_STRAGGLER_CHILD, encoding="utf-8")
    run_id, attempt_id, slow_rank, world_size = "straggler-run", "attempt-0", 2, 4

    processes = []
    for rank in range(world_size):
        env = os.environ.copy()
        env.update(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(free_tcp_port),
                "TORCHTITAN_RUN_ID": run_id,
                "TORCHTITAN_ATTEMPT_ID": attempt_id,
                "QFV_DUMP": str(tmp_path),
                "QFV_SLOW_RANK": str(slow_rank),
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": os.pathsep.join(
                    [_REPO_ROOT, env.get("PYTHONPATH", "")]
                ).rstrip(os.pathsep),
            }
        )
        env.pop("TORCHELASTIC_RESTART_COUNT", None)
        processes.append(
            subprocess.Popen(
                [sys.executable, str(child)],
                cwd=_REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )

    outcomes = {}
    for rank, process in enumerate(processes):
        stdout, stderr = process.communicate(timeout=300)
        outcomes[rank] = (process.returncode, stdout, stderr)

    for rank, (code, stdout, stderr) in outcomes.items():
        assert code == 0, (rank, stdout, stderr[-500:])
        # Unanimous, and naming the rank that was actually delayed.
        assert stdout.strip() == f"FLAGGED {slow_rank}", (rank, stdout)
        incidents = _incidents_for(tmp_path, run_id, attempt_id, rank)
        assert len(incidents) == 1, (rank, incidents)
        assert incidents[0]["incident_class"] == "compute_straggler"
        assert incidents[0]["metadata"]["straggler_ranks"] == [slow_rank]
