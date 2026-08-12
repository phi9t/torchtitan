# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flight Recorder native artifact lifecycle tests."""

import json
import os
from pathlib import Path

import pytest
from torchtitan.config import CommConfig
from torchtitan.distributed.utils import _configure_flight_recorder
from torchtitan.observability.run_evidence import RunEvidence


@pytest.fixture
def active_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("TORCHTITAN_RUN_ID", "distributed-evidence-run")
    monkeypatch.setenv("TORCHTITAN_ATTEMPT_ID", "distributed-evidence-attempt")
    with RunEvidence(
        RunEvidence.Config(),
        dump_folder=str(tmp_path),
        job_config={"training": {"steps": 8}},
        role="trainer",
        actor_id="core",
    ) as evidence:
        yield evidence


def artifact_rows(active_evidence, *, kind):
    index_path = next(
        (
            Path(active_evidence.dump_folder)
            / "run_evidence"
            / active_evidence.run_id
            / active_evidence.attempt_id
            / "indexes"
        ).glob("artifacts.*.jsonl")
    )
    return [
        row
        for row in (json.loads(line) for line in index_path.read_text().splitlines())
        if row["kind"] == kind
    ]


def test_flight_recorder_prefix_is_declared(tmp_path, active_evidence, monkeypatch):
    """A missing declaration would hide failure-triggered dump prefixes from evidence."""
    config = CommConfig(
        trace_buf_size=32,
        save_traces_folder="comm_traces",
        save_traces_file_prefix="rank_",
    )
    for name in (
        "TORCH_FR_BUFFER_SIZE",
        "TORCH_NCCL_DUMP_ON_TIMEOUT",
        "TORCH_FR_DUMP_TEMP_FILE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    ):
        monkeypatch.delenv(name, raising=False)

    _configure_flight_recorder(config, str(tmp_path))

    assert os.environ["TORCH_FR_BUFFER_SIZE"] == "32"
    assert os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "1"
    assert os.environ["TORCH_FR_DUMP_TEMP_FILE"] == str(
        tmp_path / "comm_traces" / "rank_"
    )
    assert os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "3"
    rows = artifact_rows(active_evidence, kind="pytorch.flight_recorder.dump")
    assert [row["state"] for row in rows] == ["declared"]
    assert rows[0]["producer"] == "pytorch_flight_recorder"
    assert rows[0]["metadata"]["path_semantics"] == "prefix"


def test_disabled_flight_recorder_declares_no_evidence(tmp_path, active_evidence):
    """A disabled buffer does not have a dump prefix to index."""
    _configure_flight_recorder(
        CommConfig(trace_buf_size=0),
        str(tmp_path),
    )

    assert artifact_rows(active_evidence, kind="pytorch.flight_recorder.dump") == []
