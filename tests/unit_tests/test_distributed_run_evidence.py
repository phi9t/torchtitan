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


@pytest.fixture
def relative_active_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("TORCHTITAN_RUN_ID", "distributed-relative-run")
    monkeypatch.setenv("TORCHTITAN_ATTEMPT_ID", "distributed-relative-attempt")
    with RunEvidence(
        RunEvidence.Config(),
        dump_folder="outputs",
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


def test_flight_recorder_uses_relative_native_prefix_and_normalized_evidence_path(
    relative_active_evidence, monkeypatch
):
    """A relative dump folder must not be duplicated in the indexed prefix."""
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

    _configure_flight_recorder(config, "outputs")

    assert os.environ["TORCH_FR_DUMP_TEMP_FILE"] == "outputs/comm_traces/rank_"
    rows = artifact_rows(relative_active_evidence, kind="pytorch.flight_recorder.dump")
    assert [row["state"] for row in rows] == ["declared"]
    assert rows[0]["path"] == "comm_traces/rank_"
    assert rows[0]["path_type"] == "dump_relative"


@pytest.mark.parametrize("configured_prefix", ("/absolute/rank_", "nested/rank_"))
def test_flight_recorder_preserves_legacy_prefix_string(
    relative_active_evidence, monkeypatch, configured_prefix
):
    """Configured prefixes retain the exact legacy string while evidence resolves it."""
    config = CommConfig(
        trace_buf_size=32,
        save_traces_folder="comm_traces",
        save_traces_file_prefix=configured_prefix,
    )
    base_folder = "outputs"
    for name in (
        "TORCH_FR_BUFFER_SIZE",
        "TORCH_NCCL_DUMP_ON_TIMEOUT",
        "TORCH_FR_DUMP_TEMP_FILE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    ):
        monkeypatch.delenv(name, raising=False)

    _configure_flight_recorder(config, base_folder)

    dump_dir = os.path.join(base_folder, config.save_traces_folder)
    native_prefix = f"{dump_dir}/{configured_prefix}"
    assert os.environ["TORCH_FR_DUMP_TEMP_FILE"] == native_prefix
    rows = artifact_rows(relative_active_evidence, kind="pytorch.flight_recorder.dump")
    assert (
        rows[0]["path"]
        == Path(native_prefix)
        .resolve()
        .relative_to(Path(base_folder).resolve())
        .as_posix()
    )
    assert rows[0]["path_type"] == "dump_relative"


def test_disabled_flight_recorder_declares_no_evidence(tmp_path, active_evidence):
    """A disabled buffer does not have a dump prefix to index."""
    _configure_flight_recorder(
        CommConfig(trace_buf_size=0),
        str(tmp_path),
    )

    assert artifact_rows(active_evidence, kind="pytorch.flight_recorder.dump") == []
