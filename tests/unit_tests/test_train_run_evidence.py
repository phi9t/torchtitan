# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core entrypoint lifecycle tests for run evidence."""

import json
from contextlib import contextmanager, ExitStack
from types import SimpleNamespace
from unittest import mock

import pytest
from torchtitan import train

from torchtitan.config import ConfigManager


@pytest.fixture
def launcher_identity(monkeypatch):
    identity = SimpleNamespace(run_id="research-run", attempt_id="launch-17")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("TORCHTITAN_RUN_ID", identity.run_id)
    monkeypatch.setenv("TORCHTITAN_ATTEMPT_ID", identity.attempt_id)
    monkeypatch.delenv("TORCHELASTIC_RESTART_COUNT", raising=False)
    return identity


def make_trainer_config():
    return ConfigManager().parse_args(
        [
            "--module",
            "llama3",
            "--config",
            "llama3_debugmodel",
            "--comm.mode=local_tensor",
        ]
    )


def read_outcome(tmp_path, identity):
    path = (
        tmp_path
        / "run_evidence"
        / identity.run_id
        / identity.attempt_id
        / "processes"
        / "trainer.core.global_rank_000000"
        / "outcome.json"
    )
    return json.loads(path.read_text())


@contextmanager
def patched_main(config):
    manager = mock.Mock()
    manager.parse_args.return_value = config
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(train, "ConfigManager", return_value=manager)
        )
        stack.enter_context(mock.patch.object(train, "init_logger"))
        stack.enter_context(mock.patch.object(train.sl, "init_structured_logger"))
        stack.enter_context(mock.patch.object(train.sl, "log_trace_instant"))
        yield


def test_main_records_success_for_local_tensor_early_return(
    tmp_path, launcher_identity
):
    config = make_trainer_config()
    config.dump_folder = str(tmp_path)

    with patched_main(config):
        train.main()

    assert read_outcome(tmp_path, launcher_identity)["outcome"] == "succeeded"


def test_main_records_failure_when_trainer_build_raises(tmp_path, launcher_identity):
    config = make_trainer_config()
    config.dump_folder = str(tmp_path)
    config.comm.mode = "fake_backend"

    with patched_main(config), mock.patch.object(
        type(config), "build", side_effect=RuntimeError("build failed")
    ):
        with pytest.raises(RuntimeError, match="build failed"):
            train.main()

    outcome = read_outcome(tmp_path, launcher_identity)
    assert outcome["outcome"] == "failed"
    assert outcome["exception_type"] == "RuntimeError"
