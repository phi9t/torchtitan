# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core entrypoint lifecycle tests for run evidence."""

import json
import logging
from contextlib import contextmanager, ExitStack
from types import SimpleNamespace
from unittest import mock

import pytest
from torchtitan import train

from torchtitan.config import ConfigManager
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.structured_logger.jsonl_handler import TraceJsonlHandler


def attach_jsonl_handler_factory(*, structured_logger, rank, source, output_dir, **kw):
    """Attach the native handler before a later configured factory fails."""
    structured_logger.addHandler(
        TraceJsonlHandler(rank=rank, source=source, output_dir=output_dir)
    )


def fail_handler_factory(**kw):
    """Simulate a later configured structured logging factory failure."""
    raise RuntimeError("structured handler factory failed")


class FailingCloseHandler(logging.Handler):
    """Purpose-built handler which fails only while closing."""

    def __init__(self):
        super().__init__()
        self.close_attempts = 0

    def emit(self, record):
        pass

    def close(self):
        self.close_attempts += 1
        super().close()
        raise OSError("structured handler close failed")


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


def read_artifact_rows(tmp_path, identity):
    index = next(
        (
            tmp_path
            / "run_evidence"
            / identity.run_id
            / identity.attempt_id
            / "indexes"
        ).glob("artifacts.*.jsonl")
    )
    return [json.loads(line) for line in index.read_text().splitlines()]


@pytest.fixture
def structured_logger_lifecycle():
    """Isolate the process-global structured logger for real entrypoint tests."""
    import torchtitan.observability.structured_logger.structured_logging as sl_mod

    def reset():
        for handler in sl_mod._structured_logger.handlers[:]:
            sl_mod._structured_logger.removeHandler(handler)
            try:
                handler.close()
            except OSError:
                pass
        sl_mod._is_initialized = False
        sl_mod._disabled = False

    reset()
    yield sl_mod
    reset()


@contextmanager
def patched_main(config, *, stub_structured_logging=True):
    manager = mock.Mock()
    manager.parse_args.return_value = config
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(train, "ConfigManager", return_value=manager)
        )
        stack.enter_context(mock.patch.object(train, "init_logger"))
        if stub_structured_logging:
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


def test_main_preserves_training_error_when_structured_handler_close_fails(
    tmp_path, launcher_identity, structured_logger_lifecycle
):
    config = make_trainer_config()
    config.dump_folder = str(tmp_path)
    config.comm.mode = "fake_backend"
    failing_handler = FailingCloseHandler()

    def fail_training_build():
        structured_logger_lifecycle._structured_logger.addHandler(failing_handler)
        raise RuntimeError("training failed")

    with patched_main(config, stub_structured_logging=False), mock.patch.object(
        type(config), "build", side_effect=fail_training_build
    ):
        with pytest.raises(RuntimeError, match="training failed"):
            train.main()

    assert failing_handler.close_attempts == 1
    assert structured_logger_lifecycle._structured_logger.handlers == []
    assert structured_logger_lifecycle._is_initialized is False
    assert structured_logger_lifecycle._disabled is False
    assert (
        read_outcome(tmp_path, launcher_identity)["exception_message"]
        == "training failed"
    )

    sl.init_structured_logger(rank=0, source="training", output_dir=str(tmp_path))
    assert len(structured_logger_lifecycle._structured_logger.handlers) == 1
    sl.close_structured_logger()


def test_main_closes_partial_structured_factory_initialization(
    tmp_path, launcher_identity, monkeypatch, structured_logger_lifecycle
):
    config = make_trainer_config()
    config.dump_folder = str(tmp_path)
    monkeypatch.setenv(
        "TITAN_STRUCT_LOGGER_HANDLERS",
        f"{__name__}.attach_jsonl_handler_factory,{__name__}.fail_handler_factory",
    )

    with patched_main(config, stub_structured_logging=False):
        with pytest.raises(RuntimeError, match="structured handler factory failed"):
            train.main()

    assert structured_logger_lifecycle._structured_logger.handlers == []
    assert structured_logger_lifecycle._is_initialized is False
    assert [
        row["state"] for row in read_artifact_rows(tmp_path, launcher_identity)
    ] == [
        "declared",
        "complete",
    ]
    assert read_outcome(tmp_path, launcher_identity)["outcome"] == "failed"

    monkeypatch.delenv("TITAN_STRUCT_LOGGER_HANDLERS")
    sl.init_structured_logger(rank=0, source="training", output_dir=str(tmp_path))
    assert len(structured_logger_lifecycle._structured_logger.handlers) == 1
    sl.close_structured_logger()
