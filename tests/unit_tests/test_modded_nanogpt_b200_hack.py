# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from experiments.modded_nanogpt_b200 import hack


def test_hack_import_exposes_notebook_probe_without_initializing_dist():
    assert callable(hack.probe)
    assert callable(hack.launch_jupyter_workers)
    assert callable(hack.shutdown_jupyter_workers)


def test_worker_env_uses_shared_rendezvous_and_rank_specific_local_rank():
    base_env = {"PATH": "/bin", "TORCHTITAN_IN_ROOTFS": "1"}

    env = hack.worker_env(
        rank=3,
        world_size=8,
        master_addr="127.0.0.1",
        master_port="29577",
        base_env=base_env,
    )

    assert env["MASTER_ADDR"] == "127.0.0.1"
    assert env["MASTER_PORT"] == "29577"
    assert env["WORLD_SIZE"] == "8"
    assert env["RANK"] == "3"
    assert env["LOCAL_RANK"] == "3"
    assert env["NCCL_DEBUG"] == "WARN"
    assert env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "1"
    assert env["TORCH_NCCL_BLOCKING_WAIT"] == "1"
    assert env["PATH"] == "/bin"


def test_launch_jupyter_workers_starts_ranks_one_through_world_size(monkeypatch, tmp_path):
    calls = []

    class FakeProcess:
        def __init__(self, rank: int):
            self.rank = rank

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return FakeProcess(rank=int(kwargs["env"]["RANK"]))

    monkeypatch.setattr(hack.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(hack.sys, "executable", "/workspace/torchtitan/.venv-rootfs/bin/python")
    monkeypatch.setattr(hack.Path, "cwd", lambda: Path("/workspace/torchtitan"))

    handle = hack.launch_jupyter_workers(
        world_size=4,
        master_addr="127.0.0.1",
        master_port="29601",
        log_dir=tmp_path,
    )

    assert [proc.rank for proc in handle.processes] == [1, 2, 3]
    assert len(calls) == 3
    for rank, (cmd, kwargs) in enumerate(calls, start=1):
        assert cmd == [
            "/workspace/torchtitan/.venv-rootfs/bin/python",
            "-m",
            "experiments.modded_nanogpt_b200.hack",
            "--worker",
        ]
        assert kwargs["cwd"] == "/workspace/torchtitan"
        assert kwargs["env"]["RANK"] == str(rank)
        assert kwargs["env"]["WORLD_SIZE"] == "4"
        assert kwargs["env"]["MASTER_PORT"] == "29601"
        assert (tmp_path / f"rank{rank}.log").exists()


def test_launch_jupyter_workers_rejects_existing_process_group(monkeypatch, tmp_path):
    monkeypatch.setattr(hack.dist, "is_initialized", lambda: True)

    try:
        hack.launch_jupyter_workers(log_dir=tmp_path)
    except RuntimeError as exc:
        assert "destroy_process_group" in str(exc)
    else:
        raise AssertionError("worker launch accepted an existing process group")


def test_probe_defaults_to_single_rank_when_dist_env_is_absent(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    assert hack.resolve_rank_config(multi_rank=False).world_size == 1


def test_probe_requires_dist_env_for_multi_rank(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    try:
        hack.resolve_rank_config(multi_rank=True)
    except RuntimeError as exc:
        assert "WORLD_SIZE" in str(exc)
    else:
        raise AssertionError("multi-rank probe accepted missing distributed env")
