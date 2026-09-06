# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from experiments.mini_kimi_k3.training_smoke import run_training_smoke


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "experiments" / "mini_kimi_k3" / "run.sh"


def _write_shard(path: Path, tokens: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(tokens, dtype="<u4").tofile(path)


def _write_manifest(path: Path, sources: dict[str, list[str]]) -> Path:
    path.write_text(
        json.dumps(
            {"sources": {name: {"shards": shards} for name, shards in sources.items()}},
            sort_keys=True,
        )
    )
    return path


@pytest.fixture(autouse=True)
def _rootfs_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")


def test_training_smoke_runs_loader_model_backward_and_optimizer(tmp_path: Path):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", list(range(16)))
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    report_path = tmp_path / "smoke.json"

    report = run_training_smoke(
        token_manifest=manifest,
        tokens_dir=tmp_path / "tokens",
        report_path=report_path,
        seq_len=4,
        batch_size=2,
        steps=2,
        seed=123,
    )

    assert report["status"] == "pass"
    assert report["kind"] == "mini_kimi_k3_training_smoke"
    assert report["model"]["variant"] == "tiny"
    assert report["data"]["tokens_emitted"] == 16
    assert report["optimization"]["steps"] == 2
    assert report["optimization"]["optimizer"] == "AdamW"
    assert report["optimization"]["losses"][0] != report["optimization"]["losses"][-1]
    assert report["optimization"]["max_grad_norm"] > 0
    assert json.loads(report_path.read_text()) == report


def test_training_smoke_rejects_tokens_outside_tiny_vocab(tmp_path: Path):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [0, 1, 2, 999])
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})

    with pytest.raises(ValueError, match="token id 999 exceeds tiny vocab"):
        run_training_smoke(
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            report_path=tmp_path / "smoke.json",
            seq_len=4,
            batch_size=1,
            steps=1,
        )


def test_training_smoke_cli_runs_through_runner(tmp_path: Path):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", list(range(16)))
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    report = tmp_path / "smoke.json"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "tiny-smoke",
            "--token-manifest",
            str(manifest),
            "--tokens-dir",
            str(tmp_path / "tokens"),
            "--report",
            str(report),
            "--seq-len",
            "4",
            "--batch-size",
            "2",
            "--steps",
            "1",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    data = json.loads(report.read_text())
    assert data["status"] == "pass"
    assert data["rootfs"]["marker"] == "1"
    assert data["optimization"]["steps"] == 1
