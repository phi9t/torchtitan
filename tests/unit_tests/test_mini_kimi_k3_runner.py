# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.mini_kimi_k3 import (
    build_decontamination_index,
    completion_audit,
    corpus_disk_readiness,
    launch,
    materialize_stage1_local,
    r1_training_smoke,
    run_stage1_remote,
    stage1_remote_readiness,
)
from experiments.mini_kimi_k3.dump_torchtitan_logits import (
    align_first_party_state_dict,
    align_first_party_weights,
)
from experiments.mini_kimi_k3.evidence import initialize_launch_evidence_bundle
from torch import nn
from torchtitan.experiments.execution.executor import CommandResult


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "experiments" / "mini_kimi_k3" / "run.sh"


def test_mini_kimi_k3_runner_fails_closed_inside_rootfs():
    env = {**os.environ, "TORCHTITAN_IN_ROOTFS": "1"}

    proc = subprocess.run(
        ["bash", str(RUNNER), "--mode", "tiny-plumbing"],
        cwd=Path("/tmp"),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "Mini Kimi K3 runner requires an explicit command" in proc.stdout
    assert "Use preflight before launch" in proc.stdout
    assert "Traceback" not in proc.stdout


def test_mini_kimi_k3_runner_initializes_evidence_bundle_inside_rootfs(tmp_path: Path):
    results_root = tmp_path / "results"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "init-evidence",
            "--results-root",
            str(results_root),
            "--run-id",
            "mini-k3-r1-test",
            "--attempt-id",
            "attempt-001",
            "--mode",
            "full",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0
    assert (
        results_root / "runs" / "mini-k3-r1-test" / "attempt-001" / "manifest.json"
    ).is_file()


def test_mini_kimi_k3_runner_initializes_full_manifest_template_inside_rootfs(
    tmp_path: Path,
):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "init-manifest",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--source",
            "fineweb-edu",
            "--dataset-id",
            "HuggingFaceFW/fineweb-edu",
            "--snapshot",
            "2026-08-fixture",
            "--license",
            "odc-by",
            "--source-provenance-report",
            "reports/fineweb-edu-provenance.json",
            "--source-provenance-sha256",
            "source-provenance-sha",
            "--decontamination-report",
            "reports/decontamination.json",
            "--decontamination-sha256",
            "decontam-sha",
            "--tokenizer-sha256",
            "tokenizer-sha",
            "--tokenizer-asset-report",
            "assets/kimi-k3-tokenizer.json",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    assert tokens_dir.is_dir()
    data = json.loads(manifest.read_text())
    assert data["schema_version"] == 1
    assert data["kind"] == "mini_kimi_k3_token_manifest"
    assert data["mode"] == "full"
    assert data["tokenizer"] == {
        "name": "Kimi K3",
        "vocab_size": 163_840,
        "sha256": "tokenizer-sha",
        "asset_report": "assets/kimi-k3-tokenizer.json",
    }
    assert data["decontamination"] == {
        "status": "completed",
        "report": "reports/decontamination.json",
        "sha256": "decontam-sha",
    }
    assert data["sources"] == {
        "fineweb-edu": {
            "dataset_id": "HuggingFaceFW/fineweb-edu",
            "provenance": {
                "snapshot": "2026-08-fixture",
                "license": "odc-by",
                "report": "reports/fineweb-edu-provenance.json",
                "sha256": "source-provenance-sha",
            },
            "shards": [],
        }
    }


def test_mini_kimi_k3_runner_checks_r1_smoke_gpu_readiness_inside_rootfs(
    tmp_path: Path,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "printf '0, NVIDIA B200, 183359, 110000, 73359\\n'\n"
        "printf '1, NVIDIA B200, 183359, 180000, 3359\\n'\n"
    )
    nvidia_smi.chmod(0o755)
    report = tmp_path / "gpu-readiness.json"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "r1-smoke-gpu-readiness",
            "--report",
            str(report),
            "--min-free-gpu-memory-mib",
            "120000",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "TORCHTITAN_IN_ROOTFS": "1",
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "r1 smoke GPU readiness: blocked" in proc.stdout
    data = json.loads(report.read_text())
    assert data["kind"] == "mini_kimi_k3_r1_smoke_gpu_readiness"
    assert data["status"] == "blocked"
    assert data["required_free_mib"] == 120_000
    assert data["selected"] is None
    assert data["devices"][0]["memory_free_mib"] == 73_359


def test_mini_kimi_k3_runner_installs_tokenizer_assets_inside_rootfs(tmp_path: Path):
    source = tmp_path / "source-tokenizer"
    output = tmp_path / "assets" / "kimi-k3-tokenizer"
    report = tmp_path / "tokenizer_report.json"
    source.mkdir()
    files = {
        "config.json": b'{"model_type":"kimi_k3"}\n',
        "tokenizer_config.json": b'{"tokenizer_class":"KimiTokenizer"}\n',
        "generation_config.json": b"{}\n",
        "tokenization_kimi.py": b"# tokenizer\n",
        "encoding_k3.py": b"# encoding\n",
        "tiktoken.model": b"fixture tokens\n",
    }
    for name, payload in files.items():
        (source / name).write_bytes(payload)

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "install-tokenizer-assets",
            "--source-tokenizer-dir",
            str(source),
            "--output-dir",
            str(output),
            "--fingerprint-report",
            str(report),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    assert sorted(path.name for path in output.iterdir()) == sorted(files)
    data = json.loads(report.read_text())
    assert data["kind"] == "mini_kimi_k3_tokenizer_assets"
    assert data["schema_version"] == 1
    assert data["output_dir"] == str(output)
    assert sorted(data["files"]) == sorted(files)
    for name, payload in files.items():
        assert data["files"][name]["bytes"] == len(payload)
        assert data["files"][name]["sha256"] == hashlib.sha256(payload).hexdigest()
    aggregate = hashlib.sha256()
    for name in sorted(files):
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(files[name])
    assert data["sha256"] == aggregate.hexdigest()


def test_mini_kimi_k3_runner_writes_stage1_input_manifest_inside_rootfs(
    tmp_path: Path,
):
    first = tmp_path / "part-00000.jsonl"
    second = tmp_path / "part-00001.jsonl"
    first.write_text(json.dumps({"tokens": [1, 2]}) + "\n")
    second.write_text(json.dumps({"tokens": [3, 4]}) + "\n")
    output = tmp_path / "source-inputs.json"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "write-stage1-input-manifest",
            "--output",
            str(output),
            "--input-format",
            "jsonl-tokens",
            "--tokens-field",
            "tokens",
            "--input",
            str(first),
            "--input",
            str(second),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    data = json.loads(output.read_text())
    assert data["kind"] == "mini_kimi_k3_source_input_manifest"
    assert data["schema_version"] == 1
    assert data["input_format"] == "jsonl-tokens"
    assert data["tokens_field"] == "tokens"
    assert "text_field" not in data
    assert data["inputs"] == [
        {
            "path": str(first.resolve()),
            "bytes": first.stat().st_size,
            "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
        },
        {
            "path": str(second.resolve()),
            "bytes": second.stat().st_size,
            "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
        },
    ]


def test_mini_kimi_k3_runner_plans_r1_corpus_from_first_party_mix(
    tmp_path: Path,
):
    mix_plan = tmp_path / "mix_plan.json"
    mix_plan.write_text(
        json.dumps(
            {
                "total_tokens": 55_000_000_000,
                "source_tokens": {
                    "fineweb-edu": 29_260_000_000,
                    "code-python": 6_737_500_000,
                },
                "files": {
                    "fineweb-edu": 39,
                    "code-python": 446,
                },
                "tokens_per_file": {
                    "fineweb-edu": 755_008_927.2,
                    "code-python": 15_127_250.5,
                },
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "r1-corpus-plan.json"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "plan-r1-corpus",
            "--mix-plan",
            str(mix_plan),
            "--report",
            str(report),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    data = json.loads(report.read_text())
    assert data["kind"] == "mini_kimi_k3_r1_corpus_plan"
    assert data["status"] == "planned"
    assert data["target_tokens"] == 5_000_000_000
    assert data["source_tokens"]["fineweb-edu"] == 2_660_000_000
    assert data["source_tokens"]["code-python"] == 612_500_000
    assert data["total_planned_tokens"] == 3_272_500_000
    assert data["sources"]["fineweb-edu"]["estimated_files"] == 4
    assert data["sources"]["code-python"]["estimated_files"] == 41


def test_mini_kimi_k3_runner_audits_r1_corpus_plan_deficits(tmp_path: Path):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    plan = tmp_path / "reports" / "r1-corpus-plan.json"
    report = tmp_path / "reports" / "r1-corpus-plan-audit.json"
    fineweb_dir = tokens_dir / "fineweb-edu"
    fineweb_dir.mkdir(parents=True)
    shard = fineweb_dir / "part-00000.bin"
    np.asarray([1, 2, 3, 4], dtype="<u4").tofile(shard)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_token_manifest",
                "sources": {
                    "fineweb-edu": {
                        "shards": [shard.name],
                        "shard_metadata": {
                            shard.name: {
                                "bytes": shard.stat().st_size,
                                "num_tokens": 4,
                                "sha256": hashlib.sha256(
                                    shard.read_bytes()
                                ).hexdigest(),
                            }
                        },
                    }
                },
            },
            sort_keys=True,
        )
    )
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan",
                "status": "planned",
                "target_tokens": 7,
                "total_planned_tokens": 7,
                "sources": {
                    "fineweb-edu": {"target_tokens": 4},
                    "code-python": {"target_tokens": 3},
                },
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "audit-r1-corpus-plan",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--corpus-plan",
            str(plan),
            "--report",
            str(report),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "r1 corpus plan audit: blocked" in proc.stdout
    data = json.loads(report.read_text())
    assert data["kind"] == "mini_kimi_k3_r1_corpus_plan_audit"
    assert data["schema_version"] == 1
    assert data["status"] == "blocked"
    assert data["target_tokens"] == 7
    assert data["available_tokens"] == 4
    assert data["deficit_tokens"] == 3
    assert data["sources"]["fineweb-edu"] == {
        "required_tokens": 4,
        "available_tokens": 4,
        "deficit_tokens": 0,
        "status": "ready",
    }
    assert data["sources"]["code-python"] == {
        "required_tokens": 3,
        "available_tokens": 0,
        "deficit_tokens": 3,
        "status": "missing",
    }


def test_mini_kimi_k3_corpus_disk_readiness_blocks_when_space_is_short(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    plan = tmp_path / "reports" / "r1-corpus-plan.json"
    report = tmp_path / "reports" / "corpus-disk-readiness.json"
    target = tmp_path / "data" / "tokens"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan",
                "status": "planned",
                "target_tokens": 1_000,
                "total_planned_tokens": 1_000,
                "sources": {"fineweb-edu": {"target_tokens": 1_000}},
            },
            sort_keys=True,
        )
    )

    result = corpus_disk_readiness.check_corpus_disk_readiness(
        corpus_plan_path=plan,
        target_path=target,
        report_path=report,
        overhead_fraction=0.25,
        disk_usage_fn=lambda _path: shutil._ntuple_diskusage(
            total=10_000,
            used=7_500,
            free=2_500,
        ),
    )

    assert result["status"] == "blocked"
    assert result["target_tokens"] == 1_000
    assert result["required_raw_bytes"] == 4_000
    assert result["overhead_fraction"] == 0.25
    assert result["required_bytes"] == 5_000
    assert result["free_bytes"] == 2_500
    assert result["deficit_bytes"] == 2_500
    assert result["path"] == str(target)
    assert result["corpus_plan"]["path"] == str(plan)
    assert json.loads(report.read_text()) == result


def test_mini_kimi_k3_runner_checks_corpus_disk_readiness_inside_rootfs(
    tmp_path: Path,
):
    plan = tmp_path / "reports" / "r1-corpus-plan.json"
    report = tmp_path / "reports" / "corpus-disk-readiness.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan",
                "status": "planned",
                "target_tokens": 1_000,
                "total_planned_tokens": 1_000,
                "sources": {"fineweb-edu": {"target_tokens": 1_000}},
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "corpus-disk-readiness",
            "--corpus-plan",
            str(plan),
            "--target-path",
            str(tmp_path / "data" / "tokens"),
            "--report",
            str(report),
            "--overhead-fraction",
            "0.20",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    assert "corpus disk readiness: ready" in proc.stdout
    data = json.loads(report.read_text())
    assert data["kind"] == "mini_kimi_k3_corpus_disk_readiness"
    assert data["schema_version"] == 1
    assert data["status"] == "ready"
    assert data["required_raw_bytes"] == 4_000
    assert data["required_bytes"] == 4_800
    assert data["deficit_bytes"] == 0
    assert data["free_bytes"] >= data["required_bytes"]


def test_mini_kimi_k3_runner_probe_stage1_sources_summarizes_resolved_json(
    tmp_path: Path,
):
    source_root = tmp_path / "first-party"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    resolved = data_dir / "resolved.json"
    resolved.write_text(
        json.dumps(
            [
                {
                    "kind": "benchmark",
                    "name": "mmlu",
                    "chosen": {"repo": "cais/mmlu", "config": "all", "split": "test"},
                    "tried": [],
                },
                {
                    "kind": "corpus",
                    "name": "fineweb-edu",
                    "chosen": {
                        "repo": "HuggingFaceFW/fineweb-edu",
                        "config": "sample-100BT",
                        "split": "train",
                        "text_field": "text",
                        "path_prefix": "sample/100BT/",
                        "n_files": 140,
                    },
                    "tried": [],
                    "role": "web",
                },
                {
                    "kind": "corpus",
                    "name": "web-diverse",
                    "chosen": None,
                    "tried": [
                        {
                            "repo": "nvidia/Nemotron-CC-v2",
                            "status": "ok",
                            "gated": True,
                        }
                    ],
                    "role": "web",
                },
            ],
            sort_keys=True,
        )
    )
    report = tmp_path / "reports" / "stage1-source-resolution.json"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "probe-stage1-sources",
            "--source-root",
            str(source_root),
            "--report",
            str(report),
            "--skip-probe",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "stage1 source probe: blocked" in proc.stdout
    data = json.loads(report.read_text())
    assert data["kind"] == "mini_kimi_k3_stage1_source_resolution"
    assert data["schema_version"] == 1
    assert data["status"] == "blocked"
    assert data["unresolved_benchmarks"] == []
    assert data["unresolved_corpora"] == ["web-diverse"]
    assert data["sources"]["fineweb-edu"] == {
        "status": "resolved",
        "repo": "HuggingFaceFW/fineweb-edu",
        "config": "sample-100BT",
        "split": "train",
        "text_field": "text",
        "path_prefix": "sample/100BT/",
        "filter_field": None,
        "filter_values": None,
        "n_files": 140,
        "role": "web",
    }
    assert data["sources"]["web-diverse"]["status"] == "unresolved"
    assert data["sources"]["web-diverse"]["tried"][0]["repo"] == "nvidia/Nemotron-CC-v2"


def test_mini_kimi_k3_stage1_remote_readiness_blocks_without_modal_auth(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    report = tmp_path / "reports" / "stage1-remote-readiness.json"

    class FakeReadinessExecutor:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(
            self,
            argv: list[str],
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> CommandResult:
            del cwd
            del env
            self.calls.append(list(argv))
            if argv == ["getent", "hosts", "huggingface.co"]:
                return CommandResult(return_code=0, stdout="1.2.3.4 huggingface.co\n")
            if argv == ["uv", "tool", "run", "--from", "modal", "modal", "--version"]:
                return CommandResult(
                    return_code=0, stdout="modal client version: 1.5.4\n"
                )
            if argv == [
                "uv",
                "tool",
                "run",
                "--from",
                "modal",
                "modal",
                "token",
                "info",
            ]:
                return CommandResult(
                    return_code=1,
                    stdout="Token missing. Could not authenticate client.\n",
                )
            raise AssertionError(f"unexpected command: {argv}")

    result = stage1_remote_readiness.check_stage1_remote_readiness(
        report_path=report,
        executor=FakeReadinessExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["checks"]["huggingface_dns"]["status"] == "pass"
    assert result["checks"]["modal_cli"]["status"] == "pass"
    assert result["checks"]["modal_auth"]["status"] == "blocked"
    assert result["checks"]["modal_auth"]["detail"] == "Token missing"
    assert result["checks"]["modal_auth"]["env"] == {
        "MODAL_CONFIG_PATH": "unset",
        "MODAL_PROFILE": "unset",
        "MODAL_TOKEN_ID": "unset",
        "MODAL_TOKEN_SECRET": "unset",
    }
    assert result["checks"]["modal_auth"]["config_locations"] == [
        {"exists": False, "path": str(tmp_path / "home" / ".modal.toml")},
        {"exists": False, "path": str(tmp_path / "home" / ".modal" / "config.toml")},
        {"exists": False, "path": str(tmp_path / "xdg" / "modal.toml")},
        {"exists": False, "path": str(tmp_path / "xdg" / "modal" / "config.toml")},
    ]
    assert json.loads(report.read_text()) == result


def test_mini_kimi_k3_stage1_remote_readiness_reports_modal_config_path(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    modal_config = tmp_path / "modal.toml"
    modal_config.write_text("[default]\n")
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(modal_config))
    report = tmp_path / "reports" / "stage1-remote-readiness.json"

    class FakeReadinessExecutor:
        def run(
            self,
            argv: list[str],
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> CommandResult:
            del cwd
            del env
            if argv == ["getent", "hosts", "huggingface.co"]:
                return CommandResult(return_code=0, stdout="1.2.3.4 huggingface.co\n")
            if argv == ["uv", "tool", "run", "--from", "modal", "modal", "--version"]:
                return CommandResult(
                    return_code=0, stdout="modal client version: 1.5.4\n"
                )
            if argv == [
                "uv",
                "tool",
                "run",
                "--from",
                "modal",
                "modal",
                "token",
                "info",
            ]:
                return CommandResult(
                    return_code=1,
                    stdout="Token missing. Could not authenticate client.\n",
                )
            raise AssertionError(f"unexpected command: {argv}")

    result = stage1_remote_readiness.check_stage1_remote_readiness(
        report_path=report,
        executor=FakeReadinessExecutor(),
    )

    assert result["checks"]["modal_auth"]["env"]["MODAL_CONFIG_PATH"] == "set"
    assert result["checks"]["modal_auth"]["config_locations"][0] == {
        "exists": True,
        "path": str(modal_config),
        "source": "MODAL_CONFIG_PATH",
    }
    assert json.loads(report.read_text()) == result


def test_mini_kimi_k3_stage1_remote_readiness_loads_modal_env_file(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    monkeypatch.setenv("TORCHTITAN_ROOTFS_NETWORK", "networked")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "MODAL_TOKEN_ID=ak-test-id",
                "MODAL_TOKEN_SECRET='sk-test-secret'",
                "MODAL_PROFILE=mini-k3",
                "",
            ]
        )
    )
    report = tmp_path / "reports" / "stage1-remote-readiness.json"

    class FakeReadinessExecutor:
        def __init__(self) -> None:
            self.env_by_call: list[dict[str, str] | None] = []

        def run(
            self,
            argv: list[str],
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> CommandResult:
            del cwd
            self.env_by_call.append(dict(env) if env is not None else None)
            if argv == ["getent", "hosts", "huggingface.co"]:
                return CommandResult(return_code=0, stdout="1.2.3.4 huggingface.co\n")
            if argv == ["uv", "tool", "run", "--from", "modal", "modal", "--version"]:
                return CommandResult(
                    return_code=0, stdout="modal client version: 1.5.4\n"
                )
            if argv == [
                "uv",
                "tool",
                "run",
                "--from",
                "modal",
                "modal",
                "token",
                "info",
            ]:
                return CommandResult(
                    return_code=0,
                    stdout="profile = mini-k3 ak-test-id sk-test-secret\n",
                )
            raise AssertionError(f"unexpected command: {argv}")

    executor = FakeReadinessExecutor()

    result = stage1_remote_readiness.check_stage1_remote_readiness(
        report_path=report,
        env_file=env_file,
        executor=executor,
    )

    assert result["status"] == "ready"
    assert result["checks"]["modal_auth"]["env"] == {
        "MODAL_CONFIG_PATH": "unset",
        "MODAL_PROFILE": "set",
        "MODAL_TOKEN_ID": "set",
        "MODAL_TOKEN_SECRET": "set",
    }
    assert result["env_file"] == {
        "path": str(env_file),
        "status": "loaded",
        "keys": ["MODAL_PROFILE", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"],
    }
    assert executor.env_by_call[-1]["MODAL_TOKEN_ID"] == "ak-test-id"
    assert executor.env_by_call[-1]["MODAL_TOKEN_SECRET"] == "sk-test-secret"
    report_text = report.read_text()
    assert "ak-test-id" not in report_text
    assert "sk-test-secret" not in report_text
    assert json.loads(report_text) == result


def test_mini_kimi_k3_stage1_remote_readiness_blocks_without_networked_rootfs(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    monkeypatch.delenv("TORCHTITAN_ROOTFS_NETWORK", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    report = tmp_path / "reports" / "stage1-remote-readiness.json"

    class FakeReadinessExecutor:
        def run(
            self,
            argv: list[str],
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> CommandResult:
            del cwd
            del env
            if argv == ["getent", "hosts", "huggingface.co"]:
                return CommandResult(return_code=0, stdout="1.2.3.4 huggingface.co\n")
            if argv == ["uv", "tool", "run", "--from", "modal", "modal", "--version"]:
                return CommandResult(
                    return_code=0, stdout="modal client version: 1.5.4\n"
                )
            if argv == [
                "uv",
                "tool",
                "run",
                "--from",
                "modal",
                "modal",
                "token",
                "info",
            ]:
                return CommandResult(return_code=0, stdout="profile = mini-k3\n")
            raise AssertionError(f"unexpected command: {argv}")

    result = stage1_remote_readiness.check_stage1_remote_readiness(
        report_path=report,
        executor=FakeReadinessExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["checks"]["rootfs_network"]["status"] == "blocked"
    assert result["checks"]["rootfs_network"]["network_mode"] == "offline"
    assert result["rootfs"]["marker"] == "1"
    assert result["rootfs"]["network_mode"] == "offline"
    assert json.loads(report.read_text()) == result


def test_mini_kimi_k3_stage1_remote_env_template_is_parseable(monkeypatch):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    template = REPO_ROOT / ".scratch" / "mini-kimi-k3-replication" / ".env.example"

    command_env, report = stage1_remote_readiness.build_command_env(template)

    assert report == {
        "path": str(template),
        "status": "loaded",
        "keys": ["MODAL_PROFILE", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"],
    }
    assert command_env["MODAL_PROFILE"] == "mini-k3"
    assert command_env["MODAL_TOKEN_ID"] == "replace-with-modal-token-id"
    assert command_env["MODAL_TOKEN_SECRET"] == "replace-with-modal-token-secret"


def test_mini_kimi_k3_stage1_remote_readiness_reports_missing_env_file(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    report = tmp_path / "reports" / "stage1-remote-readiness.json"
    missing_env_file = tmp_path / "missing.env"

    result = stage1_remote_readiness.main(
        [
            "--env-file",
            str(missing_env_file),
            "--report",
            str(report),
        ]
    )

    assert result == 21
    data = json.loads(report.read_text())
    assert data["status"] == "blocked"
    assert data["checks"]["env_file"] == {
        "status": "blocked",
        "path": str(missing_env_file),
        "detail": f"env file does not exist: {missing_env_file}",
        "template": ".scratch/mini-kimi-k3-replication/.env.example",
        "required_keys": [
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "MODAL_PROFILE",
        ],
    }
    assert data["rootfs"]["marker"] == "1"


def test_mini_kimi_k3_stage1_remote_readiness_blocks_placeholder_env_file(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    monkeypatch.setenv("PATH", "")
    report = tmp_path / "reports" / "stage1-remote-readiness.json"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "MODAL_TOKEN_ID=replace-with-modal-token-id",
                "MODAL_TOKEN_SECRET=replace-with-modal-token-secret",
                "",
            ]
        )
    )

    result = stage1_remote_readiness.main(
        [
            "--env-file",
            str(env_file),
            "--report",
            str(report),
        ]
    )

    assert result == 21
    data = json.loads(report.read_text())
    assert data["status"] == "blocked"
    assert data["checks"]["env_file"] == {
        "status": "blocked",
        "path": str(env_file),
        "detail": (
            "env file is missing required values: MODAL_PROFILE; "
            "env file still contains placeholder values: MODAL_TOKEN_ID, "
            "MODAL_TOKEN_SECRET"
        ),
        "template": ".scratch/mini-kimi-k3-replication/.env.example",
        "required_keys": [
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "MODAL_PROFILE",
        ],
    }
    assert data["rootfs"]["marker"] == "1"


def test_mini_kimi_k3_completion_audit_rejects_blocked_preflight_and_attempt(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    preflight_report = tmp_path / "preflight.json"
    audit_report = tmp_path / "completion-audit.json"
    bundle = initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )
    (bundle / "outcome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "attempt_outcome",
                "attempt_id": "attempt-001",
                "execution_outcome": "blocked",
                "stage_invocation_ids": ["inv-preflight"],
                "evaluations": {
                    "preflight": {
                        "execution_outcome": "blocked",
                        "measurement": "not_run",
                        "promotion": "not_evaluated",
                    }
                },
                "run_gate": {"has_real_measurement": False},
            },
            sort_keys=True,
        )
    )
    (bundle / "processes" / "coordinator").mkdir(parents=True)
    (bundle / "processes" / "coordinator" / "events.jsonl").write_text(
        json.dumps(
            {
                "kind": "stage_blocked",
                "stage_id": "preflight",
                "stage_invocation_id": "inv-preflight",
                "return_code": 21,
            },
            sort_keys=True,
        )
        + "\n"
    )
    preflight_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_preflight",
                "mode": "full",
                "status": "blocked",
                "gates": [
                    {
                        "name": "trainable_config",
                        "status": "fail",
                        "evidence": {
                            "r1_training_smoke": {
                                "report": "results/r1_training_smoke.json",
                                "status": "blocked",
                                "steps": 0,
                                "gpu_memory": {
                                    "status": "blocked",
                                    "required_free_mib": 12_000,
                                    "selected": None,
                                },
                            }
                        },
                    },
                    {"name": "model_fidelity", "status": "pass"},
                    {
                        "name": "real_corpus_manifest",
                        "status": "fail",
                        "evidence": {
                            "corpus_route": "local_or_imported",
                            "requires_stage1_input_inspection": True,
                            "r1_corpus_plan_audit": {
                                "status": "blocked",
                                "target_tokens": 5_000_000_000,
                                "available_tokens": 4097,
                                "deficit_tokens": 4_999_995_903,
                                "sources": {
                                    "fineweb-edu": {
                                        "required_tokens": 2_660_000_000,
                                        "available_tokens": 4097,
                                        "deficit_tokens": 2_659_995_903,
                                        "status": "short",
                                    },
                                    "code-python": {
                                        "required_tokens": 612_500_000,
                                        "available_tokens": 0,
                                        "deficit_tokens": 612_500_000,
                                        "status": "missing",
                                    },
                                    "ready-source": {
                                        "required_tokens": 1,
                                        "available_tokens": 1,
                                        "deficit_tokens": 0,
                                        "status": "ready",
                                    },
                                },
                            },
                            "corpus_disk_readiness": {
                                "report": "reports/corpus-disk-readiness.json",
                                "status": "ready",
                                "target_tokens": 5_000_000_000,
                                "required_bytes": 24_000_000_000,
                                "free_bytes": 48_000_000_000,
                                "deficit_bytes": 0,
                            },
                            "stage1_remote_readiness": {
                                "report": "reports/stage1-remote-readiness.json",
                                "status": "blocked",
                                "env_file": {
                                    "status": "blocked",
                                    "path": ".scratch/mini-kimi-k3-replication/.env",
                                    "detail": (
                                        "env file does not exist: "
                                        ".scratch/mini-kimi-k3-replication/.env"
                                    ),
                                    "template": (
                                        ".scratch/mini-kimi-k3-replication/.env.example"
                                    ),
                                    "required_keys": [
                                        "MODAL_TOKEN_ID",
                                        "MODAL_TOKEN_SECRET",
                                        "MODAL_PROFILE",
                                    ],
                                },
                                "huggingface_dns": "pass",
                                "modal_cli": "pass",
                                "modal_auth": "blocked",
                                "network_mode": "networked",
                            },
                        },
                    },
                    {
                        "name": "launch_evidence_bundle",
                        "status": "pass",
                        "evidence": {
                            "manifest": str(bundle / "manifest.json"),
                            "run_id": "mini-k3-r1-test",
                            "attempt_id": "attempt-001",
                            "family": "mini_kimi_k3",
                            "task": "pretraining",
                            "lane": "full",
                            "model_variant": "r1",
                        },
                    },
                ],
            },
            sort_keys=True,
        )
    )

    result = completion_audit.run_completion_audit(
        report_path=audit_report,
        preflight_report=preflight_report,
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
    )

    assert result["status"] == "blocked"
    assert result["success_criteria"] == [
        "full_preflight_ready",
        "r1_smoke_gpu_capacity",
        "r1_trainer_optimizer_step",
        "model_fidelity",
        "real_5b_corpus_manifest",
        "r1_corpus_plan_audit",
        "raw_shard_disk_capacity",
        "local_stage1_input_inspection",
        "stage1_source_resolution",
        "stage1_remote_readiness",
        "launch_evidence_bundle",
        "guarded_launch_completed",
        "training_measurement_real",
        "train_stage_succeeded",
    ]
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["full_preflight_ready"]["status"] == "fail"
    assert checks["r1_smoke_gpu_capacity_ready"]["status"] == "fail"
    assert checks["local_stage1_input_inspection_ready"]["status"] == "fail"
    assert checks["stage1_remote_readiness_ready"]["status"] == "fail"
    assert checks["guarded_launch_completed"]["status"] == "fail"
    assert checks["train_stage_succeeded"]["status"] == "fail"
    checklist = {
        item["requirement"]: item for item in result["prompt_to_artifact_checklist"]
    }
    assert checklist["full_preflight_ready"]["evidence"] == str(preflight_report)
    assert checklist["full_preflight_ready"]["status"] == "missing"
    assert checklist["r1_smoke_gpu_capacity"]["status"] == "missing"
    assert checklist["r1_trainer_optimizer_step"]["status"] == "missing"
    assert checklist["model_fidelity"]["status"] == "covered"
    assert checklist["real_5b_corpus_manifest"]["status"] == "missing"
    assert checklist["r1_corpus_plan_audit"]["status"] == "missing"
    assert checklist["raw_shard_disk_capacity"]["status"] == "covered"
    assert checklist["raw_shard_disk_capacity"]["evidence"] == str(preflight_report)
    assert checklist["local_stage1_input_inspection"]["status"] == "missing"
    assert checklist["stage1_source_resolution"]["status"] == "missing"
    assert checklist["stage1_remote_readiness"]["status"] == "missing"
    assert checklist["guarded_launch_completed"]["evidence"] == str(
        bundle / "outcome.json"
    )
    assert checklist["train_stage_succeeded"]["evidence"] == str(
        bundle / "processes" / "coordinator" / "events.jsonl"
    )
    assert result["missing"] == [
        "full_preflight_ready",
        "r1_smoke_gpu_capacity",
        "r1_trainer_optimizer_step",
        "real_5b_corpus_manifest",
        "r1_corpus_plan_audit",
        "local_stage1_input_inspection",
        "stage1_source_resolution",
        "stage1_remote_readiness",
        "guarded_launch_completed",
        "training_measurement_real",
        "train_stage_succeeded",
    ]
    assert result["covered"] == [
        "model_fidelity",
        "raw_shard_disk_capacity",
        "launch_evidence_bundle",
    ]
    assert result["not_required"] == []
    next_actions = {item["requirement"]: item for item in result["next_actions"]}
    assert next_actions["full_preflight_ready"]["evidence"] == str(preflight_report)
    assert next_actions["full_preflight_ready"]["guidance"] == {
        "blocked_preflight_requirements": [
            "r1_smoke_gpu_capacity",
            "r1_trainer_optimizer_step",
            "real_5b_corpus_manifest",
            "r1_corpus_plan_audit",
            "local_stage1_input_inspection",
            "stage1_source_resolution",
            "stage1_remote_readiness",
        ]
    }
    assert next_actions["r1_smoke_gpu_capacity"]["action"]
    assert next_actions["r1_trainer_optimizer_step"]["action"]
    assert next_actions["r1_smoke_gpu_capacity"]["guidance"] == {
        "r1_training_smoke_report": "results/r1_training_smoke.json",
        "required_free_mib": 12_000,
        "selected": None,
        "status": "blocked",
    }
    assert next_actions["r1_trainer_optimizer_step"]["guidance"] == {
        "r1_training_smoke_report": "results/r1_training_smoke.json",
        "steps": 0,
        "status": "blocked",
    }
    assert next_actions["real_5b_corpus_manifest"]["action"]
    assert next_actions["r1_corpus_plan_audit"]["action"]
    assert next_actions["real_5b_corpus_manifest"]["guidance"] == {
        "target_tokens": 5_000_000_000,
        "available_tokens": 4097,
        "deficit_tokens": 4_999_995_903,
        "source_deficits": {
            "code-python": {
                "available_tokens": 0,
                "deficit_tokens": 612_500_000,
                "required_tokens": 612_500_000,
                "status": "missing",
            },
            "fineweb-edu": {
                "available_tokens": 4097,
                "deficit_tokens": 2_659_995_903,
                "required_tokens": 2_660_000_000,
                "status": "short",
            },
        },
    }
    assert next_actions["r1_corpus_plan_audit"]["guidance"] == {
        "target_tokens": 5_000_000_000,
        "available_tokens": 4097,
        "deficit_tokens": 4_999_995_903,
        "source_deficits": {
            "code-python": {
                "available_tokens": 0,
                "deficit_tokens": 612_500_000,
                "required_tokens": 612_500_000,
                "status": "missing",
            },
            "fineweb-edu": {
                "available_tokens": 4097,
                "deficit_tokens": 2_659_995_903,
                "required_tokens": 2_660_000_000,
                "status": "short",
            },
        },
    }
    assert next_actions["local_stage1_input_inspection"]["action"]
    assert next_actions["stage1_source_resolution"]["action"]
    assert next_actions["stage1_remote_readiness"]["action"]
    assert next_actions["stage1_remote_readiness"]["guidance"] == {
        "env_file": ".scratch/mini-kimi-k3-replication/.env",
        "env_file_template": ".scratch/mini-kimi-k3-replication/.env.example",
        "required_keys": [
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "MODAL_PROFILE",
        ],
    }
    assert next_actions["guarded_launch_completed"]["evidence"] == str(
        bundle / "outcome.json"
    )
    assert next_actions["guarded_launch_completed"]["guidance"] == {
        "attempt_outcome_report": str(bundle / "outcome.json"),
        "execution_outcome": "blocked",
        "kind": "attempt_outcome",
        "status": "fail",
    }
    assert next_actions["training_measurement_real"]["guidance"] == {
        "attempt_outcome_report": str(bundle / "outcome.json"),
        "execution_outcome": None,
        "measurement": None,
        "status": "missing",
    }
    assert next_actions["train_stage_succeeded"]["evidence"] == str(
        bundle / "processes" / "coordinator" / "events.jsonl"
    )
    assert next_actions["train_stage_succeeded"]["guidance"] == {
        "event_stream": str(bundle / "processes" / "coordinator" / "events.jsonl"),
        "status": "fail",
        "terminal_events": [
            {
                "kind": "stage_blocked",
                "stage_invocation_id": "inv-preflight",
                "return_code": 21,
            }
        ],
    }
    assert "model_fidelity" not in next_actions
    assert "raw_shard_disk_capacity" not in next_actions
    assert json.loads(audit_report.read_text()) == result


def test_mini_kimi_k3_completion_audit_cli_prints_missing_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    preflight_report = tmp_path / "preflight.json"
    audit_report = tmp_path / "completion-audit.json"
    bundle = initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )
    (bundle / "outcome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "attempt_outcome",
                "attempt_id": "attempt-001",
                "execution_outcome": "blocked",
                "stage_invocation_ids": ["inv-preflight"],
                "run_gate": {"has_real_measurement": False},
            },
            sort_keys=True,
        )
    )
    (bundle / "processes" / "coordinator").mkdir(parents=True)
    (bundle / "processes" / "coordinator" / "events.jsonl").write_text("")
    preflight_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_preflight",
                "mode": "full",
                "status": "blocked",
                "gates": [
                    {"name": "trainable_config", "status": "fail"},
                    {"name": "model_fidelity", "status": "pass"},
                    {"name": "real_corpus_manifest", "status": "fail"},
                    {
                        "name": "launch_evidence_bundle",
                        "status": "pass",
                        "evidence": {
                            "manifest": str(bundle / "manifest.json"),
                            "run_id": "mini-k3-r1-test",
                            "attempt_id": "attempt-001",
                            "family": "mini_kimi_k3",
                            "task": "pretraining",
                            "lane": "full",
                            "model_variant": "r1",
                        },
                    },
                ],
            },
            sort_keys=True,
        )
    )

    result = completion_audit.main(
        [
            "--report",
            str(audit_report),
            "--preflight-report",
            str(preflight_report),
            "--results-root",
            str(results_root),
            "--run-id",
            "mini-k3-r1-test",
            "--attempt-id",
            "attempt-001",
        ]
    )

    assert result == 21
    stderr = capsys.readouterr().err
    assert "Mini Kimi K3 completion audit: blocked" in stderr
    assert "missing:" in stderr
    assert "full_preflight_ready" in stderr
    assert "guarded_launch_completed" in stderr
    assert "next actions:" in stderr
    assert "full_preflight_ready -> Refresh the full preflight" in stderr
    assert "guarded_launch_completed -> Run the guarded launch" in stderr
    assert "guidance: blocked_preflight_requirements=" in stderr
    assert "r1_smoke_gpu_capacity" in stderr
    assert "guidance: attempt_outcome_report=" in stderr
    assert "execution_outcome=blocked" in stderr


def test_mini_kimi_k3_completion_audit_passes_completed_full_launch(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    preflight_report = tmp_path / "preflight.json"
    audit_report = tmp_path / "completion-audit.json"
    bundle = initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )
    (bundle / "outcome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "attempt_outcome",
                "attempt_id": "attempt-001",
                "execution_outcome": "completed",
                "stage_invocation_ids": ["inv-preflight", "inv-train"],
                "evaluations": {
                    "preflight": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                    "training": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                        "stage_invocation_id": "inv-train",
                    },
                },
                "run_gate": {"has_real_measurement": True},
            },
            sort_keys=True,
        )
    )
    events = bundle / "processes" / "coordinator" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        "\n".join(
            json.dumps(event, sort_keys=True)
            for event in [
                {
                    "kind": "stage_succeeded",
                    "stage_id": "preflight",
                    "stage_invocation_id": "inv-preflight",
                    "return_code": 0,
                },
                {
                    "kind": "stage_succeeded",
                    "stage_id": "train",
                    "stage_invocation_id": "inv-train",
                    "return_code": 0,
                },
            ]
        )
        + "\n"
    )
    preflight_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_preflight",
                "mode": "full",
                "status": "ready",
                "gates": [
                    {
                        "name": "trainable_config",
                        "status": "pass",
                        "evidence": {
                            "r1_training_smoke": {
                                "report": "results/r1_training_smoke.json",
                                "status": "pass",
                                "steps": 1,
                                "rootfs": {
                                    "marker": "1",
                                    "cwd": "/workspace/torchtitan",
                                },
                                "command": {
                                    "argv": [
                                        "/usr/bin/env",
                                        "MODULE=mini_kimi_k3",
                                        "CONFIG=mini_kimi_k3_r1_contract",
                                        "COMM_MODE=fake_backend",
                                        "NGPU=1",
                                        "./run_train.sh",
                                    ],
                                    "cwd": "/workspace/torchtitan",
                                    "return_code": 0,
                                },
                                "gpu_memory": {
                                    "status": "pass",
                                    "required_free_mib": 12000,
                                    "selected": {
                                        "index": 0,
                                        "memory_free_mib": 48000,
                                    },
                                },
                            }
                        },
                    },
                    {"name": "model_fidelity", "status": "pass"},
                    {
                        "name": "real_corpus_manifest",
                        "status": "pass",
                        "evidence": {
                            "corpus_disk_readiness": {
                                "report": "reports/corpus-disk-readiness.json",
                                "status": "ready",
                                "target_tokens": 5_000_000_000,
                                "required_bytes": 24_000_000_000,
                                "free_bytes": 48_000_000_000,
                                "deficit_bytes": 0,
                            },
                            "r1_corpus_plan_audit": {
                                "status": "ready",
                                "target_tokens": 5_000_000_000,
                                "available_tokens": 5_000_000_000,
                                "deficit_tokens": 0,
                            },
                            "stage1_source_resolution": {
                                "status": "ready",
                                "num_benchmarks": 8,
                                "num_corpora": 6,
                                "unresolved_benchmarks": [],
                                "unresolved_corpora": [],
                            },
                            "stage1_remote_readiness": {
                                "report": "reports/stage1-remote-readiness.json",
                                "status": "ready",
                                "huggingface_dns": "pass",
                                "modal_cli": "pass",
                                "modal_auth": "pass",
                                "network_mode": "networked",
                                "rootfs_marker": "1",
                            },
                        },
                    },
                    {
                        "name": "launch_evidence_bundle",
                        "status": "pass",
                        "evidence": {
                            "manifest": str(bundle / "manifest.json"),
                            "run_id": "mini-k3-r1-test",
                            "attempt_id": "attempt-001",
                            "family": "mini_kimi_k3",
                            "task": "pretraining",
                            "lane": "full",
                            "model_variant": "r1",
                        },
                    },
                ],
            },
            sort_keys=True,
        )
    )

    result = completion_audit.run_completion_audit(
        report_path=audit_report,
        preflight_report=preflight_report,
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
    )

    assert result["status"] == "complete"
    assert result["success_criteria"] == [
        "full_preflight_ready",
        "r1_smoke_gpu_capacity",
        "r1_trainer_optimizer_step",
        "model_fidelity",
        "real_5b_corpus_manifest",
        "r1_corpus_plan_audit",
        "raw_shard_disk_capacity",
        "local_stage1_input_inspection",
        "stage1_source_resolution",
        "stage1_remote_readiness",
        "launch_evidence_bundle",
        "guarded_launch_completed",
        "training_measurement_real",
        "train_stage_succeeded",
    ]
    assert all(check["status"] == "pass" for check in result["checks"])
    checklist = {
        item["requirement"]: item for item in result["prompt_to_artifact_checklist"]
    }
    assert checklist["local_stage1_input_inspection"]["status"] == "not_required"
    assert all(
        item["status"] in {"covered", "not_required"}
        for item in result["prompt_to_artifact_checklist"]
    )
    assert result["missing"] == []
    assert result["covered"] == [
        "full_preflight_ready",
        "r1_smoke_gpu_capacity",
        "r1_trainer_optimizer_step",
        "model_fidelity",
        "real_5b_corpus_manifest",
        "r1_corpus_plan_audit",
        "raw_shard_disk_capacity",
        "stage1_source_resolution",
        "stage1_remote_readiness",
        "launch_evidence_bundle",
        "guarded_launch_completed",
        "training_measurement_real",
        "train_stage_succeeded",
    ]
    assert result["not_required"] == ["local_stage1_input_inspection"]
    assert result["next_actions"] == []


def test_mini_kimi_k3_completion_audit_rejects_run_gate_without_real_measurement(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    preflight_report = tmp_path / "preflight.json"
    audit_report = tmp_path / "completion-audit.json"
    bundle = initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )
    (bundle / "outcome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "attempt_outcome",
                "attempt_id": "attempt-001",
                "execution_outcome": "completed",
                "stage_invocation_ids": ["inv-preflight", "inv-train"],
                "evaluations": {
                    "preflight": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                    "training": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                },
                "run_gate": {"has_real_measurement": False},
            },
            sort_keys=True,
        )
    )
    events = bundle / "processes" / "coordinator" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        "\n".join(
            json.dumps(event, sort_keys=True)
            for event in [
                {
                    "kind": "stage_succeeded",
                    "stage_id": "preflight",
                    "stage_invocation_id": "inv-preflight",
                    "return_code": 0,
                },
                {
                    "kind": "stage_succeeded",
                    "stage_id": "train",
                    "stage_invocation_id": "inv-train",
                    "return_code": 0,
                },
            ]
        )
        + "\n"
    )
    preflight_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_preflight",
                "mode": "full",
                "status": "ready",
                "gates": [
                    {
                        "name": "trainable_config",
                        "status": "pass",
                        "evidence": {
                            "r1_training_smoke": {
                                "report": "results/r1_training_smoke.json",
                                "status": "pass",
                                "steps": 1,
                                "gpu_memory": {
                                    "status": "ready",
                                    "required_free_mib": 12000,
                                    "selected": {
                                        "index": 0,
                                        "memory_free_mib": 48000,
                                    },
                                },
                            }
                        },
                    },
                    {"name": "model_fidelity", "status": "pass"},
                    {
                        "name": "real_corpus_manifest",
                        "status": "pass",
                        "evidence": {
                            "corpus_disk_readiness": {
                                "report": "reports/corpus-disk-readiness.json",
                                "status": "ready",
                                "target_tokens": 5_000_000_000,
                                "required_bytes": 24_000_000_000,
                                "free_bytes": 48_000_000_000,
                                "deficit_bytes": 0,
                            },
                            "r1_corpus_plan_audit": {
                                "status": "ready",
                                "target_tokens": 5_000_000_000,
                                "available_tokens": 5_000_000_000,
                                "deficit_tokens": 0,
                            },
                            "stage1_source_resolution": {
                                "status": "ready",
                                "num_benchmarks": 8,
                                "num_corpora": 6,
                                "unresolved_benchmarks": [],
                                "unresolved_corpora": [],
                            },
                            "stage1_remote_readiness": {
                                "report": "reports/stage1-remote-readiness.json",
                                "status": "ready",
                                "huggingface_dns": "pass",
                                "modal_cli": "pass",
                                "modal_auth": "pass",
                                "network_mode": "networked",
                                "rootfs_marker": "1",
                            },
                        },
                    },
                    {
                        "name": "launch_evidence_bundle",
                        "status": "pass",
                        "evidence": {
                            "manifest": str(bundle / "manifest.json"),
                            "run_id": "mini-k3-r1-test",
                            "attempt_id": "attempt-001",
                            "family": "mini_kimi_k3",
                            "task": "pretraining",
                            "lane": "full",
                            "model_variant": "r1",
                        },
                    },
                ],
            },
            sort_keys=True,
        )
    )

    result = completion_audit.run_completion_audit(
        report_path=audit_report,
        preflight_report=preflight_report,
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
    )

    assert result["status"] == "blocked"
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["training_measurement_real"]["status"] == "fail"
    assert checks["training_measurement_real"]["evidence"]["run_gate"] == {
        "has_real_measurement": False
    }
    assert "training_measurement_real" in result["missing"]


def test_mini_kimi_k3_completion_audit_rejects_training_measurement_without_stage_invocation(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    preflight_report = tmp_path / "preflight.json"
    audit_report = tmp_path / "completion-audit.json"
    bundle = initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )
    (bundle / "outcome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "attempt_outcome",
                "attempt_id": "attempt-001",
                "execution_outcome": "completed",
                "stage_invocation_ids": ["inv-preflight", "inv-train"],
                "evaluations": {
                    "preflight": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                    "training": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                },
                "run_gate": {"has_real_measurement": True},
            },
            sort_keys=True,
        )
    )
    events = bundle / "processes" / "coordinator" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        "\n".join(
            json.dumps(event, sort_keys=True)
            for event in [
                {
                    "kind": "stage_succeeded",
                    "stage_id": "preflight",
                    "stage_invocation_id": "inv-preflight",
                    "return_code": 0,
                },
                {
                    "kind": "stage_succeeded",
                    "stage_id": "train",
                    "stage_invocation_id": "inv-train",
                    "return_code": 0,
                },
            ]
        )
        + "\n"
    )
    preflight_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_preflight",
                "mode": "full",
                "status": "ready",
                "gates": [
                    {
                        "name": "trainable_config",
                        "status": "pass",
                        "evidence": {
                            "r1_training_smoke": {
                                "status": "pass",
                                "steps": 1,
                                "rootfs": {
                                    "marker": "1",
                                    "cwd": "/workspace/torchtitan",
                                },
                                "command": {
                                    "argv": [
                                        "/usr/bin/env",
                                        "MODULE=mini_kimi_k3",
                                        "CONFIG=mini_kimi_k3_r1_contract",
                                        "COMM_MODE=fake_backend",
                                        "NGPU=1",
                                        "./run_train.sh",
                                    ],
                                    "cwd": "/workspace/torchtitan",
                                    "return_code": 0,
                                },
                                "gpu_memory": {
                                    "status": "ready",
                                    "selected": {"index": 0},
                                },
                            }
                        },
                    },
                    {"name": "model_fidelity", "status": "pass"},
                    {
                        "name": "real_corpus_manifest",
                        "status": "pass",
                        "evidence": {
                            "corpus_disk_readiness": {
                                "status": "ready",
                                "deficit_bytes": 0,
                            },
                            "r1_corpus_plan_audit": {
                                "status": "ready",
                                "deficit_tokens": 0,
                            },
                            "stage1_source_resolution": {
                                "status": "ready",
                                "unresolved_benchmarks": [],
                                "unresolved_corpora": [],
                            },
                            "stage1_remote_readiness": {"status": "not_required"},
                        },
                    },
                    {
                        "name": "launch_evidence_bundle",
                        "status": "pass",
                        "evidence": {
                            "manifest": str(bundle / "manifest.json"),
                            "run_id": "mini-k3-r1-test",
                            "attempt_id": "attempt-001",
                            "family": "mini_kimi_k3",
                            "task": "pretraining",
                            "lane": "full",
                            "model_variant": "r1",
                        },
                    },
                ],
            },
            sort_keys=True,
        )
    )

    result = completion_audit.run_completion_audit(
        report_path=audit_report,
        preflight_report=preflight_report,
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
    )

    assert result["status"] == "blocked"
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["training_measurement_real"]["status"] == "fail"
    assert (
        checks["training_measurement_real"]["evidence"]["stage_invocation_id"] is None
    )
    assert checks["training_measurement_real"]["evidence"][
        "matched_train_stage_invocation_ids"
    ] == ["inv-train"]
    assert "training_measurement_real" in result["missing"]


def test_mini_kimi_k3_completion_audit_rejects_remote_readiness_without_rootfs_marker(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    preflight_report = tmp_path / "preflight.json"
    audit_report = tmp_path / "completion-audit.json"
    bundle = initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )
    (bundle / "outcome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "attempt_outcome",
                "attempt_id": "attempt-001",
                "execution_outcome": "completed",
                "stage_invocation_ids": ["inv-preflight", "inv-train"],
                "evaluations": {
                    "preflight": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                    "training": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                        "stage_invocation_id": "inv-train",
                    },
                },
                "run_gate": {"has_real_measurement": True},
            },
            sort_keys=True,
        )
    )
    events = bundle / "processes" / "coordinator" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        "\n".join(
            json.dumps(event, sort_keys=True)
            for event in [
                {
                    "kind": "stage_succeeded",
                    "stage_id": "preflight",
                    "stage_invocation_id": "inv-preflight",
                    "return_code": 0,
                },
                {
                    "kind": "stage_succeeded",
                    "stage_id": "train",
                    "stage_invocation_id": "inv-train",
                    "return_code": 0,
                },
            ]
        )
        + "\n"
    )
    preflight_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_preflight",
                "mode": "full",
                "status": "ready",
                "gates": [
                    {
                        "name": "trainable_config",
                        "status": "pass",
                        "evidence": {
                            "r1_training_smoke": {
                                "status": "pass",
                                "steps": 1,
                                "rootfs": {
                                    "marker": "1",
                                    "cwd": "/workspace/torchtitan",
                                },
                                "command": {
                                    "argv": [
                                        "/usr/bin/env",
                                        "MODULE=mini_kimi_k3",
                                        "CONFIG=mini_kimi_k3_r1_contract",
                                        "COMM_MODE=fake_backend",
                                        "NGPU=1",
                                        "./run_train.sh",
                                    ],
                                    "cwd": "/workspace/torchtitan",
                                    "return_code": 0,
                                },
                                "gpu_memory": {
                                    "status": "ready",
                                    "selected": {"index": 0},
                                },
                            }
                        },
                    },
                    {"name": "model_fidelity", "status": "pass"},
                    {
                        "name": "real_corpus_manifest",
                        "status": "pass",
                        "evidence": {
                            "corpus_disk_readiness": {
                                "status": "ready",
                                "deficit_bytes": 0,
                            },
                            "r1_corpus_plan_audit": {
                                "status": "ready",
                                "deficit_tokens": 0,
                            },
                            "stage1_source_resolution": {
                                "status": "ready",
                                "unresolved_benchmarks": [],
                                "unresolved_corpora": [],
                            },
                            "stage1_remote_readiness": {
                                "report": "reports/stage1-remote-readiness.json",
                                "status": "ready",
                                "huggingface_dns": "pass",
                                "modal_cli": "pass",
                                "modal_auth": "pass",
                                "network_mode": "networked",
                            },
                        },
                    },
                    {
                        "name": "launch_evidence_bundle",
                        "status": "pass",
                        "evidence": {
                            "manifest": str(bundle / "manifest.json"),
                            "run_id": "mini-k3-r1-test",
                            "attempt_id": "attempt-001",
                            "family": "mini_kimi_k3",
                            "task": "pretraining",
                            "lane": "full",
                            "model_variant": "r1",
                        },
                    },
                ],
            },
            sort_keys=True,
        )
    )

    result = completion_audit.run_completion_audit(
        report_path=audit_report,
        preflight_report=preflight_report,
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
    )

    assert result["status"] == "blocked"
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["stage1_remote_readiness_ready"]["status"] == "fail"
    assert checks["stage1_remote_readiness_ready"]["evidence"]["rootfs_marker"] is None
    assert "stage1_remote_readiness" in result["missing"]


def test_mini_kimi_k3_completion_audit_rejects_trainable_gate_without_smoke_step(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    preflight_report = tmp_path / "preflight.json"
    audit_report = tmp_path / "completion-audit.json"
    bundle = initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )
    (bundle / "outcome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "attempt_outcome",
                "attempt_id": "attempt-001",
                "execution_outcome": "completed",
                "stage_invocation_ids": ["inv-preflight", "inv-train"],
                "evaluations": {
                    "preflight": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                    "training": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                        "stage_invocation_id": "inv-train",
                    },
                },
                "run_gate": {"has_real_measurement": True},
            },
            sort_keys=True,
        )
    )
    events = bundle / "processes" / "coordinator" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        "\n".join(
            json.dumps(event, sort_keys=True)
            for event in [
                {
                    "kind": "stage_succeeded",
                    "stage_id": "preflight",
                    "stage_invocation_id": "inv-preflight",
                    "return_code": 0,
                },
                {
                    "kind": "stage_succeeded",
                    "stage_id": "train",
                    "stage_invocation_id": "inv-train",
                    "return_code": 0,
                },
            ]
        )
        + "\n"
    )
    preflight_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_preflight",
                "mode": "full",
                "status": "ready",
                "gates": [
                    {
                        "name": "trainable_config",
                        "status": "pass",
                        "evidence": {
                            "r1_training_smoke": {
                                "report": "results/r1_training_smoke.json",
                                "status": "pass",
                                "steps": 0,
                                "gpu_memory": {
                                    "status": "pass",
                                    "required_free_mib": 12000,
                                    "selected": {
                                        "index": 0,
                                        "memory_free_mib": 48000,
                                    },
                                },
                            }
                        },
                    },
                    {"name": "model_fidelity", "status": "pass"},
                    {
                        "name": "real_corpus_manifest",
                        "status": "pass",
                        "evidence": {
                            "corpus_disk_readiness": {
                                "report": "reports/corpus-disk-readiness.json",
                                "status": "ready",
                                "target_tokens": 5_000_000_000,
                                "required_bytes": 24_000_000_000,
                                "free_bytes": 48_000_000_000,
                                "deficit_bytes": 0,
                            },
                            "r1_corpus_plan_audit": {
                                "status": "ready",
                                "target_tokens": 5_000_000_000,
                                "available_tokens": 5_000_000_000,
                                "deficit_tokens": 0,
                            },
                            "stage1_source_resolution": {
                                "status": "ready",
                                "num_benchmarks": 8,
                                "num_corpora": 6,
                                "unresolved_benchmarks": [],
                                "unresolved_corpora": [],
                            },
                            "stage1_remote_readiness": {
                                "report": "reports/stage1-remote-readiness.json",
                                "status": "ready",
                                "huggingface_dns": "pass",
                                "modal_cli": "pass",
                                "modal_auth": "pass",
                                "network_mode": "networked",
                                "rootfs_marker": "1",
                            },
                        },
                    },
                    {
                        "name": "launch_evidence_bundle",
                        "status": "pass",
                        "evidence": {
                            "manifest": str(bundle / "manifest.json"),
                            "run_id": "mini-k3-r1-test",
                            "attempt_id": "attempt-001",
                            "family": "mini_kimi_k3",
                            "task": "pretraining",
                            "lane": "full",
                            "model_variant": "r1",
                        },
                    },
                ],
            },
            sort_keys=True,
        )
    )

    result = completion_audit.run_completion_audit(
        report_path=audit_report,
        preflight_report=preflight_report,
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
    )

    assert result["status"] == "blocked"
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["trainable_config_passed"]["status"] == "fail"
    assert (
        checks["trainable_config_passed"]["evidence"]["r1_training_smoke"]["steps"] == 0
    )
    assert "r1_trainer_optimizer_step" in result["missing"]


def test_mini_kimi_k3_completion_audit_rejects_trainable_gate_without_smoke_provenance(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    preflight_report = tmp_path / "preflight.json"
    audit_report = tmp_path / "completion-audit.json"
    bundle = initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )
    (bundle / "outcome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "attempt_outcome",
                "attempt_id": "attempt-001",
                "execution_outcome": "completed",
                "stage_invocation_ids": ["inv-preflight", "inv-train"],
                "evaluations": {
                    "training": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "stage_invocation_id": "inv-train",
                    },
                },
                "run_gate": {"has_real_measurement": True},
            },
            sort_keys=True,
        )
    )
    events = bundle / "processes" / "coordinator" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        json.dumps(
            {
                "kind": "stage_succeeded",
                "stage_id": "train",
                "stage_invocation_id": "inv-train",
                "return_code": 0,
            },
            sort_keys=True,
        )
        + "\n"
    )
    preflight_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_preflight",
                "mode": "full",
                "status": "ready",
                "gates": [
                    {
                        "name": "trainable_config",
                        "status": "pass",
                        "evidence": {
                            "r1_training_smoke": {
                                "report": "results/r1_training_smoke.json",
                                "status": "pass",
                                "steps": 1,
                                "gpu_memory": {
                                    "status": "pass",
                                    "required_free_mib": 12000,
                                    "selected": {
                                        "index": 0,
                                        "memory_free_mib": 48000,
                                    },
                                },
                            }
                        },
                    },
                    {"name": "model_fidelity", "status": "pass"},
                    {
                        "name": "real_corpus_manifest",
                        "status": "pass",
                        "evidence": {
                            "corpus_disk_readiness": {
                                "status": "ready",
                                "deficit_bytes": 0,
                            },
                            "r1_corpus_plan_audit": {
                                "status": "ready",
                                "deficit_tokens": 0,
                            },
                            "stage1_source_resolution": {
                                "status": "ready",
                                "unresolved_benchmarks": [],
                                "unresolved_corpora": [],
                            },
                            "stage1_remote_readiness": {"status": "not_required"},
                        },
                    },
                    {
                        "name": "launch_evidence_bundle",
                        "status": "pass",
                        "evidence": {
                            "manifest": str(bundle / "manifest.json"),
                            "run_id": "mini-k3-r1-test",
                            "attempt_id": "attempt-001",
                            "family": "mini_kimi_k3",
                            "task": "pretraining",
                            "lane": "full",
                            "model_variant": "r1",
                        },
                    },
                ],
            },
            sort_keys=True,
        )
    )

    result = completion_audit.run_completion_audit(
        report_path=audit_report,
        preflight_report=preflight_report,
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
    )

    assert result["status"] == "blocked"
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["trainable_config_passed"]["status"] == "fail"
    assert checks["trainable_config_passed"]["evidence"]["r1_training_smoke"][
        "rootfs"
    ] == {"status": "missing"}
    assert checks["trainable_config_passed"]["evidence"]["r1_training_smoke"][
        "command"
    ] == {"status": "missing"}
    assert "r1_trainer_optimizer_step" in result["missing"]


def test_mini_kimi_k3_completion_audit_rejects_stale_preflight_schema(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    preflight_report = tmp_path / "preflight.json"
    audit_report = tmp_path / "completion-audit.json"
    bundle = initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )
    (bundle / "outcome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "attempt_outcome",
                "attempt_id": "attempt-001",
                "execution_outcome": "completed",
                "stage_invocation_ids": ["inv-preflight", "inv-train"],
                "evaluations": {
                    "training": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                },
                "run_gate": {"has_real_measurement": True},
            },
            sort_keys=True,
        )
    )
    events = bundle / "processes" / "coordinator" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        json.dumps(
            {
                "kind": "stage_succeeded",
                "stage_id": "train",
                "stage_invocation_id": "inv-train",
                "return_code": 0,
            },
            sort_keys=True,
        )
        + "\n"
    )
    preflight_report.write_text(
        json.dumps(
            {
                "schema_version": 0,
                "kind": "mini_kimi_k3_preflight",
                "mode": "full",
                "status": "ready",
                "gates": [
                    {
                        "name": "trainable_config",
                        "status": "pass",
                        "evidence": {
                            "r1_training_smoke": {
                                "status": "pass",
                                "steps": 1,
                                "rootfs": {
                                    "marker": "1",
                                    "cwd": "/workspace/torchtitan",
                                },
                                "command": {
                                    "argv": [
                                        "/usr/bin/env",
                                        "MODULE=mini_kimi_k3",
                                        "CONFIG=mini_kimi_k3_r1_contract",
                                        "COMM_MODE=fake_backend",
                                        "NGPU=1",
                                        "./run_train.sh",
                                    ],
                                    "cwd": "/workspace/torchtitan",
                                    "return_code": 0,
                                },
                                "gpu_memory": {
                                    "status": "ready",
                                    "selected": {"index": 0},
                                },
                            }
                        },
                    },
                    {"name": "model_fidelity", "status": "pass"},
                    {
                        "name": "real_corpus_manifest",
                        "status": "pass",
                        "evidence": {
                            "corpus_disk_readiness": {
                                "status": "ready",
                                "deficit_bytes": 0,
                            },
                            "r1_corpus_plan_audit": {
                                "status": "ready",
                                "deficit_tokens": 0,
                            },
                            "stage1_source_resolution": {
                                "status": "ready",
                                "unresolved_benchmarks": [],
                                "unresolved_corpora": [],
                            },
                            "stage1_remote_readiness": {"status": "not_required"},
                        },
                    },
                    {
                        "name": "launch_evidence_bundle",
                        "status": "pass",
                        "evidence": {
                            "manifest": str(bundle / "manifest.json"),
                            "run_id": "mini-k3-r1-test",
                            "attempt_id": "attempt-001",
                            "family": "mini_kimi_k3",
                            "task": "pretraining",
                            "lane": "full",
                            "model_variant": "r1",
                        },
                    },
                ],
            },
            sort_keys=True,
        )
    )

    result = completion_audit.run_completion_audit(
        report_path=audit_report,
        preflight_report=preflight_report,
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
    )

    assert result["status"] == "blocked"
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["full_preflight_ready"]["status"] == "fail"
    assert checks["full_preflight_ready"]["evidence"]["schema_version"] == 0
    assert "full_preflight_ready" in result["missing"]


def test_mini_kimi_k3_completion_audit_rejects_stale_outcome_schema(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    preflight_report = tmp_path / "preflight.json"
    audit_report = tmp_path / "completion-audit.json"
    bundle = initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )
    (bundle / "outcome.json").write_text(
        json.dumps(
            {
                "schema_version": 0,
                "kind": "attempt_outcome",
                "attempt_id": "attempt-001",
                "execution_outcome": "completed",
                "stage_invocation_ids": ["inv-preflight", "inv-train"],
                "evaluations": {
                    "training": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                },
                "run_gate": {"has_real_measurement": True},
            },
            sort_keys=True,
        )
    )
    events = bundle / "processes" / "coordinator" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        json.dumps(
            {
                "kind": "stage_succeeded",
                "stage_id": "train",
                "stage_invocation_id": "inv-train",
                "return_code": 0,
            },
            sort_keys=True,
        )
        + "\n"
    )
    preflight_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_preflight",
                "mode": "full",
                "status": "ready",
                "gates": [
                    {
                        "name": "trainable_config",
                        "status": "pass",
                        "evidence": {
                            "r1_training_smoke": {
                                "status": "pass",
                                "steps": 1,
                                "gpu_memory": {
                                    "status": "ready",
                                    "selected": {"index": 0},
                                },
                            }
                        },
                    },
                    {"name": "model_fidelity", "status": "pass"},
                    {
                        "name": "real_corpus_manifest",
                        "status": "pass",
                        "evidence": {
                            "corpus_disk_readiness": {
                                "status": "ready",
                                "deficit_bytes": 0,
                            },
                            "r1_corpus_plan_audit": {
                                "status": "ready",
                                "deficit_tokens": 0,
                            },
                            "stage1_source_resolution": {
                                "status": "ready",
                                "unresolved_benchmarks": [],
                                "unresolved_corpora": [],
                            },
                            "stage1_remote_readiness": {"status": "not_required"},
                        },
                    },
                    {
                        "name": "launch_evidence_bundle",
                        "status": "pass",
                        "evidence": {
                            "manifest": str(bundle / "manifest.json"),
                            "run_id": "mini-k3-r1-test",
                            "attempt_id": "attempt-001",
                            "family": "mini_kimi_k3",
                            "task": "pretraining",
                            "lane": "full",
                            "model_variant": "r1",
                        },
                    },
                ],
            },
            sort_keys=True,
        )
    )

    result = completion_audit.run_completion_audit(
        report_path=audit_report,
        preflight_report=preflight_report,
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
    )

    assert result["status"] == "blocked"
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["guarded_launch_completed"]["status"] == "fail"
    assert checks["guarded_launch_completed"]["evidence"]["schema_version"] == 0
    assert "guarded_launch_completed" in result["missing"]


def test_mini_kimi_k3_completion_audit_rejects_wrong_lane_launch_bundle(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    preflight_report = tmp_path / "preflight.json"
    audit_report = tmp_path / "completion-audit.json"
    bundle = initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )
    (bundle / "outcome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "attempt_outcome",
                "attempt_id": "attempt-001",
                "execution_outcome": "completed",
                "stage_invocation_ids": ["inv-preflight", "inv-train"],
                "evaluations": {
                    "preflight": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                    "training": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                },
            },
            sort_keys=True,
        )
    )
    events = bundle / "processes" / "coordinator" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        "\n".join(
            json.dumps(event, sort_keys=True)
            for event in [
                {
                    "kind": "stage_succeeded",
                    "stage_id": "preflight",
                    "stage_invocation_id": "inv-preflight",
                    "return_code": 0,
                },
                {
                    "kind": "stage_succeeded",
                    "stage_id": "train",
                    "stage_invocation_id": "inv-train",
                    "return_code": 0,
                },
            ]
        )
        + "\n"
    )
    preflight_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_preflight",
                "mode": "full",
                "status": "ready",
                "gates": [
                    {
                        "name": "trainable_config",
                        "status": "pass",
                        "evidence": {
                            "r1_training_smoke": {
                                "status": "pass",
                                "steps": 1,
                                "gpu_memory": {
                                    "status": "ready",
                                    "selected": {"index": 0},
                                },
                            }
                        },
                    },
                    {"name": "model_fidelity", "status": "pass"},
                    {
                        "name": "real_corpus_manifest",
                        "status": "pass",
                        "evidence": {
                            "corpus_disk_readiness": {
                                "status": "ready",
                                "deficit_bytes": 0,
                            },
                            "r1_corpus_plan_audit": {
                                "status": "ready",
                                "deficit_tokens": 0,
                            },
                            "stage1_source_resolution": {
                                "status": "ready",
                                "unresolved_benchmarks": [],
                                "unresolved_corpora": [],
                            },
                            "stage1_remote_readiness": {"status": "not_required"},
                        },
                    },
                    {
                        "name": "launch_evidence_bundle",
                        "status": "pass",
                        "evidence": {
                            "manifest": str(bundle / "manifest.json"),
                            "run_id": "mini-k3-r1-test",
                            "attempt_id": "attempt-001",
                            "family": "mini_kimi_k3",
                            "task": "pretraining",
                            "lane": "tiny-plumbing",
                            "model_variant": "r1",
                        },
                    },
                ],
            },
            sort_keys=True,
        )
    )

    result = completion_audit.run_completion_audit(
        report_path=audit_report,
        preflight_report=preflight_report,
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
    )

    assert result["status"] == "blocked"
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["launch_evidence_bundle_passed"]["status"] == "fail"
    assert checks["launch_evidence_bundle_passed"]["evidence"]["lane"] == (
        "tiny-plumbing"
    )
    assert "launch_evidence_bundle" in result["missing"]


def test_mini_kimi_k3_completion_audit_rejects_wrong_attempt_outcome(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    preflight_report = tmp_path / "preflight.json"
    audit_report = tmp_path / "completion-audit.json"
    bundle = initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )
    (bundle / "outcome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "attempt_outcome",
                "attempt_id": "stale-attempt",
                "execution_outcome": "completed",
                "stage_invocation_ids": ["inv-preflight", "inv-train"],
                "evaluations": {
                    "preflight": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                    "training": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                },
            },
            sort_keys=True,
        )
    )
    events = bundle / "processes" / "coordinator" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        "\n".join(
            json.dumps(event, sort_keys=True)
            for event in [
                {
                    "kind": "stage_succeeded",
                    "stage_id": "preflight",
                    "stage_invocation_id": "inv-preflight",
                    "return_code": 0,
                },
                {
                    "kind": "stage_succeeded",
                    "stage_id": "train",
                    "stage_invocation_id": "inv-train",
                    "return_code": 0,
                },
            ]
        )
        + "\n"
    )
    preflight_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_preflight",
                "mode": "full",
                "status": "ready",
                "gates": [
                    {
                        "name": "trainable_config",
                        "status": "pass",
                        "evidence": {
                            "r1_training_smoke": {
                                "status": "pass",
                                "steps": 1,
                                "gpu_memory": {
                                    "status": "ready",
                                    "selected": {"index": 0},
                                },
                            }
                        },
                    },
                    {"name": "model_fidelity", "status": "pass"},
                    {
                        "name": "real_corpus_manifest",
                        "status": "pass",
                        "evidence": {
                            "corpus_disk_readiness": {
                                "status": "ready",
                                "deficit_bytes": 0,
                            },
                            "r1_corpus_plan_audit": {
                                "status": "ready",
                                "deficit_tokens": 0,
                            },
                            "stage1_source_resolution": {
                                "status": "ready",
                                "unresolved_benchmarks": [],
                                "unresolved_corpora": [],
                            },
                            "stage1_remote_readiness": {"status": "not_required"},
                        },
                    },
                    {
                        "name": "launch_evidence_bundle",
                        "status": "pass",
                        "evidence": {
                            "manifest": str(bundle / "manifest.json"),
                            "run_id": "mini-k3-r1-test",
                            "attempt_id": "attempt-001",
                            "family": "mini_kimi_k3",
                            "task": "pretraining",
                            "lane": "full",
                            "model_variant": "r1",
                        },
                    },
                ],
            },
            sort_keys=True,
        )
    )

    result = completion_audit.run_completion_audit(
        report_path=audit_report,
        preflight_report=preflight_report,
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
    )

    assert result["status"] == "blocked"
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["guarded_launch_completed"]["status"] == "fail"
    assert checks["guarded_launch_completed"]["evidence"]["attempt_id"] == (
        "stale-attempt"
    )
    assert "guarded_launch_completed" in result["missing"]


def test_mini_kimi_k3_completion_audit_rejects_unlisted_train_event(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    preflight_report = tmp_path / "preflight.json"
    audit_report = tmp_path / "completion-audit.json"
    bundle = initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )
    (bundle / "outcome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "attempt_outcome",
                "attempt_id": "attempt-001",
                "execution_outcome": "completed",
                "stage_invocation_ids": ["inv-preflight", "inv-train"],
                "evaluations": {
                    "preflight": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                    "training": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                },
            },
            sort_keys=True,
        )
    )
    events = bundle / "processes" / "coordinator" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        "\n".join(
            json.dumps(event, sort_keys=True)
            for event in [
                {
                    "kind": "stage_succeeded",
                    "stage_id": "preflight",
                    "stage_invocation_id": "inv-preflight",
                    "return_code": 0,
                },
                {
                    "kind": "stage_succeeded",
                    "stage_id": "train",
                    "stage_invocation_id": "stale-train-invocation",
                    "return_code": 0,
                },
            ]
        )
        + "\n"
    )
    preflight_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_preflight",
                "mode": "full",
                "status": "ready",
                "gates": [
                    {
                        "name": "trainable_config",
                        "status": "pass",
                        "evidence": {
                            "r1_training_smoke": {
                                "status": "pass",
                                "steps": 1,
                                "gpu_memory": {
                                    "status": "ready",
                                    "selected": {"index": 0},
                                },
                            }
                        },
                    },
                    {"name": "model_fidelity", "status": "pass"},
                    {
                        "name": "real_corpus_manifest",
                        "status": "pass",
                        "evidence": {
                            "corpus_disk_readiness": {
                                "status": "ready",
                                "deficit_bytes": 0,
                            },
                            "r1_corpus_plan_audit": {
                                "status": "ready",
                                "deficit_tokens": 0,
                            },
                            "stage1_source_resolution": {
                                "status": "ready",
                                "unresolved_benchmarks": [],
                                "unresolved_corpora": [],
                            },
                            "stage1_remote_readiness": {"status": "not_required"},
                        },
                    },
                    {
                        "name": "launch_evidence_bundle",
                        "status": "pass",
                        "evidence": {
                            "manifest": str(bundle / "manifest.json"),
                            "run_id": "mini-k3-r1-test",
                            "attempt_id": "attempt-001",
                            "family": "mini_kimi_k3",
                            "task": "pretraining",
                            "lane": "full",
                            "model_variant": "r1",
                        },
                    },
                ],
            },
            sort_keys=True,
        )
    )

    result = completion_audit.run_completion_audit(
        report_path=audit_report,
        preflight_report=preflight_report,
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
    )

    assert result["status"] == "blocked"
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["training_measurement_real"]["status"] == "fail"
    assert checks["training_measurement_real"]["evidence"][
        "outcome_stage_invocation_ids"
    ] == [
        "inv-preflight",
        "inv-train",
    ]
    assert (
        checks["training_measurement_real"]["evidence"][
            "matched_train_stage_invocation_ids"
        ]
        == []
    )
    assert checks["train_stage_succeeded"]["status"] == "fail"
    assert checks["train_stage_succeeded"]["evidence"][
        "outcome_stage_invocation_ids"
    ] == [
        "inv-preflight",
        "inv-train",
    ]
    assert checks["train_stage_succeeded"]["evidence"]["train_terminal_events"] == [
        {
            "kind": "stage_succeeded",
            "return_code": 0,
            "stage_invocation_id": "stale-train-invocation",
        }
    ]
    assert "train_stage_succeeded" in result["missing"]


def test_mini_kimi_k3_completion_audit_rejects_corrupt_event_stream(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    preflight_report = tmp_path / "preflight.json"
    audit_report = tmp_path / "completion-audit.json"
    bundle = initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )
    (bundle / "outcome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "attempt_outcome",
                "attempt_id": "attempt-001",
                "execution_outcome": "completed",
                "stage_invocation_ids": ["inv-preflight", "inv-train"],
                "evaluations": {
                    "preflight": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                    "training": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "promotion": "not_evaluated",
                    },
                },
                "run_gate": {"has_real_measurement": True},
            },
            sort_keys=True,
        )
    )
    events = bundle / "processes" / "coordinator" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "stage_succeeded",
                        "stage_id": "train",
                        "stage_invocation_id": "inv-train",
                        "return_code": 0,
                    },
                    sort_keys=True,
                ),
                "{not json",
            ]
        )
        + "\n"
    )
    preflight_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_preflight",
                "mode": "full",
                "status": "ready",
                "gates": [
                    {
                        "name": "trainable_config",
                        "status": "pass",
                        "evidence": {
                            "r1_training_smoke": {
                                "status": "pass",
                                "steps": 1,
                                "gpu_memory": {
                                    "status": "ready",
                                    "selected": {"index": 0},
                                },
                            }
                        },
                    },
                    {"name": "model_fidelity", "status": "pass"},
                    {
                        "name": "real_corpus_manifest",
                        "status": "pass",
                        "evidence": {
                            "corpus_disk_readiness": {
                                "status": "ready",
                                "deficit_bytes": 0,
                            },
                            "r1_corpus_plan_audit": {
                                "status": "ready",
                                "deficit_tokens": 0,
                            },
                            "stage1_source_resolution": {
                                "status": "ready",
                                "unresolved_benchmarks": [],
                                "unresolved_corpora": [],
                            },
                            "stage1_remote_readiness": {"status": "not_required"},
                        },
                    },
                    {
                        "name": "launch_evidence_bundle",
                        "status": "pass",
                        "evidence": {
                            "manifest": str(bundle / "manifest.json"),
                            "run_id": "mini-k3-r1-test",
                            "attempt_id": "attempt-001",
                            "family": "mini_kimi_k3",
                            "task": "pretraining",
                            "lane": "full",
                            "model_variant": "r1",
                        },
                    },
                ],
            },
            sort_keys=True,
        )
    )

    result = completion_audit.run_completion_audit(
        report_path=audit_report,
        preflight_report=preflight_report,
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
    )

    assert result["status"] == "blocked"
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["train_stage_succeeded"]["status"] == "fail"
    assert checks["train_stage_succeeded"]["evidence"]["event_stream_errors"] == [
        {"line": 2, "error": "Expecting property name enclosed in double quotes"}
    ]
    next_actions = {item["requirement"]: item for item in result["next_actions"]}
    assert next_actions["training_measurement_real"]["guidance"][
        "event_stream_errors"
    ] == [{"line": 2, "error": "Expecting property name enclosed in double quotes"}]
    assert next_actions["train_stage_succeeded"]["guidance"]["event_stream_errors"] == [
        {"line": 2, "error": "Expecting property name enclosed in double quotes"}
    ]
    assert "train_stage_succeeded" in result["missing"]


def test_mini_kimi_k3_completion_audit_accepts_local_corpus_without_remote_readiness(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    preflight_report = tmp_path / "preflight.json"
    audit_report = tmp_path / "completion-audit.json"
    bundle = initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )
    (bundle / "outcome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "attempt_outcome",
                "attempt_id": "attempt-001",
                "execution_outcome": "completed",
                "stage_invocation_ids": ["inv-preflight", "inv-train"],
                "evaluations": {
                    "training": {
                        "execution_outcome": "completed",
                        "measurement": "real",
                        "stage_invocation_id": "inv-train",
                    },
                },
                "run_gate": {"has_real_measurement": True},
            },
            sort_keys=True,
        )
    )
    events = bundle / "processes" / "coordinator" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        json.dumps(
            {
                "kind": "stage_succeeded",
                "stage_id": "train",
                "stage_invocation_id": "inv-train",
                "return_code": 0,
            },
            sort_keys=True,
        )
        + "\n"
    )
    preflight_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_preflight",
                "mode": "full",
                "status": "ready",
                "gates": [
                    {
                        "name": "trainable_config",
                        "status": "pass",
                        "evidence": {
                            "r1_training_smoke": {
                                "status": "pass",
                                "steps": 1,
                                "rootfs": {
                                    "marker": "1",
                                    "cwd": "/workspace/torchtitan",
                                },
                                "command": {
                                    "argv": [
                                        "/usr/bin/env",
                                        "MODULE=mini_kimi_k3",
                                        "CONFIG=mini_kimi_k3_r1_contract",
                                        "COMM_MODE=fake_backend",
                                        "NGPU=1",
                                        "./run_train.sh",
                                    ],
                                    "cwd": "/workspace/torchtitan",
                                    "return_code": 0,
                                },
                                "gpu_memory": {
                                    "status": "ready",
                                    "selected": {"index": 0},
                                },
                            }
                        },
                    },
                    {"name": "model_fidelity", "status": "pass"},
                    {
                        "name": "real_corpus_manifest",
                        "status": "pass",
                        "evidence": {
                            "corpus_route": "local_or_imported",
                            "requires_stage1_input_inspection": True,
                            "r1_corpus_plan_audit": {
                                "status": "ready",
                                "target_tokens": 5_000_000_000,
                                "available_tokens": 5_000_000_000,
                                "deficit_tokens": 0,
                            },
                            "corpus_disk_readiness": {
                                "status": "ready",
                                "deficit_bytes": 0,
                            },
                            "stage1_source_resolution": {
                                "status": "ready",
                                "unresolved_benchmarks": [],
                                "unresolved_corpora": [],
                            },
                            "stage1_input_inspection": {
                                "status": "pass",
                                "input_format": "jsonl-tokens",
                                "min_total_tokens": 5_000_000_000,
                                "total_files": 2,
                                "total_documents": 10,
                                "total_tokens": 5_000_000_000,
                            },
                            "stage1_remote_readiness": {
                                "status": "not_required",
                                "route": "local_or_imported",
                            },
                        },
                    },
                    {
                        "name": "launch_evidence_bundle",
                        "status": "pass",
                        "evidence": {
                            "manifest": str(bundle / "manifest.json"),
                            "run_id": "mini-k3-r1-test",
                            "attempt_id": "attempt-001",
                            "family": "mini_kimi_k3",
                            "task": "pretraining",
                            "lane": "full",
                            "model_variant": "r1",
                        },
                    },
                ],
            },
            sort_keys=True,
        )
    )

    result = completion_audit.run_completion_audit(
        report_path=audit_report,
        preflight_report=preflight_report,
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
    )

    checklist = {
        item["requirement"]: item for item in result["prompt_to_artifact_checklist"]
    }
    assert result["status"] == "complete"
    assert checklist["r1_corpus_plan_audit"]["status"] == "covered"
    assert checklist["local_stage1_input_inspection"]["status"] == "covered"
    assert checklist["stage1_source_resolution"]["status"] == "covered"
    assert checklist["stage1_remote_readiness"]["status"] == "not_required"
    assert result["missing"] == []
    assert result["not_required"] == ["stage1_remote_readiness"]
    assert result["next_actions"] == []


def test_mini_kimi_k3_stage1_remote_blocks_without_ready_readiness_report(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    source_root = tmp_path / "mini-kimi-k3-vizuara"
    (source_root / "data").mkdir(parents=True)
    (source_root / "data" / "stage1.py").write_text("# fixture\n")
    readiness = tmp_path / "stage1-remote-readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_remote_readiness",
                "status": "blocked",
                "checks": {
                    "rootfs_network": {"status": "pass"},
                    "huggingface_dns": {"status": "pass"},
                    "modal_cli": {"status": "pass"},
                    "modal_auth": {"status": "blocked"},
                },
                "rootfs": {"marker": "1", "network_mode": "networked"},
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "stage1-remote-step.json"

    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(self, argv: list[str], *, cwd: Path | None = None) -> CommandResult:
            del cwd
            self.calls.append(list(argv))
            return CommandResult(return_code=0)

    executor = RecordingExecutor()

    result = run_stage1_remote.run_stage1_remote(
        source_root=source_root,
        step="dry_run",
        readiness_report=readiness,
        report_path=report,
        executor=executor,
    )

    assert result["status"] == "blocked"
    assert "readiness report must be ready" in result["detail"]
    assert executor.calls == []
    assert json.loads(report.read_text()) == result


def test_mini_kimi_k3_stage1_remote_requires_ingest_authorization(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    source_root = tmp_path / "mini-kimi-k3-vizuara"
    (source_root / "data").mkdir(parents=True)
    (source_root / "data" / "stage1.py").write_text("# fixture\n")
    readiness = tmp_path / "stage1-remote-readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_remote_readiness",
                "status": "ready",
                "checks": {
                    "rootfs_network": {"status": "pass"},
                    "huggingface_dns": {"status": "pass"},
                    "modal_cli": {"status": "pass"},
                    "modal_auth": {"status": "pass"},
                },
                "rootfs": {"marker": "1", "network_mode": "networked"},
            },
            sort_keys=True,
        )
    )

    result = run_stage1_remote.run_stage1_remote(
        source_root=source_root,
        step="ingest_all",
        readiness_report=readiness,
        report_path=tmp_path / "stage1-remote-step.json",
    )

    assert result["status"] == "blocked"
    assert "requires --allow-spend" in result["detail"]


def test_mini_kimi_k3_stage1_remote_checks_spend_guard_before_env_file(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    source_root = tmp_path / "mini-kimi-k3-vizuara"
    (source_root / "data").mkdir(parents=True)
    (source_root / "data" / "stage1.py").write_text("# fixture\n")
    readiness = tmp_path / "stage1-remote-readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_remote_readiness",
                "status": "ready",
                "checks": {
                    "rootfs_network": {"status": "pass"},
                    "huggingface_dns": {"status": "pass"},
                    "modal_cli": {"status": "pass"},
                    "modal_auth": {"status": "pass"},
                },
                "rootfs": {"marker": "1", "network_mode": "networked"},
            },
            sort_keys=True,
        )
    )
    missing_env_file = tmp_path / "missing.env"
    report_path = tmp_path / "stage1-remote-step.json"

    result = run_stage1_remote.run_stage1_remote(
        source_root=source_root,
        step="ingest_all",
        readiness_report=readiness,
        report_path=report_path,
        env_file=missing_env_file,
    )

    assert result["status"] == "blocked"
    assert "requires --allow-spend" in result["detail"]
    assert "env_file" not in result
    assert json.loads(report_path.read_text()) == result


def test_mini_kimi_k3_stage1_remote_runs_safe_modal_step(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    source_root = tmp_path / "mini-kimi-k3-vizuara"
    (source_root / "data").mkdir(parents=True)
    (source_root / "data" / "stage1.py").write_text("# fixture\n")
    readiness = tmp_path / "stage1-remote-readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_remote_readiness",
                "status": "ready",
                "checks": {
                    "rootfs_network": {"status": "pass"},
                    "huggingface_dns": {"status": "pass"},
                    "modal_cli": {"status": "pass"},
                    "modal_auth": {"status": "pass"},
                },
                "rootfs": {"marker": "1", "network_mode": "networked"},
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "stage1-remote-step.json"

    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(
            self,
            argv: list[str],
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> CommandResult:
            del env
            self.calls.append(list(argv))
            assert cwd == source_root
            return CommandResult(return_code=0, stdout="ok\n")

    executor = RecordingExecutor()

    result = run_stage1_remote.run_stage1_remote(
        source_root=source_root,
        step="dry_run",
        readiness_report=readiness,
        report_path=report,
        executor=executor,
    )

    assert result["status"] == "completed"
    assert executor.calls == [
        [
            "uv",
            "tool",
            "run",
            "--from",
            "modal",
            "modal",
            "run",
            "data/stage1.py::dry_run",
        ]
    ]
    assert json.loads(report.read_text()) == result


def test_mini_kimi_k3_stage1_remote_loads_modal_env_file(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    source_root = tmp_path / "mini-kimi-k3-vizuara"
    (source_root / "data").mkdir(parents=True)
    (source_root / "data" / "stage1.py").write_text("# fixture\n")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "MODAL_TOKEN_ID=ak-test-id",
                "MODAL_TOKEN_SECRET=sk-test-secret",
                "",
            ]
        )
    )
    readiness = tmp_path / "stage1-remote-readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_remote_readiness",
                "status": "ready",
                "checks": {
                    "rootfs_network": {"status": "pass"},
                    "huggingface_dns": {"status": "pass"},
                    "modal_cli": {"status": "pass"},
                    "modal_auth": {"status": "pass"},
                },
                "rootfs": {"marker": "1", "network_mode": "networked"},
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "stage1-remote-step.json"

    class RecordingExecutor:
        def __init__(self) -> None:
            self.env_by_call: list[dict[str, str] | None] = []

        def run(
            self,
            argv: list[str],
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> CommandResult:
            del argv
            self.env_by_call.append(dict(env) if env is not None else None)
            assert cwd == source_root
            return CommandResult(return_code=0, stdout="ok ak-test-id sk-test-secret\n")

    executor = RecordingExecutor()

    result = run_stage1_remote.run_stage1_remote(
        source_root=source_root,
        step="dry_run",
        readiness_report=readiness,
        report_path=report,
        env_file=env_file,
        executor=executor,
    )

    assert result["status"] == "completed"
    assert result["env_file"] == {
        "path": str(env_file),
        "status": "loaded",
        "keys": ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"],
    }
    assert executor.env_by_call[0]["MODAL_TOKEN_ID"] == "ak-test-id"
    assert executor.env_by_call[0]["MODAL_TOKEN_SECRET"] == "sk-test-secret"
    report_text = report.read_text()
    assert "ak-test-id" not in report_text
    assert "sk-test-secret" not in report_text
    assert json.loads(report_text) == result


def test_mini_kimi_k3_stage1_remote_rejects_mismatched_readiness_env_file(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    source_root = tmp_path / "mini-kimi-k3-vizuara"
    (source_root / "data").mkdir(parents=True)
    (source_root / "data" / "stage1.py").write_text("# fixture\n")
    checked_env_file = tmp_path / "checked.env"
    invoked_env_file = tmp_path / "invoked.env"
    invoked_env_file.write_text(
        "\n".join(
            [
                "MODAL_TOKEN_ID=ak-test-id",
                "MODAL_TOKEN_SECRET=sk-test-secret",
                "",
            ]
        )
    )
    readiness = tmp_path / "stage1-remote-readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_remote_readiness",
                "status": "ready",
                "env_file": {
                    "path": str(checked_env_file),
                    "status": "loaded",
                    "keys": ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"],
                },
                "checks": {
                    "rootfs_network": {"status": "pass"},
                    "huggingface_dns": {"status": "pass"},
                    "modal_cli": {"status": "pass"},
                    "modal_auth": {"status": "pass"},
                },
                "rootfs": {"marker": "1", "network_mode": "networked"},
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "stage1-remote-step.json"

    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(
            self,
            argv: list[str],
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> CommandResult:
            del cwd, env
            self.calls.append(list(argv))
            return CommandResult(return_code=0)

    executor = RecordingExecutor()

    result = run_stage1_remote.run_stage1_remote(
        source_root=source_root,
        step="dry_run",
        readiness_report=readiness,
        report_path=report,
        env_file=invoked_env_file,
        executor=executor,
    )

    assert result["status"] == "blocked"
    assert "readiness env file" in result["detail"]
    assert executor.calls == []
    assert json.loads(report.read_text()) == result


def test_mini_kimi_k3_stage1_remote_rejects_missing_requested_env_file_after_readiness_check(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    source_root = tmp_path / "mini-kimi-k3-vizuara"
    (source_root / "data").mkdir(parents=True)
    (source_root / "data" / "stage1.py").write_text("# fixture\n")
    checked_env_file = tmp_path / "checked.env"
    readiness = tmp_path / "stage1-remote-readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_remote_readiness",
                "status": "ready",
                "env_file": {
                    "path": str(checked_env_file),
                    "status": "loaded",
                    "keys": ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"],
                },
                "checks": {
                    "rootfs_network": {"status": "pass"},
                    "huggingface_dns": {"status": "pass"},
                    "modal_cli": {"status": "pass"},
                    "modal_auth": {"status": "pass"},
                },
                "rootfs": {"marker": "1", "network_mode": "networked"},
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "stage1-remote-step.json"

    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(
            self,
            argv: list[str],
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> CommandResult:
            del cwd, env
            self.calls.append(list(argv))
            return CommandResult(return_code=0)

    executor = RecordingExecutor()

    result = run_stage1_remote.run_stage1_remote(
        source_root=source_root,
        step="dry_run",
        readiness_report=readiness,
        report_path=report,
        executor=executor,
    )

    assert result["status"] == "blocked"
    assert "readiness env file" in result["detail"]
    assert executor.calls == []
    assert json.loads(report.read_text()) == result


def test_mini_kimi_k3_stage1_remote_rejects_ready_report_missing_required_check(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    source_root = tmp_path / "mini-kimi-k3-vizuara"
    (source_root / "data").mkdir(parents=True)
    (source_root / "data" / "stage1.py").write_text("# fixture\n")
    readiness = tmp_path / "stage1-remote-readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_remote_readiness",
                "status": "ready",
                "checks": {
                    "rootfs_network": {"status": "pass"},
                    "huggingface_dns": {"status": "pass"},
                    "modal_cli": {"status": "pass"},
                },
                "rootfs": {"marker": "1", "network_mode": "networked"},
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "stage1-remote-step.json"

    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(
            self,
            argv: list[str],
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> CommandResult:
            del cwd, env
            self.calls.append(list(argv))
            return CommandResult(return_code=0)

    executor = RecordingExecutor()

    result = run_stage1_remote.run_stage1_remote(
        source_root=source_root,
        step="dry_run",
        readiness_report=readiness,
        report_path=report,
        executor=executor,
    )

    assert result["status"] == "blocked"
    assert "missing modal_auth check" in result["detail"]
    assert executor.calls == []
    assert json.loads(report.read_text()) == result


def test_mini_kimi_k3_stage1_remote_rejects_ready_report_without_rootfs_marker(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    source_root = tmp_path / "mini-kimi-k3-vizuara"
    (source_root / "data").mkdir(parents=True)
    (source_root / "data" / "stage1.py").write_text("# fixture\n")
    readiness = tmp_path / "stage1-remote-readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_remote_readiness",
                "status": "ready",
                "checks": {
                    "rootfs_network": {"status": "pass"},
                    "huggingface_dns": {"status": "pass"},
                    "modal_cli": {"status": "pass"},
                    "modal_auth": {"status": "pass"},
                },
                "rootfs": {"network_mode": "networked"},
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "stage1-remote-step.json"

    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(
            self,
            argv: list[str],
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> CommandResult:
            del cwd, env
            self.calls.append(list(argv))
            return CommandResult(return_code=0)

    executor = RecordingExecutor()

    result = run_stage1_remote.run_stage1_remote(
        source_root=source_root,
        step="dry_run",
        readiness_report=readiness,
        report_path=report,
        executor=executor,
    )

    assert result["status"] == "blocked"
    assert "rootfs marker" in result["detail"]
    assert executor.calls == []
    assert json.loads(report.read_text()) == result


def test_mini_kimi_k3_stage1_remote_rejects_blocked_rootfs_network_check(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    source_root = tmp_path / "mini-kimi-k3-vizuara"
    (source_root / "data").mkdir(parents=True)
    (source_root / "data" / "stage1.py").write_text("# fixture\n")
    readiness = tmp_path / "stage1-remote-readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_remote_readiness",
                "status": "ready",
                "checks": {
                    "rootfs_network": {
                        "status": "blocked",
                        "network_mode": "offline",
                    },
                    "huggingface_dns": {"status": "pass"},
                    "modal_cli": {"status": "pass"},
                    "modal_auth": {"status": "pass"},
                },
                "rootfs": {"marker": "1", "network_mode": "networked"},
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "stage1-remote-step.json"

    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(
            self,
            argv: list[str],
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> CommandResult:
            del cwd, env
            self.calls.append(list(argv))
            return CommandResult(return_code=0)

    executor = RecordingExecutor()

    result = run_stage1_remote.run_stage1_remote(
        source_root=source_root,
        step="dry_run",
        readiness_report=readiness,
        report_path=report,
        executor=executor,
    )

    assert result["status"] == "blocked"
    assert "rootfs_network=blocked" in result["detail"]
    assert executor.calls == []
    assert json.loads(report.read_text()) == result


def test_mini_kimi_k3_stage1_remote_writes_report_for_missing_env_file(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    source_root = tmp_path / "mini-kimi-k3-vizuara"
    (source_root / "data").mkdir(parents=True)
    (source_root / "data" / "stage1.py").write_text("# fixture\n")
    readiness = tmp_path / "stage1-remote-readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_remote_readiness",
                "status": "ready",
                "checks": {
                    "huggingface_dns": {"status": "pass"},
                    "modal_cli": {"status": "pass"},
                    "modal_auth": {"status": "pass"},
                },
                "rootfs": {"network_mode": "networked"},
            },
            sort_keys=True,
        )
    )
    missing_env_file = tmp_path / "missing.env"
    report = tmp_path / "stage1-remote-step.json"

    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(
            self,
            argv: list[str],
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> CommandResult:
            del cwd, env
            self.calls.append(list(argv))
            return CommandResult(return_code=0)

    executor = RecordingExecutor()

    result = run_stage1_remote.run_stage1_remote(
        source_root=source_root,
        step="dry_run",
        readiness_report=readiness,
        report_path=report,
        env_file=missing_env_file,
        executor=executor,
    )

    assert result["status"] == "blocked"
    assert result["detail"] == f"env file does not exist: {missing_env_file}"
    assert result["env_file"] == {
        "status": "blocked",
        "path": str(missing_env_file),
        "detail": f"env file does not exist: {missing_env_file}",
    }
    assert executor.calls == []
    assert json.loads(report.read_text()) == result


def test_mini_kimi_k3_runner_audit_r1_corpus_plan_passes_when_manifest_covers_plan(
    tmp_path: Path,
):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    plan = tmp_path / "reports" / "r1-corpus-plan.json"
    report = tmp_path / "reports" / "r1-corpus-plan-audit.json"
    for source, values in {
        "fineweb-edu": [1, 2, 3, 4],
        "code-python": [5, 6, 7],
    }.items():
        source_dir = tokens_dir / source
        source_dir.mkdir(parents=True)
        shard = source_dir / "part-00000.bin"
        np.asarray(values, dtype="<u4").tofile(shard)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_token_manifest",
                "sources": {
                    "fineweb-edu": {"shards": ["part-00000.bin"]},
                    "code-python": {"shards": ["part-00000.bin"]},
                },
            },
            sort_keys=True,
        )
    )
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan",
                "status": "planned",
                "target_tokens": 7,
                "total_planned_tokens": 7,
                "source_tokens": {"fineweb-edu": 4, "code-python": 3},
                "sources": {
                    "fineweb-edu": {"target_tokens": 4},
                    "code-python": {"target_tokens": 3},
                },
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "audit-r1-corpus-plan",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--corpus-plan",
            str(plan),
            "--report",
            str(report),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    assert "r1 corpus plan audit: ready" in proc.stdout
    data = json.loads(report.read_text())
    assert data["status"] == "ready"
    assert data["available_tokens"] == 7
    assert data["deficit_tokens"] == 0


def test_mini_kimi_k3_runner_audit_r1_corpus_plan_rejects_stale_shard_metadata(
    tmp_path: Path,
):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    plan = tmp_path / "reports" / "r1-corpus-plan.json"
    report = tmp_path / "reports" / "r1-corpus-plan-audit.json"
    source_dir = tokens_dir / "fineweb-edu"
    source_dir.mkdir(parents=True)
    shard = source_dir / "part-00000.bin"
    np.asarray([1, 2, 3, 4], dtype="<u4").tofile(shard)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_token_manifest",
                "sources": {
                    "fineweb-edu": {
                        "shards": [shard.name],
                        "shard_metadata": {shard.name: {"num_tokens": 100}},
                    }
                },
            },
            sort_keys=True,
        )
    )
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan",
                "status": "planned",
                "target_tokens": 100,
                "total_planned_tokens": 100,
                "sources": {"fineweb-edu": {"target_tokens": 100}},
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "audit-r1-corpus-plan",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--corpus-plan",
            str(plan),
            "--report",
            str(report),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "shard_metadata num_tokens mismatch" in proc.stdout
    assert not report.exists()


def test_mini_kimi_k3_runner_audit_r1_corpus_plan_rejects_duplicate_shards(
    tmp_path: Path,
):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    plan = tmp_path / "reports" / "r1-corpus-plan.json"
    report = tmp_path / "reports" / "r1-corpus-plan-audit.json"
    source_dir = tokens_dir / "fineweb-edu"
    source_dir.mkdir(parents=True)
    shard = source_dir / "part-00000.bin"
    np.asarray([1, 2, 3, 4], dtype="<u4").tofile(shard)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_token_manifest",
                "sources": {"fineweb-edu": {"shards": [shard.name, shard.name]}},
            },
            sort_keys=True,
        )
    )
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan",
                "status": "planned",
                "target_tokens": 8,
                "total_planned_tokens": 8,
                "sources": {"fineweb-edu": {"target_tokens": 8}},
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "audit-r1-corpus-plan",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--corpus-plan",
            str(plan),
            "--report",
            str(report),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "duplicate shard" in proc.stdout
    assert not report.exists()


def test_mini_kimi_k3_runner_audit_r1_corpus_plan_rejects_shard_path_escape(
    tmp_path: Path,
):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    plan = tmp_path / "reports" / "r1-corpus-plan.json"
    report = tmp_path / "reports" / "r1-corpus-plan-audit.json"
    escape = tokens_dir / "outside.bin"
    escape.parent.mkdir(parents=True)
    np.asarray([1, 2, 3, 4], dtype="<u4").tofile(escape)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_token_manifest",
                "sources": {"fineweb-edu": {"shards": ["../outside.bin"]}},
            },
            sort_keys=True,
        )
    )
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan",
                "status": "planned",
                "target_tokens": 4,
                "total_planned_tokens": 4,
                "sources": {"fineweb-edu": {"target_tokens": 4, "estimated_files": 2}},
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "audit-r1-corpus-plan",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--corpus-plan",
            str(plan),
            "--report",
            str(report),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "must stay under" in proc.stdout
    assert not report.exists()


def test_mini_kimi_k3_runner_audit_r1_corpus_plan_rejects_noninteger_metadata(
    tmp_path: Path,
):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    plan = tmp_path / "reports" / "r1-corpus-plan.json"
    report = tmp_path / "reports" / "r1-corpus-plan-audit.json"
    source_dir = tokens_dir / "fineweb-edu"
    source_dir.mkdir(parents=True)
    shard = source_dir / "part-00000.bin"
    np.asarray([1, 2, 3, 4], dtype="<u4").tofile(shard)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_token_manifest",
                "sources": {
                    "fineweb-edu": {
                        "shards": [shard.name],
                        "shard_metadata": {shard.name: {"num_tokens": "4"}},
                    }
                },
            },
            sort_keys=True,
        )
    )
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan",
                "status": "planned",
                "target_tokens": 4,
                "total_planned_tokens": 4,
                "sources": {"fineweb-edu": {"target_tokens": 4, "estimated_files": 2}},
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "audit-r1-corpus-plan",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--corpus-plan",
            str(plan),
            "--report",
            str(report),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "shard_metadata num_tokens must be an integer" in proc.stdout
    assert not report.exists()


def test_mini_kimi_k3_runner_audit_r1_corpus_plan_rejects_inconsistent_plan_total(
    tmp_path: Path,
):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    plan = tmp_path / "reports" / "r1-corpus-plan.json"
    report = tmp_path / "reports" / "r1-corpus-plan-audit.json"
    source_dir = tokens_dir / "fineweb-edu"
    source_dir.mkdir(parents=True)
    shard = source_dir / "part-00000.bin"
    np.asarray([1, 2, 3, 4], dtype="<u4").tofile(shard)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_token_manifest",
                "sources": {"fineweb-edu": {"shards": [shard.name]}},
            },
            sort_keys=True,
        )
    )
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan",
                "status": "planned",
                "target_tokens": 8,
                "total_planned_tokens": 4,
                "sources": {"fineweb-edu": {"target_tokens": 4}},
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "audit-r1-corpus-plan",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--corpus-plan",
            str(plan),
            "--report",
            str(report),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "planned source tokens do not sum to target_tokens" in proc.stdout
    assert not report.exists()


def test_mini_kimi_k3_runner_audit_r1_corpus_plan_rejects_source_token_mismatch(
    tmp_path: Path,
):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    plan = tmp_path / "reports" / "r1-corpus-plan.json"
    report = tmp_path / "reports" / "r1-corpus-plan-audit.json"
    source_dir = tokens_dir / "fineweb-edu"
    source_dir.mkdir(parents=True)
    shard = source_dir / "part-00000.bin"
    np.asarray([1, 2, 3, 4], dtype="<u4").tofile(shard)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_token_manifest",
                "sources": {"fineweb-edu": {"shards": [shard.name]}},
            },
            sort_keys=True,
        )
    )
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan",
                "status": "planned",
                "target_tokens": 4,
                "total_planned_tokens": 4,
                "source_tokens": {"fineweb-edu": 3},
                "sources": {"fineweb-edu": {"target_tokens": 4}},
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "audit-r1-corpus-plan",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--corpus-plan",
            str(plan),
            "--report",
            str(report),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "source_tokens mismatch" in proc.stdout
    assert not report.exists()


def test_mini_kimi_k3_runner_write_stage1_input_manifest_refuses_overwrite(
    tmp_path: Path,
):
    input_path = tmp_path / "part-00000.jsonl"
    input_path.write_text(json.dumps({"tokens": [1, 2]}) + "\n")
    output = tmp_path / "source-inputs.json"
    output.write_text("existing\n")

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "write-stage1-input-manifest",
            "--output",
            str(output),
            "--input-format",
            "jsonl-tokens",
            "--input",
            str(input_path),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "already exists" in proc.stdout
    assert output.read_text() == "existing\n"


def test_mini_kimi_k3_runner_write_stage1_input_manifest_rejects_duplicate_inputs(
    tmp_path: Path,
):
    input_path = tmp_path / "part-00000.jsonl"
    input_path.write_text(json.dumps({"tokens": [1, 2]}) + "\n")

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "write-stage1-input-manifest",
            "--output",
            str(tmp_path / "source-inputs.json"),
            "--input-format",
            "jsonl-tokens",
            "--input",
            str(input_path),
            "--input",
            str(input_path.resolve()),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "duplicate input file" in proc.stdout


def test_mini_kimi_k3_materialize_stage1_local_blocks_without_download_authority(
    tmp_path: Path,
):
    resolved = tmp_path / "stage1-source-resolution.json"
    plan = tmp_path / "r1-corpus-plan.json"
    report = tmp_path / "materialize.json"
    resolved.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_source_resolution",
                "status": "ready",
                "sources": {
                    "fineweb-edu": {
                        "status": "resolved",
                        "repo": "HuggingFaceFW/fineweb-edu",
                        "config": "sample-100BT",
                        "split": "train",
                        "text_field": "text",
                        "path_prefix": "sample/100BT/",
                    }
                },
            },
            sort_keys=True,
        )
    )
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan",
                "status": "planned",
                "target_tokens": 4,
                "total_planned_tokens": 4,
                "sources": {"fineweb-edu": {"target_tokens": 4, "estimated_files": 2}},
            },
            sort_keys=True,
        )
    )

    result = materialize_stage1_local.materialize_stage1_local(
        source_resolution_report=resolved,
        corpus_plan_report=plan,
        output_root=tmp_path / "stage1-local",
        report_path=report,
        allow_download=False,
        max_files_per_source=None,
        file_lister=lambda repo, prefix: ["sample/100BT/a.parquet"],
        downloader=lambda repo, path, out: out,
    )

    assert result["status"] == "blocked"
    assert result["detail"] == "materialize-stage1-local requires --allow-download"
    assert result["sources"] == {}
    assert report.is_file()


def test_mini_kimi_k3_runner_materialize_stage1_local_dispatches_inside_rootfs(
    tmp_path: Path,
):
    resolved = tmp_path / "stage1-source-resolution.json"
    plan = tmp_path / "r1-corpus-plan.json"
    report = tmp_path / "materialize.json"
    resolved.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_source_resolution",
                "status": "ready",
                "sources": {
                    "fineweb-edu": {
                        "status": "resolved",
                        "repo": "HuggingFaceFW/fineweb-edu",
                        "config": "sample-100BT",
                        "split": "train",
                        "text_field": "text",
                        "path_prefix": "sample/100BT/",
                    }
                },
            },
            sort_keys=True,
        )
    )
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan",
                "status": "planned",
                "target_tokens": 4,
                "total_planned_tokens": 4,
                "sources": {"fineweb-edu": {"target_tokens": 4}},
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "materialize-stage1-local",
            "--source-resolution-report",
            str(resolved),
            "--corpus-plan-report",
            str(plan),
            "--output-root",
            str(tmp_path / "stage1-local"),
            "--report",
            str(report),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "materialize-stage1-local requires --allow-download" in proc.stdout
    assert json.loads(report.read_text())["kind"] == (
        "mini_kimi_k3_stage1_local_materialization"
    )


def test_mini_kimi_k3_materialize_stage1_local_writes_input_manifest_from_resolved_sources(
    tmp_path: Path,
):
    source_file = tmp_path / "hf-cache" / "sample" / "100BT" / "a.parquet"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"PAR1fixture")
    resolved = tmp_path / "stage1-source-resolution.json"
    plan = tmp_path / "r1-corpus-plan.json"
    output_root = tmp_path / "stage1-local"
    report = tmp_path / "materialize.json"
    resolved.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_source_resolution",
                "status": "ready",
                "sources": {
                    "fineweb-edu": {
                        "status": "resolved",
                        "repo": "HuggingFaceFW/fineweb-edu",
                        "config": "sample-100BT",
                        "split": "train",
                        "text_field": "text",
                        "path_prefix": "sample/100BT/",
                    }
                },
            },
            sort_keys=True,
        )
    )
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan",
                "status": "planned",
                "target_tokens": 4,
                "total_planned_tokens": 4,
                "sources": {"fineweb-edu": {"target_tokens": 4, "estimated_files": 2}},
            },
            sort_keys=True,
        )
    )

    result = materialize_stage1_local.materialize_stage1_local(
        source_resolution_report=resolved,
        corpus_plan_report=plan,
        output_root=output_root,
        report_path=report,
        allow_download=True,
        max_files_per_source=1,
        file_lister=lambda repo, prefix: [
            "sample/100BT/a.parquet",
            "sample/100BT/b.parquet",
        ],
        downloader=lambda repo, path, out: source_file,
    )

    assert result["status"] == "ready"
    assert result["sources"]["fineweb-edu"]["target_tokens"] == 4
    assert result["sources"]["fineweb-edu"]["files_selected"] == 1
    input_manifest = output_root / "fineweb-edu-inputs.json"
    assert input_manifest.is_file()
    manifest = json.loads(input_manifest.read_text())
    assert manifest["kind"] == "mini_kimi_k3_source_input_manifest"
    assert manifest["input_format"] == "parquet-text"
    assert manifest["text_field"] == "text"
    assert manifest["inputs"] == [
        {
            "path": str(source_file.resolve()),
            "bytes": len(b"PAR1fixture"),
            "sha256": hashlib.sha256(b"PAR1fixture").hexdigest(),
        }
    ]


def test_mini_kimi_k3_materialize_stage1_local_uses_plan_estimated_files(
    tmp_path: Path,
):
    source_file = tmp_path / "hf-cache" / "a.parquet"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"PAR1fixture")
    resolved = tmp_path / "stage1-source-resolution.json"
    plan = tmp_path / "r1-corpus-plan.json"
    resolved.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_source_resolution",
                "status": "ready",
                "sources": {
                    "fineweb-edu": {
                        "status": "resolved",
                        "repo": "HuggingFaceFW/fineweb-edu",
                        "config": "sample-100BT",
                        "split": "train",
                        "text_field": "text",
                        "path_prefix": "sample/100BT/",
                    }
                },
            },
            sort_keys=True,
        )
    )
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan",
                "status": "planned",
                "target_tokens": 8,
                "total_planned_tokens": 8,
                "sources": {"fineweb-edu": {"target_tokens": 8, "estimated_files": 2}},
            },
            sort_keys=True,
        )
    )
    downloaded = []

    def fake_download(repo: str, path: str, out: Path) -> Path:
        downloaded.append(path)
        return source_file

    result = materialize_stage1_local.materialize_stage1_local(
        source_resolution_report=resolved,
        corpus_plan_report=plan,
        output_root=tmp_path / "stage1-local",
        report_path=tmp_path / "materialize.json",
        allow_download=True,
        max_files_per_source=None,
        file_lister=lambda repo, prefix: [
            "sample/100BT/a.parquet",
            "sample/100BT/b.parquet",
            "sample/100BT/c.parquet",
        ],
        downloader=fake_download,
    )

    assert downloaded == ["sample/100BT/a.parquet", "sample/100BT/b.parquet"]
    assert result["sources"]["fineweb-edu"]["file_budget"] == 2
    assert result["sources"]["fineweb-edu"]["files_selected"] == 2


def test_mini_kimi_k3_materialize_stage1_local_blocks_when_plan_needs_too_many_files(
    tmp_path: Path,
):
    resolved = tmp_path / "stage1-source-resolution.json"
    plan = tmp_path / "r1-corpus-plan.json"
    report = tmp_path / "materialize.json"
    resolved.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_source_resolution",
                "status": "ready",
                "sources": {
                    "fineweb-edu": {
                        "status": "resolved",
                        "repo": "HuggingFaceFW/fineweb-edu",
                        "config": "sample-100BT",
                        "split": "train",
                        "text_field": "text",
                        "path_prefix": "sample/100BT/",
                    }
                },
            },
            sort_keys=True,
        )
    )
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan",
                "status": "planned",
                "target_tokens": 8,
                "total_planned_tokens": 8,
                "sources": {"fineweb-edu": {"target_tokens": 8, "estimated_files": 3}},
            },
            sort_keys=True,
        )
    )

    result = materialize_stage1_local.materialize_stage1_local(
        source_resolution_report=resolved,
        corpus_plan_report=plan,
        output_root=tmp_path / "stage1-local",
        report_path=report,
        allow_download=True,
        max_files_per_source=None,
        file_lister=lambda repo, prefix: [
            "sample/100BT/a.parquet",
            "sample/100BT/b.parquet",
        ],
        downloader=lambda repo, path, out: out,
    )

    assert result["status"] == "blocked"
    assert "requires 3 parquet file(s), found 2" in result["detail"]
    assert result["sources"]["fineweb-edu"]["status"] == "blocked"
    assert not (tmp_path / "stage1-local" / "fineweb-edu-inputs.json").exists()


def test_mini_kimi_k3_build_decontamination_index_writes_launch_grade_report(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    resolved = tmp_path / "stage1-source-resolution.json"
    index_path = tmp_path / "decon" / "index.npz"
    report = tmp_path / "reports" / "decontamination.json"
    resolved.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_source_resolution",
                "status": "ready",
                "benchmarks": {
                    "bench": {
                        "status": "resolved",
                        "repo": "local/bench",
                        "config": None,
                        "split": "test",
                    }
                },
            },
            sort_keys=True,
        )
    )

    result = build_decontamination_index.build_decontamination_index(
        source_resolution_report=resolved,
        index_path=index_path,
        report_path=report,
        benchmark_loader=lambda benchmark, resolved: [
            {
                "question": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu",
                "choices": ["first choice", "second choice"],
            }
        ],
        benchmark_specs=[
            build_decontamination_index.BenchmarkSpec(
                name="bench",
                fields=["question"],
                list_fields=["choices"],
            )
        ],
    )

    assert result["kind"] == "mini_kimi_k3_decontamination_report"
    assert result["schema_version"] == 1
    assert result["status"] == "completed"
    assert result["scope"] == "launch_grade_benchmark_suite"
    assert result["benchmarks_covered"] == ["bench"]
    assert result["benchmark_stats"]["bench"]["rows"] == 1
    assert result["self_check"]["rows_missed"] == 0
    assert index_path.is_file()
    assert report.is_file()


def test_mini_kimi_k3_runner_tokenizer_install_refuses_overwrite(tmp_path: Path):
    source = tmp_path / "source-tokenizer"
    output = tmp_path / "assets" / "kimi-k3-tokenizer"
    source.mkdir()
    output.mkdir(parents=True)
    (output / "config.json").write_text("existing\n")
    for name in (
        "config.json",
        "tokenizer_config.json",
        "generation_config.json",
        "tokenization_kimi.py",
        "encoding_k3.py",
        "tiktoken.model",
    ):
        (source / name).write_text(f"{name}\n")

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "install-tokenizer-assets",
            "--source-tokenizer-dir",
            str(source),
            "--output-dir",
            str(output),
            "--fingerprint-report",
            str(tmp_path / "report.json"),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "already exists" in proc.stdout
    assert (output / "config.json").read_text() == "existing\n"


def test_mini_kimi_k3_runner_imports_first_party_stage1_shards(tmp_path: Path):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    source_dir = tmp_path / "stage1" / "tokens" / "fineweb-edu"
    source_dir.mkdir(parents=True)
    shard = source_dir / "fineweb-edu-w000-s0000.bin"
    payload = np.asarray([1, 2, 3], dtype="<u4")
    payload.tofile(shard)
    (source_dir / "fineweb-edu-w000-s0000.json").write_text(
        json.dumps(
            {
                "source": "fineweb-edu",
                "worker": 0,
                "seq": 0,
                "tokens": 3,
                "dtype": "uint32",
            },
            sort_keys=True,
        )
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_token_manifest",
                "sources": {"fineweb-edu": {"shards": []}},
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "import-shards",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--source",
            "fineweb-edu",
            "--source-dir",
            str(source_dir),
            "--link-mode",
            "symlink",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    imported = tokens_dir / "fineweb-edu" / shard.name
    assert imported.is_file()
    assert imported.is_symlink()
    data = json.loads(manifest.read_text())
    assert data["sources"]["fineweb-edu"]["shards"] == [shard.name]
    metadata = data["sources"]["fineweb-edu"]["shard_metadata"][shard.name]
    assert metadata["bytes"] == 12
    assert metadata["num_tokens"] == 3
    assert metadata["source_sidecar"] == str(shard.with_suffix(".json"))
    assert metadata["sha256"] == hashlib.sha256(payload.tobytes()).hexdigest()
    assert (
        metadata["source_sidecar_sha256"]
        == hashlib.sha256(shard.with_suffix(".json").read_bytes()).hexdigest()
    )


def test_mini_kimi_k3_runner_import_shards_rejects_sidecar_token_mismatch(
    tmp_path: Path,
):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    source_dir = tmp_path / "stage1" / "tokens" / "fineweb-edu"
    source_dir.mkdir(parents=True)
    shard = source_dir / "fineweb-edu-w000-s0000.bin"
    np.asarray([1, 2, 3], dtype="<u4").tofile(shard)
    (source_dir / "fineweb-edu-w000-s0000.json").write_text(
        json.dumps({"source": "fineweb-edu", "tokens": 2, "dtype": "uint32"})
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"sources": {"fineweb-edu": {"shards": []}}}, sort_keys=True)
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "import-shards",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--source",
            "fineweb-edu",
            "--source-dir",
            str(source_dir),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "sidecar token count mismatch" in proc.stdout
    assert json.loads(manifest.read_text())["sources"]["fineweb-edu"]["shards"] == []
    assert not (tokens_dir / "fineweb-edu" / shard.name).exists()


def test_mini_kimi_k3_runner_builds_local_pretokenized_shards(tmp_path: Path):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    input_path = tmp_path / "tokens.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"tokens": [1, 2, 3]}),
                json.dumps({"tokens": [4, 5]}),
            ]
        )
        + "\n"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_token_manifest",
                "sources": {"fineweb-edu": {"shards": []}},
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "build-local-shards",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--source",
            "fineweb-edu",
            "--input",
            str(input_path),
            "--input-format",
            "jsonl-tokens",
            "--shard-tokens",
            "4",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    data = json.loads(manifest.read_text())
    assert data["sources"]["fineweb-edu"]["shards"] == [
        "fineweb-edu-local-s0000.bin",
        "fineweb-edu-local-s0001.bin",
    ]
    first = tokens_dir / "fineweb-edu" / "fineweb-edu-local-s0000.bin"
    second = tokens_dir / "fineweb-edu" / "fineweb-edu-local-s0001.bin"
    assert np.fromfile(first, dtype="<u4").tolist() == [1, 2, 3, 4]
    assert np.fromfile(second, dtype="<u4").tolist() == [5]
    sidecar = json.loads(first.with_suffix(".json").read_text())
    assert sidecar["source"] == "fineweb-edu"
    assert sidecar["dtype"] == "uint32"
    assert sidecar["tokens"] == 4
    assert sidecar["input_format"] == "jsonl-tokens"
    assert (
        data["sources"]["fineweb-edu"]["shard_metadata"]["fineweb-edu-local-s0000.bin"][
            "num_tokens"
        ]
        == 4
    )


def test_mini_kimi_k3_runner_build_local_shards_rejects_token_overflow(
    tmp_path: Path,
):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    input_path = tmp_path / "tokens.jsonl"
    input_path.write_text(json.dumps({"tokens": [163_840]}) + "\n")
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"sources": {"fineweb-edu": {"shards": []}}}, sort_keys=True)
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "build-local-shards",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--source",
            "fineweb-edu",
            "--input",
            str(input_path),
            "--input-format",
            "jsonl-tokens",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "outside Kimi K3 vocabulary" in proc.stdout
    assert json.loads(manifest.read_text())["sources"]["fineweb-edu"]["shards"] == []


def test_mini_kimi_k3_runner_build_local_shards_applies_decontamination_index(
    tmp_path: Path,
):
    from experiments.mini_kimi_k3.decontaminate_local import NgramIndex

    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    source = tmp_path / "source.jsonl"
    benchmark_text = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    )
    source.write_text(
        "\n".join(
            [
                json.dumps({"text": f"prefix {benchmark_text} suffix"}),
                json.dumps(
                    {
                        "text": "clean document with enough words to avoid the benchmark phrase entirely"
                    }
                ),
            ]
        )
        + "\n"
    )
    decontamination_index = tmp_path / "decontamination-index.npz"
    NgramIndex.build({"bench": [benchmark_text]}).save(decontamination_index)
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"sources": {"fineweb-edu": {"shards": []}}}, sort_keys=True)
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "build-local-shards",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--source",
            "fineweb-edu",
            "--input",
            str(source),
            "--input-format",
            "jsonl-text",
            "--tokenizer-dir",
            str(
                REPO_ROOT
                / "experiments"
                / "mini_kimi_k3"
                / "assets"
                / "kimi-k3-tokenizer"
            ),
            "--shard-tokens",
            "1000",
            "--decontamination-index",
            str(decontamination_index),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    data = json.loads(manifest.read_text())
    local_build = data["sources"]["fineweb-edu"]["local_builds"][0]
    assert local_build["documents"] == 1
    assert local_build["decontamination"] == {
        "applied": True,
        "index": str(decontamination_index),
        "min_matches": 1,
        "documents_seen": 2,
        "documents_kept": 1,
        "documents_removed": 1,
        "hits_by_benchmark": {"bench": 1},
    }


def test_mini_kimi_k3_runner_prepares_local_stage1_from_token_jsonl(
    tmp_path: Path,
):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    reports_dir = tmp_path / "reports"
    input_path = tmp_path / "tokens.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"tokens": [1, 2, 3]}),
                json.dumps({"tokens": [4, 5, 6]}),
            ]
        )
        + "\n"
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "prepare-stage1-local",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--reports-dir",
            str(reports_dir),
            "--source",
            "fineweb-edu",
            "--dataset-id",
            "local/fineweb-edu-fixture",
            "--snapshot",
            "2026-08-fixture",
            "--license",
            "fixture-license",
            "--tokenizer-sha256",
            "tokenizer-sha",
            "--input",
            str(input_path),
            "--input-format",
            "jsonl-tokens",
            "--shard-tokens",
            "4",
            "--allow-placeholder-decontamination",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    manifest_data = json.loads(manifest.read_text())
    assert manifest_data["kind"] == "mini_kimi_k3_token_manifest"
    assert manifest_data["tokenizer"]["sha256"] == "tokenizer-sha"
    assert manifest_data["decontamination"]["status"] == "pending-local-review"
    assert manifest_data["decontamination"]["sha256"].startswith("pending-")
    source = manifest_data["sources"]["fineweb-edu"]
    assert source["dataset_id"] == "local/fineweb-edu-fixture"
    assert source["provenance"]["snapshot"] == "2026-08-fixture"
    assert source["provenance"]["license"] == "fixture-license"
    assert source["shards"] == [
        "fineweb-edu-local-s0000.bin",
        "fineweb-edu-local-s0001.bin",
    ]
    provenance_report = Path(source["provenance"]["report"])
    assert provenance_report.is_file()
    assert (
        hashlib.sha256(provenance_report.read_bytes()).hexdigest()
        == source["provenance"]["sha256"]
    )
    provenance = json.loads(provenance_report.read_text())
    assert provenance["kind"] == "mini_kimi_k3_source_provenance"
    assert provenance["source"] == "fineweb-edu"
    assert provenance["total_tokens"] == 6
    assert "sha256" not in provenance
    assert (
        provenance["input_files"][0]["sha256"]
        == hashlib.sha256(input_path.read_bytes()).hexdigest()
    )


def test_mini_kimi_k3_runner_prepare_stage1_caps_source_tokens(tmp_path: Path):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    reports_dir = tmp_path / "reports"
    input_path = tmp_path / "tokens.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"tokens": [1, 2, 3]}),
                json.dumps({"tokens": [4, 5, 6, 7]}),
            ]
        )
        + "\n"
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "prepare-stage1-local",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--reports-dir",
            str(reports_dir),
            "--source",
            "fineweb-edu",
            "--dataset-id",
            "local/fineweb-edu-fixture",
            "--snapshot",
            "2026-08-fixture",
            "--license",
            "fixture-license",
            "--tokenizer-sha256",
            "tokenizer-sha",
            "--input",
            str(input_path),
            "--input-format",
            "jsonl-tokens",
            "--shard-tokens",
            "4",
            "--max-tokens",
            "5",
            "--allow-placeholder-decontamination",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    first = tokens_dir / "fineweb-edu" / "fineweb-edu-local-s0000.bin"
    second = tokens_dir / "fineweb-edu" / "fineweb-edu-local-s0001.bin"
    assert np.fromfile(first, dtype="<u4").tolist() == [1, 2, 3, 4]
    assert np.fromfile(second, dtype="<u4").tolist() == [5]
    provenance = json.loads((reports_dir / "fineweb-edu-provenance.json").read_text())
    assert provenance["total_tokens"] == 5
    assert provenance["max_tokens"] == 5
    local_build = json.loads(manifest.read_text())["sources"]["fineweb-edu"][
        "local_builds"
    ][0]
    assert local_build["max_tokens"] == 5
    assert local_build["shards"] == [
        {"name": "fineweb-edu-local-s0000.bin", "tokens": 4},
        {"name": "fineweb-edu-local-s0001.bin", "tokens": 1},
    ]


def test_mini_kimi_k3_runner_prepare_stage1_applies_decontamination_index(
    tmp_path: Path,
):
    from experiments.mini_kimi_k3.decontaminate_local import NgramIndex

    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    reports_dir = tmp_path / "reports"
    input_path = tmp_path / "source.jsonl"
    benchmark_text = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    )
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"text": f"prefix {benchmark_text} suffix"}),
                json.dumps(
                    {
                        "text": "clean document with enough words to avoid the benchmark phrase entirely"
                    }
                ),
            ]
        )
        + "\n"
    )
    decontamination_index = tmp_path / "decontamination-index.npz"
    NgramIndex.build({"bench": [benchmark_text]}).save(decontamination_index)

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "prepare-stage1-local",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--reports-dir",
            str(reports_dir),
            "--source",
            "fineweb-edu",
            "--dataset-id",
            "local/fineweb-edu-fixture",
            "--snapshot",
            "2026-08-fixture",
            "--license",
            "fixture-license",
            "--tokenizer-sha256",
            "tokenizer-sha",
            "--input",
            str(input_path),
            "--input-format",
            "jsonl-text",
            "--tokenizer-dir",
            str(
                REPO_ROOT
                / "experiments"
                / "mini_kimi_k3"
                / "assets"
                / "kimi-k3-tokenizer"
            ),
            "--shard-tokens",
            "1000",
            "--decontamination-index",
            str(decontamination_index),
            "--allow-placeholder-decontamination",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    manifest_data = json.loads(manifest.read_text())
    local_build = manifest_data["sources"]["fineweb-edu"]["local_builds"][0]
    assert local_build["documents"] == 1
    assert local_build["decontamination"] == {
        "applied": True,
        "index": str(decontamination_index),
        "min_matches": 1,
        "documents_seen": 2,
        "documents_kept": 1,
        "documents_removed": 1,
        "hits_by_benchmark": {"bench": 1},
    }


def test_mini_kimi_k3_runner_prepares_local_stage1_from_token_parquet(
    tmp_path: Path,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    reports_dir = tmp_path / "reports"
    input_path = tmp_path / "tokens.parquet"
    table = pa.table({"tokens": [[1, 2, 3], [4, 5, 6]]})
    pq.write_table(table, input_path)

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "prepare-stage1-local",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--reports-dir",
            str(reports_dir),
            "--source",
            "fineweb-edu",
            "--dataset-id",
            "local/fineweb-edu-fixture",
            "--snapshot",
            "2026-08-fixture",
            "--license",
            "fixture-license",
            "--tokenizer-sha256",
            "tokenizer-sha",
            "--input",
            str(input_path),
            "--input-format",
            "parquet-tokens",
            "--shard-tokens",
            "4",
            "--allow-placeholder-decontamination",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    first = tokens_dir / "fineweb-edu" / "fineweb-edu-local-s0000.bin"
    second = tokens_dir / "fineweb-edu" / "fineweb-edu-local-s0001.bin"
    assert np.fromfile(first, dtype="<u4").tolist() == [1, 2, 3, 4]
    assert np.fromfile(second, dtype="<u4").tolist() == [5, 6]
    provenance = json.loads((reports_dir / "fineweb-edu-provenance.json").read_text())
    assert provenance["input_format"] == "parquet-tokens"
    assert provenance["total_tokens"] == 6


def test_mini_kimi_k3_runner_prepare_stage1_accepts_input_manifest(
    tmp_path: Path,
):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    reports_dir = tmp_path / "reports"
    input_path = tmp_path / "tokens.jsonl"
    input_path.write_text(json.dumps({"tokens": [1, 2, 3, 4]}) + "\n")
    input_manifest = tmp_path / "source-inputs.json"
    input_manifest.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_source_input_manifest",
                "schema_version": 1,
                "input_format": "jsonl-tokens",
                "inputs": [
                    {
                        "path": input_path.name,
                        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "prepare-stage1-local",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--reports-dir",
            str(reports_dir),
            "--source",
            "fineweb-edu",
            "--dataset-id",
            "local/fineweb-edu-fixture",
            "--snapshot",
            "2026-08-fixture",
            "--license",
            "fixture-license",
            "--tokenizer-sha256",
            "tokenizer-sha",
            "--input-manifest",
            str(input_manifest),
            "--allow-placeholder-decontamination",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    shard = tokens_dir / "fineweb-edu" / "fineweb-edu-local-s0000.bin"
    assert np.fromfile(shard, dtype="<u4").tolist() == [1, 2, 3, 4]
    provenance = json.loads((reports_dir / "fineweb-edu-provenance.json").read_text())
    assert provenance["input_manifest"]["path"] == str(input_manifest)
    assert (
        provenance["input_manifest"]["sha256"]
        == hashlib.sha256(input_manifest.read_bytes()).hexdigest()
    )
    assert provenance["input_files"][0]["path"] == str(input_path.resolve())


def test_mini_kimi_k3_runner_inspects_stage1_input_manifest_without_writing_shards(
    tmp_path: Path,
):
    input_path = tmp_path / "tokens.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"tokens": [1, 2, 3]}),
                json.dumps({"tokens": [4, 5]}),
            ]
        )
        + "\n"
    )
    input_manifest = tmp_path / "source-inputs.json"
    input_manifest.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_source_input_manifest",
                "schema_version": 1,
                "input_format": "jsonl-tokens",
                "inputs": [
                    {
                        "path": input_path.name,
                        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "reports" / "source-inputs-inspection.json"
    tokens_dir = tmp_path / "tokens"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "inspect-stage1-inputs",
            "--input-manifest",
            str(input_manifest),
            "--report",
            str(report),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    assert not tokens_dir.exists()
    data = json.loads(report.read_text())
    assert data["kind"] == "mini_kimi_k3_stage1_input_inspection"
    assert data["status"] == "pass"
    assert data["input_manifest"]["path"] == str(input_manifest)
    assert (
        data["input_manifest"]["sha256"]
        == hashlib.sha256(input_manifest.read_bytes()).hexdigest()
    )
    assert data["input_format"] == "jsonl-tokens"
    assert data["total_documents"] == 2
    assert data["total_tokens"] == 5
    assert data["input_files"] == [
        {
            "path": str(input_path.resolve()),
            "bytes": input_path.stat().st_size,
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "documents": 2,
            "tokens": 5,
            "text_field": "text",
            "tokens_field": "tokens",
        }
    ]


def test_mini_kimi_k3_runner_inspects_multiple_stage1_input_manifests(
    tmp_path: Path,
):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps({"tokens": [1, 2, 3]}) + "\n")
    second.write_text(json.dumps({"tokens": [4, 5]}) + "\n")
    first_manifest = tmp_path / "first-inputs.json"
    second_manifest = tmp_path / "second-inputs.json"
    first_manifest.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_source_input_manifest",
                "schema_version": 1,
                "input_format": "jsonl-tokens",
                "inputs": [
                    {
                        "path": first.name,
                        "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        )
    )
    second_manifest.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_source_input_manifest",
                "schema_version": 1,
                "input_format": "jsonl-tokens",
                "inputs": [
                    {
                        "path": second.name,
                        "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "reports" / "source-inputs-inspection.json"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "inspect-stage1-inputs",
            "--input-manifest",
            str(first_manifest),
            "--input-manifest",
            str(second_manifest),
            "--report",
            str(report),
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
    assert data["total_files"] == 2
    assert data["total_documents"] == 2
    assert data["total_tokens"] == 5
    assert data["input_manifest"]["paths"] == [
        str(first_manifest),
        str(second_manifest),
    ]
    assert len(data["input_files"]) == 2


def test_mini_kimi_k3_runner_inspects_multiple_stage1_input_manifests_with_mixed_text_fields(
    tmp_path: Path,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    code = tmp_path / "code.parquet"
    text = tmp_path / "text.parquet"
    pq.write_table(pa.table({"code": ["def f():\n    return 1\n"]}), code)
    pq.write_table(pa.table({"text": ["A short document."]}), text)
    code_manifest = tmp_path / "code-inputs.json"
    text_manifest = tmp_path / "text-inputs.json"
    code_manifest.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_source_input_manifest",
                "schema_version": 1,
                "input_format": "parquet-text",
                "text_field": "code",
                "inputs": [
                    {
                        "path": code.name,
                        "sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        )
    )
    text_manifest.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_source_input_manifest",
                "schema_version": 1,
                "input_format": "parquet-text",
                "text_field": "text",
                "inputs": [
                    {
                        "path": text.name,
                        "sha256": hashlib.sha256(text.read_bytes()).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "reports" / "source-inputs-inspection.json"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "inspect-stage1-inputs",
            "--input-manifest",
            str(code_manifest),
            "--input-manifest",
            str(text_manifest),
            "--report",
            str(report),
            "--tokenizer-dir",
            str(
                REPO_ROOT
                / "experiments"
                / "mini_kimi_k3"
                / "assets"
                / "kimi-k3-tokenizer"
            ),
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
    assert data["total_files"] == 2
    assert data["total_documents"] == 2
    assert data["input_files"][0]["text_field"] == "code"
    assert data["input_files"][1]["text_field"] == "text"


def test_mini_kimi_k3_runner_inspect_stage1_rejects_input_manifest_hash_mismatch(
    tmp_path: Path,
):
    input_path = tmp_path / "tokens.jsonl"
    input_path.write_text(json.dumps({"tokens": [1, 2, 3, 4]}) + "\n")
    input_manifest = tmp_path / "source-inputs.json"
    input_manifest.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_source_input_manifest",
                "schema_version": 1,
                "input_format": "jsonl-tokens",
                "inputs": [{"path": str(input_path), "sha256": "stale"}],
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "inspect-stage1-inputs",
            "--input-manifest",
            str(input_manifest),
            "--report",
            str(tmp_path / "reports" / "source-inputs-inspection.json"),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "input manifest sha256 mismatch" in proc.stdout


def test_mini_kimi_k3_runner_inspect_stage1_rejects_duplicate_input_files(
    tmp_path: Path,
):
    input_path = tmp_path / "tokens.jsonl"
    input_path.write_text(json.dumps({"tokens": [1, 2, 3, 4]}) + "\n")
    input_manifest = tmp_path / "source-inputs.json"
    input_manifest.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_source_input_manifest",
                "schema_version": 1,
                "input_format": "jsonl-tokens",
                "inputs": [
                    {
                        "path": input_path.name,
                        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    },
                    {
                        "path": str(input_path.resolve()),
                        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    },
                ],
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "inspect-stage1-inputs",
            "--input-manifest",
            str(input_manifest),
            "--report",
            str(tmp_path / "reports" / "source-inputs-inspection.json"),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "duplicate input file" in proc.stdout


def test_mini_kimi_k3_runner_inspect_stage1_fails_below_min_tokens(
    tmp_path: Path,
):
    input_path = tmp_path / "tokens.jsonl"
    input_path.write_text(json.dumps({"tokens": [1, 2, 3, 4]}) + "\n")
    input_manifest = tmp_path / "source-inputs.json"
    input_manifest.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_source_input_manifest",
                "schema_version": 1,
                "input_format": "jsonl-tokens",
                "inputs": [
                    {
                        "path": input_path.name,
                        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "reports" / "source-inputs-inspection.json"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "inspect-stage1-inputs",
            "--input-manifest",
            str(input_manifest),
            "--report",
            str(report),
            "--min-total-tokens",
            "5",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "below required minimum" in proc.stdout
    data = json.loads(report.read_text())
    assert data["status"] == "fail"
    assert data["total_tokens"] == 4
    assert data["min_total_tokens"] == 5
    assert data["detail"] == "total token count below required minimum (4 < 5)"


def test_mini_kimi_k3_runner_prepare_stage1_rejects_input_manifest_hash_mismatch(
    tmp_path: Path,
):
    input_path = tmp_path / "tokens.jsonl"
    input_path.write_text(json.dumps({"tokens": [1, 2, 3, 4]}) + "\n")
    input_manifest = tmp_path / "source-inputs.json"
    input_manifest.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_source_input_manifest",
                "schema_version": 1,
                "input_format": "jsonl-tokens",
                "inputs": [{"path": str(input_path), "sha256": "stale"}],
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "prepare-stage1-local",
            "--manifest",
            str(tmp_path / "data" / "manifest.json"),
            "--tokens-dir",
            str(tmp_path / "data" / "tokens"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--source",
            "fineweb-edu",
            "--dataset-id",
            "local/fineweb-edu-fixture",
            "--snapshot",
            "2026-08-fixture",
            "--license",
            "fixture-license",
            "--tokenizer-sha256",
            "tokenizer-sha",
            "--input-manifest",
            str(input_manifest),
            "--allow-placeholder-decontamination",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "input manifest sha256 mismatch" in proc.stdout


def test_mini_kimi_k3_runner_prepare_stage1_rejects_duplicate_input_files(
    tmp_path: Path,
):
    input_path = tmp_path / "tokens.jsonl"
    input_path.write_text(json.dumps({"tokens": [1, 2, 3, 4]}) + "\n")
    input_manifest = tmp_path / "source-inputs.json"
    input_manifest.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_source_input_manifest",
                "schema_version": 1,
                "input_format": "jsonl-tokens",
                "inputs": [
                    {
                        "path": input_path.name,
                        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "prepare-stage1-local",
            "--manifest",
            str(tmp_path / "data" / "manifest.json"),
            "--tokens-dir",
            str(tmp_path / "data" / "tokens"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--source",
            "fineweb-edu",
            "--dataset-id",
            "local/fineweb-edu-fixture",
            "--snapshot",
            "2026-08-fixture",
            "--license",
            "fixture-license",
            "--tokenizer-sha256",
            "tokenizer-sha",
            "--input-manifest",
            str(input_manifest),
            "--input",
            str(input_path.resolve()),
            "--allow-placeholder-decontamination",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "duplicate input file" in proc.stdout


def test_mini_kimi_k3_runner_prepare_stage1_requires_reviewed_decontam_report(
    tmp_path: Path,
):
    input_path = tmp_path / "tokens.jsonl"
    input_path.write_text(json.dumps({"tokens": [1, 2, 3, 4]}) + "\n")

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "prepare-stage1-local",
            "--manifest",
            str(tmp_path / "data" / "manifest.json"),
            "--tokens-dir",
            str(tmp_path / "data" / "tokens"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--source",
            "fineweb-edu",
            "--dataset-id",
            "local/fineweb-edu-fixture",
            "--snapshot",
            "2026-08-fixture",
            "--license",
            "fixture-license",
            "--tokenizer-sha256",
            "tokenizer-sha",
            "--input",
            str(input_path),
            "--input-format",
            "jsonl-tokens",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "requires --decontamination-report" in proc.stdout


def test_mini_kimi_k3_runner_decontaminates_local_jsonl_text(tmp_path: Path):
    benchmark = tmp_path / "bench.jsonl"
    source = tmp_path / "source.jsonl"
    output = tmp_path / "clean.jsonl"
    report = tmp_path / "decontamination.json"
    benchmark_text = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    )
    benchmark.write_text(json.dumps({"text": benchmark_text}) + "\n")
    source.write_text(
        "\n".join(
            [
                json.dumps({"text": f"prefix {benchmark_text} suffix"}),
                json.dumps(
                    {
                        "text": "clean document with enough words to avoid the benchmark phrase entirely"
                    }
                ),
            ]
        )
        + "\n"
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "decontaminate-local",
            "--benchmark",
            str(benchmark),
            "--input",
            str(source),
            "--output",
            str(output),
            "--report",
            str(report),
            "--source",
            "fineweb-edu",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    kept_rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert kept_rows == [
        {
            "text": "clean document with enough words to avoid the benchmark phrase entirely"
        }
    ]
    data = json.loads(report.read_text())
    assert data["kind"] == "mini_kimi_k3_decontamination_report"
    assert data["schema_version"] == 1
    assert data["status"] == "completed"
    assert data["scope"] == "local_explicit_benchmarks"
    assert data["source"] == "fineweb-edu"
    assert data["documents_scanned"] == 2
    assert data["documents_removed"] == 1
    assert data["documents_kept"] == 1
    assert data["benchmarks_covered"] == ["bench"]
    assert data["ngram_hits_by_benchmark"] == {"bench": 1}
    assert (
        data["input_files"][0]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )


def test_mini_kimi_k3_runner_decontaminate_local_rejects_empty_benchmark(
    tmp_path: Path,
):
    benchmark = tmp_path / "bench.jsonl"
    source = tmp_path / "source.jsonl"
    benchmark.write_text(json.dumps({"text": "too short"}) + "\n")
    source.write_text(
        json.dumps({"text": "some long enough document for scanning"}) + "\n"
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "decontaminate-local",
            "--benchmark",
            str(benchmark),
            "--input",
            str(source),
            "--output",
            str(tmp_path / "clean.jsonl"),
            "--report",
            str(tmp_path / "decontamination.json"),
            "--source",
            "fineweb-edu",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "yielded no 13-gram hashes" in proc.stdout


def test_mini_kimi_k3_runner_compares_forward_oracle_logits(tmp_path: Path):
    first_party_logits = tmp_path / "first_party.pt"
    torchtitan_logits = tmp_path / "torchtitan.pt"
    report = tmp_path / "forward_oracle.json"
    logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    torch.save(logits, first_party_logits)
    torch.save({"logits": logits + 1.0e-7}, torchtitan_logits)

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "compare-forward-oracle",
            "--first-party-logits",
            str(first_party_logits),
            "--torchtitan-logits",
            str(torchtitan_logits),
            "--report",
            str(report),
            "--model-flavor",
            "r1",
            "--max-abs-diff",
            "1e-5",
            "--max-rel-diff",
            "1e-5",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    data = json.loads(report.read_text())
    assert data["kind"] == "mini_kimi_k3_forward_oracle"
    assert data["schema_version"] == 1
    assert data["status"] == "pass"
    assert data["model_flavor"] == "r1"
    assert data["source"]["first_party_logits"] == str(first_party_logits)
    assert data["source"]["torchtitan_logits"] == str(torchtitan_logits)
    assert data["results"]["shape"] == [1, 2, 2]
    assert data["results"]["max_abs_diff"] <= 1.0e-5
    assert data["results"]["max_rel_diff"] <= 1.0e-5


def test_mini_kimi_k3_runner_forward_oracle_compare_fails_on_mismatch(
    tmp_path: Path,
):
    first_party_logits = tmp_path / "first_party.pt"
    torchtitan_logits = tmp_path / "torchtitan.pt"
    report = tmp_path / "forward_oracle.json"
    torch.save(torch.tensor([[1.0, 2.0]]), first_party_logits)
    torch.save(torch.tensor([[1.0, 2.5]]), torchtitan_logits)

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "compare-forward-oracle",
            "--first-party-logits",
            str(first_party_logits),
            "--torchtitan-logits",
            str(torchtitan_logits),
            "--report",
            str(report),
            "--model-flavor",
            "r1",
            "--max-abs-diff",
            "1e-5",
            "--max-rel-diff",
            "1e-5",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "forward oracle comparison failed" in proc.stdout
    data = json.loads(report.read_text())
    assert data["status"] == "fail"
    assert data["results"]["max_abs_diff"] == 0.5


def test_mini_kimi_k3_runner_forward_oracle_compare_rejects_nonfinite_logits(
    tmp_path: Path,
):
    first_party_logits = tmp_path / "first_party.pt"
    torchtitan_logits = tmp_path / "torchtitan.pt"
    report = tmp_path / "forward_oracle.json"
    torch.save(torch.tensor([[1.0, float("nan")]]), first_party_logits)
    torch.save(torch.tensor([[1.0, 2.0]]), torchtitan_logits)

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "compare-forward-oracle",
            "--first-party-logits",
            str(first_party_logits),
            "--torchtitan-logits",
            str(torchtitan_logits),
            "--report",
            str(report),
            "--model-flavor",
            "r1",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    data = json.loads(report.read_text())
    assert data["status"] == "fail"
    assert data["error"] == "first-party logits contain non-finite values"


def test_mini_kimi_k3_runner_reviews_launch_backend_reports(tmp_path: Path):
    forward_oracle = tmp_path / "forward_oracle.json"
    forward_trace = tmp_path / "forward_trace.json"
    output = tmp_path / "launch_backend.json"
    forward_oracle.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_forward_oracle",
                "schema_version": 1,
                "status": "pass",
                "model_flavor": "r1",
                "tolerances": {"max_abs_diff": 1.0e-5, "max_rel_diff": 1.0e-5},
                "results": {"max_abs_diff": 0.0, "max_rel_diff": 0.0},
            },
            sort_keys=True,
        )
    )
    forward_trace.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_forward_trace",
                "schema_version": 1,
                "status": "pass",
                "config": "mini_kimi_k3_r1_contract",
                "model_flavor": "r1_contract",
                "device": "cuda:7",
                "final": {
                    "logits": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0},
                    "norm": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0},
                    "output_attention_residual": {
                        "max_abs_diff": 0.0,
                        "mean_abs_diff": 0.0,
                    },
                },
            },
            sort_keys=True,
        )
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "review-launch-backend",
            "--forward-oracle-report",
            str(forward_oracle),
            "--forward-trace-report",
            str(forward_trace),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    report = json.loads(output.read_text())
    assert report["kind"] == "mini_kimi_k3_launch_backend_review"
    assert report["schema_version"] == 1
    assert report["status"] == "pass"
    assert report["model_flavor"] == "r1"
    assert report["backend"] == "torchtitan.experiments.mini_kimi_k3"
    assert report["config"] == "mini_kimi_k3_r1_contract"
    assert report["evidence"]["forward_oracle_report"] == str(forward_oracle)
    assert report["evidence"]["forward_trace_report"] == str(forward_trace)
    assert report["evidence"]["final_logit_max_abs_diff"] == 0.0


def test_mini_kimi_k3_runner_exposes_forward_oracle_trace_command(tmp_path: Path):
    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "trace-forward-oracle",
            "--help",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    assert "usage: trace_forward_oracle.py" in proc.stdout
    assert "--first-party-state-dict" in proc.stdout


def test_mini_kimi_k3_runner_dumps_torchtitan_tiny_logits(tmp_path: Path):
    output = tmp_path / "torchtitan_logits.pt"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "dump-torchtitan-logits",
            "--config",
            "mini_kimi_k3_tiny_plumbing",
            "--output",
            str(output),
            "--input-ids",
            "1,2,3,4",
            "--seed",
            "1234",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert set(payload) == {
        "config",
        "fidelity_status",
        "input_ids",
        "logits",
        "model_flavor",
        "seed",
    }
    assert payload["config"] == "mini_kimi_k3_tiny_plumbing"
    assert payload["model_flavor"] == "tiny_plumbing"
    assert payload["fidelity_status"] == "tiny_plumbing"
    assert payload["seed"] == 1234
    assert payload["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert tuple(payload["logits"].shape) == (1, 4, 32)


def test_mini_kimi_k3_runner_dump_logits_fails_closed_for_r1_contract(
    tmp_path: Path,
):
    output = tmp_path / "r1_logits.pt"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "dump-torchtitan-logits",
            "--config",
            "mini_kimi_k3_r1_contract",
            "--output",
            str(output),
            "--input-ids",
            "1,2,3,4",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert (
        "Mini-K3 r1 logits dump requires a verified forward-oracle path" in proc.stdout
    )
    assert not output.exists()


def test_mini_kimi_k3_runner_dumps_unverified_r1_candidate_logits(tmp_path: Path):
    output = tmp_path / "r1_candidate_logits.pt"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "dump-torchtitan-logits",
            "--config",
            "mini_kimi_k3_r1_contract",
            "--output",
            str(output),
            "--input-ids",
            "1,2,3,4",
            "--seed",
            "1234",
            "--allow-unverified-r1",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert payload["config"] == "mini_kimi_k3_r1_contract"
    assert payload["model_flavor"] == "r1_contract"
    assert payload["fidelity_status"] == "unverified_candidate"
    assert payload["seed"] == 1234
    assert payload["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert tuple(payload["logits"].shape) == (1, 4, 163_840)
    assert torch.isfinite(payload["logits"]).all()


def test_mini_kimi_k3_aligns_first_party_weights_with_complete_mapping():
    first_party = nn.Module()
    first_party.model = nn.Module()
    first_party.model.embed_tokens = nn.Embedding(3, 2)
    first_party.model.layers = nn.ModuleList([nn.Module()])
    first_party.model.layers[0].mlp = nn.Module()
    first_party.model.layers[0].mlp.gate_proj = nn.Linear(2, 4, bias=False)
    first_party.model.layers[0].block_sparse_moe = nn.Module()
    first_party.model.layers[0].block_sparse_moe.gate = nn.Linear(2, 5, bias=False)

    torch_titan = nn.Module()
    torch_titan.inner = nn.Module()
    torch_titan.inner.embed_tokens = nn.Embedding(3, 2)
    torch_titan.inner.layers = nn.ModuleList([nn.Module()])
    torch_titan.inner.layers[0].feed_forward = nn.Module()
    torch_titan.inner.layers[0].feed_forward.gate_proj = nn.Linear(2, 4, bias=False)
    torch_titan.inner.layers[0].block_sparse_moe = nn.Module()
    torch_titan.inner.layers[0].block_sparse_moe.router = nn.Linear(2, 5, bias=False)

    with torch.no_grad():
        for index, parameter in enumerate(first_party.parameters(), start=1):
            parameter.fill_(float(index))

    report = align_first_party_weights(first_party, torch_titan)

    assert report == {
        "status": "pass",
        "first_party_parameters": 3,
        "torch_titan_parameters": 3,
        "mapped_parameters": 3,
        "first_party_elements": 24,
        "torch_titan_elements": 24,
        "mapped_elements": 24,
        "missing": [],
        "extra": [],
        "shape_mismatches": [],
    }
    assert torch.equal(
        torch_titan.inner.embed_tokens.weight,
        first_party.model.embed_tokens.weight,
    )
    assert torch.equal(
        torch_titan.inner.layers[0].feed_forward.gate_proj.weight,
        first_party.model.layers[0].mlp.gate_proj.weight,
    )
    assert torch.equal(
        torch_titan.inner.layers[0].block_sparse_moe.router.weight,
        first_party.model.layers[0].block_sparse_moe.gate.weight,
    )


def test_mini_kimi_k3_weight_alignment_fails_on_unmapped_parameters():
    first_party = nn.Module()
    first_party.model = nn.Module()
    first_party.model.unmapped = nn.Linear(2, 2, bias=False)
    torch_titan = nn.Module()
    torch_titan.inner = nn.Module()

    try:
        align_first_party_weights(first_party, torch_titan)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected Mini-K3 weight alignment to fail")

    assert "Mini-K3 first-party weight alignment is incomplete" in message
    assert "model.unmapped.weight" in message


def test_mini_kimi_k3_aligns_from_first_party_state_dict():
    state_dict = {
        "model.embed_tokens.weight": torch.full((3, 2), 1.0),
        "lm_head.weight": torch.full((3, 2), 1.0),
        "model.layers.0.mlp.gate_proj.weight": torch.full((4, 2), 2.0),
        "model.layers.0.block_sparse_moe.gate.weight": torch.full((5, 2), 3.0),
        "model.layers.0.block_sparse_moe.gate.expert_load": torch.ones(5),
    }
    torch_titan = nn.Module()
    torch_titan.inner = nn.Module()
    torch_titan.inner.embed_tokens = nn.Embedding(3, 2)
    torch_titan.inner.layers = nn.ModuleList([nn.Module()])
    torch_titan.inner.layers[0].feed_forward = nn.Module()
    torch_titan.inner.layers[0].feed_forward.gate_proj = nn.Linear(2, 4, bias=False)
    torch_titan.inner.layers[0].block_sparse_moe = nn.Module()
    torch_titan.inner.layers[0].block_sparse_moe.router = nn.Linear(2, 5, bias=False)

    report = align_first_party_state_dict(state_dict, torch_titan)

    assert report["status"] == "pass"
    assert report["first_party_parameters"] == 3
    assert report["mapped_elements"] == 24
    assert torch.equal(
        torch_titan.inner.layers[0].block_sparse_moe.router.weight,
        state_dict["model.layers.0.block_sparse_moe.gate.weight"],
    )


def test_mini_kimi_k3_runner_dump_first_party_logits_requires_visible_source_root(
    tmp_path: Path,
):
    output = tmp_path / "first_party_logits.pt"
    source_root = tmp_path / "missing-first-party-source"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "dump-first-party-logits",
            "--source-root",
            str(source_root),
            "--output",
            str(output),
            "--input-ids",
            "1,2,3,4",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "first-party Mini-K3 source root is not visible" in proc.stdout
    assert str(source_root) in proc.stdout
    assert "Mini Kimi K3 scaffold is not trainable yet" not in proc.stdout
    assert not output.exists()


def test_mini_kimi_k3_runner_dump_first_party_logits_requires_complete_source_root(
    tmp_path: Path,
):
    output = tmp_path / "first_party_logits.pt"
    source_root = tmp_path / "first-party-source"
    source_root.mkdir()

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "dump-first-party-logits",
            "--source-root",
            str(source_root),
            "--output",
            str(output),
            "--input-ids",
            "1,2,3,4",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "first-party Mini-K3 source root is incomplete" in proc.stdout
    assert "model/_ref/configuration_kimi_k3.py" in proc.stdout
    assert "model/_ref/modeling_kimi_linear.py" in proc.stdout
    assert "train/ladder.py" in proc.stdout
    assert not output.exists()


def test_mini_kimi_k3_runner_dump_first_party_logits_reports_import_errors(
    tmp_path: Path,
):
    output = tmp_path / "first_party_logits.pt"
    source_root = tmp_path / "first-party-source"
    (source_root / "model" / "_ref").mkdir(parents=True)
    (source_root / "train").mkdir()
    (source_root / "model" / "_ref" / "configuration_kimi_k3.py").write_text(
        "class KimiLinearConfig:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.kwargs = kwargs\n"
    )
    (source_root / "model" / "_ref" / "modeling_kimi_linear.py").write_text(
        "raise ImportError('missing first-party dependency')\n"
    )
    (source_root / "train" / "ladder.py").write_text(
        "def all_configs():\n" "    return {'r1': {'vocab_size': 8}}\n"
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "dump-first-party-logits",
            "--source-root",
            str(source_root),
            "--output",
            str(output),
            "--input-ids",
            "1,2,3,4",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "source root is visible but not importable" in proc.stdout
    assert "missing first-party dependency" in proc.stdout
    assert "Traceback" not in proc.stdout
    assert not output.exists()


def test_mini_kimi_k3_runner_dump_first_party_logits_shims_transformers_hooks(
    tmp_path: Path,
):
    output = tmp_path / "first_party_logits.pt"
    source_root = tmp_path / "first-party-source"
    (source_root / "model" / "_ref").mkdir(parents=True)
    (source_root / "train").mkdir()
    (source_root / "model" / "_ref" / "configuration_kimi_k3.py").write_text(
        "class KimiLinearConfig:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.kwargs = kwargs\n"
        "        self.vocab_size = kwargs['vocab_size']\n"
    )
    (source_root / "model" / "_ref" / "modeling_kimi_linear.py").write_text(
        "import torch\n"
        "from transformers.utils.generic import OutputRecorder, check_model_inputs\n"
        "RECORDER = OutputRecorder(object, index=1)\n"
        "class Output:\n"
        "    def __init__(self, logits):\n"
        "        self.logits = logits\n"
        "class KimiLinearForCausalLM(torch.nn.Module):\n"
        "    def __init__(self, config):\n"
        "        super().__init__()\n"
        "        self.config = config\n"
        "    @check_model_inputs\n"
        "    def forward(self, input_ids):\n"
        "        base = input_ids.to(torch.float32).unsqueeze(-1)\n"
        "        logits = base.repeat(1, 1, self.config.vocab_size)\n"
        "        return Output(logits)\n"
    )
    (source_root / "train" / "ladder.py").write_text(
        "def all_configs():\n" "    return {'r1': {'vocab_size': 8}}\n"
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "dump-first-party-logits",
            "--source-root",
            str(source_root),
            "--output",
            str(output),
            "--input-ids",
            "1,2,3,4",
            "--seed",
            "1234",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert payload["model_flavor"] == "r1"
    assert payload["seed"] == 1234
    assert payload["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert tuple(payload["logits"].shape) == (1, 4, 8)


def test_mini_kimi_k3_runner_dump_first_party_logits_uses_oracle_python(
    tmp_path: Path,
):
    output = tmp_path / "first_party_logits.pt"
    source_root = tmp_path / "first-party-source"
    marker = tmp_path / "oracle_python_invoked.txt"
    oracle_python = tmp_path / "oracle-python"
    (source_root / "model" / "_ref").mkdir(parents=True)
    (source_root / "train").mkdir()
    (source_root / "model" / "_ref" / "configuration_kimi_k3.py").write_text(
        "class KimiLinearConfig:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.vocab_size = kwargs['vocab_size']\n"
    )
    (source_root / "model" / "_ref" / "modeling_kimi_linear.py").write_text(
        "import torch\n"
        "class Output:\n"
        "    def __init__(self, logits):\n"
        "        self.logits = logits\n"
        "class KimiLinearForCausalLM(torch.nn.Module):\n"
        "    def __init__(self, config):\n"
        "        super().__init__()\n"
        "        self.config = config\n"
        "    def forward(self, input_ids, use_cache=None):\n"
        "        self.last_use_cache = use_cache\n"
        "        logits = input_ids.to(torch.float32).unsqueeze(-1)\n"
        "        return Output(logits.repeat(1, 1, self.config.vocab_size))\n"
    )
    (source_root / "train" / "ladder.py").write_text(
        "def all_configs():\n" "    return {'r1': {'vocab_size': 8}}\n"
    )
    oracle_python.write_text(
        "#!/usr/bin/env bash\n" f"printf invoked > {marker}\n" 'exec python "$@"\n'
    )
    oracle_python.chmod(0o755)

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "dump-first-party-logits",
            "--source-root",
            str(source_root),
            "--output",
            str(output),
            "--input-ids",
            "1,2,3,4",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "TORCHTITAN_IN_ROOTFS": "1",
            "MINI_KIMI_K3_ORACLE_PYTHON": str(oracle_python),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    assert marker.read_text() == "invoked"
    assert output.is_file()


def test_mini_kimi_k3_runner_dump_first_party_logits_accepts_device(
    tmp_path: Path,
):
    output = tmp_path / "first_party_logits.pt"
    source_root = tmp_path / "first-party-source"
    (source_root / "model" / "_ref").mkdir(parents=True)
    (source_root / "train").mkdir()
    (source_root / "model" / "_ref" / "configuration_kimi_k3.py").write_text(
        "class KimiLinearConfig:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.vocab_size = kwargs['vocab_size']\n"
    )
    (source_root / "model" / "_ref" / "modeling_kimi_linear.py").write_text(
        "import torch\n"
        "class Output:\n"
        "    def __init__(self, logits):\n"
        "        self.logits = logits\n"
        "class KimiLinearForCausalLM(torch.nn.Module):\n"
        "    def __init__(self, config):\n"
        "        super().__init__()\n"
        "        self.config = config\n"
        "    def forward(self, input_ids):\n"
        "        logits = input_ids.to(torch.float32).unsqueeze(-1)\n"
        "        return Output(logits.repeat(1, 1, self.config.vocab_size))\n"
    )
    (source_root / "train" / "ladder.py").write_text(
        "def all_configs():\n" "    return {'r1': {'vocab_size': 8}}\n"
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "dump-first-party-logits",
            "--source-root",
            str(source_root),
            "--output",
            str(output),
            "--input-ids",
            "1,2,3,4",
            "--device",
            "cpu",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert payload["device"] == "cpu"
    assert payload["prepared_for_training"] is False
    assert payload["prepare_report"] is None
    assert payload["use_cache"] is False
    assert tuple(payload["logits"].shape) == (1, 4, 8)


def test_mini_kimi_k3_runner_dump_first_party_logits_can_prepare_model(
    tmp_path: Path,
):
    output = tmp_path / "first_party_logits.pt"
    source_root = tmp_path / "first-party-source"
    (source_root / "model" / "_ref").mkdir(parents=True)
    (source_root / "train").mkdir()
    (source_root / "model" / "_ref" / "configuration_kimi_k3.py").write_text(
        "class KimiLinearConfig:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.vocab_size = kwargs['vocab_size']\n"
    )
    (source_root / "model" / "_ref" / "modeling_kimi_linear.py").write_text(
        "import torch\n"
        "class Output:\n"
        "    def __init__(self, logits):\n"
        "        self.logits = logits\n"
        "class KimiLinearForCausalLM(torch.nn.Module):\n"
        "    def __init__(self, config):\n"
        "        super().__init__()\n"
        "        self.config = config\n"
        "        self.bias = 0.0\n"
        "    def forward(self, input_ids):\n"
        "        logits = input_ids.to(torch.float32).unsqueeze(-1) + self.bias\n"
        "        return Output(logits.repeat(1, 1, self.config.vocab_size))\n"
    )
    (source_root / "train" / "ladder.py").write_text(
        "def all_configs():\n" "    return {'r1': {'vocab_size': 8}}\n"
    )
    (source_root / "train" / "init_patch.py").write_text(
        "def prepare_for_training(model):\n"
        "    model.bias = 5.0\n"
        "    return {'patched': True}\n"
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "dump-first-party-logits",
            "--source-root",
            str(source_root),
            "--output",
            str(output),
            "--input-ids",
            "1,2,3,4",
            "--device",
            "cpu",
            "--prepare-for-training",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert payload["prepared_for_training"] is True
    assert payload["prepare_report"] == {"patched": True}
    assert float(payload["logits"][0, 0, 0].item()) == 6.0


def test_mini_kimi_k3_runner_manifest_template_refuses_overwrite(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "init-manifest",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tmp_path / "tokens"),
            "--source",
            "fineweb-edu",
            "--dataset-id",
            "HuggingFaceFW/fineweb-edu",
            "--snapshot",
            "2026-08-fixture",
            "--license",
            "odc-by",
            "--decontamination-report",
            "reports/decontamination.json",
            "--decontamination-sha256",
            "decontam-sha",
            "--tokenizer-sha256",
            "tokenizer-sha",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "already exists" in proc.stdout
    assert json.loads(manifest.read_text()) == {}


def test_mini_kimi_k3_runner_registers_token_shards_with_metadata(tmp_path: Path):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    source_dir = tokens_dir / "fineweb-edu"
    source_dir.mkdir(parents=True)
    shard = source_dir / "part-00000.bin"
    payload = b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00"
    shard.write_bytes(payload)
    env = {**os.environ, "TORCHTITAN_IN_ROOTFS": "1"}

    init = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "init-manifest",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--source",
            "fineweb-edu",
            "--dataset-id",
            "HuggingFaceFW/fineweb-edu",
            "--snapshot",
            "2026-08-fixture",
            "--license",
            "odc-by",
            "--decontamination-report",
            "reports/decontamination.json",
            "--decontamination-sha256",
            "decontam-sha",
            "--tokenizer-sha256",
            "tokenizer-sha",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert init.returncode == 0, init.stdout

    register = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "register-shards",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--source",
            "fineweb-edu",
            "--shard",
            str(shard),
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert register.returncode == 0, register.stdout
    data = json.loads(manifest.read_text())
    assert data["sources"]["fineweb-edu"]["shards"] == ["part-00000.bin"]
    assert data["sources"]["fineweb-edu"]["shard_metadata"] == {
        "part-00000.bin": {
            "bytes": len(payload),
            "num_tokens": 4,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    }


def test_mini_kimi_k3_runner_register_shards_rejects_outside_source_dir(
    tmp_path: Path,
):
    manifest = tmp_path / "data" / "manifest.json"
    tokens_dir = tmp_path / "data" / "tokens"
    outside = tmp_path / "other.bin"
    outside.write_bytes(b"\x01\x00\x00\x00")
    env = {**os.environ, "TORCHTITAN_IN_ROOTFS": "1"}

    init = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "init-manifest",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--source",
            "fineweb-edu",
            "--dataset-id",
            "HuggingFaceFW/fineweb-edu",
            "--snapshot",
            "2026-08-fixture",
            "--license",
            "odc-by",
            "--decontamination-report",
            "reports/decontamination.json",
            "--decontamination-sha256",
            "decontam-sha",
            "--tokenizer-sha256",
            "tokenizer-sha",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert init.returncode == 0, init.stdout

    register = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "register-shards",
            "--manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--source",
            "fineweb-edu",
            "--shard",
            str(outside),
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert register.returncode == 21
    assert "must live under" in register.stdout
    assert json.loads(manifest.read_text())["sources"]["fineweb-edu"]["shards"] == []


def test_mini_kimi_k3_runner_launch_records_blocked_preflight_stage(tmp_path: Path):
    results_root = tmp_path / "results"
    tokens_dir = tmp_path / "tokens"
    manifest = tmp_path / "manifest.json"
    (tokens_dir / "alpha").mkdir(parents=True)
    (tokens_dir / "alpha" / "a.bin").write_bytes(
        b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00"
    )
    manifest.write_text(
        json.dumps({"sources": {"alpha": {"shards": ["a.bin"]}}}, sort_keys=True)
    )

    init = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "init-evidence",
            "--results-root",
            str(results_root),
            "--run-id",
            "mini-k3-r1-test",
            "--attempt-id",
            "attempt-001",
            "--mode",
            "full",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert init.returncode == 0, init.stdout

    launch = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "launch",
            "--results-root",
            str(results_root),
            "--run-id",
            "mini-k3-r1-test",
            "--attempt-id",
            "attempt-001",
            "--token-manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--seq-len",
            "4",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert launch.returncode == 21
    bundle = results_root / "runs" / "mini-k3-r1-test" / "attempt-001"
    preflight_report = json.loads((bundle / "derived" / "preflight.json").read_text())
    assert preflight_report["status"] == "blocked"
    events = [
        json.loads(line)
        for line in (bundle / "processes" / "coordinator" / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [event["kind"] for event in events] == ["stage_started", "stage_blocked"]
    assert events[-1]["return_code"] == 21
    assert events[-1]["preflight_report"]["status"] == "blocked"
    outcome = json.loads((bundle / "outcome.json").read_text())
    assert outcome["execution_outcome"] == "blocked"
    assert outcome["evaluations"]["preflight"]["measurement"] == "not_run"
    assert outcome["run_gate"]["has_real_measurement"] is False


def test_mini_kimi_k3_runner_launch_passes_forward_oracle_report(tmp_path: Path):
    results_root = tmp_path / "results"
    tokens_dir = tmp_path / "tokens"
    manifest = tmp_path / "manifest.json"
    forward_oracle = tmp_path / "forward_oracle.json"
    r1_training_smoke = tmp_path / "r1_training_smoke.json"
    corpus_audit = tmp_path / "r1-corpus-plan-audit.json"
    disk_readiness = tmp_path / "corpus-disk-readiness.json"
    source_resolution = tmp_path / "stage1-source-resolution.json"
    remote_readiness = tmp_path / "stage1-remote-readiness.json"
    (tokens_dir / "alpha").mkdir(parents=True)
    (tokens_dir / "alpha" / "a.bin").write_bytes(
        b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00"
    )
    manifest.write_text(
        json.dumps({"sources": {"alpha": {"shards": ["a.bin"]}}}, sort_keys=True)
    )
    forward_oracle.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_forward_oracle",
                "schema_version": 1,
                "status": "pass",
                "model_flavor": "r1",
                "tolerances": {"max_abs_diff": 1.0e-5, "max_rel_diff": 1.0e-5},
                "results": {"max_abs_diff": 0.0, "max_rel_diff": 0.0},
            },
            sort_keys=True,
        )
    )
    env = {**os.environ, "TORCHTITAN_IN_ROOTFS": "1"}

    init = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "init-evidence",
            "--results-root",
            str(results_root),
            "--run-id",
            "mini-k3-r1-test",
            "--attempt-id",
            "attempt-001",
            "--mode",
            "full",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert init.returncode == 0, init.stdout

    launch = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "launch",
            "--results-root",
            str(results_root),
            "--run-id",
            "mini-k3-r1-test",
            "--attempt-id",
            "attempt-001",
            "--token-manifest",
            str(manifest),
            "--tokens-dir",
            str(tokens_dir),
            "--seq-len",
            "4",
            "--forward-oracle-report",
            str(forward_oracle),
            "--r1-training-smoke-report",
            str(r1_training_smoke),
            "--r1-corpus-plan-audit-report",
            str(corpus_audit),
            "--corpus-disk-readiness-report",
            str(disk_readiness),
            "--stage1-source-resolution-report",
            str(source_resolution),
            "--stage1-remote-readiness-report",
            str(remote_readiness),
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert launch.returncode == 21
    bundle = results_root / "runs" / "mini-k3-r1-test" / "attempt-001"
    preflight_report = json.loads((bundle / "derived" / "preflight.json").read_text())
    gates = {gate["name"]: gate for gate in preflight_report["gates"]}
    assert gates["model_fidelity"]["evidence"]["forward_oracle"]["report"] == str(
        forward_oracle
    )
    assert gates["model_fidelity"]["evidence"]["forward_oracle"]["status"] == "pass"
    assert gates["trainable_config"]["evidence"]["r1_training_smoke"]["report"] == str(
        r1_training_smoke
    )
    assert "r1_corpus_plan_audit" in gates["real_corpus_manifest"]["evidence"]
    assert gates["real_corpus_manifest"]["evidence"]["r1_corpus_plan_audit"][
        "report"
    ] == str(corpus_audit)
    assert "corpus_disk_readiness" in gates["real_corpus_manifest"]["evidence"]
    assert gates["real_corpus_manifest"]["evidence"]["corpus_disk_readiness"][
        "report"
    ] == str(disk_readiness)
    assert "stage1_source_resolution" in gates["real_corpus_manifest"]["evidence"]
    assert gates["real_corpus_manifest"]["evidence"]["stage1_source_resolution"][
        "report"
    ] == str(source_resolution)
    assert "stage1_remote_readiness" in gates["real_corpus_manifest"]["evidence"]
    assert gates["real_corpus_manifest"]["evidence"]["stage1_remote_readiness"][
        "report"
    ] == str(remote_readiness)


def test_mini_kimi_k3_launch_runs_train_stage_after_ready_preflight(tmp_path: Path):
    results_root = tmp_path / "results"
    token_manifest = tmp_path / "manifest.json"
    tokens_dir = tmp_path / "tokens"
    stage1_input_inspection = tmp_path / "reports" / "stage1-input-inspection.json"
    initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-ready-test",
        attempt_id="attempt-001",
        mode="full",
    )

    class ReadyPreflightExecutor:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(self, argv: list[str], *, cwd: Path | None = None) -> CommandResult:
            del cwd
            self.calls.append(list(argv))
            if argv[:3] == [
                sys.executable,
                "-m",
                "experiments.mini_kimi_k3.preflight",
            ]:
                report_path = Path(argv[argv.index("--report") + 1])
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "kind": "mini_kimi_k3_preflight",
                            "mode": "full",
                            "status": "ready",
                            "gates": [],
                        },
                        sort_keys=True,
                    )
                )
            return CommandResult(return_code=0)

    executor = ReadyPreflightExecutor()

    rc = launch.run_launch(
        argparse.Namespace(
            results_root=results_root,
            run_id="mini-k3-r1-ready-test",
            attempt_id="attempt-001",
            mode="full",
            token_manifest=token_manifest,
            tokens_dir=tokens_dir,
            seq_len=4096,
            tiny_smoke_report=None,
            oracle_root=None,
            oracle_r1_config=None,
            forward_oracle_report=None,
            launch_backend_report=None,
            r1_training_smoke_report=None,
            r1_corpus_plan_audit_report=None,
            corpus_disk_readiness_report=None,
            stage1_source_resolution_report=None,
            stage1_input_inspection_report=stage1_input_inspection,
            stage1_remote_readiness_report=None,
        ),
        executor=executor,
    )

    assert rc == 0
    assert len(executor.calls) == 2
    preflight_argv = executor.calls[0]
    assert "--stage1-input-inspection-report" in preflight_argv
    assert preflight_argv[
        preflight_argv.index("--stage1-input-inspection-report") + 1
    ] == str(stage1_input_inspection)
    train_argv = executor.calls[1]
    assert train_argv[:6] == [
        "/usr/bin/env",
        "MODULE=mini_kimi_k3",
        "CONFIG=mini_kimi_k3_r1_contract",
        "NGPU=1",
        "TORCHTITAN_RUN_ID=mini-k3-r1-ready-test",
        "TORCHTITAN_ATTEMPT_ID=attempt-001",
    ]
    assert train_argv[6] == "./run_train.sh"
    assert f"--dataloader.token_manifest={token_manifest}" in train_argv
    assert f"--dataloader.tokens_dir={tokens_dir}" in train_argv
    assert "--checkpoint.keep_latest_k=2" in train_argv

    bundle = results_root / "runs" / "mini-k3-r1-ready-test" / "attempt-001"
    events = [
        json.loads(line)
        for line in (bundle / "processes" / "coordinator" / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [event["stage_id"] for event in events] == [
        "preflight",
        "preflight",
        "train",
        "train",
    ]
    assert [event["kind"] for event in events] == [
        "stage_started",
        "stage_succeeded",
        "stage_started",
        "stage_succeeded",
    ]
    outcome = json.loads((bundle / "outcome.json").read_text())
    assert outcome["execution_outcome"] == "completed"
    assert (
        outcome["evaluations"]["training"]["stage_invocation_id"]
        == events[-1]["stage_invocation_id"]
    )
    assert outcome["evaluations"]["training"] == {
        "execution_outcome": "completed",
        "measurement": "real",
        "promotion": "not_evaluated",
        "stage_invocation_id": events[-1]["stage_invocation_id"],
    }


def test_mini_kimi_k3_launch_blocks_on_stale_preflight_report(tmp_path: Path):
    results_root = tmp_path / "results"
    initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-stale-preflight-test",
        attempt_id="attempt-001",
        mode="full",
    )

    class StalePreflightExecutor:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(self, argv: list[str], *, cwd: Path | None = None) -> CommandResult:
            del cwd
            self.calls.append(list(argv))
            if argv[:3] == [
                sys.executable,
                "-m",
                "experiments.mini_kimi_k3.preflight",
            ]:
                report_path = Path(argv[argv.index("--report") + 1])
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "kind": "mini_kimi_k3_preflight",
                            "mode": "full",
                            "status": "blocked",
                            "gates": [
                                {"name": "trainable_config", "status": "partial"}
                            ],
                        },
                        sort_keys=True,
                    )
                )
            return CommandResult(return_code=0)

    executor = StalePreflightExecutor()

    rc = launch.run_launch(
        argparse.Namespace(
            results_root=results_root,
            run_id="mini-k3-r1-stale-preflight-test",
            attempt_id="attempt-001",
            mode="full",
            token_manifest=None,
            tokens_dir=None,
            seq_len=4096,
            tiny_smoke_report=None,
            oracle_root=None,
            oracle_r1_config=None,
            forward_oracle_report=None,
            launch_backend_report=None,
            r1_training_smoke_report=None,
            r1_corpus_plan_audit_report=None,
            corpus_disk_readiness_report=None,
            stage1_source_resolution_report=None,
            stage1_input_inspection_report=None,
            stage1_remote_readiness_report=None,
        ),
        executor=executor,
    )

    assert rc == 21
    assert len(executor.calls) == 1
    bundle = results_root / "runs" / "mini-k3-r1-stale-preflight-test" / "attempt-001"
    events = [
        json.loads(line)
        for line in (bundle / "processes" / "coordinator" / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [event["stage_id"] for event in events] == ["preflight", "preflight"]
    outcome = json.loads((bundle / "outcome.json").read_text())
    assert outcome["execution_outcome"] == "blocked"
    assert outcome["evaluations"]["preflight"] == {
        "execution_outcome": "blocked",
        "measurement": "not_run",
        "promotion": "not_evaluated",
    }


def test_mini_kimi_k3_r1_training_smoke_writes_pass_report(tmp_path: Path):
    token_manifest = tmp_path / "manifest.json"
    tokens_dir = tmp_path / "tokens"
    report = tmp_path / "r1_training_smoke.json"

    class SuccessfulExecutor:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(self, argv: list[str], *, cwd: Path | None = None) -> CommandResult:
            del cwd
            self.calls.append(list(argv))
            return CommandResult(
                return_code=0,
                stdout="Train | Step: 1 grad_norm: 2.5\n",
                stderr="",
            )

    executor = SuccessfulExecutor()

    result = r1_training_smoke.run_r1_training_smoke(
        token_manifest=token_manifest,
        tokens_dir=tokens_dir,
        report_path=report,
        executor=executor,
        gpu_memory_probe=lambda: [
            {
                "index": 0,
                "name": "NVIDIA B200",
                "memory_total_mib": 183_359,
                "memory_used_mib": 1_024,
                "memory_free_mib": 182_335,
            }
        ],
    )

    assert result["kind"] == "mini_kimi_k3_r1_training_smoke"
    assert result["status"] == "pass"
    assert result["config"] == "mini_kimi_k3_r1_contract"
    assert result["model_flavor"] == "r1"
    assert result["data"] == {
        "manifest": str(token_manifest),
        "tokens_dir": str(tokens_dir),
        "seq_len": 4096,
        "hf_assets_path": "tests/assets/tokenizer",
    }
    assert result["optimization"]["steps"] == 1
    assert result["optimization"]["max_grad_norm"] == 2.5
    assert result["command"]["return_code"] == 0
    assert json.loads(report.read_text()) == result
    argv = executor.calls[0]
    assert argv[0] == "/usr/bin/env"
    assert "CUDA_VISIBLE_DEVICES=0" in argv
    assert "MODULE=mini_kimi_k3" in argv
    assert "CONFIG=mini_kimi_k3_r1_contract" in argv
    assert "COMM_MODE=fake_backend" in argv
    assert "NGPU=1" in argv
    assert "./run_train.sh" in argv
    assert "--hf_assets_path=tests/assets/tokenizer" in argv
    assert "--training.steps" in argv
    assert "1" in argv
    assert "--training.local_batch_size=1" in argv
    assert "--training.global_batch_size=1" in argv
    assert "--training.seq_len=4096" in argv
    assert "--checkpoint.no-enable" in argv
    assert f"--dataloader.token_manifest={token_manifest}" in argv
    assert f"--dataloader.tokens_dir={tokens_dir}" in argv


def test_mini_kimi_k3_r1_training_smoke_writes_fail_report(tmp_path: Path):
    report = tmp_path / "r1_training_smoke.json"

    class FailingExecutor:
        def run(self, argv: list[str], *, cwd: Path | None = None) -> CommandResult:
            del argv, cwd
            return CommandResult(
                return_code=17,
                stdout="",
                stderr="RuntimeError: training failed\n",
            )

    result = r1_training_smoke.run_r1_training_smoke(
        token_manifest=tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        report_path=report,
        executor=FailingExecutor(),
        gpu_memory_probe=lambda: [
            {
                "index": 0,
                "name": "NVIDIA B200",
                "memory_total_mib": 183_359,
                "memory_used_mib": 1_024,
                "memory_free_mib": 182_335,
            }
        ],
    )

    assert result["status"] == "fail"
    assert result["command"]["return_code"] == 17
    assert "RuntimeError: training failed" in result["command"]["stderr_tail"]
    assert json.loads(report.read_text()) == result


def test_mini_kimi_k3_r1_training_smoke_reports_cuda_oom_as_blocked(
    tmp_path: Path,
):
    report = tmp_path / "r1_training_smoke.json"

    class OomExecutor:
        def run(self, argv: list[str], *, cwd: Path | None = None) -> CommandResult:
            del argv, cwd
            return CommandResult(
                return_code=1,
                stdout="",
                stderr="torch.OutOfMemoryError: CUDA out of memory\n",
            )

    result = r1_training_smoke.run_r1_training_smoke(
        token_manifest=tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        report_path=report,
        executor=OomExecutor(),
        gpu_memory_probe=lambda: [
            {
                "index": 0,
                "name": "NVIDIA B200",
                "memory_total_mib": 183_359,
                "memory_used_mib": 173_144,
                "memory_free_mib": 9_489,
            }
        ],
        min_free_gpu_memory_mib=9_000,
    )

    assert result["status"] == "blocked"
    assert result["error"] == "CUDA out of memory during r1 Trainer smoke"
    assert result["command"]["return_code"] == 1
    assert "CUDA out of memory" in result["command"]["stderr_tail"]
    assert json.loads(report.read_text()) == result


def test_mini_kimi_k3_r1_training_smoke_pins_selected_gpu(tmp_path: Path):
    report = tmp_path / "r1_training_smoke.json"

    class SuccessfulExecutor:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(self, argv: list[str], *, cwd: Path | None = None) -> CommandResult:
            del cwd
            self.calls.append(list(argv))
            return CommandResult(
                return_code=0,
                stdout="Train | Step: 1 grad_norm: 2.5\n",
                stderr="",
            )

    executor = SuccessfulExecutor()

    result = r1_training_smoke.run_r1_training_smoke(
        token_manifest=tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        report_path=report,
        executor=executor,
        min_free_gpu_memory_mib=12_000,
        gpu_memory_probe=lambda: [
            {
                "index": 0,
                "name": "NVIDIA B200",
                "memory_total_mib": 183_359,
                "memory_used_mib": 173_144,
                "memory_free_mib": 9_489,
            },
            {
                "index": 1,
                "name": "NVIDIA B200",
                "memory_total_mib": 183_359,
                "memory_used_mib": 120_000,
                "memory_free_mib": 63_359,
            },
        ],
    )

    assert result["status"] == "pass"
    assert result["gpu_memory"]["selected"]["index"] == 1
    assert "CUDA_VISIBLE_DEVICES=1" in executor.calls[0]


def test_mini_kimi_k3_r1_training_smoke_blocks_when_gpu_memory_is_low(
    tmp_path: Path,
):
    report = tmp_path / "r1_training_smoke.json"

    class UnusedExecutor:
        def run(self, argv: list[str], *, cwd: Path | None = None) -> CommandResult:
            raise AssertionError("run_train.sh should not launch when precheck blocks")

    result = r1_training_smoke.run_r1_training_smoke(
        token_manifest=tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        report_path=report,
        executor=UnusedExecutor(),
        min_free_gpu_memory_mib=12_000,
        gpu_memory_probe=lambda: [
            {
                "index": 0,
                "name": "NVIDIA B200",
                "memory_total_mib": 183_359,
                "memory_used_mib": 173_144,
                "memory_free_mib": 9_489,
            }
        ],
    )

    assert result["status"] == "blocked"
    assert result["error"] == "insufficient free GPU memory for r1 Trainer smoke"
    assert result["gpu_memory"]["required_free_mib"] == 12_000
    assert result["gpu_memory"]["selected"] is None
    assert result["command"]["return_code"] is None
    assert json.loads(report.read_text()) == result


def test_mini_kimi_k3_r1_training_smoke_records_gpu_holder_evidence(
    tmp_path: Path,
):
    report = tmp_path / "r1_training_smoke.json"

    class UnusedExecutor:
        def run(self, argv: list[str], *, cwd: Path | None = None) -> CommandResult:
            raise AssertionError("run_train.sh should not launch when precheck blocks")

    result = r1_training_smoke.run_r1_training_smoke(
        token_manifest=tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        report_path=report,
        executor=UnusedExecutor(),
        min_free_gpu_memory_mib=12_000,
        gpu_memory_probe=lambda: [
            {
                "index": 0,
                "name": "NVIDIA B200",
                "memory_total_mib": 183_359,
                "memory_used_mib": 173_144,
                "memory_free_mib": 9_489,
            }
        ],
        gpu_holder_probe=lambda: [
            {
                "pid": 1958391,
                "ppid": 1958387,
                "user": "philip.yang",
                "stat": "Sl+",
                "etime": "17:28:38",
                "command": "python",
                "cwd": "/workspace/monarch",
            },
            {
                "pid": 1959989,
                "ppid": 1958391,
                "user": "philip.yang",
                "stat": "Rl+",
                "etime": "17:28:24",
                "command": "sglang::scheduler_TP0",
                "cwd": "/workspace/monarch",
            },
        ],
    )

    assert result["status"] == "blocked"
    assert result["gpu_memory"]["holders"] == [
        {
            "pid": 1958391,
            "ppid": 1958387,
            "user": "philip.yang",
            "stat": "Sl+",
            "etime": "17:28:38",
            "command": "python",
            "cwd": "/workspace/monarch",
        },
        {
            "pid": 1959989,
            "ppid": 1958391,
            "user": "philip.yang",
            "stat": "Rl+",
            "etime": "17:28:24",
            "command": "sglang::scheduler_TP0",
            "cwd": "/workspace/monarch",
        },
    ]
    assert json.loads(report.read_text()) == result


def test_mini_kimi_k3_r1_training_smoke_finds_holders_from_proc(
    tmp_path: Path,
):
    proc_root = tmp_path / "proc"
    dev_root = tmp_path / "dev"
    nvidia0 = dev_root / "nvidia0"
    fd_dir = proc_root / "123" / "fd"
    fd_dir.mkdir(parents=True)
    dev_root.mkdir()
    nvidia0.write_text("")
    (fd_dir / "7").symlink_to(nvidia0)
    (proc_root / "123" / "status").write_text(
        "Name:\tsglang::scheduler_TP0\n"
        "State:\tS (sleeping)\n"
        "PPid:\t99\n"
        "Uid:\t1000\t1000\t1000\t1000\n"
    )
    (proc_root / "123" / "cmdline").write_bytes(b"sglang::scheduler_TP0\0")
    (proc_root / "123" / "cwd").symlink_to("/workspace/monarch")

    holders = r1_training_smoke._query_gpu_holders_from_proc(
        proc_root=proc_root,
        device_root=dev_root,
    )

    assert holders == [
        {
            "pid": 123,
            "ppid": 99,
            "user": "1000",
            "stat": "S",
            "etime": None,
            "command": "sglang::scheduler_TP0",
            "cwd": "/workspace/monarch",
        }
    ]


def test_mini_kimi_k3_r1_training_smoke_falls_back_when_fuser_is_missing(
    monkeypatch,
):
    def missing_fuser(*args, **kwargs):
        del args, kwargs
        raise FileNotFoundError("fuser")

    monkeypatch.setattr(r1_training_smoke.subprocess, "run", missing_fuser)
    monkeypatch.setattr(
        r1_training_smoke,
        "_query_gpu_holders_from_proc",
        lambda **kwargs: [{"pid": 123, "command": "python"}],
    )

    assert r1_training_smoke._query_gpu_holders() == [{"pid": 123, "command": "python"}]


def test_mini_kimi_k3_r1_training_smoke_expands_fuser_devices(
    tmp_path: Path,
    monkeypatch,
):
    dev_root = tmp_path / "dev"
    (dev_root / "nvidia-caps").mkdir(parents=True)
    for name in ["nvidiactl", "nvidia-uvm", "nvidia0", "nvidia1"]:
        (dev_root / name).write_text("")

    calls = []

    def fake_run(argv, *, cwd=None, capture_output=None, text=None, check=None):
        del cwd, capture_output, text, check
        calls.append(argv)
        return CommandResult(return_code=0, stderr=" 123 456\n")

    monkeypatch.setattr(r1_training_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(
        r1_training_smoke,
        "_process_holder_record",
        lambda pid: {"pid": pid, "command": f"pid-{pid}"},
    )

    assert r1_training_smoke._query_gpu_holders(device_root=dev_root) == [
        {"pid": 123, "command": "pid-123"},
        {"pid": 456, "command": "pid-456"},
    ]
    assert calls == [
        [
            "fuser",
            "-v",
            str(dev_root / "nvidia-uvm"),
            str(dev_root / "nvidia0"),
            str(dev_root / "nvidia1"),
            str(dev_root / "nvidiactl"),
        ]
    ]


def test_mini_kimi_k3_r1_training_smoke_records_hidden_proc_metadata():
    monkeypatch_pid = 123456

    record = r1_training_smoke._process_holder_record(monkeypatch_pid)

    assert record == {
        "pid": monkeypatch_pid,
        "detail": "process metadata is not visible from this namespace",
    }


def test_mini_kimi_k3_runner_rejects_rerun_of_finished_attempt(tmp_path: Path):
    results_root = tmp_path / "results"
    tokens_dir = tmp_path / "tokens"
    manifest = tmp_path / "manifest.json"
    (tokens_dir / "alpha").mkdir(parents=True)
    (tokens_dir / "alpha" / "a.bin").write_bytes(
        b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00"
    )
    manifest.write_text(
        json.dumps({"sources": {"alpha": {"shards": ["a.bin"]}}}, sort_keys=True)
    )
    env = {**os.environ, "TORCHTITAN_IN_ROOTFS": "1"}

    init = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "init-evidence",
            "--results-root",
            str(results_root),
            "--run-id",
            "mini-k3-r1-test",
            "--attempt-id",
            "attempt-001",
            "--mode",
            "full",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert init.returncode == 0, init.stdout

    launch_cmd = [
        "bash",
        str(RUNNER),
        "launch",
        "--results-root",
        str(results_root),
        "--run-id",
        "mini-k3-r1-test",
        "--attempt-id",
        "attempt-001",
        "--token-manifest",
        str(manifest),
        "--tokens-dir",
        str(tokens_dir),
        "--seq-len",
        "4",
    ]
    first_launch = subprocess.run(
        launch_cmd,
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert first_launch.returncode == 21, first_launch.stdout
    bundle = results_root / "runs" / "mini-k3-r1-test" / "attempt-001"
    outcome_before_rerun = json.loads((bundle / "outcome.json").read_text())

    second_launch = subprocess.run(
        launch_cmd,
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert second_launch.returncode == 21
    assert "already has terminal outcome" in second_launch.stdout
    assert "attempt-001" in second_launch.stdout
    assert "Traceback" not in second_launch.stdout
    assert json.loads((bundle / "outcome.json").read_text()) == outcome_before_rerun
    events = [
        json.loads(line)
        for line in (bundle / "processes" / "coordinator" / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [event["kind"] for event in events] == ["stage_started", "stage_blocked"]


def test_mini_kimi_k3_runner_reexecs_through_rootfs_and_preserves_args(
    tmp_path: Path,
):
    plan_path = tmp_path / "rootfs_plan.json"
    host_state = (
        REPO_ROOT
        / ".cache"
        / "trae-pytest-tmp"
        / f"mini-kimi-k3-rootfs-state-{tmp_path.name}"
    )
    shutil.rmtree(host_state, ignore_errors=True)
    env = {
        **os.environ,
        "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
        "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(plan_path),
        "TORCHTITAN_ROOTFS_HOST_STATE": str(host_state),
    }
    env.pop("TORCHTITAN_IN_ROOTFS", None)

    try:
        proc = subprocess.run(
            [
                "bash",
                str(RUNNER),
                "--mode",
                "tiny-plumbing",
                "--note",
                "space value",
            ],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        shutil.rmtree(host_state, ignore_errors=True)

    assert proc.returncode == 0, proc.stdout
    plan = json.loads(plan_path.read_text())
    command = plan["inner_argv"]
    assert command[:3] == [
        "/bin/bash",
        "-lc",
        'cd /workspace/torchtitan && exec experiments/mini_kimi_k3/run.sh "$@"',
    ]
    assert command[3:] == [
        "experiments/mini_kimi_k3/run.sh",
        "--mode",
        "tiny-plumbing",
        "--note",
        "space value",
    ]


def test_mini_kimi_k3_unblock_wizard_refreshes_corpus_disk_readiness():
    wizard = (
        REPO_ROOT
        / ".scratch"
        / "mini-kimi-k3-replication"
        / "unblock-full-launch-wizard.sh"
    )

    body = wizard.read_text()

    assert (
        'CORPUS_DISK_READINESS_REPORT="experiments/mini_kimi_k3/reports/'
        'corpus-disk-readiness.json"'
    ) in body
    assert (
        'run_rootfs "./experiments/mini_kimi_k3/run.sh corpus-disk-readiness '
        "--corpus-plan experiments/mini_kimi_k3/reports/r1-corpus-plan.json "
        "--target-path experiments/mini_kimi_k3/data/tokens "
        '--report $CORPUS_DISK_READINESS_REPORT --overhead-fraction 0.20" || true'
    ) in body
    assert (
        "experiments/mini_kimi_k3/run.sh corpus-disk-readiness "
        "--corpus-plan experiments/mini_kimi_k3/reports/r1-corpus-plan.json "
        "--target-path experiments/mini_kimi_k3/data/tokens "
        "--report $CORPUS_DISK_READINESS_REPORT --overhead-fraction 0.20"
    ) in body


def test_mini_kimi_k3_unblock_wizard_forwards_stage1_input_inspection():
    wizard = (
        REPO_ROOT
        / ".scratch"
        / "mini-kimi-k3-replication"
        / "unblock-full-launch-wizard.sh"
    )

    body = wizard.read_text()

    assert (
        'STAGE1_INPUT_INSPECTION_REPORT="experiments/mini_kimi_k3/reports/'
        'source-inputs-inspection.json"'
    ) in body
    assert (
        "$(stage1_input_inspection_arg) "
        "--stage1-remote-readiness-report $REMOTE_READINESS_REPORT"
    ) in body


def test_mini_kimi_k3_unblock_wizard_makes_stage1_input_inspection_optional():
    wizard = (
        REPO_ROOT
        / ".scratch"
        / "mini-kimi-k3-replication"
        / "unblock-full-launch-wizard.sh"
    )

    body = wizard.read_text()

    assert "stage1_input_inspection_arg()" in body
    assert '[[ -f "$STAGE1_INPUT_INSPECTION_REPORT" ]] || return 0' in body
    assert (
        "--stage1-source-resolution-report experiments/mini_kimi_k3/reports/"
        "stage1-source-resolution.json$(stage1_input_inspection_arg) "
        "--stage1-remote-readiness-report $REMOTE_READINESS_REPORT"
    ) in body


def test_mini_kimi_k3_unblock_wizard_final_preflight_uses_oracle_root():
    wizard = (
        REPO_ROOT
        / ".scratch"
        / "mini-kimi-k3-replication"
        / "unblock-full-launch-wizard.sh"
    )

    body = wizard.read_text()
    final_stage = body.split('stage "Final Preflight And Launch"', maxsplit=1)[1]

    assert (
        "--oracle-root experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara"
    ) in final_stage


def test_mini_kimi_k3_unblock_wizard_inspection_preflight_uses_oracle_root():
    wizard = (
        REPO_ROOT
        / ".scratch"
        / "mini-kimi-k3-replication"
        / "unblock-full-launch-wizard.sh"
    )

    body = wizard.read_text()
    inspect_stage = body.split('stage "Inspect Current Gates"', maxsplit=1)[1].split(
        'stage "GPU Capacity"', maxsplit=1
    )[0]

    assert (
        "--oracle-root experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara"
    ) in inspect_stage


def test_mini_kimi_k3_unblock_wizard_shows_networked_commands_through_rootfs():
    wizard = (
        REPO_ROOT
        / ".scratch"
        / "mini-kimi-k3-replication"
        / "unblock-full-launch-wizard.sh"
    )

    body = wizard.read_text()
    gpu_stage = body.split('stage "GPU Capacity"', maxsplit=1)[1].split(
        'stage "Corpus Route"', maxsplit=1
    )[0]
    corpus_stage = body.split('stage "Corpus Route"', maxsplit=1)[1].split(
        'stage "Final Preflight And Launch"', maxsplit=1
    )[0]

    assert (
        "TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- "
        'bash -lc \\"cd /workspace/torchtitan && experiments/mini_kimi_k3/run.sh '
        "stage1-remote-readiness --env-file $ENV_FILE "
        '--report $REMOTE_READINESS_REPORT\\"'
    ) in gpu_stage
    assert (
        "TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- "
        'bash -lc \\"cd /workspace/torchtitan && experiments/mini_kimi_k3/run.sh '
        "run-stage1-remote --source-root "
        "experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara "
        "--readiness-report $REMOTE_READINESS_REPORT --env-file $ENV_FILE "
        "--step ingest_all --allow-spend "
        '--report experiments/mini_kimi_k3/reports/stage1-remote-ingest-all.json\\"'
    ) in corpus_stage


def test_mini_kimi_k3_unblock_wizard_imports_local_corpus_inside_rootfs():
    wizard = (
        REPO_ROOT
        / ".scratch"
        / "mini-kimi-k3-replication"
        / "unblock-full-launch-wizard.sh"
    )

    body = wizard.read_text()
    corpus_stage = body.split('stage "Corpus Route"', maxsplit=1)[1].split(
        'stage "Final Preflight And Launch"', maxsplit=1
    )[0]

    assert (
        'ask SOURCE_EXPORT "Rootfs-visible path to reviewed Stage 1 export directory:"'
    ) in corpus_stage
    assert (
        'run_rootfs "./experiments/mini_kimi_k3/run.sh import-shards '
        "--manifest experiments/mini_kimi_k3/data/manifest.json "
        "--tokens-dir experiments/mini_kimi_k3/data/tokens --source $SOURCE_NAME "
        '--source-dir $SOURCE_EXPORT/tokens/$SOURCE_NAME --link-mode symlink"'
    ) in corpus_stage


def test_mini_kimi_k3_unblock_wizard_imports_every_planned_local_source():
    wizard = (
        REPO_ROOT
        / ".scratch"
        / "mini-kimi-k3-replication"
        / "unblock-full-launch-wizard.sh"
    )

    body = wizard.read_text()
    corpus_stage = body.split('stage "Corpus Route"', maxsplit=1)[1].split(
        'stage "Final Preflight And Launch"', maxsplit=1
    )[0]

    assert (
        'PLANNED_SOURCES=("fineweb-edu" "web-diverse" "code-python" '
        '"finemath" "open-web-math" "cosmopedia")'
    ) in corpus_stage
    assert 'for SOURCE_NAME in "${PLANNED_SOURCES[@]}"; do' in corpus_stage
    assert 'run_rootfs "./experiments/mini_kimi_k3/run.sh import-shards' in corpus_stage


def test_mini_kimi_k3_unblock_wizard_points_to_completion_audit_cli_guidance():
    wizard = (
        REPO_ROOT
        / ".scratch"
        / "mini-kimi-k3-replication"
        / "unblock-full-launch-wizard.sh"
    )

    body = wizard.read_text()
    inspect_stage = body.split('stage "Inspect Current Gates"', maxsplit=1)[1].split(
        'stage "GPU Capacity"', maxsplit=1
    )[0]

    assert "next actions:" in inspect_stage
    assert "guidance:" in inspect_stage
    assert "source_deficits" in inspect_stage


def test_mini_kimi_k3_unblock_wizard_documents_modal_env_template():
    wizard = (
        REPO_ROOT
        / ".scratch"
        / "mini-kimi-k3-replication"
        / "unblock-full-launch-wizard.sh"
    )

    body = wizard.read_text()
    modal_stage = body.split('stage "Modal Credentials"', maxsplit=1)[1].split(
        'stage "Corpus Route"', maxsplit=1
    )[0]

    assert ('ENV_EXAMPLE=".scratch/mini-kimi-k3-replication/.env.example"') in body
    assert 'say "Credential template: $ENV_EXAMPLE"' in modal_stage
    assert 'say "Required keys: MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, MODAL_PROFILE"' in (
        modal_stage
    )


def test_mini_kimi_k3_unblock_wizard_writes_required_modal_profile():
    wizard = (
        REPO_ROOT
        / ".scratch"
        / "mini-kimi-k3-replication"
        / "unblock-full-launch-wizard.sh"
    )

    body = wizard.read_text()
    modal_stage = body.split('stage "Modal Credentials"', maxsplit=1)[1].split(
        'stage "Corpus Route"', maxsplit=1
    )[0]

    assert 'ask MODAL_PROFILE "Modal profile [mini-k3]:"' in modal_stage
    assert 'MODAL_PROFILE="${MODAL_PROFILE:-mini-k3}"' in modal_stage
    assert 'write_env MODAL_PROFILE "$MODAL_PROFILE"' in modal_stage


def test_mini_kimi_k3_unblock_wizard_keeps_first_stage_non_gpu():
    wizard = (
        REPO_ROOT
        / ".scratch"
        / "mini-kimi-k3-replication"
        / "unblock-full-launch-wizard.sh"
    )

    body = wizard.read_text()
    first_stage, gpu_stage = body.split('stage "GPU Capacity"', maxsplit=1)

    assert "r1-smoke-gpu-readiness" not in first_stage
    assert "r1-smoke-gpu-readiness" in gpu_stage


def test_mini_kimi_k3_readme_documents_completion_audit_guidance():
    readme = REPO_ROOT / "experiments" / "mini_kimi_k3" / "README.md"

    text = readme.read_text()

    assert "`next_actions[].guidance`" in text
    assert "`blocked_preflight_requirements`" in text
    assert "`target_tokens`" in text
    assert "`available_tokens`" in text
    assert "`deficit_tokens`" in text
    assert "`source_deficits`" in text
    assert "`r1_training_smoke_report`" in text
    assert "`env_file_template`" in text
    assert "`attempt_outcome_report`" in text
    assert "`execution_outcome`" in text
    assert "`measurement`" in text
    assert "`event_stream`" in text
    assert "`terminal_events`" in text
    assert "`event_stream_errors`" in text
    assert "`next actions:`" in text
    assert "`guidance:`" in text
