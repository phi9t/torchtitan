# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from experiments.mini_kimi_k3 import preflight
from experiments.mini_kimi_k3.evidence import initialize_launch_evidence_bundle

from experiments.mini_kimi_k3.training_smoke import run_training_smoke
from torchtitan.experiments.mini_kimi_k3.model_contract import (
    mini_k3_flagship_config,
    mini_k3_r1_config,
    to_kimi_linear_config_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "experiments" / "mini_kimi_k3" / "run.sh"


def _write_shard(path: Path, tokens: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(tokens, dtype="<u4").tofile(path)


def _set_full_token_budget(monkeypatch: pytest.MonkeyPatch, target_tokens: int) -> None:
    real_recipe = preflight.mini_k3_r1_launch_recipe
    monkeypatch.setattr(
        preflight,
        "mini_k3_r1_launch_recipe",
        lambda: real_recipe(target_tokens=target_tokens),
    )


def _write_manifest(path: Path, sources: dict[str, list[str]]) -> Path:
    path.write_text(
        json.dumps(
            {"sources": {name: {"shards": shards} for name, shards in sources.items()}},
            sort_keys=True,
        )
    )
    return path


def _write_full_manifest(path: Path, sources: dict[str, list[str]]) -> Path:
    source_specs = {}
    for name, shard_paths in sources.items():
        source_specs[name] = {
            "dataset_id": f"source/{name}",
            "provenance": {
                "snapshot": "2026-08-mini-k3-fixture",
                "license": "fixture",
            },
            "shards": shard_paths,
        }
    path.write_text(
        json.dumps(
            {
                "tokenizer": {
                    "name": "Kimi K3",
                    "vocab_size": 163_840,
                    "sha256": "tokenizer-fingerprint",
                },
                "decontamination": {
                    "status": "completed",
                    "report": "reports/decontamination.json",
                    "sha256": "decontamination-report-fingerprint",
                },
                "sources": source_specs,
            },
            sort_keys=True,
        )
    )
    return path


def _write_full_manifest_with_shard_metadata(
    path: Path,
    *,
    tokens_dir: Path,
    sources: dict[str, list[str]],
) -> Path:
    manifest = _write_full_manifest(path, sources)
    data = json.loads(manifest.read_text())
    for source, shard_names in sources.items():
        shard_metadata = {}
        for shard_name in shard_names:
            shard = tokens_dir / source / shard_name
            payload = shard.read_bytes()
            shard_metadata[shard_name] = {
                "bytes": len(payload),
                "num_tokens": len(payload) // 4,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        data["sources"][source]["shard_metadata"] = shard_metadata
    manifest.write_text(json.dumps(data, sort_keys=True))
    return manifest


def _attach_decontamination_report(manifest: Path, report: Path) -> None:
    payload = b'{"status":"completed"}\n'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_bytes(payload)
    data = json.loads(manifest.read_text())
    data["decontamination"]["report"] = str(report)
    data["decontamination"]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest.write_text(json.dumps(data, sort_keys=True))


def _attach_local_stage1_decontaminated_build(
    manifest: Path,
    *,
    source: str,
    shard_name: str,
    input_format: str,
    tokens: int,
) -> None:
    data = json.loads(manifest.read_text())
    data["sources"][source]["local_builds"] = [
        {
            "source": source,
            "documents": 1,
            "input_format": input_format,
            "decontamination": {
                "applied": True,
                "index": str(manifest.parent / "decontamination-index.npz"),
                "min_matches": 1,
                "documents_seen": 1,
                "documents_kept": 1,
                "documents_removed": 0,
                "hits_by_benchmark": {},
            },
            "shards": [{"name": shard_name, "tokens": tokens}],
        }
    ]
    manifest.write_text(json.dumps(data, sort_keys=True))


def _write_oracle(root: Path) -> Path:
    (root / "model").mkdir(parents=True)
    (root / "train").mkdir(parents=True)
    (root / "model" / "config_mini.json").write_text(
        json.dumps(
            to_kimi_linear_config_dict(mini_k3_flagship_config()), sort_keys=True
        )
    )
    (root / "train" / "ladder_r1.json").write_text(
        json.dumps(to_kimi_linear_config_dict(mini_k3_r1_config()), sort_keys=True)
    )
    return root


def _write_forward_oracle_report(
    path: Path,
    *,
    status: str = "pass",
    model_flavor: str = "r1",
    max_abs_diff: float = 0.0,
    max_rel_diff: float = 0.0,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_forward_oracle",
                "schema_version": 1,
                "status": status,
                "model_flavor": model_flavor,
                "source": {
                    "oracle": "first_party_kimi_k3",
                    "torch_titan": "torchtitan.experiments.mini_kimi_k3",
                },
                "tolerances": {
                    "max_abs_diff": 1.0e-5,
                    "max_rel_diff": 1.0e-5,
                },
                "results": {
                    "max_abs_diff": max_abs_diff,
                    "max_rel_diff": max_rel_diff,
                },
            },
            sort_keys=True,
        )
    )
    return path


def _write_launch_backend_report(
    path: Path,
    *,
    status: str = "pass",
    model_flavor: str = "r1",
    forward_oracle_report: str = "forward_oracle.json",
    forward_trace_report: str = "forward_trace.json",
    final_logit_max_abs_diff: float = 0.0,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_launch_backend_review",
                "schema_version": 1,
                "status": status,
                "backend": "torchtitan.experiments.mini_kimi_k3",
                "config": "mini_kimi_k3_r1_contract",
                "model_flavor": model_flavor,
                "evidence": {
                    "forward_oracle_report": forward_oracle_report,
                    "forward_trace_report": forward_trace_report,
                    "final_logit_max_abs_diff": final_logit_max_abs_diff,
                },
            },
            sort_keys=True,
        )
    )
    return path


def _write_r1_training_smoke_report(
    path: Path,
    *,
    status: str = "pass",
    config: str = "mini_kimi_k3_r1_contract",
    model_flavor: str = "r1",
    manifest: str = "manifest.json",
    tokens_dir: str = "tokens",
    hf_assets_path: str = "tests/assets/tokenizer",
    dump_folder: str = "r1_training_smoke_train",
    seq_len: int = 4096,
    steps: int = 1,
    max_grad_norm: float = 1.0,
    error: str | None = None,
    gpu_memory: dict | None = None,
    rootfs: dict | None = None,
    command: dict | None = None,
) -> Path:
    if gpu_memory is None:
        gpu_memory = {
            "status": "pass",
            "required_free_mib": 12_000,
            "selected": {"index": 0, "memory_free_mib": 48_000},
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA B200",
                    "memory_total_mib": 183_359,
                    "memory_used_mib": 135_359,
                    "memory_free_mib": 48_000,
                }
            ],
        }
    if rootfs is None:
        rootfs = {"marker": "1", "cwd": "/workspace/torchtitan"}
    if command is None:
        command_steps = steps if steps > 0 else 1
        command = {
            "argv": [
                "/usr/bin/env",
                "CUDA_VISIBLE_DEVICES=0",
                "MODULE=mini_kimi_k3",
                "CONFIG=mini_kimi_k3_r1_contract",
                "COMM_MODE=fake_backend",
                "NGPU=1",
                "./run_train.sh",
                f"--hf_assets_path={hf_assets_path}",
                f"--dump_folder={dump_folder}",
                "--training.steps",
                str(command_steps),
                "--training.local_batch_size=1",
                "--training.global_batch_size=1",
                f"--training.seq_len={seq_len}",
                "--checkpoint.no-enable",
                f"--dataloader.token_manifest={manifest}",
                f"--dataloader.tokens_dir={tokens_dir}",
            ],
            "cwd": "/workspace/torchtitan",
            "return_code": 0,
            "stdout_tail": "Train | Step: 1 grad_norm: 1.0\n",
            "stderr_tail": "",
        }
    report = {
        "kind": "mini_kimi_k3_r1_training_smoke",
        "schema_version": 1,
        "status": status,
        "config": config,
        "model_flavor": model_flavor,
        "rootfs": rootfs,
        "data": {
            "manifest": manifest,
            "tokens_dir": tokens_dir,
            "seq_len": seq_len,
            "hf_assets_path": hf_assets_path,
        },
        "optimization": {
            "steps": steps,
            "max_grad_norm": max_grad_norm,
        },
        "command": command,
    }
    if error is not None:
        report["error"] = error
    report["gpu_memory"] = gpu_memory
    path.write_text(json.dumps(report, sort_keys=True))
    return path


def _write_stage1_remote_readiness_report(
    path: Path,
    *,
    status: str = "blocked",
    modal_auth_status: str = "blocked",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_remote_readiness",
                "status": status,
                "checks": {
                    "huggingface_dns": {
                        "status": "pass",
                        "detail": "huggingface.co resolves",
                    },
                    "modal_cli": {
                        "status": "pass",
                        "detail": "modal CLI is available through uv tool",
                        "version": "1.5.4",
                    },
                    "modal_auth": {
                        "status": modal_auth_status,
                        "detail": "Token missing",
                    },
                },
                "rootfs": {
                    "marker": "1",
                    "network_mode": "networked",
                    "cwd": "/workspace/torchtitan",
                },
            },
            sort_keys=True,
        )
    )
    return path


def _write_stage1_input_inspection_report(
    path: Path,
    *,
    status: str = "pass",
    min_total_tokens: int = 5_000_000_000,
    total_tokens: int = 5_000_000_000,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "kind": "mini_kimi_k3_stage1_input_inspection",
        "status": status,
        "input_manifest": {
            "path": str(path.parent / "source-inputs.json"),
            "sha256": "input-manifest-sha",
        },
        "input_format": "jsonl-tokens",
        "text_field": "text",
        "tokens_field": "tokens",
        "min_total_tokens": min_total_tokens,
        "total_files": 2,
        "total_documents": 3,
        "total_tokens": total_tokens,
    }
    if status != "pass":
        report["detail"] = "total token count below required minimum"
    path.write_text(json.dumps(report, sort_keys=True))
    return path


def _write_multi_stage1_input_inspection_report(
    path: Path,
    *,
    min_total_tokens: int,
    total_tokens: int,
) -> Path:
    input_a = path.parent / "source-a-inputs.json"
    input_b = path.parent / "source-b-inputs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_input_inspection",
                "status": "pass",
                "input_manifest": {
                    "paths": [str(input_a), str(input_b)],
                    "sha256": "multi-input-manifest-sha",
                },
                "input_format": "parquet-text",
                "text_fields": ["code", "text"],
                "tokens_fields": [],
                "min_total_tokens": min_total_tokens,
                "total_files": 4,
                "total_documents": 8,
                "total_tokens": total_tokens,
            },
            sort_keys=True,
        )
    )
    return path


def _write_ready_full_corpus_reports(
    reports_dir: Path,
    *,
    manifest: Path,
    tokens_dir: Path,
    available_tokens: int,
    required_tokens: int,
) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    corpus_plan_audit = reports_dir / "r1-corpus-plan-audit.json"
    corpus_plan_audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan_audit",
                "status": "ready",
                "target_tokens": required_tokens,
                "planned_tokens": required_tokens,
                "available_tokens": available_tokens,
                "deficit_tokens": 0,
                "manifest": str(manifest),
                "tokens_dir": str(tokens_dir),
            },
            sort_keys=True,
        )
    )
    source_resolution = reports_dir / "stage1-source-resolution.json"
    source_resolution.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_source_resolution",
                "status": "ready",
                "num_benchmarks": 0,
                "num_corpora": 1,
                "unresolved_benchmarks": [],
                "unresolved_corpora": [],
            },
            sort_keys=True,
        )
    )
    return corpus_plan_audit, source_resolution


def test_preflight_trainable_config_reports_blocked_r1_training_smoke(
    tmp_path: Path,
):
    forward_oracle = _write_forward_oracle_report(tmp_path / "forward_oracle.json")
    launch_backend = _write_launch_backend_report(
        tmp_path / "launch_backend.json",
        forward_oracle_report=str(forward_oracle),
    )
    training_smoke = _write_r1_training_smoke_report(
        tmp_path / "r1_training_smoke.json",
        status="blocked",
        steps=0,
        max_grad_norm=None,
        error="insufficient free GPU memory for r1 Trainer smoke",
        gpu_memory={
            "status": "blocked",
            "required_free_mib": 12_000,
            "selected": None,
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA B200",
                    "memory_total_mib": 183_359,
                    "memory_used_mib": 173_144,
                    "memory_free_mib": 9_489,
                }
            ],
        },
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            forward_oracle_report=forward_oracle,
            launch_backend_report=launch_backend,
            r1_training_smoke_report=training_smoke,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    smoke = gates["trainable_config"]["evidence"]["r1_training_smoke"]
    assert gates["trainable_config"]["status"] == "fail"
    assert (
        gates["trainable_config"]["detail"]
        == "insufficient free GPU memory for r1 Trainer smoke"
    )
    assert smoke == {
        "report": str(training_smoke),
        "status": "blocked",
        "config": "mini_kimi_k3_r1_contract",
        "model_flavor": "r1",
        "rootfs": {"marker": "1", "cwd": "/workspace/torchtitan"},
        "steps": 0,
        "max_grad_norm": None,
        "error": "insufficient free GPU memory for r1 Trainer smoke",
        "gpu_memory": {
            "status": "blocked",
            "required_free_mib": 12_000,
            "selected": None,
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA B200",
                    "memory_total_mib": 183_359,
                    "memory_used_mib": 173_144,
                    "memory_free_mib": 9_489,
                }
            ],
        },
        "command": {
            "argv": [
                "/usr/bin/env",
                "CUDA_VISIBLE_DEVICES=0",
                "MODULE=mini_kimi_k3",
                "CONFIG=mini_kimi_k3_r1_contract",
                "COMM_MODE=fake_backend",
                "NGPU=1",
                "./run_train.sh",
                "--hf_assets_path=tests/assets/tokenizer",
                "--dump_folder=r1_training_smoke_train",
                "--training.steps",
                "1",
                "--training.local_batch_size=1",
                "--training.global_batch_size=1",
                "--training.seq_len=4096",
                "--checkpoint.no-enable",
                "--dataloader.token_manifest=manifest.json",
                "--dataloader.tokens_dir=tokens",
            ],
            "cwd": "/workspace/torchtitan",
            "return_code": 0,
        },
    }


def test_preflight_report_marks_full_launch_not_ready(tmp_path: Path):
    report = tmp_path / "preflight.json"

    with pytest.raises(
        preflight.CheckFailure, match="Mini Kimi K3 full launch is not ready"
    ):
        preflight.run_preflight(mode="full", report_path=report)

    data = json.loads(report.read_text())
    assert data["schema_version"] == 1
    assert data["status"] == "blocked"
    assert data["mode"] == "full"
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["rootfs_runner"]["status"] == "pass"
    assert gates["token_shard_loader"]["status"] == "pass"
    assert gates["trainable_config"]["status"] == "partial"
    assert gates["trainable_config"]["evidence"]["module"] == "mini_kimi_k3"
    assert gates["trainable_config"]["evidence"]["config"] == "mini_kimi_k3_r1_contract"
    assert gates["trainable_config"]["evidence"]["model_spec"] == {
        "name": "mini_kimi_k3",
        "flavor": "r1_contract",
    }
    assert gates["trainable_config"]["evidence"]["build_status"] == "buildable"
    assert gates["trainable_config"]["evidence"]["model_fidelity_status"] == "partial"
    assert gates["trainable_config"]["evidence"]["r1_training_smoke"] == {
        "status": "missing"
    }
    assert gates["trainable_config"]["evidence"]["model_variant"] == "r1"
    assert gates["trainable_config"]["evidence"]["seq_len"] == 4096
    assert gates["trainable_config"]["evidence"]["local_batch_size"] == 4
    assert gates["trainable_config"]["evidence"]["global_batch_size"] == 32
    assert gates["trainable_config"]["evidence"]["tokens_per_step"] == 131_072
    assert gates["trainable_config"]["evidence"]["optimizer_steps"] == 38_147
    assert gates["trainable_config"]["evidence"]["checkpoint_interval"] == 100
    assert gates["trainable_config"]["evidence"]["adam_betas"] == [0.9, 0.95]
    assert gates["trainable_config"]["evidence"]["adam_eps"] == 1.0e-8
    assert gates["trainable_config"]["evidence"]["weight_decay_policy"] == (
        "exclude_router_parameters_and_1d_parameters"
    )
    assert "model_fidelity" in gates["trainable_config"]["detail"]
    assert gates["model_fidelity"]["status"] == "partial"
    assert gates["model_fidelity"]["evidence"]["activation"] == (
        "torchtitan/experiments/mini_kimi_k3/activation.py"
    )
    assert gates["model_fidelity"]["evidence"]["activation_tests"] == (
        "tests/unit_tests/test_mini_kimi_k3_activation.py"
    )
    assert gates["model_fidelity"]["evidence"]["router"] == (
        "torchtitan/experiments/mini_kimi_k3/router.py"
    )
    assert gates["model_fidelity"]["evidence"]["router_tests"] == (
        "tests/unit_tests/test_mini_kimi_k3_router.py"
    )
    assert gates["model_fidelity"]["evidence"]["moe"] == (
        "torchtitan/experiments/mini_kimi_k3/moe.py"
    )
    assert gates["model_fidelity"]["evidence"]["moe_tests"] == (
        "tests/unit_tests/test_mini_kimi_k3_moe.py"
    )
    assert gates["model_fidelity"]["evidence"]["kda"] == (
        "torchtitan/experiments/mini_kimi_k3/kda.py"
    )
    assert gates["model_fidelity"]["evidence"]["kda_tests"] == (
        "tests/unit_tests/test_mini_kimi_k3_kda.py"
    )
    assert gates["model_fidelity"]["evidence"]["mla"] == (
        "torchtitan/experiments/mini_kimi_k3/mla.py"
    )
    assert gates["model_fidelity"]["evidence"]["mla_tests"] == (
        "tests/unit_tests/test_mini_kimi_k3_mla.py"
    )
    assert gates["model_fidelity"]["evidence"]["model"] == (
        "torchtitan/experiments/mini_kimi_k3/model.py"
    )
    assert gates["model_fidelity"]["evidence"]["model_tests"] == (
        "tests/unit_tests/test_mini_kimi_k3_model.py"
    )
    assert gates["model_fidelity"]["evidence"]["parameter_budget"]["status"] == "pass"
    assert (
        gates["model_fidelity"]["evidence"]["parameter_budget"]["counts"][
            "total_parameters"
        ]
        == 1_023_206_628
    )
    assert (
        gates["model_fidelity"]["evidence"]["parameter_budget"]["targets"][
            "total_parameters"
        ]
        == 1_020_000_000
    )
    assert "forward oracle" in gates["model_fidelity"]["detail"]
    assert gates["model_fidelity"]["evidence"]["forward_oracle"]["status"] == "missing"
    assert gates["real_corpus_manifest"]["status"] == "missing"
    assert gates["launch_evidence_bundle"]["status"] == "missing"


def test_preflight_model_gate_records_matching_config_oracle(tmp_path: Path):
    oracle = _write_oracle(tmp_path / "oracle")
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            oracle_root=oracle,
            oracle_r1_config=oracle / "train" / "ladder_r1.json",
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["model_fidelity"]["status"] == "partial"
    assert gates["model_fidelity"]["evidence"]["oracle"] == {
        "flagship": "match",
        "r1": "match",
    }
    assert gates["model_fidelity"]["evidence"]["parameter_budget"]["status"] == "pass"
    assert gates["model_fidelity"]["evidence"]["forward_oracle"]["status"] == "missing"
    assert "forward oracle" in gates["model_fidelity"]["detail"]


def test_preflight_model_gate_records_matching_forward_oracle(tmp_path: Path):
    oracle = _write_oracle(tmp_path / "oracle")
    forward_oracle = _write_forward_oracle_report(tmp_path / "forward_oracle.json")
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            oracle_root=oracle,
            oracle_r1_config=oracle / "train" / "ladder_r1.json",
            forward_oracle_report=forward_oracle,
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["model_fidelity"]["status"] == "partial"
    assert gates["model_fidelity"]["evidence"]["oracle"] == {
        "flagship": "match",
        "r1": "match",
    }
    assert gates["model_fidelity"]["evidence"]["parameter_budget"]["status"] == "pass"
    assert gates["model_fidelity"]["evidence"]["forward_oracle"] == {
        "report": str(forward_oracle),
        "status": "pass",
        "model_flavor": "r1",
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
    }
    assert "launch-backend report" in gates["model_fidelity"]["detail"]


def test_preflight_model_gate_passes_with_launch_backend_report(tmp_path: Path):
    oracle = _write_oracle(tmp_path / "oracle")
    forward_oracle = _write_forward_oracle_report(tmp_path / "forward_oracle.json")
    launch_backend = _write_launch_backend_report(
        tmp_path / "launch_backend.json",
        forward_oracle_report=str(forward_oracle),
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            oracle_root=oracle,
            oracle_r1_config=oracle / "train" / "ladder_r1.json",
            forward_oracle_report=forward_oracle,
            launch_backend_report=launch_backend,
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert data["status"] == "blocked"
    assert gates["trainable_config"]["status"] == "partial"
    assert gates["trainable_config"]["evidence"]["build_status"] == "buildable"
    assert gates["trainable_config"]["evidence"]["model_fidelity_status"] == "pass"
    assert gates["trainable_config"]["evidence"]["launch_backend_report"] == str(
        launch_backend
    )
    assert gates["trainable_config"]["evidence"]["r1_training_smoke"] == {
        "status": "missing"
    }
    assert "r1 training smoke" in gates["trainable_config"]["detail"]
    assert gates["model_fidelity"]["status"] == "pass"
    assert gates["model_fidelity"]["evidence"]["launch_backend"] == {
        "report": str(launch_backend),
        "status": "pass",
        "model_flavor": "r1",
        "backend": "torchtitan.experiments.mini_kimi_k3",
        "config": "mini_kimi_k3_r1_contract",
        "forward_oracle_report": str(forward_oracle),
        "forward_trace_report": "forward_trace.json",
        "final_logit_max_abs_diff": 0.0,
    }
    assert gates["real_corpus_manifest"]["status"] == "missing"


def test_preflight_trainable_config_passes_with_r1_training_smoke_report(
    tmp_path: Path,
):
    oracle = _write_oracle(tmp_path / "oracle")
    forward_oracle = _write_forward_oracle_report(tmp_path / "forward_oracle.json")
    launch_backend = _write_launch_backend_report(
        tmp_path / "launch_backend.json",
        forward_oracle_report=str(forward_oracle),
    )
    token_manifest = tmp_path / "manifest.json"
    tokens_dir = tmp_path / "tokens"
    training_smoke = _write_r1_training_smoke_report(
        tmp_path / "r1_training_smoke.json",
        manifest=str(token_manifest),
        tokens_dir=str(tokens_dir),
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            oracle_root=oracle,
            oracle_r1_config=oracle / "train" / "ladder_r1.json",
            forward_oracle_report=forward_oracle,
            launch_backend_report=launch_backend,
            r1_training_smoke_report=training_smoke,
            token_manifest=token_manifest,
            tokens_dir=tokens_dir,
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert data["status"] == "blocked"
    assert gates["trainable_config"]["status"] == "pass"
    assert gates["trainable_config"]["evidence"]["build_status"] == "launchable"
    assert gates["trainable_config"]["evidence"]["r1_training_smoke"] == {
        "report": str(training_smoke),
        "status": "pass",
        "config": "mini_kimi_k3_r1_contract",
        "model_flavor": "r1",
        "rootfs": {"marker": "1", "cwd": "/workspace/torchtitan"},
        "steps": 1,
        "max_grad_norm": 1.0,
        "gpu_memory": {
            "status": "pass",
            "required_free_mib": 12_000,
            "selected": {"index": 0, "memory_free_mib": 48_000},
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA B200",
                    "memory_total_mib": 183_359,
                    "memory_used_mib": 135_359,
                    "memory_free_mib": 48_000,
                }
            ],
        },
        "command": {
            "argv": [
                "/usr/bin/env",
                "CUDA_VISIBLE_DEVICES=0",
                "MODULE=mini_kimi_k3",
                "CONFIG=mini_kimi_k3_r1_contract",
                "COMM_MODE=fake_backend",
                "NGPU=1",
                "./run_train.sh",
                "--hf_assets_path=tests/assets/tokenizer",
                "--dump_folder=r1_training_smoke_train",
                "--training.steps",
                "1",
                "--training.local_batch_size=1",
                "--training.global_batch_size=1",
                "--training.seq_len=4096",
                "--checkpoint.no-enable",
                f"--dataloader.token_manifest={token_manifest}",
                f"--dataloader.tokens_dir={tokens_dir}",
            ],
            "cwd": "/workspace/torchtitan",
            "return_code": 0,
        },
    }
    assert gates["model_fidelity"]["status"] == "pass"
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert (
        gates["real_corpus_manifest"]["description"]
        == "token manifest failed validation"
    )
    assert str(token_manifest) in gates["real_corpus_manifest"]["detail"]


def test_preflight_trainable_config_rejects_r1_smoke_without_requested_corpus_paths(
    tmp_path: Path,
):
    oracle = _write_oracle(tmp_path / "oracle")
    forward_oracle = _write_forward_oracle_report(tmp_path / "forward_oracle.json")
    launch_backend = _write_launch_backend_report(
        tmp_path / "launch_backend.json",
        forward_oracle_report=str(forward_oracle),
    )
    training_smoke = _write_r1_training_smoke_report(
        tmp_path / "r1_training_smoke.json",
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            oracle_root=oracle,
            oracle_r1_config=oracle / "train" / "ladder_r1.json",
            forward_oracle_report=forward_oracle,
            launch_backend_report=launch_backend,
            r1_training_smoke_report=training_smoke,
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["trainable_config"]["status"] == "fail"
    assert (
        "requested token manifest and tokens dir" in gates["trainable_config"]["detail"]
    )


def test_preflight_trainable_config_rejects_wrong_r1_training_smoke_report(
    tmp_path: Path,
):
    forward_oracle = _write_forward_oracle_report(tmp_path / "forward_oracle.json")
    launch_backend = _write_launch_backend_report(
        tmp_path / "launch_backend.json",
        forward_oracle_report=str(forward_oracle),
    )
    training_smoke = _write_r1_training_smoke_report(
        tmp_path / "r1_training_smoke.json",
        model_flavor="tiny",
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            forward_oracle_report=forward_oracle,
            launch_backend_report=launch_backend,
            r1_training_smoke_report=training_smoke,
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["trainable_config"]["status"] == "fail"
    assert "model_flavor r1" in gates["trainable_config"]["detail"]


def test_preflight_trainable_config_rejects_r1_smoke_without_gpu_evidence(
    tmp_path: Path,
):
    forward_oracle = _write_forward_oracle_report(tmp_path / "forward_oracle.json")
    launch_backend = _write_launch_backend_report(
        tmp_path / "launch_backend.json",
        forward_oracle_report=str(forward_oracle),
    )
    training_smoke = _write_r1_training_smoke_report(
        tmp_path / "r1_training_smoke.json",
        gpu_memory={
            "status": "skipped",
            "required_free_mib": 12_000,
            "selected": None,
            "devices": [],
        },
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            forward_oracle_report=forward_oracle,
            launch_backend_report=launch_backend,
            r1_training_smoke_report=training_smoke,
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["trainable_config"]["status"] == "fail"
    assert (
        gates["trainable_config"]["detail"]
        == "r1 training smoke report has no selected GPU capacity evidence"
    )


def test_preflight_trainable_config_rejects_r1_smoke_without_rootfs_command(
    tmp_path: Path,
):
    forward_oracle = _write_forward_oracle_report(tmp_path / "forward_oracle.json")
    launch_backend = _write_launch_backend_report(
        tmp_path / "launch_backend.json",
        forward_oracle_report=str(forward_oracle),
    )
    training_smoke = _write_r1_training_smoke_report(
        tmp_path / "r1_training_smoke.json",
        rootfs={"marker": None, "cwd": "/tmp"},
        command={
            "argv": ["/usr/bin/env", "MODULE=mini_kimi_k3", "python", "train.py"],
            "cwd": "/tmp",
            "return_code": 0,
        },
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            forward_oracle_report=forward_oracle,
            launch_backend_report=launch_backend,
            r1_training_smoke_report=training_smoke,
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["trainable_config"]["status"] == "fail"
    assert (
        gates["trainable_config"]["detail"]
        == "r1 training smoke report was not produced inside TorchTitan rootfs"
    )


@pytest.mark.parametrize(
    ("command_override", "detail_fragment"),
    [
        (
            "--dataloader.token_manifest=other-manifest.json",
            "exactly one token manifest",
        ),
        (
            "--dataloader.tokens_dir=other-tokens",
            "exactly one tokens dir",
        ),
        (
            "--training.seq_len=2048",
            "exactly one seq_len",
        ),
    ],
)
def test_preflight_trainable_config_rejects_r1_smoke_command_data_mismatch(
    tmp_path: Path,
    command_override: str,
    detail_fragment: str,
):
    forward_oracle = _write_forward_oracle_report(tmp_path / "forward_oracle.json")
    launch_backend = _write_launch_backend_report(
        tmp_path / "launch_backend.json",
        forward_oracle_report=str(forward_oracle),
    )
    manifest = tmp_path / "manifest.json"
    tokens_dir = tmp_path / "tokens"
    command = {
        "argv": [
            "/usr/bin/env",
            "CUDA_VISIBLE_DEVICES=0",
            "MODULE=mini_kimi_k3",
            "CONFIG=mini_kimi_k3_r1_contract",
            "COMM_MODE=fake_backend",
            "NGPU=1",
            "./run_train.sh",
            "--hf_assets_path=tests/assets/tokenizer",
            "--dump_folder=r1_training_smoke_train",
            "--training.steps",
            "1",
            "--training.local_batch_size=1",
            "--training.global_batch_size=1",
            f"--dataloader.token_manifest={manifest}",
            f"--dataloader.tokens_dir={tokens_dir}",
            "--training.seq_len=4096",
            "--checkpoint.no-enable",
            command_override,
        ],
        "cwd": "/workspace/torchtitan",
        "return_code": 0,
    }
    training_smoke = _write_r1_training_smoke_report(
        tmp_path / "r1_training_smoke.json",
        manifest=str(manifest),
        tokens_dir=str(tokens_dir),
        command=command,
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            forward_oracle_report=forward_oracle,
            launch_backend_report=launch_backend,
            r1_training_smoke_report=training_smoke,
            token_manifest=manifest,
            tokens_dir=tokens_dir,
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["trainable_config"]["status"] == "fail"
    assert detail_fragment in gates["trainable_config"]["detail"]


@pytest.mark.parametrize(
    ("report_kwargs", "command", "detail_fragment"),
    [
        (
            {"rootfs": {"marker": "1", "cwd": "/tmp"}},
            None,
            "rootfs cwd",
        ),
        (
            {},
            {"cwd": "/tmp"},
            "command cwd",
        ),
        (
            {},
            {"argv_append": ["--training.steps", "0"]},
            "training steps",
        ),
        (
            {"gpu_memory": {"selected": {"index": 1}}},
            None,
            "GPU selection",
        ),
        (
            {},
            {"argv_remove": ["--checkpoint.no-enable"]},
            "checkpoint disabling",
        ),
        (
            {},
            {
                "argv_remove": ["MODULE=mini_kimi_k3"],
                "argv_append": ["MODULE=mini_kimi_k3"],
            },
            "environment arguments",
        ),
        (
            {},
            {"argv_append": ["./run_train.sh"]},
            "exactly one ./run_train.sh",
        ),
        (
            {},
            {
                "argv_replace": {
                    "--hf_assets_path=tests/assets/tokenizer": "--hf_assets_path=other-tokenizer"
                }
            },
            "hf assets path",
        ),
        (
            {},
            {"argv_remove": ["--dump_folder=r1_training_smoke_train"]},
            "dump folder",
        ),
    ],
)
def test_preflight_trainable_config_rejects_r1_smoke_bad_command_contract(
    tmp_path: Path,
    report_kwargs: dict,
    command: dict | None,
    detail_fragment: str,
):
    forward_oracle = _write_forward_oracle_report(tmp_path / "forward_oracle.json")
    launch_backend = _write_launch_backend_report(
        tmp_path / "launch_backend.json",
        forward_oracle_report=str(forward_oracle),
    )
    token_manifest = tmp_path / "manifest.json"
    tokens_dir = tmp_path / "tokens"
    base_command = {
        "argv": [
            "/usr/bin/env",
            "CUDA_VISIBLE_DEVICES=0",
            "MODULE=mini_kimi_k3",
            "CONFIG=mini_kimi_k3_r1_contract",
            "COMM_MODE=fake_backend",
            "NGPU=1",
            "./run_train.sh",
            "--hf_assets_path=tests/assets/tokenizer",
            "--dump_folder=r1_training_smoke_train",
            "--training.steps",
            "1",
            "--training.local_batch_size=1",
            "--training.global_batch_size=1",
            "--training.seq_len=4096",
            "--checkpoint.no-enable",
            f"--dataloader.token_manifest={token_manifest}",
            f"--dataloader.tokens_dir={tokens_dir}",
        ],
        "cwd": "/workspace/torchtitan",
        "return_code": 0,
    }
    if command is not None:
        if "cwd" in command:
            base_command["cwd"] = command["cwd"]
        for old, new in command.get("argv_replace", {}).items():
            base_command["argv"] = [
                new if item == old else item for item in base_command["argv"]
            ]
        for removed in command.get("argv_remove", []):
            base_command["argv"] = [
                item for item in base_command["argv"] if item != removed
            ]
        if "argv_append" in command:
            base_command["argv"].extend(command["argv_append"])
    gpu_memory = report_kwargs.pop("gpu_memory", None)
    if gpu_memory is not None:
        gpu_memory = {
            "status": "pass",
            "required_free_mib": 12_000,
            "selected": {"index": 0, "memory_free_mib": 48_000},
            "devices": [],
        } | gpu_memory
    training_smoke = _write_r1_training_smoke_report(
        tmp_path / "r1_training_smoke.json",
        manifest=str(token_manifest),
        tokens_dir=str(tokens_dir),
        command=base_command,
        gpu_memory=gpu_memory,
        **report_kwargs,
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            forward_oracle_report=forward_oracle,
            launch_backend_report=launch_backend,
            r1_training_smoke_report=training_smoke,
            token_manifest=token_manifest,
            tokens_dir=tokens_dir,
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["trainable_config"]["status"] == "fail"
    assert detail_fragment in gates["trainable_config"]["detail"]


def test_preflight_model_gate_fails_on_launch_backend_report_mismatch(
    tmp_path: Path,
):
    forward_oracle = _write_forward_oracle_report(tmp_path / "forward_oracle.json")
    launch_backend = _write_launch_backend_report(
        tmp_path / "launch_backend.json",
        model_flavor="tiny",
        forward_oracle_report=str(forward_oracle),
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            forward_oracle_report=forward_oracle,
            launch_backend_report=launch_backend,
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["model_fidelity"]["status"] == "fail"
    assert "model_flavor r1" in gates["model_fidelity"]["detail"]


def test_preflight_model_gate_fails_on_forward_oracle_mismatch(tmp_path: Path):
    forward_oracle = _write_forward_oracle_report(
        tmp_path / "forward_oracle.json",
        status="fail",
        max_abs_diff=0.125,
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            forward_oracle_report=forward_oracle,
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["model_fidelity"]["status"] == "fail"
    assert gates["model_fidelity"]["evidence"]["forward_oracle"]["status"] == "fail"
    assert "did not pass" in gates["model_fidelity"]["detail"]


def test_preflight_model_gate_surfaces_forward_oracle_error(tmp_path: Path):
    forward_oracle = _write_forward_oracle_report(
        tmp_path / "forward_oracle.json",
        status="fail",
    )
    data = json.loads(forward_oracle.read_text())
    data["error"] = "logits file not found: torchtitan_r1_logits.pt"
    forward_oracle.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            forward_oracle_report=forward_oracle,
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["model_fidelity"]["status"] == "fail"
    assert "logits file not found" in gates["model_fidelity"]["detail"]


def test_preflight_model_gate_fails_on_oracle_mismatch(tmp_path: Path):
    oracle = _write_oracle(tmp_path / "oracle")
    r1_path = oracle / "train" / "ladder_r1.json"
    r1 = json.loads(r1_path.read_text())
    r1["hidden_size"] = 768
    r1_path.write_text(json.dumps(r1, sort_keys=True))
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            oracle_root=oracle,
            oracle_r1_config=r1_path,
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["model_fidelity"]["status"] == "fail"
    assert gates["model_fidelity"]["evidence"]["oracle"]["flagship"] == "match"
    assert gates["model_fidelity"]["evidence"]["oracle"]["r1"].startswith("mismatch:")


def test_preflight_reports_host_python_as_rootfs_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    report = tmp_path / "preflight.json"
    monkeypatch.delenv("TORCHTITAN_IN_ROOTFS", raising=False)

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(mode="full", report_path=report)

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["rootfs_runner"]["status"] == "fail"
    assert (
        "not running inside the TorchTitan rootfs" in gates["rootfs_runner"]["detail"]
    )


def test_preflight_validates_tiny_token_manifest(tmp_path: Path):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="tiny-plumbing",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "pass"
    assert gates["real_corpus_manifest"]["evidence"]["manifest"] == str(manifest)
    assert gates["real_corpus_manifest"]["evidence"]["tokens_dir"] == str(
        tmp_path / "tokens"
    )
    assert gates["trainable_config"]["status"] == "pass"
    assert (
        gates["trainable_config"]["evidence"]["config"] == "mini_kimi_k3_tiny_plumbing"
    )
    assert gates["tiny_training_smoke"]["status"] == "missing"


def test_preflight_accepts_tiny_training_smoke_report(tmp_path: Path):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", list(range(16)))
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    smoke_report = tmp_path / "smoke.json"
    run_training_smoke(
        token_manifest=manifest,
        tokens_dir=tmp_path / "tokens",
        report_path=smoke_report,
        seq_len=4,
        batch_size=2,
        steps=1,
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="tiny-plumbing",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            tiny_smoke_report=smoke_report,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["tiny_training_smoke"]["status"] == "pass"
    assert gates["tiny_training_smoke"]["evidence"]["report"] == str(smoke_report)
    assert gates["tiny_training_smoke"]["evidence"]["steps"] == 1


def test_preflight_rejects_stale_tiny_training_smoke_report(tmp_path: Path):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", list(range(16)))
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    smoke_report = tmp_path / "smoke.json"
    smoke_report.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_training_smoke",
                "status": "pass",
                "schema_version": 1,
                "data": {
                    "manifest": "different.json",
                    "tokens_dir": str(tmp_path / "tokens"),
                    "seq_len": 4,
                },
                "optimization": {"steps": 1},
            }
        )
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="tiny-plumbing",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            tiny_smoke_report=smoke_report,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["tiny_training_smoke"]["status"] == "fail"
    assert (
        "does not match requested token manifest"
        in gates["tiny_training_smoke"]["detail"]
    )


def test_full_preflight_rejects_token_manifest_below_r1_budget(tmp_path: Path):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert "fewer tokens than the r1 target" in gates["real_corpus_manifest"]["detail"]
    assert gates["real_corpus_manifest"]["evidence"]["available_tokens"] == 4
    assert gates["real_corpus_manifest"]["evidence"]["required_tokens"] == 5_000_000_000


def test_full_preflight_includes_r1_corpus_plan_audit_evidence(tmp_path: Path):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    audit = tmp_path / "reports" / "r1-corpus-plan-audit.json"
    audit.parent.mkdir()
    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan_audit",
                "status": "blocked",
                "target_tokens": 5_000_000_000,
                "planned_tokens": 5_000_000_000,
                "available_tokens": 4,
                "deficit_tokens": 4_999_999_996,
                "manifest": str(manifest),
                "tokens_dir": str(tmp_path / "tokens"),
                "corpus_plan": str(tmp_path / "reports" / "r1-corpus-plan.json"),
                "sources": {
                    "alpha": {
                        "required_tokens": 5_000_000_000,
                        "available_tokens": 4,
                        "deficit_tokens": 4_999_999_996,
                        "status": "short",
                    }
                },
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=audit,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "fail"
    assert corpus["evidence"]["r1_corpus_plan_audit"] == {
        "report": str(audit),
        "status": "blocked",
        "target_tokens": 5_000_000_000,
        "planned_tokens": 5_000_000_000,
        "available_tokens": 4,
        "deficit_tokens": 4_999_999_996,
        "sources": {
            "alpha": {
                "required_tokens": 5_000_000_000,
                "available_tokens": 4,
                "deficit_tokens": 4_999_999_996,
                "status": "short",
            }
        },
    }


def test_full_preflight_requires_r1_corpus_plan_audit_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    _attach_decontamination_report(manifest, tmp_path / "reports" / "decontam.json")
    _set_full_token_budget(monkeypatch, 4)
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "fail"
    assert "corpus plan audit report is required" in corpus["detail"]


def test_full_preflight_rejects_unresolved_stage1_source_resolution(
    tmp_path: Path,
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    source_resolution = tmp_path / "reports" / "stage1-source-resolution.json"
    source_resolution.parent.mkdir()
    source_resolution.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_source_resolution",
                "status": "blocked",
                "num_benchmarks": 1,
                "num_corpora": 1,
                "unresolved_benchmarks": [],
                "unresolved_corpora": ["web-diverse"],
                "sources": {
                    "web-diverse": {
                        "status": "unresolved",
                        "tried": [{"repo": "nvidia/Nemotron-CC-v2"}],
                    }
                },
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            stage1_source_resolution_report=source_resolution,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "fail"
    assert corpus["evidence"]["stage1_source_resolution"] == {
        "report": str(source_resolution),
        "status": "blocked",
        "num_benchmarks": 1,
        "num_corpora": 1,
        "unresolved_benchmarks": [],
        "unresolved_corpora": ["web-diverse"],
    }
    assert any(
        "stage1 source resolution report must be ready" in error
        for error in corpus["evidence"]["metadata_errors"]
    )


def test_full_preflight_requires_stage1_source_resolution_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    _attach_decontamination_report(manifest, tmp_path / "reports" / "decontam.json")
    _set_full_token_budget(monkeypatch, 4)
    audit = tmp_path / "reports" / "r1-corpus-plan-audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan_audit",
                "status": "ready",
                "target_tokens": 4,
                "planned_tokens": 4,
                "available_tokens": 4,
                "deficit_tokens": 0,
                "manifest": str(manifest),
                "tokens_dir": str(tmp_path / "tokens"),
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=audit,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "fail"
    assert "stage1 source resolution report is required" in corpus["detail"]


def test_full_preflight_rejects_blocked_stage1_remote_readiness(
    tmp_path: Path,
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    remote_readiness = _write_stage1_remote_readiness_report(
        tmp_path / "reports" / "stage1-remote-readiness.json"
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            stage1_remote_readiness_report=remote_readiness,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "fail"
    assert corpus["evidence"]["stage1_remote_readiness"] == {
        "report": str(remote_readiness),
        "status": "blocked",
        "huggingface_dns": "pass",
        "modal_cli": "pass",
        "modal_auth": "blocked",
        "rootfs_marker": "1",
        "network_mode": "networked",
    }
    assert any(
        "stage1 remote readiness report must be ready" in error
        for error in corpus["evidence"]["metadata_errors"]
    )


def test_full_preflight_reports_blocked_stage1_remote_env_file(
    tmp_path: Path,
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    remote_readiness = tmp_path / "reports" / "stage1-remote-readiness.json"
    remote_readiness.parent.mkdir()
    remote_readiness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_remote_readiness",
                "status": "blocked",
                "checks": {
                    "env_file": {
                        "status": "blocked",
                        "path": ".scratch/mini-kimi-k3-replication/.env",
                        "detail": (
                            "env file does not exist: "
                            ".scratch/mini-kimi-k3-replication/.env"
                        ),
                        "template": ".scratch/mini-kimi-k3-replication/.env.example",
                        "required_keys": [
                            "MODAL_TOKEN_ID",
                            "MODAL_TOKEN_SECRET",
                            "MODAL_PROFILE",
                        ],
                    }
                },
                "rootfs": {"marker": "1", "network_mode": "networked"},
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            stage1_remote_readiness_report=remote_readiness,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "fail"
    assert corpus["evidence"]["stage1_remote_readiness"]["env_file"] == {
        "status": "blocked",
        "path": ".scratch/mini-kimi-k3-replication/.env",
        "detail": (
            "env file does not exist: " ".scratch/mini-kimi-k3-replication/.env"
        ),
        "template": ".scratch/mini-kimi-k3-replication/.env.example",
        "required_keys": [
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "MODAL_PROFILE",
        ],
    }
    assert any(
        "stage1 remote readiness env file is blocked" in error
        for error in corpus["evidence"]["metadata_errors"]
    )


def test_full_preflight_preserves_ready_stage1_remote_env_file_evidence(
    tmp_path: Path,
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    env_file = tmp_path / ".env"
    remote_readiness = tmp_path / "reports" / "stage1-remote-readiness.json"
    remote_readiness.parent.mkdir()
    remote_readiness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_remote_readiness",
                "status": "ready",
                "env_file": {
                    "status": "loaded",
                    "path": str(env_file),
                    "keys": [
                        "MODAL_PROFILE",
                        "MODAL_TOKEN_ID",
                        "MODAL_TOKEN_SECRET",
                    ],
                },
                "checks": {
                    "huggingface_dns": {"status": "pass"},
                    "modal_cli": {"status": "pass"},
                    "modal_auth": {"status": "pass"},
                },
                "rootfs": {"marker": "1", "network_mode": "networked"},
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            stage1_remote_readiness_report=remote_readiness,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "fail"
    assert corpus["evidence"]["stage1_remote_readiness"]["env_file"] == {
        "status": "loaded",
        "path": str(env_file),
        "keys": [
            "MODAL_PROFILE",
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
        ],
    }
    assert corpus["evidence"]["stage1_remote_readiness"]["rootfs_marker"] == "1"


def test_full_preflight_rejects_ready_stage1_remote_without_rootfs_marker(
    tmp_path: Path,
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    remote_readiness = tmp_path / "reports" / "stage1-remote-readiness.json"
    remote_readiness.parent.mkdir()
    remote_readiness.write_text(
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
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            stage1_remote_readiness_report=remote_readiness,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "fail"
    assert any(
        "stage1 remote readiness report rootfs marker must be 1" in error
        for error in corpus["evidence"]["metadata_errors"]
    )


def test_full_preflight_treats_remote_readiness_as_optional_for_local_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    _attach_decontamination_report(
        manifest, tmp_path / "reports" / "decontamination.json"
    )
    local_report = tmp_path / "reports" / "alpha-provenance.json"
    local_report.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_source_provenance",
                "schema_version": 1,
                "source": "alpha",
                "total_tokens": 4,
            },
            sort_keys=True,
        )
    )
    data = json.loads(manifest.read_text())
    data["sources"]["alpha"]["provenance"] = {
        "snapshot": "2026-08-mini-k3-fixture",
        "license": "fixture",
        "report": str(local_report),
        "sha256": hashlib.sha256(local_report.read_bytes()).hexdigest(),
    }
    data["sources"]["alpha"]["local_stage1"] = {
        "report": str(local_report),
        "sha256": hashlib.sha256(local_report.read_bytes()).hexdigest(),
        "total_tokens": 4,
        "input_format": "jsonl",
    }
    manifest.write_text(json.dumps(data, sort_keys=True))
    remote_readiness = _write_stage1_remote_readiness_report(
        tmp_path / "reports" / "stage1-remote-readiness.json"
    )
    corpus_plan_audit = tmp_path / "reports" / "r1-corpus-plan-audit.json"
    corpus_plan_audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan_audit",
                "status": "ready",
                "target_tokens": 4,
                "planned_tokens": 4,
                "available_tokens": 4,
                "deficit_tokens": 0,
                "manifest": str(manifest),
                "tokens_dir": str(tmp_path / "tokens"),
            },
            sort_keys=True,
        )
    )
    source_resolution = tmp_path / "reports" / "stage1-source-resolution.json"
    source_resolution.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_source_resolution",
                "status": "ready",
                "num_benchmarks": 0,
                "num_corpora": 1,
                "unresolved_benchmarks": [],
                "unresolved_corpora": [],
            },
            sort_keys=True,
        )
    )
    input_inspection = _write_stage1_input_inspection_report(
        tmp_path / "reports" / "stage1-input-inspection.json",
        min_total_tokens=4,
        total_tokens=4,
    )
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=corpus_plan_audit,
            stage1_source_resolution_report=source_resolution,
            stage1_remote_readiness_report=remote_readiness,
            stage1_input_inspection_report=input_inspection,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "pass"
    assert corpus["evidence"]["stage1_remote_readiness"]["status"] == "not_required"
    assert corpus["evidence"]["stage1_remote_readiness"]["report"] == str(
        remote_readiness
    )
    assert "metadata_errors" not in corpus["evidence"]


def test_full_preflight_rejects_local_stage1_report_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    _attach_decontamination_report(
        manifest, tmp_path / "reports" / "decontamination.json"
    )
    local_report = tmp_path / "reports" / "alpha-provenance.json"
    local_report.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_source_provenance",
                "schema_version": 1,
                "source": "alpha",
                "total_tokens": 4,
            },
            sort_keys=True,
        )
    )
    data = json.loads(manifest.read_text())
    data["sources"]["alpha"]["provenance"] = {
        "snapshot": "2026-08-mini-k3-fixture",
        "license": "fixture",
        "report": str(local_report),
        "sha256": hashlib.sha256(local_report.read_bytes()).hexdigest(),
    }
    data["sources"]["alpha"]["local_stage1"] = {
        "report": str(local_report),
        "sha256": "stale-local-stage1-hash",
        "total_tokens": 4,
        "input_format": "jsonl-tokens",
    }
    manifest.write_text(json.dumps(data, sort_keys=True))
    remote_readiness = _write_stage1_remote_readiness_report(
        tmp_path / "reports" / "stage1-remote-readiness.json"
    )
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            stage1_remote_readiness_report=remote_readiness,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "fail"
    assert "local_stage1 report sha256 mismatch" in corpus["detail"]


def test_full_preflight_rejects_local_stage1_token_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    _attach_decontamination_report(
        manifest, tmp_path / "reports" / "decontamination.json"
    )
    local_report = tmp_path / "reports" / "alpha-provenance.json"
    local_report.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_source_provenance",
                "schema_version": 1,
                "source": "alpha",
                "total_tokens": 3,
            },
            sort_keys=True,
        )
    )
    data = json.loads(manifest.read_text())
    data["sources"]["alpha"]["provenance"] = {
        "snapshot": "2026-08-mini-k3-fixture",
        "license": "fixture",
        "report": str(local_report),
        "sha256": hashlib.sha256(local_report.read_bytes()).hexdigest(),
    }
    data["sources"]["alpha"]["local_stage1"] = {
        "report": str(local_report),
        "sha256": hashlib.sha256(local_report.read_bytes()).hexdigest(),
        "total_tokens": 4,
        "input_format": "jsonl-tokens",
    }
    manifest.write_text(json.dumps(data, sort_keys=True))
    remote_readiness = _write_stage1_remote_readiness_report(
        tmp_path / "reports" / "stage1-remote-readiness.json"
    )
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            stage1_remote_readiness_report=remote_readiness,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "fail"
    assert "local_stage1 total_tokens mismatch" in corpus["detail"]


def test_full_preflight_requires_input_inspection_for_local_stage1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    _attach_decontamination_report(
        manifest, tmp_path / "reports" / "decontamination.json"
    )
    local_report = tmp_path / "reports" / "alpha-provenance.json"
    local_report.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_source_provenance",
                "schema_version": 1,
                "source": "alpha",
                "total_tokens": 4,
            },
            sort_keys=True,
        )
    )
    data = json.loads(manifest.read_text())
    data["sources"]["alpha"]["provenance"] = {
        "snapshot": "2026-08-mini-k3-fixture",
        "license": "fixture",
        "report": str(local_report),
        "sha256": hashlib.sha256(local_report.read_bytes()).hexdigest(),
    }
    data["sources"]["alpha"]["local_stage1"] = {
        "report": str(local_report),
        "sha256": hashlib.sha256(local_report.read_bytes()).hexdigest(),
        "total_tokens": 4,
        "input_format": "jsonl-tokens",
    }
    manifest.write_text(json.dumps(data, sort_keys=True))
    _attach_local_stage1_decontaminated_build(
        manifest,
        source="alpha",
        shard_name="a.bin",
        input_format="jsonl-tokens",
        tokens=4,
    )
    remote_readiness = _write_stage1_remote_readiness_report(
        tmp_path / "reports" / "stage1-remote-readiness.json"
    )
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            stage1_remote_readiness_report=remote_readiness,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "fail"
    assert "stage1 input inspection report is required" in corpus["detail"]


def test_full_preflight_accepts_input_inspection_for_local_stage1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    _attach_decontamination_report(
        manifest, tmp_path / "reports" / "decontamination.json"
    )
    local_report = tmp_path / "reports" / "alpha-provenance.json"
    local_report.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_source_provenance",
                "schema_version": 1,
                "source": "alpha",
                "total_tokens": 4,
            },
            sort_keys=True,
        )
    )
    data = json.loads(manifest.read_text())
    data["sources"]["alpha"]["provenance"] = {
        "snapshot": "2026-08-mini-k3-fixture",
        "license": "fixture",
        "report": str(local_report),
        "sha256": hashlib.sha256(local_report.read_bytes()).hexdigest(),
    }
    data["sources"]["alpha"]["local_stage1"] = {
        "report": str(local_report),
        "sha256": hashlib.sha256(local_report.read_bytes()).hexdigest(),
        "total_tokens": 4,
        "input_format": "jsonl-tokens",
    }
    manifest.write_text(json.dumps(data, sort_keys=True))
    _attach_local_stage1_decontaminated_build(
        manifest,
        source="alpha",
        shard_name="a.bin",
        input_format="jsonl-tokens",
        tokens=4,
    )
    remote_readiness = _write_stage1_remote_readiness_report(
        tmp_path / "reports" / "stage1-remote-readiness.json"
    )
    corpus_plan_audit = tmp_path / "reports" / "r1-corpus-plan-audit.json"
    corpus_plan_audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan_audit",
                "status": "ready",
                "target_tokens": 4,
                "planned_tokens": 4,
                "available_tokens": 4,
                "deficit_tokens": 0,
                "manifest": str(manifest),
                "tokens_dir": str(tmp_path / "tokens"),
            },
            sort_keys=True,
        )
    )
    source_resolution = tmp_path / "reports" / "stage1-source-resolution.json"
    source_resolution.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_source_resolution",
                "status": "ready",
                "num_benchmarks": 0,
                "num_corpora": 1,
                "unresolved_benchmarks": [],
                "unresolved_corpora": [],
            },
            sort_keys=True,
        )
    )
    input_inspection = _write_stage1_input_inspection_report(
        tmp_path / "reports" / "stage1-input-inspection.json",
        min_total_tokens=4,
        total_tokens=4,
    )
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=corpus_plan_audit,
            stage1_source_resolution_report=source_resolution,
            stage1_remote_readiness_report=remote_readiness,
            stage1_input_inspection_report=input_inspection,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "pass"
    assert corpus["evidence"]["stage1_input_inspection"] == {
        "report": str(input_inspection),
        "status": "pass",
        "input_manifest": {
            "path": str(input_inspection.parent / "source-inputs.json"),
            "sha256": "input-manifest-sha",
        },
        "input_format": "jsonl-tokens",
        "min_total_tokens": 4,
        "total_files": 2,
        "total_documents": 3,
        "total_tokens": 4,
    }


def test_full_preflight_accepts_multi_manifest_input_inspection_for_local_stage1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    _write_shard(tmp_path / "tokens" / "beta" / "b.bin", [5, 6, 7, 8])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"], "beta": ["b.bin"]},
    )
    _attach_decontamination_report(
        manifest, tmp_path / "reports" / "decontamination.json"
    )
    for source in ("alpha", "beta"):
        local_report = tmp_path / "reports" / f"{source}-provenance.json"
        local_report.write_text(
            json.dumps(
                {
                    "kind": "mini_kimi_k3_source_provenance",
                    "schema_version": 1,
                    "source": source,
                    "total_tokens": 4,
                },
                sort_keys=True,
            )
        )
        data = json.loads(manifest.read_text())
        data["sources"][source]["provenance"] = {
            "snapshot": "2026-08-mini-k3-fixture",
            "license": "fixture",
            "report": str(local_report),
            "sha256": hashlib.sha256(local_report.read_bytes()).hexdigest(),
        }
        data["sources"][source]["local_stage1"] = {
            "report": str(local_report),
            "sha256": hashlib.sha256(local_report.read_bytes()).hexdigest(),
            "total_tokens": 4,
            "input_format": "parquet-text",
        }
        manifest.write_text(json.dumps(data, sort_keys=True))
        _attach_local_stage1_decontaminated_build(
            manifest,
            source=source,
            shard_name={"alpha": "a.bin", "beta": "b.bin"}[source],
            input_format="parquet-text",
            tokens=4,
        )
    corpus_plan_audit = tmp_path / "reports" / "r1-corpus-plan-audit.json"
    corpus_plan_audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan_audit",
                "status": "ready",
                "target_tokens": 8,
                "planned_tokens": 8,
                "available_tokens": 8,
                "deficit_tokens": 0,
                "manifest": str(manifest),
                "tokens_dir": str(tmp_path / "tokens"),
            },
            sort_keys=True,
        )
    )
    source_resolution = tmp_path / "reports" / "stage1-source-resolution.json"
    source_resolution.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_stage1_source_resolution",
                "status": "ready",
                "num_benchmarks": 0,
                "num_corpora": 2,
                "unresolved_benchmarks": [],
                "unresolved_corpora": [],
            },
            sort_keys=True,
        )
    )
    input_inspection = _write_multi_stage1_input_inspection_report(
        tmp_path / "reports" / "stage1-input-inspection.json",
        min_total_tokens=8,
        total_tokens=8,
    )
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 8)

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=corpus_plan_audit,
            stage1_source_resolution_report=source_resolution,
            stage1_input_inspection_report=input_inspection,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "pass"
    assert corpus["evidence"]["stage1_input_inspection"]["input_manifest"] == {
        "paths": [
            str(input_inspection.parent / "source-a-inputs.json"),
            str(input_inspection.parent / "source-b-inputs.json"),
        ],
        "sha256": "multi-input-manifest-sha",
    }


def test_full_preflight_rejects_failing_stage1_input_inspection_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    inspection = _write_stage1_input_inspection_report(
        tmp_path / "reports" / "stage1-input-inspection.json",
        status="fail",
        min_total_tokens=5_000_000_000,
        total_tokens=4,
    )
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            stage1_input_inspection_report=inspection,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "fail"
    assert "stage1 input inspection report must pass" in corpus["detail"]


def test_full_preflight_rejects_blocked_corpus_disk_readiness(
    tmp_path: Path,
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    disk_readiness = tmp_path / "reports" / "corpus-disk-readiness.json"
    disk_readiness.parent.mkdir()
    disk_readiness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_corpus_disk_readiness",
                "status": "blocked",
                "path": str(tmp_path / "tokens"),
                "measured_path": str(tmp_path),
                "target_tokens": 5_000_000_000,
                "uint32_bytes_per_token": 4,
                "required_raw_bytes": 20_000_000_000,
                "overhead_fraction": 0.20,
                "required_bytes": 24_000_000_000,
                "free_bytes": 12_000_000_000,
                "deficit_bytes": 12_000_000_000,
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            corpus_disk_readiness_report=disk_readiness,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "fail"
    assert corpus["evidence"]["corpus_disk_readiness"] == {
        "report": str(disk_readiness),
        "status": "blocked",
        "path": str(tmp_path / "tokens"),
        "target_tokens": 5_000_000_000,
        "required_bytes": 24_000_000_000,
        "free_bytes": 12_000_000_000,
        "deficit_bytes": 12_000_000_000,
    }
    assert any(
        "corpus disk readiness report must be ready" in error
        for error in corpus["evidence"]["metadata_errors"]
    )


def test_full_preflight_rejects_mismatched_r1_corpus_plan_audit(tmp_path: Path):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    audit = tmp_path / "reports" / "r1-corpus-plan-audit.json"
    audit.parent.mkdir()
    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mini_kimi_k3_r1_corpus_plan_audit",
                "status": "ready",
                "target_tokens": 5_000_000_000,
                "planned_tokens": 5_000_000_000,
                "available_tokens": 5_000_000_000,
                "deficit_tokens": 0,
                "manifest": "other-manifest.json",
                "tokens_dir": str(tmp_path / "tokens"),
                "corpus_plan": str(tmp_path / "reports" / "r1-corpus-plan.json"),
                "sources": {
                    "alpha": {
                        "required_tokens": 5_000_000_000,
                        "available_tokens": 5_000_000_000,
                        "deficit_tokens": 0,
                        "status": "ready",
                    }
                },
            },
            sort_keys=True,
        )
    )
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=audit,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "fail"
    assert any(
        "corpus plan audit report does not match requested token manifest" in error
        for error in corpus["evidence"]["metadata_errors"]
    )


def test_full_preflight_reports_metadata_errors_even_when_manifest_is_too_small(
    tmp_path: Path,
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    data = json.loads(manifest.read_text())
    data["tokenizer"]["sha256"] = "pending-kimi-k3-tokenizer-sha256"
    data["decontamination"]["sha256"] = "pending-decontamination-report-sha256"
    data["sources"]["alpha"]["provenance"]["snapshot"] = "pending-real-snapshot"
    data["sources"]["alpha"]["provenance"]["license"] = "pending-license-review"
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    corpus = gates["real_corpus_manifest"]
    assert corpus["status"] == "fail"
    assert "fewer tokens than the r1 target" in corpus["detail"]
    assert "metadata_errors" in corpus["evidence"]
    assert any("tokenizer" in error for error in corpus["evidence"]["metadata_errors"])
    assert any(
        "decontamination" in error for error in corpus["evidence"]["metadata_errors"]
    )
    assert any("provenance" in error for error in corpus["evidence"]["metadata_errors"])


def test_full_preflight_rejects_manifest_without_k3_tokenizer_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    data = json.loads(manifest.read_text())
    data.pop("tokenizer")
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)
    corpus_plan_audit, source_resolution = _write_ready_full_corpus_reports(
        tmp_path / "reports",
        manifest=manifest,
        tokens_dir=tmp_path / "tokens",
        available_tokens=4,
        required_tokens=4,
    )
    input_inspection = _write_stage1_input_inspection_report(
        tmp_path / "reports" / "stage1-input-inspection.json",
        min_total_tokens=4,
        total_tokens=4,
    )

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=corpus_plan_audit,
            stage1_source_resolution_report=source_resolution,
            stage1_input_inspection_report=input_inspection,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert "tokenizer" in gates["real_corpus_manifest"]["detail"]


def test_full_preflight_rejects_manifest_with_mismatched_tokenizer_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    tokenizer_report = tmp_path / "assets" / "kimi-k3-tokenizer.json"
    tokenizer_report.parent.mkdir()
    tokenizer_report.write_text(
        json.dumps(
            {
                "kind": "mini_kimi_k3_tokenizer_assets",
                "schema_version": 1,
                "sha256": "actual-tokenizer-sha",
                "files": {},
            },
            sort_keys=True,
        )
    )
    data = json.loads(manifest.read_text())
    data["tokenizer"]["asset_report"] = str(tokenizer_report)
    data["tokenizer"]["sha256"] = "manifest-tokenizer-sha"
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)
    corpus_plan_audit, source_resolution = _write_ready_full_corpus_reports(
        tmp_path / "reports",
        manifest=manifest,
        tokens_dir=tmp_path / "tokens",
        available_tokens=4,
        required_tokens=4,
    )
    input_inspection = _write_stage1_input_inspection_report(
        tmp_path / "reports" / "stage1-input-inspection.json",
        min_total_tokens=4,
        total_tokens=4,
    )

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=corpus_plan_audit,
            stage1_source_resolution_report=source_resolution,
            stage1_input_inspection_report=input_inspection,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert (
        "tokenizer asset report sha256 mismatch"
        in gates["real_corpus_manifest"]["detail"]
    )


def test_full_preflight_rejects_manifest_without_decontamination_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    data = json.loads(manifest.read_text())
    data["decontamination"] = {"status": "pending"}
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)
    corpus_plan_audit, source_resolution = _write_ready_full_corpus_reports(
        tmp_path / "reports",
        manifest=manifest,
        tokens_dir=tmp_path / "tokens",
        available_tokens=4,
        required_tokens=4,
    )
    input_inspection = _write_stage1_input_inspection_report(
        tmp_path / "reports" / "stage1-input-inspection.json",
        min_total_tokens=4,
        total_tokens=4,
    )

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=corpus_plan_audit,
            stage1_source_resolution_report=source_resolution,
            stage1_input_inspection_report=input_inspection,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert "decontamination" in gates["real_corpus_manifest"]["detail"]


def test_full_preflight_rejects_manifest_with_mismatched_decontamination_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    decontamination_report = tmp_path / "reports" / "decontamination.json"
    decontamination_report.parent.mkdir()
    decontamination_report.write_text('{"status":"completed"}\n')
    data = json.loads(manifest.read_text())
    data["decontamination"]["report"] = str(decontamination_report)
    data["decontamination"]["sha256"] = "manifest-decontamination-sha"
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)
    corpus_plan_audit, source_resolution = _write_ready_full_corpus_reports(
        tmp_path / "reports",
        manifest=manifest,
        tokens_dir=tmp_path / "tokens",
        available_tokens=4,
        required_tokens=4,
    )
    input_inspection = _write_stage1_input_inspection_report(
        tmp_path / "reports" / "stage1-input-inspection.json",
        min_total_tokens=4,
        total_tokens=4,
    )

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=corpus_plan_audit,
            stage1_source_resolution_report=source_resolution,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert (
        "decontamination report sha256 mismatch"
        in gates["real_corpus_manifest"]["detail"]
    )


def test_full_preflight_rejects_local_scope_decontamination_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    decontamination_report = tmp_path / "reports" / "decontamination.json"
    decontamination_report.parent.mkdir()
    decontamination_payload = json.dumps(
        {
            "kind": "mini_kimi_k3_decontamination_report",
            "schema_version": 1,
            "status": "completed",
            "scope": "local_explicit_benchmarks",
        },
        sort_keys=True,
    ).encode()
    decontamination_report.write_bytes(decontamination_payload)
    data = json.loads(manifest.read_text())
    data["decontamination"]["report"] = str(decontamination_report)
    data["decontamination"]["sha256"] = hashlib.sha256(
        decontamination_payload
    ).hexdigest()
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert "decontamination report scope" in gates["real_corpus_manifest"]["detail"]


def test_full_preflight_accepts_launch_grade_decontamination_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    decontamination_report = tmp_path / "reports" / "decontamination.json"
    decontamination_report.parent.mkdir()
    decontamination_payload = json.dumps(
        {
            "kind": "mini_kimi_k3_decontamination_report",
            "schema_version": 1,
            "status": "completed",
            "scope": "launch_grade_benchmark_suite",
            "benchmarks_covered": ["hellaswag", "piqa", "arc_challenge"],
        },
        sort_keys=True,
    ).encode()
    decontamination_report.write_bytes(decontamination_payload)
    data = json.loads(manifest.read_text())
    data["decontamination"]["report"] = str(decontamination_report)
    data["decontamination"]["sha256"] = hashlib.sha256(
        decontamination_payload
    ).hexdigest()
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)
    corpus_plan_audit, source_resolution = _write_ready_full_corpus_reports(
        tmp_path / "reports",
        manifest=manifest,
        tokens_dir=tmp_path / "tokens",
        available_tokens=4,
        required_tokens=4,
    )
    input_inspection = _write_stage1_input_inspection_report(
        tmp_path / "reports" / "stage1-input-inspection.json",
        min_total_tokens=4,
        total_tokens=4,
    )

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=corpus_plan_audit,
            stage1_source_resolution_report=source_resolution,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "pass"
    assert gates["real_corpus_manifest"]["evidence"]["decontamination"]["scope"] == (
        "launch_grade_benchmark_suite"
    )


def test_full_preflight_rejects_local_stage1_without_decontamination_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    decontamination_report = tmp_path / "reports" / "decontamination.json"
    decontamination_report.parent.mkdir()
    decontamination_payload = json.dumps(
        {
            "kind": "mini_kimi_k3_decontamination_report",
            "schema_version": 1,
            "status": "completed",
            "scope": "launch_grade_benchmark_suite",
            "benchmarks_covered": ["hellaswag", "piqa", "arc_challenge"],
        },
        sort_keys=True,
    ).encode()
    decontamination_report.write_bytes(decontamination_payload)
    data = json.loads(manifest.read_text())
    data["decontamination"]["report"] = str(decontamination_report)
    data["decontamination"]["sha256"] = hashlib.sha256(
        decontamination_payload
    ).hexdigest()
    data["sources"]["alpha"]["local_stage1"] = {
        "report": str(tmp_path / "reports" / "alpha-provenance.json"),
        "sha256": "local-stage1-sha",
        "total_tokens": 4,
    }
    data["sources"]["alpha"]["local_builds"] = [
        {
            "source": "alpha",
            "documents": 1,
            "input_format": "jsonl-text",
            "shards": [{"name": "a.bin", "tokens": 4}],
        }
    ]
    local_stage1_payload = json.dumps(
        {
            "schema_version": 1,
            "kind": "mini_kimi_k3_source_provenance",
            "source": "alpha",
            "total_tokens": 4,
        },
        sort_keys=True,
    ).encode()
    local_stage1_report = tmp_path / "reports" / "alpha-provenance.json"
    local_stage1_report.write_bytes(local_stage1_payload)
    data["sources"]["alpha"]["local_stage1"]["sha256"] = hashlib.sha256(
        local_stage1_payload
    ).hexdigest()
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)
    corpus_plan_audit, source_resolution = _write_ready_full_corpus_reports(
        tmp_path / "reports",
        manifest=manifest,
        tokens_dir=tmp_path / "tokens",
        available_tokens=4,
        required_tokens=4,
    )
    input_inspection = _write_stage1_input_inspection_report(
        tmp_path / "reports" / "stage1-input-inspection.json",
        min_total_tokens=4,
        total_tokens=4,
    )

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=corpus_plan_audit,
            stage1_source_resolution_report=source_resolution,
            stage1_input_inspection_report=input_inspection,
        )

    corpus = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}[
        "real_corpus_manifest"
    ]
    assert corpus["status"] == "fail"
    assert "local_stage1 decontamination evidence" in corpus["detail"]


def test_full_preflight_accepts_cwd_relative_decontamination_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    _write_shard(tmp_path / "data" / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "data" / "manifest.json",
        tokens_dir=tmp_path / "data" / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    decontamination_report = tmp_path / "reports" / "decontamination.json"
    decontamination_report.parent.mkdir()
    decontamination_payload = b'{"status":"completed"}\n'
    decontamination_report.write_bytes(decontamination_payload)
    data = json.loads(manifest.read_text())
    data["decontamination"]["report"] = "reports/decontamination.json"
    data["decontamination"]["sha256"] = hashlib.sha256(
        decontamination_payload
    ).hexdigest()
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)
    corpus_plan_audit, source_resolution = _write_ready_full_corpus_reports(
        tmp_path / "reports",
        manifest=manifest,
        tokens_dir=tmp_path / "data" / "tokens",
        available_tokens=4,
        required_tokens=4,
    )

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "data" / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=corpus_plan_audit,
            stage1_source_resolution_report=source_resolution,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "pass"


def test_full_preflight_rejects_missing_import_source_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    _attach_decontamination_report(
        manifest, tmp_path / "reports" / "decontamination.json"
    )
    data = json.loads(manifest.read_text())
    data["sources"]["alpha"]["shard_metadata"]["a.bin"]["source_sidecar"] = str(
        tmp_path / "missing-sidecar.json"
    )
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)
    corpus_plan_audit, source_resolution = _write_ready_full_corpus_reports(
        tmp_path / "reports",
        manifest=manifest,
        tokens_dir=tmp_path / "tokens",
        available_tokens=4,
        required_tokens=4,
    )

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=corpus_plan_audit,
            stage1_source_resolution_report=source_resolution,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert "source_sidecar" in gates["real_corpus_manifest"]["detail"]


def test_full_preflight_rejects_stale_import_source_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    sidecar = tmp_path / "sidecars" / "a.json"
    sidecar.parent.mkdir()
    sidecar.write_text(
        json.dumps(
            {
                "source": "other-source",
                "tokens": 4,
                "dtype": "uint32",
            },
            sort_keys=True,
        )
    )
    _attach_decontamination_report(
        manifest, tmp_path / "reports" / "decontamination.json"
    )
    data = json.loads(manifest.read_text())
    data["sources"]["alpha"]["shard_metadata"]["a.bin"]["source_sidecar"] = str(sidecar)
    data["sources"]["alpha"]["shard_metadata"]["a.bin"][
        "source_sidecar_sha256"
    ] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)
    corpus_plan_audit, source_resolution = _write_ready_full_corpus_reports(
        tmp_path / "reports",
        manifest=manifest,
        tokens_dir=tmp_path / "tokens",
        available_tokens=4,
        required_tokens=4,
    )

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=corpus_plan_audit,
            stage1_source_resolution_report=source_resolution,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert "source_sidecar source mismatch" in gates["real_corpus_manifest"]["detail"]


def test_full_preflight_rejects_import_source_sidecar_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    sidecar = tmp_path / "sidecars" / "a.json"
    sidecar.parent.mkdir()
    sidecar.write_text(
        json.dumps(
            {
                "source": "alpha",
                "tokens": 4,
                "dtype": "uint32",
            },
            sort_keys=True,
        )
    )
    _attach_decontamination_report(
        manifest, tmp_path / "reports" / "decontamination.json"
    )
    data = json.loads(manifest.read_text())
    data["sources"]["alpha"]["shard_metadata"]["a.bin"]["source_sidecar"] = str(sidecar)
    data["sources"]["alpha"]["shard_metadata"]["a.bin"][
        "source_sidecar_sha256"
    ] = "stale"
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert "source_sidecar sha256 mismatch" in gates["real_corpus_manifest"]["detail"]


def test_full_preflight_rejects_import_source_sidecar_without_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    sidecar = tmp_path / "sidecars" / "a.json"
    sidecar.parent.mkdir()
    sidecar.write_text(
        json.dumps(
            {
                "source": "alpha",
                "tokens": 4,
                "dtype": "uint32",
            },
            sort_keys=True,
        )
    )
    _attach_decontamination_report(
        manifest, tmp_path / "reports" / "decontamination.json"
    )
    data = json.loads(manifest.read_text())
    data["sources"]["alpha"]["shard_metadata"]["a.bin"]["source_sidecar"] = str(sidecar)
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert "source_sidecar_sha256" in gates["real_corpus_manifest"]["detail"]


def test_full_preflight_rejects_manifest_without_source_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    data = json.loads(manifest.read_text())
    data["sources"]["alpha"].pop("dataset_id")
    data["sources"]["alpha"].pop("provenance")
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert "source alpha" in gates["real_corpus_manifest"]["detail"]


def test_full_preflight_rejects_source_provenance_report_sha_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    source_report = tmp_path / "reports" / "alpha_provenance.json"
    source_report.parent.mkdir()
    source_report.write_text('{"snapshot":"fixture"}\n')
    _attach_decontamination_report(
        manifest, tmp_path / "reports" / "decontamination.json"
    )
    data = json.loads(manifest.read_text())
    data["sources"]["alpha"]["provenance"]["report"] = str(source_report)
    data["sources"]["alpha"]["provenance"]["sha256"] = "manifest-source-sha"
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert (
        "source alpha provenance report sha256 mismatch"
        in gates["real_corpus_manifest"]["detail"]
    )


def test_full_preflight_accepts_source_provenance_report_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    source_report = tmp_path / "reports" / "alpha_provenance.json"
    source_report.parent.mkdir()
    source_payload = b'{"dataset_id":"source/alpha","snapshot":"fixture"}\n'
    source_report.write_bytes(source_payload)
    _attach_decontamination_report(
        manifest, tmp_path / "reports" / "decontamination.json"
    )
    data = json.loads(manifest.read_text())
    data["sources"]["alpha"]["provenance"] = {
        "snapshot": "pending-real-snapshot",
        "license": "pending-license-review",
        "report": str(source_report),
        "sha256": hashlib.sha256(source_payload).hexdigest(),
    }
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)
    corpus_plan_audit, source_resolution = _write_ready_full_corpus_reports(
        tmp_path / "reports",
        manifest=manifest,
        tokens_dir=tmp_path / "tokens",
        available_tokens=4,
        required_tokens=4,
    )

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=corpus_plan_audit,
            stage1_source_resolution_report=source_resolution,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "pass"


def test_full_preflight_accepts_manifest_with_corpus_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    decontamination_report = tmp_path / "reports" / "decontamination.json"
    decontamination_report.parent.mkdir()
    decontamination_payload = b'{"status":"completed"}\n'
    decontamination_report.write_bytes(decontamination_payload)
    data = json.loads(manifest.read_text())
    data["decontamination"]["report"] = str(decontamination_report)
    data["decontamination"]["sha256"] = hashlib.sha256(
        decontamination_payload
    ).hexdigest()
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)
    corpus_plan_audit, source_resolution = _write_ready_full_corpus_reports(
        tmp_path / "reports",
        manifest=manifest,
        tokens_dir=tmp_path / "tokens",
        available_tokens=4,
        required_tokens=4,
    )

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
            r1_corpus_plan_audit_report=corpus_plan_audit,
            stage1_source_resolution_report=source_resolution,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "pass"
    assert gates["real_corpus_manifest"]["evidence"]["available_tokens"] == 4
    assert (
        gates["real_corpus_manifest"]["evidence"]["tokenizer"]["vocab_size"] == 163_840
    )
    assert (
        gates["real_corpus_manifest"]["evidence"]["decontamination"]["status"]
        == "completed"
    )
    assert gates["real_corpus_manifest"]["evidence"]["source_count"] == 1


def test_full_preflight_rejects_stale_registered_shard_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest_with_shard_metadata(
        tmp_path / "manifest.json",
        tokens_dir=tmp_path / "tokens",
        sources={"alpha": ["a.bin"]},
    )
    data = json.loads(manifest.read_text())
    data["sources"]["alpha"]["shard_metadata"]["a.bin"]["sha256"] = "stale"
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert "metadata sha256" in gates["real_corpus_manifest"]["detail"]


def test_full_preflight_rejects_missing_registered_shard_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_full_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert "shard_metadata" in gates["real_corpus_manifest"]["detail"]


def test_full_preflight_rejects_registered_metadata_for_missing_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = _write_full_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    data = json.loads(manifest.read_text())
    data["sources"]["alpha"]["shard_metadata"] = {
        "a.bin": {
            "bytes": 16,
            "num_tokens": 4,
            "sha256": "missing-shard-sha",
        }
    }
    manifest.write_text(json.dumps(data, sort_keys=True))
    report = tmp_path / "preflight.json"
    _set_full_token_budget(monkeypatch, 4)

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
        )

    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert "a.bin" in gates["real_corpus_manifest"]["detail"]


def test_preflight_accepts_initialized_launch_evidence_bundle(tmp_path: Path):
    results_root = tmp_path / "results"
    report = tmp_path / "preflight.json"
    initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )

    with pytest.raises(preflight.CheckFailure):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            evidence_results_root=results_root,
            run_id="mini-k3-r1-test",
            attempt_id="attempt-001",
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["launch_evidence_bundle"]["status"] == "pass"
    assert gates["launch_evidence_bundle"]["evidence"] == {
        "manifest": str(
            results_root / "runs" / "mini-k3-r1-test" / "attempt-001" / "manifest.json"
        ),
        "run_id": "mini-k3-r1-test",
        "attempt_id": "attempt-001",
        "family": "mini_kimi_k3",
        "task": "pretraining",
        "lane": "full",
        "model_variant": "r1",
    }


def test_preflight_rejects_wrong_lane_launch_evidence_bundle(tmp_path: Path):
    results_root = tmp_path / "results"
    report = tmp_path / "preflight.json"
    initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="tiny-plumbing",
    )

    with pytest.raises(
        preflight.CheckFailure, match="Mini Kimi K3 full launch is not ready"
    ):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            evidence_results_root=results_root,
            run_id="mini-k3-r1-test",
            attempt_id="attempt-001",
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["launch_evidence_bundle"]["status"] == "fail"
    assert gates["launch_evidence_bundle"]["detail"] == (
        "run-attempt evidence bundle lane 'tiny-plumbing' does not match "
        "preflight lane 'full'"
    )


def test_preflight_reports_invalid_token_manifest(tmp_path: Path):
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["missing.bin"]})
    report = tmp_path / "preflight.json"

    with pytest.raises(
        preflight.CheckFailure, match="Mini Kimi K3 full launch is not ready"
    ):
        preflight.run_preflight(
            mode="full",
            report_path=report,
            token_manifest=manifest,
            tokens_dir=tmp_path / "tokens",
            seq_len=4,
        )

    data = json.loads(report.read_text())
    gates = {gate["name"]: gate for gate in data["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "fail"
    assert "missing.bin" in gates["real_corpus_manifest"]["detail"]


def test_preflight_cli_writes_report_and_exits_21(tmp_path: Path):
    report = tmp_path / "preflight.json"

    proc = subprocess.run(
        [
            "python",
            "-m",
            "experiments.mini_kimi_k3.preflight",
            "--mode",
            "full",
            "--report",
            str(report),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    assert "Mini Kimi K3 full launch is not ready" in proc.stdout
    assert json.loads(report.read_text())["status"] == "blocked"


def test_preflight_cli_accepts_token_manifest(tmp_path: Path):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    report = tmp_path / "preflight.json"

    proc = subprocess.run(
        [
            "python",
            "-m",
            "experiments.mini_kimi_k3.preflight",
            "--mode",
            "tiny-plumbing",
            "--report",
            str(report),
            "--token-manifest",
            str(manifest),
            "--tokens-dir",
            str(tmp_path / "tokens"),
            "--seq-len",
            "4",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "pass"
    assert gates["tiny_training_smoke"]["status"] == "missing"


def test_preflight_cli_accepts_launch_evidence_bundle_args(tmp_path: Path):
    results_root = tmp_path / "results"
    report = tmp_path / "preflight.json"
    initialize_launch_evidence_bundle(
        results_root=results_root,
        run_id="mini-k3-r1-test",
        attempt_id="attempt-001",
        mode="full",
    )

    proc = subprocess.run(
        [
            "python",
            "-m",
            "experiments.mini_kimi_k3.preflight",
            "--mode",
            "full",
            "--report",
            str(report),
            "--evidence-results-root",
            str(results_root),
            "--run-id",
            "mini-k3-r1-test",
            "--attempt-id",
            "attempt-001",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["launch_evidence_bundle"]["status"] == "pass"


def test_runner_preflight_mode_reaches_preflight_cli_inside_rootfs(tmp_path: Path):
    report = tmp_path / "runner-preflight.json"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "preflight",
            "--mode",
            "full",
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
    assert "Mini Kimi K3 full launch is not ready" in proc.stdout
    assert json.loads(report.read_text())["status"] == "blocked"


def test_runner_preflight_mode_accepts_token_manifest(tmp_path: Path):
    _write_shard(tmp_path / "tokens" / "alpha" / "a.bin", [1, 2, 3, 4])
    manifest = _write_manifest(tmp_path / "manifest.json", {"alpha": ["a.bin"]})
    report = tmp_path / "runner-preflight.json"

    proc = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "preflight",
            "--mode",
            "tiny-plumbing",
            "--report",
            str(report),
            "--token-manifest",
            str(manifest),
            "--tokens-dir",
            str(tmp_path / "tokens"),
            "--seq-len",
            "4",
        ],
        cwd=tmp_path,
        env={**os.environ, "TORCHTITAN_IN_ROOTFS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode == 21
    gates = {gate["name"]: gate for gate in json.loads(report.read_text())["gates"]}
    assert gates["real_corpus_manifest"]["status"] == "pass"
