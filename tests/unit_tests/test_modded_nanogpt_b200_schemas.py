# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from experiments.modded_nanogpt_b200.runtime import (
    schema_validation,
    validate_attempt_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "experiments" / "modded_nanogpt_b200" / "runtime" / "schemas"

REQUIRED_SCHEMAS = (
    "rootfs_manifest.schema.json",
    "runtime_env.schema.json",
    "python_env_report.schema.json",
    "tool_env_report.schema.json",
    "filesystem_report.schema.json",
    "launch_prerequisites.schema.json",
    "runtime_verification.schema.json",
    "attempt.schema.json",
    "preflight_report.schema.json",
    "launch_readiness.schema.json",
    "command_env.schema.json",
    "command_argv.schema.json",
    "nanogpt_component_manifest.schema.json",
    "optimized_kernel_report.schema.json",
    "summary.schema.json",
)


TWO_B200_GPUS = [
    {"index": 0, "name": "NVIDIA B200", "compute_capability": [10, 0]},
    {"index": 1, "name": "NVIDIA B200", "compute_capability": [10, 0]},
]

PRELIGHT_ENVIRONMENT = {
    "cuda_runtime": "13.2",
    "flash_attention": "2.8.3.post1",
    "python": "3.12.3",
    "torch": "2.13.0+cu132",
    "triton": "3.7.1",
}
UPSTREAM_COMMIT = "ecbb586296d3dac36fd206211f25d63bad4a6b35"


def test_optimized_kernel_schema_copy_matches_runtime_schema():
    top_level_schema = (
        REPO_ROOT
        / "experiments"
        / "modded_nanogpt_b200"
        / "optimized_kernel_report.schema.json"
    )
    runtime_schema = SCHEMA_DIR / "optimized_kernel_report.schema.json"

    assert json.loads(top_level_schema.read_text()) == json.loads(
        runtime_schema.read_text()
    )


def test_required_schema_files_exist_and_are_strict_objects():
    for name in REQUIRED_SCHEMAS:
        path = SCHEMA_DIR / name
        assert path.is_file(), name
        schema = json.loads(path.read_text())
        assert schema["schema_version"] == 1
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert "schema_version" in schema["required"]


def test_missing_required_fields_fail():
    with pytest.raises(schema_validation.SchemaValidationError, match="missing"):
        schema_validation.validate("runtime_verification", {"schema_version": 1})


def test_unknown_top_level_fields_fail_for_launch_gating_reports():
    payload = {
        "schema_version": 1,
        "kind": "runtime_verification",
        "run_id": "run",
        "attempt_id": "attempt",
        "ok": True,
        "training_launch_allowed": True,
        "command_env_path": "command.env.json",
        "command_env_digest": {"algorithm": "sha256", "sha256": "0" * 64},
        "blockers": [],
        "surprise": True,
    }
    with pytest.raises(schema_validation.SchemaValidationError, match="unknown"):
        schema_validation.validate("runtime_verification", payload)


def test_unknown_nested_fields_fail_for_launch_gating_reports():
    payload = {
        "schema_version": 1,
        "kind": "runtime_verification",
        "run_id": "run",
        "attempt_id": "attempt",
        "ok": True,
        "training_launch_allowed": True,
        "command_env_path": "command.env.json",
        "command_env_digest": {
            "algorithm": "sha256",
            "sha256": "0" * 64,
            "surprise": True,
        },
        "blockers": [],
    }
    with pytest.raises(schema_validation.SchemaValidationError, match="unknown"):
        schema_validation.validate("runtime_verification", payload)


def test_runtime_verification_validates_current_shape():
    payload = {
        "schema_version": 1,
        "kind": "runtime_verification",
        "run_id": "run",
        "attempt_id": "attempt",
        "ok": True,
        "training_launch_allowed": True,
        "command_env_path": "command.env.json",
        "command_env_digest": {"algorithm": "sha256", "sha256": "0" * 64},
        "blockers": [],
    }
    assert schema_validation.validate("runtime_verification", payload) == payload


def test_command_env_validates_current_shape():
    payload = {
        "schema_version": 1,
        "kind": "command_environment",
        "run_id": "run",
        "attempt_id": "attempt",
        "environment": {
            "TORCHTITAN_IN_ROOTFS": "1",
            "TORCHTITAN_ROOTFS_PROJECT": "/workspace/torchtitan",
            "TORCHTITAN_ROOTFS_NETWORK": "offline",
            "PYTHON": "/project/venvs/b200-runtime/bin/python",
        },
        "environment_digest": {"algorithm": "sha256", "sha256": "1" * 64},
    }
    assert schema_validation.validate("command_env", payload) == payload


def test_command_env_rejects_missing_rootfs_critical_environment():
    payload = {
        "schema_version": 1,
        "kind": "command_environment",
        "run_id": "run",
        "attempt_id": "attempt",
        "environment": {
            "TORCHTITAN_IN_ROOTFS": "1",
            "TORCHTITAN_ROOTFS_PROJECT": "/workspace/torchtitan",
            "PYTHON": "/project/venvs/b200-runtime/bin/python",
        },
        "environment_digest": {"algorithm": "sha256", "sha256": "1" * 64},
    }

    with pytest.raises(
        schema_validation.SchemaValidationError,
        match="missing required field TORCHTITAN_ROOTFS_NETWORK",
    ):
        schema_validation.validate("command_env", payload)


def _attempt_payload() -> dict:
    return {
        "schema_version": 1,
        "classification": {
            "lane": "B",
            "mode": "full",
            "arm": "B0",
            "claim_label": "B200 compatibility patchset",
            "evidence_tier": "launch_prerequisite",
            "run_id": "run",
            "attempt_id": "attempt",
            "environment_class": "torchtitan-rootfs-b200",
            "claim_eligible": False,
        },
        "source": "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa",
        "data_manifest": "experiments/modded_nanogpt_b200/results/manifest/data_manifest.json",
        "command": {
            "argv": ["experiments/modded_nanogpt_b200/run_speedrun.sh"],
            "training_argv": ["torchrun", "--nproc_per_node=2", "train_gpt.py"],
            "skip_run": True,
            "launch_authorization_present": False,
            "attention_backend": "fa2",
            "mlp_backend": "triton",
        },
        "gpu_topology": {
            "num_gpus": 2,
            "gpu_ids": [0, 1],
            "visible_devices": "0,1",
        },
        "environment": {
            "TORCHTITAN_IN_ROOTFS": "1",
            "TORCHTITAN_ROOTFS_PROJECT": "/workspace/torchtitan",
        },
    }


def test_attempt_validates_current_shape():
    payload = _attempt_payload()
    assert schema_validation.validate("attempt", payload) == payload


def test_attempt_rejects_unknown_command_fields():
    payload = _attempt_payload()
    payload["command"]["surprise"] = True

    with pytest.raises(schema_validation.SchemaValidationError, match="unknown"):
        schema_validation.validate("attempt", payload)


def _preflight_report_payload() -> dict:
    return {
        "schema_version": 1,
        "ok": True,
        "classification": {
            "lane": "B",
            "mode": "full",
            "run_id": "run",
            "attempt_id": "attempt",
        },
        "environment": PRELIGHT_ENVIRONMENT,
        "gpus": TWO_B200_GPUS,
        "checks": [
            {
                "name": "mode_policy",
                "ok": True,
                "detail": {"expected_gpus": 2},
            },
            {
                "name": "gpu_inventory",
                "ok": True,
                "detail": TWO_B200_GPUS,
            },
            {
                "name": "data_manifest",
                "ok": True,
                "detail": [{"path": "shard.bin", "sha256": "0" * 64}],
            },
        ],
        "failures": [],
    }


def test_preflight_report_validates_current_shape():
    payload = _preflight_report_payload()
    assert schema_validation.validate("preflight_report", payload) == payload


def test_preflight_report_rejects_unknown_top_level_fields():
    payload = _preflight_report_payload()
    payload["surprise"] = True

    with pytest.raises(schema_validation.SchemaValidationError, match="unknown"):
        schema_validation.validate("preflight_report", payload)


def test_preflight_report_rejects_unknown_check_fields():
    payload = _preflight_report_payload()
    payload["checks"][0]["surprise"] = True

    with pytest.raises(schema_validation.SchemaValidationError, match="unknown"):
        schema_validation.validate("preflight_report", payload)


def _summary_payload() -> dict:
    return {
        "schema_version": 1,
        "ok": False,
        "classification": {
            "lane": "B",
            "mode": "full",
            "arm": "B0",
            "claim_label": "B200 compatibility patchset",
            "evidence_tier": "full-single-attempt",
            "run_id": "run",
            "attempt_id": "attempt",
            "environment_class": "torchtitan-rootfs-b200",
            "claim_eligible": True,
        },
        "final_validation_reached": False,
        "included_in_baseline_stats": False,
        "launch_readiness": {
            "schema_version": 1,
            "classification": {"run_id": "run", "attempt_id": "attempt"},
            "run_id": "run",
            "attempt_id": "attempt",
            "lane": "B",
            "mode": "full",
            "arm": "B0",
            "ready_to_launch": True,
            "training_launched": False,
            "launch_authority_required": True,
            "launch_authorization_required_token": "launch-full-b200",
            "preflight_ok": True,
            "preflight_returncode": 0,
            "verify_sha": True,
            "allow_previous_stall": False,
            "skip_run": True,
            "source": "experiments/modded_nanogpt_b200/sources/source",
            "data_manifest": "experiments/modded_nanogpt_b200/results/manifest/data_manifest.json",
            "attention_backend": "fa2",
            "mlp_backend": "triton",
            "full_mode_gates": {"nccl_checked": True, "verified_sha": True},
            "blocked_by": [],
            "command_env_digest": {"algorithm": "sha256", "sha256": "2" * 64},
            "runtime_verification": {
                "path": "experiments/modded_nanogpt_b200/results/run/runtime/runtime_verification.json",
                "ok": True,
                "training_launch_allowed": True,
                "command_env_digest": {
                    "algorithm": "sha256",
                    "sha256": "2" * 64,
                },
            },
        },
        "preflight_ok": True,
        "attempt_command": {
            "argv": ["experiments/modded_nanogpt_b200/run_speedrun.sh"],
            "training_argv": ["torchrun", "--nproc_per_node=2", "train_gpt.py"],
            "skip_run": True,
            "launch_authorization_present": False,
            "attention_backend": "fa2",
            "mlp_backend": "triton",
        },
        "blocker": {
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
        "final_metrics": {
            "peak_allocated_memory": None,
            "peak_reserved_memory": None,
            "step_avg": None,
            "train_time": None,
            "val_loss": None,
        },
        "last_50_meaningful_log_lines": [
            "skip-run requested; training was not launched",
        ],
        "preflight_report": "experiments/modded_nanogpt_b200/results/run/preflight_report.json",
    }


def test_summary_validates_current_shape():
    payload = _summary_payload()
    assert schema_validation.validate("summary", payload) == payload


def test_summary_rejects_unknown_launch_readiness_fields():
    payload = _summary_payload()
    payload["launch_readiness"]["surprise"] = True

    with pytest.raises(schema_validation.SchemaValidationError, match="unknown"):
        schema_validation.validate("summary", payload)


def test_summary_requires_classification_contract_fields():
    payload = _summary_payload()
    del payload["classification"]["environment_class"]

    with pytest.raises(
        schema_validation.SchemaValidationError, match="environment_class"
    ):
        schema_validation.validate("summary", payload)


def test_python_and_tool_reports_validate_current_shape():
    python_report = {
        "schema_version": 1,
        "kind": "python_env_report",
        "ok": True,
        "venv": "/project/venvs/b200-runtime",
        "lock": "experiments/modded_nanogpt_b200/runtime/requirements.lock",
        "network_mode": "offline",
    }
    tool_report = {
        "schema_version": 1,
        "kind": "tool_env_report",
        "ok": True,
        "mise_data_dir": "/project/mise/data",
        "mise_cache_dir": "/project/mise/cache",
        "mise_config_dir": "experiments/modded_nanogpt_b200/runtime",
        "mise_install_path": "/project/mise/installs",
        "mise_shims_dir": "/project/mise/data/shims",
        "shellcheck_version": "ShellCheck - shell script analysis tool",
    }
    assert (
        schema_validation.validate("python_env_report", python_report) == python_report
    )
    assert schema_validation.validate("tool_env_report", tool_report) == tool_report


def _optimized_kernel_report_payload() -> dict:
    payload = {
        "schema_version": 1,
        "schema_name": "optimized_kernel_report",
        "schema_digest": {
            "path": "experiments/modded_nanogpt_b200/optimized_kernel_report.schema.json",
            "sha256": "a" * 64,
        },
        "generated_at_epoch": 1.0,
        "launch_eligible": True,
        "selected_tuple": {
            "attention_backend": "fa2",
            "mlp_backend": "triton",
        },
        "classification": {"run_id": "run", "attempt_id": "attempt"},
        "environment": {"torch": "2.13.0+cu132"},
        "rootfs": {"torchtitan_in_rootfs": "1"},
        "source": {"path": "experiments/modded_nanogpt_b200/sources/source"},
        "gpu_arch_evidence": [],
        "selected_backend_env": {"MODDED_NANOGPT_ATTN_BACKEND": "fa2"},
        "cache_directories": {"TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor"},
        "rows": [
            {
                "name": "attention.fa2",
                "kind": "package",
                "role": "attention_backend",
                "requested": True,
                "selected_for_launch": True,
                "required_for_selected_launch": True,
                "diagnostic_only": False,
                "support_status": "supported",
                "build_status": "not_required",
                "smoke_status": "passed",
                "launch_eligible": True,
                "blockers": [],
                "failure_class": "none",
                "detail": {"backend": "fa2"},
                "packages": {"flash-attn": "2.8.3.post1"},
            }
        ],
        "blockers": [],
    }
    payload["report_digest"] = validate_attempt_artifacts._json_report_digest(
        payload, digest_field="report_digest"
    )
    return payload


def test_optimized_kernel_report_validates_current_shape():
    payload = _optimized_kernel_report_payload()
    assert schema_validation.validate("optimized_kernel_report", payload) == payload


def test_optimized_kernel_report_requires_row_gate_fields():
    payload = _optimized_kernel_report_payload()
    del payload["rows"][0]["smoke_status"]

    with pytest.raises(schema_validation.SchemaValidationError, match="smoke_status"):
        schema_validation.validate("optimized_kernel_report", payload)


def test_optimized_kernel_report_rejects_unknown_row_fields():
    payload = _optimized_kernel_report_payload()
    payload["rows"][0]["surprise"] = True

    with pytest.raises(schema_validation.SchemaValidationError, match="unknown"):
        schema_validation.validate("optimized_kernel_report", payload)


def test_validate_and_write_rejects_invalid_payload(tmp_path: Path):
    output = tmp_path / "runtime_verification.json"
    with pytest.raises(schema_validation.SchemaValidationError):
        schema_validation.validate_and_write(
            output,
            "runtime_verification",
            {"schema_version": 1, "kind": "runtime_verification"},
        )
    assert not output.exists()


def _write_complete_attempt_bundle(tmp_path: Path) -> Path:
    result_dir = tmp_path / "attempt"
    runtime_dir = result_dir / "runtime"
    manifest_dir = (
        result_dir / "experiments" / "modded_nanogpt_b200" / "results" / "manifest"
    )
    runtime_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    (result_dir / "attempt.json").write_text(json.dumps(_attempt_payload()) + "\n")
    manifest_path = (
        "experiments/modded_nanogpt_b200/results/manifest/data_manifest.json"
    )
    (manifest_dir / "data_manifest.json").write_text(
        json.dumps(_full_data_manifest_payload()) + "\n"
    )
    (result_dir / "data_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "data_manifest_pointer",
                "path": manifest_path,
            }
        )
        + "\n"
    )
    (result_dir / "hardware.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "preflight_gpus",
                "gpus": TWO_B200_GPUS,
            }
        )
        + "\n"
    )
    (result_dir / "environment.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "preflight_environment",
                "environment": PRELIGHT_ENVIRONMENT,
            }
        )
        + "\n"
    )
    telemetry_dir = result_dir / "telemetry"
    telemetry_dir.mkdir()
    (telemetry_dir / "rootfs_environment.json").write_text(
        json.dumps(
            {
                "torchtitan_in_rootfs": "1",
                "cwd": "/workspace/torchtitan",
                "python_executable": "/usr/bin/python",
                "workspace_sentinel": "scripts/rootfs/enter_rootfs.sh",
                "workspace_sentinel_exists": True,
            }
        )
        + "\n"
    )
    command_environment = {
        "TORCHTITAN_IN_ROOTFS": "1",
        "TORCHTITAN_ROOTFS_PROJECT": "/workspace/torchtitan",
        "TORCHTITAN_ROOTFS_NETWORK": "offline",
        "PYTHON": "/project/venvs/b200-runtime/bin/python",
    }
    command_env_digest = validate_attempt_artifacts._json_digest(command_environment)
    (result_dir / "command.env.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "command_environment",
                "run_id": "run",
                "attempt_id": "attempt",
                "environment": command_environment,
                "environment_digest": command_env_digest,
            }
        )
        + "\n"
    )
    replay_argv = ["experiments/modded_nanogpt_b200/run_speedrun.sh"]
    (result_dir / "command.argv.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "command_argv",
                "run_id": "run",
                "attempt_id": "attempt",
                "argv": replay_argv,
                "training_argv": ["torchrun", "train_gpt.py"],
                "argv_digest": validate_attempt_artifacts._json_digest(replay_argv),
            }
        )
        + "\n"
    )
    (result_dir / "preflight_report.json").write_text(
        json.dumps(_preflight_report_payload()) + "\n"
    )
    summary = _summary_payload()
    summary["data_manifest"] = manifest_path
    summary["data_manifest_summary"] = {
        "schema_version": 1,
        "dataset": "fineweb10B",
        "token_budget": "900M",
        "num_files": 10,
        "total_bytes": 2000010240,
        "verified_sha": True,
        "sha256_entries": 10,
    }
    summary["preflight_data_manifest"] = {
        "dataset": "fineweb10B",
        "token_budget": "900M",
        "num_files": 10,
        "total_bytes": 2000010240,
        "verified_sha": True,
        "manifest_verified_sha": True,
    }
    summary["gpus"] = TWO_B200_GPUS
    summary["hardware_sidecar"] = {
        "schema_version": 1,
        "kind": "preflight_gpus",
        "gpus": TWO_B200_GPUS,
    }
    summary["environment"] = PRELIGHT_ENVIRONMENT
    summary["environment_sidecar"] = {
        "schema_version": 1,
        "kind": "preflight_environment",
        "environment": PRELIGHT_ENVIRONMENT,
    }
    summary["telemetry"] = {
        "rootfs": {
            "torchtitan_in_rootfs": "1",
            "cwd": "/workspace/torchtitan",
            "python_executable": "/usr/bin/python",
            "workspace_sentinel_exists": True,
        }
    }
    summary["launch_readiness"]["command_env_digest"] = command_env_digest
    summary["launch_readiness"]["runtime_verification"][
        "command_env_digest"
    ] = command_env_digest
    summary["launch_readiness"]["runtime_verification"]["path"] = str(
        runtime_dir / "runtime_verification.json"
    )
    optimized_kernel_report = _optimized_kernel_report_payload()
    summary["launch_readiness"]["optimized_kernel_report"] = {
        "path": str(runtime_dir / "optimized_kernel_report.json"),
        "launch_eligible": True,
        "report_digest": optimized_kernel_report["report_digest"],
        "schema_digest": {
            "path": "experiments/modded_nanogpt_b200/optimized_kernel_report.schema.json",
            "sha256": "a" * 64,
        },
    }
    (result_dir / "launch_readiness.json").write_text(
        json.dumps(summary["launch_readiness"]) + "\n"
    )
    (runtime_dir / "runtime_verification.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "runtime_verification",
                "run_id": "run",
                "attempt_id": "attempt",
                "ok": True,
                "training_launch_allowed": True,
                "command_env_path": "command.env.json",
                "command_env_digest": command_env_digest,
                "blockers": [],
            }
        )
        + "\n"
    )
    (runtime_dir / "optimized_kernel_report.json").write_text(
        json.dumps(optimized_kernel_report) + "\n"
    )
    (result_dir / "summary.json").write_text(json.dumps(summary) + "\n")
    return result_dir


def _full_data_manifest_payload() -> dict:
    return {
        "schema_version": 1,
        "dataset": "fineweb10B",
        "token_budget": "900M",
        "num_files": 10,
        "total_bytes": 2000010240,
        "verified_sha": True,
        "source": {
            "path": "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa",
            "commit": UPSTREAM_COMMIT,
        },
        "command": ["python", "data/cached_fineweb10B.py", "10"],
        "files": [
            {
                "path": f"shard_{index:06d}.bin",
                "bytes": 200001024,
                "sha256": f"{index:064x}",
            }
            for index in range(10)
        ],
    }


def test_validate_attempt_artifacts_accepts_complete_attempt_bundle(tmp_path: Path):
    result_dir = _write_complete_attempt_bundle(tmp_path)

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is True
    assert {sidecar["schema"] for sidecar in report["sidecars"]} == {
        "attempt",
        "command_env",
        "command_argv",
        "data_manifest_pointer",
        "referenced_data_manifest",
        "hardware_sidecar",
        "environment_sidecar",
        "rootfs_environment",
        "preflight_report",
        "launch_readiness",
        "runtime_verification",
        "summary",
        "optimized_kernel_report",
        "referenced_runtime_verification",
        "referenced_optimized_kernel_digest",
        "cross_artifact_identity",
        "cross_artifact_command_env_digest",
        "cross_artifact_data_manifest",
        "cross_artifact_data_manifest_metadata",
        "cross_artifact_gpu_inventory",
        "cross_artifact_environment",
        "cross_artifact_rootfs_environment",
        "command_env_digest_integrity",
        "command_argv_digest_integrity",
    }
    assert all(sidecar["message"] == "ok" for sidecar in report["sidecars"])


def test_validate_attempt_artifacts_cli_writes_report_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    report_path = tmp_path / "validation" / "report.json"

    exit_code = validate_attempt_artifacts.main(
        [str(result_dir), "--report", str(report_path)]
    )

    assert exit_code == 0
    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(report_path.read_text())
    assert stdout_report == file_report
    assert file_report["kind"] == "attempt_artifact_validation"
    assert file_report["ok"] is True


def test_validate_attempt_artifacts_script_path_imports_repo_package(tmp_path: Path):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    script = (
        REPO_ROOT
        / "experiments"
        / "modded_nanogpt_b200"
        / "runtime"
        / "validate_attempt_artifacts.py"
    )

    proc = subprocess.run(
        ["python", str(script), str(result_dir)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["kind"] == "attempt_artifact_validation"
    assert report["ok"] is True


def test_validate_attempt_artifacts_cli_returns_21_for_invalid_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    report_path = tmp_path / "validation" / "report.json"

    exit_code = validate_attempt_artifacts.main(
        [str(tmp_path / "missing"), "--report", str(report_path)]
    )

    assert exit_code == 21
    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(report_path.read_text())
    assert stdout_report == file_report
    assert file_report["ok"] is False
    assert any(
        sidecar["schema"] == "command_env" and sidecar["message"] == "missing"
        for sidecar in file_report["sidecars"]
    )


def test_validate_attempt_artifacts_rejects_cross_artifact_identity_mismatch(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    command_argv_path = result_dir / "command.argv.json"
    command_argv = json.loads(command_argv_path.read_text())
    command_argv["attempt_id"] = "copied_attempt"
    command_argv_path.write_text(json.dumps(command_argv) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    identity = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "cross_artifact_identity"
    )
    assert identity["ok"] is False
    assert "identity mismatch" in identity["message"]
    assert "copied_attempt" in identity["message"]


def test_validate_attempt_artifacts_rejects_command_env_digest_mismatch(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    runtime_path = result_dir / "runtime" / "runtime_verification.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["command_env_digest"] = {"algorithm": "sha256", "sha256": "3" * 64}
    runtime_path.write_text(json.dumps(runtime) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    digest = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "cross_artifact_command_env_digest"
    )
    assert digest["ok"] is False
    assert "command_env_digest mismatch" in digest["message"]
    assert "runtime_verification.command_env_digest" in digest["message"]


def test_validate_attempt_artifacts_rejects_unexpected_runtime_reference_path(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    launch_readiness_path = result_dir / "launch_readiness.json"
    launch_readiness = json.loads(launch_readiness_path.read_text())
    launch_readiness["runtime_verification"]["path"] = str(
        result_dir / "copied" / "runtime_verification.json"
    )
    launch_readiness_path.write_text(json.dumps(launch_readiness) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    runtime_ref = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "referenced_runtime_verification"
    )
    assert runtime_ref["ok"] is False
    assert "unexpected path" in runtime_ref["message"]


def test_validate_attempt_artifacts_rejects_stale_runtime_reference_status(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    launch_readiness_path = result_dir / "launch_readiness.json"
    launch_readiness = json.loads(launch_readiness_path.read_text())
    launch_readiness["runtime_verification"]["training_launch_allowed"] = False
    launch_readiness_path.write_text(json.dumps(launch_readiness) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    runtime_ref = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "referenced_runtime_verification"
    )
    assert runtime_ref["ok"] is False
    assert "training_launch_allowed=False expected True" in runtime_ref["message"]


def test_validate_attempt_artifacts_rejects_optimized_kernel_digest_mismatch(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    launch_readiness_path = result_dir / "launch_readiness.json"
    launch_readiness = json.loads(launch_readiness_path.read_text())
    launch_readiness["optimized_kernel_report"]["report_digest"] = {"sha256": "4" * 64}
    launch_readiness_path.write_text(json.dumps(launch_readiness) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    kernel_digest = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "referenced_optimized_kernel_digest"
    )
    assert kernel_digest["ok"] is False
    assert "report_digest mismatch" in kernel_digest["message"]


def test_validate_attempt_artifacts_rejects_tampered_optimized_kernel_report(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    optimized_kernel_path = result_dir / "runtime" / "optimized_kernel_report.json"
    optimized_kernel = json.loads(optimized_kernel_path.read_text())
    optimized_kernel["rows"][0]["smoke_status"] = "failed"
    optimized_kernel_path.write_text(json.dumps(optimized_kernel) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    kernel_digest = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "referenced_optimized_kernel_digest"
    )
    assert kernel_digest["ok"] is False
    assert "content digest mismatch" in kernel_digest["message"]


def test_validate_attempt_artifacts_rejects_malformed_data_manifest_pointer(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    data_manifest_path = result_dir / "data_manifest.json"
    data_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "data_manifest_copy",
                "path": "experiments/modded_nanogpt_b200/results/manifest/data_manifest.json",
            }
        )
        + "\n"
    )

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    manifest = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "cross_artifact_data_manifest"
    )
    assert manifest["ok"] is False
    assert "kind must be data_manifest_pointer" in manifest["message"]


def test_validate_attempt_artifacts_rejects_missing_referenced_data_manifest(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    (
        result_dir
        / "experiments"
        / "modded_nanogpt_b200"
        / "results"
        / "manifest"
        / "data_manifest.json"
    ).unlink()

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    manifest = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "referenced_data_manifest"
    )
    assert manifest["ok"] is False
    assert manifest["message"] == "missing"


def test_validate_attempt_artifacts_rejects_smoke_shaped_data_manifest(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    manifest_path = (
        result_dir
        / "experiments"
        / "modded_nanogpt_b200"
        / "results"
        / "manifest"
        / "data_manifest.json"
    )
    manifest = _full_data_manifest_payload()
    manifest["token_budget"] = "smoke"
    manifest["num_files"] = 1
    manifest["files"] = manifest["files"][:1]
    manifest_path.write_text(json.dumps(manifest) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    manifest_sidecar = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "referenced_data_manifest"
    )
    assert manifest_sidecar["ok"] is False
    assert "token_budget mismatch" in manifest_sidecar["message"]


def test_validate_attempt_artifacts_rejects_manifest_entry_byte_mismatch(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    manifest_path = (
        result_dir
        / "experiments"
        / "modded_nanogpt_b200"
        / "results"
        / "manifest"
        / "data_manifest.json"
    )
    manifest = _full_data_manifest_payload()
    manifest["files"][0]["bytes"] = 1
    manifest_path.write_text(json.dumps(manifest) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    manifest_sidecar = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "referenced_data_manifest"
    )
    assert manifest_sidecar["ok"] is False
    assert "files bytes sum mismatch" in manifest_sidecar["message"]


def test_validate_attempt_artifacts_rejects_manifest_entry_non_hex_sha256(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    manifest_path = (
        result_dir
        / "experiments"
        / "modded_nanogpt_b200"
        / "results"
        / "manifest"
        / "data_manifest.json"
    )
    manifest = _full_data_manifest_payload()
    manifest["files"][0]["sha256"] = "z" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    manifest_sidecar = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "referenced_data_manifest"
    )
    assert manifest_sidecar["ok"] is False
    assert (
        "files[0].sha256 must be a 64-character lowercase hex string"
        in manifest_sidecar["message"]
    )


def test_validate_attempt_artifacts_rejects_manifest_source_commit_mismatch(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    manifest_path = (
        result_dir
        / "experiments"
        / "modded_nanogpt_b200"
        / "results"
        / "manifest"
        / "data_manifest.json"
    )
    manifest = _full_data_manifest_payload()
    manifest["source"]["commit"] = "bad"
    manifest_path.write_text(json.dumps(manifest) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    manifest_sidecar = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "referenced_data_manifest"
    )
    assert manifest_sidecar["ok"] is False
    assert "source.commit mismatch" in manifest_sidecar["message"]


def test_validate_attempt_artifacts_rejects_data_manifest_path_mismatch(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    launch_readiness_path = result_dir / "launch_readiness.json"
    launch_readiness = json.loads(launch_readiness_path.read_text())
    launch_readiness[
        "data_manifest"
    ] = "experiments/modded_nanogpt_b200/results/copied/data_manifest.json"
    launch_readiness_path.write_text(json.dumps(launch_readiness) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    manifest = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "cross_artifact_data_manifest"
    )
    assert manifest["ok"] is False
    assert "data_manifest mismatch" in manifest["message"]
    assert "launch_readiness.data_manifest" in manifest["message"]


def test_validate_attempt_artifacts_rejects_missing_hardware_sidecar(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    (result_dir / "hardware.json").unlink()

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    hardware = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "hardware_sidecar"
    )
    assert hardware["ok"] is False
    assert hardware["message"] == "missing"


def test_validate_attempt_artifacts_rejects_wrong_hardware_allocation(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    hardware_path = result_dir / "hardware.json"
    hardware = json.loads(hardware_path.read_text())
    hardware["gpus"] = hardware["gpus"][:1]
    hardware_path.write_text(json.dumps(hardware) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    hardware_sidecar = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "hardware_sidecar"
    )
    assert hardware_sidecar["ok"] is False
    assert "exactly two visible NVIDIA B200 GPUs" in hardware_sidecar["message"]


def test_validate_attempt_artifacts_rejects_stale_summary_gpu_inventory(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["gpus"] = summary["gpus"][:1]
    summary_path.write_text(json.dumps(summary) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    gpu_inventory = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "cross_artifact_gpu_inventory"
    )
    assert gpu_inventory["ok"] is False
    assert "summary.gpus" in gpu_inventory["message"]


def test_validate_attempt_artifacts_rejects_missing_environment_sidecar(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    (result_dir / "environment.json").unlink()

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    environment = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "environment_sidecar"
    )
    assert environment["ok"] is False
    assert environment["message"] == "missing"


def test_validate_attempt_artifacts_rejects_stale_summary_environment(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["environment"]["torch"] = "stale"
    summary_path.write_text(json.dumps(summary) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    environment = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "cross_artifact_environment"
    )
    assert environment["ok"] is False
    assert "summary.environment.torch=stale" in environment["message"]


def test_validate_attempt_artifacts_rejects_noncanonical_rootfs_environment(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    rootfs_path = result_dir / "telemetry" / "rootfs_environment.json"
    rootfs = json.loads(rootfs_path.read_text())
    rootfs["cwd"] = "/tmp/torchtitan"
    rootfs_path.write_text(json.dumps(rootfs) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    rootfs_environment = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "rootfs_environment"
    )
    assert rootfs_environment["ok"] is False
    assert "cwd mismatch" in rootfs_environment["message"]


def test_validate_attempt_artifacts_rejects_empty_rootfs_python_executable(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    rootfs_path = result_dir / "telemetry" / "rootfs_environment.json"
    rootfs = json.loads(rootfs_path.read_text())
    rootfs["python_executable"] = ""
    rootfs_path.write_text(json.dumps(rootfs) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    rootfs_environment = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "rootfs_environment"
    )
    assert rootfs_environment["ok"] is False
    assert (
        "python_executable must be a non-empty string" in rootfs_environment["message"]
    )


def test_validate_attempt_artifacts_rejects_stale_summary_rootfs_environment(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["telemetry"]["rootfs"]["workspace_sentinel_exists"] = False
    summary_path.write_text(json.dumps(summary) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    rootfs = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "cross_artifact_rootfs_environment"
    )
    assert rootfs["ok"] is False
    assert "workspace_sentinel_exists=False expected True" in rootfs["message"]


def test_validate_attempt_artifacts_rejects_stale_summary_rootfs_python(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["telemetry"]["rootfs"]["python_executable"] = "/tmp/python"
    summary_path.write_text(json.dumps(summary) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    rootfs = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "cross_artifact_rootfs_environment"
    )
    assert rootfs["ok"] is False
    assert (
        "summary.telemetry.rootfs.python_executable=/tmp/python "
        "expected /usr/bin/python"
    ) in rootfs["message"]


def test_validate_attempt_artifacts_rejects_stale_summary_manifest_metadata(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["data_manifest_summary"]["total_bytes"] = 1
    summary_path.write_text(json.dumps(summary) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    metadata = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "cross_artifact_data_manifest_metadata"
    )
    assert metadata["ok"] is False
    assert "summary.data_manifest_summary.total_bytes=1" in metadata["message"]


def test_validate_attempt_artifacts_rejects_stale_preflight_manifest_metadata(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["preflight_data_manifest"]["manifest_verified_sha"] = False
    summary_path.write_text(json.dumps(summary) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    metadata = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "cross_artifact_data_manifest_metadata"
    )
    assert metadata["ok"] is False
    assert "manifest_verified_sha=False expected True" in metadata["message"]


def test_validate_attempt_artifacts_rejects_tampered_command_environment_digest(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    command_env_path = result_dir / "command.env.json"
    command_env = json.loads(command_env_path.read_text())
    command_env["environment"]["CUDA_VISIBLE_DEVICES"] = "0,1"
    command_env_path.write_text(json.dumps(command_env) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    digest = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "command_env_digest_integrity"
    )
    assert digest["ok"] is False
    assert "environment_digest mismatch" in digest["message"]


def test_validate_attempt_artifacts_rejects_tampered_command_argv_digest(
    tmp_path: Path,
):
    result_dir = _write_complete_attempt_bundle(tmp_path)
    command_argv_path = result_dir / "command.argv.json"
    command_argv = json.loads(command_argv_path.read_text())
    command_argv["argv"].append("--skip-run")
    command_argv_path.write_text(json.dumps(command_argv) + "\n")

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    digest = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "command_argv_digest_integrity"
    )
    assert digest["ok"] is False
    assert "argv_digest mismatch" in digest["message"]


def test_validate_attempt_artifacts_fails_on_missing_launch_sidecar(tmp_path: Path):
    report = validate_attempt_artifacts.validate_attempt_dir(tmp_path / "missing")

    assert report["ok"] is False
    command_env = next(
        sidecar for sidecar in report["sidecars"] if sidecar["schema"] == "command_env"
    )
    assert command_env["message"] == "missing"


def test_validate_attempt_artifacts_fails_on_missing_referenced_kernel_report(
    tmp_path: Path,
):
    result_dir = tmp_path / "attempt"
    result_dir.mkdir()
    launch_readiness = _summary_payload()["launch_readiness"]
    launch_readiness["optimized_kernel_report"] = {
        "path": str(result_dir / "runtime" / "optimized_kernel_report.json"),
        "launch_eligible": True,
        "report_digest": {"sha256": "b" * 64},
        "schema_digest": {
            "path": "experiments/modded_nanogpt_b200/optimized_kernel_report.schema.json",
            "sha256": "a" * 64,
        },
    }
    (result_dir / "launch_readiness.json").write_text(
        json.dumps(launch_readiness) + "\n"
    )

    report = validate_attempt_artifacts.validate_attempt_dir(result_dir)

    assert report["ok"] is False
    optimized_kernel = next(
        sidecar
        for sidecar in report["sidecars"]
        if sidecar["schema"] == "optimized_kernel_report"
    )
    assert optimized_kernel["message"] == "missing"
