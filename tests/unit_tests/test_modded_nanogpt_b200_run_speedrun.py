# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from experiments.modded_nanogpt_b200 import run_speedrun


def _is_fa2_setup_command(cmd: list[str]) -> bool:
    return "experiments/modded_nanogpt_b200/setup_flash_attention.sh" in cmd


def _is_preflight_command(cmd: list[str]) -> bool:
    return "experiments/modded_nanogpt_b200/run_preflight.sh" in cmd


def _make_source(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "train_gpt.py").write_text("print('fixture')\n")
    (path / "triton_kernels.py").write_text("# fixture\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=path,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _make_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "fineweb10B",
                "token_budget": "smoke",
                "files": [{"path": "fixture.bin", "bytes": 1, "sha256": "abc"}],
            }
        )
        + "\n"
    )


def _write_success_preflight_report(
    path: Path,
    *,
    lane: str,
    mode: str,
    run_id: str,
    attention_backend: str = "fa2",
) -> None:
    claim_label = (
        "B200 compatibility patchset"
        if mode == "full" and lane == "B" and attention_backend == "fa2"
        else "diagnostic"
    )
    path.write_text(
        json.dumps(
            {
                "ok": True,
                "classification": {
                    "lane": lane,
                    "mode": mode,
                    "arm": "B0" if lane == "B" else "A0",
                    "claim_label": claim_label,
                    "evidence_tier": "full-single-attempt" if mode == "full" else mode,
                    "run_id": run_id,
                    "attempt_id": f"{run_id}_attempt_001",
                    "environment_class": "torchtitan-rootfs-b200",
                    "claim_eligible": mode == "full",
                },
                "checks": [
                    {"name": "mode_policy", "ok": True},
                    {"name": "nccl_all_reduce", "ok": True},
                    {
                        "name": "data_manifest",
                        "ok": True,
                        "detail": {
                            "dataset": "fineweb10B",
                            "token_budget": "900M" if mode == "full" else "smoke",
                            "num_files": 10,
                            "total_bytes": 2000010240,
                            "verified_sha": mode == "full",
                        },
                    },
                ],
            }
        )
        + "\n"
    )


def test_lane_b_fa2_setup_runs_before_preflight(tmp_path: Path):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_fa2_setup_order"
    calls: list[list[str]] = []

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        calls.append(cmd)
        if "experiments/modded_nanogpt_b200/setup_flash_attention.sh" in cmd:
            return run_speedrun.CommandResult(returncode=0, stdout="fa2 setup ok\n")
        assert "experiments/modded_nanogpt_b200/run_preflight.sh" in cmd
        _write_success_preflight_report(
            result_dir / "preflight_report.json",
            lane="B",
            mode="diagnostic",
            run_id="lane_b_fa2_setup_order",
        )
        return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="diagnostic",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="torch",
            allow_previous_stall=True,
            skip_run=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 0
    assert calls[0] == ["experiments/modded_nanogpt_b200/setup_flash_attention.sh"]
    assert "experiments/modded_nanogpt_b200/run_preflight.sh" in calls[1]
    assert "fa2 setup ok" in (result_dir / "run.log").read_text()


def test_lane_b_fa2_setup_failure_stops_before_preflight(tmp_path: Path):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_fa2_setup_failed"
    calls: list[list[str]] = []

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        calls.append(cmd)
        assert "experiments/modded_nanogpt_b200/setup_flash_attention.sh" in cmd
        return run_speedrun.CommandResult(returncode=17, stdout="fa2 setup failed\n")

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="diagnostic",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="torch",
            allow_previous_stall=True,
            skip_run=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 17
    assert calls == [["experiments/modded_nanogpt_b200/setup_flash_attention.sh"]]
    assert "fa2 setup failed" in (result_dir / "run.log").read_text()
    blocker = json.loads((result_dir / "blocker.json").read_text())
    assert blocker == {
        "phase": "flash_attention_setup",
        "message": "Lane B FA2 setup failed with exit code 17",
    }
    exit_code_record = json.loads((result_dir / "exit_code.json").read_text())
    assert exit_code_record == {"exit_code": 17, "phase": "flash_attention_setup"}
    watcher_status = json.loads(
        (result_dir / "telemetry" / "watcher_status.json").read_text()
    )
    assert watcher_status["state"] == "not_started"
    assert watcher_status["reason"] == "flash_attention_setup_failed"
    assert not (result_dir / "preflight_report.json").exists()


def test_data_path_comes_from_repo_relative_manifest_shard_root(
    monkeypatch, tmp_path: Path
):
    source = tmp_path / "source"
    _make_source(source)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    manifest_dir = (
        repo_root / "experiments" / "modded_nanogpt_b200" / "results" / "attempt"
    )
    manifest_dir.mkdir(parents=True)
    data_parent = repo_root / "experiments" / "modded_nanogpt_b200"
    data_root = data_parent / "data"
    shard_dir = data_root / "fineweb10B"
    shard_dir.mkdir(parents=True)
    shard = shard_dir / "fineweb_train_000000.bin"
    shard.write_bytes(b"x")
    repo_relative_shard = shard.relative_to(repo_root)
    manifest = manifest_dir / "data_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "fineweb10B",
                "token_budget": "smoke",
                "files": [
                    {"path": str(repo_relative_shard), "bytes": 1, "sha256": "fixture"},
                ],
            }
        )
        + "\n"
    )
    result_dir = tmp_path / "results" / "lane_b_data_root"
    seen_data_paths: list[str] = []

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        seen_data_paths.append(kwargs["env"]["DATA_PATH"])
        if "experiments/modded_nanogpt_b200/setup_flash_attention.sh" in cmd:
            return run_speedrun.CommandResult(returncode=0, stdout="fa2 setup ok\n")
        _write_success_preflight_report(
            result_dir / "preflight_report.json",
            lane="B",
            mode="diagnostic",
            run_id="lane_b_data_root",
        )
        return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="diagnostic",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="torch",
            allow_previous_stall=True,
            skip_run=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 0
    assert seen_data_paths
    assert set(seen_data_paths) == {str(data_parent.resolve())}
    assert str(manifest_dir.resolve()) not in seen_data_paths
    env_text = (result_dir / "command.env").read_text()
    assert f"DATA_PATH={data_parent.resolve()}" in env_text


def test_data_path_comes_from_absolute_manifest_shard_root(tmp_path: Path):
    source = tmp_path / "source"
    _make_source(source)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    data_parent = tmp_path / "upstream_data"
    data_root = data_parent / "data"
    shard_dir = data_root / "fineweb10B"
    shard_dir.mkdir(parents=True)
    shard = shard_dir / "fineweb_train_000000.bin"
    shard.write_bytes(b"x")
    manifest = manifest_dir / "data_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "fineweb10B",
                "token_budget": "smoke",
                "files": [
                    {"path": str(shard), "bytes": 1, "sha256": "fixture"},
                ],
            }
        )
        + "\n"
    )
    result_dir = tmp_path / "results" / "lane_b_absolute_data_root"
    seen_data_paths: list[str] = []

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        seen_data_paths.append(kwargs["env"]["DATA_PATH"])
        if "experiments/modded_nanogpt_b200/setup_flash_attention.sh" in cmd:
            return run_speedrun.CommandResult(returncode=0, stdout="fa2 setup ok\n")
        _write_success_preflight_report(
            result_dir / "preflight_report.json",
            lane="B",
            mode="diagnostic",
            run_id="lane_b_absolute_data_root",
        )
        return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="diagnostic",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="torch",
            allow_previous_stall=True,
            skip_run=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 0
    assert seen_data_paths
    assert set(seen_data_paths) == {str(data_parent.resolve())}
    env_text = (result_dir / "command.env").read_text()
    assert f"DATA_PATH={data_parent.resolve()}" in env_text


def test_source_data_path_blocker_catches_doubled_data_root(tmp_path: Path):
    source = tmp_path / "source"
    _make_source(source)
    shard_dir = source / "data" / "fineweb10B"
    shard_dir.mkdir(parents=True)
    (shard_dir / "fineweb_train_000000.bin").write_bytes(b"x")
    (shard_dir / "fineweb_val_000000.bin").write_bytes(b"y")

    blocker = run_speedrun._source_data_path_blocker(
        source,
        {"DATA_PATH": str(source / "data")},
    )

    assert blocker is not None
    assert blocker["phase"] == "data_path"
    assert "data/data/fineweb10B/fineweb_val_*.bin" in blocker["message"]
    assert "data/fineweb10B/fineweb_val_*.bin" in blocker["message"]


def test_full_launch_rejects_bad_source_visible_data_path_before_torchrun(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source"
    _make_source(source)
    shard_dir = source / "data" / "fineweb10B"
    shard_dir.mkdir(parents=True)
    train_shard = shard_dir / "fineweb_train_000000.bin"
    val_shard = shard_dir / "fineweb_val_000000.bin"
    train_shard.write_bytes(b"x")
    val_shard.write_bytes(b"y")
    subprocess.run(["git", "add", "data"], cwd=source, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add data fixture"],
        cwd=source,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "fineweb10B",
                "token_budget": "900M",
                "files": [
                    {"path": str(train_shard), "bytes": 1, "sha256": "fixture"},
                    {"path": str(val_shard), "bytes": 1, "sha256": "fixture"},
                ],
            }
        )
        + "\n"
    )
    result_dir = tmp_path / "results" / "lane_b_full_data_path_guard"
    calls: list[list[str]] = []
    original_run_env = run_speedrun._run_env

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        calls.append(cmd)
        if _is_fa2_setup_command(cmd):
            return run_speedrun.CommandResult(returncode=0, stdout="fa2 setup ok\n")
        if _is_preflight_command(cmd):
            _write_success_preflight_report(
                result_dir / "preflight_report.json",
                lane="B",
                mode="full",
                run_id="lane_b_full_data_path_guard",
            )
            return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")
        raise AssertionError(
            f"torchrun must not start with invalid source-visible data paths: {cmd}"
        )

    def fake_check_active_jobs(output=None):
        report = {
            "schema_version": 1,
            "ok": True,
            "active_job_count": 0,
            "active_jobs": [],
            "ignored_match_count": 0,
            "ignored_matches": [],
        }
        if output is not None:
            run_speedrun._write_json_atomic(output, report)
        return report

    def bad_run_env(config: run_speedrun.RunConfig) -> dict[str, str]:
        env = original_run_env(config)
        env["DATA_PATH"] = str(source / "data")
        return env

    monkeypatch.setattr(run_speedrun, "check_active_jobs", fake_check_active_jobs)
    monkeypatch.setattr(run_speedrun, "_run_env", bad_run_env)

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="full",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="triton",
            verify_sha=True,
            launch_authorization=run_speedrun.FULL_LAUNCH_AUTHORIZATION_TOKEN,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 21
    assert calls == [
        ["experiments/modded_nanogpt_b200/setup_flash_attention.sh"],
        [
            "experiments/modded_nanogpt_b200/run_preflight.sh",
            "--mode",
            "full",
            "--lane",
            "B",
            "--run-id",
            "lane_b_full_data_path_guard",
            "--attempt-id",
            "lane_b_full_data_path_guard_attempt_001",
            "--arm",
            "B0",
            "--source",
            str(source),
            "--data-manifest",
            str(manifest),
            "--attention-backend",
            "fa2",
            "--mlp-backend",
            "triton",
            "--report",
            str(result_dir / "preflight_report.json"),
            "--expected-gpus",
            "2",
            "--verify-sha",
        ],
    ]
    blocker = json.loads((result_dir / "blocker.json").read_text())
    assert blocker["phase"] == "data_path"
    assert "data/data/fineweb10B/fineweb_val_*.bin" in blocker["message"]
    assert "data/fineweb10B/fineweb_val_*.bin" in blocker["message"]
    exit_code_record = json.loads((result_dir / "exit_code.json").read_text())
    assert exit_code_record == {"exit_code": 21, "phase": "data_path"}
    readiness = json.loads((result_dir / "launch_readiness.json").read_text())
    assert readiness["ready_to_launch"] is False
    assert readiness["training_launched"] is False
    assert readiness["blocked_by"] == [blocker]


def test_full_attempt_initial_metadata_is_not_claim_eligible_before_preflight(
    tmp_path: Path,
):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_full_initial_claim"

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        if "experiments/modded_nanogpt_b200/setup_flash_attention.sh" in cmd:
            return run_speedrun.CommandResult(returncode=0, stdout="fa2 setup ok\n")
        _write_success_preflight_report(
            result_dir / "preflight_report.json",
            lane="B",
            mode="full",
            run_id="lane_b_full_initial_claim",
        )
        return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="full",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="triton",
            verify_sha=True,
            skip_run=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 0
    attempt = json.loads((result_dir / "attempt.json").read_text())
    assert attempt["classification"]["claim_eligible"] is False
    readiness = json.loads((result_dir / "launch_readiness.json").read_text())
    assert readiness["ready_to_launch"] is True


def test_skip_run_attempt_writes_metadata_preflight_and_summary(tmp_path: Path):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_diag"
    calls: list[list[str]] = []

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        calls.append(cmd)
        if _is_fa2_setup_command(cmd):
            return run_speedrun.CommandResult(returncode=0, stdout="fa2 setup ok\n")
        report = result_dir / "preflight_report.json"
        report.write_text(
            json.dumps(
                {
                    "ok": True,
                    "classification": {
                        "lane": "B",
                        "mode": "diagnostic",
                        "arm": "B0",
                        "claim_label": "diagnostic",
                        "evidence_tier": "diagnostic",
                        "run_id": "lane_b_diag",
                        "attempt_id": "lane_b_diag_attempt_001",
                        "environment_class": "torchtitan-rootfs-b200",
                        "claim_eligible": False,
                    },
                    "environment": {"torch": "2.13.0+cu132"},
                    "gpus": [{"index": i, "name": "NVIDIA B200"} for i in range(8)],
                }
            )
            + "\n"
        )
        return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="diagnostic",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="torch",
            allow_previous_stall=True,
            skip_run=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 0
    assert calls and _is_fa2_setup_command(calls[0])
    assert len(calls) > 1 and _is_preflight_command(calls[1])
    attempt = json.loads((result_dir / "attempt.json").read_text())
    assert attempt["classification"]["lane"] == "B"
    assert attempt["classification"]["mode"] == "diagnostic"
    assert attempt["command"]["skip_run"] is True
    command_argv = (result_dir / "command.argv").read_text()
    assert command_argv.startswith("experiments/modded_nanogpt_b200/run_speedrun.sh")
    assert "--lane B" in command_argv
    assert "--mode diagnostic" in command_argv
    assert f"--source {source}" in command_argv
    assert f"--data-manifest {manifest}" in command_argv
    assert f"--result-dir {result_dir}" in command_argv
    assert "--attention-backend fa2" in command_argv
    assert "--mlp-backend torch" in command_argv
    assert "--allow-previous-stall" in command_argv
    assert "--skip-run" in command_argv
    result_manifest = json.loads((result_dir / "data_manifest.json").read_text())
    assert result_manifest == {
        "schema_version": 1,
        "kind": "data_manifest_pointer",
        "path": str(manifest),
    }
    environment = json.loads((result_dir / "environment.json").read_text())
    assert environment == {
        "schema_version": 1,
        "kind": "preflight_environment",
        "environment": {"torch": "2.13.0+cu132"},
    }
    hardware = json.loads((result_dir / "hardware.json").read_text())
    assert hardware == {
        "schema_version": 1,
        "kind": "preflight_gpus",
        "gpus": [{"index": i, "name": "NVIDIA B200"} for i in range(8)],
    }
    env_text = (result_dir / "command.env").read_text()
    assert "TORCHINDUCTOR_CACHE_DIR=" in env_text
    assert "TRITON_CACHE_DIR=" in env_text
    assert (result_dir / "source_status_before.txt").exists()
    assert (result_dir / "source_status_after.txt").exists()
    assert (result_dir / "operator_notes.md").read_text().startswith("# Attempt Notes")
    telemetry = result_dir / "telemetry"
    assert (telemetry / "rootfs_environment.json").exists()
    assert (telemetry / "source_status_before.txt").exists()
    assert (telemetry / "source_status_after.txt").exists()
    dcgm_status = json.loads((telemetry / "dcgm_status.json").read_text())
    assert dcgm_status["state"] in {"available", "unavailable"}
    assert "checked_commands" in dcgm_status
    assert "not started: skip_run" in (telemetry / "process_watch.log").read_text()
    assert "not started: skip_run" in (telemetry / "disk_watch.log").read_text()
    assert "not started: skip_run" in (telemetry / "gpu_processes.log").read_text()
    watcher_status = json.loads((telemetry / "watcher_status.json").read_text())
    assert watcher_status["state"] == "not_started"
    assert watcher_status["reason"] == "skip_run"
    exit_code = json.loads((result_dir / "exit_code.json").read_text())
    assert exit_code == {"exit_code": 0, "phase": "not_launched"}
    blocker = json.loads((result_dir / "blocker.json").read_text())
    assert blocker == {
        "phase": "not_launched",
        "message": "skip-run requested; training was not launched",
    }
    summary = json.loads((result_dir / "summary.json").read_text())
    assert summary["classification"]["lane"] == "B"
    assert summary["final_metrics"]["val_loss"] is None
    assert summary["included_in_baseline_stats"] is False


def test_attempt_metadata_redacts_sensitive_environment_values(
    monkeypatch, tmp_path: Path
):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_redaction"
    original_run_env = run_speedrun._run_env

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        report = result_dir / "preflight_report.json"
        report.write_text(
            json.dumps(
                {
                    "ok": True,
                    "classification": {
                        "lane": "B",
                        "mode": "diagnostic",
                        "arm": "B0",
                        "claim_label": "diagnostic",
                        "evidence_tier": "diagnostic",
                        "run_id": "lane_b_redaction",
                        "attempt_id": "lane_b_redaction_attempt_001",
                        "environment_class": "torchtitan-rootfs-b200",
                        "claim_eligible": False,
                    },
                    "environment": {"torch": "2.13.0+cu132"},
                    "gpus": [{"index": i, "name": "NVIDIA B200"} for i in range(8)],
                }
            )
            + "\n"
        )
        return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")

    def env_with_sensitive_values(config: run_speedrun.RunConfig) -> dict[str, str]:
        env = original_run_env(config)
        env["AWS_CREDENTIALS_FILE"] = "raw-aws-credential-path"
        env["WANDB_API_KEY"] = "raw-wandb-key"
        env["SERVICE_AUTH_HEADER"] = "raw-auth-header"
        return env

    monkeypatch.setattr(run_speedrun, "_run_env", env_with_sensitive_values)

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="diagnostic",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="torch",
            allow_previous_stall=True,
            skip_run=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 0
    attempt_text = (result_dir / "attempt.json").read_text()
    env_text = (result_dir / "command.env").read_text()
    assert "raw-aws-credential-path" not in attempt_text
    assert "raw-wandb-key" not in attempt_text
    assert "raw-auth-header" not in attempt_text
    assert "raw-aws-credential-path" not in env_text
    assert "raw-wandb-key" not in env_text
    assert "raw-auth-header" not in env_text
    attempt = json.loads(attempt_text)
    assert attempt["environment"]["AWS_CREDENTIALS_FILE"] == "<REDACTED>"
    assert attempt["environment"]["WANDB_API_KEY"] == "<REDACTED>"
    assert attempt["environment"]["SERVICE_AUTH_HEADER"] == "<REDACTED>"
    assert "AWS_CREDENTIALS_FILE=<REDACTED>" in env_text
    assert "WANDB_API_KEY=<REDACTED>" in env_text
    assert "SERVICE_AUTH_HEADER=<REDACTED>" in env_text


def test_run_attempt_rejects_existing_nonempty_result_dir_before_artifacts(
    capsys, tmp_path: Path
):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_existing_attempt"
    result_dir.mkdir(parents=True)
    prior_marker = result_dir / "prior_attempt_marker.txt"
    prior_marker.write_text("previous attempt\n")

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        raise AssertionError(f"preflight should not run for reused result-dir: {cmd}")

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="diagnostic",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="torch",
            allow_previous_stall=True,
            skip_run=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 21
    assert prior_marker.read_text() == "previous attempt\n"
    captured_err = capsys.readouterr().err
    assert "attempt_reuse" in captured_err
    assert str(result_dir) in captured_err
    assert not (result_dir / "attempt.json").exists()
    assert not (result_dir / "command.env").exists()
    assert not (result_dir / "command.argv").exists()
    assert not (result_dir / "run.log").exists()
    assert sorted(path.name for path in result_dir.iterdir()) == [
        "prior_attempt_marker.txt"
    ]


def test_run_attempt_preserves_prelaunch_operator_notes(tmp_path: Path):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_with_prelaunch_notes"
    result_dir.mkdir(parents=True)
    notes = result_dir / "operator_notes.md"
    notes.write_text("# Phase 0 Resume\n\n- baseline_stats.count: 0\n")

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        report = result_dir / "preflight_report.json"
        report.write_text(
            json.dumps(
                {
                    "ok": True,
                    "classification": {
                        "lane": "B",
                        "mode": "diagnostic",
                        "arm": "B0",
                        "claim_label": "diagnostic",
                        "evidence_tier": "diagnostic",
                        "run_id": "lane_b_with_prelaunch_notes",
                        "attempt_id": "lane_b_with_prelaunch_notes_attempt_001",
                        "environment_class": "torchtitan-rootfs-b200",
                        "claim_eligible": False,
                    },
                    "environment": {"torch": "2.13.0+cu132"},
                    "gpus": [{"index": i, "name": "NVIDIA B200"} for i in range(8)],
                }
            )
            + "\n"
        )
        return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="diagnostic",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="torch",
            allow_previous_stall=True,
            skip_run=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 0
    notes_text = notes.read_text()
    assert notes_text.startswith("# Phase 0 Resume")
    assert "# Attempt Notes" in notes_text
    assert sorted(path.name for path in result_dir.iterdir()) != ["operator_notes.md"]


def test_run_attempt_accepts_prelaunch_operator_notes_and_active_job_scan(
    tmp_path: Path,
):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_with_prelaunch_scan"
    result_dir.mkdir(parents=True)
    (result_dir / "operator_notes.md").write_text("# Phase 0 Resume\n")
    (result_dir / "active_jobs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "active_job_count": 0,
                "active_jobs": [],
            }
        )
        + "\n"
    )

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        report = result_dir / "preflight_report.json"
        report.write_text(
            json.dumps(
                {
                    "ok": True,
                    "classification": {
                        "lane": "B",
                        "mode": "diagnostic",
                        "arm": "B0",
                        "claim_label": "diagnostic",
                        "evidence_tier": "diagnostic",
                        "run_id": "lane_b_with_prelaunch_scan",
                        "attempt_id": "lane_b_with_prelaunch_scan_attempt_001",
                        "environment_class": "torchtitan-rootfs-b200",
                        "claim_eligible": False,
                    },
                    "environment": {"torch": "2.13.0+cu132"},
                    "gpus": [{"index": i, "name": "NVIDIA B200"} for i in range(8)],
                }
            )
            + "\n"
        )
        return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="diagnostic",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="torch",
            allow_previous_stall=True,
            skip_run=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 0
    assert (result_dir / "active_jobs.json").exists()
    assert "# Attempt Notes" in (result_dir / "operator_notes.md").read_text()
    assert (result_dir / "attempt.json").exists()


def test_run_attempt_accepts_prelaunch_active_jobs_snapshot_but_rescans(
    monkeypatch,
    tmp_path: Path,
):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_with_prelaunch_scan_snapshot"
    result_dir.mkdir(parents=True)
    (result_dir / "operator_notes.md").write_text("# Phase 0 Resume\n")
    (result_dir / "active_jobs_prelaunch.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": False,
                "active_job_count": 1,
                "active_jobs": [{"pid": 1234, "command": "stale prelaunch evidence"}],
            }
        )
        + "\n"
    )

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        if _is_fa2_setup_command(cmd):
            return run_speedrun.CommandResult(returncode=0, stdout="fa2 setup ok\n")
        if _is_preflight_command(cmd):
            _write_success_preflight_report(
                result_dir / "preflight_report.json",
                lane="B",
                mode="full",
                run_id="lane_b_with_prelaunch_scan_snapshot",
            )
            return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")
        raise AssertionError(f"training must not start during a skip-run test: {cmd}")

    def fake_check_active_jobs(output=None):
        report = {
            "schema_version": 1,
            "ok": True,
            "active_job_count": 0,
            "active_jobs": [],
            "ignored_match_count": 0,
            "ignored_matches": [],
            "scanner": "in-harness",
        }
        if output is not None:
            run_speedrun._write_json_atomic(output, report)
        return report

    monkeypatch.setattr(run_speedrun, "check_active_jobs", fake_check_active_jobs)

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="full",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="triton",
            verify_sha=True,
            skip_run=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 0
    assert (result_dir / "attempt.json").exists()
    assert (result_dir / "operator_notes.md").read_text().startswith("# Phase 0 Resume")
    prelaunch_scan = json.loads((result_dir / "active_jobs_prelaunch.json").read_text())
    assert prelaunch_scan["active_job_count"] == 1
    active_jobs = json.loads((result_dir / "active_jobs.json").read_text())
    assert active_jobs["scanner"] == "in-harness"
    assert active_jobs["active_job_count"] == 0


def test_run_attempt_rejects_cache_directory_outside_result_dir(
    monkeypatch,
    tmp_path: Path,
):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_bad_cache"
    original_run_env = run_speedrun._run_env

    def fake_run_env(config: run_speedrun.RunConfig) -> dict[str, str]:
        env = original_run_env(config)
        env["TRITON_CACHE_DIR"] = str(tmp_path / "outside_cache")
        return env

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        raise AssertionError("preflight should not run with invalid cache dirs")

    monkeypatch.setattr(run_speedrun, "_run_env", fake_run_env)

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="diagnostic",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="torch",
            allow_previous_stall=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 21
    blocker = json.loads((result_dir / "blocker.json").read_text())
    assert blocker["phase"] == "cache_directories"
    assert "TRITON_CACHE_DIR" in blocker["message"]
    assert "outside result_dir" in blocker["message"]
    exit_code_record = json.loads((result_dir / "exit_code.json").read_text())
    assert exit_code_record == {"exit_code": 21, "phase": "cache_directories"}
    readiness = json.loads((result_dir / "launch_readiness.json").read_text())
    assert readiness["ready_to_launch"] is False
    assert readiness["blocked_by"] == [blocker]
    watcher_status = json.loads(
        (result_dir / "telemetry" / "watcher_status.json").read_text()
    )
    assert watcher_status["state"] == "not_started"
    assert watcher_status["reason"] == "cache_directories"
    assert not (result_dir / "preflight_report.json").exists()
    assert not (result_dir / "run.log").exists()


def test_lane_b_source_capture_classifies_variant_patch_files(tmp_path: Path):
    source = tmp_path / "source"
    _make_source(source)
    (source / "train_gpt.py").write_text("print('fixture with fa2 override')\n")
    (source / "triton_kernels.py").write_text("# fixture with b200 kernel patch\n")
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_patch_classification"
    result_dir.mkdir(parents=True)
    config = run_speedrun.RunConfig(
        lane="B",
        mode="diagnostic",
        source=source,
        data_manifest=manifest,
        result_dir=result_dir,
        attention_backend="fa2",
        mlp_backend="triton",
        allow_previous_stall=True,
        result_root=tmp_path / "results",
    )

    run_speedrun._capture_source(config)

    patch_classification = json.loads(
        (result_dir / "variant_patch_classification.json").read_text()
    )
    entries = {entry["path"]: entry for entry in patch_classification["files"]}
    assert entries["train_gpt.py"]["patch_class"] == "hardware-detection"
    assert (
        entries["train_gpt.py"]["first_lane_a_blocker"] == "FA3 no kernel image on B200"
    )
    assert entries["triton_kernels.py"]["patch_class"] == "kernel-compat"
    assert (
        entries["triton_kernels.py"]["first_lane_a_blocker"]
        == "Triton sm100 custom kernel compile/runtime blocker"
    )
    assert patch_classification["schema_version"] == 1
    assert patch_classification["source"] == str(source)
    assert patch_classification["diff_artifact"] == str(
        result_dir / "variant_patch.diff"
    )


def test_lane_b_variant_patch_is_preserved_before_preflight(tmp_path: Path):
    source = tmp_path / "source"
    _make_source(source)
    (source / "train_gpt.py").write_text("print('fixture with fa2 override')\n")
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_prelaunch_patch"

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        assert (result_dir / "variant_patch.diff").exists()
        assert (result_dir / "variant_patch_classification.json").exists()
        (result_dir / "preflight_report.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "classification": {
                        "lane": "B",
                        "mode": "diagnostic",
                        "arm": "B0",
                        "claim_label": "diagnostic",
                        "evidence_tier": "diagnostic",
                        "run_id": "lane_b_prelaunch_patch",
                        "attempt_id": "lane_b_prelaunch_patch_attempt_001",
                        "environment_class": "torchtitan-rootfs-b200",
                        "claim_eligible": False,
                    },
                    "failures": [
                        {"name": "attention_backend", "error": "fixture preflight stop"}
                    ],
                }
            )
            + "\n"
        )
        return run_speedrun.CommandResult(returncode=21, stdout="preflight failed\n")

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="diagnostic",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="triton",
            allow_previous_stall=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 21


def test_full_lane_b_rejects_unclassified_variant_patch_before_preflight(
    tmp_path: Path,
):
    source = tmp_path / "source"
    _make_source(source)
    (source / "notes.md").write_text("unclassified operator note\n")
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_unclassified_patch"

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        raise AssertionError(f"preflight should not run for unclassified patch: {cmd}")

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="full",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="triton",
            verify_sha=True,
            launch_authorization="launch-full-b200",
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 21
    blocker = json.loads((result_dir / "blocker.json").read_text())
    assert blocker == {
        "phase": "variant_patch_classification",
        "message": "full Lane B launch has unclassified source patches: notes.md",
    }
    exit_code_record = json.loads((result_dir / "exit_code.json").read_text())
    assert exit_code_record == {
        "exit_code": 21,
        "phase": "variant_patch_classification",
    }
    readiness = json.loads((result_dir / "launch_readiness.json").read_text())
    assert readiness["ready_to_launch"] is False
    assert readiness["training_launched"] is False
    assert readiness["blocked_by"] == [blocker]
    classification = json.loads(
        (result_dir / "variant_patch_classification.json").read_text()
    )
    assert classification["files"][0]["patch_class"] == "unclassified"
    assert classification["files"][0]["path"] == "notes.md"
    summary = json.loads((result_dir / "summary.json").read_text())
    assert summary["blocker"] == blocker
    assert summary["included_in_baseline_stats"] is False
    attempt = json.loads((result_dir / "attempt.json").read_text())
    assert attempt["command"]["launch_authorization_present"] is True
    assert "launch_authorization" not in attempt["command"]
    assert "launch-full-b200" not in (result_dir / "attempt.json").read_text()


def test_default_runner_extends_no_output_timeout_when_progress_probe_is_active(
    tmp_path: Path,
):
    log_path = tmp_path / "quiet_compile.log"
    script = tmp_path / "quiet_compile.py"
    script.write_text(
        textwrap.dedent(
            """
            import time

            print("compile started", flush=True)
            time.sleep(0.25)
            print("compile finished", flush=True)
            """
        )
    )
    probe_calls = 0

    def progress_probe() -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return probe_calls <= 3

    result = run_speedrun._default_runner(
        [sys.executable, str(script)],
        log_path=log_path,
        no_output_timeout_seconds=0.1,
        progress_probe=progress_probe,
    )

    assert result.returncode == 0
    assert probe_calls >= 1
    assert "compile finished" in log_path.read_text()
    assert "no output for 0.1 seconds" not in log_path.read_text()


def test_compile_worker_progress_probe_requires_active_cpu(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = """
          0.0 /usr/bin/python3 /usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_worker/__main__.py
          3.5 /usr/bin/python3 /usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_worker/__main__.py
        """

    def fake_run(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(run_speedrun.subprocess, "run", fake_run)

    assert run_speedrun._compile_worker_progress_active() is True


def test_compile_worker_progress_probe_rejects_idle_workers(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = """
          0.0 /usr/bin/python3 /usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_worker/__main__.py
          0.0 /usr/bin/python3 train_gpt.py
        """

    def fake_run(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(run_speedrun.subprocess, "run", fake_run)

    assert run_speedrun._compile_worker_progress_active() is False


def test_preflight_failure_stops_before_training_but_still_writes_summary(
    tmp_path: Path,
):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_a_fail"
    calls: list[list[str]] = []

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        calls.append(cmd)
        (result_dir / "preflight_report.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "classification": {
                        "lane": "A",
                        "mode": "full",
                        "arm": "A0",
                        "claim_label": "B200 upstream reproduction",
                        "evidence_tier": "full-single-attempt",
                        "run_id": "lane_a_fail",
                        "attempt_id": "lane_a_fail_attempt_001",
                        "environment_class": "torchtitan-rootfs-b200",
                        "claim_eligible": False,
                    },
                    "failures": [
                        {
                            "name": "attention_backend",
                            "error": "FA3 no kernel image is available",
                        }
                    ],
                }
            )
            + "\n"
        )
        return run_speedrun.CommandResult(returncode=21, stdout="preflight failed\n")

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="A",
            mode="full",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa3",
            mlp_backend="triton",
            verify_sha=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 21
    assert len(calls) == 1
    run_log = (result_dir / "run.log").read_text()
    assert "preflight failed" in run_log
    exit_code_record = json.loads((result_dir / "exit_code.json").read_text())
    assert exit_code_record == {"exit_code": 21, "phase": "preflight"}
    blocker = json.loads((result_dir / "blocker.json").read_text())
    assert blocker == {
        "phase": "attention_backend",
        "message": "FA3 no kernel image is available",
    }
    summary = json.loads((result_dir / "summary.json").read_text())
    assert summary["ok"] is False
    assert summary["classification"]["lane"] == "A"
    assert summary["final_metrics"]["train_time"] is None
    assert summary["blocker"] == blocker


def test_preflight_report_not_ok_stops_even_with_zero_process_exit(tmp_path: Path):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_preflight_report_not_ok"
    calls: list[list[str]] = []

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        calls.append(cmd)
        if _is_fa2_setup_command(cmd):
            return run_speedrun.CommandResult(returncode=0, stdout="fa2 setup ok\n")
        assert (
            len(calls) == 2
        ), "training or skip-run path must not continue after ok=false"
        (result_dir / "preflight_report.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "classification": {
                        "lane": "B",
                        "mode": "diagnostic",
                        "arm": "B0",
                        "claim_label": "diagnostic",
                        "evidence_tier": "diagnostic",
                        "run_id": "lane_b_preflight_report_not_ok",
                        "attempt_id": "lane_b_preflight_report_not_ok_attempt_001",
                        "environment_class": "torchtitan-rootfs-b200",
                        "claim_eligible": False,
                    },
                    "failures": [
                        {
                            "name": "data_manifest",
                            "error": "manifest size mismatch",
                        }
                    ],
                }
            )
            + "\n"
        )
        return run_speedrun.CommandResult(
            returncode=0, stdout="preflight wrapper returned zero\n"
        )

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="diagnostic",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="torch",
            allow_previous_stall=True,
            skip_run=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 21
    assert len(calls) == 2
    assert _is_fa2_setup_command(calls[0])
    assert _is_preflight_command(calls[1])
    exit_code_record = json.loads((result_dir / "exit_code.json").read_text())
    assert exit_code_record == {"exit_code": 21, "phase": "preflight"}
    blocker = json.loads((result_dir / "blocker.json").read_text())
    assert blocker == {"phase": "data_manifest", "message": "manifest size mismatch"}
    readiness = json.loads((result_dir / "launch_readiness.json").read_text())
    assert readiness["ready_to_launch"] is False
    assert readiness["training_launched"] is False
    assert readiness["preflight_ok"] is False
    assert readiness["blocked_by"] == [blocker]
    summary = json.loads((result_dir / "summary.json").read_text())
    assert summary["ok"] is False
    assert summary["blocker"] == blocker
    assert "skip-run requested" not in (result_dir / "run.log").read_text()


def test_training_attempt_starts_and_stops_telemetry(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    shard_dir = source / "data" / "fineweb10B"
    shard_dir.mkdir(parents=True)
    train_shard = shard_dir / "fineweb_train_000000.bin"
    val_shard = shard_dir / "fineweb_val_000000.bin"
    train_shard.write_bytes(b"x")
    val_shard.write_bytes(b"y")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "fineweb10B",
                "token_budget": "smoke",
                "files": [
                    {"path": str(train_shard), "bytes": 1, "sha256": "fixture"},
                    {"path": str(val_shard), "bytes": 1, "sha256": "fixture"},
                ],
            }
        )
        + "\n"
    )
    result_dir = tmp_path / "results" / "lane_b_launch"
    calls: list[list[str]] = []
    telemetry_events: list[str] = []

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        calls.append(cmd)
        if _is_fa2_setup_command(cmd):
            return run_speedrun.CommandResult(returncode=0, stdout="fa2 setup ok\n")
        if _is_preflight_command(cmd):
            (result_dir / "preflight_report.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "classification": {
                            "lane": "B",
                            "mode": "diagnostic",
                            "arm": "B0",
                            "claim_label": "diagnostic",
                            "evidence_tier": "diagnostic",
                            "run_id": "lane_b_launch",
                            "attempt_id": "lane_b_launch_attempt_001",
                            "environment_class": "torchtitan-rootfs-b200",
                            "claim_eligible": False,
                        },
                        "environment": {"torch": "2.13.0+cu132"},
                        "gpus": [{"index": i, "name": "NVIDIA B200"} for i in range(8)],
                    }
                )
                + "\n"
            )
            return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")
        assert kwargs["cwd"] == source
        assert kwargs["log_path"] == result_dir / "run.log"
        assert kwargs["no_output_timeout_seconds"] == 600
        assert kwargs["progress_probe"] is run_speedrun._compile_worker_progress_active
        (result_dir / "run.log").write_text(
            "NCCL communicator abort during all_reduce\n"
        )
        return run_speedrun.CommandResult(returncode=7, stdout="")

    class FakeProcess:
        pid = 4321
        returncode = 0

    def fake_start(config: run_speedrun.RunConfig):
        telemetry_events.append("start")
        return [
            run_speedrun.TelemetryProcess(
                name="process_watch",
                proc=FakeProcess(),
                output=config.result_dir / "telemetry" / "process_watch.log",
            )
        ]

    def fake_stop(config: run_speedrun.RunConfig, processes, reason: str):
        telemetry_events.append(f"stop:{reason}:{len(processes)}")
        (config.result_dir / "telemetry").mkdir(parents=True, exist_ok=True)
        (config.result_dir / "telemetry" / "watcher_status.json").write_text(
            json.dumps(
                {
                    "state": "stopped",
                    "reason": reason,
                    "processes": [
                        {
                            "name": item.name,
                            "pid": item.proc.pid,
                            "output": str(item.output),
                        }
                        for item in processes
                    ],
                }
            )
            + "\n"
        )

    def fake_subprocess_run(cmd, **kwargs):
        if cmd == ["ps", "-eo", "pid=,args="]:
            return subprocess.CompletedProcess(cmd, 0, stdout="")
        if cmd[:3] == ["ps", "-eo", "pid,ppid,pgid,etime,rss,pcpu,pmem,args"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="PID PPID PGID ELAPSED RSS %CPU %MEM COMMAND\n"
            )
        if cmd == ["nvidia-smi", "pmon", "-c", "1"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="# gpu pid type sm mem enc dec command\n"
            )
        raise AssertionError(f"unexpected subprocess.run command: {cmd}")

    monkeypatch.setattr(run_speedrun, "_start_telemetry", fake_start)
    monkeypatch.setattr(run_speedrun, "_stop_telemetry", fake_stop)
    monkeypatch.setattr(run_speedrun.subprocess, "run", fake_subprocess_run)

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="diagnostic",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="torch",
            allow_previous_stall=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 7
    assert calls == [
        ["experiments/modded_nanogpt_b200/setup_flash_attention.sh"],
        [
            "experiments/modded_nanogpt_b200/run_preflight.sh",
            "--mode",
            "diagnostic",
            "--lane",
            "B",
            "--run-id",
            "lane_b_launch",
            "--attempt-id",
            "lane_b_launch_attempt_001",
            "--arm",
            "B0",
            "--source",
            str(source),
            "--data-manifest",
            str(manifest),
            "--attention-backend",
            "fa2",
            "--mlp-backend",
            "torch",
            "--report",
            str(result_dir / "preflight_report.json"),
            "--expected-gpus",
            "2",
            "--allow-previous-stall",
        ],
        ["torchrun", "--standalone", "--nproc_per_node=2", "train_gpt.py"],
    ]
    assert telemetry_events == ["start", "stop:training_finished:1"]
    watcher_status = json.loads(
        (result_dir / "telemetry" / "watcher_status.json").read_text()
    )
    assert watcher_status["state"] == "stopped"
    assert watcher_status["reason"] == "training_finished"
    assert (
        (result_dir / "telemetry" / "stop_ps_tree.txt")
        .read_text()
        .startswith("PID PPID")
    )
    assert (
        "# gpu pid"
        in (result_dir / "telemetry" / "stop_nvidia_smi_processes.txt").read_text()
    )
    stop_snapshot = json.loads(
        (result_dir / "telemetry" / "stop_snapshot.json").read_text()
    )
    assert stop_snapshot["reason"] == "training_exit_nonzero"
    assert stop_snapshot["exit_code"] == 7
    assert stop_snapshot["commands"][0]["artifact"] == "stop_ps_tree.txt"
    assert stop_snapshot["commands"][1]["artifact"] == "stop_nvidia_smi_processes.txt"
    exit_code_record = json.loads((result_dir / "exit_code.json").read_text())
    assert exit_code_record == {"exit_code": 7, "phase": "training"}
    blocker = json.loads((result_dir / "blocker.json").read_text())
    assert blocker == {"phase": "training", "message": "training exited with code 7"}
    summary = json.loads((result_dir / "summary.json").read_text())
    assert summary["telemetry"]["state"] == "stopped"
    assert summary["telemetry"]["reason"] == "training_finished"
    assert summary["blocker"]["phase"] == "nccl"
    assert "communicator abort" in summary["blocker"]["message"]


def test_full_attempt_limits_setup_preflight_and_training_to_first_two_gpus(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    shard_dir = source / "data" / "fineweb10B"
    shard_dir.mkdir(parents=True)
    train_shard = shard_dir / "fineweb_train_000000.bin"
    val_shard = shard_dir / "fineweb_val_000000.bin"
    train_shard.write_bytes(b"x")
    val_shard.write_bytes(b"y")
    subprocess.run(["git", "add", "data"], cwd=source, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add data fixture"],
        cwd=source,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "fineweb10B",
                "token_budget": "900M",
                "files": [
                    {"path": str(train_shard), "bytes": 1, "sha256": "fixture"},
                    {"path": str(val_shard), "bytes": 1, "sha256": "fixture"},
                ],
            }
        )
        + "\n"
    )
    result_dir = tmp_path / "results" / "lane_b_full_gpu_mask"
    command_envs: list[dict[str, str]] = []

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        command_envs.append(dict(kwargs["env"]))
        if _is_fa2_setup_command(cmd):
            return run_speedrun.CommandResult(returncode=0, stdout="fa2 setup ok\n")
        if _is_preflight_command(cmd):
            _write_success_preflight_report(
                result_dir / "preflight_report.json",
                lane="B",
                mode="full",
                run_id="lane_b_full_gpu_mask",
            )
            return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")
        assert cmd == ["torchrun", "--standalone", "--nproc_per_node=2", "train_gpt.py"]
        assert kwargs["cwd"] == source
        return run_speedrun.CommandResult(returncode=0, stdout="training ok\n")

    class FakeProcess:
        pid = 4321
        returncode = 0

        def terminate(self) -> None:
            return None

        def wait(self, timeout=None) -> int:
            return self.returncode

        def kill(self) -> None:
            return None

    def fake_start(config: run_speedrun.RunConfig):
        return [
            run_speedrun.TelemetryProcess(
                name="process_watch",
                proc=FakeProcess(),
                output=config.result_dir / "telemetry" / "process_watch.log",
            )
        ]

    def fake_check_active_jobs(output=None):
        report = {
            "schema_version": 1,
            "ok": True,
            "active_job_count": 0,
            "active_jobs": [],
            "ignored_match_count": 0,
            "ignored_matches": [],
        }
        if output is not None:
            run_speedrun._write_json_atomic(output, report)
        return report

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setattr(run_speedrun, "_start_telemetry", fake_start)
    monkeypatch.setattr(run_speedrun, "check_active_jobs", fake_check_active_jobs)

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="full",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="triton",
            verify_sha=True,
            launch_authorization=run_speedrun.FULL_LAUNCH_AUTHORIZATION_TOKEN,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 0
    assert [env["CUDA_VISIBLE_DEVICES"] for env in command_envs] == ["0,1"] * 3
    assert [env["NVIDIA_VISIBLE_DEVICES"] for env in command_envs] == ["0,1"] * 3
    env_text = (result_dir / "command.env").read_text()
    assert "CUDA_VISIBLE_DEVICES=0,1" in env_text
    assert "NVIDIA_VISIBLE_DEVICES=0,1" in env_text
    attempt_text = (result_dir / "attempt.json").read_text()
    attempt = json.loads(attempt_text)
    assert attempt["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert attempt["environment"]["NVIDIA_VISIBLE_DEVICES"] == "0,1"


def test_full_skip_run_writes_launch_readiness_report(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_full_gate"

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        assert "torchrun" not in cmd, "training command should not run for skip-run"
        (result_dir / "preflight_report.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "classification": {
                        "lane": "B",
                        "mode": "full",
                        "arm": "B0",
                        "claim_label": "B200 compatibility patchset",
                        "evidence_tier": "full-single-attempt",
                        "run_id": "lane_b_full_gate",
                        "attempt_id": "lane_b_full_gate_attempt_001",
                        "environment_class": "torchtitan-rootfs-b200",
                        "claim_eligible": True,
                    },
                    "checks": [
                        {"name": "mode_policy", "ok": True},
                        {"name": "nccl_all_reduce", "ok": True},
                        {
                            "name": "data_manifest",
                            "ok": True,
                            "detail": {
                                "dataset": "fineweb10B",
                                "token_budget": "900M",
                                "num_files": 10,
                                "total_bytes": 2000010240,
                                "verified_sha": True,
                            },
                        },
                    ],
                }
            )
            + "\n"
        )
        return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")

    def fake_check_active_jobs(output=None):
        report = {
            "schema_version": 1,
            "ok": True,
            "active_job_count": 0,
            "active_jobs": [],
            "ignored_match_count": 0,
            "ignored_matches": [],
        }
        if output is not None:
            run_speedrun._write_json_atomic(output, report)
        return report

    monkeypatch.setattr(run_speedrun, "check_active_jobs", fake_check_active_jobs)

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="full",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="torch",
            verify_sha=True,
            skip_run=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 0
    report = json.loads((result_dir / "launch_readiness.json").read_text())
    assert report["ready_to_launch"] is True
    assert report["training_launched"] is False
    assert report["launch_authority_required"] is True
    assert report["launch_authorization_required_token"] == "launch-full-b200"
    assert report["preflight_ok"] is True
    assert report["mode"] == "full"
    assert report["lane"] == "B"
    assert report["verify_sha"] is True
    assert report["blocked_by"] == []
    assert report["full_mode_gates"]["verified_sha"] is True
    assert report["full_mode_gates"]["nccl_checked"] is True
    assert report["full_mode_gates"]["manifest_token_budget"] == "900M"
    active_jobs = json.loads((result_dir / "active_jobs.json").read_text())
    assert active_jobs["ok"] is True
    assert active_jobs["active_job_count"] == 0
    summary = json.loads((result_dir / "summary.json").read_text())
    assert summary["active_jobs"] == active_jobs
    markdown = (result_dir / "launch_readiness.md").read_text()
    assert "- ready_to_launch: True" in markdown
    assert "- training_launched: False" in markdown
    assert "- launch_authority_required: True" in markdown
    assert "- launch_authorization_required_token: launch-full-b200" in markdown
    assert "- blocked_by: none" in markdown


def test_full_launch_requires_explicit_authorization_token(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_full_no_authority"
    calls: list[list[str]] = []

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        calls.append(cmd)
        if _is_fa2_setup_command(cmd):
            return run_speedrun.CommandResult(returncode=0, stdout="fa2 setup ok\n")
        assert (
            len(calls) == 2
        ), "training command should not run without full-launch authorization"
        (result_dir / "preflight_report.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "classification": {
                        "lane": "B",
                        "mode": "full",
                        "arm": "B0",
                        "claim_label": "B200 compatibility patchset",
                        "evidence_tier": "full-single-attempt",
                        "run_id": "lane_b_full_no_authority",
                        "attempt_id": "lane_b_full_no_authority_attempt_001",
                        "environment_class": "torchtitan-rootfs-b200",
                        "claim_eligible": True,
                    },
                    "checks": [
                        {"name": "mode_policy", "ok": True},
                        {"name": "nccl_all_reduce", "ok": True},
                        {
                            "name": "data_manifest",
                            "ok": True,
                            "detail": {
                                "dataset": "fineweb10B",
                                "token_budget": "900M",
                                "num_files": 10,
                                "total_bytes": 2000010240,
                                "verified_sha": True,
                            },
                        },
                    ],
                }
            )
            + "\n"
        )
        return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")

    def fake_check_active_jobs(output=None):
        report = {
            "schema_version": 1,
            "ok": True,
            "active_job_count": 0,
            "active_jobs": [],
            "ignored_match_count": 0,
            "ignored_matches": [],
        }
        if output is not None:
            run_speedrun._write_json_atomic(output, report)
        return report

    monkeypatch.setattr(run_speedrun, "check_active_jobs", fake_check_active_jobs)

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="full",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="triton",
            verify_sha=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 21
    assert len(calls) == 2
    assert _is_fa2_setup_command(calls[0])
    assert _is_preflight_command(calls[1])
    readiness = json.loads((result_dir / "launch_readiness.json").read_text())
    assert readiness["ready_to_launch"] is True
    assert readiness["training_launched"] is False
    assert readiness["launch_authority_required"] is True
    assert readiness["launch_authorization_required_token"] == "launch-full-b200"
    active_jobs = json.loads((result_dir / "active_jobs.json").read_text())
    assert active_jobs["ok"] is True
    assert active_jobs["active_job_count"] == 0
    blocker = json.loads((result_dir / "blocker.json").read_text())
    assert blocker == {
        "phase": "launch_authority",
        "message": "full launch requires --launch-authorization=launch-full-b200",
    }
    exit_code_record = json.loads((result_dir / "exit_code.json").read_text())
    assert exit_code_record == {"exit_code": 21, "phase": "launch_authority"}
    summary = json.loads((result_dir / "summary.json").read_text())
    assert summary["blocker"] == blocker
    assert summary["active_jobs"] == active_jobs


def test_full_skip_run_blocks_when_active_jobs_are_present(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_full_skip_active_job"
    calls: list[list[str]] = []

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        calls.append(cmd)
        assert "torchrun" not in cmd, "training command should not run for skip-run"
        if _is_fa2_setup_command(cmd):
            return run_speedrun.CommandResult(returncode=0, stdout="fa2 setup ok\n")
        assert (
            len(calls) == 2
        ), "skip-run path should stop after preflight and active-job scan"
        _write_success_preflight_report(
            result_dir / "preflight_report.json",
            lane="B",
            mode="full",
            run_id="lane_b_full_skip_active_job",
        )
        return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")

    def fake_check_active_jobs(output=None):
        report = {
            "schema_version": 1,
            "ok": False,
            "active_job_count": 1,
            "active_jobs": [{"pid": 4321, "kind": "data-prep"}],
            "ignored_match_count": 0,
            "ignored_matches": [],
        }
        if output is not None:
            run_speedrun._write_json_atomic(output, report)
        return report

    monkeypatch.setattr(run_speedrun, "check_active_jobs", fake_check_active_jobs)

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="full",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="torch",
            verify_sha=True,
            skip_run=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 21
    assert len(calls) == 2
    assert _is_fa2_setup_command(calls[0])
    assert _is_preflight_command(calls[1])
    active_jobs = json.loads((result_dir / "active_jobs.json").read_text())
    assert active_jobs["active_job_count"] == 1
    blocker = json.loads((result_dir / "blocker.json").read_text())
    assert blocker == {
        "phase": "active_jobs",
        "message": "active training or data-prep jobs found: 4321 data-prep",
    }
    exit_code_record = json.loads((result_dir / "exit_code.json").read_text())
    assert exit_code_record == {"exit_code": 21, "phase": "active_jobs"}
    readiness = json.loads((result_dir / "launch_readiness.json").read_text())
    assert readiness["ready_to_launch"] is False
    assert readiness["training_launched"] is False
    assert readiness["blocked_by"] == [blocker]
    summary = json.loads((result_dir / "summary.json").read_text())
    assert summary["blocker"] == blocker
    assert summary["active_jobs"] == active_jobs


def test_full_missing_authority_blocks_on_active_jobs_first(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_full_no_authority_active_job"
    calls: list[list[str]] = []

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        calls.append(cmd)
        if _is_fa2_setup_command(cmd):
            return run_speedrun.CommandResult(returncode=0, stdout="fa2 setup ok\n")
        assert (
            len(calls) == 2
        ), "training command should not run without full-launch authorization"
        _write_success_preflight_report(
            result_dir / "preflight_report.json",
            lane="B",
            mode="full",
            run_id="lane_b_full_no_authority_active_job",
        )
        return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")

    def fake_check_active_jobs(output=None):
        report = {
            "schema_version": 1,
            "ok": False,
            "active_job_count": 1,
            "active_jobs": [{"pid": 8765, "kind": "torchrun"}],
            "ignored_match_count": 0,
            "ignored_matches": [],
        }
        if output is not None:
            run_speedrun._write_json_atomic(output, report)
        return report

    monkeypatch.setattr(run_speedrun, "check_active_jobs", fake_check_active_jobs)

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="full",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="triton",
            verify_sha=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 21
    assert len(calls) == 2
    assert _is_fa2_setup_command(calls[0])
    assert _is_preflight_command(calls[1])
    active_jobs = json.loads((result_dir / "active_jobs.json").read_text())
    assert active_jobs["active_job_count"] == 1
    blocker = json.loads((result_dir / "blocker.json").read_text())
    assert blocker == {
        "phase": "active_jobs",
        "message": "active training or data-prep jobs found: 8765 torchrun",
    }
    exit_code_record = json.loads((result_dir / "exit_code.json").read_text())
    assert exit_code_record == {"exit_code": 21, "phase": "active_jobs"}
    readiness = json.loads((result_dir / "launch_readiness.json").read_text())
    assert readiness["ready_to_launch"] is False
    assert readiness["training_launched"] is False
    assert readiness["blocked_by"] == [blocker]
    summary = json.loads((result_dir / "summary.json").read_text())
    assert summary["blocker"] == blocker
    assert summary["active_jobs"] == active_jobs


def test_full_launch_stops_when_active_job_safety_check_finds_process(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_full_active_job"
    calls: list[list[str]] = []

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        calls.append(cmd)
        if _is_fa2_setup_command(cmd):
            return run_speedrun.CommandResult(returncode=0, stdout="fa2 setup ok\n")
        assert (
            len(calls) == 2
        ), "training command should not run when another job is active"
        (result_dir / "preflight_report.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "classification": {
                        "lane": "B",
                        "mode": "full",
                        "arm": "B0",
                        "claim_label": "B200 compatibility patchset",
                        "evidence_tier": "full-single-attempt",
                        "run_id": "lane_b_full_active_job",
                        "attempt_id": "lane_b_full_active_job_attempt_001",
                        "environment_class": "torchtitan-rootfs-b200",
                        "claim_eligible": True,
                    },
                    "checks": [
                        {"name": "mode_policy", "ok": True},
                        {"name": "nccl_all_reduce", "ok": True},
                        {
                            "name": "data_manifest",
                            "ok": True,
                            "detail": {
                                "dataset": "fineweb10B",
                                "token_budget": "900M",
                                "num_files": 10,
                                "total_bytes": 2000010240,
                                "verified_sha": True,
                            },
                        },
                    ],
                }
            )
            + "\n"
        )
        return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")

    def fake_check_active_jobs(output=None):
        report = {
            "schema_version": 1,
            "ok": False,
            "active_job_count": 1,
            "active_jobs": [
                {
                    "pid": 1234,
                    "kind": "torchrun",
                    "args": "torchrun --standalone --nproc_per_node=8 train_gpt.py",
                }
            ],
            "ignored_match_count": 0,
            "ignored_matches": [],
        }
        if output is not None:
            run_speedrun._write_json_atomic(output, report)
        return report

    monkeypatch.setattr(run_speedrun, "check_active_jobs", fake_check_active_jobs)

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="full",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="triton",
            verify_sha=True,
            launch_authorization=run_speedrun.FULL_LAUNCH_AUTHORIZATION_TOKEN,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 21
    assert len(calls) == 2
    assert _is_fa2_setup_command(calls[0])
    assert _is_preflight_command(calls[1])
    active_jobs = json.loads((result_dir / "active_jobs.json").read_text())
    assert active_jobs["active_job_count"] == 1
    readiness = json.loads((result_dir / "launch_readiness.json").read_text())
    assert readiness["ready_to_launch"] is False
    assert readiness["training_launched"] is False
    assert readiness["blocked_by"] == [
        {
            "phase": "active_jobs",
            "message": "active training or data-prep jobs found: 1234 torchrun",
        }
    ]
    blocker = json.loads((result_dir / "blocker.json").read_text())
    assert blocker == {
        "phase": "active_jobs",
        "message": "active training or data-prep jobs found: 1234 torchrun",
    }
    exit_code_record = json.loads((result_dir / "exit_code.json").read_text())
    assert exit_code_record == {"exit_code": 21, "phase": "active_jobs"}


def test_full_launch_reports_active_job_scan_failure_as_blocker(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_full_active_job_scan_failed"
    calls: list[list[str]] = []

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        calls.append(cmd)
        if _is_fa2_setup_command(cmd):
            return run_speedrun.CommandResult(returncode=0, stdout="fa2 setup ok\n")
        assert (
            len(calls) == 2
        ), "training command should not run when active-job scan fails"
        (result_dir / "preflight_report.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "classification": {
                        "lane": "B",
                        "mode": "full",
                        "arm": "B0",
                        "claim_label": "B200 compatibility patchset",
                        "evidence_tier": "full-single-attempt",
                        "run_id": "lane_b_full_active_job_scan_failed",
                        "attempt_id": "lane_b_full_active_job_scan_failed_attempt_001",
                        "environment_class": "torchtitan-rootfs-b200",
                        "claim_eligible": True,
                    },
                    "checks": [
                        {"name": "mode_policy", "ok": True},
                        {"name": "nccl_all_reduce", "ok": True},
                        {
                            "name": "data_manifest",
                            "ok": True,
                            "detail": {
                                "dataset": "fineweb10B",
                                "token_budget": "900M",
                                "num_files": 10,
                                "total_bytes": 2000010240,
                                "verified_sha": True,
                            },
                        },
                    ],
                }
            )
            + "\n"
        )
        return run_speedrun.CommandResult(returncode=0, stdout="preflight ok\n")

    def fake_check_active_jobs(output=None):
        report = {
            "schema_version": 1,
            "ok": False,
            "active_job_count": 0,
            "active_jobs": [],
            "ignored_match_count": 0,
            "ignored_matches": [],
            "ps_returncode": 17,
            "scan_error": "ps exited with code 17",
        }
        if output is not None:
            run_speedrun._write_json_atomic(output, report)
        return report

    monkeypatch.setattr(run_speedrun, "check_active_jobs", fake_check_active_jobs)

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="full",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="triton",
            verify_sha=True,
            launch_authorization=run_speedrun.FULL_LAUNCH_AUTHORIZATION_TOKEN,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 21
    assert len(calls) == 2
    assert _is_fa2_setup_command(calls[0])
    assert _is_preflight_command(calls[1])
    active_jobs = json.loads((result_dir / "active_jobs.json").read_text())
    assert active_jobs["scan_error"] == "ps exited with code 17"
    blocker = json.loads((result_dir / "blocker.json").read_text())
    assert blocker == {
        "phase": "active_jobs",
        "message": "active-job scan failed: ps exited with code 17",
    }
    readiness = json.loads((result_dir / "launch_readiness.json").read_text())
    assert readiness["ready_to_launch"] is False
    assert readiness["training_launched"] is False
    assert readiness["blocked_by"] == [blocker]


def test_full_preflight_failure_does_not_invent_unchecked_manifest_blocker(
    tmp_path: Path,
):
    source = tmp_path / "source"
    _make_source(source)
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_full_mlp_blocked"

    def fake_runner(cmd: list[str], **kwargs) -> run_speedrun.CommandResult:
        if _is_fa2_setup_command(cmd):
            return run_speedrun.CommandResult(returncode=0, stdout="fa2 setup ok\n")
        (result_dir / "preflight_report.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "classification": {
                        "lane": "B",
                        "mode": "full",
                        "arm": "B0",
                        "claim_label": "B200 compatibility patchset",
                        "evidence_tier": "full-single-attempt",
                        "run_id": "lane_b_full_mlp_blocked",
                        "attempt_id": "lane_b_full_mlp_blocked_attempt_001",
                        "environment_class": "torchtitan-rootfs-b200",
                        "claim_eligible": False,
                    },
                    "checks": [
                        {"name": "mode_policy", "ok": True},
                        {"name": "nccl_all_reduce", "ok": True},
                        {
                            "name": "mlp_backend",
                            "ok": False,
                            "error": "known full-job blocker",
                        },
                    ],
                    "failures": [
                        {"name": "mlp_backend", "error": "known full-job blocker"}
                    ],
                }
            )
            + "\n"
        )
        return run_speedrun.CommandResult(returncode=21, stdout="preflight failed\n")

    exit_code = run_speedrun.run_attempt(
        run_speedrun.RunConfig(
            lane="B",
            mode="full",
            source=source,
            data_manifest=manifest,
            result_dir=result_dir,
            attention_backend="fa2",
            mlp_backend="torch",
            verify_sha=True,
            skip_run=True,
            result_root=tmp_path / "results",
        ),
        command_runner=fake_runner,
    )

    assert exit_code == 21
    report = json.loads((result_dir / "launch_readiness.json").read_text())
    assert report["ready_to_launch"] is False
    assert report["full_mode_gates"]["data_manifest_checked"] is False
    assert report["full_mode_gates"]["verified_sha"] is None
    assert report["blocked_by"] == [
        {"phase": "mlp_backend", "message": "known full-job blocker"}
    ]
    markdown = (result_dir / "launch_readiness.md").read_text()
    assert "- manifest_checked: False" in markdown
    assert "- manifest_verified_sha: None" in markdown
    assert (
        "data_manifest: full mode requires SHA-verified data manifest" not in markdown
    )


def test_default_runner_streams_stdout_to_log_path(tmp_path: Path):
    log_path = tmp_path / "run.log"

    result = run_speedrun._default_runner(
        [sys.executable, "-c", "print('streamed line')"],
        log_path=log_path,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert "streamed line" in log_path.read_text()


def test_default_runner_times_out_when_log_is_silent(tmp_path: Path):
    log_path = tmp_path / "run.log"

    result = run_speedrun._default_runner(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        log_path=log_path,
        no_output_timeout_seconds=0.1,
    )

    assert result.returncode == run_speedrun.NO_OUTPUT_TIMEOUT_EXIT_CODE
    assert "no output for 0.1 seconds; terminating command" in log_path.read_text()


def test_start_telemetry_captures_dcgm_when_command_is_available(
    tmp_path: Path, monkeypatch
):
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    result_dir = tmp_path / "results" / "lane_b_diag"
    config = run_speedrun.RunConfig(
        lane="B",
        mode="diagnostic",
        source=tmp_path / "source",
        data_manifest=manifest,
        result_dir=result_dir,
        attention_backend="fa2",
        mlp_backend="torch",
        allow_previous_stall=True,
        result_root=tmp_path / "results",
    )
    popen_calls = []

    class FakeProcess:
        def __init__(self, cmd, stdout=None, stderr=None, text=None):
            self.cmd = cmd
            self.stdout = stdout
            self.stderr = stderr
            self.text = text
            self.pid = 1234 + len(popen_calls)
            self.returncode = None
            popen_calls.append(cmd)

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    def fake_which(command: str) -> str | None:
        if command == "dcgmi":
            return "/usr/bin/dcgmi"
        return None

    monkeypatch.setattr(run_speedrun.shutil, "which", fake_which)
    monkeypatch.setattr(run_speedrun.subprocess, "Popen", FakeProcess)

    processes = run_speedrun._start_telemetry(config)
    try:
        dcgm_processes = [item for item in processes if item.name == "dcgm_dmon"]
        assert len(dcgm_processes) == 1
        assert dcgm_processes[0].output == result_dir / "telemetry" / "dcgm_dmon.log"
        assert ["dcgmi", "dmon", "-e", "100,101,150,155,203", "-d", "5"] in popen_calls
        watcher_status = json.loads(
            (result_dir / "telemetry" / "watcher_status.json").read_text()
        )
        assert any(item["name"] == "dcgm_dmon" for item in watcher_status["processes"])
    finally:
        run_speedrun._stop_telemetry(config, processes, "test_cleanup")


def test_active_job_scan_filters_search_shell_but_reports_training_processes():
    ps_output = "\n".join(
        [
            " 100 bash -lc ps -eo pid,args | rg 'torchrun|train_gpt.py|cached_fineweb10B.py'",
            " 101 rg torchrun|train_gpt.py|cached_fineweb10B.py",
            " 202 torchrun --standalone --nproc_per_node=8 train_gpt.py",
            " 303 python data/cached_fineweb10B.py 9",
            "",
        ]
    )

    report = run_speedrun._active_job_report(ps_output)

    assert report["ok"] is False
    assert report["active_job_count"] == 2
    assert [item["pid"] for item in report["active_jobs"]] == [202, 303]
    assert report["active_jobs"][0]["kind"] == "torchrun"
    assert report["active_jobs"][1]["kind"] == "cached_fineweb10B.py"
    assert report["ignored_match_count"] == 2


def test_active_job_scan_is_ok_when_only_search_process_matches():
    ps_output = " 100 bash -lc pgrep -af 'torchrun|train_gpt.py|cached_fineweb10B.py'\n"

    report = run_speedrun._active_job_report(ps_output)

    assert report["ok"] is True
    assert report["active_job_count"] == 0
    assert report["active_jobs"] == []
    assert report["ignored_match_count"] == 1


def test_active_job_scan_ignores_substring_lookalike_commands():
    ps_output = "\n".join(
        [
            " 100 python tools/nottrain_gpt.py",
            " 101 python data/precached_fineweb10B.py",
            " 102 /tmp/mytorchrun_helper --dry-run",
            " 202 python data/cached_fineweb10B.py 9",
            " 303 /usr/local/bin/torchrun --standalone train_gpt.py",
            "",
        ]
    )

    report = run_speedrun._active_job_report(ps_output)

    assert report["ok"] is False
    assert report["active_job_count"] == 2
    assert [item["pid"] for item in report["active_jobs"]] == [202, 303]
    assert [item["kind"] for item in report["active_jobs"]] == [
        "cached_fineweb10B.py",
        "torchrun",
    ]


def test_active_job_scan_fails_closed_when_ps_fails(monkeypatch, tmp_path: Path):
    def fake_run(cmd: list[str], **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            17,
            stdout="ps: failed to read process table\n",
        )

    monkeypatch.setattr(run_speedrun.subprocess, "run", fake_run)
    output = tmp_path / "active_jobs.json"

    report = run_speedrun.check_active_jobs(output)

    assert report["ok"] is False
    assert report["ps_returncode"] == 17
    assert report["active_job_count"] == 0
    assert report["scan_error"] == "ps exited with code 17"
    persisted = json.loads(output.read_text())
    assert persisted["ok"] is False
    assert persisted["scan_error"] == "ps exited with code 17"


def test_full_mode_rejects_allow_previous_stall(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)

    try:
        run_speedrun.RunConfig(
            lane="B",
            mode="full",
            source=source,
            data_manifest=manifest,
            result_dir=tmp_path / "results" / "bad",
            attention_backend="fa2",
            mlp_backend="torch",
            allow_previous_stall=True,
        )
    except ValueError as exc:
        assert "full mode cannot use --allow-previous-stall" in str(exc)
    else:
        raise AssertionError("expected full mode validation failure")


def test_direct_script_execution_fails_closed_outside_rootfs(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    manifest = tmp_path / "data_manifest.json"
    _make_manifest(manifest)
    child_env = dict(os.environ)
    child_env.pop("TORCHTITAN_IN_ROOTFS", None)

    proc = subprocess.run(
        [
            sys.executable,
            "experiments/modded_nanogpt_b200/run_speedrun.py",
            "--lane",
            "B",
            "--mode",
            "full",
            "--source",
            str(source),
            "--data-manifest",
            str(manifest),
            "--result-dir",
            "experiments/modded_nanogpt_b200/results/direct_script_fixture",
            "--attention-backend",
            "fa2",
            "--mlp-backend",
            "torch",
            "--allow-previous-stall",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=child_env,
    )

    assert proc.returncode == 21
    assert (
        "must run through experiments/modded_nanogpt_b200/run_speedrun.sh"
        in proc.stdout
    )
    assert "TORCHTITAN_IN_ROOTFS=1 is missing" in proc.stdout
    assert "full mode cannot use --allow-previous-stall" not in proc.stdout
    assert "ModuleNotFoundError" not in proc.stdout
    assert "Traceback" not in proc.stdout
