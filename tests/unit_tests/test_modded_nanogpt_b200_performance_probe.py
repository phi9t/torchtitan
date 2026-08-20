# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from types import SimpleNamespace

from experiments.modded_nanogpt_b200 import performance_probe


def test_static_probe_writes_issue_14_diagnostic_artifacts(monkeypatch, tmp_path):
    result_dir = tmp_path / "results" / "diag_static"
    source = tmp_path / "source"
    source.mkdir()
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text("{}\n")
    args = SimpleNamespace(
        probe_name="diag_static_wrapper_preflight",
        probe_kind="static_wrapper_preflight",
        result_dir=result_dir,
        source=source,
        data_manifest=manifest,
        attention_backend="fa2",
        mlp_backend="triton",
        run_id="diag_static",
        attempt_id="diag_static_attempt_001",
        gpu_ids="0,1",
        world_size=2,
        observability_profile="tier0",
    )

    monkeypatch.setattr(performance_probe, "parse_args", lambda: args)
    monkeypatch.setattr(performance_probe.preflight, "check_rootfs", lambda: None)
    monkeypatch.setattr(
        performance_probe.preflight,
        "collect_rootfs_detail",
        lambda: {"marker": "1", "cwd": "/workspace/torchtitan"},
    )
    monkeypatch.setattr(
        performance_probe.preflight,
        "collect_environment_detail",
        lambda: {"torch": "2.13.0+cu132", "triton": "3.7.1"},
    )
    monkeypatch.setattr(
        performance_probe,
        "_source_commit",
        lambda source: "ecbb586296d3dac36fd206211f25d63bad4a6b35",
    )
    monkeypatch.setattr(performance_probe, "_artifact_bytes", lambda path: 123)

    assert performance_probe.main() == 0

    probe = json.loads((result_dir / "probe.json").read_text())
    assert probe["schema_version"] == 1
    assert probe["probe_name"] == "diag_static_wrapper_preflight"
    assert probe["probe_kind"] == "static_wrapper_preflight"
    assert probe["mode"] == "diagnostic"
    assert probe["claim_eligible"] is False
    assert probe["source_commit"] == "ecbb586296d3dac36fd206211f25d63bad4a6b35"
    assert probe["attention_backend"] == "fa2"
    assert probe["mlp_backend"] == "triton"
    assert probe["visible_gpu_ids"] == [0, 1]
    assert probe["declared_world_size"] == 2
    assert probe["artifact_bytes"] == 123
    assert "rootfs_context" in probe["phases"]
    assert "environment_capture" in probe["phases"]

    attempt = json.loads((result_dir / "attempt.json").read_text())
    assert attempt["classification"] == {
        "lane": "B",
        "mode": "diagnostic",
        "arm": "diag_static_wrapper_preflight",
        "claim_label": "diagnostic",
        "evidence_tier": "diagnostic",
        "run_id": "diag_static",
        "attempt_id": "diag_static_attempt_001",
        "environment_class": "torchtitan-rootfs-b200",
        "claim_eligible": False,
    }
    command_argv = json.loads((result_dir / "command.argv.json").read_text())
    assert command_argv["run_id"] == "diag_static"
    assert command_argv["attempt_id"] == "diag_static_attempt_001"
    assert (
        command_argv["argv"][0]
        == "experiments/modded_nanogpt_b200/run_performance_probe.sh"
    )
    assert command_argv["argv_digest"]["algorithm"] == "sha256"
    assert len(command_argv["argv_digest"]["sha256"]) == 64
    command_env = json.loads((result_dir / "command.env.json").read_text())
    assert command_env["run_id"] == "diag_static"
    assert command_env["attempt_id"] == "diag_static_attempt_001"
    assert command_env["environment"]["TORCHTITAN_IN_ROOTFS"] == "1"
    assert command_env["environment_digest"]["algorithm"] == "sha256"
    assert len(command_env["environment_digest"]["sha256"]) == 64
    assert not (result_dir / "command.argv").exists()
    assert not (result_dir / "command.env").exists()
    summary = json.loads((result_dir / "summary.json").read_text())
    assert summary["classification"]["claim_eligible"] is False
    assert summary["probe"]["status"] == "passed"


def test_import_probe_records_import_phase(monkeypatch, tmp_path):
    result_dir = tmp_path / "results" / "diag_import"
    source = tmp_path / "source"
    source.mkdir()
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text("{}\n")
    args = SimpleNamespace(
        probe_name="diag_import_construction",
        probe_kind="import_construction",
        result_dir=result_dir,
        source=source,
        data_manifest=manifest,
        attention_backend="fa2",
        mlp_backend="triton",
        run_id="diag_import",
        attempt_id="diag_import_attempt_001",
        gpu_ids="0",
        world_size=1,
        observability_profile="tier0",
    )

    monkeypatch.setattr(performance_probe, "parse_args", lambda: args)
    monkeypatch.setattr(performance_probe.preflight, "check_rootfs", lambda: None)
    monkeypatch.setattr(
        performance_probe.preflight, "collect_rootfs_detail", lambda: {}
    )
    monkeypatch.setattr(
        performance_probe.preflight, "collect_environment_detail", lambda: {}
    )
    monkeypatch.setattr(performance_probe, "_source_commit", lambda source: "abc123")
    monkeypatch.setattr(performance_probe, "_artifact_bytes", lambda path: 0)
    monkeypatch.setattr(
        performance_probe, "_import_torch_detail", lambda: {"torch": "ok"}
    )

    assert performance_probe.main() == 0

    probe = json.loads((result_dir / "probe.json").read_text())
    assert "torch_import" in probe["phases"]
    assert probe["phases"]["torch_import"]["detail"] == {"torch": "ok"}
    assert probe["visible_gpu_ids"] == [0]
    assert probe["declared_world_size"] == 1


def test_one_gpu_microstep_probe_records_microstep_phases(monkeypatch, tmp_path):
    result_dir = tmp_path / "results" / "diag_one_gpu"
    source = tmp_path / "source"
    source.mkdir()
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text("{}\n")
    args = SimpleNamespace(
        probe_name="diag_microstep_1gpu",
        probe_kind="one_gpu_microstep",
        result_dir=result_dir,
        source=source,
        data_manifest=manifest,
        attention_backend="fa2",
        mlp_backend="torch",
        run_id="diag_one_gpu",
        attempt_id="diag_one_gpu_attempt_001",
        gpu_ids="0",
        world_size=1,
        observability_profile="tier0",
    )

    monkeypatch.setattr(performance_probe, "parse_args", lambda: args)
    monkeypatch.setattr(performance_probe.preflight, "check_rootfs", lambda: None)
    monkeypatch.setattr(
        performance_probe.preflight, "collect_rootfs_detail", lambda: {}
    )
    monkeypatch.setattr(
        performance_probe.preflight, "collect_environment_detail", lambda: {}
    )
    monkeypatch.setattr(performance_probe, "_source_commit", lambda source: "abc123")
    monkeypatch.setattr(performance_probe, "_artifact_bytes", lambda path: 0)
    monkeypatch.setattr(
        performance_probe,
        "_one_gpu_microstep_detail",
        lambda source, mlp_backend: {
            "first_forward_elapsed_ns": 10,
            "first_backward_elapsed_ns": 20,
            "optimizer_step_elapsed_ns": 30,
            "steady_microstep_median_ns": 40,
            "max_memory_allocated": 512,
        },
    )

    assert performance_probe.main() == 0

    probe = json.loads((result_dir / "probe.json").read_text())
    assert probe["phases"]["one_gpu_microstep"]["detail"] == {
        "first_forward_elapsed_ns": 10,
        "first_backward_elapsed_ns": 20,
        "optimizer_step_elapsed_ns": 30,
        "steady_microstep_median_ns": 40,
        "max_memory_allocated": 512,
    }


def test_two_gpu_microstep_probe_records_nccl_and_microstep(monkeypatch, tmp_path):
    result_dir = tmp_path / "results" / "diag_two_gpu"
    source = tmp_path / "source"
    source.mkdir()
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text("{}\n")
    args = SimpleNamespace(
        probe_name="diag_microstep_2gpu",
        probe_kind="two_gpu_microstep",
        result_dir=result_dir,
        source=source,
        data_manifest=manifest,
        attention_backend="fa2",
        mlp_backend="torch",
        run_id="diag_two_gpu",
        attempt_id="diag_two_gpu_attempt_001",
        gpu_ids="0,1",
        world_size=2,
        observability_profile="tier0",
    )

    monkeypatch.setattr(performance_probe, "parse_args", lambda: args)
    monkeypatch.setattr(performance_probe.preflight, "check_rootfs", lambda: None)
    monkeypatch.setattr(
        performance_probe.preflight, "collect_rootfs_detail", lambda: {}
    )
    monkeypatch.setattr(
        performance_probe.preflight, "collect_environment_detail", lambda: {}
    )
    monkeypatch.setattr(performance_probe, "_source_commit", lambda source: "abc123")
    monkeypatch.setattr(performance_probe, "_artifact_bytes", lambda path: 0)
    monkeypatch.setattr(
        performance_probe,
        "_two_gpu_microstep_detail",
        lambda source, mlp_backend, world_size: {
            "declared_world_size": world_size,
            "nccl_all_reduce": "passed",
            "per_rank_microstep": "passed",
        },
    )

    assert performance_probe.main() == 0

    probe = json.loads((result_dir / "probe.json").read_text())
    assert probe["visible_gpu_ids"] == [0, 1]
    assert probe["declared_world_size"] == 2
    assert probe["phases"]["two_gpu_microstep"]["detail"] == {
        "declared_world_size": 2,
        "nccl_all_reduce": "passed",
        "per_rank_microstep": "passed",
    }


def test_watchdog_probe_records_forward_progress_classification(monkeypatch, tmp_path):
    result_dir = tmp_path / "results" / "diag_watchdog"
    source = tmp_path / "source"
    source.mkdir()
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text("{}\n")
    args = SimpleNamespace(
        probe_name="diag_watchdog_heartbeat",
        probe_kind="watchdog_heartbeat",
        result_dir=result_dir,
        source=source,
        data_manifest=manifest,
        attention_backend="fa2",
        mlp_backend="triton",
        run_id="diag_watchdog",
        attempt_id="diag_watchdog_attempt_001",
        gpu_ids="0",
        world_size=1,
        observability_profile="tier0",
    )

    monkeypatch.setattr(performance_probe, "parse_args", lambda: args)
    monkeypatch.setattr(performance_probe.preflight, "check_rootfs", lambda: None)
    monkeypatch.setattr(
        performance_probe.preflight, "collect_rootfs_detail", lambda: {}
    )
    monkeypatch.setattr(
        performance_probe.preflight, "collect_environment_detail", lambda: {}
    )
    monkeypatch.setattr(performance_probe, "_source_commit", lambda source: "abc123")
    monkeypatch.setattr(performance_probe, "_artifact_bytes", lambda path: 0)
    monkeypatch.setattr(
        performance_probe,
        "_watchdog_heartbeat_detail",
        lambda: {
            "heartbeats": ["before_compile", "during_warmup"],
            "classification": "forward_progress",
            "recommendation": "raise_watchdog_with_measured_compile_evidence",
        },
    )

    assert performance_probe.main() == 0

    probe = json.loads((result_dir / "probe.json").read_text())
    detail = probe["phases"]["watchdog_heartbeat"]["detail"]
    assert detail["classification"] == "forward_progress"
    assert detail["recommendation"] == "raise_watchdog_with_measured_compile_evidence"


def test_observability_overhead_probe_compares_profiles(monkeypatch, tmp_path):
    result_dir = tmp_path / "results" / "diag_observability"
    source = tmp_path / "source"
    source.mkdir()
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text("{}\n")
    args = SimpleNamespace(
        probe_name="diag_observability_overhead",
        probe_kind="observability_overhead",
        result_dir=result_dir,
        source=source,
        data_manifest=manifest,
        attention_backend="fa2",
        mlp_backend="triton",
        run_id="diag_observability",
        attempt_id="diag_observability_attempt_001",
        gpu_ids="0",
        world_size=1,
        observability_profile="eroica",
    )

    monkeypatch.setattr(performance_probe, "parse_args", lambda: args)
    monkeypatch.setattr(performance_probe.preflight, "check_rootfs", lambda: None)
    monkeypatch.setattr(
        performance_probe.preflight, "collect_rootfs_detail", lambda: {}
    )
    monkeypatch.setattr(
        performance_probe.preflight, "collect_environment_detail", lambda: {}
    )
    monkeypatch.setattr(performance_probe, "_source_commit", lambda source: "abc123")
    monkeypatch.setattr(performance_probe, "_artifact_bytes", lambda path: 0)
    monkeypatch.setattr(
        performance_probe,
        "_observability_overhead_detail",
        lambda profile: {
            "baseline_profile": "tier0",
            "comparison_profile": profile,
            "phase_timing_delta_ns": {"environment_capture": 100},
            "artifact_bytes_delta": 200,
            "missing_evidence": [],
        },
    )

    assert performance_probe.main() == 0

    probe = json.loads((result_dir / "probe.json").read_text())
    detail = probe["phases"]["observability_overhead"]["detail"]
    assert detail["baseline_profile"] == "tier0"
    assert detail["comparison_profile"] == "eroica"
    assert detail["artifact_bytes_delta"] == 200


def test_failed_probe_preserves_diagnostic_artifacts(monkeypatch, tmp_path):
    result_dir = tmp_path / "results" / "diag_failed"
    source = tmp_path / "source"
    source.mkdir()
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text("{}\n")
    args = SimpleNamespace(
        probe_name="diag_microstep_1gpu",
        probe_kind="one_gpu_microstep",
        result_dir=result_dir,
        source=source,
        data_manifest=manifest,
        attention_backend="fa2",
        mlp_backend="torch",
        run_id="diag_failed",
        attempt_id="diag_failed_attempt_001",
        gpu_ids="0",
        world_size=1,
        observability_profile="tier0",
    )

    monkeypatch.setattr(performance_probe, "parse_args", lambda: args)
    monkeypatch.setattr(performance_probe.preflight, "check_rootfs", lambda: None)
    monkeypatch.setattr(
        performance_probe.preflight, "collect_rootfs_detail", lambda: {}
    )
    monkeypatch.setattr(
        performance_probe.preflight, "collect_environment_detail", lambda: {}
    )
    monkeypatch.setattr(performance_probe, "_source_commit", lambda source: "abc123")
    monkeypatch.setattr(performance_probe, "_artifact_bytes", lambda path: 0)

    def fail_microstep(source, mlp_backend):
        raise RuntimeError("synthetic blocker")

    monkeypatch.setattr(performance_probe, "_one_gpu_microstep_detail", fail_microstep)

    assert performance_probe.main() == 21

    probe = json.loads((result_dir / "probe.json").read_text())
    assert probe["ok"] is False
    assert probe["classification"]["claim_eligible"] is False
    assert probe["blocker"] == {
        "phase": "one_gpu_microstep",
        "message": "RuntimeError: synthetic blocker",
    }
    assert (result_dir / "attempt.json").exists()
    summary = json.loads((result_dir / "summary.json").read_text())
    assert summary["probe"]["status"] == "failed"
    assert summary["blocker"]["phase"] == "one_gpu_microstep"
