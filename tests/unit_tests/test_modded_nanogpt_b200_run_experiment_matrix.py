# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path

from experiments.modded_nanogpt_b200 import run_experiment_matrix


def _write_spec(path: Path, result_root: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "gpu_ladder_prerequisite",
                "description": "B200 prerequisite ladder",
                "source": "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa",
                "data_manifest": "experiments/modded_nanogpt_b200/results/full_manifest/data_manifest.json",
                "result_root": str(result_root),
                "execution_policy": {
                    "rootfs_required": True,
                    "sequential": True,
                    "active_job_policy": "fail_if_active",
                },
                "defaults": {
                    "mode": "full",
                    "experiment_kind": "prerequisite",
                    "legacy_lane": "B",
                    "attention_backend": "fa2",
                    "mlp_backend": "torch",
                    "compile_policy": "disabled_prerequisite",
                    "ce_compute_capability": "100",
                    "verify_sha": True,
                    "observability_profile": "byterobust",
                },
                "arms": [
                    {"name": "g1", "num_gpus": 1},
                    {
                        "name": "g2_pair",
                        "num_gpus": 2,
                        "gpu_ids": [2, 3],
                        "observability_profile": "mycroft",
                    },
                    {"name": "g4_pending", "num_gpus": 4},
                ],
            }
        )
        + "\n"
    )


def _write_observability_spec(path: Path, result_root: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "observability_matrix",
                "description": "Observability profile matrix",
                "source": "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa",
                "data_manifest": "experiments/modded_nanogpt_b200/results/full_manifest/data_manifest.json",
                "result_root": str(result_root),
                "execution_policy": {
                    "rootfs_required": True,
                    "sequential": True,
                    "active_job_policy": "fail_if_active",
                },
                "defaults": {
                    "mode": "full",
                    "experiment_kind": "prerequisite",
                    "legacy_lane": "B",
                    "attention_backend": "fa2",
                    "mlp_backend": "torch",
                    "compile_policy": "disabled_prerequisite",
                    "verify_sha": True,
                },
                "arms": [
                    {
                        "name": "timeline",
                        "num_gpus": 1,
                        "observability_profile": "mycroft",
                    },
                    {
                        "name": "planner",
                        "num_gpus": 1,
                        "observability_profile": "argus",
                    },
                    {
                        "name": "rsi",
                        "num_gpus": 1,
                        "observability_profile": "eroica",
                    },
                ],
            }
        )
        + "\n"
    )


def _write_active_ladder_spec(path: Path, result_root: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "gpu_ladder_prerequisite",
                "description": "B200 prerequisite ladder",
                "source": "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa",
                "data_manifest": "experiments/modded_nanogpt_b200/results/full_manifest/data_manifest.json",
                "result_root": str(result_root),
                "execution_policy": {
                    "rootfs_required": True,
                    "sequential": True,
                    "active_job_policy": "fail_if_active",
                },
                "defaults": {
                    "mode": "full",
                    "experiment_kind": "prerequisite",
                    "legacy_lane": "B",
                    "attention_backend": "fa2",
                    "mlp_backend": "triton",
                    "compile_policy": "disabled_prerequisite",
                    "verify_sha": True,
                    "observability_profile": "byterobust",
                },
                "arms": [
                    {"name": "g1_prerequisite", "num_gpus": 1},
                    {"name": "g2_prerequisite", "num_gpus": 2},
                    {"name": "g4_prerequisite", "num_gpus": 4},
                    {"name": "g8_prerequisite", "num_gpus": 8},
                ],
            }
        )
        + "\n"
    )


def test_dry_run_writes_plan_and_does_not_run_arms(tmp_path: Path, monkeypatch):
    spec_path = tmp_path / "matrix.json"
    result_root = tmp_path / "results"
    _write_spec(spec_path, result_root)
    calls = []
    monkeypatch.setattr(
        run_experiment_matrix.run_speedrun,
        "run_attempt",
        lambda *args, **kwargs: calls.append((args, kwargs)) or 0,
    )

    exit_code = run_experiment_matrix.run_matrix(spec_path, dry_run=True)

    assert exit_code == 0
    assert calls == []
    matrix_report = json.loads(
        (result_root / "gpu_ladder_prerequisite_matrix_report.json").read_text()
    )
    assert matrix_report["dry_run"] is True
    assert matrix_report["arms"][0]["status"] == "planned"
    assert matrix_report["arms"][1]["observability"]["profile"] == "mycroft"
    assert (
        json.loads((result_root / "gpu_ladder_prerequisite_plan.json").read_text())[
            "arms"
        ][1]["visible_devices"]
        == "2,3"
    )


def test_dry_run_report_marks_only_active_two_gpu_ladder_arm_claim_eligible(
    tmp_path: Path, monkeypatch
):
    spec_path = tmp_path / "matrix.json"
    result_root = tmp_path / "results"
    _write_active_ladder_spec(spec_path, result_root)
    calls = []
    monkeypatch.setattr(
        run_experiment_matrix.run_speedrun,
        "run_attempt",
        lambda *args, **kwargs: calls.append((args, kwargs)) or 0,
    )

    exit_code = run_experiment_matrix.run_matrix(spec_path, dry_run=True)

    assert exit_code == 0
    assert calls == []
    plan = json.loads((result_root / "gpu_ladder_prerequisite_plan.json").read_text())
    report = json.loads(
        (result_root / "gpu_ladder_prerequisite_matrix_report.json").read_text()
    )
    assert set(report) == {
        "arms",
        "claim_status",
        "dry_run",
        "experiment_id",
        "generated_epoch",
        "rsi_evidence",
        "run_index",
        "schema_version",
        "spec_path",
        "status",
    }
    assert report["schema_version"] == 1
    assert report["experiment_id"] == "gpu_ladder_prerequisite"
    assert report["spec_path"] == str(spec_path)
    assert report["dry_run"] is True
    assert report["status"] == "planned"
    assert report["claim_status"] == "planned"
    assert report["run_index"] is None
    assert {arm["name"]: arm["claim_eligible"] for arm in plan["arms"]} == {
        "g1_prerequisite": False,
        "g2_prerequisite": True,
        "g4_prerequisite": False,
        "g8_prerequisite": False,
    }
    assert {arm["name"]: arm["claim_eligible"] for arm in report["arms"]} == {
        "g1_prerequisite": False,
        "g2_prerequisite": True,
        "g4_prerequisite": False,
        "g8_prerequisite": False,
    }
    assert {arm["name"]: arm["claim_label"] for arm in report["arms"]} == {
        "g1_prerequisite": "B200 compatibility patchset",
        "g2_prerequisite": "B200 compatibility patchset",
        "g4_prerequisite": "B200 compatibility patchset",
        "g8_prerequisite": "B200 compatibility patchset",
    }
    assert {arm["name"]: arm["status"] for arm in report["arms"]} == {
        "g1_prerequisite": "planned",
        "g2_prerequisite": "planned",
        "g4_prerequisite": "planned",
        "g8_prerequisite": "planned",
    }
    assert {arm["name"]: arm["claim_status"] for arm in report["arms"]} == {
        "g1_prerequisite": "planned",
        "g2_prerequisite": "planned",
        "g4_prerequisite": "planned",
        "g8_prerequisite": "planned",
    }
    assert report["rsi_evidence"]["planned_arms"] == [
        {"run_id": "gpu_ladder_prerequisite_g1_prerequisite", "reason": "dry_run"},
        {"run_id": "gpu_ladder_prerequisite_g2_prerequisite", "reason": "dry_run"},
        {"run_id": "gpu_ladder_prerequisite_g4_prerequisite", "reason": "dry_run"},
        {"run_id": "gpu_ladder_prerequisite_g8_prerequisite", "reason": "dry_run"},
    ]
    assert set(report["rsi_evidence"]) == {
        "arm_count",
        "blockers",
        "claim_rejections",
        "excluded_authority",
        "observability_warnings",
        "planned_arms",
        "recommendation_quality_inputs",
        "skipped_arms",
        "status",
        "successful_arm_count",
        "training_metrics",
    }
    assert report["rsi_evidence"]["status"] == "advisory"
    assert report["rsi_evidence"]["arm_count"] == 4
    assert report["rsi_evidence"]["successful_arm_count"] == 0
    assert report["rsi_evidence"]["training_metrics"] == []
    assert report["rsi_evidence"]["blockers"] == []
    assert report["rsi_evidence"]["claim_rejections"] == []
    assert report["rsi_evidence"]["skipped_arms"] == []
    assert report["rsi_evidence"]["observability_warnings"] == []


def test_cli_defaults_to_dry_run_and_requires_execute_for_real_arms():
    args = run_experiment_matrix.parse_args([])

    assert args.dry_run is True
    assert args.execute is False

    execute_args = run_experiment_matrix.parse_args(["--execute"])

    assert execute_args.dry_run is False
    assert execute_args.execute is True


def test_matrix_runner_stops_after_first_failed_arm_and_refreshes_index(
    tmp_path: Path, monkeypatch
):
    spec_path = tmp_path / "matrix.json"
    result_root = tmp_path / "results"
    _write_spec(spec_path, result_root)
    configs = []

    def fake_run_attempt(config):
        configs.append(config)
        config.result_dir.mkdir(parents=True, exist_ok=True)
        (config.result_dir / "summary.json").write_text(
            json.dumps(
                {
                    "classification": {
                        "run_id": config.run_id,
                        "attempt_id": config.attempt_id,
                        "lane": config.lane,
                        "mode": config.mode,
                        "claim_eligible": False,
                    },
                    "exit_code": {"exit_code": len(configs) - 1, "phase": "training"},
                }
            )
            + "\n"
        )
        return len(configs) - 1

    monkeypatch.setattr(
        run_experiment_matrix.run_speedrun, "run_attempt", fake_run_attempt
    )
    monkeypatch.setattr(
        run_experiment_matrix.summarize,
        "write_index",
        lambda results, output: {"results_root": str(results), "output": str(output)},
    )

    exit_code = run_experiment_matrix.run_matrix(spec_path)

    assert exit_code == 1
    assert [config.num_gpus for config in configs] == [1, 2]
    assert [config.gpu_ids for config in configs] == [(0,), (2, 3)]
    assert [config.run_id for config in configs] == [
        "gpu_ladder_prerequisite_g1",
        "gpu_ladder_prerequisite_g2_pair",
    ]
    report = json.loads(
        (result_root / "gpu_ladder_prerequisite_matrix_report.json").read_text()
    )
    assert [arm["status"] for arm in report["arms"]] == [
        "passed",
        "failed",
        "skipped",
    ]
    assert [arm["claim_status"] for arm in report["arms"]] == [
        "rejected",
        "failed",
        "skipped",
    ]
    assert [arm["exit_code"] for arm in report["arms"]] == [0, 1, None]
    assert report["arms"][2]["skip_reason"] == "previous_arm_failed"
    assert report["status"] == "failed"
    assert report["claim_status"] == "failed"
    assert report["run_index"]["output"] == str(result_root / "run_index.json")
    assert (
        json.loads((configs[1].result_dir / "matrix_arm.json").read_text())[
            "observability"
        ]["profile"]
        == "mycroft"
    )


def test_matrix_rsi_success_count_requires_summary_evidence(
    tmp_path: Path, monkeypatch
):
    spec_path = tmp_path / "matrix.json"
    result_root = tmp_path / "results"
    _write_spec(spec_path, result_root)

    def fake_run_attempt(config):
        config.result_dir.mkdir(parents=True, exist_ok=True)
        return 0

    monkeypatch.setattr(
        run_experiment_matrix.run_speedrun, "run_attempt", fake_run_attempt
    )
    monkeypatch.setattr(
        run_experiment_matrix.summarize,
        "write_index",
        lambda results, output: {"groups": [], "launch_ready_attempts": []},
    )

    exit_code = run_experiment_matrix.run_matrix(spec_path)

    assert exit_code == 0
    report = json.loads(
        (result_root / "gpu_ladder_prerequisite_matrix_report.json").read_text()
    )
    assert [arm["status"] for arm in report["arms"]] == ["passed", "passed", "passed"]
    assert [arm["claim_status"] for arm in report["arms"]] == [
        "missing_summary",
        "missing_summary",
        "missing_summary",
    ]
    assert report["status"] == "passed"
    assert report["claim_status"] == "missing_summary"
    assert report["rsi_evidence"]["successful_arm_count"] == 0
    assert report["rsi_evidence"]["training_metrics"] == []
    assert report["rsi_evidence"]["observability_warnings"] == [
        {
            "run_id": "gpu_ladder_prerequisite_g1",
            "warning": "missing summary.json",
        },
        {
            "run_id": "gpu_ladder_prerequisite_g2_pair",
            "warning": "missing summary.json",
        },
        {
            "run_id": "gpu_ladder_prerequisite_g4_pending",
            "warning": "missing summary.json",
        },
    ]


def test_matrix_rsi_success_count_requires_claim_valid_summary(
    tmp_path: Path, monkeypatch
):
    spec_path = tmp_path / "matrix.json"
    result_root = tmp_path / "results"
    _write_spec(spec_path, result_root)

    def fake_run_attempt(config):
        config.result_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "ok": False,
            "included_in_baseline_stats": False,
            "classification": {
                "run_id": config.run_id,
                "attempt_id": config.attempt_id,
                "lane": config.lane,
                "mode": config.mode,
                "claim_eligible": True,
            },
            "metrics": {
                "train_time_seconds": 12.0,
                "validation_loss": 3.25,
                "step_avg_seconds": 0.5,
            },
            "blocker": {
                "phase": "claim_validation",
                "message": "summary parser rejected the attempted result",
            },
            "exit_code": {"exit_code": 0, "phase": "training"},
        }
        (config.result_dir / "summary.json").write_text(json.dumps(summary) + "\n")
        return 0

    monkeypatch.setattr(
        run_experiment_matrix.run_speedrun, "run_attempt", fake_run_attempt
    )
    monkeypatch.setattr(
        run_experiment_matrix.summarize,
        "write_index",
        lambda results, output: {"groups": [], "launch_ready_attempts": []},
    )

    exit_code = run_experiment_matrix.run_matrix(spec_path)

    assert exit_code == 0
    report = json.loads(
        (result_root / "gpu_ladder_prerequisite_matrix_report.json").read_text()
    )
    assert [arm["status"] for arm in report["arms"]] == ["passed", "passed", "passed"]
    assert [arm["claim_status"] for arm in report["arms"]] == [
        "rejected",
        "rejected",
        "rejected",
    ]
    assert report["status"] == "passed"
    assert report["claim_status"] == "rejected"
    assert report["rsi_evidence"]["successful_arm_count"] == 0
    assert report["rsi_evidence"]["training_metrics"] == []
    assert report["rsi_evidence"]["blockers"] == [
        {
            "run_id": "gpu_ladder_prerequisite_g1",
            "phase": "claim_validation",
            "message": "summary parser rejected the attempted result",
        },
        {
            "run_id": "gpu_ladder_prerequisite_g2_pair",
            "phase": "claim_validation",
            "message": "summary parser rejected the attempted result",
        },
        {
            "run_id": "gpu_ladder_prerequisite_g4_pending",
            "phase": "claim_validation",
            "message": "summary parser rejected the attempted result",
        },
    ]


def test_matrix_rsi_training_metrics_accept_parser_final_metrics(
    tmp_path: Path, monkeypatch
):
    spec_path = tmp_path / "matrix.json"
    result_root = tmp_path / "results"
    _write_spec(spec_path, result_root)

    def fake_run_attempt(config):
        config.result_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "ok": True,
            "included_in_baseline_stats": True,
            "classification": {
                "run_id": config.run_id,
                "attempt_id": config.attempt_id,
                "lane": config.lane,
                "mode": config.mode,
                "claim_eligible": True,
            },
            "final_metrics": {
                "val_loss": 3.25,
                "train_time": 72.5,
                "step_avg": 0.42,
            },
            "exit_code": {"exit_code": 0, "phase": "training"},
        }
        (config.result_dir / "summary.json").write_text(json.dumps(summary) + "\n")
        return 0

    monkeypatch.setattr(
        run_experiment_matrix.run_speedrun, "run_attempt", fake_run_attempt
    )
    monkeypatch.setattr(
        run_experiment_matrix.summarize,
        "write_index",
        lambda results, output: {"groups": [], "launch_ready_attempts": []},
    )

    exit_code = run_experiment_matrix.run_matrix(spec_path)

    assert exit_code == 0
    report = json.loads(
        (result_root / "gpu_ladder_prerequisite_matrix_report.json").read_text()
    )
    assert report["claim_status"] == "accepted"
    assert report["rsi_evidence"]["successful_arm_count"] == 3
    assert report["rsi_evidence"]["training_metrics"] == [
        {
            "run_id": "gpu_ladder_prerequisite_g1",
            "val_loss": 3.25,
            "train_time": 72.5,
            "step_avg": 0.42,
        },
        {
            "run_id": "gpu_ladder_prerequisite_g2_pair",
            "val_loss": 3.25,
            "train_time": 72.5,
            "step_avg": 0.42,
        },
        {
            "run_id": "gpu_ladder_prerequisite_g4_pending",
            "val_loss": 3.25,
            "train_time": 72.5,
            "step_avg": 0.42,
        },
    ]


def test_observability_profiles_add_advisory_matrix_evidence(
    tmp_path: Path, monkeypatch
):
    spec_path = tmp_path / "observability.json"
    result_root = tmp_path / "results"
    _write_observability_spec(spec_path, result_root)

    def fake_run_attempt(config):
        config.result_dir.mkdir(parents=True, exist_ok=True)
        exit_code = 124 if config.run_id.endswith("_planner") else 0
        summary = {
            "ok": exit_code == 0,
            "included_in_baseline_stats": exit_code == 0,
            "classification": {
                "run_id": config.run_id,
                "attempt_id": config.attempt_id,
                "lane": config.lane,
                "mode": config.mode,
                "claim_eligible": False,
            },
            "metrics": {
                "train_time_seconds": 12.0,
                "validation_loss": 3.25,
                "step_avg_seconds": 0.5,
            },
            "exit_code": {"exit_code": exit_code, "phase": "training"},
        }
        if exit_code != 0:
            summary["blocker"] = {
                "phase": "stall",
                "message": "training produced no output",
            }
        (config.result_dir / "summary.json").write_text(json.dumps(summary) + "\n")
        return exit_code

    monkeypatch.setattr(
        run_experiment_matrix.run_speedrun, "run_attempt", fake_run_attempt
    )
    monkeypatch.setattr(
        run_experiment_matrix.summarize,
        "write_index",
        lambda results, output: {"groups": [], "launch_ready_attempts": []},
    )

    exit_code = run_experiment_matrix.run_matrix(spec_path)

    assert exit_code == 124
    report = json.loads(
        (result_root / "observability_matrix_matrix_report.json").read_text()
    )
    timeline = report["arms"][0]["observability_summary"]
    assert timeline["profile"] == "mycroft"
    assert timeline["semantic_timeline"]["status"] == "advisory"
    assert timeline["feature_status"]["semantic_timeline"] == "advisory"
    planner = report["arms"][1]["observability_summary"]
    assert planner["diagnostic_recommendations"][0]["action"] == "inspect_stop_snapshot"
    assert planner["diagnostic_recommendations"][0]["launches_probe"] is False
    rsi = report["rsi_evidence"]
    assert rsi["status"] == "advisory"
    assert rsi["arm_count"] == 3
    assert [arm["claim_status"] for arm in report["arms"]] == [
        "accepted",
        "failed",
        "skipped",
    ]
    assert report["status"] == "failed"
    assert report["claim_status"] == "failed"
    assert rsi["successful_arm_count"] == 1
    assert rsi["training_metrics"] == [
        {
            "run_id": "observability_matrix_timeline",
            "train_time_seconds": 12.0,
            "validation_loss": 3.25,
            "step_avg_seconds": 0.5,
        }
    ]
    assert rsi["skipped_arms"] == [
        {
            "reason": "previous_arm_failed",
            "run_id": "observability_matrix_rsi",
        }
    ]
    assert rsi["observability_warnings"] == []
    assert "automatic_recovery" in rsi["excluded_authority"]


def test_observability_advisory_treats_parser_rejection_as_incident(
    tmp_path: Path, monkeypatch
):
    spec_path = tmp_path / "observability.json"
    result_root = tmp_path / "results"
    _write_observability_spec(spec_path, result_root)

    def fake_run_attempt(config):
        config.result_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "ok": False,
            "included_in_baseline_stats": False,
            "classification": {
                "run_id": config.run_id,
                "attempt_id": config.attempt_id,
                "lane": config.lane,
                "mode": config.mode,
                "claim_eligible": True,
            },
            "metrics": {
                "train_time_seconds": 12.0,
                "validation_loss": 3.25,
                "step_avg_seconds": 0.5,
            },
            "exit_code": {"exit_code": 0, "phase": "training"},
        }
        if config.run_id.endswith("_planner"):
            summary["blocker"] = {
                "phase": "claim_validation",
                "message": "summary parser rejected the attempted result",
            }
        (config.result_dir / "summary.json").write_text(json.dumps(summary) + "\n")
        return 0

    monkeypatch.setattr(
        run_experiment_matrix.run_speedrun, "run_attempt", fake_run_attempt
    )
    monkeypatch.setattr(
        run_experiment_matrix.summarize,
        "write_index",
        lambda results, output: {"groups": [], "launch_ready_attempts": []},
    )

    exit_code = run_experiment_matrix.run_matrix(spec_path)

    assert exit_code == 0
    report = json.loads(
        (result_root / "observability_matrix_matrix_report.json").read_text()
    )
    assert report["claim_status"] == "rejected"
    timeline = report["arms"][0]["observability_summary"]
    assert timeline["semantic_timeline"]["center"] == "incident"
    assert timeline["semantic_timeline"]["events"][1] == {
        "kind": "summary_ok",
        "ok": False,
    }
    planner = report["arms"][1]["observability_summary"]
    assert planner["diagnostic_recommendations"][0] == {
        "action": "inspect_claim_validation",
        "reason": "summary parser rejected the attempted result",
        "launches_probe": False,
        "risk": "none",
    }
    assert report["rsi_evidence"]["successful_arm_count"] == 0
    assert report["rsi_evidence"]["training_metrics"] == []


def test_observability_advisory_treats_missing_summary_ok_as_unaccepted(
    tmp_path: Path, monkeypatch
):
    spec_path = tmp_path / "observability.json"
    result_root = tmp_path / "results"
    _write_observability_spec(spec_path, result_root)

    def fake_run_attempt(config):
        config.result_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "included_in_baseline_stats": False,
            "classification": {
                "run_id": config.run_id,
                "attempt_id": config.attempt_id,
                "lane": config.lane,
                "mode": config.mode,
                "claim_eligible": True,
            },
            "metrics": {
                "train_time_seconds": 12.0,
                "validation_loss": 3.25,
                "step_avg_seconds": 0.5,
            },
            "exit_code": {"exit_code": 0, "phase": "training"},
        }
        (config.result_dir / "summary.json").write_text(json.dumps(summary) + "\n")
        return 0

    monkeypatch.setattr(
        run_experiment_matrix.run_speedrun, "run_attempt", fake_run_attempt
    )
    monkeypatch.setattr(
        run_experiment_matrix.summarize,
        "write_index",
        lambda results, output: {"groups": [], "launch_ready_attempts": []},
    )

    exit_code = run_experiment_matrix.run_matrix(spec_path)

    assert exit_code == 0
    report = json.loads(
        (result_root / "observability_matrix_matrix_report.json").read_text()
    )
    assert report["claim_status"] == "rejected"
    timeline = report["arms"][0]["observability_summary"]
    assert timeline["semantic_timeline"]["center"] == "incident"
    assert timeline["semantic_timeline"]["events"][1] == {
        "kind": "summary_ok",
        "ok": None,
    }
    assert timeline["feature_status"]["active_job_scan"] == "stale"
    assert timeline["feature_status"]["structured_log_parsing"] == "stale"
    assert timeline["feature_status"]["semantic_timeline"] == "advisory"
    planner = report["arms"][1]["observability_summary"]
    assert planner["diagnostic_recommendations"][0] == {
        "action": "inspect_claim_validation",
        "reason": "summary parser did not mark the attempted result ok",
        "launches_probe": False,
        "risk": "none",
    }
    assert report["rsi_evidence"]["successful_arm_count"] == 0
    assert report["rsi_evidence"]["training_metrics"] == []
    assert report["rsi_evidence"]["claim_rejections"] == [
        {
            "run_id": "observability_matrix_timeline",
            "summary_ok": None,
            "reason": "summary parser did not mark the attempted result ok",
        },
        {
            "run_id": "observability_matrix_planner",
            "summary_ok": None,
            "reason": "summary parser did not mark the attempted result ok",
        },
        {
            "run_id": "observability_matrix_rsi",
            "summary_ok": None,
            "reason": "summary parser did not mark the attempted result ok",
        },
    ]
