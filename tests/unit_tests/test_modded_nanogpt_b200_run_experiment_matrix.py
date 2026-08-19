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


def test_matrix_runner_executes_arms_sequentially_and_refreshes_index(
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
    assert [arm["exit_code"] for arm in report["arms"]] == [0, 1]
    assert report["status"] == "failed"
    assert report["run_index"]["output"] == str(result_root / "run_index.json")
    assert (
        json.loads((configs[1].result_dir / "matrix_arm.json").read_text())[
            "observability"
        ]["profile"]
        == "mycroft"
    )


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
    assert rsi["successful_arm_count"] == 2
    assert rsi["training_metrics"][0]["run_id"] == "observability_matrix_timeline"
    assert "automatic_recovery" in rsi["excluded_authority"]
