#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run schematized modded-nanogpt B200 experiment matrices."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.modded_nanogpt_b200 import (
    cli_guard,
    experiment_config,
    run_speedrun,
    summarize,
)


def run_matrix(spec_path: Path, *, dry_run: bool = False) -> int:
    spec = experiment_config.load_experiment_spec(spec_path)
    plan = experiment_config.materialize_experiment_plan(spec)
    result_root = Path(spec.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        result_root / f"{spec.experiment_id}_plan.json",
        plan.to_dict(),
    )

    matrix_arms: list[dict[str, Any]] = []
    exit_code = 0
    for arm, arm_plan in zip(spec.arms, plan.arms):
        if dry_run:
            arm_record = {
                **arm_plan.to_dict(),
                "status": "planned",
                "exit_code": None,
            }
            matrix_arms.append(arm_record)
            continue

        config = _run_config_from_arm(arm, arm_plan)
        config.result_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(config.result_dir / "matrix_arm.json", arm_plan.to_dict())
        arm_exit = run_speedrun.run_attempt(config)
        observability_summary = _observability_summary(
            arm_plan=arm_plan,
            summary_path=config.result_dir / "summary.json",
            exit_code=arm_exit,
        )
        matrix_arms.append(
            {
                **arm_plan.to_dict(),
                "status": "passed" if arm_exit == 0 else "failed",
                "exit_code": arm_exit,
                "summary": str(config.result_dir / "summary.json"),
                "observability_summary": observability_summary,
            }
        )
        if arm_exit != 0:
            exit_code = arm_exit

    run_index = None
    if not dry_run:
        run_index = summarize.write_index(result_root, result_root / "run_index.json")

    report = {
        "schema_version": experiment_config.SUPPORTED_SCHEMA_VERSION,
        "experiment_id": spec.experiment_id,
        "spec_path": str(spec_path),
        "dry_run": dry_run,
        "status": "planned" if dry_run else ("passed" if exit_code == 0 else "failed"),
        "generated_epoch": time.time(),
        "arms": matrix_arms,
        "run_index": run_index,
        "rsi_evidence": _rsi_evidence(matrix_arms),
    }
    _write_json_atomic(result_root / f"{spec.experiment_id}_matrix_report.json", report)
    return exit_code


def _run_config_from_arm(
    arm: experiment_config.ExperimentArm,
    arm_plan: experiment_config.ExperimentArmPlan,
) -> run_speedrun.RunConfig:
    values = arm.values
    return run_speedrun.RunConfig(
        lane=arm.legacy_lane,
        mode=str(values.get("mode", "full")),
        source=Path(str(values["source"])),
        data_manifest=Path(str(values["data_manifest"])),
        result_dir=arm_plan.result_dir,
        attention_backend=str(values.get("attention_backend", "fa3")),
        mlp_backend=str(values.get("mlp_backend", "triton")),
        verify_sha=bool(values.get("verify_sha", False)),
        run_id=arm_plan.run_id,
        attempt_id=arm_plan.attempt_id,
        arm=arm.name,
        num_gpus=arm.world_size,
        gpu_ids=arm.gpu_ids,
        result_root=Path(str(values["result_root"])),
    )


def _observability_summary(
    *,
    arm_plan: experiment_config.ExperimentArmPlan,
    summary_path: Path,
    exit_code: int,
) -> dict[str, Any]:
    summary = _read_json(summary_path)
    feature_status = _feature_status(arm_plan.observability.features, summary)
    result = {
        "profile": arm_plan.observability.profile,
        "feature_status": feature_status,
        "missing_evidence": [
            feature
            for feature, status in feature_status.items()
            if status in {"missing", "stale"}
        ],
        "advisory_only": True,
        "excluded_authority": _excluded_authority(),
    }
    if feature_status.get("semantic_timeline") == "advisory":
        result["semantic_timeline"] = _semantic_timeline(summary, exit_code)
    if feature_status.get("active_diagnostic_planner") == "advisory":
        result["diagnostic_recommendations"] = _diagnostic_recommendations(
            summary, exit_code
        )
    return result


def _feature_status(
    features: dict[str, str], summary: dict[str, Any] | None
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for feature, declared in features.items():
        if declared == "observed":
            statuses[feature] = "observed" if summary is not None else "missing"
        else:
            statuses[feature] = declared
    return dict(sorted(statuses.items()))


def _semantic_timeline(
    summary: dict[str, Any] | None, exit_code: int
) -> dict[str, Any]:
    if summary is None:
        return {"status": "advisory", "center": "missing_summary", "events": []}
    blocker = summary.get("blocker") if isinstance(summary.get("blocker"), dict) else {}
    exit_record = (
        summary.get("exit_code") if isinstance(summary.get("exit_code"), dict) else {}
    )
    center = "incident" if exit_code != 0 or blocker else "completion"
    return {
        "status": "advisory",
        "center": center,
        "events": [
            {
                "kind": "exit_code",
                "phase": exit_record.get("phase", "unknown"),
                "exit_code": exit_record.get("exit_code", exit_code),
            },
            {
                "kind": "blocker",
                "phase": blocker.get("phase"),
                "message": blocker.get("message"),
            },
        ],
    }


def _diagnostic_recommendations(
    summary: dict[str, Any] | None, exit_code: int
) -> list[dict[str, Any]]:
    if summary is None:
        return [
            {
                "action": "inspect_missing_summary",
                "reason": "summary.json was not found",
                "launches_probe": False,
                "risk": "none",
            }
        ]
    blocker = summary.get("blocker") if isinstance(summary.get("blocker"), dict) else {}
    phase = blocker.get("phase")
    if exit_code == run_speedrun.NO_OUTPUT_TIMEOUT_EXIT_CODE or phase == "stall":
        return [
            {
                "action": "inspect_stop_snapshot",
                "reason": "stall or no-output timeout evidence is present",
                "launches_probe": False,
                "risk": "none",
            },
            {
                "action": "review_process_watch",
                "reason": "process telemetry may distinguish compile from hang",
                "launches_probe": False,
                "risk": "none",
            },
        ]
    return [
        {
            "action": "compare_arm_metrics",
            "reason": "completed arm can be compared with peer arms",
            "launches_probe": False,
            "risk": "none",
        }
    ]


def _rsi_evidence(matrix_arms: list[dict[str, Any]]) -> dict[str, Any]:
    training_metrics = []
    blockers = []
    observability_warnings = []
    for arm in matrix_arms:
        summary = _read_json(Path(str(arm.get("summary", ""))))
        if summary is None:
            observability_warnings.append(
                {"run_id": arm.get("run_id"), "warning": "missing summary.json"}
            )
            continue
        metrics = (
            summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
        )
        if metrics:
            training_metrics.append({"run_id": arm.get("run_id"), **metrics})
        blocker = (
            summary.get("blocker") if isinstance(summary.get("blocker"), dict) else None
        )
        if blocker is not None:
            blockers.append({"run_id": arm.get("run_id"), **blocker})
    return {
        "status": "advisory",
        "arm_count": len(matrix_arms),
        "successful_arm_count": sum(
            1 for arm in matrix_arms if arm.get("exit_code") == 0
        ),
        "training_metrics": training_metrics,
        "blockers": blockers,
        "observability_warnings": observability_warnings,
        "recommendation_quality_inputs": {
            "detection": "advisory",
            "localization": "advisory",
            "recommendation": "advisory",
            "training_quality": "advisory",
        },
        "excluded_authority": _excluded_authority(),
    }


def _excluded_authority() -> list[str]:
    return [
        "automatic_recovery",
        "restart",
        "rollback",
        "quarantine",
        "eviction",
        "learned_safety_decision",
    ]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as fp:
            data = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/modded_nanogpt_b200/configs/gpu_ladder_prerequisite.json"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main(*, enforce_rootfs: bool = False) -> int:
    if enforce_rootfs:
        guard_exit = cli_guard.guard_rootfs_cli(
            "experiments/modded_nanogpt_b200/run_experiment_matrix.sh"
        )
        if guard_exit is not None:
            return guard_exit
    args = parse_args()
    try:
        return run_matrix(args.config, dry_run=args.dry_run)
    except (ValueError, experiment_config.SchemaValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 21


if __name__ == "__main__":
    raise SystemExit(main(enforce_rootfs=True))
