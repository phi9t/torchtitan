#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build a conservative run index from modded-nanogpt B200 summaries."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.modded_nanogpt_b200 import cli_guard


SCHEMA_VERSION = 1
GROUP_FIELDS = (
    "lane",
    "mode",
    "arm",
    "claim_label",
    "environment_class",
    "evidence_tier",
)
GROUP_ATTEMPT_FIELDS = (
    "source_commit",
    "data_manifest",
)
BASELINE_NUMERIC_METRICS = ("val_loss", "train_time", "step_avg")
BASELINE_MAX_VAL_LOSS = 3.28
FULL_MANIFEST_TOKEN_BUDGET = "900M"
FULL_MANIFEST_NUM_FILES = 10
FULL_MANIFEST_TOTAL_BYTES = 2_000_010_240
FULL_LAUNCH_AUTHORIZATION_TOKEN = "launch-full-b200"
FULL_TRIAL_GPU_COUNT = 2
CANONICAL_RUNTIME_PYTHON = "/project/venvs/b200-runtime/bin/python"
UPSTREAM_COMMIT = "ecbb586296d3dac36fd206211f25d63bad4a6b35"
MAX_STALE_OR_DEMOTED_ARTIFACTS = 20
STALE_OR_DEMOTED_OPERATOR_ACTION = (
    "Archived or demoted artifact; do not use as launch-ready or baseline evidence."
)


def _load_summary(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _group_key(attempt: dict[str, Any]) -> str:
    classification = attempt["classification"]
    fields = [f"{field}={classification.get(field)}" for field in GROUP_FIELDS]
    fields.extend(f"{field}={attempt.get(field)}" for field in GROUP_ATTEMPT_FIELDS)
    return "|".join(fields)


def _normalize_classification(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        claim_label = value
    else:
        claim_label = "legacy summary without classification"
    return {
        "lane": "unknown",
        "mode": "legacy",
        "arm": "unknown",
        "claim_label": claim_label,
        "environment_class": "unknown",
        "evidence_tier": "legacy",
    }


def _normalize_blocker(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    return {"phase": "unknown", "message": str(value)}


def _summary_shape_error(path: Path, field: str, expected: str) -> dict[str, str]:
    return {
        "phase": "summary",
        "message": f"{path}: expected {expected} in {field}",
    }


def _baseline_metrics_error(
    path: Path, metrics: dict[str, Any]
) -> dict[str, str] | None:
    for name in BASELINE_NUMERIC_METRICS:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return _summary_shape_error(path, f"final_metrics.{name}", "number")
    if metrics["val_loss"] > BASELINE_MAX_VAL_LOSS:
        return _summary_shape_error(
            path,
            "final_metrics.val_loss",
            f"number <= {BASELINE_MAX_VAL_LOSS}",
        )
    return None


def _has_rootfs_sentinel_evidence(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("torchtitan_in_rootfs") == "1"
        and value.get("cwd") == "/workspace/torchtitan"
        and value.get("workspace_sentinel_exists") is True
    )


def _has_valid_lane_b_patch_classification(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    diff_artifact = value.get("diff_artifact")
    if not isinstance(diff_artifact, str) or not diff_artifact:
        return False
    files = value.get("files")
    if not isinstance(files, list) or not files:
        return False
    for item in files:
        if not isinstance(item, dict):
            return False
        path = item.get("path")
        patch_class = item.get("patch_class")
        first_blocker = item.get("first_lane_a_blocker")
        if not isinstance(path, str) or not path:
            return False
        if (
            not isinstance(patch_class, str)
            or not patch_class
            or patch_class == "unclassified"
        ):
            return False
        if not isinstance(first_blocker, str) or not first_blocker:
            return False
    return True


def _has_clean_active_job_scan(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("ok") is True
        and value.get("active_job_count") == 0
        and not value.get("scan_error")
    )


def _runtime_verification_exclusion(summary_path: Path) -> dict[str, str] | None:
    result_dir = summary_path.parent
    command_env, command_env_error = _load_json_sidecar(result_dir / "command.env.json")
    if command_env_error is not None or command_env is None:
        return {
            "phase": "runtime_verification",
            "message": "missing current runtime verification evidence",
        }
    environment = command_env.get("environment")
    if not isinstance(environment, dict):
        return {
            "phase": "runtime_verification",
            "message": "command environment is missing runtime environment evidence",
        }
    required_env = {
        "TORCHTITAN_IN_ROOTFS": "1",
        "TORCHTITAN_ROOTFS_PROJECT": "/workspace/torchtitan",
        "TORCHTITAN_ROOTFS_NETWORK": "offline",
        "PYTHON": CANONICAL_RUNTIME_PYTHON,
    }
    for key, expected in required_env.items():
        if environment.get(key) != expected:
            return {
                "phase": "runtime_verification",
                "message": f"command environment {key} must be {expected}",
            }

    runtime_verification, runtime_error = _load_json_sidecar(
        result_dir / "runtime" / "runtime_verification.json"
    )
    if runtime_error is not None or runtime_verification is None:
        return {
            "phase": "runtime_verification",
            "message": "missing current runtime verification evidence",
        }
    if runtime_verification.get("ok") is not True:
        return {
            "phase": "runtime_verification",
            "message": "runtime verification must be ok",
        }
    if runtime_verification.get("training_launch_allowed") is not True:
        return {
            "phase": "runtime_verification",
            "message": "runtime verification must allow training launch",
        }
    if runtime_verification.get("command_env_digest") != command_env.get(
        "environment_digest"
    ):
        return {
            "phase": "runtime_verification",
            "message": "runtime verification command-env digest does not match",
        }
    return None


def _malformed_summary_attempt(path: Path, exc: Exception) -> dict[str, Any]:
    blocker = {"phase": "summary", "message": str(exc)}
    return {
        "summary": str(path),
        "classification": {
            "lane": "unknown",
            "mode": "malformed",
            "arm": "unknown",
            "claim_label": "malformed summary",
            "environment_class": "unknown",
            "evidence_tier": "malformed",
        },
        "upstream_commit": None,
        "source_commit": None,
        "source": None,
        "data_manifest": None,
        "environment": {},
        "gpus": [],
        "gpu_count": 0,
        "environment_sidecar": {},
        "hardware_sidecar": {},
        "final_metrics": {},
        "wall_clock": {},
        "included_in_baseline_stats": False,
        "blocker": blocker,
        "summary_error": blocker,
    }


def _sidecar_error(path: Path, exc: Exception | str) -> dict[str, str]:
    detail = str(exc)
    return {
        "phase": "launch_readiness",
        "message": f"{path}: {detail}",
    }


def _load_launch_readiness(
    summary_path: Path,
    summary: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    path = summary_path.parent / "launch_readiness.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            return None, _sidecar_error(path, exc)
    else:
        data = summary.get("launch_readiness")
        if data is None:
            return None, None
    if not isinstance(data, dict):
        return None, _sidecar_error(path, "expected JSON object")
    return data, None


def _load_json_sidecar(
    path: Path,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    if not path.exists():
        return None, _summary_shape_error(path, path.name, "JSON object")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, _summary_shape_error(path, path.name, f"valid JSON object: {exc}")
    if not isinstance(data, dict):
        return None, _summary_shape_error(path, path.name, "JSON object")
    return data, None


def _included_in_baseline_stats(
    summary: dict[str, Any], classification: dict[str, Any]
) -> bool:
    if summary.get("included_in_baseline_stats") is not True:
        return False
    return (
        classification.get("lane") in {"A", "B"}
        and classification.get("mode") == "full"
        and classification.get("claim_eligible") is True
        and summary.get("ok") is True
    )


def _baseline_evidence_error(
    path: Path,
    summary: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, str] | None:
    if classification.get("mode") != "full":
        return None
    launch_readiness = summary.get("launch_readiness")
    if not isinstance(launch_readiness, dict):
        return _summary_shape_error(path, "launch_readiness", "JSON object")
    full_mode_gates = launch_readiness.get("full_mode_gates")
    if not isinstance(full_mode_gates, dict):
        return _summary_shape_error(
            path, "launch_readiness.full_mode_gates", "JSON object"
        )
    if launch_readiness.get("training_launched") is not True:
        return _summary_shape_error(path, "launch_readiness.training_launched", "true")
    if launch_readiness.get("skip_run") is True:
        return _summary_shape_error(path, "launch_readiness.skip_run", "not true")
    if launch_readiness.get("allow_previous_stall") is True:
        return _summary_shape_error(
            path, "launch_readiness.allow_previous_stall", "not true"
        )
    if launch_readiness.get("preflight_ok") is not True:
        return _summary_shape_error(path, "launch_readiness.preflight_ok", "true")
    exit_code, exit_code_error = _load_json_sidecar(path.parent / "exit_code.json")
    if exit_code_error is not None:
        return exit_code_error
    if exit_code is None:
        return _summary_shape_error(
            path.parent / "exit_code.json", "exit_code.json", "JSON object"
        )
    if exit_code.get("phase") != "training" or exit_code.get("exit_code") != 0:
        return _summary_shape_error(
            path.parent / "exit_code.json",
            "exit_code.json",
            "phase=training and exit_code=0",
        )
    telemetry = summary.get("telemetry")
    rootfs = telemetry.get("rootfs") if isinstance(telemetry, dict) else None
    if not isinstance(rootfs, dict):
        return _summary_shape_error(path, "telemetry.rootfs", "JSON object")
    if not _has_rootfs_sentinel_evidence(rootfs):
        return _summary_shape_error(
            path, "telemetry.rootfs", "rootfs sentinel evidence"
        )
    if full_mode_gates.get("verify_sha_requested") is not True:
        return _summary_shape_error(
            path, "launch_readiness.full_mode_gates.verify_sha_requested", "true"
        )
    if full_mode_gates.get("data_manifest_checked") is not True:
        return _summary_shape_error(
            path, "launch_readiness.full_mode_gates.data_manifest_checked", "true"
        )
    data_manifest = summary.get("data_manifest_summary")
    if not isinstance(data_manifest, dict):
        return _summary_shape_error(path, "data_manifest_summary", "JSON object")
    preflight_data_manifest = summary.get("preflight_data_manifest")
    if not isinstance(preflight_data_manifest, dict):
        return _summary_shape_error(path, "preflight_data_manifest", "JSON object")
    manifest_error = _manifest_evidence_error(
        path=path,
        launch_readiness=launch_readiness,
        data_manifest=data_manifest,
        preflight_data_manifest=preflight_data_manifest,
        require_sha=True,
    )
    if manifest_error is not None:
        return manifest_error
    if full_mode_gates.get("verified_sha") is not True:
        return _summary_shape_error(
            path, "launch_readiness.full_mode_gates.verified_sha", "true"
        )
    if full_mode_gates.get("manifest_token_budget") != FULL_MANIFEST_TOKEN_BUDGET:
        return _summary_shape_error(
            path,
            "launch_readiness.full_mode_gates.manifest_token_budget",
            FULL_MANIFEST_TOKEN_BUDGET,
        )
    if full_mode_gates.get("manifest_num_files") != FULL_MANIFEST_NUM_FILES:
        return _summary_shape_error(
            path,
            "launch_readiness.full_mode_gates.manifest_num_files",
            str(FULL_MANIFEST_NUM_FILES),
        )
    if full_mode_gates.get("manifest_total_bytes") != FULL_MANIFEST_TOTAL_BYTES:
        return _summary_shape_error(
            path,
            "launch_readiness.full_mode_gates.manifest_total_bytes",
            str(FULL_MANIFEST_TOTAL_BYTES),
        )
    if full_mode_gates.get("nccl_checked") is not True:
        return _summary_shape_error(
            path, "launch_readiness.full_mode_gates.nccl_checked", "true"
        )
    gpus = summary.get("gpus")
    if not isinstance(gpus, list):
        return _summary_shape_error(path, "gpus", "JSON array")
    if len(gpus) != FULL_TRIAL_GPU_COUNT or any(
        not isinstance(gpu, dict) or "B200" not in str(gpu.get("name", ""))
        for gpu in gpus
    ):
        return _summary_shape_error(
            path,
            "gpus",
            f"{FULL_TRIAL_GPU_COUNT} B200 GPU entries",
        )
    return None


def _manifest_evidence_error(
    *,
    path: Path,
    launch_readiness: dict[str, Any],
    data_manifest: dict[str, Any],
    preflight_data_manifest: dict[str, Any] | None = None,
    require_sha: bool,
) -> dict[str, str] | None:
    readiness_manifest = launch_readiness.get("data_manifest")
    accepted_manifest_paths = {
        value
        for value in (
            data_manifest.get("path"),
            data_manifest.get("resolved_path"),
        )
        if isinstance(value, str) and value
    }
    if (
        isinstance(readiness_manifest, str)
        and readiness_manifest
        and readiness_manifest not in accepted_manifest_paths
    ):
        return _summary_shape_error(
            path,
            "launch_readiness.data_manifest",
            "matching data_manifest_summary path",
        )
    if require_sha:
        if preflight_data_manifest is None:
            return _summary_shape_error(path, "preflight_data_manifest", "JSON object")
        if preflight_data_manifest.get("verified_sha") is not True:
            return {
                "phase": "preflight_data_manifest.verified_sha",
                "message": "preflight_data_manifest.verified_sha must be true",
            }
        if data_manifest.get("verified_sha") is not True:
            return {
                "phase": "data_manifest_summary.verified_sha",
                "message": "data_manifest_summary.verified_sha must be true",
            }
    if data_manifest.get("token_budget") != FULL_MANIFEST_TOKEN_BUDGET:
        return _summary_shape_error(
            path, "data_manifest_summary.token_budget", FULL_MANIFEST_TOKEN_BUDGET
        )
    if data_manifest.get("num_files") != FULL_MANIFEST_NUM_FILES:
        return _summary_shape_error(
            path, "data_manifest_summary.num_files", str(FULL_MANIFEST_NUM_FILES)
        )
    if data_manifest.get("total_bytes") != FULL_MANIFEST_TOTAL_BYTES:
        return _summary_shape_error(
            path,
            "data_manifest_summary.total_bytes",
            str(FULL_MANIFEST_TOTAL_BYTES),
        )
    if data_manifest.get("source_commit") != UPSTREAM_COMMIT:
        return _summary_shape_error(
            path, "data_manifest_summary.source_commit", UPSTREAM_COMMIT
        )
    return None


def _attempt_record(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    classification = _normalize_classification(summary.get("classification"))
    raw_metrics = summary.get("final_metrics", {})
    metrics_error = None
    baseline_evidence_error = None
    if isinstance(raw_metrics, dict):
        metrics = raw_metrics
    else:
        metrics = {}
        metrics_error = _summary_shape_error(path, "final_metrics", "JSON object")
    baseline_inclusion = metrics_error is None and _included_in_baseline_stats(
        summary, classification
    )
    if baseline_inclusion:
        baseline_evidence_error = _baseline_evidence_error(
            path, summary, classification
        )
        if baseline_evidence_error is None:
            metrics_error = _baseline_metrics_error(path, metrics)
        baseline_inclusion = metrics_error is None and baseline_evidence_error is None
    source = summary.get("source")
    source_commit = source.get("commit") if isinstance(source, dict) else None
    gpus = summary.get("gpus")
    telemetry = summary.get("telemetry")
    telemetry_signs = telemetry.get("signs") if isinstance(telemetry, dict) else None
    claim_validation = summary.get("claim_validation")
    variant_patch_classification = summary.get("variant_patch_classification")
    active_jobs = summary.get("active_jobs")
    attempt = {
        "summary": str(path),
        "classification": classification,
        "upstream_commit": summary.get("upstream_commit"),
        "source_commit": source_commit or summary.get("upstream_commit"),
        "source": source,
        "data_manifest": summary.get("data_manifest"),
        "data_manifest_summary": summary.get("data_manifest_summary", {}),
        "preflight_data_manifest": summary.get("preflight_data_manifest", {}),
        "environment": summary.get("environment", {}),
        "gpus": gpus if isinstance(gpus, list) else [],
        "gpu_count": len(gpus) if isinstance(gpus, list) else 0,
        "environment_sidecar": summary.get("environment_sidecar", {}),
        "hardware_sidecar": summary.get("hardware_sidecar", {}),
        "claim_validation": claim_validation
        if isinstance(claim_validation, dict)
        else {},
        "variant_patch_classification": variant_patch_classification
        if isinstance(variant_patch_classification, dict)
        else {},
        "active_jobs": active_jobs if isinstance(active_jobs, dict) else {},
        "telemetry": telemetry if isinstance(telemetry, dict) else {},
        "telemetry_signs": telemetry_signs if isinstance(telemetry_signs, dict) else {},
        "final_metrics": metrics,
        "wall_clock": summary.get("wall_clock", {}),
        "included_in_baseline_stats": baseline_inclusion,
        "blocker": _normalize_blocker(summary.get("blocker")),
    }
    if metrics_error is not None:
        attempt["metrics_error"] = metrics_error
        if attempt["blocker"] is None:
            attempt["blocker"] = metrics_error
    if baseline_evidence_error is not None:
        attempt["baseline_evidence_error"] = baseline_evidence_error
        if attempt["blocker"] is None:
            attempt["blocker"] = baseline_evidence_error
    launch_readiness, launch_readiness_error = _load_launch_readiness(path, summary)
    if launch_readiness is not None:
        attempt["launch_readiness"] = launch_readiness
    if launch_readiness_error is not None:
        attempt["launch_readiness_error"] = launch_readiness_error
    return attempt


def _baseline_stats(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    train_times = [
        attempt["final_metrics"].get("train_time")
        for attempt in attempts
        if attempt["included_in_baseline_stats"]
        and attempt.get("final_metrics", {}).get("train_time") is not None
    ]
    if not train_times:
        return {"count": 0, "best_train_time": None, "median_train_time": None}
    return {
        "count": len(train_times),
        "best_train_time": min(train_times),
        "median_train_time": statistics.median(train_times),
    }


def _sidecar_field_matches(
    classification: dict[str, Any],
    launch_readiness: dict[str, Any],
    field: str,
) -> bool:
    readiness_values = []
    readiness_value = launch_readiness.get(field)
    if readiness_value is not None:
        readiness_values.append(readiness_value)
    readiness_classification = launch_readiness.get("classification")
    if isinstance(readiness_classification, dict):
        readiness_value = readiness_classification.get(field)
        if readiness_value is not None:
            readiness_values.append(readiness_value)
    return all(value == classification.get(field) for value in readiness_values)


def _launch_readiness_exclusion(attempt: dict[str, Any]) -> dict[str, str] | None:
    classification = attempt.get("classification")
    launch_readiness = attempt.get("launch_readiness")
    if not isinstance(classification, dict) or not isinstance(launch_readiness, dict):
        return {
            "phase": "launch_readiness",
            "message": "missing launch-readiness evidence",
        }
    for field in ("run_id", "attempt_id", "lane", "mode"):
        if not _sidecar_field_matches(classification, launch_readiness, field):
            return {
                "phase": "launch_readiness",
                "message": f"launch-readiness {field} does not match summary",
            }
    full_mode_gates = launch_readiness.get("full_mode_gates")
    if not isinstance(full_mode_gates, dict):
        return {
            "phase": "launch_readiness",
            "message": "missing launch-readiness full-mode gate evidence",
        }
    summary_path = Path(attempt["summary"])
    data_manifest = attempt.get("data_manifest_summary")
    if not isinstance(data_manifest, dict) or not data_manifest:
        return {
            "phase": "data_manifest",
            "message": "summary is missing parsed data-manifest evidence",
        }
    manifest_error = _manifest_evidence_error(
        path=summary_path,
        launch_readiness=launch_readiness,
        data_manifest=data_manifest,
        preflight_data_manifest=attempt.get("preflight_data_manifest"),
        require_sha=True,
    )
    if manifest_error is not None:
        return manifest_error
    gpus = attempt.get("gpus")
    if not isinstance(gpus, list) or len(gpus) != FULL_TRIAL_GPU_COUNT:
        return {
            "phase": "gpu",
            "message": f"summary is missing {FULL_TRIAL_GPU_COUNT}x B200 GPU evidence",
        }
    if any(
        not isinstance(gpu, dict) or "B200" not in str(gpu.get("name", ""))
        for gpu in gpus
    ):
        return {
            "phase": "gpu",
            "message": f"summary is missing {FULL_TRIAL_GPU_COUNT}x B200 GPU evidence",
        }
    telemetry = attempt.get("telemetry")
    rootfs = telemetry.get("rootfs") if isinstance(telemetry, dict) else None
    if not _has_rootfs_sentinel_evidence(rootfs):
        return {
            "phase": "rootfs",
            "message": "summary is missing rootfs sentinel evidence",
        }
    if attempt.get("source_commit") != UPSTREAM_COMMIT:
        return {
            "phase": "source",
            "message": "summary source commit does not match pinned upstream commit",
        }
    if classification.get("lane") == "B" and not _has_valid_lane_b_patch_classification(
        attempt.get("variant_patch_classification")
    ):
        return {
            "phase": "variant_patch_classification",
            "message": "summary is missing valid Lane B patch classification",
        }
    if not _has_clean_active_job_scan(attempt.get("active_jobs")):
        return {
            "phase": "active_jobs",
            "message": "summary is missing clean active-job scan evidence",
        }
    runtime_exclusion = _runtime_verification_exclusion(summary_path)
    if runtime_exclusion is not None:
        return runtime_exclusion
    blocked_by = launch_readiness.get("blocked_by")
    if blocked_by:
        return {"phase": "launch_readiness", "message": "launch-readiness has blockers"}
    checks = (
        (classification.get("lane") in {"A", "B"}, "classification.lane", "A or B"),
        (classification.get("mode") == "full", "classification.mode", "full"),
        (
            classification.get("claim_eligible") is True,
            "classification.claim_eligible",
            "true",
        ),
        (launch_readiness.get("preflight_ok") is True, "preflight_ok", "true"),
        (full_mode_gates.get("full_mode") is True, "full_mode_gates.full_mode", "true"),
        (
            full_mode_gates.get("data_manifest_checked") is True,
            "full_mode_gates.data_manifest_checked",
            "true",
        ),
        (
            full_mode_gates.get("verified_sha") is True,
            "full_mode_gates.verified_sha",
            "true",
        ),
        (
            full_mode_gates.get("nccl_checked") is True,
            "full_mode_gates.nccl_checked",
            "true",
        ),
        (
            full_mode_gates.get("manifest_token_budget") == FULL_MANIFEST_TOKEN_BUDGET,
            "full_mode_gates.manifest_token_budget",
            FULL_MANIFEST_TOKEN_BUDGET,
        ),
        (
            full_mode_gates.get("manifest_num_files") == FULL_MANIFEST_NUM_FILES,
            "full_mode_gates.manifest_num_files",
            str(FULL_MANIFEST_NUM_FILES),
        ),
        (
            full_mode_gates.get("manifest_total_bytes") == FULL_MANIFEST_TOTAL_BYTES,
            "full_mode_gates.manifest_total_bytes",
            str(FULL_MANIFEST_TOTAL_BYTES),
        ),
        (launch_readiness.get("ready_to_launch") is True, "ready_to_launch", "true"),
        (
            launch_readiness.get("launch_authority_required") is True,
            "launch_authority_required",
            "true",
        ),
        (
            launch_readiness.get("launch_authorization_required_token")
            == FULL_LAUNCH_AUTHORIZATION_TOKEN,
            "launch_authorization_required_token",
            FULL_LAUNCH_AUTHORIZATION_TOKEN,
        ),
        (
            launch_readiness.get("training_launched") is not True,
            "training_launched",
            "not true",
        ),
    )
    for ok, field, expected in checks:
        if not ok:
            return {
                "phase": "launch_readiness",
                "message": f"{field} must be {expected}",
            }
    return None


def _is_launch_ready_prerequisite(attempt: dict[str, Any]) -> bool:
    return _launch_readiness_exclusion(attempt) is None


def _is_launch_ready_attempt(attempt: dict[str, Any]) -> bool:
    launch_readiness = attempt.get("launch_readiness")
    if not isinstance(launch_readiness, dict):
        return False
    return (
        _is_launch_ready_prerequisite(attempt)
        and launch_readiness.get("skip_run") is not True
    )


def _launch_readiness_exclusion_stats(
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    by_phase: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        exclusion = attempt.get("launch_readiness_exclusion")
        if not isinstance(exclusion, dict):
            continue
        phase = exclusion.get("phase")
        if not isinstance(phase, str) or not phase:
            phase = "unknown"
        phase_stats = by_phase.setdefault(phase, {"count": 0, "example_summaries": []})
        phase_stats["count"] += 1
        summary = attempt.get("summary")
        if isinstance(summary, str) and len(phase_stats["example_summaries"]) < 3:
            phase_stats["example_summaries"].append(summary)
    return {
        "count": sum(stats["count"] for stats in by_phase.values()),
        "by_phase": dict(sorted(by_phase.items())),
    }


def _stale_or_demoted_record(attempt: dict[str, Any]) -> dict[str, Any] | None:
    classification = attempt.get("classification")
    if not isinstance(classification, dict):
        classification = {}
    mode = classification.get("mode")
    reason = None
    message = None
    launch_readiness_exclusion = attempt.get("launch_readiness_exclusion")
    if mode in {"legacy", "malformed"}:
        reason = "legacy_summary_schema"
        message = "legacy or malformed summary schema"
    elif isinstance(launch_readiness_exclusion, dict):
        reason = "launch_readiness_exclusion"
        message = str(launch_readiness_exclusion.get("message", "demoted artifact"))
    else:
        return None

    record = {
        "summary": attempt.get("summary"),
        "reason": reason,
        "message": message,
        "operator_action": STALE_OR_DEMOTED_OPERATOR_ACTION,
        "lane": classification.get("lane"),
        "mode": mode,
        "claim_label": classification.get("claim_label"),
    }
    if isinstance(launch_readiness_exclusion, dict):
        record["launch_readiness_exclusion"] = launch_readiness_exclusion
    return record


def _stale_or_demoted_artifacts(
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for attempt in attempts:
        record = _stale_or_demoted_record(attempt)
        if record is None:
            continue
        records.append(record)
        if len(records) >= MAX_STALE_OR_DEMOTED_ARTIFACTS:
            break
    return records


def build_index(results_root: Path) -> dict[str, Any]:
    summary_paths = sorted(results_root.glob("*/summary.json"))
    attempts = []
    for path in summary_paths:
        try:
            attempts.append(_attempt_record(path, _load_summary(path)))
        except ValueError as exc:
            attempts.append(_malformed_summary_attempt(path, exc))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        grouped.setdefault(_group_key(attempt), []).append(attempt)

    groups = []
    for key, group_attempts in sorted(grouped.items()):
        groups.append(
            {
                "key": key,
                "count": len(group_attempts),
                "baseline_stats": _baseline_stats(group_attempts),
            }
        )

    diagnostic_or_failed = [
        attempt
        for attempt in attempts
        if not attempt["included_in_baseline_stats"]
        or attempt["classification"].get("mode") == "diagnostic"
    ]
    for attempt in diagnostic_or_failed:
        if "launch_readiness" in attempt:
            exclusion = _launch_readiness_exclusion(attempt)
            if exclusion is not None:
                attempt["launch_readiness_exclusion"] = exclusion
    launch_prerequisite_attempts = [
        attempt for attempt in attempts if _is_launch_ready_prerequisite(attempt)
    ]
    launch_ready_attempts = [
        attempt
        for attempt in launch_prerequisite_attempts
        if _is_launch_ready_attempt(attempt)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "results_root": str(results_root),
        "total_attempts": len(attempts),
        "baseline_stats": _baseline_stats(attempts),
        "groups": groups,
        "diagnostic_or_failed_attempts": diagnostic_or_failed,
        "stale_or_demoted_artifacts": _stale_or_demoted_artifacts(diagnostic_or_failed),
        "launch_readiness_exclusion_stats": _launch_readiness_exclusion_stats(
            diagnostic_or_failed
        ),
        "launch_prerequisite_attempts": launch_prerequisite_attempts,
        "launch_ready_attempts": launch_ready_attempts,
    }


def write_index(results_root: Path, output: Path) -> dict[str, Any]:
    index = build_index(results_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    tmp.replace(output)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("experiments/modded_nanogpt_b200/results"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main(*, enforce_rootfs: bool = False) -> int:
    args = parse_args()
    if enforce_rootfs:
        guard_exit = cli_guard.guard_rootfs_cli(
            "experiments/modded_nanogpt_b200/summarize.sh"
        )
        if guard_exit is not None:
            return guard_exit
    index = write_index(args.results_root, args.output)
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(enforce_rootfs=True))
