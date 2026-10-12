# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Promotion accounting for Falcon continuation gates."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


GDN_ADDITION_DECISION_SCHEMA_VERSION = 1
GDN_ADDITION_B200_HOUR_LIMIT = 2.0
GDN_ADDITION_REQUIRED_DRAWS = ("id", "ood", "challenge")
GDN_ADDITION_REQUIRED_SEEDS = (0, 1, 2)
MECHANISM_SCREEN_ELIGIBILITY_SCHEMA_VERSION = 1
MECHANISM_SCREEN_B200_HOUR_LIMIT = 1.0
MECHANISM_SCREEN_DISCRIMINABILITY_EPS = 1.0e-6
MECHANISM_SCREEN_ARMS = ("M0", "M1", "M2", "M3", "M4")
MECHANISM_SCREEN_EQUIVALENT_PAIRS = (("M2", "M4"),)
MECHANISM_SCREEN_REQUIRED_REGIONS = (
    "val_head_0000",
    "val_mid_10m",
    "val_mid_20m",
)
MECHANISM_SCREEN_REQUIRED_SHAPE = {
    "num_hidden_layers": 4,
    "hidden_size": 256,
    "num_heads": 8,
    "head_dim": 32,
    "seq_len": 512,
    "tokens_per_step": 16_384,
}


class GdnAdditionProjectionError(ValueError):
    """A GDN addition projection is missing required spend accounting."""


@dataclass(frozen=True)
class GdnAdditionProjectionInputs:
    free_bytes: int
    projected_peak_bytes: int
    retained_artifact_bytes: int
    reserve_bytes: int
    warmup_steps: int
    steady_state_step_seconds: list[float]
    per_draw_eval_seconds: dict[str, float]
    checkpoint_seconds_per_seed: float | None
    preflight_seconds: float | None
    failed_attempt_seconds: list[float] | None
    retry_seconds_per_logical_run: float | None
    training_steps_per_seed: int = 2_000
    required_seeds: list[int] | None = None
    covered_seeds: list[int] | None = None
    device_count_per_logical_run: int = 1


@dataclass(frozen=True)
class GdnAdditionDecision:
    schema_version: int
    record_type: str
    decision: str
    reason: str
    projected_b200_hours: float
    inputs: dict[str, Any]
    disk: dict[str, Any]
    arithmetic: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path


@dataclass(frozen=True)
class MechanismScreenEligibility:
    schema_version: int
    record_type: str
    decision: str
    reason: str
    projected_b200_hours: float
    checks: dict[str, Any]
    inputs: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path


def _require_finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GdnAdditionProjectionError(f"{field} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise GdnAdditionProjectionError(f"{field} must be a finite nonnegative number")
    return number


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GdnAdditionProjectionError(f"{field} must be an integer >= {minimum}")
    return value


def _validate_inputs(inputs: GdnAdditionProjectionInputs) -> None:
    _require_int(inputs.free_bytes, "free_bytes")
    _require_int(inputs.projected_peak_bytes, "projected_peak_bytes")
    _require_int(inputs.retained_artifact_bytes, "retained_artifact_bytes")
    _require_int(inputs.reserve_bytes, "reserve_bytes")
    _require_int(inputs.warmup_steps, "warmup_steps")
    _require_int(inputs.training_steps_per_seed, "training_steps_per_seed", minimum=1)
    _require_int(
        inputs.device_count_per_logical_run,
        "device wall time device_count_per_logical_run",
        minimum=1,
    )
    if not inputs.steady_state_step_seconds:
        raise GdnAdditionProjectionError(
            "steady-state step timing is required after warmup"
        )
    for sample in inputs.steady_state_step_seconds:
        _require_finite_nonnegative(sample, "steady-state step timing")
    if set(inputs.per_draw_eval_seconds) != set(GDN_ADDITION_REQUIRED_DRAWS):
        raise GdnAdditionProjectionError(
            "evaluation draw accounting must cover id, ood, and challenge"
        )
    for draw_id in GDN_ADDITION_REQUIRED_DRAWS:
        _require_finite_nonnegative(
            inputs.per_draw_eval_seconds[draw_id],
            f"evaluation draw {draw_id}",
        )
    _require_finite_nonnegative(
        inputs.checkpoint_seconds_per_seed,
        "checkpoint allowance",
    )
    _require_finite_nonnegative(inputs.preflight_seconds, "preflight time")
    if inputs.failed_attempt_seconds is None:
        raise GdnAdditionProjectionError("failure accounting is required")
    for sample in inputs.failed_attempt_seconds:
        _require_finite_nonnegative(sample, "failure accounting")
    _require_finite_nonnegative(
        inputs.retry_seconds_per_logical_run,
        "retry accounting",
    )
    required_seeds = tuple(
        GDN_ADDITION_REQUIRED_SEEDS
        if inputs.required_seeds is None
        else inputs.required_seeds
    )
    covered_seeds = tuple([] if inputs.covered_seeds is None else inputs.covered_seeds)
    if len(required_seeds) != 3 or set(required_seeds) != set(covered_seeds):
        raise GdnAdditionProjectionError(
            "three-seed coverage is required before authorizing spend"
        )


def decide_gdn_addition(
    inputs: GdnAdditionProjectionInputs,
) -> GdnAdditionDecision:
    """Return the B12 binary decision with reproducible summed-device arithmetic."""
    _validate_inputs(inputs)
    input_dict = asdict(inputs)
    if input_dict["required_seeds"] is None:
        input_dict["required_seeds"] = list(GDN_ADDITION_REQUIRED_SEEDS)

    seed_count = len(input_dict["required_seeds"])
    device_count = inputs.device_count_per_logical_run
    steady_state_mean = sum(inputs.steady_state_step_seconds) / len(
        inputs.steady_state_step_seconds
    )
    training_seconds = steady_state_mean * inputs.training_steps_per_seed * seed_count
    eval_seconds = sum(inputs.per_draw_eval_seconds.values()) * seed_count
    checkpoint_seconds = float(inputs.checkpoint_seconds_per_seed) * seed_count
    failure_seconds = sum(inputs.failed_attempt_seconds or [])
    retry_seconds = float(inputs.retry_seconds_per_logical_run) * seed_count
    host_wall_seconds = (
        training_seconds
        + eval_seconds
        + checkpoint_seconds
        + float(inputs.preflight_seconds)
        + failure_seconds
        + retry_seconds
    )
    total_seconds = host_wall_seconds * device_count
    projected_b200_hours = total_seconds / 3600.0
    disk_required = (
        inputs.projected_peak_bytes
        + inputs.retained_artifact_bytes
        + inputs.reserve_bytes
    )
    disk = {
        "free_bytes": inputs.free_bytes,
        "projected_peak_bytes": inputs.projected_peak_bytes,
        "retained_artifact_bytes": inputs.retained_artifact_bytes,
        "reserve_bytes": inputs.reserve_bytes,
        "required_bytes": disk_required,
        "fits_with_reserve": inputs.free_bytes >= disk_required,
    }
    arithmetic = {
        "steady_state_step_seconds_mean": steady_state_mean,
        "training_seconds": training_seconds,
        "eval_seconds": eval_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "preflight_seconds": float(inputs.preflight_seconds),
        "failure_seconds": failure_seconds,
        "retry_seconds": retry_seconds,
        "host_wall_seconds_before_device_multiplier": host_wall_seconds,
        "device_count_per_logical_run": float(device_count),
        "total_seconds": total_seconds,
        "projected_b200_hours": projected_b200_hours,
    }
    if projected_b200_hours > GDN_ADDITION_B200_HOUR_LIMIT:
        decision = "performance_omission"
        reason = (
            f"projected {projected_b200_hours:.6f} summed B200-hours exceeds "
            "2.0 summed B200-hours"
        )
    elif not disk["fits_with_reserve"]:
        decision = "performance_omission"
        reason = (
            "projected artifacts plus retained artifacts and reserve exceed "
            "available disk"
        )
    else:
        decision = "run"
        reason = "projected summed B200-hours are within the B12 limit"
    return GdnAdditionDecision(
        schema_version=GDN_ADDITION_DECISION_SCHEMA_VERSION,
        record_type="gdn_addition_decision",
        decision=decision,
        reason=reason,
        projected_b200_hours=projected_b200_hours,
        inputs=input_dict,
        disk=disk,
        arithmetic=arithmetic,
    )


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _nonnegative_number(value: Any) -> bool:
    return _finite_number(value) and float(value) >= 0.0


def _check_mechanism_arm_record(
    *,
    arm: str,
    record: Any,
    required_seed: int,
    required_shape: dict[str, int],
    required_regions: tuple[str, ...],
    required_probe_steps: tuple[int, ...],
) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    details: dict[str, Any] = {"ok": True}
    if not isinstance(record, dict):
        return False, [f"{arm} attempt record must be an object"], {"ok": False}

    if record.get("arm") != arm:
        reasons.append(f"{arm} attempt records the wrong arm")
    if record.get("seed") != required_seed:
        reasons.append(f"{arm} attempt does not use required seed {required_seed}")
    if record.get("shape") != required_shape:
        reasons.append(f"{arm} attempt is not at the exact science shape")
    if record.get("completed_steps") != record.get("requested_steps"):
        reasons.append(f"{arm} attempt did not complete all requested steps")
    completed_steps = record.get("completed_steps")
    if not isinstance(completed_steps, int) or completed_steps < 1:
        reasons.append(f"{arm} attempt must complete at least one step")

    attempt = record.get("attempt")
    if not isinstance(attempt, dict) or attempt.get("status") != "completed":
        reasons.append(f"{arm} attempt is not a completed native bundle")

    device = record.get("device")
    if not isinstance(device, dict):
        reasons.append(f"{arm} attempt is missing CUDA B200 device evidence")
    else:
        device_type = device.get("type")
        device_name = str(device.get("name", ""))
        device_uuid = device.get("uuid")
        if (
            device_type != "cuda"
            or "B200" not in device_name
            or not isinstance(device_uuid, str)
            or not device_uuid
        ):
            reasons.append(f"{arm} attempt must record a CUDA B200 device")

    clocks = record.get("clocks")
    if (
        not isinstance(clocks, dict)
        or not isinstance(clocks.get("started_utc"), str)
        or not clocks["started_utc"]
        or not isinstance(clocks.get("ended_utc"), str)
        or not clocks["ended_utc"]
    ):
        reasons.append(f"{arm} attempt is missing per-arm clocks")

    trajectory = record.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        reasons.append(f"{arm} trajectory is missing")
    else:
        for index, row in enumerate(trajectory):
            if not isinstance(row, dict):
                reasons.append(f"{arm} trajectory row {index} must be an object")
                continue
            for metric in ("loss", "grad_norm"):
                if not _finite_number(row.get(metric)):
                    reasons.append(f"{arm} trajectory has non-finite {metric}")
            if not _nonnegative_number(row.get("step_seconds")):
                reasons.append(f"{arm} trajectory has invalid step timing")

    fixed_eval = record.get("fixed_region_eval")
    regions = fixed_eval.get("regions") if isinstance(fixed_eval, dict) else None
    region_ids = (
        {
            row.get("region_id")
            for row in regions
            if isinstance(row, dict) and isinstance(row.get("region_id"), str)
        }
        if isinstance(regions, list)
        else set()
    )
    if region_ids != set(required_regions):
        reasons.append(f"{arm} fixed regions do not cover the required fixed regions")
    if isinstance(regions, list):
        for row in regions:
            if not isinstance(row, dict):
                reasons.append(f"{arm} fixed region result must be an object")
                continue
            if not _finite_number(row.get("region_ce")):
                reasons.append(f"{arm} fixed region result has non-finite CE")
            token_count = row.get("region_token_count")
            if not isinstance(token_count, int) or token_count < 1:
                reasons.append(f"{arm} fixed region token count is invalid")

    probes = record.get("probes")
    probe_steps = (
        {
            row.get("step")
            for row in probes
            if isinstance(row, dict) and isinstance(row.get("step"), int)
        }
        if isinstance(probes, list)
        else set()
    )
    if probe_steps != set(required_probe_steps):
        reasons.append(f"{arm} probes do not cover required probe steps")
    if isinstance(probes, list):
        for row in probes:
            if not isinstance(row, dict):
                reasons.append(f"{arm} probe must be an object")
                continue
            diagnostics = row.get("diagnostics")
            if (
                not isinstance(diagnostics, dict)
                or diagnostics.get("record_type") != "falcon_mechanism_diagnostics"
                or not diagnostics.get("layers")
            ):
                reasons.append(f"{arm} probe is missing mechanism diagnostics")

    checkpoint = record.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("kind") != "model_only_checkpoint"
        or checkpoint.get("step") != completed_steps
        or not isinstance(checkpoint.get("path"), str)
        or not checkpoint.get("path")
        or not isinstance(checkpoint.get("sha256"), str)
        or len(checkpoint.get("sha256", "")) != 64
    ):
        reasons.append(f"{arm} final model-only checkpoint evidence is invalid")

    details.update(
        {
            "ok": not reasons,
            "seed": record.get("seed"),
            "shape": record.get("shape"),
            "region_ids": sorted(region_ids),
            "probe_steps": sorted(probe_steps),
            "completed_steps": completed_steps,
            "device": device,
            "clocks": clocks,
        }
    )
    return not reasons, reasons, details


def _pair_key(pair: Any) -> tuple[str, str] | None:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    left, right = pair
    if not isinstance(left, str) or not isinstance(right, str) or left == right:
        return None
    return tuple(sorted((left, right)))


def _screen_projection_hours(
    projection: Any,
) -> tuple[float, list[str], dict[str, Any]]:
    reasons: list[str] = []
    if not isinstance(projection, dict):
        return math.inf, ["complete screen projection is missing"], {"ok": False}
    required_fields = (
        "logical_runs",
        "screen_b200_hours",
        "preflight_b200_hours",
        "probe_b200_hours",
        "fixed_region_eval_b200_hours",
        "checkpoint_b200_hours",
        "failure_retry_b200_hours",
        "safety_reserve_b200_hours",
    )
    total = 0.0
    components: dict[str, Any] = {}
    if projection.get("logical_runs") != 15:
        reasons.append("screen projection must account for all 15 logical runs")
    components["logical_runs"] = projection.get("logical_runs")
    for field in required_fields[1:]:
        value = projection.get(field)
        if not _nonnegative_number(value):
            reasons.append(f"{field} must be finite and nonnegative")
            continue
        components[field] = float(value)
        total += float(value)
    components["projected_b200_hours"] = total
    components["limit_b200_hours"] = MECHANISM_SCREEN_B200_HOUR_LIMIT
    components["ok"] = not reasons and total <= MECHANISM_SCREEN_B200_HOUR_LIMIT
    if total > MECHANISM_SCREEN_B200_HOUR_LIMIT:
        reasons.append(
            f"projected {total:.6f} summed B200-hours exceeds " "1.0 summed B200-hour"
        )
    return total, reasons, components


def decide_mechanism_screen_eligibility(
    evidence: dict[str, Any],
) -> MechanismScreenEligibility:
    """Return the M03 binary decision for launching the full M04 screen."""
    reasons: list[str] = []
    checks: dict[str, Any] = {}

    if not isinstance(evidence, dict):
        evidence = {}
        reasons.append("evidence bundle must be an object")
    if evidence.get("record_type") != "mechanism_screen_preflight":
        reasons.append("evidence bundle must be a mechanism_screen_preflight record")

    required_arms = tuple(evidence.get("required_arms", MECHANISM_SCREEN_ARMS))
    if required_arms != MECHANISM_SCREEN_ARMS:
        reasons.append("required arms must be exactly M0-M4")
    try:
        required_seed = int(evidence.get("required_seed", 0))
    except (TypeError, ValueError):
        required_seed = -1
        reasons.append("required seed must be an integer")

    raw_shape = evidence.get("required_shape", MECHANISM_SCREEN_REQUIRED_SHAPE)
    required_shape = dict(raw_shape) if isinstance(raw_shape, dict) else {}
    if required_shape != MECHANISM_SCREEN_REQUIRED_SHAPE:
        reasons.append("required shape does not match the exact science shape")

    raw_regions = evidence.get("required_regions", MECHANISM_SCREEN_REQUIRED_REGIONS)
    required_regions = tuple(raw_regions) if isinstance(raw_regions, list) else ()
    if set(required_regions) != set(MECHANISM_SCREEN_REQUIRED_REGIONS):
        reasons.append("required fixed regions must match the M03 registry")

    raw_probe_steps = evidence.get("required_probe_steps", ())
    required_probe_steps = (
        tuple(raw_probe_steps) if isinstance(raw_probe_steps, list) else ()
    )
    if not required_probe_steps:
        reasons.append("required probe steps are missing")

    attempts = evidence.get("attempts")
    arm_details: dict[str, Any] = {}
    if not isinstance(attempts, dict):
        reasons.append("attempt records are missing")
        attempts = {}
    missing_arms = set(MECHANISM_SCREEN_ARMS) - set(attempts)
    extra_arms = set(attempts) - set(MECHANISM_SCREEN_ARMS)
    if missing_arms or extra_arms:
        reasons.append(
            f"attempt records must cover M0-M4 missing={sorted(missing_arms)} "
            f"extra={sorted(extra_arms)}"
        )
    for arm in MECHANISM_SCREEN_ARMS:
        ok, arm_reasons, details = _check_mechanism_arm_record(
            arm=arm,
            record=attempts.get(arm),
            required_seed=required_seed,
            required_shape=MECHANISM_SCREEN_REQUIRED_SHAPE,
            required_regions=MECHANISM_SCREEN_REQUIRED_REGIONS,
            required_probe_steps=required_probe_steps,
        )
        arm_details[arm] = details
        if not ok:
            reasons.extend(arm_reasons)
    checks["arms"] = {
        "ok": not any(not item.get("ok") for item in arm_details.values()),
        "details": arm_details,
    }

    equivalent_pairs = {
        _pair_key(row.get("arms")): row
        for row in evidence.get("equivalent_arm_pairs", [])
        if isinstance(row, dict)
    }
    identity_reasons: list[str] = []
    for pair in MECHANISM_SCREEN_EQUIVALENT_PAIRS:
        key = tuple(sorted(pair))
        row = equivalent_pairs.get(key)
        if row is None:
            identity_reasons.append(f"{pair[0]}/{pair[1]} identity proof is missing")
            continue
        for field in ("max_abs_loss_diff", "max_abs_grad_norm_diff"):
            if (
                not _finite_number(row.get(field))
                or abs(float(row[field])) > MECHANISM_SCREEN_DISCRIMINABILITY_EPS
            ):
                identity_reasons.append(
                    f"{pair[0]}/{pair[1]} identity {field} exceeds tolerance"
                )
        if (
            not isinstance(row.get("same_input_digest"), str)
            or len(row["same_input_digest"]) != 64
        ):
            identity_reasons.append(
                f"{pair[0]}/{pair[1]} identity lacks same-input digest"
            )
    reasons.extend(identity_reasons)
    checks["identity"] = {
        "ok": not identity_reasons,
        "equivalent_pairs": [list(pair) for pair in MECHANISM_SCREEN_EQUIVALENT_PAIRS],
    }

    expected_pairs = {
        tuple(sorted((left, right)))
        for left in MECHANISM_SCREEN_ARMS
        for right in MECHANISM_SCREEN_ARMS
        if left < right
        and tuple(sorted((left, right)))
        not in {tuple(sorted(pair)) for pair in MECHANISM_SCREEN_EQUIVALENT_PAIRS}
    }
    observed_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for row in evidence.get("discriminability", []):
        if not isinstance(row, dict):
            continue
        key = _pair_key(row.get("arms"))
        if key is not None:
            observed_pairs[key] = row
    discrim_reasons: list[str] = []
    for pair in sorted(expected_pairs):
        row = observed_pairs.get(pair)
        if row is None:
            discrim_reasons.append(f"discriminability missing for {pair[0]}/{pair[1]}")
            continue
        max_loss = row.get("max_abs_loss_diff")
        max_grad = row.get("max_abs_grad_norm_diff")
        if not _finite_number(max_loss) or not _finite_number(max_grad):
            discrim_reasons.append(
                f"discriminability non-finite for {pair[0]}/{pair[1]}"
            )
            continue
        if (
            max(abs(float(max_loss)), abs(float(max_grad)))
            <= MECHANISM_SCREEN_DISCRIMINABILITY_EPS
        ):
            discrim_reasons.append(
                f"discriminability for {pair[0]}/{pair[1]} is not > 1e-6"
            )
    reasons.extend(discrim_reasons)
    checks["discriminability"] = {
        "ok": not discrim_reasons,
        "threshold": MECHANISM_SCREEN_DISCRIMINABILITY_EPS,
        "required_pairs": [list(pair) for pair in sorted(expected_pairs)],
    }

    disk = evidence.get("disk")
    disk_reasons: list[str] = []
    disk_details: dict[str, Any] = {"ok": False}
    if not isinstance(disk, dict):
        disk_reasons.append("disk accounting is missing")
    else:
        for field in (
            "free_bytes",
            "projected_peak_bytes",
            "retained_artifact_bytes",
            "reserve_bytes",
        ):
            value = disk.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                disk_reasons.append(f"disk {field} must be a nonnegative integer")
        if not disk_reasons:
            required_bytes = (
                disk["projected_peak_bytes"]
                + disk["retained_artifact_bytes"]
                + disk["reserve_bytes"]
            )
            disk_details = {
                **disk,
                "required_bytes": required_bytes,
                "fits_with_reserve": disk["free_bytes"] >= required_bytes,
                "ok": disk["free_bytes"] >= required_bytes,
            }
            if not disk_details["fits_with_reserve"]:
                disk_reasons.append("disk accounting does not fit with reserve")
    reasons.extend(disk_reasons)
    checks["disk"] = disk_details

    failure_accounting = evidence.get("failure_accounting")
    failure_reasons: list[str] = []
    if not isinstance(failure_accounting, dict):
        failure_reasons.append("retry/failure accounting is missing")
    else:
        failed_attempts = failure_accounting.get("failed_attempts")
        if not isinstance(failed_attempts, list):
            failure_reasons.append("retry/failure accounting requires failed_attempts")
        else:
            for row in failed_attempts:
                if not isinstance(row, dict) or not _nonnegative_number(
                    row.get("elapsed_sec", 0.0)
                ):
                    failure_reasons.append(
                        "retry/failure accounting has invalid failed attempt"
                    )
                    break
        retries = failure_accounting.get("retry_budget_attempts", 0)
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            failure_reasons.append("retry/failure accounting has invalid retry budget")
    reasons.extend(failure_reasons)
    checks["failure_accounting"] = {
        "ok": not failure_reasons,
        "details": failure_accounting if isinstance(failure_accounting, dict) else None,
    }

    omissions = evidence.get("omission_records", [])
    omission_reasons: list[str] = []
    if not isinstance(omissions, list):
        omission_reasons.append("omission records must be a list")
    elif omissions:
        omission_reasons.append("unresolved omission records block M04 eligibility")
    reasons.extend(omission_reasons)
    checks["omissions"] = {
        "ok": not omission_reasons,
        "count": len(omissions) if isinstance(omissions, list) else None,
    }

    (
        projected_b200_hours,
        projection_reasons,
        projection_details,
    ) = _screen_projection_hours(evidence.get("projection"))
    reasons.extend(projection_reasons)
    checks["compute"] = projection_details

    decision = "eligible" if not reasons else "ineligible"
    reason = (
        "M03 evidence satisfies the M04 mechanism screen gate"
        if decision == "eligible"
        else reasons[0]
    )
    return MechanismScreenEligibility(
        schema_version=MECHANISM_SCREEN_ELIGIBILITY_SCHEMA_VERSION,
        record_type="mechanism_screen_eligibility",
        decision=decision,
        reason=reason,
        projected_b200_hours=projected_b200_hours,
        checks=checks,
        inputs={
            "required_arms": list(MECHANISM_SCREEN_ARMS),
            "required_seed": required_seed,
            "required_regions": list(MECHANISM_SCREEN_REQUIRED_REGIONS),
            "required_probe_steps": list(required_probe_steps),
            "required_shape": dict(MECHANISM_SCREEN_REQUIRED_SHAPE),
        },
    )
