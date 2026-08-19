# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_GPU_COUNTS = (1, 2, 4, 8)
SUPPORTED_KNOBS = {
    "attention_backend",
    "ce_compute_capability",
    "compile_policy",
    "data_manifest",
    "expected_gpu_name",
    "experiment_kind",
    "gpu_ids",
    "legacy_lane",
    "mlp_backend",
    "mode",
    "num_gpus",
    "observability_profile",
    "result_root",
    "source",
    "verify_sha",
}
RECORD_ONLY_TOP_LEVELS = {
    "batch_policy",
    "claim_policy",
    "model_config",
    "model_family",
    "optimizer",
    "parallelism",
    "preflight",
    "source_variant",
    "training_schedule",
    "validation",
}
EXPERIMENT_KIND_TO_LEGACY_LANE = {
    "upstream_reproduction": "A",
    "b200_compatibility": "B",
    "optimization_ablation": "C",
    "diagnostic": "B",
    "prerequisite": "B",
}
LEGACY_LANE_TO_EXPERIMENT_KIND = {
    "A": "upstream_reproduction",
    "B": "b200_compatibility",
    "C": "optimization_ablation",
}
OBSERVABILITY_PROFILES = {"tier0", "byterobust", "mycroft", "argus", "eroica"}


class SchemaValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ExecutionPolicy:
    rootfs_required: bool
    sequential: bool = True
    active_job_policy: str = "fail_if_active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rootfs_required": self.rootfs_required,
            "sequential": self.sequential,
            "active_job_policy": self.active_job_policy,
        }


@dataclass(frozen=True)
class Observability:
    profile: str
    features: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "features": dict(sorted(self.features.items())),
        }


@dataclass(frozen=True)
class ExperimentArm:
    name: str
    experiment_kind: str
    legacy_lane: str
    num_gpus: int
    gpu_ids: tuple[int, ...]
    world_size: int
    grad_accum_steps: int
    observability_profile: str
    observability: Observability
    tags: tuple[str, ...]
    knob_support: dict[str, str]
    values: dict[str, Any]

    @property
    def has_record_only_knobs(self) -> bool:
        return any(status == "record_only" for status in self.knob_support.values())

    def to_dict(self) -> dict[str, Any]:
        result = {
            key: _json_safe(value)
            for key, value in sorted(self.values.items())
            if key not in {"lane", "legacy_lane", "experiment_kind"}
        }
        result.update(
            {
                "name": self.name,
                "experiment_kind": self.experiment_kind,
                "legacy_lane": self.legacy_lane,
                "num_gpus": self.num_gpus,
                "gpu_ids": list(self.gpu_ids),
                "world_size": self.world_size,
                "grad_accum_steps": self.grad_accum_steps,
                "observability_profile": self.observability_profile,
                "observability": self.observability.to_dict(),
                "tags": list(self.tags),
                "knob_support": dict(sorted(self.knob_support.items())),
                "has_record_only_knobs": self.has_record_only_knobs,
            }
        )
        return result


@dataclass(frozen=True)
class ExperimentSpec:
    schema_version: int
    experiment_id: str
    description: str
    source: str
    data_manifest: str
    result_root: str
    execution_policy: ExecutionPolicy
    defaults: dict[str, Any]
    arms: tuple[ExperimentArm, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "source": self.source,
            "data_manifest": self.data_manifest,
            "result_root": self.result_root,
            "execution_policy": self.execution_policy.to_dict(),
            "defaults": _json_safe(self.defaults),
            "arms": [arm.to_dict() for arm in self.arms],
        }


@dataclass(frozen=True)
class ExperimentArmPlan:
    name: str
    experiment_kind: str
    legacy_lane: str
    run_id: str
    attempt_id: str
    result_dir: Path
    claim_label: str
    claim_eligible: bool
    visible_devices: str
    torchrun_nproc_per_node: int
    preflight_expected_gpus: int
    launch_env: dict[str, str]
    observability: Observability

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "experiment_kind": self.experiment_kind,
            "legacy_lane": self.legacy_lane,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "result_dir": str(self.result_dir),
            "claim_label": self.claim_label,
            "claim_eligible": self.claim_eligible,
            "visible_devices": self.visible_devices,
            "torchrun_nproc_per_node": self.torchrun_nproc_per_node,
            "preflight_expected_gpus": self.preflight_expected_gpus,
            "launch_env": dict(sorted(self.launch_env.items())),
            "observability": self.observability.to_dict(),
        }


@dataclass(frozen=True)
class ExperimentPlan:
    schema_version: int
    experiment_id: str
    arms: tuple[ExperimentArmPlan, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "arms": [arm.to_dict() for arm in self.arms],
        }


def load_experiment_spec(path: Path) -> ExperimentSpec:
    with path.open() as fp:
        raw = json.load(fp)
    if not isinstance(raw, dict):
        raise SchemaValidationError("experiment spec must be a JSON object")

    schema_version = _require_int(raw, "schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise SchemaValidationError(
            f"unsupported schema_version {schema_version}; expected {SUPPORTED_SCHEMA_VERSION}"
        )

    execution_policy = _parse_execution_policy(raw.get("execution_policy"))
    if not execution_policy.rootfs_required:
        raise SchemaValidationError(
            "full training experiment specs require rootfs_required"
        )

    defaults = _require_mapping(raw, "defaults")
    arms_raw = raw.get("arms")
    if not isinstance(arms_raw, list) or not arms_raw:
        raise SchemaValidationError("arms must be a non-empty list")

    seen_names: set[str] = set()
    arms: list[ExperimentArm] = []
    for raw_arm in arms_raw:
        if not isinstance(raw_arm, dict):
            raise SchemaValidationError("each arm must be a JSON object")
        name = _require_str(raw_arm, "name")
        if name in seen_names:
            raise SchemaValidationError(f"duplicate arm name: {name}")
        seen_names.add(name)
        arms.append(
            _parse_arm(
                raw_arm=raw_arm,
                defaults=defaults,
                spec_source=str(raw["source"]),
                spec_data_manifest=str(raw["data_manifest"]),
                spec_result_root=str(raw["result_root"]),
            )
        )

    return ExperimentSpec(
        schema_version=schema_version,
        experiment_id=_require_str(raw, "experiment_id"),
        description=_require_str(raw, "description"),
        source=_require_str(raw, "source"),
        data_manifest=_require_str(raw, "data_manifest"),
        result_root=_require_str(raw, "result_root"),
        execution_policy=execution_policy,
        defaults=_canonicalize_mapping(defaults),
        arms=tuple(arms),
    )


def materialize_experiment_plan(spec: ExperimentSpec) -> ExperimentPlan:
    return ExperimentPlan(
        schema_version=spec.schema_version,
        experiment_id=spec.experiment_id,
        arms=tuple(_materialize_arm(spec, arm) for arm in spec.arms),
    )


def _materialize_arm(spec: ExperimentSpec, arm: ExperimentArm) -> ExperimentArmPlan:
    run_id = f"{spec.experiment_id}_{arm.name}"
    visible_devices = ",".join(str(gpu_id) for gpu_id in arm.gpu_ids)
    return ExperimentArmPlan(
        name=arm.name,
        experiment_kind=arm.experiment_kind,
        legacy_lane=arm.legacy_lane,
        run_id=run_id,
        attempt_id=f"{run_id}_attempt_001",
        result_dir=Path(spec.result_root) / run_id,
        claim_label=_claim_label(
            arm.legacy_lane,
            str(arm.values.get("mode", "full")),
            str(arm.values.get("attention_backend", "fa3")),
            str(arm.values.get("mlp_backend", "triton")),
        ),
        claim_eligible=_claim_eligible(arm),
        visible_devices=visible_devices,
        torchrun_nproc_per_node=arm.world_size,
        preflight_expected_gpus=arm.world_size,
        launch_env={
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "NVIDIA_VISIBLE_DEVICES": visible_devices,
        },
        observability=arm.observability,
    )


def _claim_label(
    legacy_lane: str, mode: str, attention_backend: str, mlp_backend: str
) -> str:
    if mode == "smoke":
        return "smoke"
    if mode == "diagnostic":
        return "diagnostic"
    if legacy_lane == "A":
        return "B200 upstream reproduction"
    if mode == "full" and legacy_lane == "B" and mlp_backend == "torch":
        return "B200 prerequisite torch-MLP fallback"
    if legacy_lane == "B" and attention_backend == "fa2":
        return "B200 compatibility patchset"
    return "B200 systems-only"


def _claim_eligible(arm: ExperimentArm) -> bool:
    mode = str(arm.values.get("mode", "full"))
    return (
        arm.experiment_kind in {"upstream_reproduction", "b200_compatibility"}
        and arm.world_size == 8
        and mode == "full"
    )


def _parse_execution_policy(raw: Any) -> ExecutionPolicy:
    if not isinstance(raw, dict):
        raise SchemaValidationError("execution_policy must be a JSON object")
    return ExecutionPolicy(
        rootfs_required=bool(raw.get("rootfs_required")),
        sequential=bool(raw.get("sequential", True)),
        active_job_policy=str(raw.get("active_job_policy", "fail_if_active")),
    )


def _parse_arm(
    *,
    raw_arm: dict[str, Any],
    defaults: dict[str, Any],
    spec_source: str,
    spec_data_manifest: str,
    spec_result_root: str,
) -> ExperimentArm:
    values = {
        "source": spec_source,
        "data_manifest": spec_data_manifest,
        "result_root": spec_result_root,
        **defaults,
        **raw_arm,
    }
    experiment_kind, legacy_lane = _canonical_experiment_identity(values)
    values["experiment_kind"] = experiment_kind
    values["legacy_lane"] = legacy_lane

    num_gpus = _require_int(values, "num_gpus")
    if num_gpus not in SUPPORTED_GPU_COUNTS:
        raise SchemaValidationError(
            f"num_gpus must be one of {SUPPORTED_GPU_COUNTS}; got {num_gpus}"
        )
    if 8 % num_gpus != 0:
        raise SchemaValidationError(f"num_gpus must divide 8; got {num_gpus}")

    gpu_ids_raw = values.get("gpu_ids", list(range(num_gpus)))
    gpu_ids = _canonical_gpu_ids(gpu_ids_raw, num_gpus)
    values["gpu_ids"] = list(gpu_ids)
    values["world_size"] = num_gpus
    values["grad_accum_steps"] = 8 // num_gpus

    observability_profile = str(values.get("observability_profile", "tier0"))
    if observability_profile not in OBSERVABILITY_PROFILES:
        raise SchemaValidationError(
            f"unsupported observability_profile: {observability_profile}"
        )

    knob_support = _classify_knobs(raw_arm)
    for key in ("num_gpus", "gpu_ids"):
        knob_support.setdefault(key, "supported")

    tags_raw = values.get("tags", ())
    if not isinstance(tags_raw, (list, tuple)):
        raise SchemaValidationError("tags must be a list")

    return ExperimentArm(
        name=str(values["name"]),
        experiment_kind=experiment_kind,
        legacy_lane=legacy_lane,
        num_gpus=num_gpus,
        gpu_ids=gpu_ids,
        world_size=num_gpus,
        grad_accum_steps=8 // num_gpus,
        observability_profile=observability_profile,
        observability=Observability(
            profile=observability_profile,
            features=_observability_features(observability_profile),
        ),
        tags=tuple(str(tag) for tag in tags_raw),
        knob_support=knob_support,
        values=_canonicalize_mapping(values),
    )


def _canonical_experiment_identity(values: dict[str, Any]) -> tuple[str, str]:
    legacy_lane = values.get("legacy_lane", values.get("lane"))
    experiment_kind = values.get("experiment_kind")

    if experiment_kind is None:
        if legacy_lane is None:
            raise SchemaValidationError("experiment_kind is required")
        lane = str(legacy_lane)
        if lane not in LEGACY_LANE_TO_EXPERIMENT_KIND:
            raise SchemaValidationError(f"unsupported legacy lane: {lane}")
        experiment_kind = LEGACY_LANE_TO_EXPERIMENT_KIND[lane]
    experiment_kind = str(experiment_kind)
    if experiment_kind not in EXPERIMENT_KIND_TO_LEGACY_LANE:
        raise SchemaValidationError(f"unsupported experiment_kind: {experiment_kind}")

    if legacy_lane is None:
        legacy_lane = EXPERIMENT_KIND_TO_LEGACY_LANE[experiment_kind]
    legacy_lane = str(legacy_lane)
    if legacy_lane not in LEGACY_LANE_TO_EXPERIMENT_KIND:
        raise SchemaValidationError(f"unsupported legacy lane: {legacy_lane}")
    return experiment_kind, legacy_lane


def _canonical_gpu_ids(raw: Any, num_gpus: int) -> tuple[int, ...]:
    if not isinstance(raw, list):
        raise SchemaValidationError("gpu_ids must be a list")
    if len(raw) != num_gpus:
        raise SchemaValidationError(
            f"gpu_ids length must equal num_gpus={num_gpus}; got {len(raw)}"
        )
    gpu_ids = tuple(int(value) for value in raw)
    if len(set(gpu_ids)) != len(gpu_ids):
        raise SchemaValidationError(f"duplicate gpu_ids are not allowed: {gpu_ids}")
    return gpu_ids


def _classify_knobs(raw_arm: dict[str, Any]) -> dict[str, str]:
    support: dict[str, str] = {}
    for key, value in raw_arm.items():
        if key == "name":
            continue
        if key in SUPPORTED_KNOBS:
            support[key] = "supported"
        elif key in RECORD_ONLY_TOP_LEVELS:
            _flatten_record_only(key, value, support)
        else:
            support[key] = "record_only"
    return dict(sorted(support.items()))


def _flatten_record_only(prefix: str, value: Any, support: dict[str, str]) -> None:
    if isinstance(value, dict) and value:
        for key, nested in value.items():
            _flatten_record_only(f"{prefix}.{key}", nested, support)
        return
    support[prefix] = "record_only"


def _observability_features(profile: str) -> dict[str, str]:
    features = {
        "active_diagnostic_planner": "unavailable",
        "active_job_scan": "observed",
        "artifact_sizes": "observed",
        "blocker_classification": "unavailable",
        "command_env": "observed",
        "dcgm_availability": "observed",
        "evaluation_rsi_report": "unavailable",
        "lineage": "unavailable",
        "nccl_evidence": "unavailable",
        "no_output_progress_probe": "unavailable",
        "process_disk_gpu_watchers": "observed",
        "rootfs_sentinel": "observed",
        "run_attempt_evidence": "observed",
        "semantic_timeline": "unavailable",
        "stop_snapshot": "unavailable",
        "structured_log_parsing": "observed",
    }
    if profile in {"byterobust", "mycroft", "argus", "eroica"}:
        features.update(
            {
                "blocker_classification": "observed",
                "lineage": "observed",
                "nccl_evidence": "observed",
                "no_output_progress_probe": "observed",
                "stop_snapshot": "observed",
            }
        )
    if profile in {"mycroft", "argus", "eroica"}:
        features["semantic_timeline"] = "advisory"
    if profile in {"argus", "eroica"}:
        features["active_diagnostic_planner"] = "advisory"
    if profile == "eroica":
        features["evaluation_rsi_report"] = "advisory"
    return dict(sorted(features.items()))


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{key} must be a JSON object")
    return dict(value)


def _require_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise SchemaValidationError(f"{key} must be a non-empty string")
    return value


def _require_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise SchemaValidationError(f"{key} must be an integer")
    return value


def _canonicalize_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_safe(value) for key, value in sorted(raw.items())}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(nested) for key, nested in sorted(value.items())}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
