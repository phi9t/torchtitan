# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.modded_nanogpt_b200 import experiment_config


def _write_spec(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data) + "\n")


def _base_spec() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "gpu_ladder_prerequisite",
        "description": "B200 prerequisite ladder",
        "source": "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa",
        "data_manifest": "experiments/modded_nanogpt_b200/results/full_manifest/data_manifest.json",
        "result_root": "experiments/modded_nanogpt_b200/results",
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
            "expected_gpu_name": "B200",
            "verify_sha": True,
            "observability_profile": "byterobust",
        },
        "arms": [
            {
                "name": "g1_default",
                "num_gpus": 1,
            },
            {
                "name": "g2_pair_23",
                "num_gpus": 2,
                "gpu_ids": [2, 3],
                "tags": ["repeatability"],
            },
            {
                "name": "g8_compat",
                "experiment_kind": "b200_compatibility",
                "num_gpus": 8,
                "observability_profile": "eroica",
            },
        ],
    }


def test_load_spec_applies_defaults_and_canonicalizes_gpu_arms(tmp_path: Path):
    spec_path = tmp_path / "gpu_ladder.json"
    _write_spec(spec_path, _base_spec())

    spec = experiment_config.load_experiment_spec(spec_path)

    assert spec.schema_version == 1
    assert spec.experiment_id == "gpu_ladder_prerequisite"
    assert spec.execution_policy.rootfs_required is True
    assert [arm.name for arm in spec.arms] == [
        "g1_default",
        "g2_pair_23",
        "g8_compat",
    ]
    g1, g2, g8 = spec.arms
    assert g1.experiment_kind == "prerequisite"
    assert g1.legacy_lane == "B"
    assert g1.gpu_ids == (0,)
    assert g1.world_size == 1
    assert g1.grad_accum_steps == 8
    assert g1.observability_profile == "byterobust"
    assert g2.gpu_ids == (2, 3)
    assert g2.tags == ("repeatability",)
    assert g8.experiment_kind == "b200_compatibility"
    assert g8.legacy_lane == "B"
    assert g8.gpu_ids == tuple(range(8))
    assert g8.observability_profile == "eroica"


def test_legacy_lane_alias_maps_to_experiment_kind(tmp_path: Path):
    data = _base_spec()
    data["defaults"].pop("experiment_kind")
    data["defaults"]["lane"] = "A"
    data["defaults"].pop("legacy_lane")
    data["arms"] = [{"name": "upstream", "num_gpus": 8}]
    spec_path = tmp_path / "legacy.json"
    _write_spec(spec_path, data)

    spec = experiment_config.load_experiment_spec(spec_path)

    assert spec.arms[0].experiment_kind == "upstream_reproduction"
    assert spec.arms[0].legacy_lane == "A"


@pytest.mark.parametrize(
    ("arm_patch", "message"),
    [
        ({"num_gpus": 3}, "num_gpus must be one of"),
        ({"num_gpus": True}, "num_gpus must be an integer"),
        ({"num_gpus": 2, "gpu_ids": [0]}, "gpu_ids length"),
        ({"num_gpus": 2, "gpu_ids": [0, 0]}, "duplicate gpu_ids"),
        ({"num_gpus": 2, "gpu_ids": [0, True]}, "gpu_ids must contain integers"),
        ({"num_gpus": 2, "gpu_ids": [0, "1"]}, "gpu_ids must contain integers"),
        ({"num_gpus": 2, "gpu_ids": [-1, 0]}, "gpu_ids must be between 0 and 7"),
        ({"num_gpus": 2, "gpu_ids": [0, 8]}, "gpu_ids must be between 0 and 7"),
    ],
)
def test_rejects_invalid_gpu_ladder_arms(tmp_path: Path, arm_patch: dict, message: str):
    data = _base_spec()
    data["arms"] = [{"name": "bad", **arm_patch}]
    spec_path = tmp_path / "bad.json"
    _write_spec(spec_path, data)

    with pytest.raises(experiment_config.SchemaValidationError, match=message):
        experiment_config.load_experiment_spec(spec_path)


def test_rejects_boolean_schema_version(tmp_path: Path):
    data = _base_spec()
    data["schema_version"] = True
    spec_path = tmp_path / "bad_schema_version.json"
    _write_spec(spec_path, data)

    with pytest.raises(
        experiment_config.SchemaValidationError,
        match="schema_version must be an integer",
    ):
        experiment_config.load_experiment_spec(spec_path)


def test_rejects_duplicate_arm_names(tmp_path: Path):
    data = _base_spec()
    data["arms"] = [
        {"name": "dup", "num_gpus": 1},
        {"name": "dup", "num_gpus": 2},
    ]
    spec_path = tmp_path / "dups.json"
    _write_spec(spec_path, data)

    with pytest.raises(experiment_config.SchemaValidationError, match="duplicate arm"):
        experiment_config.load_experiment_spec(spec_path)


@pytest.mark.parametrize("arms", [[], {"name": "not-a-list"}])
def test_rejects_missing_or_non_list_arms(tmp_path: Path, arms):
    data = _base_spec()
    data["arms"] = arms
    spec_path = tmp_path / "bad_arms.json"
    _write_spec(spec_path, data)

    with pytest.raises(
        experiment_config.SchemaValidationError, match="arms must be a non-empty list"
    ):
        experiment_config.load_experiment_spec(spec_path)


@pytest.mark.parametrize(
    "field",
    ["source", "data_manifest", "result_root"],
)
def test_rejects_missing_top_level_path_fields(tmp_path: Path, field: str):
    data = _base_spec()
    data.pop(field)
    spec_path = tmp_path / f"missing_{field}.json"
    _write_spec(spec_path, data)

    with pytest.raises(
        experiment_config.SchemaValidationError,
        match=f"{field} must be a non-empty string",
    ):
        experiment_config.load_experiment_spec(spec_path)


@pytest.mark.parametrize(
    ("policy_patch", "message"),
    [
        ({"rootfs_required": "yes"}, "rootfs_required must be a boolean"),
        ({"sequential": "yes"}, "sequential must be a boolean"),
        ({"active_job_policy": "warn"}, "unsupported active_job_policy"),
    ],
)
def test_rejects_invalid_execution_policy_values(
    tmp_path: Path, policy_patch: dict, message: str
):
    data = _base_spec()
    data["execution_policy"].update(policy_patch)
    spec_path = tmp_path / "bad_policy.json"
    _write_spec(spec_path, data)

    with pytest.raises(experiment_config.SchemaValidationError, match=message):
        experiment_config.load_experiment_spec(spec_path)


def test_rejects_unsupported_mode(tmp_path: Path):
    data = _base_spec()
    data["defaults"]["mode"] = "benchmark"
    spec_path = tmp_path / "bad_mode.json"
    _write_spec(spec_path, data)

    with pytest.raises(
        experiment_config.SchemaValidationError, match="unsupported mode"
    ):
        experiment_config.load_experiment_spec(spec_path)


@pytest.mark.parametrize(
    ("default_patch", "message"),
    [
        ({"attention_backend": "sdpa"}, "unsupported attention_backend"),
        ({"mlp_backend": "cutlass"}, "unsupported mlp_backend"),
    ],
)
def test_rejects_unsupported_backend_values(
    tmp_path: Path, default_patch: dict, message: str
):
    data = _base_spec()
    data["defaults"].update(default_patch)
    spec_path = tmp_path / "bad_backend.json"
    _write_spec(spec_path, data)

    with pytest.raises(experiment_config.SchemaValidationError, match=message):
        experiment_config.load_experiment_spec(spec_path)


def test_rejects_unsupported_compile_policy(tmp_path: Path):
    data = _base_spec()
    data["defaults"]["compile_policy"] = "enable_compile"
    spec_path = tmp_path / "bad_compile_policy.json"
    _write_spec(spec_path, data)

    with pytest.raises(
        experiment_config.SchemaValidationError, match="unsupported compile_policy"
    ):
        experiment_config.load_experiment_spec(spec_path)


def test_classifies_supported_and_record_only_knobs(tmp_path: Path):
    data = _base_spec()
    data["arms"] = [
        {
            "name": "candidate",
            "num_gpus": 4,
            "model_config": {"n_layer": 20, "n_embd": 1280},
            "optimizer": {"name": "muon", "momentum": 0.95},
            "parallelism": {"tensor_parallel_degree": 2},
            "training_schedule": {
                "num_scheduled_iterations": 500,
                "stage_batch_sizes": [131072, 262144, 393216],
            },
        }
    ]
    spec_path = tmp_path / "knobs.json"
    _write_spec(spec_path, data)

    arm = experiment_config.load_experiment_spec(spec_path).arms[0]

    assert arm.knob_support["num_gpus"] == "supported"
    assert arm.knob_support["gpu_ids"] == "supported"
    assert arm.knob_support["model_config.n_layer"] == "record_only"
    assert arm.knob_support["model_config.n_embd"] == "record_only"
    assert arm.knob_support["optimizer.name"] == "record_only"
    assert arm.knob_support["optimizer.momentum"] == "record_only"
    assert arm.knob_support["parallelism.tensor_parallel_degree"] == "record_only"
    assert (
        arm.knob_support["training_schedule.num_scheduled_iterations"] == "record_only"
    )
    assert arm.has_record_only_knobs is True


def test_observability_profile_expands_features(tmp_path: Path):
    data = _base_spec()
    data["defaults"]["observability_profile"] = "mycroft"
    data["arms"] = [{"name": "timeline", "num_gpus": 1}]
    spec_path = tmp_path / "observability.json"
    _write_spec(spec_path, data)

    arm = experiment_config.load_experiment_spec(spec_path).arms[0]

    assert arm.observability.features["run_attempt_evidence"] == "observed"
    assert arm.observability.features["semantic_timeline"] == "advisory"
    assert arm.observability.features["active_diagnostic_planner"] == "unavailable"


def test_to_dict_returns_canonical_json_safe_schema(tmp_path: Path):
    spec_path = tmp_path / "gpu_ladder.json"
    _write_spec(spec_path, _base_spec())

    data = experiment_config.load_experiment_spec(spec_path).to_dict()

    assert data["schema_version"] == 1
    assert data["arms"][0]["experiment_kind"] == "prerequisite"
    assert data["arms"][0]["legacy_lane"] == "B"
    assert data["arms"][0]["gpu_ids"] == [0]
    assert data["arms"][0]["grad_accum_steps"] == 8
    assert data["arms"][0]["observability"]["profile"] == "byterobust"
    json.dumps(data, sort_keys=True)


def test_materialize_plan_derives_harness_values_for_gpu_arm(tmp_path: Path):
    spec_path = tmp_path / "gpu_ladder.json"
    data = _base_spec()
    data["arms"] = [
        {
            "name": "g4_candidate",
            "experiment_kind": "prerequisite",
            "num_gpus": 4,
            "gpu_ids": [0, 2, 4, 6],
            "mlp_backend": "torch",
            "compile_policy": "disabled_prerequisite",
        }
    ]
    _write_spec(spec_path, data)

    spec = experiment_config.load_experiment_spec(spec_path)
    plan = experiment_config.materialize_experiment_plan(spec)
    arm_plan = plan.arms[0]

    assert arm_plan.run_id == "gpu_ladder_prerequisite_g4_candidate"
    assert arm_plan.attempt_id == "gpu_ladder_prerequisite_g4_candidate_attempt_001"
    assert arm_plan.result_dir == Path(
        "experiments/modded_nanogpt_b200/results/gpu_ladder_prerequisite_g4_candidate"
    )
    assert arm_plan.legacy_lane == "B"
    assert arm_plan.claim_label == "B200 prerequisite torch-MLP fallback"
    assert arm_plan.claim_eligible is False
    assert arm_plan.visible_devices == "0,2,4,6"
    assert arm_plan.torchrun_nproc_per_node == 4
    assert arm_plan.preflight_expected_gpus == 4
    assert arm_plan.launch_env == {
        "CUDA_VISIBLE_DEVICES": "0,2,4,6",
        "NVIDIA_VISIBLE_DEVICES": "0,2,4,6",
    }
    assert arm_plan.to_dict()["observability"]["profile"] == "byterobust"
    json.dumps(plan.to_dict(), sort_keys=True)


def test_materialize_plan_marks_active_two_gpu_b200_compatibility_arm_claim_eligible(
    tmp_path: Path,
):
    spec_path = tmp_path / "gpu_ladder.json"
    data = _base_spec()
    data["defaults"]["experiment_kind"] = "b200_compatibility"
    data["defaults"]["mlp_backend"] = "triton"
    data["arms"] = [
        {
            "name": "g2_active",
            "num_gpus": 2,
            "gpu_ids": [0, 1],
        }
    ]
    _write_spec(spec_path, data)

    spec = experiment_config.load_experiment_spec(spec_path)
    arm_plan = experiment_config.materialize_experiment_plan(spec).arms[0]

    assert arm_plan.claim_label == "B200 compatibility patchset"
    assert arm_plan.claim_eligible is True


def test_materialize_plan_keeps_torch_mlp_fallback_claim_ineligible(tmp_path: Path):
    spec_path = tmp_path / "gpu_ladder.json"
    data = _base_spec()
    data["defaults"]["experiment_kind"] = "b200_compatibility"
    data["defaults"]["mlp_backend"] = "torch"
    data["arms"] = [
        {
            "name": "g2_torch_fallback",
            "num_gpus": 2,
            "gpu_ids": [0, 1],
        }
    ]
    _write_spec(spec_path, data)

    spec = experiment_config.load_experiment_spec(spec_path)
    arm_plan = experiment_config.materialize_experiment_plan(spec).arms[0]

    assert arm_plan.claim_label == "B200 prerequisite torch-MLP fallback"
    assert arm_plan.claim_eligible is False


def test_checked_in_gpu_ladder_config_tracks_current_safe_prerequisite_tuple():
    spec_path = Path(
        "experiments/modded_nanogpt_b200/configs/gpu_ladder_prerequisite.json"
    )

    spec = experiment_config.load_experiment_spec(spec_path)
    plan = experiment_config.materialize_experiment_plan(spec)

    assert (
        spec.data_manifest
        == "experiments/modded_nanogpt_b200/results/full_manifest_refresh_20260816T111021Z/data_manifest.json"
    )
    assert spec.execution_policy.rootfs_required is True
    assert spec.execution_policy.sequential is True
    assert spec.execution_policy.active_job_policy == "fail_if_active"
    assert spec.defaults["mode"] == "full"
    assert spec.defaults["experiment_kind"] == "prerequisite"
    assert spec.defaults["legacy_lane"] == "B"
    assert spec.defaults["attention_backend"] == "fa2"
    assert spec.defaults["mlp_backend"] == "triton"
    assert spec.defaults["verify_sha"] is True
    assert [arm.num_gpus for arm in spec.arms] == [1, 2, 4, 8]
    assert [arm.world_size for arm in spec.arms] == [1, 2, 4, 8]
    assert spec.arms[1].name == "g2_prerequisite"
    assert spec.arms[1].legacy_lane == "B"
    assert spec.arms[1].values["mode"] == "full"
    assert spec.arms[1].values["attention_backend"] == "fa2"
    assert spec.arms[1].values["mlp_backend"] == "triton"
    assert spec.arms[1].values["verify_sha"] is True
    assert plan.arms[1].visible_devices == "0,1"
    assert plan.arms[1].torchrun_nproc_per_node == 2
    assert plan.arms[1].preflight_expected_gpus == 2
    assert plan.arms[1].claim_label == "B200 compatibility patchset"
    assert plan.arms[1].claim_eligible is True
