# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import json

import pytest


M03_SHAPE = {
    "num_hidden_layers": 4,
    "hidden_size": 256,
    "num_heads": 8,
    "head_dim": 32,
    "seq_len": 512,
    "tokens_per_step": 16_384,
}
M03_REGIONS = ("val_head_0000", "val_mid_10m", "val_mid_20m")
M03_PROBES = (0, 1, 2, 3)
M03_ARMS = ("M0", "M1", "M2", "M3", "M4")


def _mechanism_arm_record(arm: str, *, seed: int = 0, loss_offset: float = 0.0):
    trajectory = [
        {
            "step": step,
            "loss": 4.0 + loss_offset + step * 0.01,
            "grad_norm": 1.0 + loss_offset + step * 0.02,
            "step_seconds": 0.5,
        }
        for step in (1, 2, 3)
    ]
    return {
        "arm": arm,
        "seed": seed,
        "shape": dict(M03_SHAPE),
        "requested_steps": 3,
        "completed_steps": 3,
        "trajectory": trajectory,
        "fixed_region_eval": {
            "registry_digest": "a" * 64,
            "regions": [
                {
                    "region_id": region,
                    "region_ce": 5.0 + loss_offset,
                    "region_token_count": 16_384,
                }
                for region in M03_REGIONS
            ],
            "draw_ids": list(M03_REGIONS),
        },
        "probes": [
            {
                "step": step,
                "probe_batch_digest": "b" * 64,
                "diagnostics": {
                    "schema_version": 1,
                    "record_type": "falcon_mechanism_diagnostics",
                    "layers": [{"layer": 0, "heads": [{"head": 0}]}],
                },
            }
            for step in M03_PROBES
        ],
        "checkpoint": {
            "path": f"experiments/falcon/results/mechanism_preflight/{arm}.pt",
            "kind": "model_only_checkpoint",
            "step": 3,
            "sha256": "c" * 64,
        },
        "attempt": {
            "status": "completed",
            "attempt_id": f"mechanism-preflight-{arm}-seed{seed}",
            "failure_reason": None,
        },
        "device": {
            "type": "cuda",
            "index": 0,
            "name": "NVIDIA B200",
            "uuid": "GPU-" + ("0" * 32),
        },
        "clocks": {
            "started_utc": "2026-09-08T00:00:00Z",
            "ended_utc": "2026-09-08T00:00:01Z",
        },
    }


def _mechanism_bundle(**overrides):
    records = {
        arm: _mechanism_arm_record(arm, loss_offset=index * 1.0e-3)
        for index, arm in enumerate(M03_ARMS)
    }
    # M2 and M4 are intentionally equivalent under the M01 theorem when run on
    # the same input and must carry an explicit identity proof instead of a
    # discriminability requirement.
    records["M4"]["trajectory"] = copy.deepcopy(records["M2"]["trajectory"])
    records["M4"]["fixed_region_eval"] = copy.deepcopy(
        records["M2"]["fixed_region_eval"]
    )
    records["M4"]["probes"] = copy.deepcopy(records["M2"]["probes"])
    bundle = {
        "schema_version": 1,
        "record_type": "mechanism_screen_preflight",
        "required_arms": list(M03_ARMS),
        "required_seed": 0,
        "required_regions": list(M03_REGIONS),
        "required_probe_steps": list(M03_PROBES),
        "required_shape": dict(M03_SHAPE),
        "attempts": records,
        "equivalent_arm_pairs": [
            {
                "arms": ["M2", "M4"],
                "max_abs_loss_diff": 0.0,
                "max_abs_grad_norm_diff": 0.0,
                "same_input_digest": "d" * 64,
            }
        ],
        "discriminability": [
            {
                "arms": [left, right],
                "max_abs_loss_diff": 2.0e-6,
                "max_abs_grad_norm_diff": 0.0,
            }
            for left in M03_ARMS
            for right in M03_ARMS
            if left < right and {left, right} != {"M2", "M4"}
        ],
        "disk": {
            "free_bytes": 10_000,
            "projected_peak_bytes": 1_000,
            "retained_artifact_bytes": 1_000,
            "reserve_bytes": 1_000,
        },
        "projection": {
            "logical_runs": 15,
            "screen_b200_hours": 0.94,
            "preflight_b200_hours": 0.01,
            "probe_b200_hours": 0.01,
            "fixed_region_eval_b200_hours": 0.01,
            "checkpoint_b200_hours": 0.01,
            "failure_retry_b200_hours": 0.01,
            "safety_reserve_b200_hours": 0.01,
        },
        "failure_accounting": {"failed_attempts": [], "retry_budget_attempts": 1},
        "omission_records": [],
    }
    bundle.update(overrides)
    return bundle


def _eligibility_decision(bundle=None):
    from torchtitan.experiments.falcon.promotion import (
        decide_mechanism_screen_eligibility,
        MechanismScreenEligibility,
    )

    decision = decide_mechanism_screen_eligibility(
        _mechanism_bundle() if bundle is None else bundle
    )
    assert isinstance(decision, MechanismScreenEligibility)
    return decision


def _projection_inputs(**overrides):
    from torchtitan.experiments.falcon.promotion import GdnAdditionProjectionInputs

    values = {
        "free_bytes": 10_000,
        "projected_peak_bytes": 4_000,
        "retained_artifact_bytes": 1_000,
        "reserve_bytes": 2_000,
        "warmup_steps": 20,
        "steady_state_step_seconds": [1.0],
        "training_steps_per_seed": 2_000,
        "required_seeds": [0, 1, 2],
        "covered_seeds": [0, 1, 2],
        "per_draw_eval_seconds": {"id": 10.0, "ood": 10.0, "challenge": 10.0},
        "checkpoint_seconds_per_seed": 5.0,
        "preflight_seconds": 15.0,
        "failed_attempt_seconds": [],
        "retry_seconds_per_logical_run": 0.0,
        "device_count_per_logical_run": 1,
    }
    values.update(overrides)
    return GdnAdditionProjectionInputs(**values)


def test_gdn_addition_projection_allows_exactly_two_b200_hours():
    # Break caught: the promotion boundary is inclusive at exactly 2.0 summed
    # B200-hours; changing this to a strict less-than would waste a valid run.
    from torchtitan.experiments.falcon.promotion import decide_gdn_addition

    inputs = _projection_inputs(
        steady_state_step_seconds=[1.19],
        per_draw_eval_seconds={"id": 5.0, "ood": 5.0, "challenge": 5.0},
        checkpoint_seconds_per_seed=0.0,
        preflight_seconds=15.0,
    )

    decision = decide_gdn_addition(inputs)

    assert decision.decision == "run"
    assert decision.projected_b200_hours == pytest.approx(2.0)
    assert decision.arithmetic["total_seconds"] == pytest.approx(7200.0)


def test_gdn_addition_projection_omits_anything_above_two_b200_hours():
    # Break caught: a projection of 2.000001 B200-hours is not "close enough"
    # to authorize the B13 GDN matrix.
    from torchtitan.experiments.falcon.promotion import decide_gdn_addition

    inputs = _projection_inputs(
        steady_state_step_seconds=[1.1900006],
        per_draw_eval_seconds={"id": 5.0, "ood": 5.0, "challenge": 5.0},
        checkpoint_seconds_per_seed=0.0,
        preflight_seconds=15.0,
    )

    decision = decide_gdn_addition(inputs)

    assert decision.decision == "performance_omission"
    assert decision.projected_b200_hours > 2.0
    assert "exceeds 2.0 summed B200-hours" in decision.reason


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    (
        ("steady_state_step_seconds", [], "steady-state"),
        ("per_draw_eval_seconds", {"id": 1.0, "ood": 1.0}, "evaluation draw"),
        ("checkpoint_seconds_per_seed", None, "checkpoint"),
        ("preflight_seconds", None, "preflight"),
        ("failed_attempt_seconds", None, "failure accounting"),
    ),
)
def test_gdn_addition_projection_rejects_missing_required_accounting(
    field, bad_value, message
):
    # Break caught: no missing accounting bucket may default to zero and
    # accidentally authorize spend.
    from torchtitan.experiments.falcon.promotion import (
        decide_gdn_addition,
        GdnAdditionProjectionError,
    )

    kwargs = {field: bad_value}
    with pytest.raises(GdnAdditionProjectionError, match=message):
        decide_gdn_addition(_projection_inputs(**kwargs))


def test_gdn_addition_projection_requires_three_seed_coverage():
    # Break caught: one successful seed is not evidence for the B13 matrix.
    from torchtitan.experiments.falcon.promotion import (
        decide_gdn_addition,
        GdnAdditionProjectionError,
    )

    with pytest.raises(GdnAdditionProjectionError, match="three-seed coverage"):
        decide_gdn_addition(_projection_inputs(covered_seeds=[0, 1]))


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    (
        ("retry_seconds_per_logical_run", None, "retry accounting"),
        ("retry_seconds_per_logical_run", -1.0, "retry accounting"),
        ("device_count_per_logical_run", 0, "device wall time"),
    ),
)
def test_gdn_addition_projection_rejects_invalid_retry_or_device_accounting(
    field, bad_value, message
):
    # Break caught: retries and device count are cost multipliers, so invalid
    # values cannot be coerced into a lower projection.
    from torchtitan.experiments.falcon.promotion import (
        decide_gdn_addition,
        GdnAdditionProjectionError,
    )

    with pytest.raises(GdnAdditionProjectionError, match=message):
        decide_gdn_addition(_projection_inputs(**{field: bad_value}))


def test_gdn_addition_decision_is_machine_readable_with_exact_arithmetic():
    # Break caught: downstream issue/artifact freezing needs a stable dict with
    # every projection input and the exact arithmetic components.
    from torchtitan.experiments.falcon.promotion import decide_gdn_addition

    inputs = _projection_inputs(steady_state_step_seconds=[0.25, 0.5, 0.75])
    original = copy.deepcopy(inputs)

    decision = decide_gdn_addition(inputs)
    payload = decision.to_dict()

    assert inputs == original
    assert payload["schema_version"] == 1
    assert payload["record_type"] == "gdn_addition_decision"
    assert payload["inputs"]["warmup_steps"] == 20
    assert payload["inputs"]["steady_state_step_seconds"] == [0.25, 0.5, 0.75]
    assert payload["arithmetic"]["steady_state_step_seconds_mean"] == pytest.approx(0.5)
    assert payload["arithmetic"]["training_seconds"] == pytest.approx(3000.0)
    assert payload["arithmetic"]["retry_seconds"] == pytest.approx(0.0)
    assert payload["disk"]["fits_with_reserve"] is True


def test_campaign_driver_builds_projection_inputs_from_preflight_and_disk():
    # Break caught: launch code must not omit measured preflight timing or the
    # retry/failure accounting fields that promotion requires.
    from experiments.falcon.campaign_driver import build_projection_inputs

    preflight = {
        "completed_steps": 100,
        "warmup_steps": 20,
        "steady_state_step_seconds": [1.0, 1.2, 1.4],
        "eval_seconds_by_draw": {"id": 2.0, "ood": 3.0, "challenge": 5.0},
        "checkpoint_seconds": 7.0,
        "elapsed_sec": 140.0,
    }
    disk = {
        "free_bytes": 20_000,
        "projected_peak_bytes": 4_000,
        "retained_artifact_bytes": 1_000,
        "reserve_bytes": 2_000,
    }

    inputs = build_projection_inputs(preflight, disk)

    assert inputs.warmup_steps == 20
    assert inputs.steady_state_step_seconds == [1.0, 1.2, 1.4]
    assert inputs.per_draw_eval_seconds == {"id": 2.0, "ood": 3.0, "challenge": 5.0}
    assert inputs.checkpoint_seconds_per_seed == 7.0
    assert inputs.preflight_seconds == 140.0
    assert inputs.failed_attempt_seconds == []
    assert inputs.retry_seconds_per_logical_run == pytest.approx(
        (1.2 * 2_000) + 10.0 + 7.0
    )


def test_addition_preflight_record_uses_native_lineage_value_types(tmp_path):
    # Break caught: native evidence lineage maps allow only strings or null, so
    # eval_draws cannot be stored as a JSON list in the manifest record.
    from experiments.falcon.campaign_driver import _attempt_record
    from torchtitan.experiments.falcon.evidence import write_native_attempt_bundle

    record = _attempt_record(
        attempt_id="b12-gdn-addition-preflight-A1-seed0-test",
        arm="A1",
        seed=0,
        started_utc="2026-09-08T00:00:00Z",
        ended_utc="2026-09-08T00:00:01Z",
        steps=100,
        completed_steps=100,
        artifacts=[
            {
                "path": "experiments/falcon/results/addition_preflight/preflight.json",
                "kind": "addition_preflight_accounting",
            },
            {
                "path": "experiments/falcon/results/addition_preflight/decision.json",
                "kind": "gdn_addition_decision",
            },
            {
                "path": "experiments/falcon/results/addition_preflight/manifest.json",
                "kind": "addition_split_manifest",
            },
            {
                "path": "experiments/falcon/results/addition_preflight/checkpoint.pt",
                "kind": "addition_preflight_checkpoint",
            },
        ],
        projection={
            "decision": "performance_omission",
            "projected_b200_hours": 2.1,
            "inputs": {"split_manifest_digest": "a" * 64},
        },
    )

    assert record["data_lineage"]["eval_draws"] == "id,ood,challenge"
    write_native_attempt_bundle(tmp_path / "evidence", record)


def _write_existing_preflight_artifacts(tmp_path):
    from experiments.falcon.campaign_driver import build_projection_inputs
    from torchtitan.experiments.falcon.addition import AdditionSplitManifest
    from torchtitan.experiments.falcon.promotion import decide_gdn_addition

    artifact_dir = tmp_path / "b12-gdn-addition-preflight-A1-seed0-test"
    artifact_dir.mkdir()
    manifest = AdditionSplitManifest.generate(seed=0)
    manifest.write_json(artifact_dir / "split_manifest.json")
    (artifact_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    preflight = {
        "arm": "A1",
        "seed": 0,
        "requested_steps": 100,
        "completed_steps": 100,
        "warmup_steps": 20,
        "steady_state_step_seconds": [0.1, 0.2],
        "eval_seconds_by_draw": {"id": 0.01, "ood": 0.02, "challenge": 0.03},
        "checkpoint_seconds": 0.04,
        "elapsed_sec": 10.0,
        "split_manifest_digest": manifest.digest,
        "failed_attempt_seconds": [],
    }
    (artifact_dir / "preflight.json").write_text(json.dumps(preflight))
    decision_inputs = build_projection_inputs(
        preflight,
        {
            "free_bytes": 10_000,
            "projected_peak_bytes": 1_000,
            "retained_artifact_bytes": 1_000,
            "reserve_bytes": 1_000,
        },
    )
    decision = decide_gdn_addition(decision_inputs).to_dict()
    decision["inputs"]["split_manifest_digest"] = manifest.digest
    (artifact_dir / "gdn_addition_decision.json").write_text(json.dumps(decision))
    return artifact_dir


def test_publish_existing_addition_preflight_requires_rootfs(tmp_path, monkeypatch):
    # Break caught: the no-rerun publisher writes evidence, so direct host-side
    # Python invocation must fail closed like the training entrypoint.
    from experiments.falcon.campaign_driver import publish_existing_addition_preflight

    artifact_dir = _write_existing_preflight_artifacts(tmp_path)
    monkeypatch.delenv("TORCHTITAN_IN_ROOTFS", raising=False)

    with pytest.raises(RuntimeError, match="bwrap rootfs"):
        publish_existing_addition_preflight(
            artifact_dir=artifact_dir,
            results_root=tmp_path / "evidence",
        )


def test_publish_existing_addition_preflight_rejects_stale_artifacts(
    tmp_path, monkeypatch
):
    # Break caught: publishing a recovered measured run must rebind the native
    # bundle to its manifest and recomputed projection arithmetic.
    from experiments.falcon.campaign_driver import publish_existing_addition_preflight

    artifact_dir = _write_existing_preflight_artifacts(tmp_path)
    decision_path = artifact_dir / "gdn_addition_decision.json"
    decision = json.loads(decision_path.read_text())
    decision["inputs"]["split_manifest_digest"] = "0" * 64
    decision_path.write_text(json.dumps(decision))
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")

    with pytest.raises(ValueError, match="split manifest digest"):
        publish_existing_addition_preflight(
            artifact_dir=artifact_dir,
            results_root=tmp_path / "evidence",
        )


def test_publish_existing_addition_preflight_is_idempotent(tmp_path, monkeypatch):
    # Break caught: a recovered artifact may be republished during verification;
    # existing immutable clocks must be reused rather than regenerated.
    from experiments.falcon.campaign_driver import publish_existing_addition_preflight

    artifact_dir = _write_existing_preflight_artifacts(tmp_path)
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")

    first = publish_existing_addition_preflight(
        artifact_dir=artifact_dir,
        results_root=tmp_path / "evidence",
    )
    second = publish_existing_addition_preflight(
        artifact_dir=artifact_dir,
        results_root=tmp_path / "evidence",
    )

    assert second["bundle_path"] == first["bundle_path"]


def test_mechanism_screen_eligibility_accepts_complete_actual_shape_bundle():
    # Break caught: M04 launch eligibility is a binary artifact only after all
    # five arms satisfy actual-shape M03 evidence and complete cost accounting.
    decision = _eligibility_decision()
    payload = decision.to_dict()

    assert decision.decision == "eligible"
    assert decision.projected_b200_hours == pytest.approx(1.0)
    assert payload["record_type"] == "mechanism_screen_eligibility"
    assert payload["inputs"]["required_shape"] == M03_SHAPE
    assert payload["checks"]["arms"]["ok"] is True
    assert payload["checks"]["compute"]["ok"] is True


def test_mechanism_screen_eligibility_rejects_missing_seed_or_region():
    # Break caught: a subset of arms or fixed-region draws must not certify the
    # full M03 preflight.
    missing_seed = _mechanism_bundle(required_seed=1)
    missing_region = _mechanism_bundle()
    missing_region["attempts"]["M0"]["fixed_region_eval"]["regions"] = [
        {"region_id": "val_head_0000", "region_ce": 5.0, "region_token_count": 16_384}
    ]

    assert _eligibility_decision(missing_seed).decision == "ineligible"
    assert "required seed" in _eligibility_decision(missing_seed).reason
    assert _eligibility_decision(missing_region).decision == "ineligible"
    assert "fixed regions" in _eligibility_decision(missing_region).reason


def test_mechanism_screen_eligibility_rejects_shrunk_science_shape():
    # Break caught: M03 cannot pass by using the dry-run/tiny mechanism shape.
    bundle = _mechanism_bundle()
    bundle["attempts"]["M0"]["shape"]["seq_len"] = 16

    decision = _eligibility_decision(bundle)

    assert decision.decision == "ineligible"
    assert "science shape" in decision.reason


@pytest.mark.parametrize(
    ("device", "message"),
    (
        ({"type": "cpu", "index": 0, "name": "CPU", "uuid": "CPU"}, "CUDA B200"),
        (
            {
                "type": "cuda",
                "index": 0,
                "name": "NVIDIA H100",
                "uuid": "GPU-" + ("1" * 32),
            },
            "CUDA B200",
        ),
    ),
)
def test_mechanism_screen_eligibility_requires_cuda_b200_device_records(
    device, message
):
    # Break caught: B200-hour eligibility cannot be certified from CPU or
    # non-B200 GPU evidence.
    bundle = _mechanism_bundle()
    bundle["attempts"]["M0"]["device"] = device

    decision = _eligibility_decision(bundle)

    assert decision.decision == "ineligible"
    assert message in decision.reason


@pytest.mark.parametrize("diff", (1.0e-6, 1.0e-7))
def test_mechanism_screen_eligibility_requires_discriminability_above_threshold(diff):
    # Break caught: pairwise differences exactly at 1e-6 or below are not
    # enough evidence to distinguish non-equivalent arms.
    bundle = _mechanism_bundle()
    bundle["discriminability"][0]["max_abs_loss_diff"] = diff

    decision = _eligibility_decision(bundle)

    assert decision.decision == "ineligible"
    assert "discriminability" in decision.reason


def test_mechanism_screen_eligibility_accepts_exact_one_b200_hour_but_rejects_above():
    # Break caught: the M03 projection boundary is inclusive at exactly 1.0
    # summed B200-hour and rejects even a tiny overshoot.
    exact = _mechanism_bundle(projection={**_mechanism_bundle()["projection"]})
    exact["projection"]["screen_b200_hours"] = 0.94
    elevated = _mechanism_bundle(projection={**exact["projection"]})
    elevated["projection"]["screen_b200_hours"] = 0.940001

    assert _eligibility_decision(exact).decision == "eligible"
    decision = _eligibility_decision(elevated)
    assert decision.decision == "ineligible"
    assert "1.0 summed B200-hour" in decision.reason


def test_mechanism_screen_eligibility_requires_retry_accounting():
    # Break caught: retries/failures are part of the M04 screen cost and cannot
    # silently default to zero.
    missing = _mechanism_bundle()
    missing.pop("failure_accounting")
    negative = _mechanism_bundle(
        failure_accounting={"failed_attempts": [{"elapsed_sec": -1.0}]}
    )

    assert _eligibility_decision(missing).decision == "ineligible"
    assert "retry/failure accounting" in _eligibility_decision(missing).reason
    assert _eligibility_decision(negative).decision == "ineligible"


def test_mechanism_screen_eligibility_rejects_invalid_bundle_or_omissions():
    # Break caught: incomplete native bundles and unresolved omission records
    # freeze the decision as ineligible instead of being ignored.
    invalid = _mechanism_bundle()
    invalid["attempts"]["M0"]["attempt"]["status"] = "failed"
    omitted = _mechanism_bundle(
        omission_records=[
            {"record_type": "performance_omission", "reason": "missing checkpoint"}
        ]
    )

    assert _eligibility_decision(invalid).decision == "ineligible"
    assert "attempt" in _eligibility_decision(invalid).reason
    assert _eligibility_decision(omitted).decision == "ineligible"
    assert "omission" in _eligibility_decision(omitted).reason


def test_mechanism_screen_evidence_builder_handles_failed_arm_records():
    # Break caught: the campaign driver must still emit an ineligible decision
    # artifact when one arm fails before probes, rather than crashing first.
    from experiments.falcon.campaign_driver import build_mechanism_screen_evidence

    attempts = {
        arm: _mechanism_arm_record(arm, loss_offset=index * 1.0e-3)
        for index, arm in enumerate(M03_ARMS)
    }
    attempts["M2"] = {
        "arm": "M2",
        "seed": 0,
        "shape": dict(M03_SHAPE),
        "requested_steps": 3,
        "completed_steps": 0,
        "trajectory": [],
        "fixed_region_eval": {"regions": []},
        "probes": [],
        "checkpoint": {},
        "attempt": {
            "status": "failed",
            "attempt_id": "mechanism-preflight-M2-seed0",
            "failure_reason": "synthetic failure",
        },
    }

    evidence = build_mechanism_screen_evidence(
        attempts=attempts,
        disk={
            "free_bytes": 10_000,
            "projected_peak_bytes": 1_000,
            "retained_artifact_bytes": 1_000,
            "reserve_bytes": 1_000,
        },
        failed_attempts=[{"arm": "M2", "seed": 0, "elapsed_sec": 0.1}],
    )

    json.dumps(evidence, allow_nan=False)
    decision = _eligibility_decision(evidence)
    assert decision.decision == "ineligible"
    assert "M2" in decision.reason


def test_mechanism_screen_evidence_builder_scales_measured_probe_time_to_15_runs():
    # Break caught: probe cost projection must include measured probe time for
    # the complete 15-run screen, not just the five-arm preflight probe count.
    from experiments.falcon.campaign_driver import build_mechanism_screen_evidence

    attempts = {
        arm: _mechanism_arm_record(arm, loss_offset=index * 1.0e-3)
        for index, arm in enumerate(M03_ARMS)
    }
    for record in attempts.values():
        for probe in record["probes"]:
            probe["seconds"] = 0.25

    evidence = build_mechanism_screen_evidence(
        attempts=attempts,
        disk={
            "free_bytes": 10_000,
            "projected_peak_bytes": 1_000,
            "retained_artifact_bytes": 1_000,
            "reserve_bytes": 1_000,
        },
        failed_attempts=[],
    )

    assert evidence["projection"]["probe_b200_hours"] == pytest.approx(
        (5 * 4 * 0.25) * (15 / 5) / 3600.0
    )


def test_mechanism_preflight_native_outcome_completes_ineligible_gate():
    # Break caught: M03 may complete its measurement while blocking the M04
    # promotion, so native attempt status must not mirror promotion_decision.
    from experiments.falcon.campaign_driver import _mechanism_preflight_outcome
    from torchtitan.experiments.falcon.promotion import MechanismScreenEligibility

    decision = MechanismScreenEligibility(
        schema_version=1,
        record_type="mechanism_screen_eligibility",
        decision="ineligible",
        reason="disk accounting does not fit with reserve",
        projected_b200_hours=0.75,
        checks={},
        inputs={},
    )

    outcome = _mechanism_preflight_outcome(decision)

    assert outcome == {
        "status": "completed",
        "reason": "disk accounting does not fit with reserve",
        "promotion_decision": "ineligible",
        "projected_b200_hours": 0.75,
    }
