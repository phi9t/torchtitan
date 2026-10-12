# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

import pytest
import torch

from torchtitan.experiments.falcon.model import FalconConfig, FalconForCausalLM


def _tiny_m0_config() -> FalconConfig:
    return FalconConfig(
        vocab_size=29,
        hidden_size=8,
        num_hidden_layers=2,
        num_heads=2,
        head_dim=4,
        intermediate_size=16,
        seq_len=5,
        mixer="falcon",
        variant="falcon1a",
        alignment="delayed",
        phi="rms",
        qk_norm_eps=2.0e-4,
        nlms_denom_eps=7.0e-4,
    )


def _tiny_m4_config() -> FalconConfig:
    config = _tiny_m0_config()
    config.scale_compensation = "rms_to_l2"
    return config


def _tiny_m2_config() -> FalconConfig:
    config = _tiny_m0_config()
    config.phi = "l2"
    return config


def _copy_state(source: torch.nn.Module, target: torch.nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def _assert_parameter_grads_close(
    left: torch.nn.Module,
    right: torch.nn.Module,
    *,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-5,
) -> None:
    for (left_name, left_param), (right_name, right_param) in zip(
        left.named_parameters(), right.named_parameters(), strict=True
    ):
        assert left_name == right_name
        torch.testing.assert_close(
            left_param.grad,
            right_param.grad,
            rtol=rtol,
            atol=atol,
            msg=f"gradient mismatch for {left_name}",
        )


def test_mechanism_runner_rejects_more_than_one_arm_or_seed():
    # Break caught: a mechanism attempt must own exactly one arm x seed cell, so
    # a comma-separated launch cannot silently mix evidence identities.
    from experiments.falcon.mechanism_runner import parse_one_arm, parse_one_seed

    with pytest.raises(ValueError, match="exactly one mechanism arm"):
        parse_one_arm("M0,M1")
    with pytest.raises(ValueError, match="exactly one seed"):
        parse_one_seed("0,1")
    assert parse_one_arm("M4") == "M4"
    assert parse_one_seed("7") == 7


def test_mechanism_probe_schedule_requires_screen_and_confirmation_milestones():
    # Break caught: screen runs must observe the full 0/100/500/1000/2000 set;
    # confirmation adds 4000 and 8000 rather than replacing the screen set.
    from experiments.falcon.mechanism_runner import mechanism_probe_steps

    assert mechanism_probe_steps(2000, confirmation=False) == [
        0,
        100,
        500,
        1000,
        2000,
    ]
    assert mechanism_probe_steps(8000, confirmation=True) == [
        0,
        100,
        500,
        1000,
        2000,
        4000,
        8000,
    ]
    with pytest.raises(ValueError, match="screen mechanism attempts require"):
        mechanism_probe_steps(1999, confirmation=False)
    with pytest.raises(ValueError, match="confirmation mechanism attempts require"):
        mechanism_probe_steps(4000, confirmation=True)


def test_mechanism_diagnostics_report_layer_head_stats_and_m4_canonical_state():
    # Break caught: M4 reports must name the effective epsilons, actual lambda,
    # raw S_rms, and sqrt(head_dim)-canonicalized S_rms for layer/head analysis.
    from torchtitan.experiments.falcon.mechanism import MechanismDiagnostics

    torch.manual_seed(3)
    model = FalconForCausalLM(_tiny_m4_config())
    tokens = torch.arange(10).view(2, 5) % model.config.vocab_size
    diagnostics = MechanismDiagnostics()

    logits = model(tokens, mechanism_diagnostics=diagnostics)
    summary = diagnostics.summary()

    assert logits.shape == (2, 5, model.config.vocab_size)
    assert summary["layers"][0]["layer"] == 0
    head0 = summary["layers"][0]["heads"][0]
    assert head0["labels"] == {
        "phi": "rms",
        "alignment": "delayed",
        "variant": "falcon1a",
        "qk_norm_eps_effective": pytest.approx(5.0e-5),
        "nlms_denom_eps_effective": pytest.approx(2.8e-3),
        "lambda_scale": pytest.approx(4.0),
        "state_canonical_scale": pytest.approx(2.0),
        "state_norm_canonical_label": "sqrt_head_dim_times_raw_S_rms",
        "lambda_label": "actual_lambda_after_scale",
    }
    metrics = head0["metrics"]
    expected_names = {
        "key_energy_pre",
        "key_energy_post",
        "beta",
        "actual_lambda",
        "eta",
        "gamma",
        "negative_inverse_log_gamma",
        "state_norm_raw",
        "state_norm_canonical",
        "update_norm",
        "read_norm",
        "clamp_fraction",
    }
    assert expected_names <= set(metrics)
    assert metrics["beta"]["count"] == 8
    assert metrics["state_norm_canonical"]["mean"] == pytest.approx(
        2.0 * metrics["state_norm_raw"]["mean"]
    )


def test_delayed_key_energy_pre_and_post_describe_the_same_shifted_write_key():
    # Break caught: delayed write summaries must describe k_{t-1}; recording
    # raw key energy from k_t makes pre/post energy columns incomparable.
    from torchtitan.experiments.falcon.mechanism import MechanismDiagnostics

    q = torch.ones(1, 3, 1, 2)
    k = torch.tensor([[[[1.0, 0.0]], [[2.0, 0.0]], [[3.0, 0.0]]]])
    v = torch.ones(1, 3, 1, 2)
    beta = torch.ones(1, 3, 1)
    lamb = torch.zeros(1, 3, 1)
    diagnostics = MechanismDiagnostics()

    diagnostics.record_falcon(
        layer=0,
        q_BLNK=q,
        k_BLNK=k,
        v_BLNV=v,
        beta_BLN=beta,
        lambda_BLN=lamb,
        variant="falcon1a",
        alignment="delayed",
        phi="none",
        qk_norm_eps=0.0,
        nlms_denom_eps=0.0,
        lambda_scale=1.0,
        state_canonical_scale=1.0,
        state_norm_canonical_label="raw_S_rms",
    )
    metrics = diagnostics.summary()["layers"][0]["heads"][0]["metrics"]

    assert metrics["key_energy_pre"]["count"] == 2
    assert metrics["key_energy_pre"]["mean"] == pytest.approx(2.5)
    assert metrics["key_energy_post"]["mean"] == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("config_factory", "expected_scale", "expected_label"),
    (
        (_tiny_m2_config, 1.0, "raw_S_rms"),
        (_tiny_m4_config, 2.0, "sqrt_head_dim_times_raw_S_rms"),
    ),
)
def test_m2_and_m4_canonical_state_scale_labels_are_unambiguous(
    config_factory, expected_scale, expected_label
):
    # Break caught: only M4 should canonicalize raw RMS state by sqrt(head_dim);
    # L2 arms must not report inflated state norms with an ambiguous label.
    from torchtitan.experiments.falcon.mechanism import MechanismDiagnostics

    torch.manual_seed(13)
    model = FalconForCausalLM(config_factory())
    tokens = torch.arange(10).view(2, 5) % model.config.vocab_size
    diagnostics = MechanismDiagnostics()

    model(tokens, mechanism_diagnostics=diagnostics)
    head0 = diagnostics.summary()["layers"][0]["heads"][0]

    assert head0["labels"]["state_canonical_scale"] == pytest.approx(expected_scale)
    assert head0["labels"]["state_norm_canonical_label"] == expected_label
    assert head0["metrics"]["state_norm_canonical"]["mean"] == pytest.approx(
        expected_scale * head0["metrics"]["state_norm_raw"]["mean"]
    )


def test_mechanism_diagnostics_are_not_persistent_model_state():
    # Break caught: diagnostics must be supplied per call; modules must not keep
    # a diagnostic buffer that can leak across attempts or affect checkpoints.
    from torchtitan.experiments.falcon.mechanism import MechanismDiagnostics

    model = FalconForCausalLM(_tiny_m0_config())
    tokens = torch.arange(5).view(1, 5) % model.config.vocab_size
    diagnostics = MechanismDiagnostics()
    _ = model(tokens, mechanism_diagnostics=diagnostics)

    assert diagnostics.summary()["layers"]
    assert "mechanism_diagnostics" not in dict(model.named_buffers())
    for module in model.modules():
        assert "mechanism_diagnostics" not in vars(module)


def test_mechanism_instrumentation_preserves_outputs_and_all_gradients():
    # Break caught: enabling observations must not alter outputs or any fp32
    # gradient at the required rtol=atol=1e-5 mechanism gate.
    from torchtitan.experiments.falcon.mechanism import MechanismDiagnostics

    torch.manual_seed(5)
    plain = FalconForCausalLM(_tiny_m0_config())
    observed = FalconForCausalLM(_tiny_m0_config())
    _copy_state(plain, observed)

    tokens = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=torch.long)
    cotangent = torch.randn(2, 5, plain.config.vocab_size)

    plain_logits = plain(tokens)
    (plain_logits * cotangent).sum().backward()

    diagnostics = MechanismDiagnostics()
    observed_logits = observed(tokens, mechanism_diagnostics=diagnostics)
    (observed_logits * cotangent).sum().backward()

    torch.testing.assert_close(observed_logits, plain_logits, rtol=1e-5, atol=1e-5)
    _assert_parameter_grads_close(observed, plain)


def test_mechanism_dry_run_writes_native_attempt_bundle(tmp_path):
    # Break caught: dry-run mechanism attempts must publish a native evidence
    # bundle plus a diagnostics artifact for the single arm x seed.
    from experiments.falcon.mechanism_runner import run_mechanism

    result = run_mechanism(
        arm="M0",
        seed=0,
        steps=4,
        dry_run=True,
        results_root=tmp_path / "evidence",
        artifact_root=tmp_path / "artifacts",
    )

    assert result["arm"] == "M0"
    assert result["seed"] == 0
    assert result["completed_steps"] == 4
    assert result["claim_label"] == "smoke"
    bundle = tmp_path / "evidence" / "runs" / result["run_id"] / result["attempt_id"]
    manifest = json.loads((bundle / "manifest.json").read_text())
    artifacts = json.loads((bundle / "artifact_index.json").read_text())
    outcome = json.loads((bundle / "outcome.json").read_text())
    assert manifest["logical_run"] == {"workload": "mechanism", "arm": "M0", "seed": 0}
    assert outcome["status"] == "completed"
    assert {artifact["kind"] for artifact in artifacts} >= {
        "mechanism_diagnostics",
        "mechanism_outcome",
    }


def test_mechanism_artifact_dirs_are_unique_and_derived_from_attempt_id(tmp_path):
    # Break caught: second-resolution artifact directories collide for rapid
    # same-arm/seed attempts and decouple artifacts from bundle identity.
    from experiments.falcon.mechanism_runner import (
        mechanism_artifact_dir,
        new_attempt_id,
    )

    first = new_attempt_id("M0", 0)
    second = new_attempt_id("M0", 0)

    assert first != second
    assert mechanism_artifact_dir(tmp_path, first) == tmp_path / first
    assert mechanism_artifact_dir(tmp_path, second) == tmp_path / second


def test_mechanism_runner_uses_one_fixed_probe_batch_across_milestones(tmp_path):
    # Break caught: milestone probes must use a stable fixed batch for the
    # arm/seed instead of changing the probe seed at each step.
    from experiments.falcon.mechanism_runner import run_mechanism

    result = run_mechanism(
        arm="M0",
        seed=0,
        steps=4,
        dry_run=True,
        results_root=tmp_path / "evidence",
        artifact_root=tmp_path / "artifacts",
    )
    diagnostics = json.loads(
        (
            tmp_path / "artifacts" / result["attempt_id"] / "mechanism_diagnostics.json"
        ).read_text()
    )
    probe_digests = {probe["probe_batch_digest"] for probe in diagnostics["probes"]}
    probe_ids = {probe["probe_batch_id"] for probe in diagnostics["probes"]}

    assert len(probe_digests) == 1
    assert probe_ids == {"fixed-synthetic-seed0"}


def test_completed_mechanism_attempt_requires_exact_probe_coverage():
    # Break caught: a completed bundle must not be written if a required probe
    # was missed or if training stopped before the requested/probe horizon.
    from experiments.falcon.mechanism_runner import require_completed_probe_coverage

    require_completed_probe_coverage(
        required_steps=[0, 4],
        observed_steps=[0, 4],
        completed_steps=4,
        requested_steps=4,
    )
    with pytest.raises(ValueError, match="missing required mechanism probes"):
        require_completed_probe_coverage(
            required_steps=[0, 4],
            observed_steps=[0],
            completed_steps=4,
            requested_steps=4,
        )
    with pytest.raises(ValueError, match="completed 3 of requested 4"):
        require_completed_probe_coverage(
            required_steps=[0, 4],
            observed_steps=[0, 4],
            completed_steps=3,
            requested_steps=4,
        )
