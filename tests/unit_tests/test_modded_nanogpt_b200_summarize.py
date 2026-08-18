# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path

from experiments.modded_nanogpt_b200 import summarize


def _summary(
    *,
    result_dir: Path,
    run_id: str,
    lane: str,
    mode: str,
    train_time: float | None,
    included: bool,
    blocker: dict | None = None,
    upstream_commit: str = "ecbb586296d3dac36fd206211f25d63bad4a6b35",
    data_manifest: str = "data_manifest.json",
    environment: dict | None = None,
    gpus: list[dict] | None = None,
    environment_sidecar: dict | None = None,
    hardware_sidecar: dict | None = None,
    claim_validation: dict | None = None,
    telemetry_signs: dict | None = None,
) -> None:
    result_dir.mkdir(parents=True)
    default_gpus = [
        {"index": index, "name": "NVIDIA B200", "compute_capability": [10, 0]}
        for index in range(2)
    ]
    variant_patch_classification = {}
    if lane == "B":
        variant_patch_classification = {
            "schema_version": 1,
            "source": "variant",
            "diff_artifact": "variant_patch.diff",
            "files": [
                {
                    "path": "train_gpt.py",
                    "patch_class": "hardware-detection",
                    "first_lane_a_blocker": "FA3 no kernel image on B200",
                },
                {
                    "path": "triton_kernels.py",
                    "patch_class": "kernel-compat",
                    "first_lane_a_blocker": (
                        "Triton sm100 custom kernel compile/runtime blocker"
                    ),
                },
            ],
        }
    telemetry = {
        "rootfs": {
            "torchtitan_in_rootfs": "1",
            "cwd": "/workspace/torchtitan",
            "workspace_sentinel_exists": True,
        },
        "signs": telemetry_signs or {},
    }
    result_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": included,
                "classification": {
                    "lane": lane,
                    "mode": mode,
                    "arm": "A0" if lane == "A" else "B0",
                    "claim_label": "B200 upstream reproduction"
                    if lane == "A" and mode == "full"
                    else "diagnostic",
                    "evidence_tier": "full-single-attempt" if mode == "full" else mode,
                    "run_id": run_id,
                    "attempt_id": f"{run_id}_attempt_001",
                    "environment_class": "torchtitan-rootfs-b200",
                    "claim_eligible": included,
                },
                "upstream_commit": upstream_commit,
                "source": {"path": "source", "dirty": False, "status": ""},
                "data_manifest": data_manifest,
                "data_manifest_summary": {
                    "schema_version": 1,
                    "path": data_manifest,
                    "resolved_path": None,
                    "dataset": "fineweb10B",
                    "token_budget": "900M",
                    "num_files": 10,
                    "total_bytes": 2000010240,
                    "source_commit": upstream_commit,
                    "verified_sha": True,
                },
                "preflight_data_manifest": {
                    "schema_version": 1,
                    "verified_sha": True,
                },
                "environment": environment or {},
                "gpus": gpus if gpus is not None else default_gpus,
                "environment_sidecar": environment_sidecar or {},
                "hardware_sidecar": hardware_sidecar or {},
                "claim_validation": claim_validation or {},
                "variant_patch_classification": variant_patch_classification,
                "active_jobs": {
                    "ok": True,
                    "active_job_count": 0,
                    "ignored_match_count": 0,
                    "active_jobs": [],
                },
                "telemetry": telemetry,
                "final_metrics": {
                    "val_loss": 3.27 if included else None,
                    "train_time": train_time,
                    "step_avg": 0.8 if included else None,
                    "peak_allocated_memory": None,
                    "peak_reserved_memory": None,
                },
                "wall_clock": {"elapsed_seconds": 90.0 if included else 30.0},
                "included_in_baseline_stats": included,
                "blocker": blocker,
            }
        )
        + "\n"
    )


def _launch_readiness(
    *,
    result_dir: Path,
    ready_to_launch: bool,
    training_launched: bool,
    launch_authority_required: bool,
    skip_run: bool = False,
    blocked_by: list[dict] | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> None:
    report = {
        "schema_version": 1,
        "ready_to_launch": ready_to_launch,
        "training_launched": training_launched,
        "skip_run": skip_run,
        "launch_authority_required": launch_authority_required,
        "launch_authorization_required_token": "launch-full-b200"
        if launch_authority_required
        else None,
        "preflight_ok": ready_to_launch,
        "full_mode_gates": {
            "full_mode": True,
            "data_manifest_checked": ready_to_launch,
            "verified_sha": ready_to_launch,
            "nccl_checked": ready_to_launch,
            "manifest_token_budget": "900M",
            "manifest_num_files": 10,
            "manifest_total_bytes": 2000010240,
        },
        "blocked_by": blocked_by or [],
    }
    if run_id is not None:
        report["run_id"] = run_id
    if attempt_id is not None:
        report["attempt_id"] = attempt_id
    result_dir.joinpath("launch_readiness.json").write_text(json.dumps(report) + "\n")


def _make_valid_baseline_evidence(result_dir: Path) -> None:
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    data_manifest = summary["data_manifest"]
    summary["launch_readiness"] = {
        "schema_version": 1,
        "data_manifest": data_manifest,
        "training_launched": True,
        "skip_run": False,
        "allow_previous_stall": False,
        "preflight_ok": True,
        "full_mode_gates": {
            "verify_sha_requested": True,
            "data_manifest_checked": True,
            "verified_sha": True,
            "nccl_checked": True,
            "manifest_token_budget": "900M",
            "manifest_num_files": 10,
            "manifest_total_bytes": 2000010240,
        },
    }
    summary["data_manifest_summary"] = {
        "schema_version": 1,
        "path": data_manifest,
        "resolved_path": None,
        "dataset": "fineweb10B",
        "token_budget": "900M",
        "num_files": 10,
        "total_bytes": 2000010240,
        "source_commit": "ecbb586296d3dac36fd206211f25d63bad4a6b35",
        "verified_sha": True,
    }
    summary["preflight_data_manifest"] = {
        "schema_version": 1,
        "verified_sha": True,
    }
    summary["telemetry"] = {
        "rootfs": {
            "torchtitan_in_rootfs": "1",
            "cwd": "/workspace/torchtitan",
            "workspace_sentinel_exists": True,
        },
    }
    summary["gpus"] = [{"name": "NVIDIA B200", "index": index} for index in range(2)]
    summary_path.write_text(json.dumps(summary) + "\n")
    result_dir.joinpath("exit_code.json").write_text(
        json.dumps({"phase": "training", "exit_code": 0}) + "\n"
    )


def test_run_index_excludes_diagnostics_from_baseline_stats(tmp_path: Path):
    fast_dir = tmp_path / "lane_a_full_fast"
    slow_dir = tmp_path / "lane_a_full_slow"
    _summary(
        result_dir=fast_dir,
        run_id="lane_a_full_fast",
        lane="A",
        mode="full",
        train_time=72.0,
        included=True,
    )
    _make_valid_baseline_evidence(fast_dir)
    _summary(
        result_dir=slow_dir,
        run_id="lane_a_full_slow",
        lane="A",
        mode="full",
        train_time=78.0,
        included=True,
    )
    _make_valid_baseline_evidence(slow_dir)
    _summary(
        result_dir=tmp_path / "lane_b_diag",
        run_id="lane_b_diag",
        lane="B",
        mode="diagnostic",
        train_time=None,
        included=False,
        blocker={"phase": "warmup", "message": "CUDA out of memory"},
    )

    index = summarize.build_index(tmp_path)

    assert index["schema_version"] == 1
    assert index["total_attempts"] == 3
    assert index["baseline_stats"]["count"] == 2
    assert index["baseline_stats"]["best_train_time"] == 72.0
    assert index["baseline_stats"]["median_train_time"] == 75.0
    assert len(index["diagnostic_or_failed_attempts"]) == 1
    assert index["diagnostic_or_failed_attempts"][0]["blocker"]["phase"] == "warmup"


def test_run_index_rejects_diagnostic_even_if_summary_claims_baseline_inclusion(
    tmp_path: Path,
):
    _summary(
        result_dir=tmp_path / "lane_b_diag_bad_summary",
        run_id="lane_b_diag_bad_summary",
        lane="B",
        mode="diagnostic",
        train_time=72.0,
        included=True,
    )

    index = summarize.build_index(tmp_path)

    assert index["baseline_stats"]["count"] == 0
    assert index["groups"][0]["baseline_stats"]["count"] == 0
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["classification"]["mode"] == "diagnostic"
    assert attempt["included_in_baseline_stats"] is False


def test_run_index_preserves_claim_validation_and_telemetry_signs(tmp_path: Path):
    _summary(
        result_dir=tmp_path / "lane_b_diag_with_signs",
        run_id="lane_b_diag_with_signs",
        lane="B",
        mode="diagnostic",
        train_time=None,
        included=False,
        blocker={"phase": "warmup", "message": "CUDA out of memory"},
        claim_validation={
            "successful_b200_reproduction": False,
            "first_blocker": "final validation was not reached",
        },
        telemetry_signs={
            "thermal_or_clock_throttling": True,
            "thermal_or_clock_throttling_reason": "minimum SM clock 990.0MHz",
            "cpu_or_rss_bottleneck": True,
            "cpu_or_rss_bottleneck_reason": "peak process CPU 950.0%",
        },
    )

    index = summarize.build_index(tmp_path)

    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["claim_validation"] == {
        "successful_b200_reproduction": False,
        "first_blocker": "final validation was not reached",
    }
    assert attempt["telemetry_signs"] == {
        "thermal_or_clock_throttling": True,
        "thermal_or_clock_throttling_reason": "minimum SM clock 990.0MHz",
        "cpu_or_rss_bottleneck": True,
        "cpu_or_rss_bottleneck_reason": "peak process CPU 950.0%",
    }


def test_run_index_rejects_full_baseline_without_launch_evidence(tmp_path: Path):
    result_dir = tmp_path / "lane_a_full_stale_success"
    _summary(
        result_dir=result_dir,
        run_id="lane_a_full_stale_success",
        lane="A",
        mode="full",
        train_time=72.0,
        included=True,
    )

    index = summarize.build_index(tmp_path)

    assert index["baseline_stats"]["count"] == 0
    assert index["groups"][0]["baseline_stats"]["count"] == 0
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["included_in_baseline_stats"] is False
    assert attempt["baseline_evidence_error"]["phase"] == "summary"
    assert "launch_readiness" in attempt["baseline_evidence_error"]["message"]


def test_run_index_rejects_full_baseline_with_wrong_manifest_source_commit(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_wrong_data_source"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_wrong_data_source",
        lane="B",
        mode="full",
        train_time=72.0,
        included=True,
    )
    _make_valid_baseline_evidence(result_dir)
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["data_manifest_summary"]["source_commit"] = "stale-data-source"
    summary_path.write_text(json.dumps(summary) + "\n")

    index = summarize.build_index(tmp_path)

    assert index["baseline_stats"]["count"] == 0
    assert index["groups"][0]["baseline_stats"]["count"] == 0
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["included_in_baseline_stats"] is False
    assert attempt["baseline_evidence_error"]["phase"] == "summary"
    assert (
        "data_manifest_summary.source_commit"
        in attempt["baseline_evidence_error"]["message"]
    )


def test_run_index_rejects_full_baseline_without_preflight_manifest_sha(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_missing_preflight_sha"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_missing_preflight_sha",
        lane="B",
        mode="full",
        train_time=72.0,
        included=True,
    )
    _make_valid_baseline_evidence(result_dir)
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["preflight_data_manifest"]["verified_sha"] = False
    assert summary["data_manifest_summary"]["verified_sha"] is True
    summary_path.write_text(json.dumps(summary) + "\n")

    index = summarize.build_index(tmp_path)

    assert index["baseline_stats"]["count"] == 0
    assert index["groups"][0]["baseline_stats"]["count"] == 0
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["included_in_baseline_stats"] is False
    assert attempt["data_manifest_summary"]["verified_sha"] is True
    assert attempt["preflight_data_manifest"]["verified_sha"] is False
    assert attempt["baseline_evidence_error"] == {
        "phase": "preflight_data_manifest.verified_sha",
        "message": "preflight_data_manifest.verified_sha must be true",
    }


def test_run_index_rejects_full_baseline_with_nonzero_training_exit_code(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_a_full_failed_training"
    _summary(
        result_dir=result_dir,
        run_id="lane_a_full_failed_training",
        lane="A",
        mode="full",
        train_time=72.0,
        included=True,
    )
    _make_valid_baseline_evidence(result_dir)
    (result_dir / "exit_code.json").write_text(
        json.dumps({"phase": "training", "exit_code": 7}) + "\n"
    )

    index = summarize.build_index(tmp_path)

    assert index["baseline_stats"]["count"] == 0
    assert index["groups"][0]["baseline_stats"]["count"] == 0
    assert index["launch_ready_attempts"] == []
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["included_in_baseline_stats"] is False
    assert attempt["baseline_evidence_error"]["phase"] == "summary"
    assert "exit_code.json" in attempt["baseline_evidence_error"]["message"]
    assert "exit_code=0" in attempt["baseline_evidence_error"]["message"]


def test_run_index_groups_by_claim_boundary_fields(tmp_path: Path):
    _summary(
        result_dir=tmp_path / "lane_a_full",
        run_id="lane_a_full",
        lane="A",
        mode="full",
        train_time=72.0,
        included=True,
    )
    _summary(
        result_dir=tmp_path / "lane_b_diag",
        run_id="lane_b_diag",
        lane="B",
        mode="diagnostic",
        train_time=None,
        included=False,
    )

    index = summarize.build_index(tmp_path)

    keys = {group["key"] for group in index["groups"]}
    lane_a_key = (
        "lane=A|mode=full|arm=A0|claim_label=B200 upstream reproduction|"
        "environment_class=torchtitan-rootfs-b200|evidence_tier=full-single-attempt|"
        "source_commit=ecbb586296d3dac36fd206211f25d63bad4a6b35|"
        "data_manifest=data_manifest.json"
    )
    lane_b_key = (
        "lane=B|mode=diagnostic|arm=B0|claim_label=diagnostic|"
        "environment_class=torchtitan-rootfs-b200|evidence_tier=diagnostic|"
        "source_commit=ecbb586296d3dac36fd206211f25d63bad4a6b35|"
        "data_manifest=data_manifest.json"
    )
    assert lane_a_key in keys
    assert lane_b_key in keys


def test_run_index_groups_by_source_commit_and_data_manifest(tmp_path: Path):
    _summary(
        result_dir=tmp_path / "lane_a_full_original_data",
        run_id="lane_a_full_original_data",
        lane="A",
        mode="full",
        train_time=72.0,
        included=True,
        upstream_commit="commit-a",
        data_manifest="fineweb_900m_a.json",
    )
    _summary(
        result_dir=tmp_path / "lane_a_full_other_data",
        run_id="lane_a_full_other_data",
        lane="A",
        mode="full",
        train_time=74.0,
        included=True,
        upstream_commit="commit-a",
        data_manifest="fineweb_900m_b.json",
    )
    _summary(
        result_dir=tmp_path / "lane_a_full_other_commit",
        run_id="lane_a_full_other_commit",
        lane="A",
        mode="full",
        train_time=76.0,
        included=True,
        upstream_commit="commit-b",
        data_manifest="fineweb_900m_a.json",
    )

    index = summarize.build_index(tmp_path)

    keys = {group["key"] for group in index["groups"]}
    assert len(keys) == 3
    assert any(
        "source_commit=commit-a|data_manifest=fineweb_900m_a.json" in key
        for key in keys
    )
    assert any(
        "source_commit=commit-a|data_manifest=fineweb_900m_b.json" in key
        for key in keys
    )
    assert any(
        "source_commit=commit-b|data_manifest=fineweb_900m_a.json" in key
        for key in keys
    )


def test_run_index_prefers_captured_source_commit_over_upstream_commit(tmp_path: Path):
    _summary(
        result_dir=tmp_path / "lane_b_variant_a",
        run_id="lane_b_variant_a",
        lane="B",
        mode="diagnostic",
        train_time=None,
        included=False,
        upstream_commit="pinned-upstream",
    )
    summary_path = tmp_path / "lane_b_variant_a" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["source"] = {"path": "variant", "commit": "captured-source-commit"}
    summary_path.write_text(json.dumps(summary) + "\n")

    index = summarize.build_index(tmp_path)

    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["upstream_commit"] == "pinned-upstream"
    assert attempt["source_commit"] == "captured-source-commit"
    assert "source_commit=captured-source-commit" in index["groups"][0]["key"]


def test_write_index_json_is_atomic(tmp_path: Path):
    _summary(
        result_dir=tmp_path / "lane_a_full",
        run_id="lane_a_full",
        lane="A",
        mode="full",
        train_time=72.0,
        included=True,
    )

    index = summarize.write_index(tmp_path, tmp_path / "run_index.json")

    written = json.loads((tmp_path / "run_index.json").read_text())
    assert written == index
    assert not (tmp_path / "run_index.json.tmp").exists()


def test_run_index_surfaces_launch_ready_prerequisite_attempts(tmp_path: Path):
    result_dir = tmp_path / "lane_b_full_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
        gpus=[
            {"index": 0, "name": "NVIDIA B200"},
            {"index": 1, "name": "NVIDIA B200"},
        ],
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary_path.write_text(json.dumps(summary) + "\n")

    index = summarize.build_index(tmp_path)

    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["launch_readiness"]["ready_to_launch"] is True
    assert attempt["launch_readiness"]["training_launched"] is False
    assert attempt["launch_readiness"]["launch_authority_required"] is True
    assert attempt["launch_readiness"]["full_mode_gates"]["verified_sha"] is True
    assert len(index["launch_ready_attempts"]) == 1
    assert index["launch_ready_attempts"][0]["summary"] == str(
        result_dir / "summary.json"
    )
    assert index["launch_ready_attempts"][0]["blocker"]["phase"] == "not_launched"


def test_run_index_separates_skip_run_prerequisites_from_launch_ready_attempts(
    tmp_path: Path,
):
    dry_gate_dir = tmp_path / "lane_b_full_skip_run_gate"
    _summary(
        result_dir=dry_gate_dir,
        run_id="lane_b_full_skip_run_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=dry_gate_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
        skip_run=True,
    )
    dry_summary_path = dry_gate_dir / "summary.json"
    dry_summary = json.loads(dry_summary_path.read_text())
    dry_summary["classification"]["claim_eligible"] = True
    dry_summary_path.write_text(json.dumps(dry_summary) + "\n")

    authority_gate_dir = tmp_path / "lane_b_full_authority_gate"
    _summary(
        result_dir=authority_gate_dir,
        run_id="lane_b_full_authority_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "launch_authority",
            "message": "full launch requires explicit authorization",
        },
    )
    _launch_readiness(
        result_dir=authority_gate_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    authority_summary_path = authority_gate_dir / "summary.json"
    authority_summary = json.loads(authority_summary_path.read_text())
    authority_summary["classification"]["claim_eligible"] = True
    authority_summary_path.write_text(json.dumps(authority_summary) + "\n")

    index = summarize.build_index(tmp_path)

    prerequisite_summaries = {
        attempt["summary"] for attempt in index["launch_prerequisite_attempts"]
    }
    assert prerequisite_summaries == {
        str(dry_summary_path),
        str(authority_summary_path),
    }
    assert len(index["launch_ready_attempts"]) == 1
    assert index["launch_ready_attempts"][0]["summary"] == str(authority_summary_path)
    assert (
        index["launch_ready_attempts"][0]["launch_readiness"].get("skip_run")
        is not True
    )


def test_run_index_uses_embedded_launch_readiness_when_sidecar_is_absent(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_archived_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_archived_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary["launch_readiness"] = {
        "schema_version": 1,
        "ready_to_launch": True,
        "training_launched": False,
        "launch_authority_required": True,
        "launch_authorization_required_token": "launch-full-b200",
        "preflight_ok": True,
        "full_mode_gates": {
            "full_mode": True,
            "data_manifest_checked": True,
            "verified_sha": True,
            "nccl_checked": True,
            "manifest_token_budget": "900M",
            "manifest_num_files": 10,
            "manifest_total_bytes": 2000010240,
        },
        "blocked_by": [],
    }
    summary_path.write_text(json.dumps(summary) + "\n")

    index = summarize.build_index(tmp_path)

    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["launch_readiness"]["ready_to_launch"] is True
    assert attempt["launch_readiness"]["training_launched"] is False
    assert len(index["launch_ready_attempts"]) == 1
    assert index["launch_ready_attempts"][0]["summary"] == str(summary_path)


def test_run_index_launch_ready_attempts_are_full_mode_only(tmp_path: Path):
    result_dir = tmp_path / "lane_b_diagnostic_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_diagnostic_gate",
        lane="B",
        mode="diagnostic",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "diagnostic skip-run was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=False,
    )

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    assert (
        index["diagnostic_or_failed_attempts"][0]["classification"]["mode"]
        == "diagnostic"
    )
    assert index["launch_ready_attempts"] == []


def test_run_index_launch_ready_attempts_require_sidecar_classification_match(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_stale_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_stale_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary_path.write_text(json.dumps(summary) + "\n")
    readiness_path = result_dir / "launch_readiness.json"
    readiness = json.loads(readiness_path.read_text())
    readiness["lane"] = "A"
    readiness["mode"] = "full"
    readiness_path.write_text(json.dumps(readiness) + "\n")

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    assert index["diagnostic_or_failed_attempts"][0]["launch_readiness"]["lane"] == "A"
    assert index["launch_ready_attempts"] == []


def test_run_index_launch_ready_attempts_require_sidecar_attempt_identity_match(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_copied_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_copied_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
        run_id="different_run",
        attempt_id="different_run_attempt_001",
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary_path.write_text(json.dumps(summary) + "\n")

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["classification"]["run_id"] == "lane_b_full_copied_gate"
    assert attempt["launch_readiness"]["run_id"] == "different_run"
    assert attempt["launch_readiness"]["attempt_id"] == "different_run_attempt_001"
    assert index["launch_ready_attempts"] == []


def test_run_index_launch_ready_attempts_require_nested_sidecar_identity_match(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_copied_nested_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_copied_nested_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary_path.write_text(json.dumps(summary) + "\n")
    readiness_path = result_dir / "launch_readiness.json"
    readiness = json.loads(readiness_path.read_text())
    readiness["classification"] = {
        "lane": "B",
        "mode": "full",
        "run_id": "different_run",
        "attempt_id": "different_run_attempt_001",
    }
    readiness_path.write_text(json.dumps(readiness) + "\n")

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["classification"]["run_id"] == "lane_b_full_copied_nested_gate"
    assert attempt["launch_readiness"]["classification"]["run_id"] == "different_run"
    assert (
        attempt["launch_readiness"]["classification"]["attempt_id"]
        == "different_run_attempt_001"
    )
    assert index["launch_ready_attempts"] == []


def test_run_index_launch_ready_attempts_reject_contradictory_sidecar_identity(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_contradictory_identity_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_contradictory_identity_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
        run_id="lane_b_full_contradictory_identity_gate",
        attempt_id="lane_b_full_contradictory_identity_gate_attempt_001",
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary_path.write_text(json.dumps(summary) + "\n")
    readiness_path = result_dir / "launch_readiness.json"
    readiness = json.loads(readiness_path.read_text())
    readiness["classification"] = {
        "lane": "B",
        "mode": "full",
        "run_id": "different_run",
        "attempt_id": "different_run_attempt_001",
    }
    readiness_path.write_text(json.dumps(readiness) + "\n")

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert (
        attempt["launch_readiness"]["run_id"]
        == "lane_b_full_contradictory_identity_gate"
    )
    assert attempt["launch_readiness"]["classification"]["run_id"] == "different_run"
    assert index["launch_ready_attempts"] == []


def test_run_index_launch_ready_attempts_reject_contradictory_blockers(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_contradictory_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_contradictory_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
        blocked_by=[
            {
                "phase": "data_manifest",
                "message": "full mode requires SHA-verified data manifest",
            }
        ],
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary_path.write_text(json.dumps(summary) + "\n")
    readiness_path = result_dir / "launch_readiness.json"
    readiness = json.loads(readiness_path.read_text())
    readiness["full_mode_gates"]["verified_sha"] = False
    readiness_path.write_text(json.dumps(readiness) + "\n")

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["launch_readiness"]["ready_to_launch"] is True
    assert attempt["launch_readiness"]["blocked_by"][0]["phase"] == "data_manifest"
    assert attempt["launch_readiness"]["full_mode_gates"]["verified_sha"] is False
    assert index["launch_ready_attempts"] == []


def test_run_index_launch_ready_attempts_require_full_manifest_shape(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_smoke_manifest_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_smoke_manifest_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary_path.write_text(json.dumps(summary) + "\n")
    readiness_path = result_dir / "launch_readiness.json"
    readiness = json.loads(readiness_path.read_text())
    readiness["full_mode_gates"]["manifest_token_budget"] = "smoke"
    readiness["full_mode_gates"]["manifest_num_files"] = 1
    readiness_path.write_text(json.dumps(readiness) + "\n")

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["launch_readiness"]["ready_to_launch"] is True
    assert attempt["launch_readiness"]["full_mode_gates"]["verified_sha"] is True
    assert (
        attempt["launch_readiness"]["full_mode_gates"]["manifest_token_budget"]
        == "smoke"
    )
    assert index["launch_ready_attempts"] == []


def test_run_index_launch_ready_attempts_require_manifest_gate_checked(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_manifest_unchecked_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_manifest_unchecked_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary_path.write_text(json.dumps(summary) + "\n")
    readiness_path = result_dir / "launch_readiness.json"
    readiness = json.loads(readiness_path.read_text())
    readiness["full_mode_gates"]["data_manifest_checked"] = False
    readiness_path.write_text(json.dumps(readiness) + "\n")

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["launch_readiness"]["ready_to_launch"] is True
    assert attempt["launch_readiness"]["full_mode_gates"]["verified_sha"] is True
    assert (
        attempt["launch_readiness"]["full_mode_gates"]["manifest_token_budget"]
        == "900M"
    )
    assert (
        attempt["launch_readiness"]["full_mode_gates"]["data_manifest_checked"] is False
    )
    assert index["launch_ready_attempts"] == []


def test_run_index_launch_ready_attempts_require_matching_data_manifest_path(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_copied_manifest_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_copied_manifest_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary["data_manifest_summary"] = {
        "path": "experiments/modded_nanogpt_b200/results/current/data_manifest.json",
        "resolved_path": (
            "experiments/modded_nanogpt_b200/results/full/data_manifest.json"
        ),
        "source_commit": "ecbb586296d3dac36fd206211f25d63bad4a6b35",
    }
    summary_path.write_text(json.dumps(summary) + "\n")
    readiness_path = result_dir / "launch_readiness.json"
    readiness = json.loads(readiness_path.read_text())
    readiness[
        "data_manifest"
    ] = "experiments/modded_nanogpt_b200/results/other/data_manifest.json"
    readiness_path.write_text(json.dumps(readiness) + "\n")

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["launch_readiness"]["ready_to_launch"] is True
    assert attempt["launch_readiness"]["data_manifest"].endswith(
        "other/data_manifest.json"
    )
    assert index["launch_ready_attempts"] == []


def test_run_index_launch_ready_attempts_require_summary_manifest_evidence(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_missing_summary_manifest_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_missing_summary_manifest_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary.pop("data_manifest_summary")
    summary_path.write_text(json.dumps(summary) + "\n")

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["launch_readiness"]["ready_to_launch"] is True
    assert "data_manifest_summary" not in summary
    assert attempt["data_manifest_summary"] == {}
    assert index["launch_ready_attempts"] == []


def test_run_index_launch_ready_attempts_require_summary_manifest_sha_state(
    tmp_path: Path,
):
    missing_sha_dir = tmp_path / "lane_b_full_missing_manifest_sha_gate"
    _summary(
        result_dir=missing_sha_dir,
        run_id="lane_b_full_missing_manifest_sha_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=missing_sha_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    missing_summary_path = missing_sha_dir / "summary.json"
    missing_summary = json.loads(missing_summary_path.read_text())
    missing_summary["classification"]["claim_eligible"] = True
    missing_summary["preflight_data_manifest"] = {"verified_sha": True}
    missing_summary["data_manifest_summary"].pop("verified_sha")
    missing_summary_path.write_text(json.dumps(missing_summary) + "\n")

    false_sha_dir = tmp_path / "lane_b_full_false_manifest_sha_gate"
    _summary(
        result_dir=false_sha_dir,
        run_id="lane_b_full_false_manifest_sha_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=false_sha_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    false_summary_path = false_sha_dir / "summary.json"
    false_summary = json.loads(false_summary_path.read_text())
    false_summary["classification"]["claim_eligible"] = True
    false_summary["preflight_data_manifest"] = {"verified_sha": True}
    false_summary["data_manifest_summary"]["verified_sha"] = False
    false_summary_path.write_text(json.dumps(false_summary) + "\n")

    index = summarize.build_index(tmp_path)

    assert index["launch_prerequisite_attempts"] == []
    assert index["launch_ready_attempts"] == []
    exclusions = {
        attempt["summary"]: attempt["launch_readiness_exclusion"]
        for attempt in index["diagnostic_or_failed_attempts"]
    }
    assert exclusions[str(missing_summary_path)] == {
        "phase": "data_manifest_summary.verified_sha",
        "message": "data_manifest_summary.verified_sha must be true",
    }
    assert exclusions[str(false_summary_path)] == {
        "phase": "data_manifest_summary.verified_sha",
        "message": "data_manifest_summary.verified_sha must be true",
    }


def test_run_index_launch_ready_attempts_require_summary_b200_gpu_inventory(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_bad_gpu_inventory_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_bad_gpu_inventory_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
        gpus=[
            {"index": 0, "name": "NVIDIA B200"},
        ],
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary_path.write_text(json.dumps(summary) + "\n")

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["launch_readiness"]["ready_to_launch"] is True
    assert attempt["gpu_count"] == 1
    assert index["launch_ready_attempts"] == []


def test_run_index_launch_ready_attempts_require_summary_rootfs_evidence(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_missing_rootfs_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_missing_rootfs_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary["telemetry"].pop("rootfs")
    summary_path.write_text(json.dumps(summary) + "\n")

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["launch_readiness"]["ready_to_launch"] is True
    assert "rootfs" not in summary["telemetry"]
    assert index["launch_ready_attempts"] == []


def test_run_index_launch_ready_attempts_require_pinned_source_commit(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_stale_source_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_stale_source_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary["source"]["commit"] = "stale-source-commit"
    summary_path.write_text(json.dumps(summary) + "\n")

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["launch_readiness"]["ready_to_launch"] is True
    assert attempt["source_commit"] == "stale-source-commit"
    assert index["launch_ready_attempts"] == []


def test_run_index_launch_ready_attempts_require_lane_b_patch_classification(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_missing_patch_classification_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_missing_patch_classification_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary.pop("variant_patch_classification")
    summary_path.write_text(json.dumps(summary) + "\n")

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["launch_readiness"]["ready_to_launch"] is True
    assert "variant_patch_classification" not in summary
    assert index["launch_ready_attempts"] == []


def test_run_index_launch_ready_attempts_require_active_job_scan_evidence(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_missing_active_jobs_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_missing_active_jobs_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary.pop("active_jobs")
    summary_path.write_text(json.dumps(summary) + "\n")

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["launch_readiness"]["ready_to_launch"] is True
    assert attempt["launch_readiness_exclusion"] == {
        "phase": "active_jobs",
        "message": "summary is missing clean active-job scan evidence",
    }
    assert "active_jobs" not in summary
    assert index["launch_ready_attempts"] == []


def test_stale_artifacts_include_ready_sidecar_demoted_by_active_jobs(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_missing_active_jobs_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_missing_active_jobs_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary.pop("active_jobs")
    summary_path.write_text(json.dumps(summary) + "\n")

    index = summarize.build_index(tmp_path)

    stale = index["stale_or_demoted_artifacts"]
    assert stale == [
        {
            "summary": str(summary_path),
            "reason": "launch_readiness_exclusion",
            "message": "summary is missing clean active-job scan evidence",
            "operator_action": (
                "Archived or demoted artifact; do not use as launch-ready "
                "or baseline evidence."
            ),
            "lane": "B",
            "mode": "full",
            "claim_label": "diagnostic",
            "launch_readiness_exclusion": {
                "phase": "active_jobs",
                "message": "summary is missing clean active-job scan evidence",
            },
        }
    ]
    assert index["launch_ready_attempts"] == []


def test_run_index_summarizes_launch_readiness_exclusion_phases(tmp_path: Path):
    missing_patch_dir = tmp_path / "lane_b_full_missing_patch_stats"
    _summary(
        result_dir=missing_patch_dir,
        run_id="lane_b_full_missing_patch_stats",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=missing_patch_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    missing_patch_summary_path = missing_patch_dir / "summary.json"
    missing_patch_summary = json.loads(missing_patch_summary_path.read_text())
    missing_patch_summary["classification"]["claim_eligible"] = True
    missing_patch_summary.pop("variant_patch_classification")
    missing_patch_summary_path.write_text(json.dumps(missing_patch_summary) + "\n")

    missing_active_jobs_dir = tmp_path / "lane_b_full_missing_active_jobs_stats"
    _summary(
        result_dir=missing_active_jobs_dir,
        run_id="lane_b_full_missing_active_jobs_stats",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=missing_active_jobs_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    missing_active_jobs_summary_path = missing_active_jobs_dir / "summary.json"
    missing_active_jobs_summary = json.loads(
        missing_active_jobs_summary_path.read_text()
    )
    missing_active_jobs_summary["classification"]["claim_eligible"] = True
    missing_active_jobs_summary.pop("active_jobs")
    missing_active_jobs_summary_path.write_text(
        json.dumps(missing_active_jobs_summary) + "\n"
    )

    index = summarize.build_index(tmp_path)

    assert index["launch_readiness_exclusion_stats"] == {
        "count": 2,
        "by_phase": {
            "active_jobs": {
                "count": 1,
                "example_summaries": [str(missing_active_jobs_summary_path)],
            },
            "variant_patch_classification": {
                "count": 1,
                "example_summaries": [str(missing_patch_summary_path)],
            },
        },
    }


def test_run_index_launch_ready_attempts_require_launch_authorization_token(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_missing_token_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_missing_token_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary_path.write_text(json.dumps(summary) + "\n")
    readiness_path = result_dir / "launch_readiness.json"
    readiness = json.loads(readiness_path.read_text())
    readiness.pop("launch_authorization_required_token")
    readiness_path.write_text(json.dumps(readiness) + "\n")

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["launch_readiness"]["ready_to_launch"] is True
    assert attempt["launch_readiness"]["launch_authority_required"] is True
    assert "launch_authorization_required_token" not in attempt["launch_readiness"]
    assert index["launch_ready_attempts"] == []


def test_run_index_launch_ready_attempts_require_launch_authority_gate(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_authority_not_required_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_authority_not_required_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    _launch_readiness(
        result_dir=result_dir,
        ready_to_launch=True,
        training_launched=False,
        launch_authority_required=True,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary_path.write_text(json.dumps(summary) + "\n")
    readiness_path = result_dir / "launch_readiness.json"
    readiness = json.loads(readiness_path.read_text())
    readiness["launch_authority_required"] = False
    readiness_path.write_text(json.dumps(readiness) + "\n")

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["launch_readiness"]["ready_to_launch"] is True
    assert (
        attempt["launch_readiness"]["launch_authorization_required_token"]
        == "launch-full-b200"
    )
    assert attempt["launch_readiness"]["launch_authority_required"] is False
    assert index["launch_ready_attempts"] == []


def test_run_index_preserves_malformed_launch_readiness_as_failed_attempt(
    tmp_path: Path,
):
    result_dir = tmp_path / "lane_b_full_malformed_gate"
    _summary(
        result_dir=result_dir,
        run_id="lane_b_full_malformed_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "skip-run requested; training was not launched",
        },
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"]["claim_eligible"] = True
    summary_path.write_text(json.dumps(summary) + "\n")
    (result_dir / "launch_readiness.json").write_text("{not valid json\n")

    index = summarize.build_index(tmp_path)

    assert index["baseline_stats"]["count"] == 0
    assert index["launch_ready_attempts"] == []
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["summary"] == str(summary_path)
    assert attempt["launch_readiness_error"]["phase"] == "launch_readiness"
    assert "launch_readiness.json" in attempt["launch_readiness_error"]["message"]
    assert attempt["included_in_baseline_stats"] is False


def test_run_index_preserves_malformed_summary_as_failed_attempt(tmp_path: Path):
    _summary(
        result_dir=tmp_path / "lane_b_diagnostic_valid",
        run_id="lane_b_diagnostic_valid",
        lane="B",
        mode="diagnostic",
        train_time=None,
        included=False,
        blocker={
            "phase": "not_launched",
            "message": "diagnostic skip-run was not launched",
        },
    )
    broken_dir = tmp_path / "lane_b_full_broken_summary"
    broken_dir.mkdir()
    broken_summary = broken_dir / "summary.json"
    broken_summary.write_text('{"schema_version": 1, "classification": ')

    index = summarize.build_index(tmp_path)

    assert index["total_attempts"] == 2
    assert index["baseline_stats"]["count"] == 0
    assert index["launch_ready_attempts"] == []
    broken_attempt = next(
        attempt
        for attempt in index["diagnostic_or_failed_attempts"]
        if attempt["summary"] == str(broken_summary)
    )
    assert broken_attempt["classification"]["mode"] == "malformed"
    assert broken_attempt["summary_error"]["phase"] == "summary"
    assert "summary.json" in broken_attempt["summary_error"]["message"]
    assert broken_attempt["included_in_baseline_stats"] is False


def test_stale_artifacts_include_malformed_summary_failed_attempt(tmp_path: Path):
    broken_dir = tmp_path / "lane_b_full_broken_summary"
    broken_dir.mkdir()
    broken_summary = broken_dir / "summary.json"
    broken_summary.write_text('{"schema_version": 1, "classification": ')

    index = summarize.build_index(tmp_path)

    assert len(index["diagnostic_or_failed_attempts"]) == 1
    assert index["diagnostic_or_failed_attempts"][0]["summary"] == str(
        broken_summary
    )
    assert index["diagnostic_or_failed_attempts"][0]["classification"]["mode"] == (
        "malformed"
    )
    assert index["stale_or_demoted_artifacts"] == [
        {
            "summary": str(broken_summary),
            "reason": "legacy_summary_schema",
            "message": "legacy or malformed summary schema",
            "operator_action": (
                "Archived or demoted artifact; do not use as launch-ready "
                "or baseline evidence."
            ),
            "lane": "unknown",
            "mode": "malformed",
            "claim_label": "malformed summary",
        }
    ]


def test_run_index_rejects_malformed_final_metrics_without_crashing(tmp_path: Path):
    result_dir = tmp_path / "lane_a_full_bad_metrics"
    _summary(
        result_dir=result_dir,
        run_id="lane_a_full_bad_metrics",
        lane="A",
        mode="full",
        train_time=72.0,
        included=True,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["final_metrics"] = "not a metrics object"
    summary_path.write_text(json.dumps(summary) + "\n")

    index = summarize.build_index(tmp_path)

    assert index["baseline_stats"]["count"] == 0
    assert index["launch_ready_attempts"] == []
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["summary"] == str(summary_path)
    assert attempt["final_metrics"] == {}
    assert attempt["metrics_error"]["phase"] == "summary"
    assert "final_metrics" in attempt["metrics_error"]["message"]
    assert attempt["included_in_baseline_stats"] is False


def test_run_index_rejects_nonnumeric_baseline_train_time(tmp_path: Path):
    result_dir = tmp_path / "lane_a_full_bad_train_time"
    _summary(
        result_dir=result_dir,
        run_id="lane_a_full_bad_train_time",
        lane="A",
        mode="full",
        train_time=72.0,
        included=True,
    )
    _make_valid_baseline_evidence(result_dir)
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["final_metrics"]["train_time"] = "72 seconds"
    summary_path.write_text(json.dumps(summary) + "\n")

    index = summarize.build_index(tmp_path)

    assert index["baseline_stats"]["count"] == 0
    assert index["launch_ready_attempts"] == []
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["summary"] == str(summary_path)
    assert attempt["metrics_error"]["phase"] == "summary"
    assert "final_metrics.train_time" in attempt["metrics_error"]["message"]
    assert attempt["included_in_baseline_stats"] is False


def test_run_index_rejects_high_validation_loss_baseline_claim(tmp_path: Path):
    result_dir = tmp_path / "lane_a_full_high_loss"
    _summary(
        result_dir=result_dir,
        run_id="lane_a_full_high_loss",
        lane="A",
        mode="full",
        train_time=72.0,
        included=True,
    )
    _make_valid_baseline_evidence(result_dir)
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["final_metrics"]["val_loss"] = 3.29
    summary_path.write_text(json.dumps(summary) + "\n")

    index = summarize.build_index(tmp_path)

    assert index["baseline_stats"]["count"] == 0
    assert index["launch_ready_attempts"] == []
    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["summary"] == str(summary_path)
    assert attempt["metrics_error"]["phase"] == "summary"
    assert "final_metrics.val_loss" in attempt["metrics_error"]["message"]
    assert attempt["included_in_baseline_stats"] is False


def test_run_index_preserves_environment_and_hardware_sidecar_evidence(tmp_path: Path):
    _summary(
        result_dir=tmp_path / "lane_b_full_gate",
        run_id="lane_b_full_gate",
        lane="B",
        mode="full",
        train_time=None,
        included=False,
        blocker={
            "phase": "launch_authority",
            "message": "full launch requires authorization",
        },
        environment={
            "python": "3.12.3",
            "torch": "2.13.0+cu132",
            "cuda_runtime": "13.2",
            "triton": "3.7.1",
            "flash_attention": "2.8.3.post1",
        },
        gpus=[
            {"index": 0, "name": "NVIDIA B200", "compute_capability": [10, 0]},
            {"index": 1, "name": "NVIDIA B200", "compute_capability": [10, 0]},
        ],
        environment_sidecar={
            "schema_version": 1,
            "kind": "preflight_environment",
            "environment": {"torch": "2.13.0+cu132"},
        },
        hardware_sidecar={
            "schema_version": 1,
            "kind": "preflight_gpus",
            "gpus": [{"index": 0, "name": "NVIDIA B200"}],
        },
    )

    index = summarize.build_index(tmp_path)

    attempt = index["diagnostic_or_failed_attempts"][0]
    assert attempt["environment"]["torch"] == "2.13.0+cu132"
    assert attempt["environment_sidecar"]["kind"] == "preflight_environment"
    assert attempt["hardware_sidecar"]["kind"] == "preflight_gpus"
    assert attempt["gpu_count"] == 2
    assert attempt["gpus"][0]["name"] == "NVIDIA B200"


def test_run_index_preserves_legacy_string_classification_as_failed_attempt(
    tmp_path: Path,
):
    result_dir = tmp_path / "legacy_lane_a"
    result_dir.mkdir()
    result_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "classification": "B200 upstream reproduction Lane A preflight failure",
                "success": False,
                "blocker": "Flash Attention 3 failed on B200",
                "upstream_commit": "ecbb586296d3dac36fd206211f25d63bad4a6b35",
            }
        )
        + "\n"
    )

    index = summarize.build_index(tmp_path)

    assert index["total_attempts"] == 1
    assert index["baseline_stats"]["count"] == 0
    assert index["diagnostic_or_failed_attempts"][0]["classification"] == {
        "lane": "unknown",
        "mode": "legacy",
        "arm": "unknown",
        "claim_label": "B200 upstream reproduction Lane A preflight failure",
        "environment_class": "unknown",
        "evidence_tier": "legacy",
    }
    assert index["diagnostic_or_failed_attempts"][0]["blocker"] == {
        "phase": "unknown",
        "message": "Flash Attention 3 failed on B200",
    }
    assert index["stale_or_demoted_artifacts"] == [
        {
            "summary": str(result_dir / "summary.json"),
            "reason": "legacy_summary_schema",
            "message": "legacy or malformed summary schema",
            "operator_action": (
                "Archived or demoted artifact; do not use as launch-ready "
                "or baseline evidence."
            ),
            "lane": "unknown",
            "mode": "legacy",
            "claim_label": "B200 upstream reproduction Lane A preflight failure",
        }
    ]


def test_stale_artifact_list_is_bounded(tmp_path: Path):
    for index in range(25):
        result_dir = tmp_path / f"legacy_artifact_{index:02d}"
        result_dir.mkdir()
        result_dir.joinpath("summary.json").write_text(
            json.dumps(
                {
                    "classification": f"legacy artifact {index:02d}",
                    "success": False,
                    "blocker": "legacy run-index shape",
                }
            )
            + "\n"
        )

    index = summarize.build_index(tmp_path)

    stale = index["stale_or_demoted_artifacts"]
    assert len(stale) == 20
    assert stale[0]["summary"].endswith("legacy_artifact_00/summary.json")
    assert stale[-1]["summary"].endswith("legacy_artifact_19/summary.json")
    assert all(record["reason"] == "legacy_summary_schema" for record in stale)
