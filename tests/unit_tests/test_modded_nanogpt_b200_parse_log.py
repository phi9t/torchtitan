# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path

from experiments.modded_nanogpt_b200 import parse_log


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data) + "\n")


def _classification(mode: str = "full", lane: str = "A") -> dict:
    return {
        "lane": lane,
        "mode": mode,
        "arm": "A0" if lane == "A" else "B0",
        "claim_label": "B200 upstream reproduction"
        if lane == "A" and mode == "full"
        else "diagnostic",
        "evidence_tier": "full-single-attempt" if mode == "full" else mode,
        "run_id": f"lane_{lane.lower()}_{mode}_fixture",
        "attempt_id": f"lane_{lane.lower()}_{mode}_fixture_attempt_001",
        "environment_class": "torchtitan-rootfs-b200",
        "claim_eligible": lane == "A" and mode == "full",
    }


def _write_full_attempt_evidence(result_dir: Path, *, lane: str = "A") -> None:
    manifest_files = [
        {
            "path": str(result_dir / "data" / f"shard_{idx:02d}.bin"),
            "bytes": 200001024,
            "sha256": f"{idx:064x}",
        }
        for idx in range(10)
    ]
    _write_json(
        result_dir / "data_manifest.json",
        {
            "schema_version": 1,
            "dataset": "fineweb10B",
            "token_budget": "900M",
            "source": {"commit": "ecbb586296d3dac36fd206211f25d63bad4a6b35"},
            "files": manifest_files,
            "num_files": 10,
            "total_bytes": 2000010240,
            "verified_sha": True,
        },
    )
    preflight_path = result_dir / "preflight_report.json"
    preflight = (
        json.loads(preflight_path.read_text()) if preflight_path.exists() else {}
    )
    preflight["checks"] = [
        {
            "name": "data_manifest",
            "ok": True,
            "detail": {
                "dataset": "fineweb10B",
                "token_budget": "900M",
                "num_files": 10,
                "total_bytes": 2000010240,
                "verified_sha": True,
            },
        },
        {
            "name": "nccl_all_reduce",
            "ok": True,
            "detail": {"world_size": 8},
        },
    ]
    preflight.setdefault(
        "gpus", [{"index": i, "name": "NVIDIA B200"} for i in range(8)]
    )
    _write_json(preflight_path, preflight)
    _write_json(
        result_dir / "launch_readiness.json",
        {
            "schema_version": 1,
            "lane": lane,
            "mode": "full",
            "ready_to_launch": True,
            "training_launched": True,
            "launch_authority_required": False,
            "preflight_ok": True,
            "verify_sha": True,
            "allow_previous_stall": False,
            "skip_run": False,
            "full_mode_gates": {
                "full_mode": True,
                "verify_sha_requested": True,
                "data_manifest_checked": True,
                "verified_sha": True,
                "nccl_checked": True,
                "manifest_token_budget": "900M",
                "manifest_num_files": 10,
                "manifest_total_bytes": 2000010240,
            },
            "blocked_by": [],
        },
    )
    _write_json(result_dir / "exit_code.json", {"exit_code": 0, "phase": "training"})
    telemetry = result_dir / "telemetry"
    telemetry.mkdir()
    _write_json(
        telemetry / "rootfs_environment.json",
        {
            "cwd": "/workspace/torchtitan",
            "python_executable": "/usr/bin/python",
            "torchtitan_in_rootfs": "1",
            "workspace_sentinel_exists": True,
        },
    )


def test_successful_full_lane_a_summary_requires_final_validation(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "Python 3.12.3",
                "PyTorch 2.13.0+cu132",
                "CUDA 13.2",
                "Triton 3.7.1",
                "Peak memory allocated: 123.5 GiB",
                "Peak memory reserved: 130.0 GiB",
                "step_avg: 0.812 sec",
                "final validation loss: 3.2742",
                "train_time: 72.45 seconds",
            ]
        )
        + "\n"
    )
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": _classification(),
            "environment": {"torch": "2.13.0+cu132"},
            "gpus": [{"index": i, "name": "NVIDIA B200"} for i in range(8)],
        },
    )
    wall_clock = tmp_path / "wall_clock.json"
    _write_json(wall_clock, {"elapsed_seconds": 95.0})
    _write_full_attempt_evidence(tmp_path, lane="A")
    source = tmp_path / "source"
    source.mkdir()
    _write_json(
        tmp_path / "source.json",
        {
            "path": str(source),
            "commit": "ecbb586296d3dac36fd206211f25d63bad4a6b35",
            "status_before": "",
        },
    )
    (tmp_path / "source_status_after.txt").write_text("")

    summary = parse_log.build_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=source,
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=wall_clock,
    )

    assert summary["ok"] is True
    assert summary["classification"] == _classification()
    assert summary["final_validation_reached"] is True
    assert summary["final_metrics"]["val_loss"] == 3.2742
    assert summary["final_metrics"]["train_time"] == 72.45
    assert summary["final_metrics"]["step_avg"] == 0.812
    assert summary["wall_clock"]["elapsed_seconds"] == 95.0
    assert summary["included_in_baseline_stats"] is True
    assert summary["claim_validation"] == {
        "successful_b200_reproduction": True,
        "mode_full": True,
        "lane_a": True,
        "preflight_ok": True,
        "nccl_checked": True,
        "sha_verified": True,
        "source_clean": True,
        "final_validation_reached": True,
        "val_loss_within_target": True,
        "train_time_reported": True,
        "shell_wall_clock_reported": True,
        "first_blocker": None,
    }


def test_final_metrics_parse_speedrun_ms_and_mib_format(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "step:1285/1285 train_time:443531ms step_avg:345.16ms",
                "step:1285/1285 val_loss:3.2873 train_time:443563ms step_avg:345.19ms",
                "peak memory allocated: 142696 MiB reserved: 150356 MiB",
            ]
        )
        + "\n"
    )
    preflight = tmp_path / "preflight_report.json"
    classification = {
        **_classification(mode="full", lane="B"),
        "claim_label": "B200 prerequisite torch-MLP fallback",
        "claim_eligible": False,
    }
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": classification,
            "environment": {"torch": "2.13.0+cu132"},
            "gpus": [{"index": i, "name": "NVIDIA B200"} for i in range(2)],
        },
    )

    summary = parse_log.build_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    assert summary["final_validation_reached"] is True
    assert summary["final_metrics"] == {
        "val_loss": 3.2873,
        "train_time": 443.563,
        "step_avg": 0.34519,
        "peak_allocated_memory": 142696 / 1024,
        "peak_reserved_memory": 150356 / 1024,
    }
    assert summary["included_in_baseline_stats"] is False
    assert summary["claim_validation"]["successful_b200_reproduction"] is False


def test_data_manifest_pointer_resolves_repo_relative_path(
    tmp_path: Path, monkeypatch,
):
    repo_root = tmp_path / "repo"
    monkeypatch.setattr(parse_log, "REPO_ROOT", repo_root)
    full_manifest = repo_root / "experiments" / "full_manifest.json"
    full_manifest.parent.mkdir(parents=True)
    _write_json(
        full_manifest,
        {
            "schema_version": 1,
            "dataset": "fineweb10B",
            "token_budget": "900M",
            "source": {"commit": "ecbb586296d3dac36fd206211f25d63bad4a6b35"},
            "files": [
                {
                    "path": str(tmp_path / "data" / f"shard_{idx:02d}.bin"),
                    "bytes": 200001024,
                    "sha256": f"{idx:064x}",
                }
                for idx in range(10)
            ],
            "num_files": 10,
            "total_bytes": 2000010240,
            "verified_sha": True,
        },
    )
    pointer_dir = repo_root / "results" / "run"
    pointer_dir.mkdir(parents=True)
    pointer = pointer_dir / "data_manifest.json"
    repo_relative = full_manifest.relative_to(parse_log.REPO_ROOT)
    _write_json(
        pointer,
        {
            "schema_version": 1,
            "kind": "data_manifest_pointer",
            "path": str(repo_relative),
        },
    )

    summary = parse_log._data_manifest_summary(pointer)

    assert summary["resolved_path"] == str(full_manifest)
    assert summary["resolved_exists"] is True
    assert summary["dataset"] == "fineweb10B"
    assert summary["token_budget"] == "900M"
    assert summary["verified_sha"] is True
    assert summary["num_files"] == 10


def test_full_attempt_accepts_repo_relative_launch_manifest_path(
    tmp_path: Path, monkeypatch,
):
    repo_root = tmp_path / "repo"
    monkeypatch.setattr(parse_log, "REPO_ROOT", repo_root)
    manifest = repo_root / "experiments" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n")

    blocker = parse_log._full_attempt_evidence_blocker(
        _classification(mode="full", lane="B"),
        launch_readiness={
            "training_launched": True,
            "skip_run": False,
            "allow_previous_stall": False,
            "preflight_ok": True,
            "data_manifest": "experiments/manifest.json",
            "full_mode_gates": {
                "verify_sha_requested": True,
                "data_manifest_checked": True,
                "verified_sha": True,
                "manifest_token_budget": "900M",
                "manifest_num_files": parse_log.EXPECTED_FINEWEB_SHARDS,
                "manifest_total_bytes": parse_log.EXPECTED_FINEWEB_BYTES,
                "nccl_checked": True,
            },
        },
        exit_code={"phase": "training", "exit_code": 0},
        telemetry={
            "rootfs": {
                "torchtitan_in_rootfs": "1",
                "cwd": "/workspace/torchtitan",
                "workspace_sentinel_exists": True,
            },
        },
        data_manifest={
            "path": str(repo_root / "results" / "run" / "data_manifest.json"),
            "resolved_path": str(manifest),
            "verified_sha": True,
            "token_budget": "900M",
            "num_files": parse_log.EXPECTED_FINEWEB_SHARDS,
            "total_bytes": parse_log.EXPECTED_FINEWEB_BYTES,
            "source_commit": parse_log.UPSTREAM_COMMIT,
        },
        preflight_data_manifest={"verified_sha": True},
        gpus=[{"name": "NVIDIA B200"} for _ in range(8)],
    )

    assert blocker is None


def test_full_baseline_rejects_nonzero_training_exit_code(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "Peak memory allocated: 123.5 GiB",
                "Peak memory reserved: 130.0 GiB",
                "step_avg: 0.812 sec",
                "final validation loss: 3.2742",
                "train_time: 72.45 seconds",
            ]
        )
        + "\n"
    )
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": _classification(),
            "environment": {"torch": "2.13.0+cu132"},
            "gpus": [{"index": i, "name": "NVIDIA B200"} for i in range(8)],
        },
    )
    _write_full_attempt_evidence(tmp_path, lane="A")
    _write_json(tmp_path / "exit_code.json", {"exit_code": 7, "phase": "training"})
    source = tmp_path / "source"
    source.mkdir()
    _write_json(
        tmp_path / "source.json",
        {
            "path": str(source),
            "commit": "ecbb586296d3dac36fd206211f25d63bad4a6b35",
            "status_before": "",
        },
    )
    (tmp_path / "source_status_after.txt").write_text("")

    summary = parse_log.build_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=source,
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    assert summary["ok"] is False
    assert summary["included_in_baseline_stats"] is False
    assert summary["blocker"] == {
        "phase": "training",
        "message": "full baseline requires training exit_code=0, found 7",
    }


def test_full_baseline_rejects_final_metrics_without_full_attempt_evidence(
    tmp_path: Path,
):
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "Peak memory allocated: 123.5 GiB",
                "Peak memory reserved: 130.0 GiB",
                "step_avg: 0.812 sec",
                "final validation loss: 3.2742",
                "train_time: 72.45 seconds",
            ]
        )
        + "\n"
    )
    preflight = tmp_path / "preflight_report.json"
    _write_json(preflight, {"ok": True, "classification": _classification()})
    source = tmp_path / "source"
    source.mkdir()
    _write_json(
        tmp_path / "source.json",
        {
            "path": str(source),
            "commit": "ecbb586296d3dac36fd206211f25d63bad4a6b35",
            "status_before": "",
        },
    )
    (tmp_path / "source_status_after.txt").write_text("")

    summary = parse_log.build_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=source,
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    assert summary["ok"] is False
    assert summary["included_in_baseline_stats"] is False
    assert summary["blocker"] == {
        "phase": "launch_evidence",
        "message": "full baseline requires launch_readiness.training_launched=true",
    }


def test_full_baseline_requires_launch_readiness_manifest_match(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "Peak memory allocated: 123.5 GiB",
                "Peak memory reserved: 130.0 GiB",
                "step_avg: 0.812 sec",
                "final validation loss: 3.2742",
                "train_time: 72.45 seconds",
            ]
        )
        + "\n"
    )
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": {
                **_classification(mode="full", lane="B"),
                "claim_label": "B200 compatibility patchset",
                "claim_eligible": True,
            },
            "environment": {"torch": "2.13.0+cu132"},
            "gpus": [{"index": i, "name": "NVIDIA B200"} for i in range(8)],
        },
    )
    _write_json(
        tmp_path / "variant_patch_classification.json",
        {
            "schema_version": 1,
            "source": "variant",
            "diff_artifact": "variant_patch.diff",
            "files": [
                {
                    "path": "train_gpt.py",
                    "patch_class": "hardware-detection",
                    "first_lane_a_blocker": "FA3 no kernel image on B200",
                }
            ],
        },
    )
    _write_full_attempt_evidence(tmp_path, lane="B")
    readiness_path = tmp_path / "launch_readiness.json"
    readiness = json.loads(readiness_path.read_text())
    readiness[
        "data_manifest"
    ] = "experiments/modded_nanogpt_b200/results/other/data_manifest.json"
    readiness_path.write_text(json.dumps(readiness) + "\n")

    summary = parse_log.build_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    assert summary["ok"] is False
    assert summary["included_in_baseline_stats"] is False
    assert summary["blocker"] == {
        "phase": "data_manifest",
        "message": "full baseline requires launch_readiness.data_manifest to match parsed manifest",
    }


def test_full_baseline_accepts_launch_readiness_resolved_manifest_for_pointer(
    tmp_path: Path,
):
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "Peak memory allocated: 123.5 GiB",
                "Peak memory reserved: 130.0 GiB",
                "step_avg: 0.812 sec",
                "final validation loss: 3.2742",
                "train_time: 72.45 seconds",
            ]
        )
        + "\n"
    )
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": {
                **_classification(mode="full", lane="B"),
                "claim_label": "B200 compatibility patchset",
                "claim_eligible": True,
            },
            "environment": {"torch": "2.13.0+cu132"},
            "gpus": [{"index": i, "name": "NVIDIA B200"} for i in range(8)],
        },
    )
    _write_json(
        tmp_path / "variant_patch_classification.json",
        {
            "schema_version": 1,
            "source": "variant",
            "diff_artifact": "variant_patch.diff",
            "files": [
                {
                    "path": "train_gpt.py",
                    "patch_class": "hardware-detection",
                    "first_lane_a_blocker": "FA3 no kernel image on B200",
                }
            ],
        },
    )
    full_manifest = tmp_path / "full_manifest.json"
    _write_json(
        full_manifest,
        {
            "schema_version": 1,
            "dataset": "fineweb10B",
            "token_budget": "900M",
            "source": {"commit": "ecbb586296d3dac36fd206211f25d63bad4a6b35"},
            "files": [
                {
                    "path": str(tmp_path / "data" / f"shard_{idx:02d}.bin"),
                    "bytes": 200001024,
                    "sha256": f"{idx:064x}",
                }
                for idx in range(10)
            ],
            "num_files": 10,
            "total_bytes": 2000010240,
            "verified_sha": True,
        },
    )
    _write_full_attempt_evidence(tmp_path, lane="B")
    pointer = tmp_path / "data_manifest.json"
    _write_json(
        pointer,
        {
            "schema_version": 1,
            "kind": "data_manifest_pointer",
            "path": str(full_manifest),
        },
    )
    readiness_path = tmp_path / "launch_readiness.json"
    readiness = json.loads(readiness_path.read_text())
    readiness["data_manifest"] = str(full_manifest)
    readiness_path.write_text(json.dumps(readiness) + "\n")

    summary = parse_log.build_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=pointer,
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    assert summary["ok"] is True
    assert summary["data_manifest_summary"]["path"] == str(pointer)
    assert summary["data_manifest_summary"]["resolved_path"] == str(full_manifest)


def test_full_lane_a_requires_source_cleanliness_evidence(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "Peak memory allocated: 123.5 GiB",
                "Peak memory reserved: 130.0 GiB",
                "step_avg: 0.812 sec",
                "final validation loss: 3.2742",
                "train_time: 72.45 seconds",
            ]
        )
        + "\n"
    )
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": _classification(),
        },
    )

    summary = parse_log.build_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "missing_source",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    assert summary["ok"] is False
    assert summary["included_in_baseline_stats"] is False
    assert summary["blocker"] == {
        "phase": "source_policy",
        "message": "Lane A source cleanliness evidence is missing",
    }


def test_full_lane_a_dirty_source_invalidates_success(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "Peak memory allocated: 123.5 GiB",
                "Peak memory reserved: 130.0 GiB",
                "step_avg: 0.812 sec",
                "final validation loss: 3.2742",
                "train_time: 72.45 seconds",
            ]
        )
        + "\n"
    )
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": _classification(),
        },
    )
    source = tmp_path / "source"
    source.mkdir()
    _write_json(
        tmp_path / "source.json",
        {
            "path": str(source),
            "commit": "ecbb586296d3dac36fd206211f25d63bad4a6b35",
            "status_before": "",
        },
    )
    (tmp_path / "source_status_after.txt").write_text(" M train_gpt.py\n")

    summary = parse_log.build_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=source,
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    assert summary["ok"] is False
    assert summary["included_in_baseline_stats"] is False
    assert summary["blocker"] == {
        "phase": "source_policy",
        "message": "Lane A source was dirty after launch",
    }


def test_successful_full_lane_b_summary_can_be_baseline_eligible(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "Peak memory allocated: 123.5 GiB",
                "Peak memory reserved: 130.0 GiB",
                "step_avg: 0.812 sec",
                "final validation loss: 3.2742",
                "train_time: 72.45 seconds",
            ]
        )
        + "\n"
    )
    preflight = tmp_path / "preflight_report.json"
    classification = {
        **_classification(mode="full", lane="B"),
        "claim_label": "B200 compatibility patchset",
        "claim_eligible": True,
    }
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": classification,
            "environment": {"torch": "2.13.0+cu132"},
            "gpus": [{"index": i, "name": "NVIDIA B200"} for i in range(8)],
        },
    )
    _write_json(
        tmp_path / "variant_patch_classification.json",
        {
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
                    "first_lane_a_blocker": "Triton sm100 custom kernel compile/runtime blocker",
                },
            ],
        },
    )
    _write_full_attempt_evidence(tmp_path, lane="B")

    summary = parse_log.build_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    assert summary["ok"] is True
    assert summary["classification"]["lane"] == "B"
    assert summary["classification"]["claim_eligible"] is True
    assert summary["included_in_baseline_stats"] is True


def test_full_lane_b_unclassified_patch_invalidates_baseline(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "Peak memory allocated: 123.5 GiB",
                "Peak memory reserved: 130.0 GiB",
                "step_avg: 0.812 sec",
                "final validation loss: 3.2742",
                "train_time: 72.45 seconds",
            ]
        )
        + "\n"
    )
    preflight = tmp_path / "preflight_report.json"
    classification = {
        **_classification(mode="full", lane="B"),
        "claim_label": "B200 compatibility patchset",
        "claim_eligible": True,
    }
    _write_json(preflight, {"ok": True, "classification": classification})
    _write_json(
        tmp_path / "variant_patch_classification.json",
        {
            "schema_version": 1,
            "source": "variant",
            "diff_artifact": "variant_patch.diff",
            "files": [
                {
                    "path": "notes.md",
                    "patch_class": "unclassified",
                    "first_lane_a_blocker": "unclassified Lane B source difference",
                }
            ],
        },
    )

    summary = parse_log.build_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    assert summary["ok"] is False
    assert summary["included_in_baseline_stats"] is False
    assert summary["blocker"] == {
        "phase": "variant_patch_classification",
        "message": "Lane B baseline has unclassified source patches: notes.md",
    }


def test_partial_diagnostic_summary_keeps_metrics_null_and_classifies_blocker(
    tmp_path: Path,
):
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "warmup: compiling",
                "CUDA out of memory. Tried to allocate 256.00 GiB.",
                "rank 0 exited with code 1",
            ]
        )
        + "\n"
    )
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": _classification(mode="diagnostic", lane="B"),
            "environment": {"torch": "2.13.0+cu132"},
            "gpus": [{"index": i, "name": "NVIDIA B200"} for i in range(8)],
        },
    )

    summary = parse_log.build_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    assert summary["ok"] is False
    assert summary["classification"] == _classification(mode="diagnostic", lane="B")
    assert summary["final_validation_reached"] is False
    assert summary["final_metrics"] == {
        "val_loss": None,
        "train_time": None,
        "step_avg": None,
        "peak_allocated_memory": None,
        "peak_reserved_memory": None,
    }
    assert summary["included_in_baseline_stats"] is False
    assert summary["blocker"]["phase"] == "warmup"
    assert "out of memory" in summary["blocker"]["message"].lower()


def test_write_summary_also_writes_analysis_markdown(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text("CUDA out of memory. Tried to allocate 256.00 GiB.\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": _classification(mode="diagnostic", lane="B"),
            "environment": {"torch": "2.13.0+cu132"},
            "gpus": [{"index": i, "name": "NVIDIA B200"} for i in range(8)],
        },
    )

    parse_log.write_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    analysis = (tmp_path / "analysis.md").read_text()
    assert "# Modded NanoGPT B200 Attempt Analysis" in analysis
    assert "- lane: B" in analysis
    assert "- mode: diagnostic" in analysis
    assert "- claim_eligible: False" in analysis
    assert "- val_loss: null" in analysis
    assert "- train_time: null" in analysis
    assert "## Claim Validation" in analysis
    assert "- successful_b200_reproduction: False" in analysis
    assert "- claim_first_blocker: final validation was not reached" in analysis
    assert "- first failing phase: warmup" in analysis
    assert "CUDA out of memory" in analysis


def test_analysis_includes_data_manifest_and_telemetry_status(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text("skip-run requested; training was not launched\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": _classification(mode="diagnostic", lane="B"),
            "checks": [
                {
                    "name": "data_manifest",
                    "ok": True,
                    "detail": {
                        "verified_sha": True,
                        "num_files": 1,
                        "total_bytes": 123,
                    },
                }
            ],
        },
    )
    manifest = tmp_path / "data_manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "dataset": "fineweb10B",
            "token_budget": "smoke",
            "source": {"commit": "fixture-commit"},
            "num_files": 1,
            "total_bytes": 123,
            "files": [{"path": "fixture.bin", "bytes": 123, "sha256": "abc"}],
        },
    )
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    _write_json(
        telemetry / "watcher_status.json",
        {"state": "not_started", "reason": "skip_run"},
    )
    _write_json(
        telemetry / "dcgm_status.json",
        {
            "state": "unavailable",
            "reason": "no rootfs-friendly DCGM command found",
            "checked_commands": ["dcgmi"],
        },
    )
    _write_json(
        tmp_path / "source.json",
        {
            "path": "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa",
            "commit": "fixture-source-commit",
            "status_before": " M train_gpt.py\n",
        },
    )
    _write_json(
        tmp_path / "attempt.json",
        {
            "command": {
                "argv": [
                    "experiments/modded_nanogpt_b200/run_speedrun.sh",
                    "--lane",
                    "B",
                    "--mode",
                    "diagnostic",
                    "--skip-run",
                ],
                "training_argv": [
                    "torchrun",
                    "--standalone",
                    "--nproc_per_node=8",
                    "train_gpt.py",
                ],
            }
        },
    )

    summary = parse_log.write_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=manifest,
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    assert summary["attempt_command"]["argv"] == [
        "experiments/modded_nanogpt_b200/run_speedrun.sh",
        "--lane",
        "B",
        "--mode",
        "diagnostic",
        "--skip-run",
    ]
    assert summary["attempt_command"]["training_argv"] == [
        "torchrun",
        "--standalone",
        "--nproc_per_node=8",
        "train_gpt.py",
    ]
    analysis = (tmp_path / "analysis.md").read_text()
    assert (
        "- attempt_command: experiments/modded_nanogpt_b200/run_speedrun.sh --lane B --mode diagnostic --skip-run"
        in analysis
    )
    assert (
        "- training_command: torchrun --standalone --nproc_per_node=8 train_gpt.py"
        in analysis
    )
    assert "- dataset: fineweb10B" in analysis
    assert "- token_budget: smoke" in analysis
    assert "- data_source_commit: fixture-commit" in analysis
    assert "- manifest_verified_sha: null" in analysis
    assert "- preflight_verified_sha: True" in analysis
    assert "- num_files: 1" in analysis
    assert "- total_bytes: 123" in analysis
    assert "- sha256_entries: 1" in analysis
    assert "- attempt_source_commit: fixture-source-commit" in analysis
    assert "- source_status_before:  M train_gpt.py\\n" in analysis
    assert "- source_status_current: null" in analysis
    assert "- telemetry_state: not_started" in analysis
    assert "- telemetry_reason: skip_run" in analysis
    assert "- dcgm_state: unavailable" in analysis
    assert "- dcgm_reason: no rootfs-friendly DCGM command found" in analysis


def test_data_manifest_summary_resolves_result_local_pointer(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text("skip-run requested; training was not launched\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {"ok": True, "classification": _classification(mode="diagnostic", lane="B")},
    )
    source_manifest = tmp_path / "manifests" / "full_manifest.json"
    source_manifest.parent.mkdir()
    _write_json(
        source_manifest,
        {
            "schema_version": 1,
            "dataset": "fineweb10B",
            "token_budget": "900M",
            "source": {"commit": "fixture-data-commit"},
            "num_files": 10,
            "total_bytes": 2000010240,
            "files": [
                {"path": f"shard_{idx}.bin", "bytes": 200001024, "sha256": f"sha-{idx}"}
                for idx in range(10)
            ],
        },
    )
    pointer = tmp_path / "data_manifest.json"
    _write_json(
        pointer,
        {
            "schema_version": 1,
            "kind": "data_manifest_pointer",
            "path": str(source_manifest),
        },
    )

    summary = parse_log.write_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=pointer,
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    manifest_summary = summary["data_manifest_summary"]
    assert manifest_summary["path"] == str(pointer)
    assert manifest_summary["kind"] == "data_manifest_pointer"
    assert manifest_summary["resolved_path"] == str(source_manifest)
    assert manifest_summary["dataset"] == "fineweb10B"
    assert manifest_summary["token_budget"] == "900M"
    assert manifest_summary["source_commit"] == "fixture-data-commit"
    assert manifest_summary["num_files"] == 10
    assert manifest_summary["total_bytes"] == 2000010240
    assert manifest_summary["sha256_entries"] == 10
    analysis = (tmp_path / "analysis.md").read_text()
    assert "- data_manifest: " + str(pointer) in analysis
    assert "- data_manifest_kind: data_manifest_pointer" in analysis
    assert "- data_manifest_resolved_path: " + str(source_manifest) in analysis
    assert "- dataset: fineweb10B" in analysis


def test_data_manifest_summary_propagates_verified_sha_from_resolved_pointer_target(
    tmp_path: Path,
):
    log = tmp_path / "run.log"
    log.write_text("skip-run requested; training was not launched\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {"ok": True, "classification": _classification(mode="diagnostic", lane="B")},
    )
    target_manifest = tmp_path / "manifests" / "full_manifest.json"
    target_manifest.parent.mkdir()
    _write_json(
        target_manifest,
        {
            "schema_version": 1,
            "dataset": "fineweb10B",
            "token_budget": "900M",
            "source": {"commit": "ecbb586296d3dac36fd206211f25d63bad4a6b35"},
            "num_files": 10,
            "total_bytes": 2000010240,
            "verified_sha": True,
            "files": [
                {
                    "path": f"shard_{idx}.bin",
                    "bytes": 200001024,
                    "sha256": f"{idx:064x}",
                }
                for idx in range(10)
            ],
        },
    )
    pointer = tmp_path / "data_manifest.json"
    _write_json(
        pointer,
        {
            "schema_version": 1,
            "kind": "data_manifest_pointer",
            "path": "manifests/full_manifest.json",
        },
    )

    summary = parse_log.write_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=pointer,
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    manifest_summary = summary["data_manifest_summary"]
    assert manifest_summary["resolved_path"] == str(target_manifest)
    assert manifest_summary["verified_sha"] is True
    assert manifest_summary["token_budget"] == "900M"
    assert manifest_summary["num_files"] == 10
    assert manifest_summary["total_bytes"] == 2000010240
    assert (
        manifest_summary["source_commit"]
        == "ecbb586296d3dac36fd206211f25d63bad4a6b35"
    )
    analysis = (tmp_path / "analysis.md").read_text()
    assert "- manifest_verified_sha: True" in analysis


def test_data_manifest_summary_does_not_infer_verified_sha_from_shard_hashes(
    tmp_path: Path,
):
    log = tmp_path / "run.log"
    log.write_text("skip-run requested; training was not launched\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {"ok": True, "classification": _classification(mode="diagnostic", lane="B")},
    )
    target_manifest = tmp_path / "manifests" / "full_manifest.json"
    target_manifest.parent.mkdir()
    _write_json(
        target_manifest,
        {
            "schema_version": 1,
            "dataset": "fineweb10B",
            "token_budget": "900M",
            "source": {"commit": "ecbb586296d3dac36fd206211f25d63bad4a6b35"},
            "num_files": 10,
            "total_bytes": 2000010240,
            "files": [
                {
                    "path": f"shard_{idx}.bin",
                    "bytes": 200001024,
                    "sha256": f"{idx:064x}",
                }
                for idx in range(10)
            ],
        },
    )
    pointer = tmp_path / "data_manifest.json"
    _write_json(
        pointer,
        {
            "schema_version": 1,
            "kind": "data_manifest_pointer",
            "path": "manifests/full_manifest.json",
        },
    )

    summary = parse_log.write_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=pointer,
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    manifest_summary = summary["data_manifest_summary"]
    assert manifest_summary["resolved_path"] == str(target_manifest)
    assert manifest_summary["sha256_entries"] == 10
    assert manifest_summary.get("verified_sha") is not True
    analysis = (tmp_path / "analysis.md").read_text()
    assert "- manifest_verified_sha: True" not in analysis


def test_summary_and_analysis_include_variant_patch_classification(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text("skip-run requested; training was not launched\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": _classification(mode="full", lane="B"),
        },
    )
    _write_json(
        tmp_path / "variant_patch_classification.json",
        {
            "schema_version": 1,
            "source": "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa",
            "diff_artifact": "experiments/modded_nanogpt_b200/results/fixture/variant_patch.diff",
            "files": [
                {
                    "path": "train_gpt.py",
                    "patch_class": "hardware-detection",
                    "first_lane_a_blocker": "FA3 no kernel image on B200",
                },
                {
                    "path": "triton_kernels.py",
                    "patch_class": "kernel-compat",
                    "first_lane_a_blocker": "Triton sm100 custom kernel compile/runtime blocker",
                },
            ],
        },
    )

    summary = parse_log.write_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    assert summary["variant_patch_classification"]["files"][0]["path"] == "train_gpt.py"
    assert (
        summary["variant_patch_classification"]["files"][1]["patch_class"]
        == "kernel-compat"
    )
    analysis = (tmp_path / "analysis.md").read_text()
    assert (
        "- variant_patch_diff: experiments/modded_nanogpt_b200/results/fixture/variant_patch.diff"
        in analysis
    )
    assert "- variant_patch_file_count: 2" in analysis
    expected_train_patch = (
        "- variant_patch_file: train_gpt.py | class=hardware-detection | "
        "first_lane_a_blocker=FA3 no kernel image on B200"
    )
    expected_triton_patch = (
        "- variant_patch_file: triton_kernels.py | class=kernel-compat | "
        "first_lane_a_blocker=Triton sm100 custom kernel compile/runtime blocker"
    )
    assert expected_train_patch in analysis
    assert expected_triton_patch in analysis


def test_analysis_includes_mlp_preflight_detail(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text("preflight failed\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": False,
            "classification": _classification(mode="full", lane="B"),
            "failures": [
                {
                    "name": "mlp_backend",
                    "error": "MODDED_NANOGPT_MLP_BACKEND=torch is blocked for full mode",
                }
            ],
            "checks": [
                {
                    "name": "mlp_backend",
                    "ok": False,
                    "detail": {
                        "backend": "torch",
                        "local_smoke": {
                            "backend": "torch",
                            "output_shape": [2, 16, 768],
                        },
                        "full_mode_policy": "blocked_without_allow_previous_stall",
                    },
                }
            ],
        },
    )

    parse_log.write_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    analysis = (tmp_path / "analysis.md").read_text()
    assert "- first failing phase: mlp_backend" in analysis
    assert "- failure_category: MLP" in analysis
    assert "- mlp_backend: torch" in analysis
    assert "- mlp_local_smoke_backend: torch" in analysis
    assert "- mlp_local_smoke_output_shape: [2, 16, 768]" in analysis
    assert "- mlp_full_mode_policy: blocked_without_allow_previous_stall" in analysis


def test_summary_and_analysis_include_active_job_scan_evidence(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text("skip-run requested; training was not launched\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": {
                **_classification(mode="full", lane="B"),
                "claim_eligible": True,
            },
            "environment": {"torch": "2.13.0+cu132"},
            "gpus": [{"index": i, "name": "NVIDIA B200"} for i in range(8)],
        },
    )
    _write_json(
        tmp_path / "active_jobs.json",
        {
            "ok": True,
            "active_job_count": 0,
            "ignored_match_count": 1,
            "active_jobs": [],
        },
    )

    parse_log.write_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    analysis = (tmp_path / "analysis.md").read_text()
    assert summary["active_jobs"]["ok"] is True
    assert summary["active_jobs"]["active_job_count"] == 0
    assert summary["active_jobs"]["ignored_match_count"] == 1
    assert "- active_jobs_ok: True" in analysis
    assert "- active_job_count: 0" in analysis
    assert "- active_jobs_ignored_match_count: 1" in analysis


def test_analysis_includes_triton_mlp_compile_blocker_detail(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text("preflight failed\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": False,
            "classification": _classification(mode="full", lane="B"),
            "failures": [
                {
                    "name": "mlp_backend",
                    "error": (
                        "MODDED_NANOGPT_MLP_BACKEND=triton is blocked on B200: "
                        "prior smoke/full run failed compiling linear_relu_square_kernel"
                    ),
                }
            ],
            "checks": [
                {
                    "name": "mlp_backend",
                    "ok": False,
                    "detail": {
                        "backend": "triton",
                        "blocked_kernel": "linear_relu_square_kernel",
                        "blocked_arch": "sm100",
                        "failure_class": "triton_compile",
                        "compiler_pass": "TritonNvidiaGPUOptimizeTMemLayoutsPass",
                        "full_mode_policy": "blocked_without_allow_previous_stall",
                    },
                }
            ],
        },
    )

    parse_log.write_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    analysis = (tmp_path / "analysis.md").read_text()
    assert "- first failing phase: mlp_backend" in analysis
    assert "- mlp_backend: triton" in analysis
    assert "- mlp_blocked_kernel: linear_relu_square_kernel" in analysis
    assert "- mlp_blocked_arch: sm100" in analysis
    assert "- mlp_failure_class: triton_compile" in analysis
    assert "- mlp_compiler_pass: TritonNvidiaGPUOptimizeTMemLayoutsPass" in analysis
    assert "- mlp_full_mode_policy: blocked_without_allow_previous_stall" in analysis


def test_analysis_includes_telemetry_ranges_from_watcher_artifacts(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text("CUDA out of memory. Tried to allocate 256.00 GiB.\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": _classification(mode="diagnostic", lane="B"),
        },
    )
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    _write_json(
        telemetry / "watcher_status.json",
        {"state": "stopped", "reason": "training_finished"},
    )
    (telemetry / "nvidia_smi_query.csv").write_text(
        "\n".join(
            [
                (
                    "timestamp, index, name, uuid, temperature.gpu, power.draw [W], "
                    "clocks.sm [MHz], clocks.mem [MHz], utilization.gpu [%], "
                    "utilization.memory [%], memory.used [MiB], memory.total [MiB]"
                ),
                "2026/08/15 00:00:00.000, 0, NVIDIA B200, GPU-a, 55, 610.50, 1800, 4200, 92, 40, 180000, 190000",
                "2026/08/15 00:00:05.000, 0, NVIDIA B200, GPU-a, 88, 650.00, 990, 4200, 98, 50, 181000, 190000",
                "2026/08/15 00:00:00.000, 1, NVIDIA B200, GPU-b, 52, 600.00, 1790, 4200, 88, 35, 170000, 190000",
            ]
        )
        + "\n"
    )
    (telemetry / "process_watch.log").write_text(
        " 123 1 123 00:01 52428800 950.0 1.2 torchrun --standalone\n"
    )
    (telemetry / "disk_watch.log").write_text(
        "2026-08-15T00:00:00Z\n"
        "12G /workspace/torchtitan/experiments/modded_nanogpt_b200/results/run\n"
        "2.5G /workspace/torchtitan/experiments/modded_nanogpt_b200/data\n"
    )
    _write_json(
        telemetry / "stop_snapshot.json",
        {
            "reason": "training_exit_nonzero",
            "exit_code": 7,
            "commands": [
                {"artifact": "stop_ps_tree.txt", "returncode": 0},
                {"artifact": "stop_nvidia_smi_processes.txt", "returncode": 0},
            ],
        },
    )

    summary = parse_log.write_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    ranges = summary["telemetry"]["ranges"]
    assert ranges["gpu_utilization"]["min"] == 88.0
    assert ranges["gpu_utilization"]["median"] == 92.0
    assert ranges["gpu_utilization"]["max"] == 98.0
    assert ranges["power_draw_watts"]["max"] == 650.0
    assert ranges["memory_used_mib"]["peak"] == 181000.0
    assert summary["telemetry"]["process_watch"]["peak_rss_mib"] == 51200.0
    assert summary["telemetry"]["process_watch"]["peak_cpu_percent"] == 950.0
    assert summary["telemetry"]["disk_watch"]["max_result_size_bytes"] == 12 * 1024**3
    assert summary["telemetry"]["stop_snapshot"]["reason"] == "training_exit_nonzero"
    assert summary["telemetry"]["stop_snapshot"]["exit_code"] == 7
    assert summary["telemetry"]["signs"] == {
        "thermal_or_clock_throttling": True,
        "thermal_or_clock_throttling_reason": (
            "max GPU temperature 88.0C; minimum SM clock 990.0MHz"
        ),
        "cpu_or_rss_bottleneck": True,
        "cpu_or_rss_bottleneck_reason": (
            "peak process CPU 950.0%; peak process RSS 51200.0MiB"
        ),
    }
    analysis = (tmp_path / "analysis.md").read_text()
    assert "- gpu_utilization_percent: min=88.0 median=92.0 max=98.0" in analysis
    assert "- power_draw_watts: min=600.0 median=610.5 max=650.0" in analysis
    assert "- memory_used_mib_peak: 181000.0" in analysis
    assert "- process_peak_rss_mib: 51200.0" in analysis
    assert "- thermal_or_clock_throttling_signs: True" in analysis
    assert (
        "- thermal_or_clock_throttling_reason: max GPU temperature 88.0C; minimum SM clock 990.0MHz"
        in analysis
    )
    assert "- cpu_or_rss_bottleneck_signs: True" in analysis
    assert (
        "- cpu_or_rss_bottleneck_reason: peak process CPU 950.0%; peak process RSS 51200.0MiB"
        in analysis
    )
    assert "- disk_result_size_peak_bytes: 12884901888" in analysis
    assert "- stop_snapshot_reason: training_exit_nonzero" in analysis
    assert "- stop_snapshot_exit_code: 7" in analysis
    assert "- stop_snapshot_command_count: 2" in analysis


def test_analysis_includes_result_artifact_sizes(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text("skip-run requested; training was not launched\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": _classification(mode="diagnostic", lane="B"),
        },
    )
    (tmp_path / "command.argv").write_text("run_speedrun.sh --skip-run\n")
    (tmp_path / "operator_notes.md").write_text("# Attempt Notes\n")
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    _write_json(
        telemetry / "watcher_status.json",
        {"state": "not_started", "reason": "skip_run"},
    )
    _write_json(
        telemetry / "rootfs_environment.json",
        {
            "torchtitan_in_rootfs": "1",
            "cwd": "/workspace/torchtitan",
            "python_executable": "/usr/bin/python",
            "workspace_sentinel_exists": True,
        },
    )
    (telemetry / "process_watch.log").write_text("not started: skip_run\n")

    summary = parse_log.write_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    artifact_sizes = summary["artifact_sizes"]
    assert artifact_sizes["total_known_bytes"] > 0
    assert artifact_sizes["files"]["run.log"] == log.stat().st_size
    assert artifact_sizes["files"]["preflight_report.json"] == preflight.stat().st_size
    assert (
        artifact_sizes["files"]["telemetry/process_watch.log"]
        == (telemetry / "process_watch.log").stat().st_size
    )
    analysis = (tmp_path / "analysis.md").read_text()
    assert "- artifact_total_known_bytes: " in analysis
    assert "- artifact_size: run.log=" in analysis
    assert "- artifact_size: telemetry/process_watch.log=" in analysis


def test_analysis_includes_environment_gpu_wall_clock_and_failure_category(
    tmp_path: Path,
):
    log = tmp_path / "run.log"
    log.write_text("NCCL communicator abort during all_reduce\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": _classification(mode="diagnostic", lane="B"),
            "environment": {
                "python": "3.12.3",
                "torch": "2.13.0+cu132",
                "cuda_runtime": "13.2",
                "triton": "3.7.1",
                "flash_attention": "2.8.3.post1",
            },
            "gpus": [
                {"index": 0, "name": "NVIDIA B200", "compute_capability": [10, 0]},
                {"index": 1, "name": "NVIDIA B200", "compute_capability": [10, 0]},
            ],
        },
    )
    wall_clock = tmp_path / "wall_clock.json"
    _write_json(wall_clock, {"elapsed_seconds": 123.5})

    parse_log.write_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=wall_clock,
    )

    analysis = (tmp_path / "analysis.md").read_text()
    assert "- python: 3.12.3" in analysis
    assert "- torch: 2.13.0+cu132" in analysis
    assert "- cuda_runtime: 13.2" in analysis
    assert "- triton: 3.7.1" in analysis
    assert "- flash_attention: 2.8.3.post1" in analysis
    assert "- gpu_count: 2" in analysis
    assert "- gpu_inventory: 0:NVIDIA B200 cc=10.0; 1:NVIDIA B200 cc=10.0" in analysis
    assert "- shell_wall_clock_seconds: 123.5" in analysis
    assert "- failure_category: NCCL" in analysis


def test_summary_and_analysis_prefer_result_local_environment_hardware_sidecars(
    tmp_path: Path,
):
    log = tmp_path / "run.log"
    log.write_text("NCCL communicator abort during all_reduce\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": _classification(mode="diagnostic", lane="B"),
            "environment": {
                "python": "stale",
                "torch": "stale",
                "cuda_runtime": "stale",
                "triton": "stale",
                "flash_attention": "stale",
            },
            "gpus": [{"index": 0, "name": "stale", "compute_capability": [0, 0]}],
        },
    )
    _write_json(
        tmp_path / "environment.json",
        {
            "schema_version": 1,
            "kind": "preflight_environment",
            "environment": {
                "python": "3.12.3",
                "torch": "2.13.0+cu132",
                "cuda_runtime": "13.2",
                "triton": "3.7.1",
                "flash_attention": "2.8.3.post1",
            },
        },
    )
    _write_json(
        tmp_path / "hardware.json",
        {
            "schema_version": 1,
            "kind": "preflight_gpus",
            "gpus": [
                {"index": 0, "name": "NVIDIA B200", "compute_capability": [10, 0]},
                {"index": 1, "name": "NVIDIA B200", "compute_capability": [10, 0]},
            ],
        },
    )

    summary = parse_log.write_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    assert summary["environment_sidecar"]["kind"] == "preflight_environment"
    assert summary["hardware_sidecar"]["kind"] == "preflight_gpus"
    assert summary["environment"]["torch"] == "2.13.0+cu132"
    assert summary["gpus"][0]["name"] == "NVIDIA B200"
    assert len(summary["gpus"]) == 2
    analysis = (tmp_path / "analysis.md").read_text()
    assert "- environment_sidecar_kind: preflight_environment" in analysis
    assert "- hardware_sidecar_kind: preflight_gpus" in analysis
    assert "- torch: 2.13.0+cu132" in analysis
    assert "- gpu_count: 2" in analysis
    assert "- gpu_inventory: 0:NVIDIA B200 cc=10.0; 1:NVIDIA B200 cc=10.0" in analysis
    assert "stale" not in analysis


def test_skip_run_log_gets_explicit_not_launched_blocker(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text("preflight ok\nskip-run requested; training was not launched\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": _classification(mode="diagnostic", lane="B"),
        },
    )

    summary = parse_log.build_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    assert summary["ok"] is False
    assert summary["final_metrics"]["val_loss"] is None
    assert summary["included_in_baseline_stats"] is False
    assert summary["blocker"] == {
        "phase": "not_launched",
        "message": "skip-run requested; training was not launched",
    }


def test_runner_blocker_json_beats_generic_missing_validation_blocker(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "preflight ok\n"
        "CUDA error: no kernel image is available for execution on the device\n"
        "full launch requires --launch-authorization=launch-full-b200\n"
    )
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": _classification(mode="full", lane="B"),
        },
    )
    _write_json(
        tmp_path / "blocker.json",
        {
            "phase": "launch_authority",
            "message": "full launch requires --launch-authorization=launch-full-b200",
        },
    )
    _write_json(
        tmp_path / "launch_readiness.json",
        {
            "ready_to_launch": True,
            "training_launched": False,
            "launch_authority_required": True,
            "launch_authorization_required_token": "launch-full-b200",
            "preflight_ok": True,
            "blocked_by": [],
        },
    )

    summary = parse_log.write_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    assert summary["ok"] is False
    assert summary["final_validation_reached"] is False
    assert summary["blocker"] == {
        "phase": "launch_authority",
        "message": "full launch requires --launch-authorization=launch-full-b200",
    }
    assert summary["first_relevant_error"] == (
        "CUDA error: no kernel image is available for execution on the device"
    )
    assert summary["launch_readiness"]["ready_to_launch"] is True
    assert (
        summary["launch_readiness"]["launch_authorization_required_token"]
        == "launch-full-b200"
    )
    analysis = (tmp_path / "analysis.md").read_text()
    assert "- launch_ready_to_launch: True" in analysis
    assert "- launch_training_launched: False" in analysis
    assert "- launch_authorization_required_token: launch-full-b200" in analysis
    assert (
        "- first relevant error: CUDA error: no kernel image is available for execution on the device"
        in analysis
    )


def test_first_relevant_error_ignores_benign_stall_config_keys(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "{\n"
        '  "mode": "full",\n'
        '  "allow_previous_stall": false,\n'
        "}\n"
        "no-output stall detected after 900 seconds\n"
    )
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {"ok": True, "classification": _classification(mode="diagnostic", lane="B")},
    )

    summary = parse_log.build_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    assert (
        summary["first_relevant_error"] == "no-output stall detected after 900 seconds"
    )


def test_write_summary_json_is_atomic(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text("final validation loss: 3.40\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(preflight, {"ok": False, "classification": _classification()})

    summary = parse_log.write_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "source",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
    )

    written = json.loads((tmp_path / "summary.json").read_text())
    assert written == summary
    assert not (tmp_path / "summary.json.tmp").exists()


def test_cli_classification_overrides_legacy_preflight_without_classification(
    tmp_path: Path,
):
    log = tmp_path / "run.log"
    log.write_text("CUDA out of memory during warmup\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(preflight, {"ok": True})

    summary = parse_log.build_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
        classification_override={
            "lane": "B",
            "mode": "diagnostic",
            "arm": "B0",
            "run_id": "legacy_lane_b",
            "attempt_id": "legacy_lane_b_attempt_001",
        },
    )

    assert summary["classification"]["lane"] == "B"
    assert summary["classification"]["mode"] == "diagnostic"
    assert summary["classification"]["arm"] == "B0"
    assert summary["classification"]["claim_label"] == "diagnostic"
    assert summary["classification"]["evidence_tier"] == "diagnostic"
    assert summary["classification"]["claim_eligible"] is False


def test_cli_classification_preserves_explicit_non_claimable_full_override(
    tmp_path: Path,
):
    log = tmp_path / "run.log"
    log.write_text("RuntimeError: function FusedSoftcappedCrossEntropyBackward\n")
    preflight = tmp_path / "preflight_report.json"
    _write_json(
        preflight,
        {
            "ok": True,
            "classification": {
                **_classification(mode="full", lane="B"),
                "claim_label": "B200 prerequisite torch-MLP fallback",
                "claim_eligible": False,
            },
        },
    )

    summary = parse_log.build_summary(
        log_path=log,
        result_dir=tmp_path,
        source_path=tmp_path / "variant",
        data_manifest_path=tmp_path / "data_manifest.json",
        preflight_report_path=preflight,
        wall_clock_path=None,
        classification_override={
            "lane": "B",
            "mode": "full",
            "arm": "B0",
            "claim_label": "B200 prerequisite torch-MLP fallback",
            "evidence_tier": "full-single-attempt",
            "run_id": "lane_b_torch_mlp_prereq",
            "attempt_id": "lane_b_torch_mlp_prereq_attempt_001",
            "claim_eligible": False,
        },
    )

    classification = summary["classification"]
    assert classification["claim_label"] == "B200 prerequisite torch-MLP fallback"
    assert classification["evidence_tier"] == "full-single-attempt"
    assert classification["claim_eligible"] is False
