# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wave F2 done-criteria: preflight readiness attaches to the canonical report.

These tests pin the seam between the execution lifecycle's preflight query and
the scaffold's report builder. The scaffold's build_report_input stays the
scientific schema owner; the preflight adapter only folds one boolean into the
flat checks view and adds a validated execution section. A blocked preflight
must surface as a failing check without fabricating a score, and a ready
preflight must integrate without disturbing the existing checks.
"""

from __future__ import annotations

import json
from pathlib import Path

from torchtitan.experiments.execution.preflight import profiles, query, report_attach
from torchtitan.experiments.scaffold_to_policy import report_artifacts


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _minimal_registry(path: Path) -> Path:
    # A selected registry with one split; num_problems must match the summary.
    return _write_json(
        path,
        {"selected": True, "splits": {"dev": {"num_problems": 10}}},
    )


def _minimal_summary(path: Path) -> Path:
    return _write_json(path, {"num_problems": 10, "pass_at_k": {"1": 0.4, "8": 0.7}})


def _build_report(tmp_path: Path, preflight: dict[str, object]) -> dict[str, object]:
    extra_checks, execution_section = report_attach.to_report_sections(preflight)
    registry = _minimal_registry(tmp_path / "split_registry.json")
    summary = _minimal_summary(tmp_path / "summary_dev.json")
    return report_artifacts.build_report_input(
        data_root=tmp_path / "data",
        results_root=tmp_path / "results",
        run_id="20260813T000000Z-f2",
        task="math_style",
        lane="reasoning",
        scaffold={"kind": "raw"},
        split_registry=registry,
        summary_paths={"dev": summary},
        verifier={"kind": "arithmetic"},
        summary_to_registry_split={"dev": "dev"},
        extra_checks=extra_checks,
        extra_sections={"execution": execution_section},
    )


def test_ready_preflight_attaches_without_disturbing_checks(tmp_path):
    # A fully provisioned rootfs_cpu profile: in rootfs with torch importable.
    env = profiles.ProbeEnv(
        in_rootfs=True,
        available_packages={"torch"},
        available_executables=set(),
        gpu_count=0,
    )
    preflight = query.run_preflight(profile_names=["rootfs_cpu"], env=env)
    assert preflight["readiness"] == "ready"

    report = _build_report(tmp_path, preflight)

    # The adapter's boolean rides alongside the scaffold's own checks.
    assert report["checks"]["preflight_ready"] is True
    assert report["checks"]["split_registry_selected"] is True
    assert report["checks"]["summary_split_counts_match"] is True
    # The execution section carries the full readiness verdict.
    assert report["execution"]["kind"] == "execution_preflight"
    assert report["execution"]["readiness"] == "ready"
    assert report["execution"]["blocker_codes"] == []


def test_blocked_preflight_surfaces_failing_check_without_score(tmp_path):
    # rootfs_cpu requested from the host: not in rootfs, torch absent.
    env = profiles.ProbeEnv(
        in_rootfs=False,
        available_packages=set(),
        available_executables=set(),
        gpu_count=0,
    )
    preflight = query.run_preflight(profile_names=["rootfs_cpu"], env=env)
    assert preflight["readiness"] == "blocked"
    assert preflight["execution_outcome"] == "blocked"

    report = _build_report(tmp_path, preflight)

    # A blocked preflight is a failing check, so a report gate can reject it.
    assert report["checks"]["preflight_ready"] is False
    # The blocker codes are stable and carry no score in their place.
    assert "rootfs_not_active" in report["execution"]["blocker_codes"]
    assert "torch_not_importable" in report["execution"]["blocker_codes"]
    # Metrics still reflect only the supplied summaries, never a fabricated one.
    assert set(report["metrics"]["splits"]) == {"dev"}


def test_failing_semantic_check_blocks_even_when_runtime_ready(tmp_path):
    # Runtime is ready, but a task-owned semantic check fails: still blocked.
    env = profiles.ProbeEnv(
        in_rootfs=True,
        available_packages={"torch"},
        available_executables=set(),
        gpu_count=0,
    )
    semantic = [
        query.SemanticCheck(
            name="matched_coverage",
            passed=False,
            blocker_code="insufficient_matched_coverage",
            details={"train": 120, "required": 300},
        )
    ]
    preflight = query.run_preflight(
        profile_names=["rootfs_cpu"], env=env, semantic_checks=semantic
    )
    assert preflight["readiness"] == "blocked"

    report = _build_report(tmp_path, preflight)

    assert report["checks"]["preflight_ready"] is False
    assert "insufficient_matched_coverage" in report["execution"]["blocker_codes"]
