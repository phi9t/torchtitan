# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Wave F0 truth-preserving execution-foundation tests.

These cover the scaffold-to-policy runtime foundation fixes that must land
before expensive replicated research: append-only run manifests, explicit
status dimensions, latest-valid versus latest-attempt report indexing, report
identity and pass@k-budget validation, and rootfs destination safety.

The shell tests exercise ``experiments/scaffold_to_policy/run_common.sh`` and
``scripts/rootfs/build_rootfs.sh`` directly with bash; they do not require the
bwrap rootfs, GPUs, or Docker.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from torchtitan.experiments.execution import models
from torchtitan.experiments.scaffold_to_policy import execution_status, report_artifacts


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_COMMON = REPO_ROOT / "experiments" / "scaffold_to_policy" / "run_common.sh"
BUILD_ROOTFS = REPO_ROOT / "scripts" / "rootfs" / "build_rootfs.sh"
ROOTFS_TARGET = REPO_ROOT / "scripts" / "rootfs" / "rootfs_target.sh"
ENTER_ROOTFS = REPO_ROOT / "scripts" / "rootfs" / "enter_rootfs.sh"


def _run_manifest_script(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    """Source run_common.sh outside the rootfs and run a manifest scenario.

    ``TORCHTITAN_IN_ROOTFS=1`` keeps ``scaffold_enter_rootfs_if_needed`` from
    trying to re-enter bwrap; the manifest helpers under test do not need the
    rootfs.
    """

    script = f"""
set -euo pipefail
export TORCHTITAN_IN_ROOTFS=1
source "{RUN_COMMON}"
{body}
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def _manifest_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- Manifest append-only behavior -----------------------------------------


def test_setup_run_manifest_does_not_truncate_existing_rows(tmp_path):
    results_root = tmp_path / "results"
    manifest = results_root / "manifests" / "run-a.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"stage": "prior", "return_code": 0}) + "\n")

    result = _run_manifest_script(
        tmp_path,
        f"""
RUN_ID=run-a
RESULTS_ROOT="{results_root}"
scaffold_setup_run_manifest
""",
    )

    assert result.returncode == 0, result.stderr
    rows = _manifest_rows(manifest)
    stages = [row.get("stage") for row in rows]
    assert "prior" in stages, f"prior row was erased: {stages}"


def test_setup_run_manifest_records_attempt_marker(tmp_path):
    results_root = tmp_path / "results"
    manifest = results_root / "manifests" / "run-b.jsonl"

    result = _run_manifest_script(
        tmp_path,
        f"""
RUN_ID=run-b
RESULTS_ROOT="{results_root}"
scaffold_setup_run_manifest
""",
    )

    assert result.returncode == 0, result.stderr
    rows = _manifest_rows(manifest)
    markers = [row for row in rows if row.get("kind") == "manifest_attempt"]
    assert len(markers) == 1
    marker = markers[0]
    assert marker["run_id"] == "run-b"
    assert (
        isinstance(marker["stage_invocation_id"], str) and marker["stage_invocation_id"]
    )


def test_repeated_setup_run_manifest_appends_distinct_attempts(tmp_path):
    results_root = tmp_path / "results"
    manifest = results_root / "manifests" / "run-c.jsonl"

    result = _run_manifest_script(
        tmp_path,
        f"""
RUN_ID=run-c
RESULTS_ROOT="{results_root}"
scaffold_setup_run_manifest
scaffold_setup_run_manifest
""",
    )

    assert result.returncode == 0, result.stderr
    rows = _manifest_rows(manifest)
    markers = [row for row in rows if row.get("kind") == "manifest_attempt"]
    assert len(markers) == 2
    invocation_ids = {marker["stage_invocation_id"] for marker in markers}
    assert len(invocation_ids) == 2, "attempt markers must be distinct"


# --- Explicit status dimensions --------------------------------------------


def test_condition_status_rejects_unknown_dimension_values():
    with pytest.raises(ValueError):
        models.ConditionStatus("done", "real", "promote")
    with pytest.raises(ValueError):
        models.ConditionStatus("completed", "guess", "promote")
    with pytest.raises(ValueError):
        models.ConditionStatus("completed", "real", "maybe")


def test_real_zero_measurement_is_real_not_blocked():
    # A valid score of zero is measurement=real, per roadmap 7.1.
    status = models.ConditionStatus("completed", "real", "reject")
    assert status.is_real_measurement is True
    assert status.to_dict() == {
        "execution_outcome": "completed",
        "measurement": "real",
        "promotion": "reject",
    }


def test_legacy_execution_status_reexports_condition_status():
    assert execution_status.ConditionStatus is models.ConditionStatus


def test_blocker_translates_to_not_run_without_score():
    status = execution_status.translate_legacy_work_status("blocked")
    assert status.execution_outcome == "blocked"
    assert status.measurement == "not_run"
    assert status.promotion == "not_evaluated"
    assert status.is_real_measurement is False


def test_executed_translates_to_completed_real():
    status = execution_status.translate_legacy_work_status("executed")
    assert status.execution_outcome == "completed"
    assert status.measurement == "real"


def test_unknown_legacy_work_status_is_rejected():
    with pytest.raises(ValueError):
        execution_status.translate_legacy_work_status("mystery")


# --- Latest valid versus latest attempt indexing ---------------------------


def _write_report_input(path: Path, run_id: str, checks_passed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run": {"run_id": run_id, "task": "math_style", "lane": "reasoning"},
                "checks": {"ok": checks_passed},
            }
        )
    )


def test_latest_index_reports_valid_and_attempt_separately(tmp_path):
    manifests = tmp_path / "manifests"
    _write_report_input(
        manifests / "report_input_20260812T010000Z-valid.json",
        "20260812T010000Z-valid",
        checks_passed=True,
    )
    _write_report_input(
        manifests / "report_input_20260812T020000Z-failed.json",
        "20260812T020000Z-failed",
        checks_passed=False,
    )

    index = report_artifacts.build_latest_report_index(
        manifests_dir=manifests, task="math_style"
    )

    # A later failed attempt must not hide the last valid evidence.
    assert index["latest_attempt"]["run_id"] == "20260812T020000Z-failed"
    assert index["latest_valid"]["run_id"] == "20260812T010000Z-valid"
    assert index["num_candidates"] == 2
    assert index["num_valid"] == 1


def test_latest_index_valid_is_none_when_all_attempts_fail(tmp_path):
    manifests = tmp_path / "manifests"
    _write_report_input(
        manifests / "report_input_20260812T010000Z-a.json",
        "20260812T010000Z-a",
        checks_passed=False,
    )

    index = report_artifacts.build_latest_report_index(
        manifests_dir=manifests, task="math_style"
    )

    assert index["latest_attempt"]["run_id"] == "20260812T010000Z-a"
    assert index["latest_valid"] is None
    assert index["num_valid"] == 0


# --- Report identity validation --------------------------------------------


def test_reused_run_id_with_different_digest_is_rejected():
    first = {"run": {"run_id": "run-x"}, "declaration": {"model": "qwen3-1.7b"}}
    conflicting = {"run": {"run_id": "run-x"}, "declaration": {"model": "qwen3-8b"}}
    seen = {}
    # First observation records the digest.
    report_artifacts.validate_report_identity(first, seen=seen)
    # Same run_id, different declaration digest is an error (roadmap 3.1).
    with pytest.raises(ValueError, match="run-x"):
        report_artifacts.validate_report_identity(conflicting, seen=seen)


def test_reused_run_id_with_same_digest_is_accepted():
    payload = {"run": {"run_id": "run-y"}, "declaration": {"model": "qwen3-1.7b"}}
    seen = {}
    digest_a = report_artifacts.validate_report_identity(payload, seen=seen)
    # A regenerated derived report with the identical declaration is fine.
    digest_b = report_artifacts.validate_report_identity(dict(payload), seen=seen)
    assert digest_a == digest_b


# --- pass@k over rollout budget --------------------------------------------


def test_pass_at_k_over_budget_is_rejected():
    # num_rollouts=8 means pass@16 and pass@32 exceed the actual sample budget.
    summary = {
        "num_problems": 10,
        "pass_at_k": {"1": 0.4, "8": 0.7, "16": 0.9, "32": 1.0},
    }
    with pytest.raises(ValueError, match="16"):
        report_artifacts.validate_pass_at_k_budget(summary, num_rollouts=8)


def test_pass_at_k_within_budget_is_accepted():
    summary = {"num_problems": 10, "pass_at_k": {"1": 0.4, "8": 0.7}}
    # Does not raise; returns the validated ks.
    ks = report_artifacts.validate_pass_at_k_budget(summary, num_rollouts=8)
    assert ks == [1, 8]


# --- Rootfs destination safety ---------------------------------------------


def _run_rootfs_target(body: str, cwd: Path) -> subprocess.CompletedProcess:
    script = f"""
set -euo pipefail
source "{ROOTFS_TARGET}"
{body}
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_resolver_accepts_canonical_legacy_target(tmp_path):
    canonical = REPO_ROOT / "scripts" / "rootfs" / "rootfs"
    result = _run_rootfs_target(
        f'rootfs_resolve_managed_dest "{canonical}"',
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(canonical)


def test_resolver_refuses_arbitrary_dest(tmp_path):
    parent = tmp_path / "some" / "unrelated"
    parent.mkdir(parents=True)
    hostile = parent / "dir"
    result = _run_rootfs_target(
        f'rootfs_resolve_managed_dest "{hostile}"',
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "allowlist" in (result.stderr + result.stdout).lower()


def test_resolver_refuses_empty_and_root(tmp_path):
    empty = _run_rootfs_target('rootfs_resolve_managed_dest ""', cwd=tmp_path)
    assert empty.returncode != 0
    root = _run_rootfs_target('rootfs_resolve_managed_dest "/"', cwd=tmp_path)
    assert root.returncode != 0


def test_resolver_refuses_symlinked_parent(tmp_path):
    # A symlinked parent component must not be traversed into a delete target.
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    link_parent = tmp_path / "link"
    link_parent.symlink_to(real_parent)
    target = link_parent / "rootfs"
    result = _run_rootfs_target(
        f'rootfs_resolve_managed_dest "{target}"',
        cwd=tmp_path,
    )
    assert result.returncode != 0


def test_assert_removable_refuses_dir_without_ownership_marker(tmp_path):
    victim = tmp_path / "rootfs"
    victim.mkdir()
    (victim / "important.txt").write_text("do not delete")
    result = _run_rootfs_target(
        f'rootfs_assert_removable "{victim}"',
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "ownership" in (result.stderr + result.stdout).lower()


def test_assert_removable_accepts_owned_dir(tmp_path):
    owned = tmp_path / "rootfs"
    owned.mkdir()
    result = _run_rootfs_target(
        f"""
rootfs_write_ownership_marker "{owned}"
rootfs_assert_removable "{owned}"
""",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_assert_removable_refuses_symlink_target(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    rootfs_write = _run_rootfs_target(
        f'rootfs_write_ownership_marker "{real}"', cwd=tmp_path
    )
    assert rootfs_write.returncode == 0, rootfs_write.stderr
    link = tmp_path / "link"
    link.symlink_to(real)
    result = _run_rootfs_target(
        f'rootfs_assert_removable "{link}"',
        cwd=tmp_path,
    )
    assert result.returncode != 0


def test_enter_rootfs_refuses_implicit_build_of_missing_custom_rootfs(tmp_path):
    if not (Path("/usr/bin/bwrap").exists() or _which("bwrap")):
        pytest.skip("bwrap not available on host")
    missing = tmp_path / "no_such_rootfs"
    result = subprocess.run(
        [str(ENTER_ROOTFS), "--rootfs", str(missing), "--", "true"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    combined = (result.stderr + result.stdout).lower()
    assert "refusing implicit build" in combined


def _which(name: str) -> bool:
    from shutil import which

    return which(name) is not None
