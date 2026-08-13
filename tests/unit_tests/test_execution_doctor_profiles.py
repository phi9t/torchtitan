# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wave F2 doctor clause and profile model tests (roadmap Sections 9-10).

A scientific lane is not an execution profile. Doctor clauses use pass/fail/skip
and a profile is ready only when all its required clauses pass. The decisive F2
property is that an optional package's absence affects only the profiles that
declare it. These tests use fake probes so no rootfs or GPU is required.
"""

from __future__ import annotations

import pytest

from torchtitan.experiments.execution.preflight import doctor


def _passing(name="p", *, group="packages"):
    return doctor.Clause(
        name=name,
        group=group,
        requirement="fake requirement",
        probe=lambda: doctor.ClauseResult(status="pass", details={}),
    )


def _failing(name="f", *, group="packages", blocker_code="fake_blocked"):
    return doctor.Clause(
        name=name,
        group=group,
        requirement="fake requirement",
        blocker_code=blocker_code,
        probe=lambda: doctor.ClauseResult(status="fail", details={"why": "absent"}),
    )


def _skipping(name="s", *, group="packages"):
    return doctor.Clause(
        name=name,
        group=group,
        requirement="fake requirement",
        probe=lambda: doctor.ClauseResult(status="skip", details={}),
    )


def test_clause_status_must_be_pass_fail_or_skip():
    with pytest.raises(ValueError):
        doctor.ClauseResult(status="selected", details={})


def test_profile_ready_when_all_required_clauses_pass():
    profile = doctor.Profile(name="rootfs_cpu", clauses=[_passing("a"), _passing("b")])
    report = profile.evaluate()
    assert report.readiness == "ready"
    assert report.execution_outcome == "completed"
    assert report.blocker_codes == []


def test_profile_blocked_when_a_required_clause_fails():
    profile = doctor.Profile(
        name="vllm_1gpu",
        clauses=[_passing("a"), _failing("vllm_import", blocker_code="vllm_absent")],
    )
    report = profile.evaluate()
    assert report.readiness == "blocked"
    # A failed required clause yields execution_outcome=blocked and a stable
    # blocker code, and no score is fabricated (roadmap Section 10).
    assert report.execution_outcome == "blocked"
    assert report.blocker_codes == ["vllm_absent"]
    assert "score" not in report.to_dict()


def test_skip_clause_does_not_block_a_profile():
    # An optional clause that skips must not make the profile blocked.
    profile = doctor.Profile(
        name="rootfs_cpu", clauses=[_passing("a"), _skipping("opt")]
    )
    report = profile.evaluate()
    assert report.readiness == "ready"


def test_optional_package_absence_affects_only_declaring_profile():
    # vllm is declared only by vllm_1gpu; its absence must block vllm_1gpu but
    # leave rootfs_cpu (which does not declare vllm) ready. This is the F2
    # done criterion: optional package absence is profile-local.
    vllm_absent = _failing("vllm_import", blocker_code="vllm_absent")
    vllm_1gpu = doctor.Profile(
        name="vllm_1gpu", clauses=[_passing("torch"), vllm_absent]
    )
    rootfs_cpu = doctor.Profile(name="rootfs_cpu", clauses=[_passing("torch")])

    composed = doctor.compose([rootfs_cpu, vllm_1gpu]).evaluate()
    per_profile = {r.name: r.readiness for r in composed.profiles}
    assert per_profile["rootfs_cpu"] == "ready"
    assert per_profile["vllm_1gpu"] == "blocked"
    # The composite is blocked because at least one profile is blocked, but the
    # blocker is attributed to vllm_1gpu only.
    assert composed.readiness == "blocked"
    assert composed.blocker_codes == ["vllm_absent"]


def test_doctor_artifact_has_no_journal_side_effect(tmp_path):
    # The doctor writes exactly one artifact and never initializes the attempt
    # journal (roadmap Section 10).
    profile = doctor.Profile(name="host_static", clauses=[_passing("a")])
    out = tmp_path / "doctor.json"
    doctor.write_doctor_artifact(out, profile.evaluate())
    assert out.is_file()
    # No processes/ event stream is created as a side effect.
    assert not (tmp_path / "processes").exists()
