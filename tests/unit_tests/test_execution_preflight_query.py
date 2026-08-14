# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wave F2 semantic-preflight query and report-attach tests (roadmap 7, 11).

run_preflight is a query over composed profiles plus task-owned semantic
checks: it reports ready/blocked and stable blocker codes and never
initializes the attempt journal. The report adapter attaches the doctor and
preflight readiness to the canonical report input's checks section and adds an
execution section, without owning the scientific schema. Host-testable.
"""

from __future__ import annotations

import pytest

from torchtitan.experiments.execution.preflight import profiles, query, report_attach


def _ready_env():
    return profiles.ProbeEnv(
        in_rootfs=True,
        available_packages={"torch", "datasets", "transformers", "vllm"},
        gpu_count=1,
    )


def _blocked_env():
    return profiles.ProbeEnv(in_rootfs=False, available_packages=set(), gpu_count=0)


def test_query_reports_ready_over_composed_profiles():
    result = query.run_preflight(
        profile_names=["host_static", "rootfs_cpu"], env=_ready_env()
    )
    assert result["readiness"] == "ready"
    assert result["execution_outcome"] == "completed"
    assert result["blocker_codes"] == []


def test_query_reports_blocked_with_stable_blocker_codes():
    result = query.run_preflight(
        profile_names=["rootfs_cpu", "vllm_1gpu"], env=_blocked_env()
    )
    assert result["readiness"] == "blocked"
    assert "rootfs_not_active" in result["blocker_codes"]


def test_query_with_semantic_checks_blocks_on_failing_check():
    # A task-owned semantic check that fails must block the query even when the
    # runtime profiles are ready (roadmap Section 11: readiness is necessary
    # but not sufficient).
    semantic = [
        query.SemanticCheck(
            name="parser_fixture",
            passed=False,
            blocker_code="parser_fixture_failed",
            details={},
        )
    ]
    result = query.run_preflight(
        profile_names=["host_static"], env=_ready_env(), semantic_checks=semantic
    )
    assert result["readiness"] == "blocked"
    assert "parser_fixture_failed" in result["blocker_codes"]


def test_report_attach_adds_checks_and_execution_section():
    preflight = query.run_preflight(
        profile_names=["host_static", "rootfs_cpu"], env=_ready_env()
    )
    extra_checks, execution_section = report_attach.to_report_sections(preflight)
    assert extra_checks["preflight_ready"] is True
    assert execution_section["kind"] == "execution_preflight"
    assert execution_section["readiness"] == "ready"
    # The scientific schema owner is untouched; this only supplies extras.
    assert set(extra_checks) == {"preflight_ready"}


def test_report_attach_marks_blocked_preflight_as_failing_check():
    preflight = query.run_preflight(profile_names=["rootfs_cpu"], env=_blocked_env())
    extra_checks, execution_section = report_attach.to_report_sections(preflight)
    assert extra_checks["preflight_ready"] is False
    assert execution_section["blocker_codes"]


def test_report_attach_rejects_invalid_ready_outcome():
    preflight = query.run_preflight(profile_names=["host_static"], env=_ready_env())
    preflight["execution_outcome"] = "pass"

    with pytest.raises(ValueError, match="execution_outcome"):
        report_attach.to_report_sections(preflight)


def test_report_attach_rejects_blocked_without_blocker_codes():
    preflight = query.run_preflight(profile_names=["rootfs_cpu"], env=_blocked_env())
    preflight["blocker_codes"] = []

    with pytest.raises(ValueError, match="blocked preflight"):
        report_attach.to_report_sections(preflight)


def test_report_attach_rejects_unsupported_schema_version():
    preflight = query.run_preflight(profile_names=["host_static"], env=_ready_env())
    preflight["schema_version"] = 2

    with pytest.raises(ValueError, match="schema_version"):
        report_attach.to_report_sections(preflight)


def test_report_attach_rejects_malformed_profile_report():
    preflight = query.run_preflight(profile_names=["host_static"], env=_ready_env())
    preflight["profiles"] = ["host_static"]

    with pytest.raises(ValueError, match=r"profiles\[0\]"):
        report_attach.to_report_sections(preflight)


def test_report_attach_rejects_malformed_semantic_check():
    preflight = query.run_preflight(
        profile_names=["host_static"],
        env=_ready_env(),
        semantic_checks=[
            query.SemanticCheck(
                name="fixture",
                passed=False,
                blocker_code="fixture_failed",
                details={},
            )
        ],
    )
    preflight["semantic_checks"][0]["details"] = "bad"

    with pytest.raises(ValueError, match=r"semantic_checks\[0\]"):
        report_attach.to_report_sections(preflight)

