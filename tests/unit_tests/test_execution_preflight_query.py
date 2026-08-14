# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wave F2 semantic-preflight query tests (roadmap Sections 9-11).

run_preflight is a query over composed profiles plus task-owned semantic
checks: it reports ready/blocked and stable blocker codes and never
initializes the attempt journal. Host-testable.
"""

from __future__ import annotations

from torchtitan.experiments.execution.preflight import profiles, query


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

