# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preflight query over composed profiles and semantic checks (roadmap 9-11).

run_preflight is a query, not a manifest initializer. It composes the doctor
profiles a run's stages need, evaluates them, folds in task-owned semantic
checks (roadmap Section 11), and reports one readiness verdict with stable
blocker codes. It never writes the attempt journal; a caller may persist the
returned dict as a doctor/preflight artifact.

Runtime readiness is necessary but not sufficient: a failing semantic check
blocks the query even when every runtime profile is ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan.experiments.execution.preflight import doctor, profiles


@dataclass(frozen=True)
class SemanticCheck:
    """One task-owned benchmark-semantic check (roadmap Section 11).

    The task owns the check logic; the query only records its verdict so a
    failing check contributes a stable blocker code instead of a score.
    """

    name: str
    passed: bool
    blocker_code: str
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SemanticCheck.name must be non-empty")
        if not self.blocker_code:
            raise ValueError("SemanticCheck.blocker_code must be non-empty")


def run_preflight(
    *,
    profile_names: list[str],
    env: profiles.ProbeEnv,
    semantic_checks: list[SemanticCheck] | None = None,
) -> dict[str, object]:
    """Query readiness over composed profiles plus semantic checks."""

    composed = doctor.compose(
        [profiles.build(name, env=env) for name in profile_names]
    ).evaluate()

    semantic = list(semantic_checks or [])
    semantic_blockers = [check.blocker_code for check in semantic if not check.passed]
    blocker_codes = list(composed.blocker_codes) + semantic_blockers
    readiness = "ready" if not blocker_codes else "blocked"
    return {
        "schema_version": 1,
        "kind": "execution_preflight",
        "readiness": readiness,
        "execution_outcome": "completed" if readiness == "ready" else "blocked",
        "blocker_codes": blocker_codes,
        "profiles": composed.to_dict()["profiles"],
        "semantic_checks": [
            {
                "name": check.name,
                "status": "pass" if check.passed else "fail",
                "blocker_code": check.blocker_code if not check.passed else None,
                "details": check.details,
            }
            for check in semantic
        ],
    }
