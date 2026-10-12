# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Composable doctor clauses and execution profiles (roadmap Sections 9-10).

A scientific lane (reasoning/coding/agentic) is not an execution profile. A
profile is a composable bundle of doctor clauses that proves execution
prerequisites, not benchmark semantics. Each clause reports pass, fail, or skip;
a profile is ``ready`` only when all of its required clauses pass. A skipped
clause never blocks a profile, which is why an optional package's absence
affects only the profiles that declare a clause for it.

A failed required clause yields ``execution_outcome=blocked``, a stable blocker
code, and no fabricated score. The doctor writes exactly one artifact and never
initializes the attempt journal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from torchtitan.experiments.execution import store


# Clause status vocabulary (roadmap Section 9): pass, fail, or skip.
CLAUSE_STATUSES = ("pass", "fail", "skip")

# Generic doctor clause groups (roadmap Section 10). The doctor proves these
# execution prerequisites; benchmark semantics belong to task preflight.
CLAUSE_GROUPS = (
    "attempt",
    "rootfs",
    "source",
    "packages",
    "assets",
    "gpu",
    "distributed",
    "cache",
    "external",
    "output",
)

# Profile readiness is derived, not a score.
PROFILE_READINESS = ("ready", "blocked")


@dataclass(frozen=True)
class ClauseResult:
    """The outcome of probing one clause."""

    status: str
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in CLAUSE_STATUSES:
            raise ValueError(
                f"ClauseResult.status must be one of {CLAUSE_STATUSES}, "
                f"got {self.status!r}"
            )


@dataclass(frozen=True)
class Clause:
    """One doctor prerequisite check.

    ``probe`` is injected so a clause is host-testable without the real
    dependency. ``blocker_code`` is the stable code attributed to the profile
    when a required clause fails. A clause whose probe returns ``skip`` never
    blocks its profile, so declaring a clause optional is done by having its
    probe skip when the dependency is absent.
    """

    name: str
    group: str
    requirement: str
    probe: Callable[[], ClauseResult]
    blocker_code: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Clause.name must be non-empty")
        if self.group not in CLAUSE_GROUPS:
            raise ValueError(
                f"Clause.group must be one of {CLAUSE_GROUPS}, got {self.group!r}"
            )

    def evaluate(self) -> dict[str, object]:
        result = self.probe()
        row: dict[str, object] = {
            "name": self.name,
            "group": self.group,
            "requirement": self.requirement,
            "status": result.status,
            "details": result.details,
        }
        if result.status == "fail":
            row["blocker_code"] = self.blocker_code or f"{self.name}_failed"
        return row


@dataclass(frozen=True)
class ProfileReport:
    """A derived readiness view for one profile.

    ``readiness`` is ``ready`` only when every required clause passed;
    otherwise ``blocked`` with the failing clauses' stable blocker codes.
    There is deliberately no score field: a blocked profile fabricates nothing.
    """

    name: str
    readiness: str
    clauses: list[dict[str, object]]

    @property
    def execution_outcome(self) -> str:
        return "completed" if self.readiness == "ready" else "blocked"

    @property
    def blocker_codes(self) -> list[str]:
        return [
            str(clause["blocker_code"])
            for clause in self.clauses
            if clause["status"] == "fail"
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "profile_report",
            "name": self.name,
            "readiness": self.readiness,
            "execution_outcome": self.execution_outcome,
            "blocker_codes": self.blocker_codes,
            "clauses": self.clauses,
        }


@dataclass(frozen=True)
class Profile:
    """A composable execution profile: a named bundle of doctor clauses."""

    name: str
    clauses: list[Clause]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Profile.name must be non-empty")

    def evaluate(self) -> ProfileReport:
        evaluated = [clause.evaluate() for clause in self.clauses]
        # A skip never blocks; only a fail does (roadmap Section 9).
        blocked = any(clause["status"] == "fail" for clause in evaluated)
        return ProfileReport(
            name=self.name,
            readiness="blocked" if blocked else "ready",
            clauses=evaluated,
        )


@dataclass(frozen=True)
class ComposedReport:
    """The readiness of a set of composed profiles.

    The composite is ready only when every profile is ready. Blocker codes stay
    attributed to the profile that produced them, so an optional package's
    absence is never charged to a profile that did not declare it.
    """

    profiles: list[ProfileReport]

    @property
    def readiness(self) -> str:
        return (
            "ready"
            if all(report.readiness == "ready" for report in self.profiles)
            else "blocked"
        )

    @property
    def execution_outcome(self) -> str:
        return "completed" if self.readiness == "ready" else "blocked"

    @property
    def blocker_codes(self) -> list[str]:
        codes: list[str] = []
        for report in self.profiles:
            codes.extend(report.blocker_codes)
        return codes

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "composed_doctor_report",
            "readiness": self.readiness,
            "execution_outcome": self.execution_outcome,
            "blocker_codes": self.blocker_codes,
            "profiles": [report.to_dict() for report in self.profiles],
        }


class _ComposedProfiles:
    """A set of profiles evaluated together (see compose)."""

    def __init__(self, profiles: list[Profile]):
        self._profiles = profiles

    def evaluate(self) -> ComposedReport:
        return ComposedReport(profiles=[p.evaluate() for p in self._profiles])


def compose(profiles: list[Profile]) -> _ComposedProfiles:
    """Compose the profiles a run's stages need into one evaluable set."""

    return _ComposedProfiles(list(profiles))


def write_doctor_artifact(path: Path, report: ProfileReport | ComposedReport) -> None:
    """Write the single immutable doctor artifact.

    The doctor never initializes the attempt journal; it only writes this one
    artifact (roadmap Section 10).
    """

    payload = {"schema_version": 1, **report.to_dict()}
    store.write_terminal_json(Path(path), payload)
