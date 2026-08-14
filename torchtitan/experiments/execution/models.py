# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Typed lifecycle models (runtime preflight roadmap Sections 3-7).

A RunDeclaration is one immutable scientific declaration identified by a
run_id; reusing a run_id with a different normalized declaration is an error
(Section 3.1). An Attempt is one top-level operational execution of that frozen
declaration and links its parent when it is a resume (Section 3.2). A StageSpec
declares one stage; a StageEvent records its start and single terminal outcome
(Section 5). An ArtifactRef records lineage with independent work_status and
freshness (Section 6). An AttemptOutcome is the independent terminal summary
derived from stage events (Section 7).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


__all__ = [
    "ConditionStatus",
    "RunDeclaration",
    "Attempt",
    "StageSpec",
    "StageEvent",
    "ArtifactRef",
    "AttemptOutcome",
    "STAGE_KINDS",
    "EXECUTION_ADAPTERS",
    "STAGE_EVENT_KINDS",
    "STAGE_TERMINAL_EVENTS",
    "ARTIFACT_WORK_STATUSES",
    "ARTIFACT_FRESHNESS",
    "EXECUTION_OUTCOMES",
    "MEASUREMENTS",
    "PROMOTIONS",
    "ATTEMPT_EXECUTION_OUTCOMES",
]


# Stage kinds and execution adapters (roadmap Section 5).
STAGE_KINDS = (
    "doctor",
    "preflight",
    "acquire",
    "prepare",
    "generate",
    "verify",
    "train",
    "checkpoint",
    "export",
    "evaluate",
    "ingest",
    "report",
)
EXECUTION_ADAPTERS = (
    "host_test",
    "rootfs_cpu",
    "rootfs_vllm",
    "rootfs_torchrun_sft",
    "rootfs_monarch_rl",
    "coding_sandbox",
    "external_harness",
)

# Stage event kinds. One non-terminal start plus one terminal outcome.
STAGE_EVENT_STARTED = "stage_started"
STAGE_TERMINAL_EVENTS = (
    "stage_succeeded",
    "stage_blocked",
    "stage_failed",
    "stage_interrupted",
)
STAGE_EVENT_KINDS = (STAGE_EVENT_STARTED,) + STAGE_TERMINAL_EVENTS

# Artifact lineage vocabularies (roadmap Section 6). 'fresh' is not a
# work_status value; freshness describes what provenance proves about the bytes.
ARTIFACT_WORK_STATUSES = ("produced", "reused", "resumed", "imported", "external")
ARTIFACT_FRESHNESS = ("verified_new", "verified_preexisting", "unknown")

# Status dimensions for evaluation conditions (roadmap Section 7.1).
EXECUTION_OUTCOMES = ("completed", "blocked", "failed", "interrupted")
MEASUREMENTS = ("real", "fixture", "smoke", "invalid", "not_run")
PROMOTIONS = ("promote", "hold", "reject", "not_evaluated")

# An attempt outcome is a terminal execution summary, not a score. It shares
# the execution-outcome vocabulary used by per-condition status.
ATTEMPT_EXECUTION_OUTCOMES = EXECUTION_OUTCOMES


@dataclass(frozen=True)
class ConditionStatus:
    """The three independent status dimensions for one evaluation condition."""

    execution_outcome: str
    measurement: str
    promotion: str

    def __post_init__(self) -> None:
        _require_member("execution_outcome", self.execution_outcome, EXECUTION_OUTCOMES)
        _require_member("measurement", self.measurement, MEASUREMENTS)
        _require_member("promotion", self.promotion, PROMOTIONS)

    @property
    def is_real_measurement(self) -> bool:
        """A real measurement is a scientific number, including a valid zero."""

        return self.measurement == "real"

    def to_dict(self) -> dict[str, str]:
        return {
            "execution_outcome": self.execution_outcome,
            "measurement": self.measurement,
            "promotion": self.promotion,
        }


def _require_member(field: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{field} must be one of {allowed}, got {value!r}")


@dataclass(frozen=True)
class RunDeclaration:
    """One immutable scientific declaration identified by run_id.

    ``fields`` carries the remaining normalized declaration content (model,
    tokenizer, scaffold, sampling, budgets, and so on). The digest normalizes
    the whole declaration so that reusing a run_id with a different digest can
    be rejected.
    """

    run_id: str
    family: str
    task: str
    lane: str
    fields: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("run_id", "family", "task", "lane"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"RunDeclaration.{name} must be a non-empty string")

    def normalized(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "family": self.family,
            "task": self.task,
            "lane": self.lane,
            "fields": self.fields,
        }

    def digest(self) -> str:
        payload = json.dumps(self.normalized(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Attempt:
    """One top-level operational execution of a frozen run declaration.

    A resume after a terminal outcome is a new attempt that links its parent;
    it never reopens the old attempt (roadmap 3.2).
    """

    attempt_id: str
    declaration: RunDeclaration
    parent_attempt_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise ValueError("Attempt.attempt_id must be a non-empty string")
        if self.parent_attempt_id is not None and not self.parent_attempt_id:
            raise ValueError("Attempt.parent_attempt_id must be non-empty when set")


@dataclass(frozen=True)
class StageSpec:
    """One stage declaration within an attempt (roadmap Section 5)."""

    stage_id: str
    name: str
    kind: str
    adapter: str
    argv: list[str]
    cwd: str | None = None
    depends_on: tuple[str, ...] = ()
    timeout_seconds: int | None = None
    supports_resume: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.stage_id, str) or not self.stage_id:
            raise ValueError("StageSpec.stage_id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("StageSpec.name must be a non-empty string")
        if self.kind not in STAGE_KINDS:
            raise ValueError(
                f"StageSpec.kind must be one of {STAGE_KINDS}, got {self.kind!r}"
            )
        if self.adapter not in EXECUTION_ADAPTERS:
            raise ValueError(
                f"StageSpec.adapter must be one of {EXECUTION_ADAPTERS}, "
                f"got {self.adapter!r}"
            )
        if not self.argv:
            raise ValueError("StageSpec.argv must be non-empty")


@dataclass(frozen=True)
class StageEvent:
    """A stage start or single terminal outcome (roadmap Section 5)."""

    kind: str
    stage_id: str
    stage_invocation_id: str
    return_code: int | None = None
    failure_type: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in STAGE_EVENT_KINDS:
            raise ValueError(
                f"StageEvent.kind must be one of {STAGE_EVENT_KINDS}, "
                f"got {self.kind!r}"
            )
        if not self.stage_id:
            raise ValueError("StageEvent.stage_id must be non-empty")
        if not self.stage_invocation_id:
            raise ValueError("StageEvent.stage_invocation_id must be non-empty")

    @property
    def is_terminal(self) -> bool:
        return self.kind in STAGE_TERMINAL_EVENTS


@dataclass(frozen=True)
class ArtifactRef:
    """An artifact lineage reference (roadmap Section 6).

    ``work_status`` describes how the stage relates to the artifact;
    ``freshness`` describes what provenance proves about the bytes. A run ID in
    the path is not freshness proof.
    """

    artifact_id: str
    artifact_class: str
    path: str
    work_status: str
    freshness: str

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("ArtifactRef.artifact_id must be non-empty")
        if not self.artifact_class:
            raise ValueError("ArtifactRef.artifact_class must be non-empty")
        if not self.path:
            raise ValueError("ArtifactRef.path must be non-empty")
        if self.work_status not in ARTIFACT_WORK_STATUSES:
            raise ValueError(
                f"ArtifactRef.work_status must be one of {ARTIFACT_WORK_STATUSES}, "
                f"got {self.work_status!r}"
            )
        if self.freshness not in ARTIFACT_FRESHNESS:
            raise ValueError(
                f"ArtifactRef.freshness must be one of {ARTIFACT_FRESHNESS}, "
                f"got {self.freshness!r}"
            )


@dataclass(frozen=True)
class AttemptOutcome:
    """Independent terminal summary of an attempt (roadmap Section 7).

    Derived from all stage terminal events. It does not overwrite per-condition
    evaluation status and is not a score.
    """

    attempt_id: str
    execution_outcome: str
    stage_invocation_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.attempt_id:
            raise ValueError("AttemptOutcome.attempt_id must be non-empty")
        if self.execution_outcome not in ATTEMPT_EXECUTION_OUTCOMES:
            raise ValueError(
                f"AttemptOutcome.execution_outcome must be one of "
                f"{ATTEMPT_EXECUTION_OUTCOMES}, got {self.execution_outcome!r}"
            )
