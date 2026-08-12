# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Explicit status dimensions for scaffold-to-policy evaluation conditions.

The runtime preflight roadmap (Section 7.1) defines three independent status
dimensions that must not be collapsed into a single pass/fail number:

- execution_outcome: did the operational execution complete, block, fail, or
  get interrupted;
- measurement: is the recorded number a real measurement, a fixture, a smoke
  value, an invalid artifact, or was nothing measured;
- promotion: is the condition promoted, held, rejected, or not yet evaluated.

A valid score of zero is ``measurement=real``. A runtime, access, verifier, or
integrity blocker has no fabricated score, so it is ``measurement=invalid`` or
``not_run`` with ``promotion=not_evaluated``. Process exit zero does not imply
task success or promotion.
"""

from __future__ import annotations

from dataclasses import dataclass


EXECUTION_OUTCOMES = ("completed", "blocked", "failed", "interrupted")
MEASUREMENTS = ("real", "fixture", "smoke", "invalid", "not_run")
PROMOTIONS = ("promote", "hold", "reject", "not_evaluated")


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


def translate_legacy_work_status(work_status: str) -> ConditionStatus:
    """Map a legacy ``work_status`` string onto explicit status dimensions.

    Older runners recorded a single ``work_status`` per stage row (for example
    ``executed``, ``blocked``, ``skipped``). Translating those to explicit
    dimensions lets the report layer keep ingesting historical manifests
    without inventing a score for a blocker.
    """

    normalized = work_status.strip().lower()
    if normalized in ("executed", "completed", "done", "ok"):
        return ConditionStatus("completed", "real", "not_evaluated")
    if normalized in ("blocked", "gated"):
        return ConditionStatus("blocked", "not_run", "not_evaluated")
    if normalized in ("skipped", "not_run", "pending"):
        return ConditionStatus("blocked", "not_run", "not_evaluated")
    if normalized in ("failed", "error"):
        return ConditionStatus("failed", "invalid", "not_evaluated")
    if normalized in ("interrupted", "cancelled", "canceled"):
        return ConditionStatus("interrupted", "not_run", "not_evaluated")
    if normalized in ("fixture", "fake"):
        return ConditionStatus("completed", "fixture", "not_evaluated")
    if normalized in ("smoke", "debug_smoke"):
        return ConditionStatus("completed", "smoke", "not_evaluated")
    raise ValueError(f"unknown legacy work_status: {work_status!r}")


def _require_member(field: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{field} must be one of {allowed}, got {value!r}")
