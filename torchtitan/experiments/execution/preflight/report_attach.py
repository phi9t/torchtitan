# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attach preflight readiness to the canonical report input (roadmap Section 7).

Existing task report builders remain the scientific schema owner. This adapter
only supplies the lifecycle's additions: a boolean check for the flat ``checks``
section and a validated ``execution`` section carrying the full preflight
readiness. It never rewrites metrics, verifier, or scaffold sections.
"""

from __future__ import annotations

from torchtitan.experiments.execution import models


_READINESS = ("ready", "blocked")
_READY_OUTCOMES = {"ready": "completed", "blocked": "blocked"}


def to_report_sections(
    preflight: dict[str, object],
) -> tuple[dict[str, bool], dict[str, object]]:
    """Return (extra_checks, execution_section) for build_report_input.

    ``extra_checks`` folds a single boolean into the flat checks view so a
    blocked preflight shows as a failing check. ``execution_section`` is the
    validated execution evidence added under the report's ``execution`` key.
    """

    _validate_preflight(preflight)
    ready = preflight.get("readiness") == "ready"
    extra_checks = {"preflight_ready": ready}
    execution_section = {
        "kind": "execution_preflight",
        "readiness": preflight.get("readiness"),
        "execution_outcome": preflight.get("execution_outcome"),
        "blocker_codes": list(preflight.get("blocker_codes", [])),
        "profiles": preflight.get("profiles", []),
        "semantic_checks": preflight.get("semantic_checks", []),
    }
    return extra_checks, execution_section


def _validate_preflight(preflight: dict[str, object]) -> None:
    if preflight.get("kind") != "execution_preflight":
        raise ValueError("preflight.kind must be 'execution_preflight'")
    readiness = preflight.get("readiness")
    if readiness not in _READINESS:
        raise ValueError(f"preflight.readiness must be one of {_READINESS}")
    execution_outcome = preflight.get("execution_outcome")
    if execution_outcome not in models.EXECUTION_OUTCOMES:
        raise ValueError(
            "preflight.execution_outcome must be one of "
            f"{models.EXECUTION_OUTCOMES}"
        )
    expected_outcome = _READY_OUTCOMES[readiness]
    if execution_outcome != expected_outcome:
        raise ValueError(
            "preflight.execution_outcome must be "
            f"{expected_outcome!r} when readiness is {readiness!r}"
        )
    blocker_codes = preflight.get("blocker_codes")
    if not isinstance(blocker_codes, list) or not all(
        isinstance(code, str) and code for code in blocker_codes
    ):
        raise ValueError("preflight.blocker_codes must be a list of non-empty strings")
    if readiness == "ready" and blocker_codes:
        raise ValueError("ready preflight must not have blocker_codes")
    if readiness == "blocked" and not blocker_codes:
        raise ValueError("blocked preflight must include blocker_codes")
    profiles = preflight.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("preflight.profiles must be a list")
    semantic_checks = preflight.get("semantic_checks")
    if not isinstance(semantic_checks, list):
        raise ValueError("preflight.semantic_checks must be a list")
