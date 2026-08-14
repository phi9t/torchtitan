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
from torchtitan.experiments.execution.preflight import doctor


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
    if preflight.get("schema_version") != 1:
        raise ValueError("preflight.schema_version must be 1")
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
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("preflight.profiles must be a non-empty list")
    for index, profile in enumerate(profiles):
        _validate_profile_report(profile, index)
    semantic_checks = preflight.get("semantic_checks")
    if not isinstance(semantic_checks, list):
        raise ValueError("preflight.semantic_checks must be a list")
    for index, semantic_check in enumerate(semantic_checks):
        _validate_semantic_check(semantic_check, index)


def _validate_profile_report(value: object, index: int) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"preflight.profiles[{index}] must be an object")
    if value.get("kind") != "profile_report":
        raise ValueError(f"preflight.profiles[{index}].kind must be 'profile_report'")
    if not isinstance(value.get("name"), str) or not value.get("name"):
        raise ValueError(f"preflight.profiles[{index}].name must be non-empty")
    readiness = value.get("readiness")
    if readiness not in _READINESS:
        raise ValueError(f"preflight.profiles[{index}].readiness is invalid")
    execution_outcome = value.get("execution_outcome")
    if execution_outcome != _READY_OUTCOMES[readiness]:
        raise ValueError(
            f"preflight.profiles[{index}].execution_outcome is inconsistent"
        )
    blocker_codes = value.get("blocker_codes")
    if not isinstance(blocker_codes, list) or not all(
        isinstance(code, str) and code for code in blocker_codes
    ):
        raise ValueError(
            f"preflight.profiles[{index}].blocker_codes must be a list of strings"
        )
    clauses = value.get("clauses")
    if not isinstance(clauses, list):
        raise ValueError(f"preflight.profiles[{index}].clauses must be a list")
    for clause_index, clause in enumerate(clauses):
        _validate_clause(clause, index, clause_index)


def _validate_clause(value: object, profile_index: int, clause_index: int) -> None:
    prefix = f"preflight.profiles[{profile_index}].clauses[{clause_index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{prefix} must be an object")
    for field in ("name", "group", "requirement", "status"):
        if not isinstance(value.get(field), str) or not value.get(field):
            raise ValueError(f"{prefix}.{field} must be a non-empty string")
    if value["group"] not in doctor.CLAUSE_GROUPS:
        raise ValueError(f"{prefix}.group is invalid")
    if value["status"] not in doctor.CLAUSE_STATUSES:
        raise ValueError(f"{prefix}.status is invalid")
    if not isinstance(value.get("details"), dict):
        raise ValueError(f"{prefix}.details must be an object")
    blocker_code = value.get("blocker_code")
    if value["status"] == "fail":
        if not isinstance(blocker_code, str) or not blocker_code:
            raise ValueError(f"{prefix}.blocker_code must be set on failed clauses")
    elif blocker_code is not None:
        raise ValueError(f"{prefix}.blocker_code is only valid for failed clauses")


def _validate_semantic_check(value: object, index: int) -> None:
    prefix = f"preflight.semantic_checks[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{prefix} must be an object")
    for field in ("name", "status"):
        if not isinstance(value.get(field), str) or not value.get(field):
            raise ValueError(f"{prefix}.{field} must be a non-empty string")
    if value["status"] not in ("pass", "fail"):
        raise ValueError(f"{prefix}.status must be 'pass' or 'fail'")
    blocker_code = value.get("blocker_code")
    if value["status"] == "fail":
        if not isinstance(blocker_code, str) or not blocker_code:
            raise ValueError(f"{prefix}.blocker_code must be set on failures")
    elif blocker_code is not None:
        raise ValueError(f"{prefix}.blocker_code must be null on passes")
    if not isinstance(value.get("details"), dict):
        raise ValueError(f"{prefix}.details must be an object")
