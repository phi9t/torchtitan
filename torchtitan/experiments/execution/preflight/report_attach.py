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


def to_report_sections(
    preflight: dict[str, object],
) -> tuple[dict[str, bool], dict[str, object]]:
    """Return (extra_checks, execution_section) for build_report_input.

    ``extra_checks`` folds a single boolean into the flat checks view so a
    blocked preflight shows as a failing check. ``execution_section`` is the
    validated execution evidence added under the report's ``execution`` key.
    """

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
