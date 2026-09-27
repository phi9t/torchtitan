#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pre-commit guard that keeps prose in the formal-verifier notes at 80 columns.

Why this exists: the qwen3-formal-verifier tickets are hand-reflowed prose, and
reflowing them by hand repeatedly produced orphan lines and inconsistent widths.
A checker is cheaper and more reliable than the habit.

Why it is scoped rather than repository-wide: 99 of 176 tracked markdown files
carry over-width lines, spread across several unrelated programs. Widening the
scope would demand a mechanical reflow of other people's documents, so the hook
covers only the corpus whose width this program actually maintains. Extending it
is a deliberate act -- add paths to the hook's ``files`` pattern and reflow that
corpus in the same change.

Runs on plain stdlib so the pre-commit "system" language can invoke it host-side.

What is exempt, and why each exemption is narrow enough to still catch prose:

  - fenced code blocks (``` or ~~~), where width is the content's business;
  - indented code blocks of four spaces or more;
  - table rows (a line whose first non-space character is ``|``), which cannot
    be wrapped without changing the table;
  - any line holding a single unbreakable token longer than the limit -- a URL,
    a long path, a hash -- since no reflow can fix it;
  - a line containing the literal ``width-allow``, for the rare deliberate case.

A line that is merely long, with spaces available to wrap at, is a finding.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

LIMIT = 80
ALLOW_PRAGMA = "width-allow"
FENCE_PREFIXES = ("```", "~~~")


@dataclass(frozen=True)
class Finding:
    path: Path
    line_number: int
    width: int
    text: str

    def render(self) -> str:
        return (
            f"{self.path}:{self.line_number}: {self.width} columns "
            f"(limit {LIMIT}): {self.text.rstrip()}"
        )


def _has_unbreakable_token(text: str) -> bool:
    """True when some single token already exceeds the limit.

    Such a line cannot be reflowed, so flagging it would be an instruction the
    author cannot follow.
    """
    return any(len(token) > LIMIT for token in text.split())


def scan_text(text: str, path: Path) -> list[Finding]:
    findings: list[Finding] = []
    in_fence = False
    fence_marker = ""
    for line_number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.lstrip()
        if in_fence:
            if stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            continue
        for prefix in FENCE_PREFIXES:
            if stripped.startswith(prefix):
                in_fence = True
                fence_marker = prefix
                break
        if in_fence:
            continue
        width = len(raw.rstrip("\n"))
        if width <= LIMIT:
            continue
        if ALLOW_PRAGMA in raw:
            continue
        if raw.startswith("    ") or raw.startswith("\t"):
            continue
        if stripped.startswith("|"):
            continue
        if _has_unbreakable_token(raw):
            continue
        findings.append(Finding(path, line_number, width, raw))
    return findings


def scan_path(path: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return scan_text(text, path)


def main(argv: Iterable[str]) -> int:
    findings: list[Finding] = []
    for candidate in argv:
        path = Path(candidate)
        if path.suffix != ".md" or not path.is_file():
            continue
        findings.extend(scan_path(path))

    if not findings:
        return 0

    print(f"check-line-width: {len(findings)} line(s) exceed {LIMIT} columns")
    for finding in findings:
        print(f"  {finding.render()}")
    print(
        "  Reflow the prose, or add the literal 'width-allow' to a line that "
        "must stay long."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
