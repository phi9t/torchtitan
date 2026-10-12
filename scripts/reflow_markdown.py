#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rewrap markdown prose to the width check_line_width.py enforces.

The companion formatter to scripts/check_line_width.py: that script says which
lines are too wide, this one fixes them. Kept out of pre-commit on purpose --
a formatter that rewrites prose should be run deliberately, not on every commit.

Why it exists: hand-reflowing ticket prose to 80 columns repeatedly produced
orphan lines, because fixing one long line by hand leaves the rest of its
paragraph unbalanced. Rewrapping the whole containing paragraph is the fix.

The safety property, asserted on every file before anything is written: the
whitespace-separated token sequence must be identical before and after. A file
whose tokens would change is refused and left alone rather than written, so a
bug here cannot silently alter prose. Paragraphs already within the limit are
left byte-identical, so the diff stays confined to what was actually too wide.

Structures left verbatim: fenced blocks, four-space indented blocks, table rows,
headings, and any line already inside the limit.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LIMIT = 80
FENCE = re.compile(r"^\s*(```|~~~)")
LIST_MARKER = re.compile(r"^(\s*)([-*+]|\d+[.)])(\s+)")
HEADING = re.compile(r"^\s*#")
TABLE_ROW = re.compile(r"^\s*\|")
CODE_INDENT = re.compile(r"^(    |\t)")


def _wrap(words: list[str], first_prefix: str, cont_prefix: str) -> list[str]:
    lines: list[str] = []
    current = first_prefix
    empty = True
    for word in words:
        candidate = current + ("" if empty else " ") + word
        if not empty and len(candidate) > LIMIT:
            lines.append(current)
            current, empty = cont_prefix + word, False
        else:
            current, empty = candidate, False
    if not empty:
        lines.append(current)
    return lines


def _starts_new_block(line: str) -> bool:
    return bool(
        not line.strip()
        or FENCE.match(line)
        or HEADING.match(line)
        or TABLE_ROW.match(line)
        or CODE_INDENT.match(line)
    )


def reflow(text: str) -> str:
    source = text.splitlines()
    out: list[str] = []
    index = 0
    in_fence = False
    fence_marker = ""
    while index < len(source):
        line = source[index]
        if in_fence:
            out.append(line)
            if line.lstrip().startswith(fence_marker):
                in_fence = False
            index += 1
            continue
        fence = FENCE.match(line)
        if fence:
            in_fence, fence_marker = True, fence.group(1)
            out.append(line)
            index += 1
            continue
        if _starts_new_block(line):
            out.append(line)
            index += 1
            continue
        start = index
        marker = LIST_MARKER.match(line)
        if marker:
            first_prefix = marker.group(1) + marker.group(2) + marker.group(3)
            cont_prefix = " " * len(first_prefix)
            body = [line[len(first_prefix) :]]
        else:
            indent = line[: len(line) - len(line.lstrip())]
            first_prefix = cont_prefix = indent
            body = [line.strip()]
        index += 1
        while (
            index < len(source)
            and not _starts_new_block(source[index])
            and not LIST_MARKER.match(source[index])
        ):
            body.append(source[index].strip())
            index += 1
        if max((len(item) for item in source[start:index]), default=0) <= LIMIT:
            out.extend(source[start:index])
        else:
            out.extend(_wrap(" ".join(body).split(), first_prefix, cont_prefix))
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def main(argv: list[str]) -> int:
    changed: list[str] = []
    refused: list[str] = []
    for candidate in argv:
        path = Path(candidate)
        if path.suffix != ".md" or not path.is_file():
            continue
        original = path.read_text(encoding="utf-8")
        rewrapped = reflow(original)
        if rewrapped == original:
            continue
        if original.split() != rewrapped.split():
            refused.append(candidate)
            continue
        path.write_text(rewrapped, encoding="utf-8")
        changed.append(candidate)

    for path_text in changed:
        print(f"reflowed {path_text}")
    for path_text in refused:
        print(f"REFUSED {path_text}: token stream would change", file=sys.stderr)
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
