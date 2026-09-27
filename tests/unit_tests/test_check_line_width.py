# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the markdown width guard.

The assertions that matter here are the ones proving the exemptions do not
swallow ordinary prose. A width checker whose fence or token exemption is too
broad reports success on everything, which is the "guard that cannot fail"
shape this program keeps finding in its own tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "check_line_width.py"

OVER = "word " * 20  # 100 columns of plain, wrappable prose.


def _run(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(path)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "note.md"
    path.write_text(body, encoding="utf-8")
    return path


def test_long_prose_line_is_reported_with_its_line_number(tmp_path: Path) -> None:
    result = _run(_write(tmp_path, f"short\n{OVER}\nshort\n"))
    assert result.returncode == 1, result.stdout
    assert "note.md:2:" in result.stdout
    assert "100 columns" in result.stdout


def test_a_conforming_file_passes(tmp_path: Path) -> None:
    result = _run(_write(tmp_path, "a short line\nanother short line\n"))
    assert result.returncode == 0, result.stdout


def test_fenced_code_is_exempt_but_the_fence_closes(tmp_path: Path) -> None:
    """The fence must stop exempting at its closing marker.

    A checker that never leaves the fence state would pass every file whose
    first fence is opened, which is indistinguishable from success.
    """
    body = f"```\n{OVER}\n```\n{OVER}\n"
    result = _run(_write(tmp_path, body))
    assert result.returncode == 1, result.stdout
    # Line 2 is inside the fence, line 4 is after it closed.
    assert "note.md:4:" in result.stdout
    assert "note.md:2:" not in result.stdout


def test_table_rows_and_indented_code_are_exempt(tmp_path: Path) -> None:
    body = f"| {OVER} |\n\n    {OVER}\n"
    result = _run(_write(tmp_path, body))
    assert result.returncode == 0, result.stdout


def test_an_unbreakable_token_is_exempt_but_a_short_url_is_not(
    tmp_path: Path,
) -> None:
    """Only a token that cannot be wrapped earns the exemption.

    Exempting any line that merely contains a URL would let most prose through,
    so the second case must still be reported.
    """
    unbreakable = "see https://example.com/" + "z" * 90 + "\n"
    wrappable = "see https://example.com/short and then " + "word " * 12 + "\n"
    assert len(wrappable.rstrip()) > 80

    assert _run(_write(tmp_path, unbreakable)).returncode == 0
    reported = _run(_write(tmp_path, wrappable))
    assert reported.returncode == 1, reported.stdout


def test_width_allow_pragma_suppresses_one_line(tmp_path: Path) -> None:
    result = _run(_write(tmp_path, f"{OVER} width-allow\n"))
    assert result.returncode == 0, result.stdout


def test_the_maintained_corpus_conforms() -> None:
    """The corpus the hook covers must actually pass, not merely be covered.

    Registering a hook whose corpus fails is how a guard gets disabled later.
    """
    corpus = sorted((REPO_ROOT / ".scratch" / "qwen3-formal-verifier").rglob("*.md"))
    assert corpus, "expected the formal-verifier notes to exist"
    result = subprocess.run(
        [sys.executable, str(CHECKER), *[str(p) for p in corpus]],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
