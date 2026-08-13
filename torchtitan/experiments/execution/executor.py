# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Command executor seam for the typed lifecycle.

RunAttempt.run_stage drives an executor rather than calling subprocess
directly, so tests can substitute a deterministic FakeExecutor and the default
SubprocessExecutor stays a thin adapter. This is the seam the roadmap Section 4
requires between the lifecycle and real process launches.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class CommandResult:
    """The outcome of one command launch.

    ``terminal_kind`` lets an executor declare a terminal stage state that a
    bare return code cannot express: a gate that blocked the launch, or a
    cancellation (timeout, SIGINT) that interrupted it. When None the lifecycle
    derives succeeded/failed from ``return_code`` (roadmap Section 5).
    """

    return_code: int
    stdout: str = ""
    stderr: str = ""
    terminal_kind: str | None = None


class Executor(Protocol):
    def run(self, argv: list[str], *, cwd: Path | None = None) -> CommandResult:
        """Launch argv and return its CommandResult."""


class SubprocessExecutor:
    """Default executor: run argv as a host subprocess and capture output."""

    def run(self, argv: list[str], *, cwd: Path | None = None) -> CommandResult:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        return CommandResult(
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


class FakeExecutor:
    """Deterministic executor keyed by argv tuple, for tests.

    Any argv not present in the mapping defaults to a successful result, so a
    test only has to declare the commands whose outcome it cares about.
    """

    def __init__(self, results: dict[tuple[str, ...], CommandResult] | None = None):
        self._results = dict(results or {})
        self.calls: list[list[str]] = []

    def run(self, argv: list[str], *, cwd: Path | None = None) -> CommandResult:
        self.calls.append(list(argv))
        return self._results.get(tuple(argv), CommandResult(return_code=0))
