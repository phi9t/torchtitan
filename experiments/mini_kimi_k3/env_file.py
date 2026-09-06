#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Environment-file loading helpers for Mini Kimi K3 operators."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any


_SENSITIVE_KEY_RE = re.compile(r"(token|secret|password|api[_-]?key)", re.IGNORECASE)


def build_command_env(
    env_file: Path | None,
) -> tuple[dict[str, str], dict[str, Any] | None]:
    command_env = dict(os.environ)
    if env_file is None:
        return command_env, None
    loaded = _parse_env_file(env_file)
    command_env.update(loaded)
    return command_env, {
        "path": str(env_file),
        "status": "loaded",
        "keys": sorted(loaded),
    }


def _parse_env_file(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"env file does not exist: {path}") from exc
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            raise ValueError(f"invalid env file line {line_number}: expected KEY=VALUE")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"invalid env file key on line {line_number}: {key!r}")
        try:
            parsed = shlex.split(raw_value, comments=False, posix=True)
        except ValueError as exc:
            raise ValueError(
                f"invalid env file value on line {line_number}: {exc}"
            ) from exc
        if len(parsed) > 1:
            raise ValueError(
                f"invalid env file value on line {line_number}: expected one value"
            )
        values[key] = parsed[0] if parsed else ""
    return values


def redact_sensitive_env_values(text: str, env: dict[str, str]) -> str:
    redacted = text
    for key, value in env.items():
        if not value or not _SENSITIVE_KEY_RE.search(key):
            continue
        redacted = redacted.replace(value, "<redacted>")
    return redacted
