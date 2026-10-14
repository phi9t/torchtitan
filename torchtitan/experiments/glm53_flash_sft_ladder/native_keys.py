# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Record installed GLM native build keys into training attempt artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_KEYS_FILE = Path("/project/venvs/b200-runtime/glm53_native_build_keys.json")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def installed_native_keys(keys_file: Path | None = None) -> dict[str, Any]:
    """Load the installed native build-key receipt."""

    if keys_file is None:
        keys_file = Path(os.environ.get("GLM53_NATIVE_KEYS_FILE", DEFAULT_KEYS_FILE))
    return _read_json(keys_file)


def write_native_keys_to_attempt(
    attempt_json: Path,
    *,
    keys_file: Path | None = None,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Merge installed native build keys into ``attempt_json``."""

    try:
        native_builds = installed_native_keys(keys_file)
    except FileNotFoundError:
        if not allow_missing:
            raise
        native_builds = {
            "schema_version": 1,
            "missing": True,
            "packages": {},
        }

    attempt = _read_json(attempt_json) if attempt_json.exists() else {}
    attempt["native_builds"] = native_builds
    _write_json_atomic(attempt_json, attempt)
    return attempt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-json", required=True, type=Path)
    parser.add_argument("--keys-file", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)

    payload = write_native_keys_to_attempt(
        args.attempt_json,
        keys_file=args.keys_file,
        allow_missing=args.allow_missing,
    )
    print(json.dumps(payload.get("native_builds", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

