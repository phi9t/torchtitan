#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Write a Mini Kimi K3 local Stage 1 source input manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
INPUT_FORMATS = {
    "text",
    "jsonl-text",
    "jsonl-tokens",
    "parquet-text",
    "parquet-tokens",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a mini_kimi_k3_source_input_manifest for already-local files."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Local input file. May be passed multiple times.",
    )
    parser.add_argument(
        "--input-format",
        choices=sorted(INPUT_FORMATS),
        required=True,
        help="Shared format for every listed input file.",
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokens-field", default="tokens")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = write_stage1_input_manifest(
            output=args.output,
            input_paths=args.input,
            input_format=args.input_format,
            text_field=args.text_field,
            tokens_field=args.tokens_field,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 21

    print(
        "wrote Mini Kimi K3 Stage 1 input manifest "
        f"{args.output}: {len(manifest['inputs'])} file(s)",
        file=sys.stderr,
    )
    return 0


def write_stage1_input_manifest(
    *,
    output: Path,
    input_paths: list[Path],
    input_format: str,
    text_field: str,
    tokens_field: str,
) -> dict[str, Any]:
    if input_format not in INPUT_FORMATS:
        raise ValueError(f"unsupported input format: {input_format!r}")
    if not input_paths:
        raise ValueError("at least one input file is required")
    if not text_field:
        raise ValueError("text_field must be nonempty")
    if not tokens_field:
        raise ValueError("tokens_field must be nonempty")
    if output.exists():
        raise ValueError(f"output manifest already exists: {output}")

    inputs = []
    seen_paths: set[Path] = set()
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"input file does not exist: {path}")
        resolved = path.resolve()
        if resolved in seen_paths:
            raise ValueError(f"duplicate input file: {resolved}")
        seen_paths.add(resolved)
        inputs.append(
            {
                "path": str(resolved),
                "bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_source_input_manifest",
        "input_format": input_format,
        "inputs": inputs,
    }
    if input_format in {"jsonl-text", "parquet-text"}:
        manifest["text_field"] = text_field
    if input_format in {"jsonl-tokens", "parquet-tokens"}:
        manifest["tokens_field"] = tokens_field

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
