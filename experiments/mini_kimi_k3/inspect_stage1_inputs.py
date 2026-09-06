#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inspect local Mini Kimi K3 Stage 1 inputs without writing token shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from experiments.mini_kimi_k3.build_local_shards import (
    _iter_token_sequences,
    _load_text_encoder,
    _validate_tokens,
)
from experiments.mini_kimi_k3.prepare_stage1_local import _resolve_input_spec


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a Mini Kimi K3 local Stage 1 input manifest without "
            "building shards."
        )
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        action="append",
        required=True,
        help="mini_kimi_k3_source_input_manifest JSON to inspect.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Inspection report JSON path to write.",
    )
    parser.add_argument(
        "--input-format",
        choices=(
            "text",
            "jsonl-text",
            "jsonl-tokens",
            "parquet-text",
            "parquet-tokens",
        ),
        default="jsonl-tokens",
        help="Fallback input format when the manifest does not specify one.",
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokens-field", default="tokens")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("experiments/mini_kimi_k3/assets/kimi-k3-tokenizer"),
        help="Kimi K3 tokenizer asset directory for text inputs.",
    )
    parser.add_argument(
        "--min-total-tokens",
        type=int,
        default=0,
        help="Fail after inspection if the resolved inputs contain fewer tokens.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = inspect_stage1_inputs(
            input_manifest=args.input_manifest[0],
            input_manifests=args.input_manifest,
            report_path=args.report,
            input_format=args.input_format,
            text_field=args.text_field,
            tokens_field=args.tokens_field,
            tokenizer_dir=args.tokenizer_dir,
            min_total_tokens=args.min_total_tokens,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 21

    print(
        "inspected Mini Kimi K3 Stage 1 inputs: "
        f"{report['total_documents']} documents, {report['total_tokens']} tokens",
        file=sys.stderr,
    )
    if report["status"] != "pass":
        print(report["detail"], file=sys.stderr)
    return 0 if report["status"] == "pass" else 21


def inspect_stage1_inputs(
    *,
    input_manifest: Path,
    input_manifests: list[Path] | None = None,
    report_path: Path,
    input_format: str,
    text_field: str,
    tokens_field: str,
    tokenizer_dir: Path,
    min_total_tokens: int = 0,
) -> dict[str, Any]:
    if min_total_tokens < 0:
        raise ValueError(f"min_total_tokens must be >= 0, got {min_total_tokens}")
    manifest_paths = input_manifests or [input_manifest]
    input_specs = [
        _resolve_input_spec(
            input_paths=None,
            input_manifest=manifest_path,
            input_format=input_format,
            text_field=text_field,
            tokens_field=tokens_field,
        )
        for manifest_path in manifest_paths
    ]
    formats = {spec["input_format"] for spec in input_specs}
    if len(formats) != 1:
        raise ValueError("all input manifests must use the same input_format")
    resolved_format = input_specs[0]["input_format"]
    text_input_formats = {"text", "jsonl-text", "parquet-text"}
    encoder = (
        _load_text_encoder(tokenizer_dir)
        if resolved_format in text_input_formats
        else None
    )

    input_files = []
    total_documents = 0
    total_tokens = 0
    for input_spec in input_specs:
        for input_path in input_spec["paths"]:
            documents = 0
            tokens = 0
            for token_ids in _iter_token_sequences(
                input_paths=[input_path],
                input_format=resolved_format,
                text_field=input_spec["text_field"],
                tokens_field=input_spec["tokens_field"],
                encoder=encoder,
                decontamination=None,
                decontamination_stats=None,
            ):
                arr = _validate_tokens(token_ids)
                if arr.size == 0:
                    continue
                documents += 1
                tokens += int(arr.size)
            input_files.append(
                {
                    "path": str(input_path),
                    "bytes": input_path.stat().st_size,
                    "sha256": _sha256(input_path),
                    "documents": documents,
                    "tokens": tokens,
                    "text_field": input_spec["text_field"],
                    "tokens_field": input_spec["tokens_field"],
                }
            )
            total_documents += documents
            total_tokens += tokens

    status = "pass"
    detail = None
    if total_tokens < min_total_tokens:
        status = "fail"
        detail = (
            "total token count below required minimum "
            f"({total_tokens} < {min_total_tokens})"
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_stage1_input_inspection",
        "status": status,
        "input_manifest": _input_manifest_evidence(input_specs),
        "input_format": resolved_format,
        "text_fields": sorted({spec["text_field"] for spec in input_specs}),
        "tokens_fields": sorted({spec["tokens_field"] for spec in input_specs}),
        "min_total_tokens": min_total_tokens,
        "total_files": len(input_files),
        "total_documents": total_documents,
        "total_tokens": total_tokens,
        "input_files": input_files,
    }
    if detail is not None:
        report["detail"] = detail
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _input_manifest_evidence(input_specs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(input_specs) == 1:
        return input_specs[0]["manifest"]
    manifests = [spec["manifest"] for spec in input_specs]
    aggregate = hashlib.sha256()
    for manifest in manifests:
        aggregate.update(manifest["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(manifest["sha256"].encode("utf-8"))
        aggregate.update(b"\0")
    return {
        "paths": [manifest["path"] for manifest in manifests],
        "sha256": aggregate.hexdigest(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
