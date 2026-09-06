#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prepare a local Stage 1 Mini Kimi K3 corpus manifest.

This command intentionally operates only on local files. It can build the raw
uint32 shards and provenance reports needed for a reproducible local corpus
trial, but it does not download paid or credentialed corpora and it does not
turn a placeholder decontamination note into launch-grade evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from experiments.mini_kimi_k3.build_local_shards import build_local_shards


SCHEMA_VERSION = 1
KIMI_K3_VOCAB_SIZE = 163_840


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Mini Kimi K3 Stage 1 shards from local inputs."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/mini_kimi_k3/data/manifest.json"),
        help="Manifest JSON path to create or update.",
    )
    parser.add_argument(
        "--tokens-dir",
        type=Path,
        default=Path("experiments/mini_kimi_k3/data/tokens"),
        help="Token shard root directory.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("experiments/mini_kimi_k3/reports"),
        help="Directory for local provenance reports.",
    )
    parser.add_argument("--source", required=True, help="Manifest source key.")
    parser.add_argument("--dataset-id", required=True, help="Stable dataset id.")
    parser.add_argument("--snapshot", required=True, help="Dataset snapshot label.")
    parser.add_argument("--license", required=True, help="Source license label.")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        help="Local input file. May be passed multiple times.",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help=(
            "JSON manifest listing local input files and expected hashes. "
            "Direct --input files may be omitted when this is supplied."
        ),
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
        help="How to read --input files.",
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
        "--tokenizer-sha256",
        required=True,
        help="SHA-256 or equivalent fingerprint of the Kimi K3 tokenizer assets.",
    )
    parser.add_argument(
        "--tokenizer-asset-report",
        help="Path or URI for the tokenizer asset fingerprint report.",
    )
    parser.add_argument(
        "--decontamination-report",
        type=Path,
        help="Reviewed decontamination report to record in the manifest.",
    )
    parser.add_argument(
        "--decontamination-sha256",
        help="Expected SHA-256 of --decontamination-report.",
    )
    parser.add_argument(
        "--allow-placeholder-decontamination",
        action="store_true",
        help=(
            "Write a pending decontamination marker. Full-mode preflight will "
            "continue to reject the manifest until reviewed evidence is supplied."
        ),
    )
    parser.add_argument(
        "--shard-tokens",
        type=int,
        default=100_000_000,
        help="Maximum tokens per output shard.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Stop after writing this many tokens for the source.",
    )
    parser.add_argument(
        "--decontamination-index",
        type=Path,
        help=(
            "Optional first-party NgramIndex .npz. For text inputs, matching "
            "documents are dropped before tokenization."
        ),
    )
    parser.add_argument(
        "--decontamination-min-matches",
        type=int,
        default=1,
        help="Minimum matching 13-grams required to drop a text document.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output shard prefix. Defaults to <source>-local.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = prepare_stage1_local(
            manifest_path=args.manifest,
            tokens_dir=args.tokens_dir,
            reports_dir=args.reports_dir,
            source=args.source,
            dataset_id=args.dataset_id,
            snapshot=args.snapshot,
            license_label=args.license,
            input_paths=args.input,
            input_manifest=args.input_manifest,
            input_format=args.input_format,
            text_field=args.text_field,
            tokens_field=args.tokens_field,
            tokenizer_dir=args.tokenizer_dir,
            tokenizer_sha256=args.tokenizer_sha256,
            tokenizer_asset_report=args.tokenizer_asset_report,
            decontamination_report=args.decontamination_report,
            decontamination_sha256=args.decontamination_sha256,
            allow_placeholder_decontamination=args.allow_placeholder_decontamination,
            shard_tokens=args.shard_tokens,
            max_tokens=args.max_tokens,
            decontamination_index=args.decontamination_index,
            decontamination_min_matches=args.decontamination_min_matches,
            prefix=args.prefix,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 21

    print(
        "prepared local Mini Kimi K3 Stage 1 source "
        f"{args.source}: {report['total_tokens']} tokens",
        file=sys.stderr,
    )
    return 0


def prepare_stage1_local(
    *,
    manifest_path: Path,
    tokens_dir: Path,
    reports_dir: Path,
    source: str,
    dataset_id: str,
    snapshot: str,
    license_label: str,
    input_paths: list[Path] | None,
    input_manifest: Path | None,
    input_format: str,
    text_field: str,
    tokens_field: str,
    tokenizer_dir: Path,
    tokenizer_sha256: str,
    tokenizer_asset_report: str | None,
    decontamination_report: Path | None,
    decontamination_sha256: str | None,
    allow_placeholder_decontamination: bool,
    shard_tokens: int,
    prefix: str | None,
    max_tokens: int | None = None,
    decontamination_index: Path | None = None,
    decontamination_min_matches: int = 1,
) -> dict[str, Any]:
    if not source.strip():
        raise ValueError("source must be nonempty")
    if not dataset_id.strip():
        raise ValueError("dataset_id must be nonempty")
    if not snapshot.strip():
        raise ValueError("snapshot must be nonempty")
    if not license_label.strip():
        raise ValueError("license must be nonempty")
    if not tokenizer_sha256.strip() or tokenizer_sha256.startswith("pending-"):
        raise ValueError("tokenizer_sha256 must be a non-placeholder fingerprint")
    input_spec = _resolve_input_spec(
        input_paths=input_paths,
        input_manifest=input_manifest,
        input_format=input_format,
        text_field=text_field,
        tokens_field=tokens_field,
    )

    manifest = _load_or_initialize_manifest(
        manifest_path=manifest_path,
        source=source,
        dataset_id=dataset_id,
        snapshot=snapshot,
        license_label=license_label,
        tokenizer_sha256=tokenizer_sha256,
        tokenizer_asset_report=tokenizer_asset_report,
    )
    _record_decontamination(
        manifest,
        manifest_path=manifest_path,
        decontamination_report=decontamination_report,
        decontamination_sha256=decontamination_sha256,
        allow_placeholder_decontamination=allow_placeholder_decontamination,
    )

    written = build_local_shards(
        manifest=manifest,
        manifest_path=manifest_path,
        tokens_dir=tokens_dir,
        source=source,
        input_paths=input_spec["paths"],
        input_format=input_spec["input_format"],
        text_field=input_spec["text_field"],
        tokens_field=input_spec["tokens_field"],
        tokenizer_dir=tokenizer_dir,
        shard_tokens=shard_tokens,
        max_tokens=max_tokens,
        decontamination_index=decontamination_index,
        decontamination_min_matches=decontamination_min_matches,
        prefix=prefix,
    )
    total_tokens = sum(item["tokens"] for item in written)
    report = _write_source_provenance_report(
        reports_dir=reports_dir,
        source=source,
        dataset_id=dataset_id,
        snapshot=snapshot,
        license_label=license_label,
        input_paths=input_spec["paths"],
        input_manifest=input_spec["manifest"],
        input_format=input_spec["input_format"],
        text_field=input_spec["text_field"],
        tokens_field=input_spec["tokens_field"],
        shard_tokens=shard_tokens,
        max_tokens=max_tokens,
        written=written,
        total_tokens=total_tokens,
    )
    report_hash = _sha256(report)

    manifest = json.loads(manifest_path.read_text())
    source_data = manifest["sources"][source]
    provenance = source_data.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError(f"manifest source {source!r} provenance must be an object")
    provenance.update(
        {
            "snapshot": snapshot,
            "license": license_label,
            "report": str(report),
            "sha256": report_hash,
        }
    )
    source_data["dataset_id"] = dataset_id
    source_data["local_stage1"] = {
        "report": str(report),
        "sha256": report_hash,
        "total_tokens": total_tokens,
        "input_format": input_spec["input_format"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return json.loads(report.read_text())


def _resolve_input_spec(
    *,
    input_paths: list[Path] | None,
    input_manifest: Path | None,
    input_format: str,
    text_field: str,
    tokens_field: str,
) -> dict[str, Any]:
    direct_paths = [path.resolve() for path in input_paths or []]
    if input_manifest is None:
        if not direct_paths:
            raise ValueError(
                "prepare-stage1-local requires --input or --input-manifest"
            )
        _check_unique_input_paths(direct_paths)
        return {
            "paths": direct_paths,
            "manifest": None,
            "input_format": input_format,
            "text_field": text_field,
            "tokens_field": tokens_field,
        }

    data = json.loads(input_manifest.read_text())
    if data.get("kind") != "mini_kimi_k3_source_input_manifest":
        raise ValueError(
            "input manifest kind must be mini_kimi_k3_source_input_manifest"
        )
    if data.get("schema_version") != 1:
        raise ValueError("input manifest schema_version must be 1")
    manifest_format = data.get("input_format", input_format)
    if manifest_format not in {
        "text",
        "jsonl-text",
        "jsonl-tokens",
        "parquet-text",
        "parquet-tokens",
    }:
        raise ValueError(f"unsupported input manifest format: {manifest_format!r}")
    manifest_text_field = data.get("text_field", text_field)
    manifest_tokens_field = data.get("tokens_field", tokens_field)
    if not isinstance(manifest_text_field, str) or not manifest_text_field:
        raise ValueError("input manifest text_field must be nonempty text")
    if not isinstance(manifest_tokens_field, str) or not manifest_tokens_field:
        raise ValueError("input manifest tokens_field must be nonempty text")

    entries = data.get("inputs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("input manifest must include a nonempty inputs list")
    manifest_dir = input_manifest.parent
    manifest_paths: list[Path] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"input manifest entry {index} must be an object")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"input manifest entry {index} must include path")
        path = Path(raw_path)
        if not path.is_absolute():
            path = manifest_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"input manifest file does not exist: {path}")
        expected_sha256 = entry.get("sha256")
        if expected_sha256 is not None:
            if not isinstance(expected_sha256, str) or not expected_sha256:
                raise ValueError(
                    f"input manifest entry {index} sha256 must be nonempty text"
                )
            actual_sha256 = _sha256(path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"input manifest sha256 mismatch for {path} "
                    f"({actual_sha256} != {expected_sha256})"
                )
        manifest_paths.append(path.resolve())
    manifest_paths.extend(direct_paths)
    _check_unique_input_paths(manifest_paths)
    return {
        "paths": manifest_paths,
        "manifest": {
            "path": str(input_manifest),
            "sha256": _sha256(input_manifest),
        },
        "input_format": manifest_format,
        "text_field": manifest_text_field,
        "tokens_field": manifest_tokens_field,
    }


def _check_unique_input_paths(paths: list[Path]) -> None:
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            raise ValueError(f"duplicate input file: {path}")
        seen.add(path)


def _load_or_initialize_manifest(
    *,
    manifest_path: Path,
    source: str,
    dataset_id: str,
    snapshot: str,
    license_label: str,
    tokenizer_sha256: str,
    tokenizer_asset_report: str | None,
) -> dict[str, Any]:
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "mini_kimi_k3_token_manifest",
            "mode": "full",
            "sources": {},
        }
    sources = manifest.setdefault("sources", {})
    if not isinstance(sources, dict):
        raise ValueError("manifest must define a sources object")

    tokenizer = manifest.setdefault(
        "tokenizer",
        {
            "name": "Kimi K3",
            "vocab_size": KIMI_K3_VOCAB_SIZE,
        },
    )
    if not isinstance(tokenizer, dict):
        raise ValueError("manifest tokenizer must be an object")
    tokenizer.update(
        {
            "name": tokenizer.get("name") or "Kimi K3",
            "vocab_size": KIMI_K3_VOCAB_SIZE,
            "sha256": tokenizer_sha256,
        }
    )
    if tokenizer_asset_report:
        tokenizer["asset_report"] = tokenizer_asset_report

    source_data = sources.setdefault(source, {})
    if not isinstance(source_data, dict):
        raise ValueError(f"manifest source {source!r} must be an object")
    source_data["dataset_id"] = dataset_id
    provenance = source_data.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError(f"manifest source {source!r} provenance must be an object")
    provenance.update({"snapshot": snapshot, "license": license_label})
    source_data.setdefault("shards", [])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _record_decontamination(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    decontamination_report: Path | None,
    decontamination_sha256: str | None,
    allow_placeholder_decontamination: bool,
) -> None:
    if decontamination_report is None:
        if not allow_placeholder_decontamination:
            raise ValueError(
                "prepare-stage1-local requires --decontamination-report or "
                "--allow-placeholder-decontamination"
            )
        manifest["decontamination"] = {
            "status": "pending-local-review",
            "report": "pending-local-review",
            "sha256": "pending-local-review",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return

    if not decontamination_report.is_file():
        raise FileNotFoundError(
            f"decontamination report does not exist: {decontamination_report}"
        )
    actual = _sha256(decontamination_report)
    if decontamination_sha256 is not None and decontamination_sha256 != actual:
        raise ValueError("decontamination report sha256 mismatch")
    manifest["decontamination"] = {
        "status": "completed",
        "report": str(decontamination_report),
        "sha256": actual,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _write_source_provenance_report(
    *,
    reports_dir: Path,
    source: str,
    dataset_id: str,
    snapshot: str,
    license_label: str,
    input_paths: list[Path],
    input_manifest: dict[str, Any] | None,
    input_format: str,
    text_field: str,
    tokens_field: str,
    shard_tokens: int,
    max_tokens: int | None,
    written: list[dict[str, Any]],
    total_tokens: int,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{source}-provenance.json"
    input_reports = []
    for input_path in input_paths:
        input_reports.append(
            {
                "path": str(input_path),
                "bytes": input_path.stat().st_size,
                "sha256": _sha256(input_path),
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_source_provenance",
        "source": source,
        "dataset_id": dataset_id,
        "snapshot": snapshot,
        "license": license_label,
        "input_format": input_format,
        "text_field": text_field,
        "tokens_field": tokens_field,
        "shard_tokens": shard_tokens,
        "max_tokens": max_tokens,
        "total_tokens": total_tokens,
        "input_files": input_reports,
        "input_manifest": input_manifest,
        "shards": [
            {
                "path": str(item["path"]),
                "name": item["path"].name,
                "tokens": item["tokens"],
                "sha256": _sha256(item["path"]),
            }
            for item in written
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
