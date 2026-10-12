#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare the current Mini Kimi K3 token manifest against the r1 corpus plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Mini Kimi K3 r1 corpus manifest coverage by source."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/mini_kimi_k3/data/manifest.json"),
        help="Mini Kimi K3 token manifest to audit.",
    )
    parser.add_argument(
        "--tokens-dir",
        type=Path,
        default=Path("experiments/mini_kimi_k3/data/tokens"),
        help="Token shard root directory.",
    )
    parser.add_argument(
        "--corpus-plan",
        type=Path,
        default=Path("experiments/mini_kimi_k3/reports/r1-corpus-plan.json"),
        help="Report written by plan-r1-corpus.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/reports/r1-corpus-plan-audit.json"),
        help="Coverage/deficit report to write.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit_r1_corpus_plan(
            manifest_path=args.manifest,
            tokens_dir=args.tokens_dir,
            corpus_plan_path=args.corpus_plan,
            report_path=args.report,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 21

    print(
        "r1 corpus plan audit: "
        f"{report['status']} "
        f"({report['available_tokens']}/{report['target_tokens']} tokens, "
        f"deficit {report['deficit_tokens']})",
        file=sys.stderr,
    )
    return 0 if report["status"] == "ready" else 21


def audit_r1_corpus_plan(
    *,
    manifest_path: Path,
    tokens_dir: Path,
    corpus_plan_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise ValueError(
            "Mini-K3 r1 corpus audit must run inside the TorchTitan rootfs"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"token manifest does not exist: {manifest_path}")
    if not corpus_plan_path.is_file():
        raise FileNotFoundError(f"r1 corpus plan does not exist: {corpus_plan_path}")

    manifest = json.loads(manifest_path.read_text())
    plan = json.loads(corpus_plan_path.read_text())
    _validate_plan(plan)
    manifest_sources = _source_map(manifest, key="manifest")
    required_sources = _required_source_tokens(plan)
    _validate_source_tokens_summary(plan, required_sources)
    planned_tokens = sum(required_sources.values())
    target_tokens = _nonnegative_int(plan, "target_tokens")
    total_planned_tokens = _nonnegative_int(plan, "total_planned_tokens")
    if planned_tokens != total_planned_tokens:
        raise ValueError(
            "planned source tokens do not sum to total_planned_tokens: "
            f"{planned_tokens} != {total_planned_tokens}"
        )
    if planned_tokens != target_tokens:
        raise ValueError(
            "planned source tokens do not sum to target_tokens: "
            f"{planned_tokens} != {target_tokens}"
        )

    source_reports: dict[str, dict[str, Any]] = {}
    for source, required_tokens in required_sources.items():
        available_tokens = _manifest_source_tokens(
            source=source,
            source_data=manifest_sources.get(source),
            tokens_dir=tokens_dir,
        )
        deficit_tokens = max(0, required_tokens - available_tokens)
        if available_tokens <= 0:
            status = "missing"
        elif deficit_tokens > 0:
            status = "short"
        else:
            status = "ready"
        source_reports[source] = {
            "required_tokens": required_tokens,
            "available_tokens": available_tokens,
            "deficit_tokens": deficit_tokens,
            "status": status,
        }

    extra_sources = {
        source: _manifest_source_tokens(
            source=source,
            source_data=source_data,
            tokens_dir=tokens_dir,
        )
        for source, source_data in manifest_sources.items()
        if source not in required_sources
    }
    available_tokens = sum(item["available_tokens"] for item in source_reports.values())
    deficit_tokens = sum(item["deficit_tokens"] for item in source_reports.values())
    status = "ready" if deficit_tokens == 0 else "blocked"
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_r1_corpus_plan_audit",
        "status": status,
        "target_tokens": target_tokens,
        "planned_tokens": total_planned_tokens,
        "available_tokens": available_tokens,
        "deficit_tokens": deficit_tokens,
        "manifest": str(manifest_path),
        "tokens_dir": str(tokens_dir),
        "corpus_plan": str(corpus_plan_path),
        "sources": source_reports,
        "extra_manifest_sources": extra_sources,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("kind") != "mini_kimi_k3_r1_corpus_plan":
        raise ValueError("corpus plan kind must be mini_kimi_k3_r1_corpus_plan")
    if plan.get("schema_version") != 1:
        raise ValueError("corpus plan schema_version must be 1")
    if plan.get("status") != "planned":
        raise ValueError("corpus plan status must be planned")


def _source_map(data: dict[str, Any], *, key: str) -> dict[str, Any]:
    sources = data.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError(f"{key} must define a nonempty sources object")
    for source, source_data in sources.items():
        if not isinstance(source, str) or not source:
            raise ValueError(f"{key} source names must be nonempty strings")
        if not isinstance(source_data, dict):
            raise ValueError(f"{key} source {source!r} must be an object")
    return sources


def _nonnegative_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"corpus plan {key} must be a nonnegative integer")
    return value


def _required_source_tokens(plan: dict[str, Any]) -> dict[str, int]:
    sources = _source_map(plan, key="corpus plan")
    required: dict[str, int] = {}
    for source, source_data in sources.items():
        value = source_data.get("target_tokens")
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"corpus plan source {source!r} target_tokens must be a nonnegative integer"
            )
        required[source] = value
    return required


def _validate_source_tokens_summary(
    plan: dict[str, Any], required_sources: dict[str, int]
) -> None:
    source_tokens = plan.get("source_tokens")
    if source_tokens is None:
        return
    if not isinstance(source_tokens, dict):
        raise ValueError("corpus plan source_tokens must be an object when present")
    if set(source_tokens) != set(required_sources):
        raise ValueError("corpus plan source_tokens keys must match sources")
    for source, required_tokens in required_sources.items():
        value = source_tokens[source]
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"corpus plan source_tokens.{source} must be a nonnegative integer"
            )
        if value != required_tokens:
            raise ValueError(
                f"corpus plan source_tokens mismatch for {source!r}: "
                f"{value} != {required_tokens}"
            )


def _manifest_source_tokens(
    *,
    source: str,
    source_data: Any,
    tokens_dir: Path,
) -> int:
    if not isinstance(source_data, dict):
        return 0
    shards = source_data.get("shards", [])
    if not isinstance(shards, list) or not all(
        isinstance(name, str) for name in shards
    ):
        raise ValueError(f"manifest source {source!r} shards must be a string list")
    shard_metadata = source_data.get("shard_metadata", {})
    if shard_metadata is None:
        shard_metadata = {}
    if not isinstance(shard_metadata, dict):
        raise ValueError(f"manifest source {source!r} shard_metadata must be an object")

    if len(shards) != len(set(shards)):
        raise ValueError(f"manifest source {source!r} contains a duplicate shard")

    total = 0
    source_dir = (tokens_dir / source).resolve()
    for shard_name in shards:
        shard_path = (source_dir / shard_name).resolve()
        if not shard_path.is_relative_to(source_dir):
            raise ValueError(
                f"manifest source {source!r} shard {shard_name} must stay under "
                f"{source_dir}"
            )
        if not shard_path.is_file():
            raise FileNotFoundError(
                f"manifest source {source!r} shard does not exist: {shard_path}"
            )
        size = shard_path.stat().st_size
        if size % 4 != 0:
            raise ValueError(
                f"manifest source {source!r} shard {shard_name} size is not "
                "divisible by uint32 width"
            )
        actual_tokens = size // 4
        metadata = shard_metadata.get(shard_name)
        if isinstance(metadata, dict) and "num_tokens" in metadata:
            if not isinstance(metadata["num_tokens"], int):
                raise ValueError(
                    f"manifest source {source!r} shard {shard_name} "
                    "shard_metadata num_tokens must be an integer"
                )
            metadata_tokens = metadata["num_tokens"]
            if metadata_tokens != actual_tokens:
                raise ValueError(
                    f"manifest source {source!r} shard {shard_name} "
                    "shard_metadata num_tokens mismatch: "
                    f"{metadata_tokens} != {actual_tokens}"
                )
        total += actual_tokens
    return total


if __name__ == "__main__":
    raise SystemExit(main())
