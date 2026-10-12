#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build a launch-grade Mini Kimi K3 benchmark decontamination index locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FIRST_PARTY_DATA = (
    REPO_ROOT
    / "experiments"
    / "mini_kimi_k3"
    / "assets"
    / "mini-kimi-k3-vizuara"
    / "data"
)
if FIRST_PARTY_DATA.is_dir() and str(FIRST_PARTY_DATA) not in sys.path:
    sys.path.insert(0, str(FIRST_PARTY_DATA))

try:
    from bench_extract import extract_rows, missing_fields, row_text
    from ngram import NGRAM_N, NgramIndex, text_ngram_hashes
    from sources import BENCHMARKS
except ImportError as exc:  # pragma: no cover - exercised only without assets.
    raise ImportError(
        "Mini-K3 launch-grade decontamination requires first-party data assets"
    ) from exc


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    fields: list[str]
    list_fields: list[str]


BenchmarkLoader = Callable[[BenchmarkSpec, dict[str, Any]], Iterable[dict[str, Any]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Mini Kimi K3 launch-grade 13-gram benchmark suite index."
        )
    )
    parser.add_argument(
        "--source-resolution-report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/reports/stage1-source-resolution.json"),
        help="Ready report written by probe-stage1-sources.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("experiments/mini_kimi_k3/data/decontamination/index.npz"),
        help="Output first-party NgramIndex .npz.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/reports/decontamination.json"),
        help="Output decontamination report consumed by full preflight.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_decontamination_index(
            source_resolution_report=args.source_resolution_report,
            index_path=args.index,
            report_path=args.report,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 21

    print(
        "built Mini Kimi K3 launch-grade decontamination index: "
        f"{report['n_unique_13grams']} unique 13-grams",
        file=sys.stderr,
    )
    print(f"wrote decontamination report: {args.report}", file=sys.stderr)
    return 0


def build_decontamination_index(
    *,
    source_resolution_report: Path,
    index_path: Path,
    report_path: Path,
    benchmark_loader: BenchmarkLoader | None = None,
    benchmark_specs: list[BenchmarkSpec] | None = None,
) -> dict[str, Any]:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise ValueError(
            "Mini-K3 launch-grade decontamination index build must run inside rootfs"
        )
    resolution = _read_ready_resolution(source_resolution_report)
    specs = benchmark_specs or [
        BenchmarkSpec(
            name=benchmark.name,
            fields=list(benchmark.fields),
            list_fields=list(benchmark.list_fields),
        )
        for benchmark in BENCHMARKS
    ]
    loader = benchmark_loader or _load_hf_benchmark_rows

    started = time.time()
    per_benchmark: dict[str, list[str]] = {}
    benchmark_stats: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for spec in specs:
        resolved = resolution["benchmarks"].get(spec.name)
        if not isinstance(resolved, dict) or resolved.get("status") != "resolved":
            problems.append(f"{spec.name}: unresolved in source-resolution report")
            continue
        rows = list(loader(spec, resolved))
        if not rows:
            problems.append(f"{spec.name}: loader returned no rows")
            continue
        sample = rows[0]
        miss = missing_fields(sample, spec.fields, spec.list_fields)
        texts = extract_rows(rows, spec.fields, spec.list_fields)
        ngrams = [int(text_ngram_hashes(text, n=NGRAM_N).size) for text in texts]
        too_short = sum(1 for count in ngrams if count == 0)
        per_benchmark[spec.name] = texts
        benchmark_stats[spec.name] = {
            "repo": resolved.get("repo"),
            "config": resolved.get("config"),
            "split": resolved.get("split"),
            "rows": len(rows),
            "rows_with_text": len(texts),
            "rows_too_short_for_13gram": too_short,
            "ngrams": sum(ngrams),
            "columns_used": [*spec.fields, *spec.list_fields],
            "columns_missing": miss,
        }
        if miss:
            problems.append(f"{spec.name}: expected columns absent: {miss}")

    empty = [
        name
        for name, stats in benchmark_stats.items()
        if stats["ngrams"] == 0
    ]
    if empty:
        problems.append("benchmark input yielded no 13-gram hashes: " + ", ".join(empty))

    index = NgramIndex.build(per_benchmark, n=NGRAM_N)
    if len(index) == 0:
        problems.append("benchmark suite yielded no 13-gram hashes")

    self_check = _self_check_index(index, per_benchmark)
    if self_check["rows_missed"]:
        problems.append(
            "self-check: "
            f"{self_check['rows_missed']}/{self_check['rows_checked']} "
            "benchmark rows not detected by their own index"
        )
    if problems:
        raise ValueError("; ".join(problems))

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(index_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_decontamination_report",
        "status": "completed",
        "scope": "launch_grade_benchmark_suite",
        "ngram_n": NGRAM_N,
        "source_resolution_report": {
            "path": str(source_resolution_report),
            "sha256": _sha256(source_resolution_report),
        },
        "benchmarks_covered": sorted(per_benchmark),
        "benchmark_stats": benchmark_stats,
        "self_check": self_check,
        "index": {
            "path": str(index_path),
            "bytes": index_path.stat().st_size,
            "sha256": _sha256(index_path),
        },
        "n_unique_13grams": len(index),
        "seconds": round(time.time() - started, 1),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _read_ready_resolution(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"source resolution report does not exist: {path}")
    data = json.loads(path.read_text())
    if data.get("kind") != "mini_kimi_k3_stage1_source_resolution":
        raise ValueError("source resolution kind must be mini_kimi_k3_stage1_source_resolution")
    if data.get("schema_version") != 1:
        raise ValueError("source resolution schema_version must be 1")
    if data.get("status") != "ready":
        raise ValueError("source resolution report must be ready")
    benchmarks = data.get("benchmarks")
    if not isinstance(benchmarks, dict) or not benchmarks:
        raise ValueError("source resolution report must define benchmarks")
    return data


def _load_hf_benchmark_rows(
    spec: BenchmarkSpec,
    resolved: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ValueError(
            "build-decontamination-index requires datasets and huggingface_hub "
            "in the rootfs"
        ) from exc

    repo = _nonempty_text(resolved, "repo", spec.name)
    split = _nonempty_text(resolved, "split", spec.name)
    config = resolved.get("config")
    if config is not None and not isinstance(config, str):
        raise ValueError(f"benchmark {spec.name!r} config must be text or null")
    try:
        return load_dataset(repo, config, split=split)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise

    files = HfApi().list_repo_files(
        repo,
        repo_type="dataset",
        revision="refs/convert/parquet",
    )
    wanted = [
        path
        for path in files
        if path.endswith(".parquet") and f"/{split}/" in path
    ]
    if config:
        config_prefix = config.replace(".", "_")
        narrowed = [
            path
            for path in wanted
            if path.startswith((f"{config}/", f"{config_prefix}/"))
        ]
        wanted = narrowed or wanted
    if not wanted:
        raise ValueError(
            f"benchmark {spec.name!r} has no parquet conversion files for "
            f"split={split!r}"
        )
    urls = [
        f"hf://datasets/{repo}@refs%2Fconvert%2Fparquet/{path}"
        for path in sorted(wanted)
    ]
    return load_dataset("parquet", data_files=urls, split="train")


def _self_check_index(
    index: Any,
    per_benchmark: dict[str, list[str]],
) -> dict[str, int]:
    checked = 0
    missed = 0
    for name, texts in per_benchmark.items():
        for text in texts[:200]:
            if text_ngram_hashes(text, n=NGRAM_N).size == 0:
                continue
            checked += 1
            hit, which = index.check(text, n=NGRAM_N)
            if not hit or name not in which:
                missed += 1
    return {"rows_checked": checked, "rows_missed": missed}


def _nonempty_text(data: dict[str, Any], key: str, name: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"benchmark {name!r} {key} must be nonempty text")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
