#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local 13-gram decontamination for Mini Kimi K3 source-prep trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
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
    from ngram import NGRAM_N, NgramIndex, text_ngram_hashes
except ImportError as exc:  # pragma: no cover - exercised only without assets.
    raise ImportError(
        "Mini-K3 local decontamination requires the first-party ngram.py asset"
    ) from exc


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter local text documents with Mini-K3 13-gram decontamination."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        action="append",
        required=True,
        help=(
            "Local benchmark text file, JSONL, or Parquet file. May be passed "
            "multiple times."
        ),
    )
    parser.add_argument(
        "--benchmark-format",
        choices=("text", "jsonl-text", "parquet-text"),
        default="jsonl-text",
    )
    parser.add_argument("--benchmark-text-field", default="text")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Local source text file, JSONL, or Parquet file. May be passed multiple times.",
    )
    parser.add_argument(
        "--input-format",
        choices=("text", "jsonl-text", "parquet-text"),
        default="jsonl-text",
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--min-matches", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = decontaminate_local(
            benchmark_paths=args.benchmark,
            benchmark_format=args.benchmark_format,
            benchmark_text_field=args.benchmark_text_field,
            input_paths=args.input,
            input_format=args.input_format,
            text_field=args.text_field,
            output_path=args.output,
            report_path=args.report,
            source=args.source,
            min_matches=args.min_matches,
            max_examples=args.max_examples,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 21

    print(
        "wrote Mini Kimi K3 local decontamination report: "
        f"{args.report} ({report['documents_removed']} removed)",
        file=sys.stderr,
    )
    return 0


def decontaminate_local(
    *,
    benchmark_paths: list[Path],
    benchmark_format: str,
    benchmark_text_field: str,
    input_paths: list[Path],
    input_format: str,
    text_field: str,
    output_path: Path,
    report_path: Path,
    source: str,
    min_matches: int,
    max_examples: int,
) -> dict[str, Any]:
    if min_matches <= 0:
        raise ValueError("min_matches must be positive")
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")
    if not source.strip():
        raise ValueError("source must be nonempty")

    per_benchmark = _read_benchmark_texts(
        benchmark_paths,
        input_format=benchmark_format,
        text_field=benchmark_text_field,
    )
    benchmark_stats = _benchmark_stats(per_benchmark)
    empty = [name for name, stats in benchmark_stats.items() if stats["ngrams"] == 0]
    if empty:
        raise ValueError(
            "benchmark input yielded no 13-gram hashes: " + ", ".join(empty)
        )
    index = NgramIndex.build(per_benchmark, n=NGRAM_N)
    if len(index) == 0:
        raise ValueError("benchmark input yielded no 13-gram hashes")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    documents_seen = 0
    documents_removed = 0
    hits_by_benchmark: dict[str, int] = {}
    removed_examples: list[dict[str, Any]] = []
    with output_path.open("w") as output:
        for input_path, line_no, text in _iter_text_records(
            input_paths,
            input_format=input_format,
            text_field=text_field,
        ):
            documents_seen += 1
            contaminated, hits = index.check(text, n=NGRAM_N, min_matches=min_matches)
            if contaminated:
                documents_removed += 1
                for name, count in hits.items():
                    hits_by_benchmark[name] = hits_by_benchmark.get(name, 0) + count
                if len(removed_examples) < max_examples:
                    removed_examples.append(
                        {
                            "input": str(input_path),
                            "line": line_no,
                            "benchmarks": sorted(hits),
                            "ngram_hits": sum(hits.values()),
                            "text_prefix": text[:240],
                        }
                    )
                continue
            output.write(json.dumps({"text": text}, sort_keys=True) + "\n")

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_decontamination_report",
        "status": "completed",
        "scope": "local_explicit_benchmarks",
        "source": source,
        "ngram_n": NGRAM_N,
        "min_matches": min_matches,
        "documents_scanned": documents_seen,
        "documents_removed": documents_removed,
        "documents_kept": documents_seen - documents_removed,
        "removal_rate": documents_removed / documents_seen if documents_seen else 0.0,
        "benchmarks_covered": sorted(per_benchmark),
        "ngram_hits_by_benchmark": dict(
            sorted(hits_by_benchmark.items(), key=lambda item: item[0])
        ),
        "benchmark_stats": benchmark_stats,
        "input_files": _file_reports(input_paths),
        "benchmark_files": _file_reports(benchmark_paths),
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        },
        "removed_examples": removed_examples,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _read_benchmark_texts(
    paths: list[Path],
    *,
    input_format: str,
    text_field: str,
) -> dict[str, list[str]]:
    per_benchmark: dict[str, list[str]] = {}
    for path in paths:
        name = path.stem
        texts = [
            text
            for _input_path, _line_no, text in _iter_text_records(
                [path],
                input_format=input_format,
                text_field=text_field,
            )
        ]
        if not texts:
            raise ValueError(f"benchmark input has no text rows: {path}")
        per_benchmark[name] = texts
    return per_benchmark


def _benchmark_stats(per_benchmark: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for name, texts in per_benchmark.items():
        ngrams = [int(text_ngram_hashes(text, n=NGRAM_N).size) for text in texts]
        indexed = sum(1 for count in ngrams if count > 0)
        rows = len(texts)
        stats[name] = {
            "rows": rows,
            "rows_indexed": indexed,
            "rows_too_short": rows - indexed,
            "coverage": indexed / rows if rows else 0.0,
            "ngrams": sum(ngrams),
        }
    return stats


def _iter_text_records(
    input_paths: list[Path],
    *,
    input_format: str,
    text_field: str,
) -> Iterable[tuple[Path, int, str]]:
    for input_path in input_paths:
        if not input_path.is_file():
            raise FileNotFoundError(f"input file does not exist: {input_path}")
        if input_format == "text":
            yield input_path, 1, input_path.read_text()
            continue
        if input_format == "parquet-text":
            yield from _iter_parquet_text_records(input_path, text_field=text_field)
            continue
        for line_no, line in enumerate(input_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}:{line_no} must be a JSON object")
            value = row.get(text_field)
            if not isinstance(value, str):
                raise ValueError(
                    f"{input_path}:{line_no} field {text_field!r} must be text"
                )
            yield input_path, line_no, value


def _iter_parquet_text_records(
    input_path: Path,
    *,
    text_field: str,
) -> Iterable[tuple[Path, int, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError(
            "parquet inputs require pyarrow in the TorchTitan rootfs"
        ) from exc

    try:
        parquet_file = pq.ParquetFile(input_path)
    except Exception as exc:
        raise ValueError(f"could not read parquet input {input_path}: {exc}") from exc
    if text_field not in parquet_file.schema_arrow.names:
        raise ValueError(f"{input_path} does not contain parquet field {text_field!r}")

    row_number = 0
    for batch in parquet_file.iter_batches(columns=[text_field], batch_size=1024):
        column = batch.column(0)
        for value in column.to_pylist():
            row_number += 1
            if not isinstance(value, str):
                raise ValueError(
                    f"{input_path}:row {row_number} field {text_field!r} must be text"
                )
            yield input_path, row_number, value


def _file_reports(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
