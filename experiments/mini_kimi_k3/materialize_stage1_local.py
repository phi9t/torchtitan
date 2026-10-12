#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Materialize resolved Mini Kimi K3 Stage 1 source files locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


FileLister = Callable[[str, str | None], list[str]]
Downloader = Callable[[str, str, Path], Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download resolved public Mini Kimi K3 Stage 1 parquet inputs into "
            "a rootfs-visible local directory and write source input manifests."
        )
    )
    parser.add_argument(
        "--source-resolution-report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/reports/stage1-source-resolution.json"),
        help="Report written by probe-stage1-sources.",
    )
    parser.add_argument(
        "--corpus-plan-report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/reports/r1-corpus-plan.json"),
        help="Report written by plan-r1-corpus.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/mini_kimi_k3/data/stage1-local-inputs"),
        help="Directory for downloaded inputs and generated input manifests.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/reports/stage1-local-materialize.json"),
        help="Machine-readable materialization report.",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Restrict materialization to one source. May be passed multiple times.",
    )
    parser.add_argument(
        "--max-files-per-source",
        type=int,
        default=None,
        help=(
            "Optional cap for fixture or pilot materialization. The effective "
            "file count is min(this value, the corpus plan estimated_files)."
        ),
    )
    parser.add_argument(
        "--all-available-files",
        action="store_true",
        help=(
            "Download every matching resolved parquet file instead of the r1 "
            "corpus plan estimated_files budget."
        ),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Required before any Hugging Face dataset file is downloaded.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = materialize_stage1_local(
            source_resolution_report=args.source_resolution_report,
            corpus_plan_report=args.corpus_plan_report,
            output_root=args.output_root,
            report_path=args.report,
            allow_download=args.allow_download,
            max_files_per_source=args.max_files_per_source,
            all_available_files=args.all_available_files,
            selected_sources=args.source,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 21

    print(
        "Mini Kimi K3 local Stage 1 materialization: "
        f"{report['status']} ({len(report['sources'])} source(s))",
        file=sys.stderr,
    )
    if report["status"] != "ready":
        print(report["detail"], file=sys.stderr)
    print(f"wrote local Stage 1 materialization report: {args.report}", file=sys.stderr)
    return 0 if report["status"] == "ready" else 21


def materialize_stage1_local(
    *,
    source_resolution_report: Path,
    corpus_plan_report: Path,
    output_root: Path,
    report_path: Path,
    allow_download: bool,
    max_files_per_source: int | None,
    all_available_files: bool = False,
    selected_sources: list[str] | None = None,
    file_lister: FileLister | None = None,
    downloader: Downloader | None = None,
) -> dict[str, Any]:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise ValueError("Mini-K3 local Stage 1 materialization must run inside rootfs")
    if max_files_per_source is not None and max_files_per_source <= 0:
        raise ValueError("max_files_per_source must be positive when set")
    if all_available_files and max_files_per_source is not None:
        raise ValueError(
            "--all-available-files cannot be combined with --max-files-per-source"
        )

    if not allow_download:
        return _write_report(
            report_path,
            output_root=output_root,
            status="blocked",
            detail="materialize-stage1-local requires --allow-download",
            sources={},
        )

    resolution = _read_ready_resolution(source_resolution_report)
    plan = _read_planned_corpus_plan(corpus_plan_report)
    wanted = set(selected_sources or plan["sources"])
    unknown = sorted(wanted - set(plan["sources"]))
    if unknown:
        raise ValueError(f"source(s) are not in corpus plan: {', '.join(unknown)}")

    file_lister = file_lister or _list_hf_parquet_files
    downloader = downloader or _download_hf_file
    output_root.mkdir(parents=True, exist_ok=True)

    source_reports: dict[str, Any] = {}
    blocked = False
    details: list[str] = []
    for source in sorted(wanted):
        resolved = resolution["sources"].get(source)
        if not isinstance(resolved, dict) or resolved.get("status") != "resolved":
            raise ValueError(f"source {source!r} is not resolved")
        repo = _nonempty_text(resolved, "repo", source)
        path_prefix = resolved.get("path_prefix")
        if path_prefix is not None and not isinstance(path_prefix, str):
            raise ValueError(f"source {source!r} path_prefix must be text or null")
        files = file_lister(repo, path_prefix)
        files = [name for name in files if name.endswith(".parquet")]
        if path_prefix:
            files = [name for name in files if name.startswith(path_prefix)]
        files = sorted(files)
        if not files:
            raise ValueError(f"source {source!r} has no parquet files")
        plan_file_budget = _estimated_files(plan, source)
        file_budget = len(files) if all_available_files else plan_file_budget
        if max_files_per_source is not None:
            file_budget = min(file_budget, max_files_per_source)
        if len(files) < file_budget:
            blocked = True
            details.append(
                f"{source} requires {file_budget} parquet file(s), found {len(files)}"
            )
            source_reports[source] = {
                "repo": repo,
                "config": resolved.get("config"),
                "split": resolved.get("split"),
                "text_field": _text_field(resolved),
                "path_prefix": path_prefix,
                "target_tokens": _target_tokens(plan, source),
                "file_budget": file_budget,
                "files_available": len(files),
                "files_selected": 0,
                "status": "blocked",
            }
            continue
        selected = files[:file_budget]
        source_dir = output_root / "files" / source
        local_files = [
            downloader(repo, remote_path, source_dir / Path(remote_path).name).resolve()
            for remote_path in selected
        ]
        input_manifest = output_root / f"{source}-inputs.json"
        _write_input_manifest(
            output=input_manifest,
            input_paths=local_files,
            text_field=_text_field(resolved),
        )
        source_reports[source] = {
            "repo": repo,
            "config": resolved.get("config"),
            "split": resolved.get("split"),
            "text_field": _text_field(resolved),
            "path_prefix": path_prefix,
            "target_tokens": _target_tokens(plan, source),
            "file_budget": file_budget,
            "files_available": len(files),
            "files_selected": len(selected),
            "status": "ready",
            "input_manifest": str(input_manifest),
            "inputs": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in local_files
            ],
        }

    if blocked:
        return _write_report(
            report_path,
            output_root=output_root,
            status="blocked",
            detail="; ".join(details),
            sources=source_reports,
        )

    return _write_report(
        report_path,
        output_root=output_root,
        status="ready",
        detail="local Stage 1 input manifests are ready",
        sources=source_reports,
    )


def _read_ready_resolution(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("kind") != "mini_kimi_k3_stage1_source_resolution":
        raise ValueError(
            "source resolution kind must be mini_kimi_k3_stage1_source_resolution"
        )
    if data.get("schema_version") != 1:
        raise ValueError("source resolution schema_version must be 1")
    if data.get("status") != "ready":
        raise ValueError("source resolution report must be ready")
    sources = data.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("source resolution report must define sources")
    return data


def _read_planned_corpus_plan(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("kind") != "mini_kimi_k3_r1_corpus_plan":
        raise ValueError("corpus plan kind must be mini_kimi_k3_r1_corpus_plan")
    if data.get("schema_version") != 1:
        raise ValueError("corpus plan schema_version must be 1")
    if data.get("status") != "planned":
        raise ValueError("corpus plan status must be planned")
    sources = data.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("corpus plan must define sources")
    return data


def _target_tokens(plan: dict[str, Any], source: str) -> int:
    source_plan = plan["sources"].get(source)
    if not isinstance(source_plan, dict):
        raise ValueError(f"source {source!r} missing from corpus plan")
    value = source_plan.get("target_tokens")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"corpus plan source {source!r} target_tokens must be positive"
        )
    return value


def _estimated_files(plan: dict[str, Any], source: str) -> int:
    source_plan = plan["sources"].get(source)
    if not isinstance(source_plan, dict):
        raise ValueError(f"source {source!r} missing from corpus plan")
    value = source_plan.get("estimated_files")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"corpus plan source {source!r} estimated_files must be positive"
        )
    return value


def _text_field(resolved: dict[str, Any]) -> str:
    value = resolved.get("text_field") or "text"
    if not isinstance(value, str) or not value:
        raise ValueError("resolved source text_field must be nonempty text")
    return value


def _nonempty_text(data: dict[str, Any], key: str, source: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"source {source!r} {key} must be nonempty text")
    return value


def _write_input_manifest(
    *,
    output: Path,
    input_paths: list[Path],
    text_field: str,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"input manifest already exists: {output}")
    inputs = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in input_paths
    ]
    manifest = {
        "schema_version": 1,
        "kind": "mini_kimi_k3_source_input_manifest",
        "input_format": "parquet-text",
        "text_field": text_field,
        "inputs": inputs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _list_hf_parquet_files(repo: str, path_prefix: str | None) -> list[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ValueError(
            "materialize-stage1-local requires huggingface_hub in the rootfs"
        ) from exc
    files = [
        path
        for path in HfApi().list_repo_files(repo, repo_type="dataset")
        if path.endswith(".parquet")
    ]
    if path_prefix:
        files = [path for path in files if path.startswith(path_prefix)]
    return files


def _download_hf_file(repo: str, remote_path: str, output_path: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ValueError(
            "materialize-stage1-local requires huggingface_hub in the rootfs"
        ) from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise ValueError(f"download destination already exists: {output_path}")
    downloaded = Path(
        hf_hub_download(
            repo_id=repo,
            filename=remote_path,
            repo_type="dataset",
            local_dir=output_path.parent,
        )
    )
    if downloaded.resolve() != output_path.resolve() and not output_path.exists():
        output_path.symlink_to(downloaded)
    return output_path if output_path.exists() else downloaded


def _write_report(
    report_path: Path,
    *,
    output_root: Path,
    status: str,
    detail: str,
    sources: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_stage1_local_materialization",
        "status": status,
        "detail": detail,
        "output_root": str(output_root),
        "sources": sources,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
