#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Fetch or verify the pinned modded-nanogpt source checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.modded_nanogpt_b200 import cli_guard


UPSTREAM_REPOSITORY = "https://github.com/kellerjordan/modded-nanogpt"
UPSTREAM_COMMIT = "ecbb586296d3dac36fd206211f25d63bad4a6b35"


def _run_git(args: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _ensure_source(source: Path, repository: str) -> None:
    if source.exists():
        return
    source.parent.mkdir(parents=True, exist_ok=True)
    _run_git(["clone", repository, str(source)])


def _has_origin(source: Path) -> bool:
    remotes = _run_git(["remote"], cwd=source).splitlines()
    return "origin" in remotes


def prepare_source(
    *,
    source: Path,
    result_dir: Path | None,
    pinned_commit: str = UPSTREAM_COMMIT,
    repository: str = UPSTREAM_REPOSITORY,
) -> dict[str, Any]:
    _ensure_source(source, repository)
    status = _run_git(["status", "--short"], cwd=source)
    if status.strip():
        raise ValueError(f"refusing dirty source at {source}:\n{status}")
    if _has_origin(source):
        _run_git(["fetch", "--tags", "origin"], cwd=source)
    _run_git(["checkout", pinned_commit], cwd=source)
    commit = _run_git(["rev-parse", "HEAD"], cwd=source).strip()
    if commit != pinned_commit:
        raise ValueError(f"expected source commit {pinned_commit}, found {commit}")
    status = _run_git(["status", "--short"], cwd=source)
    if status.strip():
        raise ValueError(f"refusing dirty source at {source}:\n{status}")
    record = {
        "path": str(source),
        "repository": repository,
        "commit": commit,
        "status": status,
    }
    if result_dir is not None:
        _write_json_atomic(result_dir / "source.json", record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("experiments/modded_nanogpt_b200/sources/modded-nanogpt"),
    )
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--repository", default=UPSTREAM_REPOSITORY)
    parser.add_argument("--pinned-commit", default=UPSTREAM_COMMIT)
    return parser.parse_args()


def main(*, enforce_rootfs: bool = False) -> int:
    args = parse_args()
    if enforce_rootfs:
        guard_exit = cli_guard.guard_rootfs_cli("experiments/modded_nanogpt_b200/fetch_upstream.sh")
        if guard_exit is not None:
            return guard_exit
    try:
        record = prepare_source(
            source=args.source,
            result_dir=args.result_dir,
            pinned_commit=args.pinned_commit,
            repository=args.repository,
        )
    except (subprocess.CalledProcessError, ValueError) as exc:
        print(f"error: {exc}")
        return 21
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(enforce_rootfs=True))
