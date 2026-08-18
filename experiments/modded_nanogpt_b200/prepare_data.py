#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Prepare or manifest FineWeb shards for the modded-nanogpt B200 harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.modded_nanogpt_b200 import cli_guard


SCHEMA_VERSION = 1
UPSTREAM_COMMIT = "ecbb586296d3dac36fd206211f25d63bad4a6b35"
EXPECTED_FINEWEB_BYTES = 2_000_010_240
EXPECTED_FINEWEB_SHARDS = 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _environment_snapshot() -> dict[str, Any]:
    env = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_runtime"] = torch.version.cuda
    except Exception as exc:  # noqa: BLE001
        env["torch_error"] = f"{type(exc).__name__}: {exc}"
    return env


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _bin_shards(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.bin"))


def build_manifest(
    *,
    data_dir: Path,
    source_path: Path,
    source_commit: str,
    command: list[str],
    token_budget: str,
    freshness: str,
    environment: dict[str, Any],
) -> dict[str, Any]:
    shards = _bin_shards(data_dir)
    if not shards:
        raise ValueError(f"no .bin shards found under {data_dir}")
    files = [
        {
            "path": str(shard),
            "bytes": shard.stat().st_size,
            "sha256": sha256_file(shard),
        }
        for shard in shards
    ]
    total_bytes = sum(item["bytes"] for item in files)
    if token_budget == "900M":
        if len(files) != EXPECTED_FINEWEB_SHARDS:
            raise ValueError(
                f"full manifest requires exactly {EXPECTED_FINEWEB_SHARDS} .bin shards, found {len(files)}"
            )
        if total_bytes != EXPECTED_FINEWEB_BYTES:
            raise ValueError(
                f"full manifest requires total_bytes {EXPECTED_FINEWEB_BYTES}, found {total_bytes}"
            )
        if source_commit != UPSTREAM_COMMIT:
            raise ValueError(f"full manifest source commit must be {UPSTREAM_COMMIT}, found {source_commit}")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "fineweb10B",
        "token_budget": token_budget,
        "source": {
            "path": str(source_path),
            "commit": source_commit,
        },
        "command": command,
        "freshness": freshness,
        "environment": environment,
        "files": files,
        "num_files": len(files),
        "total_bytes": total_bytes,
        "verified_sha": True,
    }


def write_manifest(
    *,
    output: Path,
    data_dir: Path,
    source_path: Path,
    source_commit: str,
    command: list[str],
    token_budget: str,
    freshness: str,
    environment: dict[str, Any],
) -> dict[str, Any]:
    manifest = build_manifest(
        data_dir=data_dir,
        source_path=source_path,
        source_commit=source_commit,
        command=command,
        token_budget=token_budget,
        freshness=freshness,
        environment=environment,
    )
    _write_json_atomic(output, manifest)
    return manifest


def _source_commit(source: Path) -> str:
    return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", choices=["900M", "smoke"], default="900M")
    parser.add_argument("--freshness", choices=["fresh", "reused"], default="reused")
    parser.add_argument("--skip-upstream-command", action="store_true")
    return parser.parse_args()


def main(*, enforce_rootfs: bool = False) -> int:
    args = parse_args()
    if enforce_rootfs:
        guard_exit = cli_guard.guard_rootfs_cli("experiments/modded_nanogpt_b200/prepare_data.sh")
        if guard_exit is not None:
            return guard_exit
    count_arg = "9" if args.token_budget == "900M" else "1"
    command = ["python", "data/cached_fineweb10B.py", count_arg]
    try:
        if not args.skip_upstream_command:
            subprocess.run(command, cwd=args.source, check=True)
        manifest = write_manifest(
            output=args.output,
            data_dir=args.data_dir,
            source_path=args.source,
            source_commit=_source_commit(args.source),
            command=command,
            token_budget=args.token_budget,
            freshness=args.freshness,
            environment=_environment_snapshot(),
        )
    except (subprocess.CalledProcessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 21
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(enforce_rootfs=True))
