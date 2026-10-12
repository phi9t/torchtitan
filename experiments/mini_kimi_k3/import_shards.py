#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Import first-party Stage 1 Mini Kimi K3 token shards into a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import first-party Stage 1 raw uint32 shards into the Mini Kimi K3 "
            "manifest-owned token tree."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/mini_kimi_k3/data/manifest.json"),
        help="Manifest JSON path to update.",
    )
    parser.add_argument(
        "--tokens-dir",
        type=Path,
        default=Path("experiments/mini_kimi_k3/data/tokens"),
        help="Manifest-owned token shard root.",
    )
    parser.add_argument("--source", required=True, help="Existing manifest source key.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing first-party .bin shards and .json sidecars.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Materialization mode under <tokens-dir>/<source>.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        imported = import_shards(
            manifest=manifest,
            manifest_path=args.manifest,
            tokens_dir=args.tokens_dir,
            source=args.source,
            source_dir=args.source_dir,
            link_mode=args.link_mode,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 21

    print(
        f"imported {len(imported)} Mini Kimi K3 shard(s) from {args.source_dir}",
        file=sys.stderr,
    )
    return 0


def import_shards(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    tokens_dir: Path,
    source: str,
    source_dir: Path,
    link_mode: str,
) -> list[str]:
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("manifest must define a sources object")
    source_data = sources.get(source)
    if not isinstance(source_data, dict):
        raise ValueError(f"manifest source {source!r} does not exist")

    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source shard directory does not exist: {source_dir}")
    destination_dir = tokens_dir / source

    shards = sorted(source_dir.glob("*.bin"))
    if not shards:
        raise ValueError(f"source shard directory has no .bin shards: {source_dir}")

    validated = [
        _validate_stage1_shard(path, expected_source=source) for path in shards
    ]
    destination_dir.mkdir(parents=True, exist_ok=True)
    for item in validated:
        _materialize_shard(
            source_path=item["source_path"],
            destination_path=destination_dir / item["name"],
            link_mode=link_mode,
        )

    shard_names = source_data.setdefault("shards", [])
    if not isinstance(shard_names, list) or not all(
        isinstance(name, str) for name in shard_names
    ):
        raise ValueError(f"manifest source {source!r} shards must be a string list")
    shard_metadata = source_data.setdefault("shard_metadata", {})
    if not isinstance(shard_metadata, dict):
        raise ValueError(f"manifest source {source!r} shard_metadata must be an object")

    imported: list[str] = []
    for item in validated:
        name = item["name"]
        if name not in shard_names:
            shard_names.append(name)
        shard_metadata[name] = {
            "bytes": item["bytes"],
            "num_tokens": item["num_tokens"],
            "sha256": item["sha256"],
            "source_sidecar": str(item["sidecar_path"]),
            "source_sidecar_sha256": item["sidecar_sha256"],
        }
        imported.append(name)

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return imported


def _validate_stage1_shard(path: Path, *, expected_source: str) -> dict[str, Any]:
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar_source = sidecar.get("source")
    if sidecar_source != expected_source:
        raise ValueError(
            f"{path.name} sidecar source mismatch ({sidecar_source!r} != "
            f"{expected_source!r})"
        )
    if sidecar.get("dtype") != "uint32":
        raise ValueError(f"{path.name} sidecar dtype must be uint32")
    size = path.stat().st_size
    if size % 4 != 0:
        raise ValueError(f"{path.name} size is not divisible by uint32 width")
    num_tokens = size // 4
    sidecar_tokens = sidecar.get("tokens")
    if sidecar_tokens != num_tokens:
        raise ValueError(
            f"{path.name} sidecar token count mismatch "
            f"({sidecar_tokens} != {num_tokens})"
        )
    return {
        "name": path.name,
        "source_path": path,
        "sidecar_path": sidecar_path,
        "sidecar_sha256": _sha256(sidecar_path),
        "bytes": size,
        "num_tokens": num_tokens,
        "sha256": _sha256(path),
    }


def _materialize_shard(
    *,
    source_path: Path,
    destination_path: Path,
    link_mode: str,
) -> None:
    if destination_path.exists() or destination_path.is_symlink():
        if destination_path.resolve() == source_path:
            return
        raise FileExistsError(f"destination shard already exists: {destination_path}")
    if link_mode == "symlink":
        os.symlink(source_path, destination_path)
        return
    shutil.copy2(source_path, destination_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
