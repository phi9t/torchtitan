#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Register Mini Kimi K3 token shards in a corpus manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register raw uint32 token shards in a Mini Kimi K3 manifest."
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
        help="Token shard root directory.",
    )
    parser.add_argument("--source", required=True, help="Existing manifest source key.")
    parser.add_argument(
        "--shard",
        type=Path,
        action="append",
        required=True,
        help="Shard file under <tokens-dir>/<source>. May be passed multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        _register_shards(
            manifest=manifest,
            tokens_dir=args.tokens_dir,
            source=args.source,
            shards=args.shard,
        )
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 21

    print(
        f"registered {len(args.shard)} Mini Kimi K3 shard(s) in {args.manifest}",
        file=sys.stderr,
    )
    return 0


def _register_shards(
    *,
    manifest: dict[str, Any],
    tokens_dir: Path,
    source: str,
    shards: list[Path],
) -> None:
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("manifest must define a sources object")
    source_data = sources.get(source)
    if not isinstance(source_data, dict):
        raise ValueError(f"manifest source {source!r} does not exist")

    source_dir = (tokens_dir / source).resolve()
    shard_names = source_data.setdefault("shards", [])
    if not isinstance(shard_names, list) or not all(
        isinstance(name, str) for name in shard_names
    ):
        raise ValueError(f"manifest source {source!r} shards must be a string list")
    shard_metadata = source_data.setdefault("shard_metadata", {})
    if not isinstance(shard_metadata, dict):
        raise ValueError(f"manifest source {source!r} shard_metadata must be an object")

    for shard in shards:
        shard_path = shard.resolve()
        if not shard_path.is_file():
            raise FileNotFoundError(f"shard does not exist: {shard}")
        if not shard_path.is_relative_to(source_dir):
            raise ValueError(f"shard {shard} must live under {source_dir}")
        rel = shard_path.relative_to(source_dir).as_posix()
        size = shard_path.stat().st_size
        if size % 4 != 0:
            raise ValueError(f"shard {shard} size is not divisible by uint32 width")
        if rel not in shard_names:
            shard_names.append(rel)
        shard_metadata[rel] = {
            "bytes": size,
            "num_tokens": size // 4,
            "sha256": _sha256(shard_path),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
