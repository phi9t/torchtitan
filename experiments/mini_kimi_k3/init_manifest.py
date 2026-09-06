#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Initialize a full Mini Kimi K3 token-manifest template."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize a Mini Kimi K3 full corpus manifest template."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/mini_kimi_k3/data/manifest.json"),
        help="Manifest JSON path to create.",
    )
    parser.add_argument(
        "--tokens-dir",
        type=Path,
        default=Path("experiments/mini_kimi_k3/data/tokens"),
        help="Token shard directory to create.",
    )
    parser.add_argument("--source", required=True, help="Manifest source key.")
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Stable corpus dataset identifier for the source.",
    )
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Corpus snapshot, commit, release, or date for provenance.",
    )
    parser.add_argument(
        "--license",
        required=True,
        help="Source license or policy label.",
    )
    parser.add_argument(
        "--source-provenance-report",
        help="Path or URI for the source provenance report.",
    )
    parser.add_argument(
        "--source-provenance-sha256",
        help="SHA-256 or equivalent fingerprint of the source provenance report.",
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
        required=True,
        help="Path or URI for the decontamination report.",
    )
    parser.add_argument(
        "--decontamination-sha256",
        required=True,
        help="SHA-256 or equivalent fingerprint of the decontamination report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.manifest.exists():
        print(f"Mini Kimi K3 manifest already exists: {args.manifest}", file=sys.stderr)
        return 21

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.tokens_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "kind": "mini_kimi_k3_token_manifest",
        "mode": "full",
        "tokenizer": {
            "name": "Kimi K3",
            "vocab_size": 163_840,
            "sha256": args.tokenizer_sha256,
        },
        "decontamination": {
            "status": "completed",
            "report": args.decontamination_report,
            "sha256": args.decontamination_sha256,
        },
        "sources": {
            args.source: {
                "dataset_id": args.dataset_id,
                "provenance": {
                    "snapshot": args.snapshot,
                    "license": args.license,
                },
                "shards": [],
            }
        },
    }
    if args.tokenizer_asset_report:
        manifest["tokenizer"]["asset_report"] = args.tokenizer_asset_report
    if args.source_provenance_report:
        manifest["sources"][args.source]["provenance"][
            "report"
        ] = args.source_provenance_report
    if args.source_provenance_sha256:
        manifest["sources"][args.source]["provenance"][
            "sha256"
        ] = args.source_provenance_sha256
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"initialized Mini Kimi K3 manifest template: {args.manifest}", file=sys.stderr
    )
    print(f"token shards directory: {args.tokens_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
