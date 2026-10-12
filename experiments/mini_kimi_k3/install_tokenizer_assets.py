#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Install Mini Kimi K3 tokenizer assets into the experiment asset tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


REQUIRED_TOKENIZER_FILES = (
    "config.json",
    "tokenizer_config.json",
    "generation_config.json",
    "tokenization_kimi.py",
    "encoding_k3.py",
    "tiktoken.model",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy and fingerprint Mini Kimi K3 tokenizer assets."
    )
    parser.add_argument(
        "--source-tokenizer-dir",
        type=Path,
        required=True,
        help="Directory containing the first-party tokenizer files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/mini_kimi_k3/assets/kimi-k3-tokenizer"),
        help="Rootfs-visible tokenizer asset directory to create.",
    )
    parser.add_argument(
        "--fingerprint-report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/assets/kimi-k3-tokenizer.json"),
        help="JSON report containing file hashes and aggregate fingerprint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing tokenizer asset directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        install_tokenizer_assets(
            source_tokenizer_dir=args.source_tokenizer_dir,
            output_dir=args.output_dir,
            fingerprint_report=args.fingerprint_report,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 21
    print(
        f"installed Mini Kimi K3 tokenizer assets: {args.output_dir}", file=sys.stderr
    )
    print(
        f"wrote tokenizer fingerprint report: {args.fingerprint_report}",
        file=sys.stderr,
    )
    return 0


def install_tokenizer_assets(
    *,
    source_tokenizer_dir: Path,
    output_dir: Path,
    fingerprint_report: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    if not source_tokenizer_dir.is_dir():
        raise FileNotFoundError(
            f"source tokenizer directory not found: {source_tokenizer_dir}"
        )

    missing = [
        name
        for name in REQUIRED_TOKENIZER_FILES
        if not (source_tokenizer_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "source tokenizer directory is missing required files: "
            + ", ".join(sorted(missing))
        )

    if output_dir.exists():
        if not overwrite:
            raise ValueError(f"tokenizer output directory already exists: {output_dir}")
        if not output_dir.is_dir():
            raise ValueError(f"tokenizer output path is not a directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    fingerprint_report.parent.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_TOKENIZER_FILES:
        shutil.copy2(source_tokenizer_dir / name, output_dir / name)

    report = _fingerprint_tokenizer_dir(output_dir)
    report["source_tokenizer_dir"] = str(source_tokenizer_dir)
    report["output_dir"] = str(output_dir)
    fingerprint_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _fingerprint_tokenizer_dir(path: Path) -> dict[str, object]:
    aggregate = hashlib.sha256()
    files: dict[str, dict[str, object]] = {}
    for name in sorted(REQUIRED_TOKENIZER_FILES):
        payload = (path / name).read_bytes()
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(payload)
        files[name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return {
        "schema_version": 1,
        "kind": "mini_kimi_k3_tokenizer_assets",
        "sha256": aggregate.hexdigest(),
        "files": files,
    }


if __name__ == "__main__":
    raise SystemExit(main())
