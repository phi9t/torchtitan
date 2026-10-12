#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build local Mini Kimi K3 token shards from local files.

This is a local producer for the same raw uint32 shard shape used by the
first-party Stage 1 pipeline. It intentionally consumes already-local inputs;
remote dataset resolution, decontamination, and provenance reporting remain
separate gates before a full launch can be trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from experiments.mini_kimi_k3.register_shards import _register_shards

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
    from ngram import NGRAM_N, NgramIndex
except ImportError:  # pragma: no cover - exercised only without first-party assets.
    NGRAM_N = 13
    NgramIndex = None  # type: ignore[assignment]


KIMI_K3_VOCAB_SIZE = 163_840
KIMI_K3_EOS_ID = 163_585


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Mini Kimi K3 raw uint32 shards from local text or token JSONL."
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
        help="Token shard root directory.",
    )
    parser.add_argument("--source", required=True, help="Existing manifest source key.")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Local input file. May be passed multiple times.",
    )
    parser.add_argument(
        "--input-format",
        choices=("text", "jsonl-text", "jsonl-tokens", "parquet-text", "parquet-tokens"),
        default="text",
        help="How to read --input files.",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="JSONL field to read when --input-format=jsonl-text.",
    )
    parser.add_argument(
        "--tokens-field",
        default="tokens",
        help="JSONL field to read when --input-format=jsonl-tokens.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("experiments/mini_kimi_k3/assets/kimi-k3-tokenizer"),
        help="Kimi K3 tokenizer asset directory for text inputs.",
    )
    parser.add_argument(
        "--shard-tokens",
        type=int,
        default=100_000_000,
        help="Maximum tokens per output shard.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Stop after writing this many tokens for the source.",
    )
    parser.add_argument(
        "--decontamination-index",
        type=Path,
        help=(
            "Optional first-party NgramIndex .npz. For text inputs, matching "
            "documents are dropped before tokenization."
        ),
    )
    parser.add_argument(
        "--decontamination-min-matches",
        type=int,
        default=1,
        help="Minimum matching 13-grams required to drop a text document.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output shard prefix. Defaults to <source>-local.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        written = build_local_shards(
            manifest=manifest,
            manifest_path=args.manifest,
            tokens_dir=args.tokens_dir,
            source=args.source,
            input_paths=args.input,
            input_format=args.input_format,
            text_field=args.text_field,
            tokens_field=args.tokens_field,
            tokenizer_dir=args.tokenizer_dir,
            shard_tokens=args.shard_tokens,
            max_tokens=args.max_tokens,
            decontamination_index=args.decontamination_index,
            decontamination_min_matches=args.decontamination_min_matches,
            prefix=args.prefix,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 21

    total = sum(item["tokens"] for item in written)
    print(
        f"built {len(written)} Mini Kimi K3 local shard(s), {total} tokens",
        file=sys.stderr,
    )
    return 0


def build_local_shards(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    tokens_dir: Path,
    source: str,
    input_paths: list[Path],
    input_format: str,
    text_field: str,
    tokens_field: str,
    tokenizer_dir: Path,
    shard_tokens: int,
    prefix: str | None,
    max_tokens: int | None = None,
    decontamination_index: Path | None = None,
    decontamination_min_matches: int = 1,
) -> list[dict[str, Any]]:
    if shard_tokens <= 0:
        raise ValueError("shard_tokens must be positive")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive when provided")
    if decontamination_min_matches <= 0:
        raise ValueError("decontamination_min_matches must be positive")
    if not input_paths:
        raise ValueError("at least one input file is required")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or source not in sources:
        raise ValueError(f"manifest source {source!r} does not exist")

    source_dir = tokens_dir / source
    source_dir.mkdir(parents=True, exist_ok=True)
    shard_prefix = prefix or f"{source}-local"
    text_input_formats = {"text", "jsonl-text", "parquet-text"}
    encoder = _load_text_encoder(tokenizer_dir) if input_format in text_input_formats else None
    decontamination = _load_decontamination_index(
        decontamination_index,
        input_format=input_format,
        min_matches=decontamination_min_matches,
    )

    written: list[dict[str, Any]] = []
    buffer: list[np.ndarray] = []
    buffered_tokens = 0
    written_tokens = 0
    shard_index = 0
    stats = {
        "documents": 0,
        "input_files": [str(path) for path in input_paths],
        "input_format": input_format,
        "source": source,
    }
    if max_tokens is not None:
        stats["max_tokens"] = max_tokens
    if decontamination is not None:
        stats["decontamination"] = {
            "applied": True,
            "index": str(decontamination_index),
            "min_matches": decontamination_min_matches,
            "documents_seen": 0,
            "documents_kept": 0,
            "documents_removed": 0,
            "hits_by_benchmark": {},
        }

    def flush() -> None:
        nonlocal buffer, buffered_tokens, shard_index, written_tokens
        if not buffer:
            return
        tokens = np.concatenate(buffer).astype("<u4", copy=False)
        name = f"{shard_prefix}-s{shard_index:04d}.bin"
        shard_path = source_dir / name
        sidecar_path = shard_path.with_suffix(".json")
        if shard_path.exists() or sidecar_path.exists():
            raise ValueError(f"output shard already exists: {shard_path}")
        tokens.tofile(shard_path)
        sidecar = {
            "source": source,
            "seq": shard_index,
            "tokens": int(tokens.size),
            "dtype": "uint32",
            "input_format": input_format,
            "input_files": [str(path) for path in input_paths],
            "sha256": _sha256(shard_path),
        }
        if decontamination is not None:
            sidecar["decontamination"] = {
                "applied": True,
                "index": str(decontamination_index),
                "min_matches": decontamination_min_matches,
            }
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        written.append({"path": shard_path, "tokens": int(tokens.size)})
        written_tokens += int(tokens.size)
        shard_index += 1
        buffer, buffered_tokens = [], 0

    for token_ids in _iter_token_sequences(
        input_paths=input_paths,
        input_format=input_format,
        text_field=text_field,
        tokens_field=tokens_field,
        encoder=encoder,
        decontamination=decontamination,
        decontamination_stats=stats.get("decontamination"),
    ):
        arr = _validate_tokens(token_ids)
        if arr.size == 0:
            continue
        if max_tokens is not None:
            remaining = max_tokens - written_tokens - buffered_tokens
            if remaining <= 0:
                break
            if arr.size > remaining:
                arr = arr[:remaining]
        offset = 0
        while offset < arr.size:
            space = shard_tokens - buffered_tokens
            take = min(space, arr.size - offset)
            buffer.append(arr[offset : offset + take])
            buffered_tokens += take
            offset += take
            if buffered_tokens == shard_tokens:
                flush()
        stats["documents"] += 1
        if max_tokens is not None and written_tokens + buffered_tokens >= max_tokens:
            break
    flush()

    if not written:
        raise ValueError("local shard build produced no tokens")

    _register_shards(
        manifest=manifest,
        tokens_dir=tokens_dir,
        source=source,
        shards=[item["path"] for item in written],
    )
    source_data = manifest["sources"][source]
    local_builds = source_data.setdefault("local_builds", [])
    if not isinstance(local_builds, list):
        raise ValueError(f"manifest source {source!r} local_builds must be a list")
    local_builds.append(
        {
            **stats,
            "shard_tokens": shard_tokens,
            "shards": [
                {"name": item["path"].name, "tokens": item["tokens"]}
                for item in written
            ],
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return written


def _iter_token_sequences(
    *,
    input_paths: list[Path],
    input_format: str,
    text_field: str,
    tokens_field: str,
    encoder: Callable[[str], list[int]] | None,
    decontamination: Any | None,
    decontamination_stats: Any | None,
) -> Iterable[list[int]]:
    for input_path in input_paths:
        if not input_path.is_file():
            raise FileNotFoundError(f"input file does not exist: {input_path}")
        if input_format == "text":
            if encoder is None:
                raise ValueError("text inputs require a tokenizer encoder")
            text = input_path.read_text()
            if _is_decontaminated_text(
                text,
                decontamination=decontamination,
                decontamination_stats=decontamination_stats,
            ):
                yield encoder(text)
            continue
        if input_format in {"parquet-text", "parquet-tokens"}:
            yield from _iter_parquet_token_sequences(
                input_path=input_path,
                input_format=input_format,
                text_field=text_field,
                tokens_field=tokens_field,
                encoder=encoder,
                decontamination=decontamination,
                decontamination_stats=decontamination_stats,
            )
            continue
        for line_no, line in enumerate(input_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}:{line_no} must be a JSON object")
            if input_format == "jsonl-tokens":
                value = row.get(tokens_field)
                if not isinstance(value, list):
                    raise ValueError(
                        f"{input_path}:{line_no} field {tokens_field!r} must be a list"
                    )
                yield value
            else:
                if encoder is None:
                    raise ValueError("jsonl-text inputs require a tokenizer encoder")
                value = row.get(text_field)
                if not isinstance(value, str):
                    raise ValueError(
                        f"{input_path}:{line_no} field {text_field!r} must be text"
                    )
                if _is_decontaminated_text(
                    value,
                    decontamination=decontamination,
                    decontamination_stats=decontamination_stats,
                ):
                    yield encoder(value)


def _iter_parquet_token_sequences(
    *,
    input_path: Path,
    input_format: str,
    text_field: str,
    tokens_field: str,
    encoder: Callable[[str], list[int]] | None,
    decontamination: Any | None,
    decontamination_stats: Any | None,
) -> Iterable[list[int]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError(
            "parquet inputs require pyarrow in the TorchTitan rootfs"
        ) from exc

    field = tokens_field if input_format == "parquet-tokens" else text_field
    try:
        parquet_file = pq.ParquetFile(input_path)
    except Exception as exc:
        raise ValueError(f"could not read parquet input {input_path}: {exc}") from exc

    if field not in parquet_file.schema_arrow.names:
        raise ValueError(f"{input_path} does not contain parquet field {field!r}")
    for batch_index, batch in enumerate(
        parquet_file.iter_batches(columns=[field], batch_size=1024),
        start=1,
    ):
        column = batch.column(0)
        for row_index, value in enumerate(column.to_pylist(), start=1):
            location = f"{input_path}:batch {batch_index} row {row_index}"
            if input_format == "parquet-tokens":
                if not isinstance(value, list):
                    raise ValueError(f"{location} field {field!r} must be a list")
                yield value
                continue
            if encoder is None:
                raise ValueError("parquet-text inputs require a tokenizer encoder")
            if not isinstance(value, str):
                raise ValueError(f"{location} field {field!r} must be text")
            if _is_decontaminated_text(
                value,
                decontamination=decontamination,
                decontamination_stats=decontamination_stats,
            ):
                yield encoder(value)


def _load_decontamination_index(
    path: Path | None,
    *,
    input_format: str,
    min_matches: int,
) -> Any | None:
    if path is None:
        return None
    if input_format not in {"text", "jsonl-text", "parquet-text"}:
        raise ValueError("decontamination index can only be used with text inputs")
    if NgramIndex is None:
        raise ValueError("decontamination index requires first-party ngram.py asset")
    if not path.is_file():
        raise FileNotFoundError(f"decontamination index does not exist: {path}")
    return NgramIndex.load(path)


def _is_decontaminated_text(
    text: str,
    *,
    decontamination: Any | None,
    decontamination_stats: Any | None,
) -> bool:
    if decontamination is None:
        return True
    assert isinstance(decontamination_stats, dict)
    decontamination_stats["documents_seen"] += 1
    contaminated, hits = decontamination.check(
        text,
        n=NGRAM_N,
        min_matches=decontamination_stats["min_matches"],
    )
    if contaminated:
        decontamination_stats["documents_removed"] += 1
        hit_counts = decontamination_stats["hits_by_benchmark"]
        for benchmark, count in hits.items():
            hit_counts[benchmark] = hit_counts.get(benchmark, 0) + count
        return False
    decontamination_stats["documents_kept"] += 1
    return True


def _validate_tokens(token_ids: list[int]) -> np.ndarray:
    if not all(isinstance(token_id, int) for token_id in token_ids):
        raise ValueError("token sequences must contain only integers")
    arr = np.asarray(token_ids, dtype=np.int64)
    if arr.size == 0:
        return np.asarray([], dtype="<u4")
    if int(arr.min()) < 0 or int(arr.max()) >= KIMI_K3_VOCAB_SIZE:
        raise ValueError(
            f"token id outside Kimi K3 vocabulary [0, {KIMI_K3_VOCAB_SIZE})"
        )
    return arr.astype("<u4")

def _load_text_encoder(tokenizer_dir: Path) -> Callable[[str], list[int]]:
    model_path = tokenizer_dir / "tiktoken.model"
    config_path = tokenizer_dir / "tokenizer_config.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Kimi K3 tokenizer model not found: {model_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Kimi K3 tokenizer config not found: {config_path}")
    try:
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
    except ImportError as exc:
        raise ValueError("text inputs require tiktoken in the TorchTitan rootfs") from exc

    mergeable_ranks = load_tiktoken_bpe(str(model_path))
    config = json.loads(config_path.read_text())
    named = {
        int(key): value["content"]
        for key, value in config.get("added_tokens_decoder", {}).items()
    }
    special_tokens = {
        named.get(idx, f"<|reserved_token_{idx}|>"): idx
        for idx in range(len(mergeable_ranks), KIMI_K3_VOCAB_SIZE)
    }
    encoding = tiktoken.Encoding(
        name="kimi-k3-local",
        pat_str=_kimi_k3_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    if encoding.n_vocab != KIMI_K3_VOCAB_SIZE:
        raise ValueError(
            f"expected Kimi K3 vocab {KIMI_K3_VOCAB_SIZE}, got {encoding.n_vocab}"
        )

    def encode(text: str) -> list[int]:
        return [*encoding.encode_ordinary(text), KIMI_K3_EOS_ID]

    return encode


def _kimi_k3_pattern() -> str:
    return "|".join(
        [
            r"""[\p{Han}]+""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
