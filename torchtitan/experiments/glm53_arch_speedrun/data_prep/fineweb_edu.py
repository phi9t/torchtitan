# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prepare FineWeb-Edu sample-10BT GPT-2 nanogpt-bin shards.

The generated shard format matches the modded-nanogpt/S01 contract:
256 int32 header values, magic 20240520, version 1, token count in header[2],
then a flat uint16 GPT-2 token stream. Each document is prefixed with GPT-2 EOT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm

from torchtitan.experiments.falcon.bin_reader import read_nanogpt_bin

DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DEFAULT_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
DEFAULT_CACHE_ROOT = Path("/data05/philip.yang/glm53-ladder-cache")
SAMPLE_RELPATH = Path("sample/10BT")
NANOGPT_MAGIC = 20240520
NANOGPT_VERSION = 1
NANOGPT_HEADER_INTS = 256
DEFAULT_SHARD_SIZE = 100_000_000
EOT_TOKEN_ID = 50256

_ENCODER: tiktoken.Encoding | None = None


@dataclass(frozen=True, slots=True)
class HfSampleFile:
    file: str
    size: int


@dataclass(frozen=True, slots=True)
class ShardRecord:
    file: str
    size: int
    md5: str
    num_tokens: int
    split: str
    shard_index: int


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def short_revision(revision: str) -> str:
    return revision[:7]


def default_raw_dir(revision: str) -> Path:
    return DEFAULT_CACHE_ROOT / "datasets-raw" / "fineweb-edu" / revision


def default_output_dir(revision: str) -> Path:
    return (
        DEFAULT_CACHE_ROOT
        / "datasets-tokenized"
        / "fineweb-edu-10BT-gpt2"
        / f"{short_revision(revision)}-v1"
    )


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def md5_file(path: Path) -> str:
    hasher = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_rootfs() -> None:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise RuntimeError(
            "FineWeb-Edu data prep must run inside scripts/rootfs/enter_rootfs.sh"
        )


def dataset_info_for_revision(revision: str) -> tuple[str, list[HfSampleFile]]:
    api = HfApi()
    info = api.dataset_info(DATASET_REPO, revision=revision, files_metadata=True)
    hf_revision = str(info.sha)
    files: list[HfSampleFile] = []
    for sibling in info.siblings or []:
        filename = sibling.rfilename
        if filename.startswith(f"{SAMPLE_RELPATH.as_posix()}/") and filename.endswith(
            ".parquet"
        ):
            size = sibling.size
            if size is None:
                raise ValueError(f"missing size metadata for {filename}")
            files.append(HfSampleFile(file=filename, size=int(size)))
    files.sort(key=lambda item: item.file)
    if not files:
        raise ValueError(
            f"no {DATASET_CONFIG} parquet files found in {DATASET_REPO}@{revision}"
        )
    return hf_revision, files


def validate_local_sample_files(
    raw_dir: Path, files: Sequence[HfSampleFile]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_file in files:
        local_path = raw_dir / sample_file.file
        if not local_path.is_file():
            raise FileNotFoundError(f"missing downloaded parquet: {local_path}")
        size = local_path.stat().st_size
        if size != sample_file.size:
            raise ValueError(
                f"size mismatch for {local_path}: expected {sample_file.size}, got {size}"
            )
        rows.append(
            {
                "file": sample_file.file,
                "local_path": str(local_path),
                "size": size,
            }
        )
    return rows


def command_download(args: argparse.Namespace) -> int:
    require_rootfs()
    started = time.monotonic()
    requested_revision = args.revision
    hf_revision, files = dataset_info_for_revision(requested_revision)
    raw_dir = args.raw_dir or default_raw_dir(hf_revision)
    raw_dir = raw_dir.resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision=hf_revision,
        allow_patterns=[f"{SAMPLE_RELPATH.as_posix()}/*.parquet"],
        local_dir=str(raw_dir),
        max_workers=args.download_workers,
    )

    manifest_rows = validate_local_sample_files(raw_dir, files)
    total_bytes = sum(row["size"] for row in manifest_rows)
    elapsed = time.monotonic() - started
    write_jsonl_atomic(raw_dir / "RAW_MANIFEST.jsonl", manifest_rows)
    (raw_dir / "DATASET_REVISION.txt").write_text(hf_revision + "\n")
    metadata = {
        "created_at": utc_now(),
        "dataset_repo": DATASET_REPO,
        "dataset_config": DATASET_CONFIG,
        "requested_revision": requested_revision,
        "hf_revision": hf_revision,
        "raw_dir": str(raw_dir),
        "num_files": len(manifest_rows),
        "total_bytes": total_bytes,
        "elapsed_sec": elapsed,
    }
    write_json_atomic(raw_dir / "RAW_METADATA.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def resolve_parquet_files(raw_dir: Path) -> list[Path]:
    sample_files = sorted((raw_dir / SAMPLE_RELPATH).glob("*.parquet"))
    if sample_files:
        return sample_files
    return sorted(raw_dir.glob("*.parquet"))


def parquet_num_rows(path: Path) -> int:
    parquet_file = pq.ParquetFile(path)
    return parquet_file.metadata.num_rows


def iter_text_batches(
    parquet_files: Sequence[Path], *, batch_rows: int
) -> Iterator[list[str]]:
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=batch_rows, columns=["text"]):
            text_column = batch.column("text").to_pylist()
            yield ["" if value is None else str(value) for value in text_column]


def init_tokenizer_worker() -> None:
    global _ENCODER
    _ENCODER = tiktoken.get_encoding("gpt2")


def require_encoder() -> tiktoken.Encoding:
    if _ENCODER is None:
        init_tokenizer_worker()
    if _ENCODER is None:
        raise RuntimeError("failed to initialize GPT-2 tokenizer")
    return _ENCODER


def tokenize_text_batch(texts: list[str]) -> list[np.ndarray]:
    encoder = require_encoder()
    encoded_batch = encoder.encode_ordinary_batch(texts, num_threads=1)
    token_arrays: list[np.ndarray] = []
    for token_ids in encoded_batch:
        if token_ids and max(token_ids) > np.iinfo(np.uint16).max:
            raise ValueError("GPT-2 token id does not fit in uint16")
        tokens = np.empty(len(token_ids) + 1, dtype=np.uint16)
        tokens[0] = EOT_TOKEN_ID
        if token_ids:
            tokens[1:] = np.asarray(token_ids, dtype=np.uint16)
        token_arrays.append(tokens)
    return token_arrays


def write_nanogpt_bin_np(path: Path, tokens: np.ndarray) -> None:
    if tokens.dtype != np.uint16:
        raise ValueError("nanogpt bin writer expects uint16 tokens")
    if tokens.size >= 2**31:
        raise ValueError("token count too large for nanogpt header")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.zeros(NANOGPT_HEADER_INTS, dtype=np.int32)
    header[0] = NANOGPT_MAGIC
    header[1] = NANOGPT_VERSION
    header[2] = tokens.size
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())


class NanogptShardWriter:
    def __init__(self, output_dir: Path, *, shard_size: int) -> None:
        if shard_size < 1:
            raise ValueError("shard_size must be positive")
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.buffer = np.empty(shard_size, dtype=np.uint16)
        self.token_count = 0
        self.shard_index = 0
        self.records: list[ShardRecord] = []

    def add_document_tokens(self, tokens: np.ndarray) -> None:
        offset = 0
        while offset < tokens.size:
            available = self.shard_size - self.token_count
            take = min(available, tokens.size - offset)
            end = self.token_count + take
            self.buffer[self.token_count : end] = tokens[offset : offset + take]
            self.token_count = end
            offset += take
            if self.token_count == self.shard_size:
                self.flush()

    def flush(self) -> None:
        if self.token_count == 0:
            return
        split = "val" if self.shard_index == 0 else "train"
        filename = f"fineweb_edu_{split}_{self.shard_index:06d}.bin"
        path = self.output_dir / filename
        write_nanogpt_bin_np(path, self.buffer[: self.token_count])
        record = ShardRecord(
            file=filename,
            size=path.stat().st_size,
            md5=md5_file(path),
            num_tokens=self.token_count,
            split=split,
            shard_index=self.shard_index,
        )
        self.records.append(record)
        self.shard_index += 1
        self.token_count = 0


def clean_generated_outputs(output_dir: Path) -> None:
    for path in output_dir.glob("fineweb_edu_*.bin"):
        path.unlink()
    for name in ("MANIFEST.jsonl", "METADATA.json", "VERIFY.json"):
        path = output_dir / name
        if path.exists():
            path.unlink()


def tokenizer_environment() -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "tiktoken": getattr(tiktoken, "__version__", "unknown"),
    }


def command_tokenize(args: argparse.Namespace) -> int:
    require_rootfs()
    started = time.monotonic()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    parquet_files = resolve_parquet_files(raw_dir)
    if not parquet_files:
        raise FileNotFoundError(f"no sample parquet files found under {raw_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_shards = sorted(output_dir.glob("fineweb_edu_*.bin"))
    if existing_shards and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains generated shards; pass --overwrite"
        )
    if args.overwrite:
        clean_generated_outputs(output_dir)

    revision_path = raw_dir / "DATASET_REVISION.txt"
    hf_revision = (
        revision_path.read_text(encoding="utf-8").strip()
        if revision_path.is_file()
        else args.revision
    )
    num_documents_expected = sum(parquet_num_rows(path) for path in parquet_files)
    writer = NanogptShardWriter(output_dir, shard_size=args.shard_size)
    num_documents = 0
    num_tokens = 0
    batch_iter = iter_text_batches(parquet_files, batch_rows=args.batch_rows)

    if args.num_processes == 1:
        init_tokenizer_worker()
        mapped_batches: Iterable[list[np.ndarray]] = map(
            tokenize_text_batch, batch_iter
        )
        progress = tqdm(total=num_documents_expected, unit="doc", smoothing=0.01)
        for token_batch in mapped_batches:
            for tokens in token_batch:
                writer.add_document_tokens(tokens)
                num_documents += 1
                num_tokens += int(tokens.size)
            progress.update(len(token_batch))
        progress.close()
    else:
        with mp.Pool(args.num_processes, initializer=init_tokenizer_worker) as pool:
            mapped_batches = pool.imap(
                tokenize_text_batch, batch_iter, chunksize=args.pool_chunksize
            )
            progress = tqdm(total=num_documents_expected, unit="doc", smoothing=0.01)
            for token_batch in mapped_batches:
                for tokens in token_batch:
                    writer.add_document_tokens(tokens)
                    num_documents += 1
                    num_tokens += int(tokens.size)
                progress.update(len(token_batch))
            progress.close()

    writer.flush()
    elapsed = time.monotonic() - started
    manifest_rows = [
        {
            "file": record.file,
            "size": record.size,
            "md5": record.md5,
            "num_tokens": record.num_tokens,
            "split": record.split,
            "shard_index": record.shard_index,
        }
        for record in writer.records
    ]
    write_jsonl_atomic(output_dir / "MANIFEST.jsonl", manifest_rows)
    metadata = {
        "created_at": utc_now(),
        "dataset_repo": DATASET_REPO,
        "dataset_config": DATASET_CONFIG,
        "hf_revision": hf_revision,
        "hf_revision_short": short_revision(hf_revision),
        "tokenizer": "tiktoken:gpt2",
        "eot_token_id": EOT_TOKEN_ID,
        "format": {
            "magic": NANOGPT_MAGIC,
            "version": NANOGPT_VERSION,
            "header_int32_count": NANOGPT_HEADER_INTS,
            "token_dtype": "uint16",
        },
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "parquet_files": [str(path) for path in parquet_files],
        "num_parquet_files": len(parquet_files),
        "num_documents": num_documents,
        "num_documents_expected": num_documents_expected,
        "num_tokens": num_tokens,
        "shard_size_tokens": args.shard_size,
        "num_shards": len(manifest_rows),
        "num_val_shards": sum(row["split"] == "val" for row in manifest_rows),
        "num_train_shards": sum(row["split"] == "train" for row in manifest_rows),
        "elapsed_sec": elapsed,
        "tokens_per_sec": num_tokens / elapsed if elapsed > 0 else 0,
        "documents_per_sec": num_documents / elapsed if elapsed > 0 else 0,
        "environment": tokenizer_environment(),
    }
    write_json_atomic(output_dir / "METADATA.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def read_nanogpt_header(path: Path) -> dict[str, Any]:
    header = np.fromfile(path, dtype=np.int32, count=NANOGPT_HEADER_INTS)
    if header.size != NANOGPT_HEADER_INTS:
        raise ValueError(f"{path} is smaller than a nanogpt header")
    magic = int(header[0])
    version = int(header[1])
    num_tokens = int(header[2])
    expected_size = NANOGPT_HEADER_INTS * 4 + num_tokens * 2
    actual_size = path.stat().st_size
    if magic != NANOGPT_MAGIC:
        raise ValueError(f"{path} magic mismatch: {magic}")
    if version != NANOGPT_VERSION:
        raise ValueError(f"{path} version mismatch: {version}")
    if actual_size != expected_size:
        raise ValueError(
            f"{path} size mismatch: expected {expected_size}, got {actual_size}"
        )
    return {
        "magic": magic,
        "version": version,
        "num_tokens": num_tokens,
        "size": actual_size,
    }


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def verify_manifest_rows(data_dir: Path, rows: Sequence[dict[str, Any]]) -> int:
    total_tokens = 0
    for row in rows:
        path = data_dir / str(row["file"])
        header = read_nanogpt_header(path)
        total_tokens += int(header["num_tokens"])
        expected_size = int(row["size"])
        expected_tokens = int(row["num_tokens"])
        if int(header["size"]) != expected_size:
            raise ValueError(
                f"manifest size mismatch for {path}: "
                f"{expected_size} vs {header['size']}"
            )
        if int(header["num_tokens"]) != expected_tokens:
            raise ValueError(
                f"manifest token mismatch for {path}: "
                f"{expected_tokens} vs {header['num_tokens']}"
            )
        expected_md5 = str(row["md5"])
        actual_md5 = md5_file(path)
        if actual_md5 != expected_md5:
            raise ValueError(
                f"manifest md5 mismatch for {path}: {expected_md5} vs {actual_md5}"
            )
    return total_tokens


def command_verify(args: argparse.Namespace) -> int:
    require_rootfs()
    started = time.monotonic()
    data_dir = args.data_dir.resolve()
    manifest_path = data_dir / "MANIFEST.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest_rows = read_manifest(manifest_path)
    if not manifest_rows:
        raise ValueError(f"empty manifest: {manifest_path}")
    total_tokens = verify_manifest_rows(data_dir, manifest_rows)

    spot_file = data_dir / str(manifest_rows[0]["file"])
    spot_tokens = read_nanogpt_bin(spot_file)
    token_prefix = spot_tokens[: args.spot_tokens].tolist()
    encoder = tiktoken.get_encoding("gpt2")
    decoded_prefix = encoder.decode(token_prefix)
    elapsed = time.monotonic() - started
    result = {
        "created_at": utc_now(),
        "data_dir": str(data_dir),
        "manifest": str(manifest_path),
        "num_shards": len(manifest_rows),
        "total_tokens": total_tokens,
        "spot_check": {
            "reader": "torchtitan.experiments.falcon.bin_reader.read_nanogpt_bin",
            "file": str(spot_file),
            "num_tokens_from_reader": int(spot_tokens.numel()),
            "prefix_token_ids": token_prefix,
            "decoded_prefix": decoded_prefix[: args.decoded_chars],
            "starts_with_eot": bool(token_prefix and token_prefix[0] == EOT_TOKEN_ID),
        },
        "elapsed_sec": elapsed,
    }
    write_json_atomic(data_dir / "VERIFY.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("--revision", default=DEFAULT_REVISION)
    download.add_argument("--raw-dir", type=Path)
    download.add_argument("--download-workers", type=int, default=8)
    download.set_defaults(func=command_download)

    tokenize = subparsers.add_parser("tokenize")
    tokenize.add_argument("--revision", default=DEFAULT_REVISION)
    tokenize.add_argument("--raw-dir", type=Path, required=True)
    tokenize.add_argument("--output-dir", type=Path, required=True)
    tokenize.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    tokenize.add_argument("--batch-rows", type=int, default=2048)
    default_processes = min(64, max(1, (os.cpu_count() or 8) - 8))
    tokenize.add_argument("--num-processes", type=int, default=default_processes)
    tokenize.add_argument("--pool-chunksize", type=int, default=1)
    tokenize.add_argument("--overwrite", action="store_true")
    tokenize.set_defaults(func=command_tokenize)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--data-dir", type=Path, required=True)
    verify.add_argument("--spot-tokens", type=int, default=128)
    verify.add_argument("--decoded-chars", type=int, default=400)
    verify.set_defaults(func=command_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
