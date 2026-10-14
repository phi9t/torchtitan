# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rank-local CUDA and RSS evidence helpers for the 4-GPU GLM lean probes."""

from __future__ import annotations

import json
import logging
import os
import resource
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def env_int_set(name: str, default: set[int] | None = None) -> set[int]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return set(default or set())
    values: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.add(int(item))
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated int list") from exc
    return values


def current_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def current_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def current_rss_bytes() -> int:
    try:
        with Path("/proc/self/status").open() as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        return 0
    return 0


def cuda_memory_record(label: str) -> dict[str, Any]:
    maxrss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record: dict[str, Any] = {
        "label": label,
        "time_unix": time.time(),
        "pid": os.getpid(),
        "rank": current_rank(),
        "local_rank": current_local_rank(),
        "rss_gib": current_rss_bytes() / (1024**3),
        "maxrss_gib": (maxrss_kib * 1024) / (1024**3),
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        return record

    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    stats = torch.cuda.memory_stats(device)
    record.update(
        {
            "device": device,
            "device_name": torch.cuda.get_device_name(device),
            "mem_get_info_free_gib": free_bytes / (1024**3),
            "mem_get_info_total_gib": total_bytes / (1024**3),
            "allocated_gib": torch.cuda.memory_allocated(device) / (1024**3),
            "reserved_gib": torch.cuda.memory_reserved(device) / (1024**3),
            "max_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "max_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
            "active_bytes_gib": stats.get("active_bytes.all.current", 0)
            / (1024**3),
            "inactive_split_bytes_gib": stats.get(
                "inactive_split_bytes.all.current", 0
            )
            / (1024**3),
            "allocated_bytes_all_peak_gib": stats.get(
                "allocated_bytes.all.peak", 0
            )
            / (1024**3),
            "reserved_bytes_all_peak_gib": stats.get("reserved_bytes.all.peak", 0)
            / (1024**3),
        }
    )
    return record


def write_memory_evidence(
    dump_folder: str | Path,
    *,
    label: str,
    include_summary: bool,
    include_snapshot: bool,
) -> dict[str, Any]:
    evidence_dir = Path(dump_folder) / "lean_memory"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rank = current_rank()
    safe_label = label.replace("/", "_").replace(" ", "_")
    record = cuda_memory_record(label)

    json_path = evidence_dir / f"rank{rank:05d}_{safe_label}.json"
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    if torch.cuda.is_available() and include_summary:
        summary_path = evidence_dir / f"rank{rank:05d}_{safe_label}_summary.txt"
        summary_path.write_text(torch.cuda.memory_summary())

    if torch.cuda.is_available() and include_snapshot:
        snapshot_path = evidence_dir / f"rank{rank:05d}_{safe_label}_snapshot.pickle"
        try:
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
        except Exception as exc:
            logger.warning("Could not dump CUDA memory snapshot: %s", exc)
            error_path = evidence_dir / f"rank{rank:05d}_{safe_label}_snapshot_error.txt"
            error_path.write_text(f"{type(exc).__name__}: {exc}\n")

    return record


def maybe_enable_cuda_memory_history() -> None:
    if not torch.cuda.is_available():
        return
    if not env_flag("GLM53_LEAN_RECORD_CUDA_MEMORY", False):
        return
    if not env_flag("GLM53_LEAN_CUDA_MEMORY_HISTORY", False):
        return
    try:
        torch.cuda.memory._record_memory_history(max_entries=200000)
    except Exception as exc:
        logger.warning("Could not enable CUDA memory history: %s", exc)

