# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optional lightweight behavior probes for speedrun ablation runs."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


_LOCK = threading.Lock()
_RECORD_COUNTS: defaultdict[tuple[str, str], int] = defaultdict(int)


def behavior_enabled() -> bool:
    return os.environ.get("GLM53_SPEEDRUN_BEHAVIOR_READS", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _rank_enabled() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _behavior_path() -> Path:
    configured = os.environ.get("GLM53_SPEEDRUN_BEHAVIOR_FILE")
    if configured:
        return Path(configured)
    dump_folder = os.environ.get("GLM53_SPEEDRUN_DUMP_FOLDER", ".")
    return Path(dump_folder) / "behavior_reads.jsonl"


def should_record(kind: str, tag: str, call_count: int) -> bool:
    if not behavior_enabled() or not _rank_enabled():
        return False
    interval = int(os.environ.get("GLM53_SPEEDRUN_BEHAVIOR_EVERY_CALLS", "8192"))
    max_records = int(os.environ.get("GLM53_SPEEDRUN_BEHAVIOR_MAX_RECORDS", "4"))
    if call_count != 1 and (interval < 1 or call_count % interval != 0):
        return False
    key = (kind, tag)
    if _RECORD_COUNTS[key] >= max_records:
        return False
    _RECORD_COUNTS[key] += 1
    return True


def scalar(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def record(payload: dict[str, Any]) -> None:
    if not behavior_enabled() or not _rank_enabled():
        return
    payload = {
        "time_unix": time.time(),
        "run_id": os.environ.get("TORCHTITAN_RUN_ID"),
        "attempt_id": os.environ.get("TORCHTITAN_ATTEMPT_ID"),
        **payload,
    }
    path = _behavior_path()
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
