# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read-only fixed-region evaluation for one Falcon checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from torchtitan.experiments.falcon.evidence import (
    canonical_native_run_id,
    write_native_attempt_bundle,
)
from torchtitan.experiments.falcon.lm_eval import (
    aggregate_region_ce,
    build_fixed_region_registry,
    evaluate_regions,
)


_ARMS = ("A0", "A1", "A5")
_RESULTS_ROOT = Path("experiments/falcon/results/evidence")
_ATTEMPT_ROOT = Path("experiments/falcon/results/checkpoint_eval")


class CheckpointMutationError(RuntimeError):
    """A historical checkpoint does not match its required immutable snapshot."""


@dataclass(frozen=True, slots=True)
class AttemptIds:
    run_id: str
    attempt_id: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_checkpoint_tree(checkpoint: str | Path) -> dict[str, Any]:
    """Return file metadata and digests for a checkpoint directory."""
    root = Path(checkpoint)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory is missing: {checkpoint}")
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "mode": stat.st_mode,
                "digest_algorithm": "sha256",
                "digest": _sha256_file(path),
            }
        )
    if not files:
        raise FileNotFoundError(f"checkpoint contains no files: {checkpoint}")
    return {"checkpoint_path": str(root), "files": files}


def checkpoint_tree_digest(snapshot: dict[str, Any]) -> str:
    payload = {
        "files": [
            {
                "path": file["path"],
                "size": file["size"],
                "digest": file["digest"],
            }
            for file in snapshot["files"]
        ]
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def require_checkpoint_digest(snapshot: dict[str, Any], expected_digest: str) -> None:
    actual = checkpoint_tree_digest(snapshot)
    if actual != expected_digest:
        raise CheckpointMutationError(
            f"checkpoint digest mismatch: got {actual}, expected {expected_digest}"
        )


def assert_checkpoint_unchanged(
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    if before != after:
        raise CheckpointMutationError("historical checkpoint mutated during eval")


def build_attempt_ids(*, arm: str, checkpoint_digest: str) -> AttemptIds:
    logical_run = {"workload": "b11-checkpoint-eval", "arm": arm, "seed": 0}
    run_id = canonical_native_run_id(logical_run)
    attempt_id = (
        f"ckpt-eval-{arm}-"
        f"{checkpoint_digest[:12]}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    return AttemptIds(run_id=run_id, attempt_id=attempt_id)


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _build_eval_config(arm: str, checkpoint: Path, attempt_dir: Path):
    from torchtitan.config import ConfigManager
    from torchtitan.experiments.falcon.config_registry import apply_arm, falcon_science

    config = falcon_science()
    apply_arm(config, arm)
    restore_step = _checkpoint_step(checkpoint)
    config.training.steps = restore_step
    config.lr_scheduler.total_steps = max(restore_step, 1)
    config.lr_scheduler.warmup_steps = min(200, max(1, restore_step // 40))
    config.debug.seed = 0
    config.debug.enable_structured_logging = False
    config.dump_folder = str(attempt_dir)
    config.checkpoint.enable = True
    config.checkpoint.load_only = True
    config.checkpoint.initial_load_path = str(checkpoint.resolve())
    config.checkpoint.initial_load_model_only = False
    config.checkpoint.load_step = -1
    config.metrics.log_freq = 1

    manager = ConfigManager()
    manager.config = config
    manager._validate_config()
    return config


def _checkpoint_step(checkpoint: Path) -> int:
    name = checkpoint.name
    if name.startswith("step-"):
        try:
            return int(name.removeprefix("step-"))
        except ValueError:
            pass
    return 0


def _set_single_process_env(*, master_port: int) -> None:
    for key, value in (
        ("LOCAL_RANK", "0"),
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("NGPU", "1"),
    ):
        existing = os.environ.get(key)
        if existing is not None and existing != value:
            raise RuntimeError(
                f"checkpoint-eval requires {key}={value}, got {existing!r}"
            )
        os.environ[key] = value
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_checkpoint_eval(
    *,
    arm: str,
    checkpoint: Path,
    master_port: int,
    expected_checkpoint_digest: str | None = None,
) -> dict[str, Any]:
    if arm not in _ARMS:
        raise ValueError(f"checkpoint-eval arm must be one of {_ARMS}, got {arm!r}")
    _set_single_process_env(master_port=master_port)
    started = _utc_now()
    before = snapshot_checkpoint_tree(checkpoint)
    checkpoint_digest = checkpoint_tree_digest(before)
    if expected_checkpoint_digest:
        require_checkpoint_digest(before, expected_checkpoint_digest)
    ids = build_attempt_ids(arm=arm, checkpoint_digest=checkpoint_digest)
    attempt_dir = _ATTEMPT_ROOT / ids.run_id / ids.attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)

    registry = build_fixed_region_registry()
    _json_write(attempt_dir / "checkpoint_before.json", before)
    _json_write(attempt_dir / "region_registry.json", registry.to_dict())

    trainer = None
    loaded = False
    try:
        config = _build_eval_config(arm, checkpoint, attempt_dir)
        trainer = config.build()
        loaded = trainer.checkpointer.load(step=config.checkpoint.load_step)
        if not loaded:
            raise RuntimeError(f"checkpoint did not load: {checkpoint}")
        inner = config.model_spec.model.config
        region_results = evaluate_regions(
            trainer.model_parts[0],
            registry,
            seq_len=config.training.seq_len,
            vocab_size=inner.vocab_size,
        )
        aggregate = aggregate_region_ce(region_results, registry)
    finally:
        if trainer is not None:
            try:
                trainer.close()
            finally:
                after = snapshot_checkpoint_tree(checkpoint)
                _json_write(attempt_dir / "checkpoint_after.json", after)
                assert_checkpoint_unchanged(before, after)
        else:
            after = snapshot_checkpoint_tree(checkpoint)
            _json_write(attempt_dir / "checkpoint_after.json", after)
            assert_checkpoint_unchanged(before, after)
    ended = _utc_now()

    result = {
        "run_id": ids.run_id,
        "attempt_id": ids.attempt_id,
        "arm": arm,
        "checkpoint_path": str(checkpoint),
        "checkpoint_digest": checkpoint_digest,
        "registry_digest": registry.registry_digest,
        "regions": region_results,
        "aggregate": aggregate,
        "attempt_dir": str(attempt_dir),
        "loaded": loaded,
        "started_utc": started,
        "ended_utc": ended,
    }
    _json_write(attempt_dir / "result.json", result)

    evidence_record = {
        "run_id": ids.run_id,
        "attempt_id": ids.attempt_id,
        "lane": "science",
        "mode": "checkpoint_eval",
        "arm": arm,
        "claim_label": "representative_eval",
        "evidence_tier": "tier0",
        "environment_class": "rootfs_b200",
        "logical_run": {"workload": "b11-checkpoint-eval", "arm": arm, "seed": 0},
        "processes": [
            {
                "process_id": "checkpoint_eval.rank0",
                "role": "evaluator",
                "global_rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "host_name": socket.gethostname(),
            }
        ],
        "mesh": {"axes": {"dp": 1}},
        "device": {"type": "cuda", "index": 0, "uuid": "unknown"},
        "clocks": {"started_utc": started, "ended_utc": ended},
        "steps": {"requested": 0, "completed": 0},
        "phases": [
            {"name": "snapshot_before", "outcome": "completed"},
            {"name": "checkpoint_restore", "outcome": "completed"},
            {"name": "fixed_region_eval", "outcome": "completed"},
            {"name": "mutation_guard", "outcome": "completed"},
        ],
        "data_lineage": {
            "region_registry": registry.registry_digest,
            "source_path": registry.source_path,
            "source_digest": registry.source_digest,
        },
        "checkpoint_lineage": {
            "input_checkpoint": str(checkpoint),
            "input_checkpoint_digest": checkpoint_digest,
        },
        "artifacts": [
            {"path": (attempt_dir / "result.json").as_posix(), "kind": "eval_result"},
            {
                "path": (attempt_dir / "region_registry.json").as_posix(),
                "kind": "region_registry",
            },
            {
                "path": (attempt_dir / "checkpoint_before.json").as_posix(),
                "kind": "checkpoint_snapshot",
            },
            {
                "path": (attempt_dir / "checkpoint_after.json").as_posix(),
                "kind": "checkpoint_snapshot",
            },
        ],
        "outcome": {
            "status": "completed",
            "reason": None,
            "aggregate_ce": aggregate["aggregate_ce"],
            "ppl": aggregate["ppl"],
            "checkpoint_unchanged": True,
        },
    }
    bundle = write_native_attempt_bundle(_RESULTS_ROOT, evidence_record)
    result["evidence_bundle"] = str(bundle)
    _json_write(attempt_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=_ARMS)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--master-port", type=int, default=29590)
    parser.add_argument("--expected-checkpoint-digest")
    args = parser.parse_args()
    started = time.time()
    result = run_checkpoint_eval(
        arm=args.arm,
        checkpoint=args.checkpoint,
        master_port=args.master_port,
        expected_checkpoint_digest=args.expected_checkpoint_digest,
    )
    result["elapsed_sec"] = round(time.time() - started, 2)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
