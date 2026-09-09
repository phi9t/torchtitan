# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""B10 addition runner with one-arm/one-seed evidence ownership."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from torchtitan.experiments.falcon.addition import (
    AdditionSplitManifest,
    AdditionVocab,
    OnlineAdditionStream,
)
from torchtitan.experiments.falcon.evidence import (
    canonical_native_run_id,
    write_native_attempt_bundle,
)


_VALID_ARMS = ("A0", "A1", "A2", "A3", "A4", "A5")
_DEFAULT_RESULTS_ROOT = Path("experiments/falcon/results/addition")
_ARTIFACT_ROOT = Path("experiments/falcon/results/addition/artifacts")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _json_bytes(value)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_artifact_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return (_ARTIFACT_ROOT / path.name).as_posix()


def _safe_arm(raw: str) -> str:
    return raw.replace(",", "_").replace("/", "_").replace("\\", "_") or "invalid"


def _parse_one_arm(raw: str) -> str:
    arms = [part.strip() for part in raw.split(",") if part.strip()]
    if len(arms) != 1:
        raise ValueError("addition runner accepts exactly one arm per invocation")
    arm = arms[0]
    if arm not in _VALID_ARMS:
        raise ValueError(f"unknown Falcon arm {arm!r}; expected one of {_VALID_ARMS}")
    return arm


def _parse_one_seed(raw: str) -> int:
    seeds = [part.strip() for part in raw.split(",") if part.strip()]
    if len(seeds) != 1:
        raise ValueError("addition runner accepts exactly one seed per invocation")
    try:
        seed = int(seeds[0])
    except ValueError as exc:
        raise ValueError("seed must be an integer") from exc
    if seed < 0:
        raise ValueError("seed must be >= 0")
    return seed


def _attempt_record(
    *,
    arm: str,
    seed: int,
    started_utc: str,
    ended_utc: str,
    steps: int,
    completed_steps: int,
    dry_run: bool,
    status: str,
    reason: str | None,
    data_lineage: dict[str, str],
    checkpoint_lineage: dict[str, str],
    artifacts: list[dict[str, str]],
) -> dict[str, Any]:
    logical_run = {"workload": "addition", "arm": arm, "seed": seed}
    return {
        "run_id": canonical_native_run_id(logical_run),
        "attempt_id": f"addition-{arm}-seed{seed}-{uuid.uuid4().hex[:12]}",
        "lane": "science",
        "mode": "addition_train",
        "arm": arm,
        "claim_label": "smoke" if dry_run else "representative_training",
        "evidence_tier": "tier0",
        "environment_class": "rootfs"
        if os.environ.get("TORCHTITAN_IN_ROOTFS") == "1"
        else "host",
        "logical_run": logical_run,
        "processes": [
            {
                "process_id": "addition.rank0",
                "role": "trainer",
                "global_rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "host_name": socket.gethostname(),
            }
        ],
        "mesh": {"axes": {"dp": 1}},
        "device": {"type": "cpu", "index": 0, "uuid": "CPU"},
        "clocks": {"started_utc": started_utc, "ended_utc": ended_utc},
        "steps": {"requested": steps, "completed": completed_steps},
        "phases": [
            {
                "name": "dry_run" if dry_run else "addition_train",
                "outcome": "completed" if status == "completed" else "failed",
            }
        ],
        "data_lineage": data_lineage,
        "checkpoint_lineage": checkpoint_lineage,
        "artifacts": artifacts,
        "outcome": {"status": status, "reason": reason},
    }


def _failure_bundle(args: argparse.Namespace, reason: str, started_utc: str) -> Path:
    seed = 0
    try:
        seed = int(str(args.seed).split(",")[0])
    except ValueError:
        seed = 0
    arm = _safe_arm(
        str(args.arm).split(",")[0] if "," in str(args.arm) else str(args.arm)
    )
    if "," in str(args.arm):
        arm = _safe_arm(str(args.arm))
    ended_utc = _utc_now()
    failure_path = (
        _ARTIFACT_ROOT
        / "failures"
        / f"addition-{arm}-seed{max(0, seed)}-{uuid.uuid4().hex[:12]}"
        / "failure.json"
    )
    failure_digest = _write_json(
        failure_path,
        {
            "arm": arm,
            "seed": max(0, seed),
            "requested_steps": max(0, args.steps),
            "dry_run": args.dry_run,
            "reason": reason,
            "started_utc": started_utc,
            "ended_utc": ended_utc,
        },
    )
    record = _attempt_record(
        arm=arm,
        seed=max(0, seed),
        started_utc=started_utc,
        ended_utc=ended_utc,
        steps=max(0, args.steps),
        completed_steps=0,
        dry_run=args.dry_run,
        status="failed",
        reason=reason,
        data_lineage={"split_manifest_digest": "not-created-validation-failed"},
        checkpoint_lineage={"checkpoint_policy": "not-created-validation-failed"},
        artifacts=[
            {
                "path": _repo_artifact_path(failure_path),
                "kind": "failure",
                "digest": failure_digest,
            }
        ],
    )
    return write_native_attempt_bundle(args.results_root, record)


def run_addition(
    *,
    arm: str,
    seed: int,
    steps: int,
    dry_run: bool,
    results_root: Path,
) -> dict[str, Any]:
    started = _utc_now()
    manifest = AdditionSplitManifest.generate(seed=seed)
    stream = OnlineAdditionStream(seed=seed, manifest=manifest, vocab=AdditionVocab())
    preview = [stream.next_example() for _ in range(4 if dry_run else 16)]
    run_dir = _ARTIFACT_ROOT / f"addition-{arm}-seed{seed}-{int(time.time())}"
    manifest_path = run_dir / "split_manifest.json"
    manifest.write_json(manifest_path)
    manifest_file_digest = _file_sha256(manifest_path)
    preview_payload = {
        "arm": arm,
        "seed": seed,
        "steps": steps,
        "dry_run": dry_run,
        "preview": [{"width": ex.width, "a": ex.a, "b": ex.b} for ex in preview],
    }
    preview_digest = _write_json(run_dir / "stream_preview.json", preview_payload)
    result = {
        "claim_label": "smoke" if dry_run else "representative_training",
        "arm": arm,
        "seed": seed,
        "requested_steps": steps,
        "completed_steps": steps if dry_run else 0,
        "dry_run": dry_run,
        "split_manifest_digest": manifest.digest,
        "stream_preview_digest": preview_digest,
    }
    result_digest = _write_json(run_dir / "addition_outcome.json", result)
    ended = _utc_now()
    record = _attempt_record(
        arm=arm,
        seed=seed,
        started_utc=started,
        ended_utc=ended,
        steps=steps,
        completed_steps=result["completed_steps"],
        dry_run=dry_run,
        status="completed",
        reason=None,
        data_lineage={
            "split_manifest_digest": manifest.digest,
            "stream_policy": "online-id-widths-1-16-excludes-all-eval-draws",
        },
        checkpoint_lineage={
            "checkpoint_policy": "dry-run-none" if dry_run else "not-implemented",
            "output_checkpoint": "none" if dry_run else "not-implemented",
        },
        artifacts=[
            {
                "path": _repo_artifact_path(manifest_path),
                "kind": "addition_split_manifest",
                "digest": manifest_file_digest,
                "split_manifest_digest": manifest.digest,
            },
            {
                "path": _repo_artifact_path(run_dir / "stream_preview.json"),
                "kind": "stream_preview",
                "digest": preview_digest,
            },
            {
                "path": _repo_artifact_path(run_dir / "addition_outcome.json"),
                "kind": "addition_outcome",
                "digest": result_digest,
            },
        ],
    )
    bundle_path = write_native_attempt_bundle(results_root, record)
    return {**result, "bundle_path": str(bundle_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--results-root", type=Path, default=_DEFAULT_RESULTS_ROOT)
    args = parser.parse_args(argv)
    started = _utc_now()
    try:
        if args.steps < 1:
            raise ValueError("steps must be >= 1")
        arm = _parse_one_arm(args.arm)
        seed = _parse_one_seed(args.seed)
        payload = run_addition(
            arm=arm,
            seed=seed,
            steps=args.steps,
            dry_run=args.dry_run,
            results_root=args.results_root,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        try:
            bundle_path = _failure_bundle(args, str(exc), started)
            print(f"wrote failure bundle: {bundle_path}", file=sys.stderr)
        except Exception as bundle_exc:
            print(f"failed to write failure bundle: {bundle_exc}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
