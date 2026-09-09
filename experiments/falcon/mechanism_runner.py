# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""M02 one-arm/one-seed mechanism runner with observable probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from torchtitan.experiments.falcon.evidence import (
    canonical_native_run_id,
    write_native_attempt_bundle,
)
from torchtitan.experiments.falcon.mechanism import MechanismDiagnostics


_VALID_MECHANISM_ARMS = ("M0", "M1", "M2", "M3", "M4")
_SCREEN_MILESTONES = (0, 100, 500, 1000, 2000)
_CONFIRMATION_EXTRA_MILESTONES = (4000, 8000)
_DEFAULT_RESULTS_ROOT = Path("experiments/falcon/results/mechanism/evidence")
_DEFAULT_ARTIFACT_ROOT = Path("experiments/falcon/results/mechanism/artifacts")


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


def new_attempt_id(arm: str, seed: int) -> str:
    return f"mechanism-{arm}-seed{seed}-{uuid.uuid4().hex[:12]}"


def mechanism_artifact_dir(artifact_root: Path, attempt_id: str) -> Path:
    return artifact_root / attempt_id


def _repo_artifact_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
        return f"external-mechanism-artifacts/{digest}/{path.name}"


def _safe_arm(raw: str) -> str:
    return raw.replace(",", "_").replace("/", "_").replace("\\", "_") or "invalid"


def parse_one_arm(raw: str) -> str:
    arms = [part.strip() for part in raw.split(",") if part.strip()]
    if len(arms) != 1:
        raise ValueError("mechanism runner accepts exactly one mechanism arm")
    arm = arms[0]
    if arm not in _VALID_MECHANISM_ARMS:
        raise ValueError(
            f"unknown Falcon mechanism arm {arm!r}; "
            f"expected one of {_VALID_MECHANISM_ARMS}"
        )
    return arm


def parse_one_seed(raw: str) -> int:
    seeds = [part.strip() for part in raw.split(",") if part.strip()]
    if len(seeds) != 1:
        raise ValueError("mechanism runner accepts exactly one seed")
    try:
        seed = int(seeds[0])
    except ValueError as exc:
        raise ValueError("seed must be an integer") from exc
    if seed < 0:
        raise ValueError("seed must be >= 0")
    return seed


def mechanism_probe_steps(steps: int, *, confirmation: bool) -> list[int]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if confirmation:
        if steps < 8000:
            raise ValueError("confirmation mechanism attempts require steps >= 8000")
        return [*_SCREEN_MILESTONES, *_CONFIRMATION_EXTRA_MILESTONES]
    if steps < 2000:
        raise ValueError("screen mechanism attempts require steps >= 2000")
    return list(_SCREEN_MILESTONES)


def _probe_steps_for_attempt(
    steps: int, *, dry_run: bool, confirmation: bool
) -> list[int]:
    if dry_run:
        return [0, steps]
    return mechanism_probe_steps(steps, confirmation=confirmation)


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
                f"mechanism runner requires {key}={value}, got {existing!r}"
            )
        os.environ[key] = value
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)


def build_mechanism_config(
    arm: str,
    *,
    seed: int,
    steps: int,
    dump_folder: Path,
    dry_run: bool,
):
    from torchtitan.config import ConfigManager
    from torchtitan.experiments.falcon.config_registry import (
        apply_mechanism_arm,
        falcon_science,
    )

    config = falcon_science()
    apply_mechanism_arm(config, arm)
    config.training.steps = steps
    config.lr_scheduler.total_steps = steps
    config.lr_scheduler.warmup_steps = min(200, max(1, steps // 40))
    config.debug.seed = seed
    config.debug.enable_structured_logging = False
    config.dump_folder = str(dump_folder)
    config.checkpoint.enable = False
    config.metrics.log_freq = 1 if dry_run else 50

    if dry_run:
        from torchtitan.experiments.falcon.repeat_data import FalconRepeatDataLoader

        inner = config.model_spec.model.config
        inner.vocab_size = 32
        inner.hidden_size = 32
        inner.num_heads = 4
        inner.head_dim = 8
        inner.num_hidden_layers = 2
        inner.intermediate_size = 64
        inner.seq_len = 16
        config.training.seq_len = 16
        config.training.local_batch_size = 4
        config.dataloader = FalconRepeatDataLoader.Config(
            num_sequences=4,
            vocab_size=32,
            seed=seed,
        )
        config.loss.global_vocab_size = 32

    manager = ConfigManager()
    manager.config = config
    manager._validate_config()
    return config


def _fixed_probe_tokens(*, vocab_size: int, seq_len: int, batch_size: int, seed: int):
    import torch

    generator = torch.Generator().manual_seed(seed + 20260908)
    return torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        generator=generator,
        dtype=torch.long,
    )


def _probe_batch_digest(tokens) -> str:
    return hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest()


def _probe_model(model, config, *, seed: int) -> dict[str, Any]:
    import torch

    inner = config.model_spec.model.config
    batch_size = min(2, config.training.local_batch_size)
    tokens = _fixed_probe_tokens(
        vocab_size=inner.vocab_size,
        seq_len=config.training.seq_len,
        batch_size=batch_size,
        seed=seed,
    )
    batch_digest = _probe_batch_digest(tokens)
    device = next(model.parameters()).device
    diagnostics = MechanismDiagnostics()
    model.eval()
    with torch.no_grad():
        logits = model(tokens.to(device), mechanism_diagnostics=diagnostics)
    model.train()
    return {
        "probe_batch_id": f"fixed-synthetic-seed{seed}",
        "probe_batch_digest": batch_digest,
        "tokens_shape": list(tokens.shape),
        "logits_checksum": float(logits.float().sum().item()),
        "diagnostics": diagnostics.summary(),
    }


def _attempt_record(
    *,
    attempt_id: str,
    arm: str,
    seed: int,
    started_utc: str,
    ended_utc: str,
    steps: int,
    completed_steps: int,
    dry_run: bool,
    confirmation: bool,
    status: str,
    reason: str | None,
    artifacts: list[dict[str, str]],
    probe_steps: list[int],
    device: dict[str, Any] | None = None,
) -> dict[str, Any]:
    logical_run = {"workload": "mechanism", "arm": arm, "seed": seed}
    return {
        "run_id": canonical_native_run_id(logical_run),
        "attempt_id": attempt_id,
        "lane": "science",
        "mode": "mechanism_observation",
        "arm": arm,
        "claim_label": "smoke"
        if dry_run
        else ("confirmation" if confirmation else "mechanism_screen"),
        "evidence_tier": "tier0",
        "environment_class": "rootfs"
        if os.environ.get("TORCHTITAN_IN_ROOTFS") == "1"
        else "host",
        "logical_run": logical_run,
        "processes": [
            {
                "process_id": "mechanism.rank0",
                "role": "trainer",
                "global_rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "host_name": socket.gethostname(),
            }
        ],
        "mesh": {"axes": {"dp": 1}},
        "device": device or {"type": "cpu", "index": 0, "uuid": "CPU"},
        "clocks": {"started_utc": started_utc, "ended_utc": ended_utc},
        "steps": {"requested": steps, "completed": completed_steps},
        "phases": [
            {
                "name": "dry_run" if dry_run else "mechanism_train",
                "outcome": "completed" if status == "completed" else "failed",
            },
            {
                "name": "mechanism_probes",
                "outcome": "completed" if status == "completed" else "failed",
                "probe_steps": probe_steps,
            },
        ],
        "data_lineage": {
            "probe_batch_policy": "fixed-synthetic-token-batches",
            "training_data": "synthetic-repeat-bank"
            if dry_run
            else "fineweb10b-train-shards",
        },
        "checkpoint_lineage": {
            "checkpoint_policy": "none",
            "output_checkpoint": "none",
        },
        "artifacts": artifacts,
        "outcome": {"status": status, "reason": reason},
    }


def _failure_bundle(
    *,
    raw_arm: str,
    raw_seed: str,
    steps: int,
    dry_run: bool,
    confirmation: bool,
    results_root: Path,
    artifact_root: Path,
    reason: str,
    started_utc: str,
) -> Path:
    try:
        seed = max(0, int(str(raw_seed).split(",")[0]))
    except ValueError:
        seed = 0
    arm = _safe_arm(str(raw_arm))
    if "," in str(raw_arm):
        arm = _safe_arm(str(raw_arm))
    ended_utc = _utc_now()
    failure_path = (
        artifact_root
        / "failures"
        / f"mechanism-{arm}-seed{seed}-{uuid.uuid4().hex[:12]}"
        / "failure.json"
    )
    failure_digest = _write_json(
        failure_path,
        {
            "arm": arm,
            "seed": seed,
            "requested_steps": max(0, steps),
            "dry_run": dry_run,
            "confirmation": confirmation,
            "reason": reason,
            "started_utc": started_utc,
            "ended_utc": ended_utc,
        },
    )
    record = _attempt_record(
        attempt_id=new_attempt_id(arm, seed),
        arm=arm,
        seed=seed,
        started_utc=started_utc,
        ended_utc=ended_utc,
        steps=max(0, steps),
        completed_steps=0,
        dry_run=dry_run,
        confirmation=confirmation,
        status="failed",
        reason=reason,
        artifacts=[
            {
                "path": _repo_artifact_path(failure_path),
                "kind": "failure",
                "digest": failure_digest,
            }
        ],
        probe_steps=[],
    )
    return write_native_attempt_bundle(results_root, record)


def _runtime_device_record(device) -> dict[str, Any]:
    import torch

    if device.type == "cuda" and torch.cuda.is_available():
        index = 0 if device.index is None else int(device.index)
        try:
            uuid_value = str(torch.cuda.get_device_properties(index).uuid)
        except AttributeError:
            uuid_value = torch.cuda.get_device_name(index)
        return {"type": "cuda", "index": index, "uuid": uuid_value}
    return {"type": device.type, "index": 0, "uuid": device.type.upper()}


def require_completed_probe_coverage(
    *,
    required_steps: list[int],
    observed_steps: list[int],
    completed_steps: int,
    requested_steps: int,
) -> None:
    if completed_steps != requested_steps:
        raise ValueError(
            f"completed {completed_steps} of requested {requested_steps} "
            "mechanism steps"
        )
    required = set(required_steps)
    observed = set(observed_steps)
    if observed != required:
        missing = sorted(required - observed)
        extra = sorted(observed - required)
        raise ValueError(
            "missing required mechanism probes" f" missing={missing} extra={extra}"
        )


def run_mechanism(
    *,
    arm: str,
    seed: int,
    steps: int,
    dry_run: bool,
    results_root: Path = _DEFAULT_RESULTS_ROOT,
    artifact_root: Path = _DEFAULT_ARTIFACT_ROOT,
    confirmation: bool = False,
    master_port: int = 29591,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    probe_steps = _probe_steps_for_attempt(
        steps, dry_run=dry_run, confirmation=confirmation
    )
    _set_single_process_env(master_port=master_port)
    started = _utc_now()
    attempt_id = new_attempt_id(arm, seed)
    run_dir = mechanism_artifact_dir(artifact_root, attempt_id)
    config = build_mechanism_config(
        arm,
        seed=seed,
        steps=steps,
        dump_folder=run_dir / "trainer",
        dry_run=dry_run,
    )
    import torch

    torch.manual_seed(seed)
    trainer = config.build()
    probes: list[dict[str, Any]] = []
    seen_probe_steps: set[int] = set()

    def collect_probe(step: int) -> None:
        if step in seen_probe_steps:
            return
        probes.append(
            {
                "step": step,
                **_probe_model(trainer.model_parts[0], config, seed=seed),
            }
        )
        seen_probe_steps.add(step)

    try:
        collect_probe(0)
        orig_train_step = trainer.train_step

        def train_step_with_probes(data_iterator):
            result = orig_train_step(data_iterator)
            if trainer.step in probe_steps:
                collect_probe(trainer.step)
            return result

        trainer.train_step = train_step_with_probes  # type: ignore[method-assign]
        trainer.train()
        completed_steps = trainer.step
        if completed_steps in probe_steps:
            collect_probe(completed_steps)
    finally:
        trainer.close()
    ended = _utc_now()
    require_completed_probe_coverage(
        required_steps=probe_steps,
        observed_steps=sorted(seen_probe_steps),
        completed_steps=completed_steps,
        requested_steps=steps,
    )

    diagnostics = {
        "schema_version": 1,
        "arm": arm,
        "seed": seed,
        "requested_steps": steps,
        "completed_steps": completed_steps,
        "probe_steps_required": probe_steps,
        "probes": probes,
    }
    diagnostics_digest = _write_json(
        run_dir / "mechanism_diagnostics.json", diagnostics
    )
    result = {
        "run_id": canonical_native_run_id(
            {"workload": "mechanism", "arm": arm, "seed": seed}
        ),
        "arm": arm,
        "seed": seed,
        "requested_steps": steps,
        "completed_steps": completed_steps,
        "dry_run": dry_run,
        "claim_label": "smoke"
        if dry_run
        else ("confirmation" if confirmation else "mechanism_screen"),
        "probe_steps_required": probe_steps,
        "diagnostics_path": _repo_artifact_path(run_dir / "mechanism_diagnostics.json"),
        "artifact_dir": str(run_dir),
    }
    result_digest = _write_json(run_dir / "mechanism_outcome.json", result)
    record = _attempt_record(
        attempt_id=attempt_id,
        arm=arm,
        seed=seed,
        started_utc=started,
        ended_utc=ended,
        steps=steps,
        completed_steps=completed_steps,
        dry_run=dry_run,
        confirmation=confirmation,
        status="completed",
        reason=None,
        artifacts=[
            {
                "path": _repo_artifact_path(run_dir / "mechanism_diagnostics.json"),
                "kind": "mechanism_diagnostics",
                "digest": diagnostics_digest,
            },
            {
                "path": _repo_artifact_path(run_dir / "mechanism_outcome.json"),
                "kind": "mechanism_outcome",
                "digest": result_digest,
            },
        ],
        probe_steps=probe_steps,
        device=_runtime_device_record(trainer.device),
    )
    bundle_path = write_native_attempt_bundle(results_root, record)
    return {
        **result,
        "attempt_id": record["attempt_id"],
        "bundle_path": str(bundle_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--master-port", type=int, default=29591)
    parser.add_argument("--results-root", type=Path, default=_DEFAULT_RESULTS_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=_DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    started = _utc_now()
    try:
        arm = parse_one_arm(args.arm)
        seed = parse_one_seed(args.seed)
        payload = run_mechanism(
            arm=arm,
            seed=seed,
            steps=args.steps,
            dry_run=args.dry_run,
            confirmation=args.confirmation,
            master_port=args.master_port,
            results_root=args.results_root,
            artifact_root=args.artifact_root,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        try:
            bundle_path = _failure_bundle(
                raw_arm=args.arm,
                raw_seed=args.seed,
                steps=args.steps,
                dry_run=args.dry_run,
                confirmation=args.confirmation,
                results_root=args.results_root,
                artifact_root=args.artifact_root,
                reason=str(exc),
                started_utc=started,
            )
            print(f"wrote failure bundle: {bundle_path}", file=sys.stderr)
        except Exception as bundle_exc:
            print(f"failed to write failure bundle: {bundle_exc}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
