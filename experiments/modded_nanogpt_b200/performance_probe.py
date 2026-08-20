#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run diagnostic performance probes for the modded-nanogpt B200 harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.modded_nanogpt_b200 import cli_guard, preflight


SCHEMA_VERSION = 1
WRAPPER = "experiments/modded_nanogpt_b200/run_performance_probe.sh"
DEFAULT_ENVIRONMENT_CLASS = "torchtitan-rootfs-b200"
SOURCE_VARIANT = "B200-compatible local setup"
PROBE_KINDS = (
    "static_wrapper_preflight",
    "import_construction",
    "one_gpu_microstep",
    "two_gpu_microstep",
    "watchdog_heartbeat",
    "observability_overhead",
)


class ProbeError(Exception):
    def __init__(
        self, phase_name: str, message: str, *, phases: dict[str, Any]
    ) -> None:
        super().__init__(message)
        self.phase_name = phase_name
        self.phases = phases


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _json_sha256(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _phase(phases: dict[str, Any], name: str, func) -> Any:
    start = time.monotonic_ns()
    try:
        detail = func()
    except Exception as exc:
        end = time.monotonic_ns()
        phases[name] = {
            "start_ns": start,
            "end_ns": end,
            "elapsed_ns": end - start,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        raise ProbeError(name, f"{type(exc).__name__}: {exc}", phases=phases) from exc
    end = time.monotonic_ns()
    phases[name] = {
        "start_ns": start,
        "end_ns": end,
        "elapsed_ns": end - start,
        "ok": True,
        "detail": detail,
    }
    return detail


def _parse_gpu_ids(raw: str) -> list[int]:
    if raw.strip() == "":
        return []
    result = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        result.append(int(stripped))
    return result


def _source_commit(source: Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _import_torch_detail() -> dict[str, Any]:
    import torch

    cuda_available = torch.cuda.is_available()
    detail: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": cuda_available,
    }
    if cuda_available:
        start = time.monotonic_ns()
        _ = torch.empty(1, device="cuda")
        if hasattr(torch.cuda, "synchronize"):
            torch.cuda.synchronize()
        end = time.monotonic_ns()
        detail["first_cuda_operation_elapsed_ns"] = end - start
        detail["max_memory_allocated"] = torch.cuda.max_memory_allocated()
    return detail


def _synthetic_cuda_microstep_detail(*, warmup_steps: int = 3) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()

    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 128),
    ).to(device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)

    start_forward = time.monotonic_ns()
    y = model(x)
    torch.cuda.synchronize()
    end_forward = time.monotonic_ns()

    start_backward = time.monotonic_ns()
    loss = (y.float() - target.float()).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    end_backward = time.monotonic_ns()

    start_optimizer = time.monotonic_ns()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    end_optimizer = time.monotonic_ns()

    steady_times = []
    for _ in range(warmup_steps):
        start = time.monotonic_ns()
        y = model(x)
        loss = (y.float() - target.float()).square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        steady_times.append(time.monotonic_ns() - start)

    return {
        "first_forward_elapsed_ns": end_forward - start_forward,
        "first_backward_elapsed_ns": end_backward - start_backward,
        "optimizer_step_elapsed_ns": end_optimizer - start_optimizer,
        "steady_microstep_median_ns": int(median(steady_times)),
        "max_memory_allocated": torch.cuda.max_memory_allocated(),
        "loss": float(loss.detach().cpu()),
    }


def _backend_smoke_detail(source: Path, mlp_backend: str) -> dict[str, Any]:
    if mlp_backend == "torch":
        return preflight._run_torch_mlp_smoke(source)
    if mlp_backend == "triton":
        return preflight._run_triton_mlp_smoke(source)
    raise ValueError(f"unsupported mlp_backend: {mlp_backend}")


def _one_gpu_microstep_detail(source: Path, mlp_backend: str) -> dict[str, Any]:
    detail = _synthetic_cuda_microstep_detail()
    detail["backend_smoke"] = _backend_smoke_detail(source, mlp_backend)
    return detail


def _two_gpu_microstep_detail(
    source: Path, mlp_backend: str, world_size: int
) -> dict[str, Any]:
    if world_size != 2:
        raise ValueError("two_gpu_microstep requires --world-size 2")
    preflight.check_nccl(world_size)
    return {
        "declared_world_size": world_size,
        "nccl_all_reduce": "passed",
        "per_rank_microstep": "rank0_synthetic_cuda_microstep",
        "rank0_detail": _one_gpu_microstep_detail(source, mlp_backend),
    }


def _watchdog_heartbeat_detail() -> dict[str, Any]:
    before_compile = time.monotonic_ns()
    compile_progress = preflight.run_checked_subprocess(
        [
            sys.executable,
            "-c",
            "import time; print('heartbeat: before_compile', flush=True); "
            "time.sleep(0.1); print('heartbeat: during_warmup', flush=True)",
        ],
        timeout_seconds=10,
    )
    after_warmup = time.monotonic_ns()
    heartbeats = [
        line.strip().removeprefix("heartbeat: ")
        for line in compile_progress.stdout.splitlines()
        if line.strip().startswith("heartbeat:")
    ]
    classification = "forward_progress" if heartbeats else "no_progress"
    recommendation = (
        "raise_watchdog_with_measured_compile_evidence"
        if classification == "forward_progress"
        else "keep_watchdog_or_repair_progress_signal"
    )
    return {
        "heartbeats": heartbeats,
        "classification": classification,
        "recommendation": recommendation,
        "elapsed_ns": after_warmup - before_compile,
    }


def _observability_overhead_detail(profile: str) -> dict[str, Any]:
    start_tier0 = time.monotonic_ns()
    tier0_env = preflight.collect_environment_detail()
    tier0_elapsed = time.monotonic_ns() - start_tier0

    start_profile = time.monotonic_ns()
    profile_env = preflight.collect_environment_detail()
    profile_elapsed = time.monotonic_ns() - start_profile

    return {
        "baseline_profile": "tier0",
        "comparison_profile": profile,
        "phase_timing_delta_ns": {
            "environment_capture": profile_elapsed - tier0_elapsed,
        },
        "artifact_bytes_delta": len(json.dumps(profile_env, sort_keys=True))
        - len(json.dumps(tier0_env, sort_keys=True)),
        "missing_evidence": [],
    }


def _classification(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "lane": "B",
        "mode": "diagnostic",
        "arm": args.probe_name,
        "claim_label": "diagnostic",
        "evidence_tier": "diagnostic",
        "run_id": args.run_id,
        "attempt_id": args.attempt_id,
        "environment_class": DEFAULT_ENVIRONMENT_CLASS,
        "claim_eligible": False,
    }


def _attempt_command(args: argparse.Namespace) -> list[str]:
    return [
        WRAPPER,
        "--probe-name",
        args.probe_name,
        "--probe-kind",
        args.probe_kind,
        "--result-dir",
        str(args.result_dir),
        "--source",
        str(args.source),
        "--data-manifest",
        str(args.data_manifest),
        "--attention-backend",
        args.attention_backend,
        "--mlp-backend",
        args.mlp_backend,
        "--run-id",
        args.run_id,
        "--attempt-id",
        args.attempt_id,
        "--gpu-ids",
        args.gpu_ids,
        "--world-size",
        str(args.world_size),
        "--observability-profile",
        args.observability_profile,
    ]


def _artifact_bytes(path: Path) -> int:
    total = 0
    for candidate in path.rglob("*"):
        if candidate.is_file():
            total += candidate.stat().st_size
    return total


def _base_probe(args: argparse.Namespace, phases: dict[str, Any]) -> dict[str, Any]:
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_name": args.probe_name,
        "probe_kind": args.probe_kind,
        "run_id": args.run_id,
        "attempt_id": args.attempt_id,
        "mode": "diagnostic",
        "claim_eligible": False,
        "source": str(args.source),
        "source_commit": _source_commit(args.source),
        "source_variant": SOURCE_VARIANT,
        "data_manifest": str(args.data_manifest),
        "attention_backend": args.attention_backend,
        "mlp_backend": args.mlp_backend,
        "visible_gpu_ids": gpu_ids,
        "declared_world_size": args.world_size,
        "observability_profile": args.observability_profile,
        "phases": phases,
    }


def _write_attempt_artifacts(
    args: argparse.Namespace,
    classification: dict[str, Any],
    probe: dict[str, Any],
) -> None:
    argv_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "command_argv",
        "run_id": args.run_id,
        "attempt_id": args.attempt_id,
        "argv": _attempt_command(args),
        "training_argv": [],
    }
    argv_payload["argv_digest"] = {
        "algorithm": "sha256",
        "sha256": _json_sha256(argv_payload),
    }
    env_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "command_environment",
        "run_id": args.run_id,
        "attempt_id": args.attempt_id,
        "environment": {
            "TORCHTITAN_IN_ROOTFS": os.environ.get("TORCHTITAN_IN_ROOTFS", ""),
        },
    }
    env_payload["environment_digest"] = {
        "algorithm": "sha256",
        "sha256": _json_sha256(env_payload),
    }
    attempt = {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "source": str(args.source),
        "data_manifest": str(args.data_manifest),
        "command": {
            "argv": _attempt_command(args),
            "skip_run": True,
            "launch_authorization_present": False,
            "attention_backend": args.attention_backend,
            "mlp_backend": args.mlp_backend,
        },
        "gpu_topology": {
            "num_gpus": len(_parse_gpu_ids(args.gpu_ids)),
            "gpu_ids": _parse_gpu_ids(args.gpu_ids),
            "visible_devices": args.gpu_ids,
        },
    }
    _write_json_atomic(args.result_dir / "attempt.json", attempt)
    _write_json_atomic(args.result_dir / "command.argv.json", argv_payload)
    _write_json_atomic(args.result_dir / "command.env.json", env_payload)


def _write_summary(
    args: argparse.Namespace,
    classification: dict[str, Any],
    probe: dict[str, Any],
    *,
    status: str,
    blocker: dict[str, str] | None = None,
) -> None:
    summary = {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "metrics": {},
        "probe": {
            "status": status,
            "probe_name": args.probe_name,
            "probe_kind": args.probe_kind,
            "artifact_bytes": probe["artifact_bytes"],
        },
    }
    if blocker is not None:
        summary["blocker"] = blocker
    _write_json_atomic(args.result_dir / "summary.json", summary)


def _write_failure_artifacts(
    args: argparse.Namespace,
    phase_name: str,
    message: str,
    phases: dict[str, Any],
) -> dict[str, Any]:
    classification = _classification(args)
    probe = _base_probe(args, phases)
    blocker = {"phase": phase_name, "message": message}
    probe.update(
        {
            "ok": False,
            "classification": classification,
            "blocker": blocker,
        }
    )
    _write_json_atomic(args.result_dir / "probe.json", probe)
    probe["artifact_bytes"] = _artifact_bytes(args.result_dir)
    _write_json_atomic(args.result_dir / "probe.json", probe)
    _write_attempt_artifacts(args, classification, probe)
    probe["artifact_bytes"] = _artifact_bytes(args.result_dir)
    _write_json_atomic(args.result_dir / "probe.json", probe)
    _write_summary(args, classification, probe, status="failed", blocker=blocker)
    return probe


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.result_dir.exists() and any(args.result_dir.iterdir()):
        raise ValueError(
            f"result-dir already exists and is non-empty: {args.result_dir}"
        )
    args.result_dir.mkdir(parents=True, exist_ok=True)
    phases: dict[str, Any] = {}
    _phase(
        phases,
        "rootfs_context",
        lambda: (preflight.check_rootfs(), preflight.collect_rootfs_detail())[1],
    )
    _phase(phases, "environment_capture", preflight.collect_environment_detail)
    if args.probe_kind == "import_construction":
        _phase(phases, "torch_import", _import_torch_detail)
    elif args.probe_kind == "one_gpu_microstep":
        _phase(
            phases,
            "one_gpu_microstep",
            lambda: _one_gpu_microstep_detail(args.source, args.mlp_backend),
        )
    elif args.probe_kind == "two_gpu_microstep":
        _phase(
            phases,
            "two_gpu_microstep",
            lambda: _two_gpu_microstep_detail(
                args.source, args.mlp_backend, args.world_size
            ),
        )
    elif args.probe_kind == "watchdog_heartbeat":
        _phase(phases, "watchdog_heartbeat", _watchdog_heartbeat_detail)
    elif args.probe_kind == "observability_overhead":
        _phase(
            phases,
            "observability_overhead",
            lambda: _observability_overhead_detail(args.observability_profile),
        )
    elif args.probe_kind != "static_wrapper_preflight":
        raise ValueError(f"unsupported probe-kind: {args.probe_kind}")
    probe = _base_probe(args, phases)
    _write_json_atomic(args.result_dir / "probe.json", probe)
    probe["artifact_bytes"] = _artifact_bytes(args.result_dir)
    _write_json_atomic(args.result_dir / "probe.json", probe)
    _write_attempt_artifacts(args, _classification(args), probe)
    probe["artifact_bytes"] = _artifact_bytes(args.result_dir)
    _write_json_atomic(args.result_dir / "probe.json", probe)
    _write_summary(args, _classification(args), probe, status="passed")
    return probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-name", required=True)
    parser.add_argument(
        "--probe-kind",
        required=True,
        choices=PROBE_KINDS,
    )
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument(
        "--attention-backend", required=True, choices=["fa3", "fa2", "flex"]
    )
    parser.add_argument("--mlp-backend", required=True, choices=["triton", "torch"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--observability-profile", default="tier0")
    return parser.parse_args()


def main(*, enforce_rootfs: bool = False) -> int:
    args = parse_args()
    if enforce_rootfs:
        guard_exit = cli_guard.guard_rootfs_cli(WRAPPER)
        if guard_exit is not None:
            return guard_exit
    try:
        probe = run_probe(args)
    except ProbeError as exc:
        args.result_dir.mkdir(parents=True, exist_ok=True)
        report = _write_failure_artifacts(args, exc.phase_name, str(exc), exc.phases)
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 21
    except Exception as exc:  # noqa: BLE001
        args.result_dir.mkdir(parents=True, exist_ok=True)
        report = _write_failure_artifacts(
            args,
            "probe",
            f"{type(exc).__name__}: {exc}",
            {},
        )
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 21
    print(json.dumps(probe, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(enforce_rootfs=True))
