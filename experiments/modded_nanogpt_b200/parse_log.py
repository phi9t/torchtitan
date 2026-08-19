#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parse modded-nanogpt B200 attempt logs into a conservative summary."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.modded_nanogpt_b200 import cli_guard


SCHEMA_VERSION = 1
UPSTREAM_COMMIT = "ecbb586296d3dac36fd206211f25d63bad4a6b35"
EXPECTED_FINEWEB_SHARDS = 10
EXPECTED_FINEWEB_BYTES = 2000010240
NULL_FINAL_METRICS = {
    "val_loss": None,
    "train_time": None,
    "step_avg": None,
    "peak_allocated_memory": None,
    "peak_reserved_memory": None,
}

_FLOAT = r"([0-9]+(?:\.[0-9]+)?)"
_VAL_LOSS_PATTERNS = [
    re.compile(
        rf"(?:final[\s_-]+)?val(?:idation)?(?:[\s_-]+loss)?\s*[:=]\s*{_FLOAT}",
        re.I,
    ),
    re.compile(rf"final\s+validation\s+loss\s*[:=]\s*{_FLOAT}", re.I),
]
_TRAIN_TIME_PATTERNS = [
    re.compile(rf"\btrain[_\s-]*time\s*[:=]\s*{_FLOAT}\s*([a-z]+)?", re.I),
]
_STEP_AVG_PATTERNS = [
    re.compile(rf"\bstep[_\s-]*avg\s*[:=]\s*{_FLOAT}\s*([a-z]+)?", re.I),
]
_PEAK_ALLOCATED_PATTERNS = [
    re.compile(rf"peak\s+memory\s+allocated\s*[:=]\s*{_FLOAT}\s*([kmgt]?i?b)", re.I),
    re.compile(rf"peak\s+allocated\s+memory\s*[:=]\s*{_FLOAT}\s*([kmgt]?i?b)", re.I),
]
_PEAK_RESERVED_PATTERNS = [
    re.compile(rf"peak\s+memory\s+reserved\s*[:=]\s*{_FLOAT}\s*([kmgt]?i?b)", re.I),
    re.compile(rf"peak\s+reserved\s+memory\s*[:=]\s*{_FLOAT}\s*([kmgt]?i?b)", re.I),
    re.compile(
        rf"peak\s+memory\s+allocated\s*[:=]\s*{_FLOAT}\s*[kmgt]?i?b\s+reserved\s*[:=]?\s*{_FLOAT}\s*([kmgt]?i?b)",
        re.I,
    ),
]
_RELEVANT_ERROR_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in (
        r"\btraceback\b",
        r"\berror\b",
        r"\bfailed\b",
        r"\bexception\b",
        r"\babort(?:ed)?\b",
        r"\bout of memory\b",
        r"\boom\b",
        r"\bkernel image\b",
        r"\bnccl\b",
        r"\bcommunicator\b",
        r"\brendezvous\b",
        r"\brank 0 exited\b",
        r"\bexited with code\b",
        r"\btimeout\b",
        r"\bstall(?:ed|ing)?\b",
    )
]


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _last_float(patterns: list[re.Pattern[str]], text: str) -> float | None:
    value = None
    for pattern in patterns:
        for match in pattern.finditer(text):
            value = float(match.group(1))
    return value


def _last_time_seconds(patterns: list[re.Pattern[str]], text: str) -> float | None:
    value = None
    for pattern in patterns:
        for match in pattern.finditer(text):
            amount = float(match.group(1))
            unit = match.group(2).lower() if match.lastindex and match.group(2) else ""
            if unit in {"ms", "millisecond", "milliseconds"}:
                amount /= 1000
            value = amount
    return value


def _last_memory_gib(patterns: list[re.Pattern[str]], text: str) -> float | None:
    value = None
    for pattern in patterns:
        for match in pattern.finditer(text):
            if pattern.groups >= 3:
                amount = float(match.group(2))
                unit = match.group(3)
            else:
                amount = float(match.group(1))
                unit = match.group(2)
            unit = unit.lower()
            if unit in {"kib", "kb"}:
                amount /= 1024 * 1024
            elif unit in {"mib", "mb"}:
                amount /= 1024
            elif unit in {"tib", "tb"}:
                amount *= 1024
            value = amount
    return value


def _source_status(source_path: Path, result_dir: Path) -> dict[str, Any]:
    captured = _read_json(result_dir / "source.json")
    captured_after_path = result_dir / "source_status_after.txt"
    captured_after = (
        captured_after_path.read_text() if captured_after_path.exists() else None
    )
    if not source_path.exists():
        return {
            "path": str(source_path),
            "exists": False,
            "dirty": None,
            "status": captured_after,
            "commit": captured.get("commit"),
            "status_before": captured.get("status_before"),
        }
    if captured_after is not None:
        return {
            "path": str(source_path),
            "exists": True,
            "dirty": bool(captured_after.strip()),
            "status": captured_after,
            "commit": captured.get("commit"),
            "status_before": captured.get("status_before"),
        }
    try:
        status = subprocess.check_output(
            ["git", "-C", str(source_path), "status", "--short"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "path": str(source_path),
            "exists": True,
            "dirty": None,
            "status": None,
            "error": f"{type(exc).__name__}: {exc}",
            "commit": captured.get("commit"),
            "status_before": captured.get("status_before"),
        }
    return {
        "path": str(source_path),
        "exists": True,
        "dirty": bool(status.strip()),
        "status": status,
        "commit": captured.get("commit"),
        "status_before": captured.get("status_before"),
    }


def _variant_patch_classification(result_dir: Path) -> dict[str, Any]:
    return _read_json(result_dir / "variant_patch_classification.json")


def _launch_readiness(result_dir: Path) -> dict[str, Any]:
    return _read_json(result_dir / "launch_readiness.json")


def _attempt_command(result_dir: Path) -> dict[str, Any]:
    attempt = _read_json(result_dir / "attempt.json")
    command = attempt.get("command")
    return command if isinstance(command, dict) else {}


def _environment_sidecar(result_dir: Path) -> dict[str, Any]:
    sidecar = _read_json(result_dir / "environment.json")
    if sidecar.get("kind") != "preflight_environment":
        return {}
    environment = sidecar.get("environment")
    if not isinstance(environment, dict):
        return {}
    return sidecar


def _hardware_sidecar(result_dir: Path) -> dict[str, Any]:
    sidecar = _read_json(result_dir / "hardware.json")
    if sidecar.get("kind") != "preflight_gpus":
        return {}
    gpus = sidecar.get("gpus")
    if not isinstance(gpus, list):
        return {}
    return sidecar


def _phase_for_line(line: str) -> str:
    lowered = line.lower()
    if "validation" in lowered:
        return "validation"
    if "nccl" in lowered or "rendezvous" in lowered or "communicator" in lowered:
        return "nccl"
    if "out of memory" in lowered or "oom" in lowered:
        return "warmup"
    if "triton" in lowered or "compile" in lowered or "torchdynamo" in lowered:
        return "compile"
    if (
        "cuda" in lowered
        or "kernel image" in lowered
        or "illegal instruction" in lowered
    ):
        return "kernel"
    if "dataloader" in lowered or "fineweb" in lowered:
        return "data"
    return "unknown"


def _first_relevant_error(lines: list[str]) -> str | None:
    for line in lines:
        if any(pattern.search(line) for pattern in _RELEVANT_ERROR_PATTERNS):
            return line
    return None


def _blocker_for_incomplete(text: str, first_error: str | None) -> dict[str, str]:
    for line in text.splitlines():
        if "skip-run requested; training was not launched" in line:
            return {
                "phase": "not_launched",
                "message": "skip-run requested; training was not launched",
            }
    message = first_error or "final validation metrics are missing or not claim-valid"
    return {"phase": _phase_for_line(message), "message": message}


def _failure_category(blocker: dict[str, Any]) -> str:
    phase = str(blocker.get("phase") or "unknown")
    message = str(blocker.get("message") or "")
    lowered = f"{phase} {message}".lower()
    if "rootfs" in lowered:
        return "rootfs"
    if "dependency" in lowered or "package" in lowered or "import" in lowered:
        return "dependency"
    if (
        "source" in lowered
        or "variant_patch" in lowered
        or "dirty" in lowered
        or "commit" in lowered
    ):
        return "source policy"
    if "data" in lowered or "manifest" in lowered or "fineweb" in lowered:
        return "data"
    if (
        "flash" in lowered
        or "attention_backend" in lowered
        or "fa2" in lowered
        or "fa3" in lowered
    ):
        return "FA"
    if "mlp" in lowered:
        return "MLP"
    if "nccl" in lowered or "communicator" in lowered or "rendezvous" in lowered:
        return "NCCL"
    if "compile" in lowered or "triton" in lowered or "torchdynamo" in lowered:
        return "compile"
    if "warmup" in lowered or "out of memory" in lowered or "oom" in lowered:
        return "warmup"
    if "train" in lowered or "training" in lowered:
        return "train"
    if "validation" in lowered:
        return "validation"
    if "parser" in lowered or "parse" in lowered:
        return "parser"
    if "telemetry" in lowered or "watcher" in lowered:
        return "telemetry"
    if phase not in {"unknown", ""}:
        return phase
    return _phase_for_line(message)


def _preflight_failure_blocker(preflight: dict[str, Any]) -> dict[str, str] | None:
    failures = preflight.get("failures")
    if not isinstance(failures, list) or not failures:
        return None
    first = failures[0]
    if not isinstance(first, dict):
        return None
    phase = first.get("name")
    message = first.get("error")
    if not phase or not message:
        return None
    return {"phase": str(phase), "message": str(message)}


def _runner_blocker(result_dir: Path) -> dict[str, str] | None:
    data = _read_json(result_dir / "blocker.json")
    phase = data.get("phase")
    message = data.get("message")
    if not phase or not message:
        return None
    return {"phase": str(phase), "message": str(message)}


def _runtime_blocker(
    result_dir: Path, text: str, first_error: str | None
) -> dict[str, str]:
    inferred = _blocker_for_incomplete(text, first_error)
    runner = _runner_blocker(result_dir)
    if runner is None:
        return inferred
    if runner["phase"] != "training":
        return runner
    if (
        inferred["phase"] == "validation"
        and inferred["message"]
        == "final validation metrics are missing or not claim-valid"
    ):
        return runner
    return inferred


def _data_manifest_summary(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    kind = manifest.get("kind")
    resolved_path: Path | None = None
    resolved_manifest = manifest
    verified_sha = manifest.get("verified_sha")
    if kind == "data_manifest_pointer":
        pointer_path = manifest.get("path")
        if isinstance(pointer_path, str) and pointer_path:
            resolved_path = Path(pointer_path)
            if not resolved_path.is_absolute():
                repo_relative_path = REPO_ROOT / resolved_path
                if repo_relative_path.exists():
                    resolved_path = repo_relative_path
                else:
                    resolved_path = path.parent / resolved_path
            resolved_manifest = _read_json(resolved_path)
            if "verified_sha" not in manifest:
                verified_sha = resolved_manifest.get("verified_sha")
    files = manifest.get("files", [])
    if kind == "data_manifest_pointer":
        files = resolved_manifest.get("files", [])
    source = resolved_manifest.get("source", {})
    sha_entries = 0
    if isinstance(files, list):
        sha_entries = sum(
            1 for item in files if isinstance(item, dict) and item.get("sha256")
        )
    return {
        "path": str(path),
        "exists": path.exists(),
        "kind": kind,
        "resolved_path": str(resolved_path) if resolved_path is not None else None,
        "resolved_exists": resolved_path.exists()
        if resolved_path is not None
        else None,
        "schema_version": resolved_manifest.get("schema_version"),
        "dataset": resolved_manifest.get("dataset"),
        "token_budget": resolved_manifest.get("token_budget"),
        "source_commit": source.get("commit") if isinstance(source, dict) else None,
        "verified_sha": verified_sha,
        "num_files": resolved_manifest.get(
            "num_files", len(files) if isinstance(files, list) else None
        ),
        "total_bytes": resolved_manifest.get("total_bytes"),
        "sha256_entries": sha_entries,
    }


def _manifest_path_id(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        repo_relative_path = REPO_ROOT / path
        if repo_relative_path.exists():
            path = repo_relative_path
    return str(path)


def _preflight_check_detail(preflight: dict[str, Any], name: str) -> dict[str, Any]:
    checks = preflight.get("checks")
    if not isinstance(checks, list):
        return {}
    for check in checks:
        if not isinstance(check, dict) or check.get("name") != name:
            continue
        detail = check.get("detail")
        return detail if isinstance(detail, dict) else {}
    return {}


def _float_value(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"-?[0-9]+(?:\.[0-9]+)?", value)
    if not match:
        return None
    return float(match.group(0))


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _read_nvidia_smi_query(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.read_text().splitlines(), skipinitialspace=True))
    gpu_utilization = []
    memory_utilization = []
    power_draw = []
    temperature = []
    memory_used = []
    clocks_sm = []
    for row in rows:
        gpu_utilization.append(_float_value(row.get("utilization.gpu [%]")))
        memory_utilization.append(_float_value(row.get("utilization.memory [%]")))
        power_draw.append(_float_value(row.get("power.draw [W]")))
        temperature.append(_float_value(row.get("temperature.gpu")))
        memory_used.append(_float_value(row.get("memory.used [MiB]")))
        clocks_sm.append(_float_value(row.get("clocks.sm [MHz]")))
    clean_gpu_utilization = [value for value in gpu_utilization if value is not None]
    clean_memory_utilization = [
        value for value in memory_utilization if value is not None
    ]
    clean_power_draw = [value for value in power_draw if value is not None]
    clean_temperature = [value for value in temperature if value is not None]
    clean_memory_used = [value for value in memory_used if value is not None]
    clean_clocks_sm = [value for value in clocks_sm if value is not None]
    return {
        "samples": len(rows),
        "gpu_utilization": _stats(clean_gpu_utilization),
        "memory_utilization": _stats(clean_memory_utilization),
        "power_draw_watts": _stats(clean_power_draw),
        "temperature_celsius": _stats(clean_temperature),
        "memory_used_mib": {"peak": max(clean_memory_used)}
        if clean_memory_used
        else None,
        "sm_clock_mhz": _stats(clean_clocks_sm),
    }


def _read_process_watch(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    peak_rss_kib = None
    peak_cpu_percent = None
    for line in path.read_text().splitlines():
        parts = line.split(maxsplit=8)
        if len(parts) < 8:
            continue
        rss_kib = _float_value(parts[4])
        cpu_percent = _float_value(parts[5])
        if rss_kib is not None:
            peak_rss_kib = (
                rss_kib if peak_rss_kib is None else max(peak_rss_kib, rss_kib)
            )
        if cpu_percent is not None:
            peak_cpu_percent = (
                cpu_percent
                if peak_cpu_percent is None
                else max(peak_cpu_percent, cpu_percent)
            )
    return {
        "peak_rss_mib": peak_rss_kib / 1024 if peak_rss_kib is not None else None,
        "peak_cpu_percent": peak_cpu_percent,
    }


def _size_to_bytes(value: str) -> int | None:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTPE]?)(i?B?)?", value, re.I)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).upper()
    scale = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
        "E": 1024**6,
    }[unit]
    return int(amount * scale)


def _read_disk_watch(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    result_sizes = []
    data_sizes = []
    for line in path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        size = _size_to_bytes(parts[0])
        if size is None:
            continue
        target = parts[1]
        if "/results/" in target or target.endswith("/results"):
            result_sizes.append(size)
        else:
            data_sizes.append(size)
    return {
        "max_result_size_bytes": max(result_sizes) if result_sizes else None,
        "max_data_size_bytes": max(data_sizes) if data_sizes else None,
    }


def _telemetry_signs(
    ranges: dict[str, Any], process_watch: dict[str, Any]
) -> dict[str, Any]:
    temperature = ranges.get("temperature_celsius")
    sm_clock = ranges.get("sm_clock_mhz")
    max_temperature = (
        temperature.get("max") if isinstance(temperature, dict) else None
    )
    min_sm_clock = sm_clock.get("min") if isinstance(sm_clock, dict) else None
    thermal_reasons = []
    if isinstance(max_temperature, int | float) and max_temperature >= 85:
        thermal_reasons.append(f"max GPU temperature {max_temperature}C")
    if isinstance(min_sm_clock, int | float) and min_sm_clock <= 1000:
        thermal_reasons.append(f"minimum SM clock {min_sm_clock}MHz")

    peak_cpu_percent = process_watch.get("peak_cpu_percent")
    peak_rss_mib = process_watch.get("peak_rss_mib")
    process_reasons = []
    if isinstance(peak_cpu_percent, int | float) and peak_cpu_percent >= 800:
        process_reasons.append(f"peak process CPU {peak_cpu_percent}%")
    if isinstance(peak_rss_mib, int | float) and peak_rss_mib >= 32768:
        process_reasons.append(f"peak process RSS {peak_rss_mib}MiB")

    return {
        "thermal_or_clock_throttling": bool(thermal_reasons),
        "thermal_or_clock_throttling_reason": "; ".join(thermal_reasons) or None,
        "cpu_or_rss_bottleneck": bool(process_reasons),
        "cpu_or_rss_bottleneck_reason": "; ".join(process_reasons) or None,
    }


def _telemetry_summary(result_dir: Path) -> dict[str, Any]:
    telemetry_dir = result_dir / "telemetry"
    watcher_status = _read_json(telemetry_dir / "watcher_status.json")
    rootfs_environment = _read_json(telemetry_dir / "rootfs_environment.json")
    dcgm_status = _read_json(telemetry_dir / "dcgm_status.json")
    stop_snapshot = _read_json(telemetry_dir / "stop_snapshot.json")
    ranges = _read_nvidia_smi_query(telemetry_dir / "nvidia_smi_query.csv")
    process_watch = _read_process_watch(telemetry_dir / "process_watch.log")
    return {
        "path": str(telemetry_dir),
        "exists": telemetry_dir.exists(),
        "state": watcher_status.get("state"),
        "reason": watcher_status.get("reason"),
        "rootfs": {
            "torchtitan_in_rootfs": rootfs_environment.get("torchtitan_in_rootfs"),
            "cwd": rootfs_environment.get("cwd"),
            "python_executable": rootfs_environment.get("python_executable"),
            "workspace_sentinel_exists": rootfs_environment.get(
                "workspace_sentinel_exists"
            ),
        },
        "ranges": ranges,
        "process_watch": process_watch,
        "disk_watch": _read_disk_watch(telemetry_dir / "disk_watch.log"),
        "signs": _telemetry_signs(ranges, process_watch),
        "dcgm": dcgm_status,
        "stop_snapshot": stop_snapshot,
    }


def _artifact_sizes(result_dir: Path) -> dict[str, Any]:
    relative_paths = (
        "attempt.json",
        "command.argv",
        "command.env",
        "operator_notes.md",
        "preflight_report.json",
        "run.log",
        "wall_clock.json",
        "exit_code.json",
        "blocker.json",
        "source.json",
        "source_status_before.txt",
        "source_status_after.txt",
        "data_manifest.json",
        "environment.json",
        "hardware.json",
        "variant_patch.diff",
        "variant_patch_classification.json",
        "launch_readiness.json",
        "launch_readiness.md",
        "telemetry/rootfs_environment.json",
        "telemetry/watcher_status.json",
        "telemetry/dcgm_status.json",
        "telemetry/nvidia_smi_query.csv",
        "telemetry/nvidia_smi_dmon.csv",
        "telemetry/process_watch.log",
        "telemetry/disk_watch.log",
        "telemetry/gpu_processes.log",
        "telemetry/dcgm_dmon.log",
        "telemetry/stop_ps_tree.txt",
        "telemetry/stop_nvidia_smi_processes.txt",
        "telemetry/stop_snapshot.json",
    )
    files = {}
    for relative_path in relative_paths:
        path = result_dir / relative_path
        if path.is_file():
            files[relative_path] = path.stat().st_size
    return {
        "files": files,
        "total_known_bytes": sum(files.values()),
    }


def _meaningful_tail(lines: list[str], count: int = 50) -> list[str]:
    meaningful = [line for line in lines if line.strip()]
    return meaningful[-count:]


def _claim_label(lane: str, mode: str) -> str:
    if mode == "smoke":
        return "smoke"
    if mode == "diagnostic":
        return "diagnostic"
    if lane == "A":
        return "B200 upstream reproduction"
    return "B200 compatibility patchset"


def _evidence_tier(mode: str) -> str:
    if mode == "full":
        return "full-single-attempt"
    return mode


def _default_classification(
    preflight: dict[str, Any],
    classification_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    classification = preflight.get("classification")
    if isinstance(classification, dict):
        base = dict(classification)
    else:
        base = {
            "lane": "A",
            "mode": "diagnostic",
            "arm": "A0",
            "claim_label": "diagnostic",
            "evidence_tier": "diagnostic",
            "run_id": "unknown",
            "attempt_id": "unknown",
            "environment_class": "torchtitan-rootfs-b200",
            "claim_eligible": False,
        }
    if classification_override:
        base.update(
            {
                key: value
                for key, value in classification_override.items()
                if value is not None
            }
        )
    lane = str(base.get("lane", "A"))
    mode = str(base.get("mode", "diagnostic"))
    base["claim_label"] = base.get("claim_label") or _claim_label(lane, mode)
    base["evidence_tier"] = base.get("evidence_tier") or _evidence_tier(mode)
    base["environment_class"] = (
        base.get("environment_class") or "torchtitan-rootfs-b200"
    )
    base["claim_eligible"] = (
        bool(base.get("claim_eligible")) and lane in {"A", "B"} and mode == "full"
    )
    if classification_override:
        base["claim_label"] = base.get("claim_label") or _claim_label(lane, mode)
        base["evidence_tier"] = base.get("evidence_tier") or _evidence_tier(mode)
        base["claim_eligible"] = (
            bool(base.get("claim_eligible")) and lane in {"A", "B"} and mode == "full"
        )
    return base


def _final_metrics(text: str) -> dict[str, float | None]:
    val_loss = _last_float(_VAL_LOSS_PATTERNS, text)
    train_time = _last_time_seconds(_TRAIN_TIME_PATTERNS, text)
    step_avg = _last_time_seconds(_STEP_AVG_PATTERNS, text)
    if val_loss is None or train_time is None or step_avg is None:
        return dict(NULL_FINAL_METRICS)
    return {
        "val_loss": val_loss,
        "train_time": train_time,
        "step_avg": step_avg,
        "peak_allocated_memory": _last_memory_gib(_PEAK_ALLOCATED_PATTERNS, text),
        "peak_reserved_memory": _last_memory_gib(_PEAK_RESERVED_PATTERNS, text),
    }


def _source_policy_blocker(
    classification: dict[str, Any], source: dict[str, Any]
) -> dict[str, str] | None:
    if classification.get("lane") != "A":
        return None
    if source.get("dirty") is None:
        return {
            "phase": "source_policy",
            "message": "Lane A source cleanliness evidence is missing",
        }
    if source.get("dirty") is True:
        return {
            "phase": "source_policy",
            "message": "Lane A source was dirty after launch",
        }
    return None


def _variant_patch_policy_blocker(
    classification: dict[str, Any],
    variant_patch: dict[str, Any],
) -> dict[str, str] | None:
    if classification.get("lane") != "B" or classification.get("mode") != "full":
        return None
    files = variant_patch.get("files")
    if not isinstance(files, list):
        return None
    paths = []
    for item in files:
        if not isinstance(item, dict):
            continue
        if item.get("patch_class") != "unclassified":
            continue
        path = item.get("path")
        if isinstance(path, str):
            paths.append(path)
    if not paths:
        return None
    return {
        "phase": "variant_patch_classification",
        "message": "Lane B baseline has unclassified source patches: "
        + ", ".join(paths),
    }


def _full_attempt_evidence_blocker(
    classification: dict[str, Any],
    *,
    launch_readiness: dict[str, Any],
    exit_code: dict[str, Any],
    telemetry: dict[str, Any],
    data_manifest: dict[str, Any],
    preflight_data_manifest: dict[str, Any],
    gpus: list[Any],
) -> dict[str, str] | None:
    if classification.get("mode") != "full":
        return None
    full_mode_gates = launch_readiness.get("full_mode_gates")
    if not isinstance(full_mode_gates, dict):
        full_mode_gates = {}
    if launch_readiness.get("training_launched") is not True:
        return {
            "phase": "launch_evidence",
            "message": "full baseline requires launch_readiness.training_launched=true",
        }
    if launch_readiness.get("skip_run") is True:
        return {
            "phase": "launch_evidence",
            "message": "full baseline cannot come from a skip-run attempt",
        }
    if launch_readiness.get("allow_previous_stall") is True:
        return {
            "phase": "launch_evidence",
            "message": "full baseline cannot use known-stall override",
        }
    if launch_readiness.get("preflight_ok") is not True:
        return {
            "phase": "preflight",
            "message": "full baseline requires launch_readiness.preflight_ok=true",
        }
    if exit_code.get("phase") != "training" or exit_code.get("exit_code") != 0:
        return {
            "phase": "training",
            "message": "full baseline requires training exit_code=0, found "
            f"{exit_code.get('exit_code')}",
        }
    rootfs = telemetry.get("rootfs")
    if not isinstance(rootfs, dict):
        rootfs = {}
    if (
        rootfs.get("torchtitan_in_rootfs") != "1"
        or rootfs.get("cwd") != "/workspace/torchtitan"
        or rootfs.get("workspace_sentinel_exists") is not True
    ):
        return {
            "phase": "rootfs",
            "message": "full baseline requires rootfs sentinel evidence",
        }
    if full_mode_gates.get("verify_sha_requested") is not True:
        return {
            "phase": "data_manifest",
            "message": "full baseline requires SHA verification to be requested",
        }
    if full_mode_gates.get("data_manifest_checked") is not True:
        return {
            "phase": "data_manifest",
            "message": "full baseline requires data manifest preflight evidence",
        }
    readiness_manifest = launch_readiness.get("data_manifest")
    accepted_manifest_paths = {
        _manifest_path_id(value)
        for value in (
            data_manifest.get("path"),
            data_manifest.get("resolved_path"),
        )
        if isinstance(value, str) and value
    }
    if (
        isinstance(readiness_manifest, str)
        and readiness_manifest
        and _manifest_path_id(readiness_manifest) not in accepted_manifest_paths
    ):
        return {
            "phase": "data_manifest",
            "message": "full baseline requires launch_readiness.data_manifest to match parsed manifest",
        }
    if (
        full_mode_gates.get("verified_sha") is not True
        or preflight_data_manifest.get("verified_sha") is not True
        or data_manifest.get("verified_sha") is not True
    ):
        return {
            "phase": "data_manifest",
            "message": "full baseline requires SHA-verified data manifest evidence",
        }
    if (
        full_mode_gates.get("manifest_token_budget") != "900M"
        or data_manifest.get("token_budget") != "900M"
    ):
        return {
            "phase": "data_manifest",
            "message": "full baseline requires a 900M FineWeb manifest",
        }
    if (
        full_mode_gates.get("manifest_num_files") != EXPECTED_FINEWEB_SHARDS
        or data_manifest.get("num_files") != EXPECTED_FINEWEB_SHARDS
    ):
        return {
            "phase": "data_manifest",
            "message": f"full baseline requires {EXPECTED_FINEWEB_SHARDS} FineWeb shards",
        }
    if (
        full_mode_gates.get("manifest_total_bytes") != EXPECTED_FINEWEB_BYTES
        or data_manifest.get("total_bytes") != EXPECTED_FINEWEB_BYTES
    ):
        return {
            "phase": "data_manifest",
            "message": f"full baseline requires FineWeb total_bytes={EXPECTED_FINEWEB_BYTES}",
        }
    if data_manifest.get("source_commit") != UPSTREAM_COMMIT:
        return {
            "phase": "data_manifest",
            "message": f"full baseline requires data source commit {UPSTREAM_COMMIT}",
        }
    if full_mode_gates.get("nccl_checked") is not True:
        return {
            "phase": "nccl_all_reduce",
            "message": "full baseline requires NCCL preflight evidence",
        }
    if len(gpus) != 8:
        return {
            "phase": "hardware",
            "message": f"full baseline requires 8 B200 GPUs, found {len(gpus)}",
        }
    for gpu in gpus:
        if not isinstance(gpu, dict) or "B200" not in str(gpu.get("name", "")):
            return {
                "phase": "hardware",
                "message": "full baseline requires every GPU inventory entry to be B200",
            }
    return None


def _is_success(
    classification: dict[str, Any],
    metrics: dict[str, float | None],
    preflight: dict[str, Any],
    source: dict[str, Any],
    variant_patch: dict[str, Any],
    full_attempt_evidence_blocker: dict[str, str] | None,
) -> bool:
    if _source_policy_blocker(classification, source) is not None:
        return False
    if _variant_patch_policy_blocker(classification, variant_patch) is not None:
        return False
    if full_attempt_evidence_blocker is not None:
        return False
    return (
        classification.get("lane") in {"A", "B"}
        and classification.get("mode") == "full"
        and classification.get("claim_eligible") is True
        and preflight.get("ok") is True
        and metrics.get("val_loss") is not None
        and metrics.get("train_time") is not None
        and metrics.get("step_avg") is not None
        and float(metrics["val_loss"]) <= 3.28
    )


def _claim_validation(
    classification: dict[str, Any],
    metrics: dict[str, float | None],
    preflight: dict[str, Any],
    source: dict[str, Any],
    launch_readiness: dict[str, Any],
    data_manifest: dict[str, Any],
    preflight_data_manifest: dict[str, Any],
    full_attempt_evidence_blocker: dict[str, str] | None,
    wall_clock: dict[str, Any],
) -> dict[str, Any]:
    full_mode_gates = launch_readiness.get("full_mode_gates")
    if not isinstance(full_mode_gates, dict):
        full_mode_gates = {}
    mode_full = classification.get("mode") == "full"
    lane_a = classification.get("lane") == "A"
    val_loss = metrics.get("val_loss")
    final_validation_reached = val_loss is not None
    val_loss_within_target = (
        isinstance(val_loss, int | float) and float(val_loss) <= 3.28
    )
    train_time_reported = metrics.get("train_time") is not None
    wall_clock_reported = wall_clock.get("elapsed_seconds") is not None
    nccl_checked = full_mode_gates.get("nccl_checked") is True
    sha_verified = (
        full_mode_gates.get("verified_sha") is True
        and preflight_data_manifest.get("verified_sha") is True
        and data_manifest.get("verified_sha") is True
    )
    source_clean = source.get("dirty") is False
    gates: dict[str, Any] = {
        "successful_b200_reproduction": False,
        "mode_full": mode_full,
        "lane_a": lane_a,
        "preflight_ok": preflight.get("ok") is True,
        "nccl_checked": nccl_checked,
        "sha_verified": sha_verified,
        "source_clean": source_clean,
        "final_validation_reached": final_validation_reached,
        "val_loss_within_target": val_loss_within_target,
        "train_time_reported": train_time_reported,
        "shell_wall_clock_reported": wall_clock_reported,
        "first_blocker": None,
    }
    blocker_checks = [
        (final_validation_reached, "final validation was not reached"),
        (val_loss_within_target, "final val_loss is missing or above 3.28"),
        (train_time_reported, "upstream train_time was not reported"),
        (mode_full, "mode is not full"),
        (lane_a, "lane is not A"),
        (preflight.get("ok") is True, "preflight did not pass"),
        (nccl_checked, "NCCL preflight evidence is missing"),
        (sha_verified, "SHA-verified full manifest evidence is missing"),
        (source_clean, "source cleanliness evidence is missing or dirty"),
        (wall_clock_reported, "shell wall-clock was not reported separately"),
        (
            full_attempt_evidence_blocker is None,
            full_attempt_evidence_blocker.get("message")
            if full_attempt_evidence_blocker is not None
            else "",
        ),
    ]
    for passed, message in blocker_checks:
        if not passed:
            gates["first_blocker"] = message
            break
    gates["successful_b200_reproduction"] = all(
        bool(gates[name])
        for name in (
            "mode_full",
            "lane_a",
            "preflight_ok",
            "nccl_checked",
            "sha_verified",
            "source_clean",
            "final_validation_reached",
            "val_loss_within_target",
            "train_time_reported",
            "shell_wall_clock_reported",
        )
    ) and full_attempt_evidence_blocker is None
    return gates


def build_summary(
    *,
    log_path: Path,
    result_dir: Path,
    source_path: Path,
    data_manifest_path: Path,
    preflight_report_path: Path,
    wall_clock_path: Path | None,
    classification_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = log_path.read_text() if log_path.exists() else ""
    lines = text.splitlines()
    preflight = _read_json(preflight_report_path)
    environment_sidecar = _environment_sidecar(result_dir)
    hardware_sidecar = _hardware_sidecar(result_dir)
    sidecar_environment = environment_sidecar.get("environment")
    sidecar_gpus = hardware_sidecar.get("gpus")
    classification = _default_classification(preflight, classification_override)
    metrics = _final_metrics(text)
    final_validation_reached = metrics["val_loss"] is not None
    source = _source_status(source_path, result_dir)
    variant_patch = _variant_patch_classification(result_dir)
    source_blocker = _source_policy_blocker(classification, source)
    variant_patch_blocker = _variant_patch_policy_blocker(classification, variant_patch)
    launch_readiness = _launch_readiness(result_dir)
    exit_code = _read_json(result_dir / "exit_code.json")
    data_manifest_summary = _data_manifest_summary(data_manifest_path)
    preflight_data_manifest = _preflight_check_detail(preflight, "data_manifest")
    telemetry = _telemetry_summary(result_dir)
    environment = (
        sidecar_environment
        if isinstance(sidecar_environment, dict)
        else preflight.get("environment", {})
    )
    gpus = sidecar_gpus if isinstance(sidecar_gpus, list) else preflight.get("gpus", [])
    if not isinstance(gpus, list):
        gpus = []
    full_attempt_blocker = _full_attempt_evidence_blocker(
        classification,
        launch_readiness=launch_readiness,
        exit_code=exit_code,
        telemetry=telemetry,
        data_manifest=data_manifest_summary,
        preflight_data_manifest=preflight_data_manifest,
        gpus=gpus,
    )
    runner_blocker = _runner_blocker(result_dir)
    explicit_runner_blocker = (
        runner_blocker
        if runner_blocker is not None and runner_blocker.get("phase") != "training"
        else None
    )
    ok = _is_success(
        classification, metrics, preflight, source, variant_patch, full_attempt_blocker
    )
    wall_clock = _read_json(wall_clock_path)
    first_error = _first_relevant_error(lines)
    claim_validation = _claim_validation(
        classification,
        metrics,
        preflight,
        source,
        launch_readiness,
        data_manifest_summary,
        preflight_data_manifest,
        full_attempt_blocker,
        wall_clock,
    )

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "classification": classification,
        "upstream_commit": UPSTREAM_COMMIT,
        "source": source,
        "variant_patch_classification": variant_patch,
        "launch_readiness": launch_readiness,
        "attempt_command": _attempt_command(result_dir),
        "data_manifest": str(data_manifest_path),
        "data_manifest_summary": data_manifest_summary,
        "preflight_data_manifest": preflight_data_manifest,
        "preflight_mlp_backend": _preflight_check_detail(preflight, "mlp_backend"),
        "telemetry": telemetry,
        "artifact_sizes": _artifact_sizes(result_dir),
        "preflight_report": str(preflight_report_path),
        "preflight_ok": preflight.get("ok"),
        "active_jobs": _read_json(result_dir / "active_jobs.json"),
        "environment_sidecar": environment_sidecar,
        "hardware_sidecar": hardware_sidecar,
        "environment": environment,
        "gpus": gpus,
        "final_validation_reached": final_validation_reached,
        "final_metrics": metrics,
        "wall_clock": wall_clock,
        "included_in_baseline_stats": ok,
        "claim_validation": claim_validation,
        "first_relevant_error": first_error,
        "last_50_meaningful_log_lines": _meaningful_tail(lines),
    }

    if not ok:
        summary["blocker"] = (
            source_blocker
            or _preflight_failure_blocker(preflight)
            or explicit_runner_blocker
            or variant_patch_blocker
            or full_attempt_blocker
            or _runtime_blocker(result_dir, text, first_error)
        )

    return summary


def write_summary(
    *,
    log_path: Path,
    result_dir: Path,
    source_path: Path,
    data_manifest_path: Path,
    preflight_report_path: Path,
    wall_clock_path: Path | None,
    classification_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = build_summary(
        log_path=log_path,
        result_dir=result_dir,
        source_path=source_path,
        data_manifest_path=data_manifest_path,
        preflight_report_path=preflight_report_path,
        wall_clock_path=wall_clock_path,
        classification_override=classification_override,
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / "summary.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    write_analysis(result_dir / "analysis.md", summary)
    return summary


def _metric_text(value: object) -> str:
    if value is None:
        return "null"
    return str(value)


def _one_line_text(value: object) -> str:
    if value is None or value == "":
        return "null"
    return str(value).replace("\n", "\\n")


def _range_text(value: object) -> str:
    if not isinstance(value, dict):
        return "null"
    return (
        f"min={_metric_text(value.get('min'))} "
        f"median={_metric_text(value.get('median'))} "
        f"max={_metric_text(value.get('max'))}"
    )


def _gpu_inventory_text(gpus: object) -> str:
    if not isinstance(gpus, list) or not gpus:
        return "null"
    items = []
    for item in gpus:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        name = item.get("name")
        capability = item.get("compute_capability")
        if isinstance(capability, list) and len(capability) >= 2:
            cc = f"{capability[0]}.{capability[1]}"
        else:
            cc = "unknown"
        items.append(f"{index}:{name} cc={cc}")
    return "; ".join(items) if items else "null"


def analysis_markdown(summary: dict[str, Any]) -> str:
    classification = summary.get("classification", {})
    metrics = summary.get("final_metrics", {})
    blocker = summary.get("blocker", {})
    first_relevant_error = summary.get("first_relevant_error")
    source = summary.get("source", {})
    variant_patch = summary.get("variant_patch_classification", {})
    launch_readiness = summary.get("launch_readiness", {})
    active_jobs = summary.get("active_jobs", {})
    attempt_command = summary.get("attempt_command", {})
    data_manifest = summary.get("data_manifest_summary", {})
    preflight_data_manifest = summary.get("preflight_data_manifest", {})
    preflight_mlp_backend = summary.get("preflight_mlp_backend", {})
    telemetry = summary.get("telemetry", {})
    environment = summary.get("environment", {})
    environment_sidecar = summary.get("environment_sidecar", {})
    hardware_sidecar = summary.get("hardware_sidecar", {})
    gpus = summary.get("gpus", [])
    wall_clock = summary.get("wall_clock", {})
    claim_validation = summary.get("claim_validation", {})
    telemetry_rootfs = telemetry.get("rootfs", {})
    telemetry_ranges = telemetry.get("ranges", {})
    process_watch = telemetry.get("process_watch", {})
    disk_watch = telemetry.get("disk_watch", {})
    telemetry_signs = telemetry.get("signs", {})
    dcgm = telemetry.get("dcgm", {})
    stop_snapshot = telemetry.get("stop_snapshot", {})
    artifact_sizes = summary.get("artifact_sizes", {})
    stop_snapshot_commands = (
        stop_snapshot.get("commands") if isinstance(stop_snapshot, dict) else None
    )
    if isinstance(launch_readiness, dict):
        launch_ready_to_launch = launch_readiness.get("ready_to_launch")
        launch_training_launched = launch_readiness.get("training_launched")
        launch_authority_required = launch_readiness.get("launch_authority_required")
        launch_authorization_required_token = launch_readiness.get(
            "launch_authorization_required_token"
        )
        launch_preflight_ok = launch_readiness.get("preflight_ok")
        launch_blocked_by = launch_readiness.get("blocked_by")
    else:
        launch_ready_to_launch = None
        launch_training_launched = None
        launch_authority_required = None
        launch_authorization_required_token = None
        launch_preflight_ok = None
        launch_blocked_by = None
    if isinstance(active_jobs, dict):
        active_jobs_ok = active_jobs.get("ok")
        active_job_count = active_jobs.get("active_job_count")
        active_jobs_ignored_match_count = active_jobs.get("ignored_match_count")
        active_jobs_scan_error = active_jobs.get("scan_error")
    else:
        active_jobs_ok = None
        active_job_count = None
        active_jobs_ignored_match_count = None
        active_jobs_scan_error = None
    if isinstance(attempt_command, dict) and isinstance(
        attempt_command.get("argv"), list
    ):
        attempt_command_text = " ".join(attempt_command["argv"])
    else:
        attempt_command_text = None
    if isinstance(attempt_command, dict) and isinstance(
        attempt_command.get("training_argv"),
        list,
    ):
        training_command_text = " ".join(attempt_command["training_argv"])
    else:
        training_command_text = None
    if isinstance(variant_patch, dict) and isinstance(variant_patch.get("files"), list):
        variant_patch_file_count = len(variant_patch["files"])
    else:
        variant_patch_file_count = 0
    if isinstance(preflight_data_manifest, dict):
        preflight_verified_sha = preflight_data_manifest.get("verified_sha")
    else:
        preflight_verified_sha = None
    if isinstance(telemetry_ranges.get("memory_used_mib"), dict):
        memory_used_mib_peak = telemetry_ranges["memory_used_mib"].get("peak")
    else:
        memory_used_mib_peak = None
    if isinstance(stop_snapshot_commands, list):
        stop_snapshot_command_count = len(stop_snapshot_commands)
    else:
        stop_snapshot_command_count = 0
    lines = [
        "# Modded NanoGPT B200 Attempt Analysis",
        "",
        "## Classification",
        "",
        f"- lane: {classification.get('lane')}",
        f"- mode: {classification.get('mode')}",
        f"- arm: {classification.get('arm')}",
        f"- claim_label: {classification.get('claim_label')}",
        f"- evidence_tier: {classification.get('evidence_tier')}",
        f"- run_id: {classification.get('run_id')}",
        f"- attempt_id: {classification.get('attempt_id')}",
        f"- claim_eligible: {classification.get('claim_eligible')}",
        "",
        "## Launch Readiness",
        "",
        f"- launch_ready_to_launch: {_metric_text(launch_ready_to_launch)}",
        f"- launch_training_launched: {_metric_text(launch_training_launched)}",
        f"- launch_authority_required: {_metric_text(launch_authority_required)}",
        f"- launch_authorization_required_token: {_metric_text(launch_authorization_required_token)}",
        f"- launch_preflight_ok: {_metric_text(launch_preflight_ok)}",
        f"- launch_blocked_by: {_one_line_text(launch_blocked_by)}",
        f"- active_jobs_ok: {_metric_text(active_jobs_ok)}",
        f"- active_job_count: {_metric_text(active_job_count)}",
        f"- active_jobs_ignored_match_count: {_metric_text(active_jobs_ignored_match_count)}",
        f"- active_jobs_scan_error: {_metric_text(active_jobs_scan_error)}",
        f"- attempt_command: {_one_line_text(attempt_command_text)}",
        f"- training_command: {_one_line_text(training_command_text)}",
        "",
        "## Source And Data",
        "",
        f"- source: {source.get('path')}",
        f"- attempt_source_commit: {source.get('commit')}",
        f"- source_dirty: {source.get('dirty')}",
        f"- source_status_before: {_one_line_text(source.get('status_before'))}",
        f"- source_status_current: {_one_line_text(source.get('status'))}",
        f"- variant_patch_diff: {_metric_text(variant_patch.get('diff_artifact') if isinstance(variant_patch, dict) else None)}",
        f"- variant_patch_file_count: {variant_patch_file_count}",
        f"- data_manifest: {summary.get('data_manifest')}",
        f"- data_manifest_kind: {data_manifest.get('kind')}",
        f"- data_manifest_resolved_path: {data_manifest.get('resolved_path')}",
        f"- dataset: {data_manifest.get('dataset')}",
        f"- token_budget: {data_manifest.get('token_budget')}",
        f"- data_source_commit: {data_manifest.get('source_commit')}",
        f"- manifest_verified_sha: {_metric_text(data_manifest.get('verified_sha'))}",
        f"- preflight_verified_sha: {_metric_text(preflight_verified_sha)}",
        f"- num_files: {data_manifest.get('num_files')}",
        f"- total_bytes: {data_manifest.get('total_bytes')}",
        f"- sha256_entries: {data_manifest.get('sha256_entries')}",
        f"- preflight_ok: {summary.get('preflight_ok')}",
        "",
        "## Environment",
        "",
        f"- environment_sidecar_kind: {environment_sidecar.get('kind') if isinstance(environment_sidecar, dict) else None}",
        f"- hardware_sidecar_kind: {hardware_sidecar.get('kind') if isinstance(hardware_sidecar, dict) else None}",
        f"- python: {environment.get('python')}",
        f"- torch: {environment.get('torch')}",
        f"- cuda_runtime: {environment.get('cuda_runtime')}",
        f"- triton: {environment.get('triton')}",
        f"- flash_attention: {environment.get('flash_attention')}",
        f"- gpu_count: {len(gpus) if isinstance(gpus, list) else 0}",
        f"- gpu_inventory: {_gpu_inventory_text(gpus)}",
        "",
        "## Telemetry",
        "",
        f"- telemetry_path: {telemetry.get('path')}",
        f"- telemetry_state: {telemetry.get('state')}",
        f"- telemetry_reason: {telemetry.get('reason')}",
        f"- dcgm_state: {dcgm.get('state') if isinstance(dcgm, dict) else None}",
        f"- dcgm_reason: {dcgm.get('reason') if isinstance(dcgm, dict) else None}",
        f"- rootfs_marker: {telemetry_rootfs.get('torchtitan_in_rootfs')}",
        f"- rootfs_cwd: {telemetry_rootfs.get('cwd')}",
        f"- rootfs_python: {telemetry_rootfs.get('python_executable')}",
        f"- workspace_sentinel_exists: {telemetry_rootfs.get('workspace_sentinel_exists')}",
        f"- gpu_utilization_percent: {_range_text(telemetry_ranges.get('gpu_utilization'))}",
        f"- memory_utilization_percent: {_range_text(telemetry_ranges.get('memory_utilization'))}",
        f"- power_draw_watts: {_range_text(telemetry_ranges.get('power_draw_watts'))}",
        f"- temperature_celsius: {_range_text(telemetry_ranges.get('temperature_celsius'))}",
        f"- sm_clock_mhz: {_range_text(telemetry_ranges.get('sm_clock_mhz'))}",
        f"- memory_used_mib_peak: {_metric_text(memory_used_mib_peak)}",
        f"- process_peak_rss_mib: {_metric_text(process_watch.get('peak_rss_mib'))}",
        f"- process_peak_cpu_percent: {_metric_text(process_watch.get('peak_cpu_percent'))}",
        f"- thermal_or_clock_throttling_signs: {_metric_text(telemetry_signs.get('thermal_or_clock_throttling') if isinstance(telemetry_signs, dict) else None)}",
        f"- thermal_or_clock_throttling_reason: {_metric_text(telemetry_signs.get('thermal_or_clock_throttling_reason') if isinstance(telemetry_signs, dict) else None)}",
        f"- cpu_or_rss_bottleneck_signs: {_metric_text(telemetry_signs.get('cpu_or_rss_bottleneck') if isinstance(telemetry_signs, dict) else None)}",
        f"- cpu_or_rss_bottleneck_reason: {_metric_text(telemetry_signs.get('cpu_or_rss_bottleneck_reason') if isinstance(telemetry_signs, dict) else None)}",
        f"- disk_result_size_peak_bytes: {_metric_text(disk_watch.get('max_result_size_bytes'))}",
        f"- disk_data_size_peak_bytes: {_metric_text(disk_watch.get('max_data_size_bytes'))}",
        f"- artifact_total_known_bytes: {_metric_text(artifact_sizes.get('total_known_bytes') if isinstance(artifact_sizes, dict) else None)}",
        f"- stop_snapshot_reason: {stop_snapshot.get('reason') if isinstance(stop_snapshot, dict) else None}",
        f"- stop_snapshot_exit_code: {stop_snapshot.get('exit_code') if isinstance(stop_snapshot, dict) else None}",
        f"- stop_snapshot_command_count: {stop_snapshot_command_count}",
        "",
        "## Wall Clock",
        "",
        f"- shell_wall_clock_seconds: {_metric_text(wall_clock.get('elapsed_seconds'))}",
        f"- wall_clock_start_epoch: {_metric_text(wall_clock.get('start_epoch'))}",
        f"- wall_clock_end_epoch: {_metric_text(wall_clock.get('end_epoch'))}",
        "",
        "## Final Metrics",
        "",
        f"- val_loss: {_metric_text(metrics.get('val_loss'))}",
        f"- train_time: {_metric_text(metrics.get('train_time'))}",
        f"- step_avg: {_metric_text(metrics.get('step_avg'))}",
        f"- peak_allocated_memory: {_metric_text(metrics.get('peak_allocated_memory'))}",
        f"- peak_reserved_memory: {_metric_text(metrics.get('peak_reserved_memory'))}",
        f"- included_in_baseline_stats: {summary.get('included_in_baseline_stats')}",
        "",
        "## Claim Validation",
        "",
        f"- successful_b200_reproduction: {_metric_text(claim_validation.get('successful_b200_reproduction') if isinstance(claim_validation, dict) else None)}",
        f"- claim_mode_full: {_metric_text(claim_validation.get('mode_full') if isinstance(claim_validation, dict) else None)}",
        f"- claim_lane_a: {_metric_text(claim_validation.get('lane_a') if isinstance(claim_validation, dict) else None)}",
        f"- claim_preflight_ok: {_metric_text(claim_validation.get('preflight_ok') if isinstance(claim_validation, dict) else None)}",
        f"- claim_nccl_checked: {_metric_text(claim_validation.get('nccl_checked') if isinstance(claim_validation, dict) else None)}",
        f"- claim_sha_verified: {_metric_text(claim_validation.get('sha_verified') if isinstance(claim_validation, dict) else None)}",
        f"- claim_source_clean: {_metric_text(claim_validation.get('source_clean') if isinstance(claim_validation, dict) else None)}",
        f"- claim_final_validation_reached: {_metric_text(claim_validation.get('final_validation_reached') if isinstance(claim_validation, dict) else None)}",
        f"- claim_val_loss_within_target: {_metric_text(claim_validation.get('val_loss_within_target') if isinstance(claim_validation, dict) else None)}",
        f"- claim_train_time_reported: {_metric_text(claim_validation.get('train_time_reported') if isinstance(claim_validation, dict) else None)}",
        f"- claim_shell_wall_clock_reported: {_metric_text(claim_validation.get('shell_wall_clock_reported') if isinstance(claim_validation, dict) else None)}",
        f"- claim_first_blocker: {_metric_text(claim_validation.get('first_blocker') if isinstance(claim_validation, dict) else None)}",
        "",
    ]
    if isinstance(variant_patch, dict) and isinstance(variant_patch.get("files"), list):
        for item in variant_patch["files"]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "- variant_patch_file: "
                f"{item.get('path')} | "
                f"class={item.get('patch_class')} | "
                f"first_lane_a_blocker={item.get('first_lane_a_blocker')}"
            )
        lines.append("")
    if isinstance(artifact_sizes, dict) and isinstance(
        artifact_sizes.get("files"), dict
    ):
        for path, size in sorted(artifact_sizes["files"].items()):
            lines.append(f"- artifact_size: {path}={size}")
        lines.append("")
    if blocker:
        lines.extend(
            [
                "## Failure Analysis",
                "",
                f"- first failing phase: {blocker.get('phase')}",
                f"- first relevant error: {_metric_text(first_relevant_error)}",
                f"- blocker_message: {blocker.get('message')}",
                f"- failure_category: {_failure_category(blocker)}",
            ]
        )
        if isinstance(preflight_mlp_backend, dict) and preflight_mlp_backend:
            local_smoke = preflight_mlp_backend.get("local_smoke")
            local_smoke = local_smoke if isinstance(local_smoke, dict) else {}
            lines.extend(
                [
                    f"- mlp_backend: {_metric_text(preflight_mlp_backend.get('backend'))}",
                    f"- mlp_local_smoke_backend: {_metric_text(local_smoke.get('backend'))}",
                    f"- mlp_local_smoke_output_shape: {_metric_text(local_smoke.get('output_shape'))}",
                    f"- mlp_blocked_kernel: {_metric_text(preflight_mlp_backend.get('blocked_kernel'))}",
                    f"- mlp_blocked_arch: {_metric_text(preflight_mlp_backend.get('blocked_arch'))}",
                    f"- mlp_failure_class: {_metric_text(preflight_mlp_backend.get('failure_class'))}",
                    f"- mlp_compiler_pass: {_metric_text(preflight_mlp_backend.get('compiler_pass'))}",
                    f"- mlp_full_mode_policy: {_metric_text(preflight_mlp_backend.get('full_mode_policy'))}",
                ]
            )
        lines.append("")
    tail = summary.get("last_50_meaningful_log_lines", [])
    if tail:
        lines.extend(["## Last Meaningful Log Lines", "", "```text"])
        lines.extend(str(line) for line in tail[-50:])
        lines.extend(["```", ""])
    return "\n".join(lines)


def write_analysis(path: Path, summary: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(analysis_markdown(summary) + "\n")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--wall-clock-json", type=Path)
    parser.add_argument("--lane", choices=["A", "B", "C"])
    parser.add_argument("--mode", choices=["full", "smoke", "diagnostic"])
    parser.add_argument("--arm")
    parser.add_argument("--run-id")
    parser.add_argument("--attempt-id")
    return parser.parse_args()


def main(*, enforce_rootfs: bool = False) -> int:
    args = parse_args()
    if enforce_rootfs:
        guard_exit = cli_guard.guard_rootfs_cli(
            "experiments/modded_nanogpt_b200/parse_log.sh"
        )
        if guard_exit is not None:
            return guard_exit
    summary = write_summary(
        log_path=args.log,
        result_dir=args.result_dir,
        source_path=args.source,
        data_manifest_path=args.data_manifest,
        preflight_report_path=args.preflight_report,
        wall_clock_path=args.wall_clock_json,
        classification_override={
            "lane": args.lane,
            "mode": args.mode,
            "arm": args.arm,
            "run_id": args.run_id,
            "attempt_id": args.attempt_id,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(enforce_rootfs=True))
