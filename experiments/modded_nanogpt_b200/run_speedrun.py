#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Orchestrate modded-nanogpt B200 speedrun attempts under the rootfs."""

from __future__ import annotations

import argparse
import glob
import json
import os
import selectors
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.modded_nanogpt_b200 import cli_guard, parse_log


SCHEMA_VERSION = 1
DEFAULT_ENVIRONMENT_CLASS = "torchtitan-rootfs-b200"
RESULT_ROOT = Path("experiments/modded_nanogpt_b200/results")
NO_OUTPUT_TIMEOUT_EXIT_CODE = 124
FULL_LAUNCH_AUTHORIZATION_TOKEN = "launch-full-b200"
FULL_TRIAL_GPU_COUNT = 2
ACTIVE_JOB_COMMANDS = ("torchrun", "train_gpt.py", "cached_fineweb10B.py")
ACTIVE_JOB_SEARCH_COMMANDS = ("pgrep", "grep", "rg", "ripgrep", "ps")
PRELAUNCH_RESULT_FILES = frozenset(
    {"operator_notes.md", "active_jobs.json", "active_jobs_prelaunch.json"}
)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str


@dataclass
class TelemetryProcess:
    name: str
    proc: subprocess.Popen
    output: Path


@dataclass(frozen=True)
class RunConfig:
    lane: str
    mode: str
    source: Path
    data_manifest: Path
    result_dir: Path
    attention_backend: str
    mlp_backend: str
    allow_previous_stall: bool = False
    verify_sha: bool = False
    skip_run: bool = False
    launch_authorization: str | None = None
    run_id: str | None = None
    attempt_id: str | None = None
    arm: str | None = None
    result_root: Path = RESULT_ROOT

    def __post_init__(self) -> None:
        if self.lane not in {"A", "B"}:
            raise ValueError(f"unsupported lane: {self.lane}")
        if self.mode not in {"full", "smoke", "diagnostic"}:
            raise ValueError(f"unsupported mode: {self.mode}")
        if self.mode == "full":
            if self.allow_previous_stall:
                raise ValueError("full mode cannot use --allow-previous-stall")
            if not self.verify_sha:
                raise ValueError("full mode requires --verify-sha")
        if not _is_under_result_root(self.result_dir, self.result_root):
            raise ValueError(
                f"result-dir must be under {self.result_root}: {self.result_dir}"
            )


def _is_under_result_root(path: Path, result_root: Path) -> bool:
    try:
        path.resolve().relative_to((Path.cwd() / result_root).resolve())
        return True
    except ValueError:
        try:
            path.resolve().relative_to(result_root.resolve())
            return True
        except ValueError:
            return False


def _claim_label(lane: str, mode: str, attention_backend: str) -> str:
    if mode == "smoke":
        return "smoke"
    if mode == "diagnostic":
        return "diagnostic"
    if lane == "A":
        return "B200 upstream reproduction"
    if attention_backend == "fa2":
        return "B200 compatibility patchset"
    return "B200 systems-only"


def _evidence_tier(mode: str) -> str:
    if mode == "full":
        return "full-single-attempt"
    return mode


def _ids(config: RunConfig) -> tuple[str, str, str]:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = (
        config.run_id
        or config.result_dir.name
        or f"lane_{config.lane.lower()}_{config.mode}_{timestamp}"
    )
    attempt_id = config.attempt_id or f"{run_id}_attempt_001"
    arm = config.arm or ("A0" if config.lane == "A" else "B0")
    return run_id, attempt_id, arm


def _classification(config: RunConfig) -> dict[str, object]:
    run_id, attempt_id, arm = _ids(config)
    return {
        "lane": config.lane,
        "mode": config.mode,
        "arm": arm,
        "claim_label": _claim_label(config.lane, config.mode, config.attention_backend),
        "evidence_tier": _evidence_tier(config.mode),
        "run_id": run_id,
        "attempt_id": attempt_id,
        "environment_class": DEFAULT_ENVIRONMENT_CLASS,
        "claim_eligible": False,
    }


def _write_json_atomic(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _git_output(source: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True, stderr=subprocess.STDOUT
        )
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}\n"


def _capture_source(config: RunConfig) -> None:
    before = _git_output(config.source, ["status", "--short"])
    (config.result_dir / "source_status_before.txt").write_text(before)
    telemetry = _telemetry_dir(config)
    (telemetry / "source_status_before.txt").write_text(before)
    source_json = {
        "path": str(config.source),
        "commit": _git_output(config.source, ["rev-parse", "HEAD"]).strip(),
        "status_before": before,
    }
    _write_json_atomic(config.result_dir / "source.json", source_json)
    if config.lane == "B":
        _write_variant_patch_artifacts(config, before)


def _capture_source_after(config: RunConfig) -> None:
    after = _git_output(config.source, ["status", "--short"])
    (config.result_dir / "source_status_after.txt").write_text(after)
    telemetry = _telemetry_dir(config)
    (telemetry / "source_status_after.txt").write_text(after)


def _write_variant_patch_artifacts(config: RunConfig, status: str) -> None:
    diff = _git_output(config.source, ["diff"])
    (config.result_dir / "variant_patch.diff").write_text(diff)
    _write_variant_patch_classification(config, status)


def _variant_patch_entry(path: str) -> dict[str, str]:
    if path == "train_gpt.py":
        return {
            "path": path,
            "patch_class": "hardware-detection",
            "first_lane_a_blocker": "FA3 no kernel image on B200",
        }
    if path == "triton_kernels.py":
        return {
            "path": path,
            "patch_class": "kernel-compat",
            "first_lane_a_blocker": "Triton sm100 custom kernel compile/runtime blocker",
        }
    return {
        "path": path,
        "patch_class": "unclassified",
        "first_lane_a_blocker": "unclassified Lane B source difference",
    }


def _changed_paths_from_status(status: str) -> list[str]:
    paths = []
    for line in status.splitlines():
        if not line:
            continue
        path = line[3:] if len(line) > 3 else line
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    return paths


def _write_variant_patch_classification(config: RunConfig, status: str) -> None:
    report = {
        "schema_version": SCHEMA_VERSION,
        "source": str(config.source),
        "diff_artifact": str(config.result_dir / "variant_patch.diff"),
        "files": [
            _variant_patch_entry(path) for path in _changed_paths_from_status(status)
        ],
    }
    _write_json_atomic(config.result_dir / "variant_patch_classification.json", report)


def _unclassified_variant_patch_paths(config: RunConfig) -> list[str]:
    try:
        data = json.loads(
            (config.result_dir / "variant_patch_classification.json").read_text()
        )
    except (OSError, json.JSONDecodeError):
        return []
    files = data.get("files", [])
    if not isinstance(files, list):
        return []
    paths = []
    for item in files:
        if not isinstance(item, dict):
            continue
        if item.get("patch_class") != "unclassified":
            continue
        path = item.get("path")
        if isinstance(path, str):
            paths.append(path)
    return paths


def _write_unclassified_variant_patch_blocker(
    config: RunConfig, paths: list[str]
) -> None:
    message = "full Lane B launch has unclassified source patches: " + ", ".join(paths)
    _append_run_log(config.result_dir, message + "\n")
    _write_watcher_status(config, "not_started", "variant_patch_classification")
    _capture_source_after(config)
    _write_exit_code(config.result_dir, 21, "variant_patch_classification")
    _write_blocker(config.result_dir, "variant_patch_classification", message)


def _manifest_shard_paths(data_manifest: Path) -> list[Path]:
    try:
        data = json.loads(data_manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"data manifest is not readable JSON: {data_manifest}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"data manifest must be a JSON object: {data_manifest}")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"data manifest has no shard file paths: {data_manifest}")
    paths = []
    for item in files:
        path: object
        if isinstance(item, dict):
            path = item.get("path")
        else:
            path = item
        if not isinstance(path, str) or not path:
            raise ValueError(
                f"data manifest has an unusable shard path: {data_manifest}"
            )
        shard_path = Path(path)
        if not shard_path.is_absolute():
            shard_path = Path.cwd() / shard_path
        paths.append(shard_path.resolve())
    return paths


def _common_path_parent(paths: list[Path]) -> Path:
    common = Path(os.path.commonpath([str(path.parent) for path in paths]))
    return common.resolve()


def _data_root_from_manifest(data_manifest: Path) -> Path:
    shard_paths = _manifest_shard_paths(data_manifest)
    shard_parent = _common_path_parent(shard_paths)
    if shard_parent.name == "fineweb10B":
        data_dir = shard_parent.parent.resolve()
        if data_dir.name == "data":
            return data_dir.parent.resolve()
        return data_dir
    return shard_parent


def _source_data_path_blocker(
    source: Path, env: dict[str, str]
) -> dict[str, str] | None:
    data_path = env.get("DATA_PATH")
    if not data_path:
        return {"phase": "data_path", "message": "DATA_PATH is not set"}
    patterns = {
        "train": "data/fineweb10B/fineweb_train_*.bin",
        "val": "data/fineweb10B/fineweb_val_*.bin",
    }
    missing = []
    for split, relative_pattern in patterns.items():
        pattern = str(Path(data_path) / relative_pattern)
        matches = glob.glob(pattern)
        if matches:
            continue
        source_pattern = str(source / relative_pattern)
        source_matches = glob.glob(source_pattern)
        message = (
            f"{split} shards are not visible from source cwd with DATA_PATH={data_path}: "
            f"looked for {pattern}"
        )
        if source_matches:
            message += f"; source-visible pattern exists at {source_pattern}"
        missing.append(message)
    if not missing:
        return None
    return {"phase": "data_path", "message": "; ".join(missing)}


def _run_env(config: RunConfig) -> dict[str, str]:
    result_abs = config.result_dir.resolve()
    data_root = _data_root_from_manifest(config.data_manifest)
    env = {
        "HF_HOME": str((Path.cwd() / ".cache/huggingface").resolve()),
        "HF_HUB_CACHE": str((Path.cwd() / ".cache/huggingface/hub").resolve()),
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "DATA_PATH": str(data_root),
        "TORCHINDUCTOR_CACHE_DIR": str(result_abs / "torchinductor_cache"),
        "TRITON_CACHE_DIR": str(result_abs / "triton_cache"),
        "MODDED_NANOGPT_ATTN_BACKEND": config.attention_backend,
        "MODDED_NANOGPT_MLP_BACKEND": config.mlp_backend,
    }
    if config.lane == "B":
        env["MODDED_NANOGPT_CE_COMPUTE_CAPABILITY"] = "100"
    return env


def _write_env(path: Path, env: dict[str, str]) -> None:
    lines = []
    for key, value in sorted(_redacted_env(env).items()):
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")


def _redacted_env(env: dict[str, str]) -> dict[str, str]:
    redacted_fragments = (
        "AUTH",
        "CREDENTIAL",
        "KEY",
        "PASSWORD",
        "SECRET",
        "TOKEN",
    )
    redacted = {}
    for key, value in env.items():
        if any(fragment in key.upper() for fragment in redacted_fragments):
            redacted[key] = "<REDACTED>"
        else:
            redacted[key] = value
    return redacted


def _write_data_manifest_pointer(config: RunConfig) -> None:
    _write_json_atomic(
        config.result_dir / "data_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "data_manifest_pointer",
            "path": str(config.data_manifest),
        },
    )


def _training_command() -> list[str]:
    return [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={FULL_TRIAL_GPU_COUNT}",
        "train_gpt.py",
    ]


def _attempt_command(config: RunConfig) -> list[str]:
    cmd = [
        "experiments/modded_nanogpt_b200/run_speedrun.sh",
        "--lane",
        config.lane,
        "--mode",
        config.mode,
        "--source",
        str(config.source),
        "--data-manifest",
        str(config.data_manifest),
        "--result-dir",
        str(config.result_dir),
        "--attention-backend",
        config.attention_backend,
        "--mlp-backend",
        config.mlp_backend,
    ]
    if config.allow_previous_stall:
        cmd.append("--allow-previous-stall")
    if config.verify_sha:
        cmd.append("--verify-sha")
    if config.skip_run:
        cmd.append("--skip-run")
    if config.launch_authorization is not None:
        cmd.append("--launch-authorization=<REDACTED>")
    if config.run_id is not None:
        cmd.extend(["--run-id", config.run_id])
    if config.attempt_id is not None:
        cmd.extend(["--attempt-id", config.attempt_id])
    if config.arm is not None:
        cmd.extend(["--arm", config.arm])
    return cmd


def _default_runner(cmd: list[str], **kwargs) -> CommandResult:
    log_path = kwargs.pop("log_path", None)
    no_output_timeout_seconds = kwargs.pop("no_output_timeout_seconds", None)
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                **kwargs,
            )
            assert proc.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)
            last_output = time.monotonic()
            while proc.poll() is None:
                timeout = None
                if no_output_timeout_seconds is not None:
                    elapsed = time.monotonic() - last_output
                    timeout = max(0.0, no_output_timeout_seconds - elapsed)
                events = selector.select(timeout=timeout)
                if not events:
                    if no_output_timeout_seconds is not None:
                        message = (
                            f"no output for {no_output_timeout_seconds} seconds; "
                            "terminating command\n"
                        )
                        log.write(message)
                        log.flush()
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=10)
                        selector.close()
                        return CommandResult(
                            returncode=NO_OUTPUT_TIMEOUT_EXIT_CODE, stdout=""
                        )
                    continue
                for key, _ in events:
                    line = key.fileobj.readline()
                    if line:
                        log.write(line)
                        log.flush()
                        last_output = time.monotonic()
            for line in proc.stdout:
                log.write(line)
                log.flush()
            selector.close()
            return CommandResult(returncode=proc.returncode or 0, stdout="")
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **kwargs,
    )
    return CommandResult(returncode=proc.returncode, stdout=proc.stdout)


def _active_job_kind(args: str) -> str | None:
    try:
        tokens = shlex.split(args)
    except ValueError:
        tokens = args.split()
    token_names = {Path(token).name for token in tokens}
    for name in ACTIVE_JOB_COMMANDS:
        if name in token_names:
            return name
    return None


def _is_search_only_active_job_match(args: str) -> bool:
    try:
        tokens = shlex.split(args)
    except ValueError:
        tokens = args.split()
    token_names = {Path(token).name for token in tokens}
    if not any(name in ACTIVE_JOB_SEARCH_COMMANDS for name in token_names):
        return False
    if not all(name in args for name in ACTIVE_JOB_COMMANDS):
        return False
    return True


def _active_job_report(ps_output: str) -> dict[str, object]:
    active_jobs = []
    ignored_matches = []
    for line in ps_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, args = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if _is_search_only_active_job_match(args):
            ignored_matches.append({"pid": pid, "kind": "search", "args": args})
            continue
        kind = _active_job_kind(args)
        if kind is None:
            continue
        record = {"pid": pid, "kind": kind, "args": args}
        active_jobs.append(record)
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": not active_jobs,
        "active_job_count": len(active_jobs),
        "active_jobs": active_jobs,
        "ignored_match_count": len(ignored_matches),
        "ignored_matches": ignored_matches,
    }


def check_active_jobs(output: Path | None = None) -> dict[str, object]:
    proc = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    report = _active_job_report(proc.stdout)
    report["ps_returncode"] = proc.returncode
    if proc.returncode != 0:
        report["ok"] = False
        report["scan_error"] = f"ps exited with code {proc.returncode}"
    report["timestamp_epoch"] = time.time()
    if output is not None:
        _write_json_atomic(output, report)
    return report


def _preflight_command(
    config: RunConfig, classification: dict[str, object]
) -> list[str]:
    cmd = [
        "experiments/modded_nanogpt_b200/run_preflight.sh",
        "--mode",
        config.mode,
        "--lane",
        config.lane,
        "--run-id",
        str(classification["run_id"]),
        "--attempt-id",
        str(classification["attempt_id"]),
        "--arm",
        str(classification["arm"]),
        "--source",
        str(config.source),
        "--data-manifest",
        str(config.data_manifest),
        "--attention-backend",
        config.attention_backend,
        "--mlp-backend",
        config.mlp_backend,
        "--report",
        str(config.result_dir / "preflight_report.json"),
        "--expected-gpus",
        str(FULL_TRIAL_GPU_COUNT),
    ]
    if config.verify_sha:
        cmd.append("--verify-sha")
    if config.allow_previous_stall:
        cmd.append("--allow-previous-stall")
    return cmd


def _requires_flash_attention_setup(config: RunConfig) -> bool:
    return config.lane == "B" and config.attention_backend == "fa2"


def _flash_attention_setup_command() -> list[str]:
    return ["experiments/modded_nanogpt_b200/setup_flash_attention.sh"]


def _append_run_log(result_dir: Path, text: str) -> None:
    with (result_dir / "run.log").open("a") as f:
        f.write(text)
        if text and not text.endswith("\n"):
            f.write("\n")


def _write_attempt(
    config: RunConfig, classification: dict[str, object], env: dict[str, str]
) -> None:
    attempt = {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "source": str(config.source),
        "data_manifest": str(config.data_manifest),
        "command": {
            "argv": _attempt_command(config),
            "training_argv": _training_command(),
            "skip_run": config.skip_run,
            "launch_authorization_present": config.launch_authorization is not None,
            "attention_backend": config.attention_backend,
            "mlp_backend": config.mlp_backend,
        },
        "environment": _redacted_env(env),
    }
    _write_json_atomic(config.result_dir / "attempt.json", attempt)


def _write_operator_notes(config: RunConfig, classification: dict[str, object]) -> None:
    lines = [
        "# Attempt Notes",
        "",
        f"- lane: {classification['lane']}",
        f"- mode: {classification['mode']}",
        f"- arm: {classification['arm']}",
        f"- run_id: {classification['run_id']}",
        f"- attempt_id: {classification['attempt_id']}",
        f"- source: {config.source}",
        f"- data_manifest: {config.data_manifest}",
        f"- attention_backend: {config.attention_backend}",
        f"- mlp_backend: {config.mlp_backend}",
        f"- skip_run: {config.skip_run}",
        f"- launch_authorization_present: {config.launch_authorization is not None}",
        f"- verify_sha: {config.verify_sha}",
        f"- allow_previous_stall: {config.allow_previous_stall}",
        "",
    ]
    if config.skip_run:
        lines.extend(
            [
                "## Known Blocker",
                "",
                "- skip-run requested; training will not be launched.",
                "",
            ]
        )
    path = config.result_dir / "operator_notes.md"
    text = "\n".join(lines)
    if path.exists():
        existing = path.read_text()
        separator = "\n\n" if existing and not existing.endswith("\n\n") else ""
        path.write_text(existing + separator + text)
        return
    path.write_text(text)


def _write_wall_clock(result_dir: Path, start: float, end: float) -> None:
    _write_json_atomic(
        result_dir / "wall_clock.json",
        {"start_epoch": start, "end_epoch": end, "elapsed_seconds": end - start},
    )


def _write_exit_code(result_dir: Path, exit_code: int, phase: str) -> None:
    _write_json_atomic(
        result_dir / "exit_code.json", {"exit_code": exit_code, "phase": phase}
    )


def _write_blocker(result_dir: Path, phase: str, message: str) -> None:
    _write_json_atomic(
        result_dir / "blocker.json", {"phase": phase, "message": message}
    )


def _cache_directory_blocker(
    config: RunConfig,
    env: dict[str, str],
) -> dict[str, str] | None:
    result_dir = config.result_dir.resolve()
    for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        path = Path(env[name]).resolve()
        try:
            path.relative_to(result_dir)
        except ValueError:
            return {
                "phase": "cache_directories",
                "message": f"{name} is outside result_dir: {path}",
            }
        if not path.is_dir():
            return {
                "phase": "cache_directories",
                "message": f"{name} is not a directory: {path}",
            }
    return None


def _write_cache_directory_blocker(
    config: RunConfig,
    blocker: dict[str, str],
) -> None:
    _write_watcher_status(config, "not_started", "cache_directories")
    _write_exit_code(config.result_dir, 21, "cache_directories")
    _write_blocker(config.result_dir, blocker["phase"], blocker["message"])


def _write_data_path_blocker(
    config: RunConfig,
    blocker: dict[str, str],
) -> None:
    _append_run_log(config.result_dir, blocker["message"] + "\n")
    _write_watcher_status(config, "not_started", "data_path")
    _capture_source_after(config)
    _write_exit_code(config.result_dir, 21, "data_path")
    _write_blocker(config.result_dir, blocker["phase"], blocker["message"])


def _write_flash_attention_setup_blocker(
    config: RunConfig,
    classification: dict[str, object],
    returncode: int,
) -> None:
    message = f"Lane B FA2 setup failed with exit code {returncode}"
    _write_launch_readiness(
        config,
        classification,
        preflight_returncode=None,
        training_launched=False,
        extra_blocked_by=[{"phase": "flash_attention_setup", "message": message}],
    )
    _write_watcher_status(config, "not_started", "flash_attention_setup_failed")
    _capture_source_after(config)
    _write_exit_code(config.result_dir, returncode, "flash_attention_setup")
    _write_blocker(config.result_dir, "flash_attention_setup", message)


def _preflight_failure_blocker(report_path: Path, returncode: int) -> dict[str, str]:
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {
            "phase": "preflight",
            "message": f"preflight failed with exit code {returncode}",
        }
    if isinstance(report, dict):
        failures = report.get("failures")
        if isinstance(failures, list) and failures:
            first = failures[0]
            if isinstance(first, dict):
                phase = first.get("name")
                message = first.get("error")
                if phase and message:
                    return {"phase": str(phase), "message": str(message)}
    return {
        "phase": "preflight",
        "message": f"preflight failed with exit code {returncode}",
    }


def _check_by_name(
    preflight_report: dict[str, object], name: str
) -> dict[str, object] | None:
    checks = preflight_report.get("checks")
    if not isinstance(checks, list):
        return None
    for check in checks:
        if isinstance(check, dict) and check.get("name") == name:
            return check
    return None


def _check_ok(preflight_report: dict[str, object], name: str) -> bool:
    check = _check_by_name(preflight_report, name)
    return check is not None and check.get("ok") is True


def _check_detail(preflight_report: dict[str, object], name: str) -> dict[str, object]:
    check = _check_by_name(preflight_report, name)
    if check is None:
        return {}
    detail = check.get("detail")
    if isinstance(detail, dict):
        return detail
    return {}


def _read_preflight_report(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _write_preflight_sidecars(config: RunConfig) -> None:
    preflight_report = _read_preflight_report(
        config.result_dir / "preflight_report.json"
    )
    environment = preflight_report.get("environment")
    if isinstance(environment, dict):
        _write_json_atomic(
            config.result_dir / "environment.json",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "preflight_environment",
                "environment": environment,
            },
        )
    gpus = preflight_report.get("gpus")
    if isinstance(gpus, list):
        _write_json_atomic(
            config.result_dir / "hardware.json",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "preflight_gpus",
                "gpus": gpus,
            },
        )


def _write_launch_readiness(
    config: RunConfig,
    classification: dict[str, object],
    *,
    preflight_returncode: int | None,
    training_launched: bool,
    extra_blocked_by: list[dict[str, str]] | None = None,
) -> None:
    preflight_ran = preflight_returncode is not None
    preflight_report = _read_preflight_report(
        config.result_dir / "preflight_report.json"
    )
    data_check = _check_by_name(preflight_report, "data_manifest")
    data_detail = _check_detail(preflight_report, "data_manifest")
    data_manifest_checked = data_check is not None
    verified_sha = (
        data_detail.get("verified_sha") is True if data_manifest_checked else None
    )
    full_mode_gates = {
        "full_mode": config.mode == "full",
        "verify_sha_requested": config.verify_sha,
        "data_manifest_checked": data_manifest_checked,
        "verified_sha": verified_sha,
        "nccl_checked": _check_ok(preflight_report, "nccl_all_reduce"),
        "manifest_token_budget": data_detail.get("token_budget"),
        "manifest_num_files": data_detail.get("num_files"),
        "manifest_total_bytes": data_detail.get("total_bytes"),
    }
    blocked_by = list(extra_blocked_by or [])
    if preflight_returncode not in {0, None}:
        blocker = _preflight_failure_blocker(
            config.result_dir / "preflight_report.json", preflight_returncode
        )
        blocked_by.append(blocker)
    if config.mode == "full":
        if not config.verify_sha:
            blocked_by.append(
                {"phase": "mode_policy", "message": "full mode requires --verify-sha"}
            )
        if preflight_ran and full_mode_gates["nccl_checked"] is not True:
            blocked_by.append(
                {
                    "phase": "nccl_all_reduce",
                    "message": "full mode requires NCCL preflight evidence",
                }
            )
        if (
            preflight_ran
            and data_manifest_checked
            and full_mode_gates["verified_sha"] is not True
        ):
            blocked_by.append(
                {
                    "phase": "data_manifest",
                    "message": "full mode requires SHA-verified data manifest",
                }
            )
    ready_to_launch = preflight_returncode == 0 and not blocked_by
    launch_authority_required = config.mode == "full" and not training_launched
    report = {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "run_id": classification["run_id"],
        "attempt_id": classification["attempt_id"],
        "lane": config.lane,
        "mode": config.mode,
        "arm": classification["arm"],
        "ready_to_launch": ready_to_launch,
        "training_launched": training_launched,
        "launch_authority_required": launch_authority_required,
        "launch_authorization_required_token": (
            FULL_LAUNCH_AUTHORIZATION_TOKEN if launch_authority_required else None
        ),
        "preflight_ok": preflight_returncode == 0
        and preflight_report.get("ok") is True,
        "preflight_returncode": preflight_returncode,
        "verify_sha": config.verify_sha,
        "allow_previous_stall": config.allow_previous_stall,
        "skip_run": config.skip_run,
        "source": str(config.source),
        "data_manifest": str(config.data_manifest),
        "attention_backend": config.attention_backend,
        "mlp_backend": config.mlp_backend,
        "full_mode_gates": full_mode_gates,
        "blocked_by": blocked_by,
    }
    _write_json_atomic(config.result_dir / "launch_readiness.json", report)
    blocked = (
        "none"
        if not blocked_by
        else "; ".join(f"{item['phase']}: {item['message']}" for item in blocked_by)
    )
    lines = [
        "# Launch Readiness",
        "",
        f"- lane: {config.lane}",
        f"- mode: {config.mode}",
        f"- arm: {classification['arm']}",
        f"- ready_to_launch: {ready_to_launch}",
        f"- training_launched: {training_launched}",
        f"- launch_authority_required: {report['launch_authority_required']}",
        f"- launch_authorization_required_token: {report['launch_authorization_required_token']}",
        f"- preflight_ok: {report['preflight_ok']}",
        f"- verify_sha: {config.verify_sha}",
        f"- nccl_checked: {full_mode_gates['nccl_checked']}",
        f"- manifest_checked: {full_mode_gates['data_manifest_checked']}",
        f"- manifest_token_budget: {full_mode_gates['manifest_token_budget']}",
        f"- manifest_num_files: {full_mode_gates['manifest_num_files']}",
        f"- manifest_total_bytes: {full_mode_gates['manifest_total_bytes']}",
        f"- manifest_verified_sha: {full_mode_gates['verified_sha']}",
        f"- blocked_by: {blocked}",
        "",
    ]
    (config.result_dir / "launch_readiness.md").write_text("\n".join(lines))


def _full_launch_authorized(config: RunConfig) -> bool:
    return (
        config.mode != "full"
        or config.launch_authorization == FULL_LAUNCH_AUTHORIZATION_TOKEN
    )


def _write_launch_authority_blocker(config: RunConfig) -> None:
    message = (
        f"full launch requires --launch-authorization={FULL_LAUNCH_AUTHORIZATION_TOKEN}"
    )
    _append_run_log(config.result_dir, message + "\n")
    _write_watcher_status(config, "not_started", "launch_authority_missing")
    _capture_source_after(config)
    _write_exit_code(config.result_dir, 21, "launch_authority")
    _write_blocker(config.result_dir, "launch_authority", message)


def _active_jobs_blocker_message(report: dict[str, object]) -> str:
    scan_error = report.get("scan_error")
    if isinstance(scan_error, str) and scan_error:
        return "active-job scan failed: " + scan_error
    jobs = report.get("active_jobs")
    if not isinstance(jobs, list):
        return "active training or data-prep jobs found"
    parts = []
    for item in jobs:
        if not isinstance(item, dict):
            continue
        pid = item.get("pid")
        kind = item.get("kind")
        if pid is None or kind is None:
            continue
        parts.append(f"{pid} {kind}")
    if not parts:
        return "active training or data-prep jobs found"
    return "active training or data-prep jobs found: " + ", ".join(parts)


def _write_active_jobs_blocker(
    config: RunConfig, report: dict[str, object]
) -> dict[str, str]:
    message = _active_jobs_blocker_message(report)
    _append_run_log(config.result_dir, message + "\n")
    _write_watcher_status(config, "not_started", "active_jobs")
    _capture_source_after(config)
    _write_exit_code(config.result_dir, 21, "active_jobs")
    _write_blocker(config.result_dir, "active_jobs", message)
    return {"phase": "active_jobs", "message": message}


def _telemetry_dir(config: RunConfig) -> Path:
    path = config.result_dir / "telemetry"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_rootfs_environment(config: RunConfig, env: dict[str, str]) -> None:
    telemetry = _telemetry_dir(config)
    rootfs = {
        "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "torchtitan_in_rootfs": os.environ.get("TORCHTITAN_IN_ROOTFS"),
        "workspace_sentinel": "scripts/rootfs/enter_rootfs.sh",
        "workspace_sentinel_exists": Path("scripts/rootfs/enter_rootfs.sh").exists(),
        "cache_environment": {
            "TORCHINDUCTOR_CACHE_DIR": env["TORCHINDUCTOR_CACHE_DIR"],
            "TRITON_CACHE_DIR": env["TRITON_CACHE_DIR"],
        },
    }
    _write_json_atomic(telemetry / "rootfs_environment.json", rootfs)


def _write_dcgm_status(config: RunConfig) -> None:
    telemetry = _telemetry_dir(config)
    checked_commands = ["dcgmi", "dcgmproftester"]
    found = [
        {"command": command, "path": shutil.which(command)}
        for command in checked_commands
        if shutil.which(command)
    ]
    if found:
        status = {
            "state": "available",
            "reason": "rootfs-friendly DCGM command found",
            "checked_commands": checked_commands,
            "commands": found,
            "timestamp_epoch": time.time(),
        }
    else:
        status = {
            "state": "unavailable",
            "reason": "no rootfs-friendly DCGM command found",
            "checked_commands": checked_commands,
            "commands": [],
            "timestamp_epoch": time.time(),
        }
    _write_json_atomic(telemetry / "dcgm_status.json", status)


def _write_watcher_status(config: RunConfig, state: str, reason: str) -> None:
    telemetry = _telemetry_dir(config)
    for name in ("process_watch.log", "disk_watch.log", "gpu_processes.log"):
        path = telemetry / name
        if not path.exists():
            path.write_text(f"not started: {reason}\n")
    _write_json_atomic(
        telemetry / "watcher_status.json",
        {"state": state, "reason": reason, "timestamp_epoch": time.time()},
    )


def _start_telemetry(config: RunConfig) -> list[TelemetryProcess]:
    telemetry = _telemetry_dir(config)
    specs = [
        (
            "nvidia_smi_query",
            telemetry / "nvidia_smi_query.csv",
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,name,uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv",
                "-l",
                "5",
            ],
        ),
        (
            "nvidia_smi_dmon",
            telemetry / "nvidia_smi_dmon.csv",
            ["nvidia-smi", "dmon", "-s", "pucvmet", "-d", "5"],
        ),
        (
            "gpu_processes",
            telemetry / "gpu_processes.log",
            [
                "bash",
                "-lc",
                "while true; do date -u; nvidia-smi pmon -c 1 || true; sleep 5; done",
            ],
        ),
        (
            "process_watch",
            telemetry / "process_watch.log",
            [
                "bash",
                "-lc",
                "while true; do date -u; ps -eo pid,ppid,pgid,etime,rss,pcpu,pmem,args | "
                "grep -E 'torchrun|train_gpt.py|python' | grep -v grep || true; sleep 5; done",
            ],
        ),
        (
            "disk_watch",
            telemetry / "disk_watch.log",
            [
                "bash",
                "-lc",
                "while true; do date -u; du -sh "
                + shlex.quote(str(config.result_dir))
                + " "
                + shlex.quote(str(config.data_manifest.parent))
                + " 2>/dev/null || true; sleep 30; done",
            ],
        ),
    ]
    if shutil.which("dcgmi"):
        specs.append(
            (
                "dcgm_dmon",
                telemetry / "dcgm_dmon.log",
                ["dcgmi", "dmon", "-e", "100,101,150,155,203", "-d", "5"],
            )
        )
    processes = []
    failures = []
    for name, output, cmd in specs:
        try:
            out = output.open("w")
            proc = subprocess.Popen(
                cmd, stdout=out, stderr=subprocess.STDOUT, text=True
            )
            out.close()
            processes.append(TelemetryProcess(name=name, proc=proc, output=output))
        except Exception as exc:  # noqa: BLE001
            failures.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
    _write_json_atomic(
        telemetry / "watcher_status.json",
        {
            "state": "started" if processes else "unavailable",
            "reason": "training",
            "processes": [
                {"name": item.name, "pid": item.proc.pid, "output": str(item.output)}
                for item in processes
            ],
            "failures": failures,
            "timestamp_epoch": time.time(),
        },
    )
    return processes


def _stop_telemetry(
    config: RunConfig, processes: list[TelemetryProcess], reason: str
) -> None:
    stopped = []
    for item in processes:
        item.proc.terminate()
        try:
            item.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            item.proc.kill()
            item.proc.wait(timeout=10)
        stopped.append(
            {
                "name": item.name,
                "pid": item.proc.pid,
                "returncode": item.proc.returncode,
                "output": str(item.output),
            }
        )
    _write_json_atomic(
        _telemetry_dir(config) / "watcher_status.json",
        {
            "state": "stopped" if processes else "not_started",
            "reason": reason,
            "processes": stopped,
            "timestamp_epoch": time.time(),
        },
    )


def _capture_stop_snapshot(config: RunConfig, *, reason: str, exit_code: int) -> None:
    telemetry = _telemetry_dir(config)
    commands = [
        {
            "artifact": "stop_ps_tree.txt",
            "argv": ["ps", "-eo", "pid,ppid,pgid,etime,rss,pcpu,pmem,args"],
        },
        {
            "artifact": "stop_nvidia_smi_processes.txt",
            "argv": ["nvidia-smi", "pmon", "-c", "1"],
        },
    ]
    records = []
    for command in commands:
        artifact = telemetry / str(command["artifact"])
        argv = command["argv"]
        assert isinstance(argv, list)
        try:
            result = subprocess.run(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            artifact.write_text(result.stdout)
            records.append(
                {
                    "artifact": command["artifact"],
                    "argv": argv,
                    "returncode": result.returncode,
                }
            )
        except Exception as exc:  # noqa: BLE001
            artifact.write_text(f"{type(exc).__name__}: {exc}\n")
            records.append(
                {
                    "artifact": command["artifact"],
                    "argv": argv,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    _write_json_atomic(
        telemetry / "stop_snapshot.json",
        {
            "reason": reason,
            "exit_code": exit_code,
            "commands": records,
            "timestamp_epoch": time.time(),
        },
    )


def _parse_summary(config: RunConfig, classification: dict[str, object]) -> None:
    parse_log.write_summary(
        log_path=config.result_dir / "run.log",
        result_dir=config.result_dir,
        source_path=config.source,
        data_manifest_path=config.data_manifest,
        preflight_report_path=config.result_dir / "preflight_report.json",
        wall_clock_path=config.result_dir / "wall_clock.json",
        classification_override=classification,
    )


def _prepare_result_dir_for_new_attempt(result_dir: Path) -> bool:
    if result_dir.exists():
        if not result_dir.is_dir():
            print(
                f"attempt_reuse blocker: result_dir exists but is not a directory: {result_dir}",
                file=sys.stderr,
            )
            return False
        entries = list(result_dir.iterdir())
        allowed_prelaunch_files = entries and all(
            entry.is_file() and entry.name in PRELAUNCH_RESULT_FILES
            for entry in entries
        )
        if entries and not allowed_prelaunch_files:
            first_entry = entries[0]
            print(
                "attempt_reuse blocker: result_dir already exists and is non-empty: "
                f"{result_dir} (first entry: {first_entry.name}); "
                "use a new run ID or attempt ID",
                file=sys.stderr,
            )
            return False
        return True

    result_dir.mkdir(parents=True, exist_ok=False)
    return True


def run_attempt(
    config: RunConfig,
    *,
    command_runner: Callable[..., CommandResult] = _default_runner,
) -> int:
    start = time.time()
    if not _prepare_result_dir_for_new_attempt(config.result_dir):
        return 21
    try:
        env = _run_env(config)
    except ValueError as exc:
        _write_watcher_status(config, "not_started", "data_manifest")
        _write_exit_code(config.result_dir, 21, "data_manifest")
        _write_blocker(config.result_dir, "data_manifest", str(exc))
        _write_wall_clock(config.result_dir, start, time.time())
        return 21
    for path in (env["TORCHINDUCTOR_CACHE_DIR"], env["TRITON_CACHE_DIR"]):
        Path(path).mkdir(parents=True, exist_ok=True)
    classification = _classification(config)
    _write_attempt(config, classification, env)
    _write_operator_notes(config, classification)
    _write_env(config.result_dir / "command.env", env)
    (config.result_dir / "command.argv").write_text(
        shlex.join(_attempt_command(config)) + "\n"
    )
    _write_data_manifest_pointer(config)
    _write_rootfs_environment(config, env)
    _write_dcgm_status(config)
    cache_blocker = _cache_directory_blocker(config, env)
    if cache_blocker is not None:
        _write_launch_readiness(
            config,
            classification,
            preflight_returncode=None,
            training_launched=False,
            extra_blocked_by=[cache_blocker],
        )
        _write_cache_directory_blocker(config, cache_blocker)
        _write_wall_clock(config.result_dir, start, time.time())
        _parse_summary(config, classification)
        return 21
    _capture_source(config)
    if config.lane == "B" and config.mode == "full":
        unclassified_paths = _unclassified_variant_patch_paths(config)
        if unclassified_paths:
            blocker = {
                "phase": "variant_patch_classification",
                "message": "full Lane B launch has unclassified source patches: "
                + ", ".join(unclassified_paths),
            }
            _write_launch_readiness(
                config,
                classification,
                preflight_returncode=None,
                training_launched=False,
                extra_blocked_by=[blocker],
            )
            _write_unclassified_variant_patch_blocker(config, unclassified_paths)
            _write_wall_clock(config.result_dir, start, time.time())
            _parse_summary(config, classification)
            return 21

    if _requires_flash_attention_setup(config):
        setup = command_runner(
            _flash_attention_setup_command(), env={**os.environ, **env}
        )
        _append_run_log(config.result_dir, setup.stdout)
        if setup.returncode != 0:
            _write_flash_attention_setup_blocker(
                config, classification, setup.returncode
            )
            _write_wall_clock(config.result_dir, start, time.time())
            _parse_summary(config, classification)
            return setup.returncode

    preflight = command_runner(
        _preflight_command(config, classification), env={**os.environ, **env}
    )
    _append_run_log(config.result_dir, preflight.stdout)
    _write_preflight_sidecars(config)
    preflight_report = _read_preflight_report(
        config.result_dir / "preflight_report.json"
    )
    preflight_returncode = preflight.returncode
    if preflight_returncode == 0 and preflight_report.get("ok") is not True:
        preflight_returncode = 21
    if preflight_returncode != 0:
        _write_launch_readiness(
            config,
            classification,
            preflight_returncode=preflight_returncode,
            training_launched=False,
        )
        _write_watcher_status(config, "not_started", "preflight_failed")
        _capture_source_after(config)
        _write_wall_clock(config.result_dir, start, time.time())
        _write_exit_code(config.result_dir, preflight_returncode, "preflight")
        blocker = _preflight_failure_blocker(
            config.result_dir / "preflight_report.json", preflight_returncode
        )
        _write_blocker(config.result_dir, blocker["phase"], blocker["message"])
        _parse_summary(config, classification)
        return preflight_returncode

    active_jobs_report = None
    if config.mode == "full":
        active_jobs_report = check_active_jobs(config.result_dir / "active_jobs.json")
        if active_jobs_report.get("ok") is not True:
            blocker = _write_active_jobs_blocker(config, active_jobs_report)
            _write_launch_readiness(
                config,
                classification,
                preflight_returncode=preflight_returncode,
                training_launched=False,
                extra_blocked_by=[blocker],
            )
            _write_wall_clock(config.result_dir, start, time.time())
            _parse_summary(config, classification)
            return 21

    if config.skip_run:
        _write_launch_readiness(
            config,
            classification,
            preflight_returncode=preflight_returncode,
            training_launched=False,
        )
        _append_run_log(
            config.result_dir, "skip-run requested; training was not launched\n"
        )
        _write_watcher_status(config, "not_started", "skip_run")
        _capture_source_after(config)
        _write_wall_clock(config.result_dir, start, time.time())
        _write_exit_code(config.result_dir, 0, "not_launched")
        _write_blocker(
            config.result_dir,
            "not_launched",
            "skip-run requested; training was not launched",
        )
        _parse_summary(config, classification)
        return 0

    if not _full_launch_authorized(config):
        _write_launch_readiness(
            config,
            classification,
            preflight_returncode=preflight_returncode,
            training_launched=False,
        )
        _write_launch_authority_blocker(config)
        _write_wall_clock(config.result_dir, start, time.time())
        _parse_summary(config, classification)
        return 21

    if active_jobs_report is None:
        active_jobs_report = check_active_jobs(config.result_dir / "active_jobs.json")
    if active_jobs_report.get("ok") is not True:
        blocker = _write_active_jobs_blocker(config, active_jobs_report)
        _write_launch_readiness(
            config,
            classification,
            preflight_returncode=preflight_returncode,
            training_launched=False,
            extra_blocked_by=[blocker],
        )
        _write_wall_clock(config.result_dir, start, time.time())
        _parse_summary(config, classification)
        return 21

    data_path_blocker = _source_data_path_blocker(config.source, env)
    if data_path_blocker is not None:
        _write_launch_readiness(
            config,
            classification,
            preflight_returncode=preflight_returncode,
            training_launched=False,
            extra_blocked_by=[data_path_blocker],
        )
        _write_data_path_blocker(config, data_path_blocker)
        _write_wall_clock(config.result_dir, start, time.time())
        _parse_summary(config, classification)
        return 21

    train_env = {**os.environ, **env}
    telemetry_processes = _start_telemetry(config)
    try:
        _write_launch_readiness(
            config,
            classification,
            preflight_returncode=preflight_returncode,
            training_launched=True,
        )
        train = command_runner(
            _training_command(),
            cwd=config.source,
            env=train_env,
            log_path=config.result_dir / "run.log",
            no_output_timeout_seconds=600,
        )
    finally:
        _stop_telemetry(config, telemetry_processes, "training_finished")
    _append_run_log(config.result_dir, train.stdout)
    _capture_source_after(config)
    _write_wall_clock(config.result_dir, start, time.time())
    _write_exit_code(config.result_dir, train.returncode, "training")
    if train.returncode == NO_OUTPUT_TIMEOUT_EXIT_CODE:
        _capture_stop_snapshot(
            config, reason="no_output_timeout", exit_code=train.returncode
        )
        _write_blocker(
            config.result_dir, "stall", "training produced no output for 600 seconds"
        )
    elif train.returncode != 0:
        _capture_stop_snapshot(
            config, reason="training_exit_nonzero", exit_code=train.returncode
        )
        _write_blocker(
            config.result_dir,
            "training",
            f"training exited with code {train.returncode}",
        )
    _parse_summary(config, classification)
    return train.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-active-jobs", action="store_true")
    parser.add_argument("--active-jobs-output", type=Path)
    parser.add_argument("--lane", choices=["A", "B"])
    parser.add_argument("--mode", choices=["full", "smoke", "diagnostic"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument(
        "--attention-backend", default="fa3", choices=["fa3", "fa2", "flex"]
    )
    parser.add_argument("--mlp-backend", default="triton", choices=["triton", "torch"])
    parser.add_argument("--allow-previous-stall", action="store_true")
    parser.add_argument("--verify-sha", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--launch-authorization")
    parser.add_argument("--run-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--arm")
    return parser.parse_args()


def main(*, enforce_rootfs: bool = False) -> int:
    args = parse_args()
    if enforce_rootfs:
        guard_exit = cli_guard.guard_rootfs_cli(
            "experiments/modded_nanogpt_b200/run_speedrun.sh"
        )
        if guard_exit is not None:
            return guard_exit
    if args.check_active_jobs:
        report = check_active_jobs(args.active_jobs_output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 21
    required = ["lane", "mode", "source", "data_manifest", "result_dir"]
    missing = [
        name.replace("_", "-") for name in required if getattr(args, name) is None
    ]
    if missing:
        print(
            f"error: missing required arguments: {', '.join('--' + name for name in missing)}",
            file=sys.stderr,
        )
        return 21
    try:
        config = RunConfig(
            lane=args.lane,
            mode=args.mode,
            source=args.source,
            data_manifest=args.data_manifest,
            result_dir=args.result_dir,
            attention_backend=args.attention_backend,
            mlp_backend=args.mlp_backend,
            allow_previous_stall=args.allow_previous_stall,
            verify_sha=args.verify_sha,
            skip_run=args.skip_run,
            launch_authorization=args.launch_authorization,
            run_id=args.run_id,
            attempt_id=args.attempt_id,
            arm=args.arm,
        )
        return run_attempt(config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 21


if __name__ == "__main__":
    raise SystemExit(main(enforce_rootfs=True))
