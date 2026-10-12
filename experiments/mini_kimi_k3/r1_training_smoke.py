# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run a bounded TorchTitan Trainer smoke for Mini Kimi K3 r1."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchtitan.experiments.execution.executor import (  # noqa: E402
    Executor,
    SubprocessExecutor,
)


SCHEMA_VERSION = 1
_GRAD_NORM_RE = re.compile(r"(?:grad(?:ient)?[_ ]?norm)[:= ]+([0-9.eE+-]+)")


def run_r1_training_smoke(
    *,
    token_manifest: Path,
    tokens_dir: Path,
    report_path: Path,
    hf_assets_path: Path = Path("tests/assets/tokenizer"),
    seq_len: int = 4096,
    steps: int = 1,
    local_batch_size: int = 1,
    global_batch_size: int = 1,
    min_free_gpu_memory_mib: int = 12_000,
    skip_gpu_memory_precheck: bool = False,
    dump_folder: Path | None = None,
    executor: Executor | None = None,
    gpu_memory_probe: Callable[[], list[dict[str, Any]]] | None = None,
    gpu_holder_probe: Callable[[], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise RuntimeError(
            "Mini-K3 r1 training smoke must run inside the TorchTitan rootfs"
        )
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    if local_batch_size <= 0:
        raise ValueError(f"local_batch_size must be > 0, got {local_batch_size}")
    if global_batch_size <= 0:
        raise ValueError(f"global_batch_size must be > 0, got {global_batch_size}")
    if min_free_gpu_memory_mib < 0:
        raise ValueError(
            "min_free_gpu_memory_mib must be >= 0, got " f"{min_free_gpu_memory_mib}"
        )

    executor = executor or SubprocessExecutor()
    dump_folder = dump_folder or (report_path.parent / "r1_training_smoke_train")
    gpu_memory = _gpu_memory_evidence(
        min_free_gpu_memory_mib=min_free_gpu_memory_mib,
        skip_gpu_memory_precheck=skip_gpu_memory_precheck,
        gpu_memory_probe=gpu_memory_probe,
        gpu_holder_probe=gpu_holder_probe,
    )
    argv = _smoke_argv(
        token_manifest=token_manifest,
        tokens_dir=tokens_dir,
        hf_assets_path=hf_assets_path,
        seq_len=seq_len,
        steps=steps,
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
        dump_folder=dump_folder,
        cuda_visible_devices=_selected_cuda_visible_devices(gpu_memory),
    )
    if gpu_memory["status"] == "blocked":
        report = _base_report(
            status="blocked",
            token_manifest=token_manifest,
            tokens_dir=tokens_dir,
            hf_assets_path=hf_assets_path,
            seq_len=seq_len,
            steps=0,
            max_grad_norm=None,
            argv=argv,
            return_code=None,
            stdout_tail="",
            stderr_tail="",
            gpu_memory=gpu_memory,
            error="insufficient free GPU memory for r1 Trainer smoke",
        )
        _write_report(report_path, report)
        return report

    result = executor.run(argv, cwd=REPO_ROOT)
    stdout_tail = _tail_text(result.stdout)
    stderr_tail = _tail_text(result.stderr)
    combined_tail = stdout_tail + "\n" + stderr_tail
    max_grad_norm = _parse_max_grad_norm(result.stdout + "\n" + result.stderr)

    status = "pass" if result.return_code == 0 and max_grad_norm is not None else "fail"
    error = None
    if result.return_code != 0:
        if _is_cuda_oom(combined_tail):
            status = "blocked"
            error = "CUDA out of memory during r1 Trainer smoke"
        else:
            error = "run_train.sh returned nonzero"
    elif max_grad_norm is None:
        error = "could not parse optimizer-step grad norm evidence"
    report = _base_report(
        status=status,
        token_manifest=token_manifest,
        tokens_dir=tokens_dir,
        hf_assets_path=hf_assets_path,
        seq_len=seq_len,
        steps=steps if result.return_code == 0 else 0,
        max_grad_norm=max_grad_norm,
        argv=argv,
        return_code=result.return_code,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        gpu_memory=gpu_memory,
    )
    if error is not None:
        report["error"] = error

    _write_report(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-step Mini Kimi K3 r1 Trainer smoke through run_train.sh."
    )
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--tokens-dir", type=Path, required=True)
    parser.add_argument(
        "--hf-assets-path",
        type=Path,
        default=Path("tests/assets/tokenizer"),
        help=(
            "Tokenizer assets used only to satisfy Trainer construction. The "
            "Mini-K3 uint32 dataloader consumes token ids directly."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/results/r1_training_smoke.json"),
    )
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument(
        "--min-free-gpu-memory-mib",
        type=int,
        default=12_000,
        help="Block before launching run_train.sh unless at least one GPU has this much free memory.",
    )
    parser.add_argument(
        "--skip-gpu-memory-precheck",
        action="store_true",
        help="Run the Trainer smoke even when the GPU memory precheck would block.",
    )
    parser.add_argument(
        "--dump-folder",
        type=Path,
        help="Optional TorchTitan dump folder for the smoke command.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_r1_training_smoke(
            token_manifest=args.token_manifest,
            tokens_dir=args.tokens_dir,
            report_path=args.report,
            hf_assets_path=args.hf_assets_path,
            seq_len=args.seq_len,
            steps=args.steps,
            local_batch_size=args.local_batch_size,
            global_batch_size=args.global_batch_size,
            min_free_gpu_memory_mib=args.min_free_gpu_memory_mib,
            skip_gpu_memory_precheck=args.skip_gpu_memory_precheck,
            dump_folder=args.dump_folder,
        )
    except Exception as exc:
        print(f"Mini Kimi K3 r1 training smoke failed: {exc}", file=sys.stderr)
        return 21
    print(f"wrote r1 training smoke report: {args.report}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 21


def _smoke_argv(
    *,
    token_manifest: Path,
    tokens_dir: Path,
    hf_assets_path: Path,
    seq_len: int,
    steps: int,
    local_batch_size: int,
    global_batch_size: int,
    dump_folder: Path,
    cuda_visible_devices: str | None,
) -> list[str]:
    argv = [
        "/usr/bin/env",
        "MODULE=mini_kimi_k3",
        "CONFIG=mini_kimi_k3_r1_contract",
        "COMM_MODE=fake_backend",
        "NGPU=1",
        "./run_train.sh",
        f"--hf_assets_path={hf_assets_path}",
        f"--dump_folder={dump_folder}",
        "--training.steps",
        str(steps),
        f"--training.local_batch_size={local_batch_size}",
        f"--training.global_batch_size={global_batch_size}",
        f"--training.seq_len={seq_len}",
        "--checkpoint.no-enable",
        f"--dataloader.token_manifest={token_manifest}",
        f"--dataloader.tokens_dir={tokens_dir}",
    ]
    if cuda_visible_devices is not None:
        argv.insert(1, f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    return argv


def _selected_cuda_visible_devices(gpu_memory: dict[str, Any]) -> str | None:
    selected = gpu_memory.get("selected")
    if not isinstance(selected, dict):
        return None
    index = selected.get("index")
    if isinstance(index, int):
        return str(index)
    if isinstance(index, str) and index.strip():
        return index.strip()
    return None


def _base_report(
    *,
    status: str,
    token_manifest: Path,
    tokens_dir: Path,
    hf_assets_path: Path,
    seq_len: int,
    steps: int,
    max_grad_norm: float | None,
    argv: list[str],
    return_code: int | None,
    stdout_tail: str,
    stderr_tail: str,
    gpu_memory: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_r1_training_smoke",
        "status": status,
        "config": "mini_kimi_k3_r1_contract",
        "model_flavor": "r1",
        "rootfs": {
            "marker": os.environ.get("TORCHTITAN_IN_ROOTFS"),
            "cwd": str(REPO_ROOT),
        },
        "gpu_memory": gpu_memory,
        "data": {
            "manifest": str(token_manifest),
            "tokens_dir": str(tokens_dir),
            "seq_len": seq_len,
            "hf_assets_path": str(hf_assets_path),
        },
        "optimization": {
            "steps": steps,
            "max_grad_norm": max_grad_norm,
        },
        "command": {
            "argv": argv,
            "cwd": str(REPO_ROOT),
            "return_code": return_code,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        },
    }
    if error is not None:
        report["error"] = error
    return report


def _gpu_memory_evidence(
    *,
    min_free_gpu_memory_mib: int,
    skip_gpu_memory_precheck: bool,
    gpu_memory_probe: Callable[[], list[dict[str, Any]]] | None,
    gpu_holder_probe: Callable[[], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if skip_gpu_memory_precheck:
        return {
            "status": "skipped",
            "required_free_mib": min_free_gpu_memory_mib,
            "selected": None,
            "devices": [],
        }
    probe = gpu_memory_probe or _query_gpu_memory
    try:
        devices = probe()
    except Exception as exc:
        return {
            "status": "unknown",
            "required_free_mib": min_free_gpu_memory_mib,
            "selected": None,
            "devices": [],
            "error": str(exc),
        }
    eligible = [
        device
        for device in devices
        if int(device.get("memory_free_mib", -1)) >= min_free_gpu_memory_mib
    ]
    selected = max(
        eligible,
        key=lambda device: int(device.get("memory_free_mib", -1)),
        default=None,
    )
    holders = _gpu_holder_evidence(
        gpu_holder_probe=gpu_holder_probe,
        include_holders=selected is None,
    )
    return {
        "status": "pass" if selected is not None else "blocked",
        "required_free_mib": min_free_gpu_memory_mib,
        "selected": selected,
        "devices": devices,
        **holders,
    }


def _query_gpu_memory() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    devices = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, name, total, used, free = [part.strip() for part in line.split(",", 4)]
        devices.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "memory_free_mib": int(free),
            }
        )
    return devices


def _gpu_holder_evidence(
    *,
    gpu_holder_probe: Callable[[], list[dict[str, Any]]] | None,
    include_holders: bool,
) -> dict[str, Any]:
    if not include_holders:
        return {}
    probe = gpu_holder_probe or _query_gpu_holders
    try:
        return _gpu_holder_metadata(probe())
    except Exception as exc:
        return {"holders": [], "holder_error": str(exc)}


def _gpu_holder_metadata(holders: list[dict[str, Any]]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"holders": holders}
    if not holders:
        metadata[
            "holder_detail"
        ] = "no GPU holder processes were visible from the current process namespace"
    return metadata


def _query_gpu_holders(device_root: Path = Path("/dev")) -> list[dict[str, Any]]:
    device_args = _nvidia_device_args(device_root)
    if not device_args:
        return []
    try:
        result = subprocess.run(
            ["fuser", "-v", *device_args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return _query_gpu_holders_from_proc(device_root=device_root)
    pids = sorted(
        {
            int(match)
            for match in re.findall(r"\b\d{2,}\b", result.stdout + result.stderr)
        }
    )
    holders = []
    for pid in pids:
        holders.append(_process_holder_record(pid))
    if holders or result.returncode == 0:
        return holders
    return _query_gpu_holders_from_proc(device_root=device_root)


def _nvidia_device_args(device_root: Path) -> list[str]:
    return [
        str(path)
        for path in sorted(device_root.glob("nvidia*"), key=lambda path: path.name)
        if path.name == "nvidiactl"
        or path.name == "nvidia-uvm"
        or (path.name.startswith("nvidia") and path.name[6:].isdigit())
    ]


def _query_gpu_holders_from_proc(
    *,
    proc_root: Path = Path("/proc"),
    device_root: Path = Path("/dev"),
) -> list[dict[str, Any]]:
    device_paths = {
        path.resolve()
        for path in device_root.glob("nvidia*")
        if path.name == "nvidiactl"
        or path.name == "nvidia-uvm"
        or path.name.startswith("nvidia")
    }
    if not device_paths:
        return []

    holders = []
    for proc in sorted(proc_root.iterdir(), key=lambda path: path.name):
        if not proc.name.isdigit():
            continue
        fd_dir = proc / "fd"
        try:
            fd_entries = list(fd_dir.iterdir())
        except OSError:
            continue
        if not any(_fd_targets_gpu(fd, device_paths) for fd in fd_entries):
            continue
        holders.append(_proc_holder_record(proc))
    return holders


def _fd_targets_gpu(fd_path: Path, device_paths: set[Path]) -> bool:
    try:
        return fd_path.resolve() in device_paths
    except OSError:
        return False


def _proc_holder_record(proc: Path) -> dict[str, Any]:
    pid = int(proc.name)
    status = _parse_proc_status(proc / "status")
    cmdline = _read_proc_cmdline(proc / "cmdline")
    record: dict[str, Any] = {
        "pid": pid,
        "ppid": status.get("ppid"),
        "user": status.get("uid"),
        "stat": status.get("state"),
        "etime": None,
        "command": _bounded_text(cmdline or status.get("name") or ""),
    }
    try:
        record["cwd"] = _bounded_text(str((proc / "cwd").resolve()))
    except OSError:
        pass
    return record


def _process_holder_record(pid: int) -> dict[str, Any]:
    record: dict[str, Any] = {"pid": pid}
    proc = Path("/proc") / str(pid)
    if not proc.exists():
        record["detail"] = "process metadata is not visible from this namespace"
        return record
    stat_result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "user=,pid=,ppid=,stat=,etime=,comm="],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    line = stat_result.stdout.strip()
    if line:
        parts = line.split(maxsplit=5)
        if len(parts) == 6:
            user, parsed_pid, ppid, stat, etime, command = parts
            record.update(
                {
                    "user": user,
                    "pid": int(parsed_pid),
                    "ppid": int(ppid),
                    "stat": stat,
                    "etime": etime,
                    "command": _bounded_text(command),
                }
            )
    cwd_path = Path("/proc") / str(pid) / "cwd"
    try:
        record["cwd"] = _bounded_text(str(cwd_path.resolve()))
    except OSError:
        pass
    return record


def _parse_proc_status(path: Path) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return fields
    for line in lines:
        if line.startswith("Name:"):
            fields["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("State:"):
            fields["state"] = line.split(":", 1)[1].strip().split(maxsplit=1)[0]
        elif line.startswith("PPid:"):
            fields["ppid"] = _parse_optional_int(line.split(":", 1)[1].strip())
        elif line.startswith("Uid:"):
            fields["uid"] = line.split(":", 1)[1].strip().split()[0]
    return fields


def _read_proc_cmdline(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    return " ".join(part.decode(errors="replace") for part in raw.split(b"\0") if part)


def _parse_optional_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _bounded_text(value: str, *, max_chars: int = 240) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _parse_max_grad_norm(output: str) -> float | None:
    values: list[float] = []
    for match in _GRAD_NORM_RE.finditer(output):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return max(values) if values else None


def _is_cuda_oom(output: str) -> bool:
    return "CUDA out of memory" in output or "torch.OutOfMemoryError" in output


def _tail_text(text: str, *, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
