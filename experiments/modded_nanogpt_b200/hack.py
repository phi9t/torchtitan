"""Scratch probe for the rootfs-hosted Jupyter kernel.

This file is intended to be imported from the bwrap rootfs hosted Jupyter
kernel. Rank 0 can stay interactive while the helper CLI keeps ranks 1..N-1
alive as subprocess workers.
"""

from __future__ import annotations

import rich
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torchtitan.distributed.parallel_dims import ParallelDims  # noqa: E402


def _free_loopback_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _rootfs_context() -> dict[str, Any]:
    cwd = Path.cwd().resolve()
    return {
        "cwd": str(cwd),
        "in_rootfs": os.environ.get("TORCHTITAN_IN_ROOTFS") == "1",
        "shell": os.environ.get("SHELL"),
        "sys_prefix": sys.prefix,
        "python": sys.executable,
    }


def require_rootfs_jupyter_context() -> None:
    context = _rootfs_context()
    errors = []
    if not context["in_rootfs"]:
        errors.append("TORCHTITAN_IN_ROOTFS=1 is missing")
    if context["cwd"] != "/workspace/torchtitan":
        errors.append(f"expected cwd /workspace/torchtitan, got {context['cwd']}")
    if not str(context["sys_prefix"]).endswith(".venv-rootfs"):
        errors.append(f"expected .venv-rootfs sys.prefix, got {context['sys_prefix']}")
    if context["shell"] != "/bin/bash":
        errors.append(f"expected SHELL=/bin/bash, got {context['shell']!r}")
    if errors:
        raise RuntimeError("; ".join(errors))


@dataclass(frozen=True)
class RankConfig:
    rank: int
    local_rank: int
    world_size: int
    master_addr: str
    master_port: str


@dataclass
class WorkerHandle:
    master_addr: str
    master_port: str
    world_size: int
    processes: list[subprocess.Popen]
    log_files: list[TextIO]
    log_dir: Path


def worker_env(
    *,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: str,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env.update(
        {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "NCCL_DEBUG": env.get("NCCL_DEBUG", "WARN"),
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": env.get(
                "TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"
            ),
            "TORCH_NCCL_BLOCKING_WAIT": env.get("TORCH_NCCL_BLOCKING_WAIT", "1"),
        }
    )
    return env


def resolve_rank_config(*, multi_rank: bool) -> RankConfig:
    if multi_rank:
        missing = [
            name
            for name in (
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "MASTER_ADDR",
                "MASTER_PORT",
            )
            if name not in os.environ
        ]
        if missing:
            raise RuntimeError(
                "multi-rank probe requires distributed env vars: "
                + ", ".join(missing)
            )
    else:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", _free_loopback_port())

    return RankConfig(
        rank=int(os.environ["RANK"]),
        local_rank=int(os.environ["LOCAL_RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        master_addr=os.environ["MASTER_ADDR"],
        master_port=os.environ["MASTER_PORT"],
    )


def _ensure_process_group(rank_config: RankConfig) -> None:
    if dist.is_initialized():
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the rootfs Jupyter kernel")
    torch.cuda.set_device(rank_config.local_rank % torch.cuda.device_count())
    dist.init_process_group(
        backend="cuda:nccl,cpu:gloo",
        init_method="env://",
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )


def _probe_parallel_dims(world_size: int) -> ParallelDims:
    return ParallelDims(
        dp_replicate=1,
        dp_shard=world_size,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=world_size,
    )


def probe(*, multi_rank: bool = False) -> dict[str, Any]:
    require_rootfs_jupyter_context()
    rank_config = resolve_rank_config(multi_rank=multi_rank)
    _ensure_process_group(rank_config)
    rank, world_size = dist.get_rank(), dist.get_world_size()

    parallel_dims = _probe_parallel_dims(world_size)
    if rank == 0:
        rich.print(parallel_dims)
    world_mesh = parallel_dims.build_mesh()
    fsdp_mesh = parallel_dims.get_mesh("fsdp")

    model = nn.Sequential(
        nn.Linear(8, 16, bias=False),
        nn.SiLU(),
        nn.Linear(16, 4, bias=False),
    ).cuda()

    fully_shard(
        model,
        mesh=fsdp_mesh,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        ),
    )
    if not isinstance(model, FSDPModule):
        raise RuntimeError(
            f"expected fully_shard to return an FSDPModule, got {type(model)!r}"
        )

    torch.manual_seed(20260918 + rank)
    x = torch.randn(4, 8, device="cuda")
    loss = model(x).float().square().mean()
    loss.backward()
    dist.barrier()
    torch.cuda.synchronize()

    return {
        "context": _rootfs_context(),
        "torch": torch.__version__,
        "torch_git": getattr(torch.version, "git_version", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "rank": rank,
        "world_size": world_size,
        "world_mesh": {
            "shape": list(world_mesh.shape),
            "names": list(world_mesh.mesh_dim_names or ()),
        },
        "fsdp_mesh": {
            "shape": list(fsdp_mesh.shape),
            "names": list(fsdp_mesh.mesh_dim_names or ()),
        },
        "loss": float(loss.detach().cpu()),
        "fsdp_module": isinstance(model, FSDPModule),
    }


def launch_jupyter_workers(
    *,
    world_size: int = 8,
    master_addr: str = "127.0.0.1",
    master_port: str | None = None,
    log_dir: str | Path = "experiments/modded_nanogpt_b200/results/jupyter_pg_probe",
) -> WorkerHandle:
    if dist.is_initialized():
        raise RuntimeError(
            "destroy the existing process group with "
            "torch.distributed.destroy_process_group() before launching workers"
        )
    if world_size < 2:
        raise ValueError("world_size must be at least 2 to launch sidecar workers")
    if master_port is None:
        master_port = _free_loopback_port()

    os.environ.update(
        {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,
            "WORLD_SIZE": str(world_size),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": os.environ.get(
                "TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"
            ),
            "TORCH_NCCL_BLOCKING_WAIT": os.environ.get("TORCH_NCCL_BLOCKING_WAIT", "1"),
        }
    )

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen] = []
    log_files: list[TextIO] = []
    for rank in range(1, world_size):
        log_file = (log_path / f"rank{rank}.log").open("w")
        log_files.append(log_file)
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "experiments.modded_nanogpt_b200.hack",
                    "--worker",
                ],
                cwd=str(Path.cwd()),
                env=worker_env(
                    rank=rank,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                ),
                stdin=subprocess.PIPE,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        )
    return WorkerHandle(
        master_addr, master_port, world_size, processes, log_files, log_path
    )


def distributed_probe(handle: WorkerHandle) -> dict[str, Any]:
    for proc in handle.processes:
        if proc.poll() is not None:
            raise RuntimeError(f"worker exited before probe: pid={proc.pid}")
        assert proc.stdin is not None
        proc.stdin.write("probe\n")
        proc.stdin.flush()
    return probe(multi_rank=True)


def shutdown_jupyter_workers(
    handle: WorkerHandle, *, timeout_seconds: float = 10.0
) -> None:
    for proc in handle.processes:
        if proc.poll() is None and proc.stdin is not None:
            proc.stdin.write("shutdown\n")
            proc.stdin.flush()
            proc.stdin.close()
    deadline = time.monotonic() + timeout_seconds
    for proc in handle.processes:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.terminate()
    for log_file in handle.log_files:
        log_file.close()


def _worker_loop() -> int:
    require_rootfs_jupyter_context()
    rank_config = resolve_rank_config(multi_rank=True)
    _ensure_process_group(rank_config)
    print(
        json.dumps(
            {
                "event": "ready",
                "rank": rank_config.rank,
                "world_size": rank_config.world_size,
                "device": torch.cuda.get_device_name(torch.cuda.current_device()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for line in sys.stdin:
        command = line.strip()
        if command == "probe":
            try:
                result = probe(multi_rank=True)
                print(
                    json.dumps({"event": "probe", "ok": True, "result": result}),
                    flush=True,
                )
            except Exception as exc:
                print(
                    json.dumps({"event": "probe", "ok": False, "error": str(exc)}),
                    flush=True,
                )
                return 1
        elif command == "shutdown":
            break
        elif command:
            print(
                json.dumps({"event": "unknown_command", "command": command}),
                flush=True,
            )
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--probe-once", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        return _worker_loop()
    if args.probe_once:
        print(json.dumps(probe(multi_rank=True), indent=2, sort_keys=True))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
