#!/usr/bin/env python3
"""Preflight gates for the modded-nanogpt B200 experiment.

Run this inside scripts/rootfs/enter_rootfs.sh. The checks intentionally fail
known-bad B200 configurations before a full trial job can spend time in warmup.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.modded_nanogpt_b200 import cli_guard


UPSTREAM_COMMIT = "ecbb586296d3dac36fd206211f25d63bad4a6b35"
EXPECTED_FINEWEB_BYTES = 2_000_010_240
EXPECTED_FINEWEB_SHARDS = 10
SCHEMA_VERSION = 1
DEFAULT_ENVIRONMENT_CLASS = "torchtitan-rootfs-b200"
FULL_TRIAL_GPU_COUNT = 2
SUPPORTED_GPU_COUNTS = (1, 2, 4, 8)
SUBPROCESS_TIMEOUT_SECONDS = 180
EXPECTED_TORCH_VERSION = "2.13.0+cu132"
EXPECTED_CUDA_RUNTIME = "13.2"
EXPECTED_TRITON_VERSION = "3.7.1"
DIRECT_RUNTIME_DEPENDENCIES = (
    ("numpy", "numpy"),
    ("tqdm", "tqdm"),
    ("huggingface_hub", "huggingface_hub"),
    ("datasets", "datasets"),
    ("tiktoken", "tiktoken"),
    ("typing_extensions", "typing_extensions"),
    ("setuptools", "setuptools"),
)
FA3_RUNTIME_DEPENDENCY = ("kernels", "kernels")


class CheckFailure(Exception):  # noqa: N818
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = detail


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_checked_subprocess(
    cmd: list[str],
    *,
    timeout_seconds: int = SUBPROCESS_TIMEOUT_SECONDS,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, _ = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, _ = proc.communicate()
        output = stdout or ""
        raise CheckFailure(
            f"subprocess timed out after {timeout_seconds}s: {' '.join(cmd)}\n{output}"
        ) from exc
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, None)


def check_rootfs() -> None:
    require(
        os.environ.get("TORCHTITAN_IN_ROOTFS") == "1",
        "must run inside scripts/rootfs/enter_rootfs.sh; TORCHTITAN_IN_ROOTFS=1 is missing",
    )
    require(
        Path.cwd() == Path("/workspace/torchtitan"),
        f"expected rootfs workspace /workspace/torchtitan, found {Path.cwd()}",
    )
    executable = Path(sys.executable).resolve()
    require(
        str(executable).startswith(("/usr/", "/opt/", "/bin/")),
        f"unexpected Python executable for rootfs verifier: {executable}",
    )
    require(
        Path("/workspace/torchtitan/scripts/rootfs/enter_rootfs.sh").exists(),
        "rootfs workspace sentinel is missing: /workspace/torchtitan/scripts/rootfs/enter_rootfs.sh",
    )


def collect_rootfs_detail() -> dict[str, Any]:
    return {
        "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "marker": os.environ.get("TORCHTITAN_IN_ROOTFS"),
        "workspace_sentinel": "/workspace/torchtitan/scripts/rootfs/enter_rootfs.sh",
    }


def load_torch():
    import torch
    import triton

    return torch, triton


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment_detail() -> dict[str, Any]:
    torch, triton = load_torch()
    return {
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "flash_attention": _package_version("flash-attn"),
    }


def check_runtime_contract(*, mode: str) -> dict[str, Any]:
    torch, triton = load_torch()
    actual = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
    }
    expected = {
        "torch": EXPECTED_TORCH_VERSION,
        "cuda_runtime": EXPECTED_CUDA_RUNTIME,
        "triton": EXPECTED_TRITON_VERSION,
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    detail = {
        "expected": expected,
        "actual": actual,
        "mismatches": mismatches,
    }
    if mode == "full" and mismatches:
        raise CheckFailure(
            "runtime contract drift: full preflight requires pinned Torch/CUDA/Triton versions",
            detail=detail,
        )
    return detail


def _required_runtime_dependencies(
    *,
    lane: str,
    attention_backend: str,
) -> list[tuple[str, str]]:
    required = list(DIRECT_RUNTIME_DEPENDENCIES)
    if lane == "A" or attention_backend == "fa3":
        required.append(FA3_RUNTIME_DEPENDENCY)
    return required


def check_direct_runtime_dependencies(
    *,
    mode: str,
    lane: str,
    attention_backend: str,
) -> dict[str, Any]:
    required = _required_runtime_dependencies(
        lane=lane,
        attention_backend=attention_backend,
    )
    required_detail = [
        {"package": package, "import_name": import_name}
        for package, import_name in required
    ]
    missing = []
    for package, import_name in required:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append({"package": package, "import_name": import_name})
    detail = {"missing": missing, "required": required_detail}
    if mode == "full" and missing:
        raise CheckFailure(
            "missing direct runtime dependencies for full preflight",
            detail=detail,
        )
    return detail


def check_gpu_inventory(expected_gpus: int, expected_name: str) -> list[dict[str, Any]]:
    torch, _ = load_torch()
    require(torch.cuda.is_available(), "torch.cuda.is_available() is false")
    count = torch.cuda.device_count()
    require(
        count == expected_gpus, f"expected {expected_gpus} CUDA devices, found {count}"
    )
    devices = []
    for idx in range(count):
        name = torch.cuda.get_device_name(idx)
        capability = torch.cuda.get_device_capability(idx)
        require(
            expected_name in name,
            f"device {idx} expected name containing {expected_name!r}, found {name!r}",
        )
        require(
            capability >= (10, 0),
            f"device {idx} expected Blackwell/B200 capability >= (10, 0), found {capability}",
        )
        devices.append(
            {"index": idx, "name": name, "compute_capability": list(capability)}
        )
    return devices


def check_torch_primitives() -> None:
    torch, _ = load_torch()
    device = "cuda:0"
    bf16 = torch.ones((16, 16), device=device, dtype=torch.bfloat16)
    require(bf16.sum().item() == 256, "BF16 allocation/reduction smoke failed")
    fp8 = torch.ones((16, 16), device=device, dtype=torch.float8_e4m3fn)
    require(fp8.dtype is torch.float8_e4m3fn, "FP8 tensor creation smoke failed")
    require(hasattr(torch, "_scaled_mm"), "torch._scaled_mm is missing")
    a = torch.randn((16, 16), device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    b = torch.randn((16, 16), device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    scale = torch.ones((), device=device, dtype=torch.float32)
    out = torch._scaled_mm(
        a,
        b,
        out_dtype=torch.bfloat16,
        scale_a=scale,
        scale_b=scale,
        use_fast_accum=True,
    )
    require(
        out.shape == (16, 16), "torch._scaled_mm FP8 smoke returned the wrong shape"
    )
    torch.cuda.synchronize()


def run_nccl_worker() -> int:
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cuda:nccl,cpu:gloo", device_id=torch.device("cuda", local_rank)
    )
    x = torch.ones(1, device=f"cuda:{local_rank}") * (local_rank + 1)
    dist.all_reduce(x)
    expected = sum(range(1, dist.get_world_size() + 1))
    ok = x.item() == expected
    dist.destroy_process_group()
    if not ok:
        print(f"NCCL all_reduce expected {expected}, got {x.item()}", file=sys.stderr)
        return 1
    return 0


def check_nccl(expected_gpus: int) -> None:
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={expected_gpus}",
        str(Path(__file__).resolve()),
        "--_nccl-worker",
    ]
    proc = run_checked_subprocess(cmd, timeout_seconds=SUBPROCESS_TIMEOUT_SECONDS)
    require(
        proc.returncode == 0,
        f"{expected_gpus}-rank NCCL all-reduce smoke failed:\n{proc.stdout}",
    )


def check_source(
    source: Path, lane: str, attention_backend: str, mlp_backend: str
) -> str:
    require(source.exists(), f"source directory does not exist: {source}")
    require((source / "train_gpt.py").exists(), f"missing train_gpt.py under {source}")
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    require(
        commit == UPSTREAM_COMMIT,
        f"expected upstream commit {UPSTREAM_COMMIT}, found {commit}",
    )
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--short"], text=True
    )
    if lane == "A":
        require(
            status.strip() == "",
            f"Lane A requires clean upstream source; dirty status:\n{status}",
        )
    if lane == "B":
        require(
            (source / "triton_kernels.py").exists(),
            f"Lane B source must include triton_kernels.py under {source}",
        )
    triton_source = (source / "triton_kernels.py").read_text()
    has_env_ce = "MODDED_NANOGPT_CE_COMPUTE_CAPABILITY" in triton_source
    if lane == "B":
        require(
            has_env_ce,
            "Lane B B200 variant must make the CE custom-kernel compute capability configurable",
        )
        require(
            os.environ.get("MODDED_NANOGPT_CE_COMPUTE_CAPABILITY") == "100",
            "set MODDED_NANOGPT_CE_COMPUTE_CAPABILITY=100 before importing the variant on B200",
        )
    if lane == "A" and 'compute_capability="90"' in triton_source:
        print(
            "warning: upstream source hardcodes the CE custom kernel for sm90; "
            "Lane A may fail after FA3 even if FA3 becomes available",
            file=sys.stderr,
        )
    if attention_backend == "flex":
        raise CheckFailure(
            "MODDED_NANOGPT_ATTN_BACKEND=flex is blocked for full jobs: prior full-width "
            "validation attempted a 256 GiB block-mask allocation"
        )
    if mlp_backend == "torch" and os.environ.get("TORCHDYNAMO_DISABLE") == "1":
        raise CheckFailure(
            "MODDED_NANOGPT_MLP_BACKEND=torch with TORCHDYNAMO_DISABLE=1 is blocked for full jobs: "
            "prior eager warmup OOMed during validation"
        )
    return status


def check_fa3_smoke() -> None:
    code = r"""
import torch
from kernels import get_kernel

interface = get_kernel("kernels-community/flash-attn3", version=1).flash_attn_interface
cu = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
q = torch.randn(256, 6, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
y = interface.flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q=cu,
    cu_seqlens_k=cu,
    max_seqlen_q=128,
    max_seqlen_k=128,
    causal=True,
    softmax_scale=0.1,
    window_size=(64, 0),
)
torch.cuda.synchronize()
assert y.shape == q.shape, y.shape
print("fa3_smoke_ok")
"""
    proc = run_checked_subprocess(
        [sys.executable, "-c", code], timeout_seconds=SUBPROCESS_TIMEOUT_SECONDS
    )
    if proc.returncode != 0:
        raise CheckFailure("FA3 varlen/window smoke failed:\n" + proc.stdout)


def check_fa2_smoke() -> None:
    code = r"""
import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func

cu = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
q = torch.randn(256, 6, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
y = flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q=cu,
    cu_seqlens_k=cu,
    max_seqlen_q=128,
    max_seqlen_k=128,
    causal=True,
    softmax_scale=0.1,
    window_size=(64, 0),
)
torch.cuda.synchronize()
assert y.shape == q.shape, y.shape
print("fa2_smoke_ok")
"""
    proc = run_checked_subprocess(
        [sys.executable, "-c", code], timeout_seconds=SUBPROCESS_TIMEOUT_SECONDS
    )
    if proc.returncode != 0:
        raise CheckFailure("FA2 varlen/window smoke failed:\n" + proc.stdout)


def check_attention(lane: str, attention_backend: str) -> None:
    if lane == "A":
        require(attention_backend == "fa3", "Lane A must use upstream FA3 attention")
        check_fa3_smoke()
        return
    if attention_backend == "fa3":
        check_fa3_smoke()
    elif attention_backend == "fa2":
        check_fa2_smoke()
    else:
        raise CheckFailure(
            f"unsupported attention backend for full jobs: {attention_backend}"
        )


def check_mlp_backend(
    source: Path, mlp_backend: str, allow_previous_stall: bool
) -> dict[str, Any]:
    if mlp_backend == "triton":
        smoke_detail = _run_triton_mlp_smoke(source)
        return {"backend": "triton", "local_smoke": smoke_detail}
    if mlp_backend != "torch":
        raise CheckFailure(f"unsupported MLP backend: {mlp_backend}")

    smoke_detail = _run_torch_mlp_smoke(source)
    if os.environ.get("TORCH_COMPILE_DISABLE") == "1":
        return {
            "backend": "torch",
            "local_smoke": smoke_detail,
            "full_mode_policy": "compile_disabled_prerequisite",
            "claim_eligible": False,
        }
    if not allow_previous_stall:
        raise CheckFailure(
            "MODDED_NANOGPT_MLP_BACKEND=torch passes a local smoke, but the only full-job options "
            "seen so far are blocked: eager OOMs and graph-break compile stalls. Use "
            "--allow-previous-stall only for diagnostic launches, not full jobs.",
            detail={
                "backend": "torch",
                "local_smoke": smoke_detail,
                "full_mode_policy": "blocked_without_allow_previous_stall",
            },
        )
    return {"backend": "torch", "local_smoke": smoke_detail}


def _triton_mlp_smoke_code(source: Path) -> str:
    return rf"""
import importlib
import json
import os
import sys

os.environ["MODDED_NANOGPT_MLP_BACKEND"] = "triton"
sys.path.insert(0, {str(source)!r})

import torch

importlib.invalidate_caches()
module = importlib.import_module("triton_kernels")
fn = module.FusedLinearReLUSquareFunction.apply
x = torch.randn(2, 16, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
w1 = torch.randn(3072, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
w2 = torch.randn(3072, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = fn(x, w1, w2)
y.float().mean().backward()
torch.cuda.synchronize()
assert y.shape == x.shape, y.shape
print(json.dumps({{"backend": "triton", "output_shape": list(y.shape)}}))
"""


def _triton_mlp_failure_detail(stdout: str, returncode: int) -> dict[str, Any]:
    failure_class = "triton_compile_or_runtime"
    failure_site = None
    if "Expected size for first two dimensions of batch2 tensor" in stdout:
        failure_class = "triton_backward_shape_mismatch"
        failure_site = "FusedLinearReLUSquareFunction.backward"
    compiler_pass = (
        "TritonNvidiaGPUOptimizeTMemLayoutsPass"
        if "TritonNvidiaGPUOptimizeTMemLayoutsPass" in stdout
        else None
    )
    return {
        "backend": "triton",
        "blocked_kernel": "linear_relu_square_kernel",
        "blocked_arch": "sm100",
        "failure_class": failure_class,
        "failure_site": failure_site,
        "compiler_pass": compiler_pass,
        "returncode": returncode,
        "stdout": stdout,
    }


def _run_triton_mlp_smoke(
    source: Path,
    *,
    timeout_seconds: int = SUBPROCESS_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    os.environ.setdefault("MODDED_NANOGPT_CE_COMPUTE_CAPABILITY", "100")
    proc = run_checked_subprocess(
        [sys.executable, "-c", _triton_mlp_smoke_code(source)],
        timeout_seconds=timeout_seconds,
    )
    if proc.returncode != 0:
        raise CheckFailure(
            "Triton MLP smoke failed",
            detail=_triton_mlp_failure_detail(proc.stdout, proc.returncode),
        )
    try:
        detail = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise CheckFailure(
            "Triton MLP smoke produced malformed output",
            detail={"backend": "triton", "stdout": proc.stdout},
        ) from exc
    if not isinstance(detail, dict):
        raise CheckFailure(
            "Triton MLP smoke output was not a JSON object",
            detail={"backend": "triton", "stdout": proc.stdout},
        )
    return detail


def _run_torch_mlp_smoke(source: Path) -> dict[str, Any]:
    torch, _ = load_torch()
    old_path = list(sys.path)
    sys.path.insert(0, str(source))
    try:
        importlib.invalidate_caches()
        module = importlib.import_module("triton_kernels")
        fn = module.FusedLinearReLUSquareFunction.apply
        x = torch.randn(
            2, 16, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w1 = torch.randn(
            3072, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w2 = torch.randn(
            3072, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        y = fn(x, w1, w2)
        y.float().mean().backward()
        torch.cuda.synchronize()
        require(y.shape == x.shape, "PyTorch MLP fallback returned the wrong shape")
        return {"backend": "torch", "output_shape": list(y.shape)}
    except Exception as exc:  # noqa: BLE001
        raise CheckFailure(f"PyTorch MLP fallback smoke failed: {exc}") from exc
    finally:
        sys.path[:] = old_path


def check_data_manifest(path: Path, *, mode: str, verify_sha: bool) -> dict[str, Any]:
    require(path.exists(), f"data manifest does not exist: {path}")
    data = json.loads(path.read_text())
    schema_version = data.get("schema_version")
    require(
        schema_version == SCHEMA_VERSION,
        f"manifest schema_version must be {SCHEMA_VERSION}, found {schema_version!r}",
    )
    require(
        data.get("dataset") == "fineweb10B",
        f"manifest dataset must be fineweb10B, found {data.get('dataset')!r}",
    )
    files = data.get("files")
    require(
        isinstance(files, list) and files,
        "manifest must contain a non-empty files list",
    )
    if mode == "full":
        require(
            data.get("token_budget") == "900M",
            f"full runs require token_budget '900M', found {data.get('token_budget')!r}",
        )
        require(
            len(files) == EXPECTED_FINEWEB_SHARDS,
            f"full runs require exactly {EXPECTED_FINEWEB_SHARDS} .bin shards, found {len(files)}",
        )
        require(
            data.get("source", {}).get("commit") == UPSTREAM_COMMIT,
            f"full data manifest source commit must be {UPSTREAM_COMMIT}, found {data.get('source', {}).get('commit')!r}",
        )
        require(
            isinstance(data.get("command"), list) and data["command"],
            "full data manifest must record preparation command",
        )
        if verify_sha:
            require(
                data.get("verified_sha") is True,
                "full data manifest must declare verified_sha true when SHA verification is requested",
            )
    total = 0
    for item in files:
        require(
            str(item.get("path", "")).endswith(".bin"),
            f"manifest shard is not a .bin file: {item.get('path')!r}",
        )
        require("bytes" in item, f"manifest shard missing bytes: {item}")
        require(
            "sha256" in item and item["sha256"],
            f"manifest shard missing sha256: {item.get('path')!r}",
        )
        shard = Path(item["path"])
        require(shard.exists(), f"manifest shard missing: {shard}")
        size = shard.stat().st_size
        require(
            size == item["bytes"],
            f"manifest size mismatch for {shard}: {size} != {item['bytes']}",
        )
        total += size
        if verify_sha:
            actual = sha256_file(shard)
            require(
                actual == item["sha256"],
                f"sha256 mismatch for {shard}: {actual} != {item['sha256']}",
            )
    if mode == "full":
        require(
            total == EXPECTED_FINEWEB_BYTES,
            f"expected FineWeb total {EXPECTED_FINEWEB_BYTES}, found {total}",
        )
        manifest_total = data.get("total_bytes")
        require(
            manifest_total == total,
            f"manifest total_bytes mismatch: {manifest_total} != {total}",
        )
    return {
        "dataset": data.get("dataset"),
        "token_budget": data.get("token_budget"),
        "num_files": len(files),
        "total_bytes": total,
        "verified_sha": verify_sha,
        "manifest_verified_sha": data.get("verified_sha"),
    }


def write_report(report_path: Path | None, report: dict[str, Any]) -> None:
    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tmp.replace(report_path)


def derive_claim_label(
    lane: str, mode: str, attention_backend: str, mlp_backend: str
) -> str:
    if mode == "smoke":
        return "smoke"
    if mode == "diagnostic":
        return "diagnostic"
    if lane == "A":
        return "B200 upstream reproduction"
    if mode == "full" and lane == "B" and mlp_backend == "torch":
        return "B200 prerequisite torch-MLP fallback"
    if lane == "B" and attention_backend == "fa2":
        return "B200 compatibility patchset"
    return "B200 systems-only"


def derive_evidence_tier(mode: str) -> str:
    if mode == "full":
        return "full-single-attempt"
    return mode


def check_mode_policy(args: argparse.Namespace) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "mode": args.mode,
        "skip_nccl": args.skip_nccl,
        "verify_sha": args.verify_sha,
        "allow_previous_stall": args.allow_previous_stall,
        "expected_gpus": args.expected_gpus,
        "attention_backend": args.attention_backend,
        "mlp_backend": args.mlp_backend,
    }
    if args.mode == "full":
        require(
            not args.skip_nccl,
            "full preflight must include NCCL; --skip-nccl is diagnostic-only",
        )
        require(
            args.verify_sha,
            "full preflight requires --verify-sha so data preservation is proven",
        )
        require(
            not args.allow_previous_stall,
            "full preflight cannot use --allow-previous-stall",
        )
        require(
            args.expected_gpus in SUPPORTED_GPU_COUNTS,
            "full preflight requires expected-gpus in [1, 2, 4, 8], "
            f"found --expected-gpus={args.expected_gpus}",
        )
        require(
            args.expected_name == "B200",
            f"full preflight requires expected GPU name B200, found {args.expected_name!r}",
        )
    if args.mode != "diagnostic":
        require(not args.skip_nccl, "--skip-nccl is allowed only for diagnostic mode")
        require(
            not args.allow_previous_stall,
            "--allow-previous-stall is allowed only for diagnostic mode",
        )
    return policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=False, choices=["full", "smoke", "diagnostic"]
    )
    parser.add_argument("--lane", choices=["A", "B"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--arm")
    parser.add_argument("--claim-label")
    parser.add_argument("--evidence-tier")
    parser.add_argument("--environment-class", default=DEFAULT_ENVIRONMENT_CLASS)
    parser.add_argument(
        "--attention-backend",
        choices=["fa3", "fa2", "flex"],
        default=os.environ.get("MODDED_NANOGPT_ATTN_BACKEND", "fa3"),
    )
    parser.add_argument(
        "--mlp-backend",
        choices=["triton", "torch"],
        default=os.environ.get("MODDED_NANOGPT_MLP_BACKEND", "triton"),
    )
    parser.add_argument("--expected-gpus", type=int, default=FULL_TRIAL_GPU_COUNT)
    parser.add_argument("--expected-name", default="B200")
    parser.add_argument(
        "--skip-nccl",
        action="store_true",
        help="only for local iteration, never before a full job",
    )
    parser.add_argument(
        "--verify-sha", action="store_true", help="rehash all FineWeb shards"
    )
    parser.add_argument(
        "--allow-previous-stall",
        action="store_true",
        help="permit known-stalled diagnostic configs",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--_nccl-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._nccl_worker:
        return args
    missing = [
        flag
        for flag, value in (
            ("--mode", args.mode),
            ("--lane", args.lane),
            ("--source", args.source),
            ("--data-manifest", args.data_manifest),
        )
        if value is None
    ]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if args.run_id is None:
        args.run_id = f"modded_nanogpt_b200_{args.lane.lower()}_{args.mode}_{timestamp}"
    if args.attempt_id is None:
        args.attempt_id = f"{args.run_id}_attempt_001"
    if args.arm is None:
        args.arm = "A0" if args.lane == "A" else "B0"
    if args.claim_label is None:
        args.claim_label = derive_claim_label(
            args.lane,
            args.mode,
            args.attention_backend,
            args.mlp_backend,
        )
    if args.evidence_tier is None:
        args.evidence_tier = derive_evidence_tier(args.mode)
    return args


def main(*, enforce_rootfs: bool = False) -> int:
    args = parse_args()
    if enforce_rootfs:
        guard_exit = cli_guard.guard_rootfs_cli(
            "experiments/modded_nanogpt_b200/run_preflight.sh"
        )
        if guard_exit is not None:
            return guard_exit
    if args._nccl_worker:
        return run_nccl_worker()

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "checks": [],
        "failures": [],
        "classification": {
            "lane": args.lane,
            "mode": args.mode,
            "arm": args.arm,
            "claim_label": args.claim_label,
            "evidence_tier": args.evidence_tier,
            "run_id": args.run_id,
            "attempt_id": args.attempt_id,
            "environment_class": args.environment_class,
            "claim_eligible": False,
        },
    }

    def checked(name: str, fn):
        try:
            value = fn()
        except CheckFailure as exc:
            item = {"name": name, "ok": False, "error": str(exc)}
            if exc.detail is not None:
                item["detail"] = exc.detail
            report["checks"].append(item)
            report["failures"].append({"name": name, "error": str(exc)})
            raise
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            report["checks"].append({"name": name, "ok": False, "error": error})
            report["failures"].append({"name": name, "error": error})
            raise CheckFailure(error) from exc
        else:
            item = {"name": name, "ok": True}
            if value is not None:
                item["detail"] = value
            report["checks"].append(item)
            return value

    def check_torch_import() -> dict[str, Any]:
        detail = collect_environment_detail()
        report["environment"] = detail
        return detail

    try:
        checked("mode_policy", lambda: check_mode_policy(args))
        checked("rootfs", lambda: (check_rootfs(), collect_rootfs_detail())[1])
        checked("torch_import", check_torch_import)
        checked("runtime_contract", lambda: check_runtime_contract(mode=args.mode))
        checked(
            "direct_runtime_dependencies",
            lambda: check_direct_runtime_dependencies(
                mode=args.mode,
                lane=args.lane,
                attention_backend=args.attention_backend,
            ),
        )
        report["gpus"] = checked(
            "gpu_inventory",
            lambda: check_gpu_inventory(args.expected_gpus, args.expected_name),
        )
        checked("torch_primitives", check_torch_primitives)
        if not args.skip_nccl:
            checked("nccl_all_reduce", lambda: check_nccl(args.expected_gpus))
        else:
            report["checks"].append(
                {"name": "nccl_all_reduce", "ok": "skipped", "mode": args.mode}
            )
        checked(
            "source_policy",
            lambda: {
                "status": check_source(
                    args.source, args.lane, args.attention_backend, args.mlp_backend
                )
            },
        )
        checked(
            "data_manifest",
            lambda: check_data_manifest(
                args.data_manifest, mode=args.mode, verify_sha=args.verify_sha
            ),
        )
        checked(
            "attention_backend",
            lambda: check_attention(args.lane, args.attention_backend),
        )
        mlp_detail = checked(
            "mlp_backend",
            lambda: check_mlp_backend(
                args.source, args.mlp_backend, args.allow_previous_stall
            ),
        )
        report["ok"] = True
        report["classification"]["claim_eligible"] = args.mode == "full" and not (
            isinstance(mlp_detail, dict) and mlp_detail.get("claim_eligible") is False
        )
        write_report(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except CheckFailure as exc:
        write_report(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        print(
            "\npreflight failed:\n" + textwrap.indent(str(exc), "  "), file=sys.stderr
        )
        return 21
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        report["checks"].append({"name": "runtime_error", "ok": False, "error": error})
        report["failures"].append({"name": "runtime_error", "error": error})
        write_report(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        print(
            "\npreflight runtime error:\n" + textwrap.indent(error, "  "),
            file=sys.stderr,
        )
        return 21


if __name__ == "__main__":
    raise SystemExit(main(enforce_rootfs=True))
