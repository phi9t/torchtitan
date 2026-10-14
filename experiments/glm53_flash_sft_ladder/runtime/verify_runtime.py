# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Verify the glm53_ep rootfs Python runtime and emit a JSON receipt."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as importlib_metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EXPECTED_ENV_NAME = "glm53_ep"
EXPECTED_PYTHON = "/project/venvs/b200-runtime/bin/python"
EXPECTED_PACKAGES = {
    "torch": "2.16.0.dev20261013+cu132",
    "triton": "3.9.0+gitfc8d31a5",
    "fla-core": "0.5.2",
    "torchao": "0.18.0",
    "nvidia-nvshmem-cu13": "3.7.2",
}
CUDA_HEADERS = (
    "/opt/cuda-synth/include/cuda_runtime.h",
    "/opt/cuda-synth/include/cublas_v2.h",
    "/opt/cuda-synth/include/nvrtc.h",
)
CUDA_LIBS = (
    "/opt/cuda-synth/lib64/libcudart.so",
    "/opt/cuda-synth/lib64/libcublas.so",
    "/opt/cuda-synth/lib64/libnvrtc.so",
)


def _blocker(phase: str, message: str, **detail: object) -> dict[str, object]:
    return {"phase": phase, "message": message, **detail}


def _check_package(
    blockers: list[dict[str, object]], packages: dict[str, str], name: str, version: str
) -> None:
    try:
        actual = importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        blockers.append(_blocker("package", f"missing package {name}", name=name))
        return
    packages[name] = actual
    if actual != version:
        blockers.append(
            _blocker(
                "package",
                f"unexpected package version for {name}",
                name=name,
                expected=version,
                actual=actual,
            )
        )


def _check_import(
    blockers: list[dict[str, object]], imports: dict[str, Any], module: str
) -> None:
    try:
        imported = importlib.import_module(module)
    except Exception as exc:
        imports[module] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        blockers.append(
            _blocker(
                "import",
                f"failed to import {module}",
                module=module,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        )
        return
    imports[module] = {"ok": True, "file": getattr(imported, "__file__", None)}


def _load_rootfs_manifest() -> dict[str, Any]:
    path = Path("/build_manifest.json")
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {"manifest_error": str(exc)}
    return payload if isinstance(payload, dict) else {"manifest_error": "not object"}


def _nvcc_version() -> dict[str, object]:
    nvcc = "/opt/cuda-synth/bin/nvcc"
    if not Path(nvcc).is_file():
        nvcc = shutil.which("nvcc") or ""
    if not nvcc:
        return {"ok": False, "error": "nvcc not found"}
    proc = subprocess.run(
        [nvcc, "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return {
        "ok": proc.returncode == 0,
        "path": nvcc,
        "returncode": proc.returncode,
        "output": proc.stdout.splitlines(),
    }


def verify_runtime() -> dict[str, Any]:
    blockers: list[dict[str, object]] = []
    packages: dict[str, str] = {}
    imports: dict[str, Any] = {}

    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        blockers.append(
            _blocker(
                "rootfs",
                "TORCHTITAN_IN_ROOTFS is not set to 1",
                actual=os.environ.get("TORCHTITAN_IN_ROOTFS"),
            )
        )
    if os.environ.get("TORCHTITAN_ROOTFS_ENV") != EXPECTED_ENV_NAME:
        blockers.append(
            _blocker(
                "rootfs_env",
                "unexpected TORCHTITAN_ROOTFS_ENV",
                expected=EXPECTED_ENV_NAME,
                actual=os.environ.get("TORCHTITAN_ROOTFS_ENV"),
            )
        )
    if os.environ.get("PYTHON") != EXPECTED_PYTHON:
        blockers.append(
            _blocker(
                "python_path",
                "unexpected PYTHON environment path",
                expected=EXPECTED_PYTHON,
                actual=os.environ.get("PYTHON"),
            )
        )
    if Path(sys.executable).resolve() != Path(EXPECTED_PYTHON).resolve():
        blockers.append(
            _blocker(
                "python_executable",
                "runtime script is not running under the managed venv Python",
                expected=EXPECTED_PYTHON,
                actual=sys.executable,
            )
        )

    for name, version in EXPECTED_PACKAGES.items():
        _check_package(blockers, packages, name, version)

    _check_import(blockers, imports, "triton")
    _check_import(blockers, imports, "fla.ops.kda")
    _check_import(blockers, imports, "torchao.prototype.mx_formats")
    _check_import(blockers, imports, "nvidia.nvshmem")

    import torch

    torch_info: dict[str, Any] = {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.version.cuda != "13.2":
        blockers.append(
            _blocker(
                "torch_cuda",
                "torch was not built for CUDA 13.2",
                expected="13.2",
                actual=torch.version.cuda,
            )
        )
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        torch_info["device_0_capability"] = list(capability)
        torch_info["device_0_name"] = torch.cuda.get_device_name(0)
        if capability != (10, 0):
            blockers.append(
                _blocker(
                    "gpu_arch",
                    "device 0 is not sm_100",
                    expected=[10, 0],
                    actual=list(capability),
                )
            )
    else:
        blockers.append(_blocker("cuda_available", "torch.cuda.is_available is false"))

    filesystem_checks: dict[str, bool] = {}
    for path in CUDA_HEADERS + CUDA_LIBS:
        exists = Path(path).exists()
        filesystem_checks[path] = exists
        if not exists:
            blockers.append(
                _blocker("cuda_file", "required CUDA file is missing", path=path)
            )

    rootfs_manifest = _load_rootfs_manifest()
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm53_ep_runtime_verification",
        "ok": not blockers,
        "env": {
            "TORCHTITAN_ROOTFS_ENV": os.environ.get("TORCHTITAN_ROOTFS_ENV"),
            "TORCHTITAN_ROOTFS_STORE_ID": os.environ.get("TORCHTITAN_ROOTFS_STORE_ID"),
            "TORCHTITAN_ROOTFS_HOST_STATE": os.environ.get(
                "TORCHTITAN_ROOTFS_HOST_STATE"
            ),
            "PYTHON": os.environ.get("PYTHON"),
            "sys_executable": sys.executable,
        },
        "rootfs_manifest": rootfs_manifest,
        "rootfs_digest": rootfs_manifest.get("exported_rootfs_digest"),
        "packages": packages,
        "imports": imports,
        "torch": torch_info,
        "nvcc": _nvcc_version(),
        "filesystem": filesystem_checks,
        "blockers": blockers,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/project/logs/glm53_ep_runtime_verification.json"),
    )
    args = parser.parse_args(argv)

    report = verify_runtime()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 21


if __name__ == "__main__":
    raise SystemExit(main())
