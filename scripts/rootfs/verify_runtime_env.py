#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validate the bwrap plan emitted by scripts/rootfs/enter_rootfs.sh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
WORKSPACE = "/workspace/torchtitan"
REQUIRED_PATH = ":".join(
    (
        "/project/venvs/b200-runtime/bin",
        "/project/mise/data/shims",
        "/usr/local/cuda/bin",
        "/opt/cuda-synth/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
    )
)
REQUIRED_ENV = {
    "PATH": REQUIRED_PATH,
    "CUDA_HOME": "/opt/cuda-synth",
    "CUDA_PATH": "/opt/cuda-synth",
    "TORCHTITAN_IN_ROOTFS": "1",
    "TORCHTITAN_ROOTFS_ENV": "modded_nanogpt_b200",
    "TORCHTITAN_ROOTFS_PROJECT": WORKSPACE,
    "TORCHTITAN_ROOTFS_LOG_DIR": "/project/logs",
    "HOME": "/project/home",
    "XDG_CACHE_HOME": "/project/xdg-cache",
    "UV_CACHE_DIR": "/project/uv-cache",
    "PIP_CACHE_DIR": "/project/pip-cache",
    "MISE_DATA_DIR": "/project/mise/data",
    "MISE_CACHE_DIR": "/project/mise/cache",
    "MISE_CONFIG_DIR": f"{WORKSPACE}/experiments/modded_nanogpt_b200/runtime",
    "PYTHON": "/project/venvs/b200-runtime/bin/python",
    "TMPDIR": "/project/tmp",
    "TEMP": "/project/tmp",
    "TMP": "/project/tmp",
    "HF_HOME": f"{WORKSPACE}/.cache/huggingface",
    "HF_HUB_CACHE": f"{WORKSPACE}/.cache/huggingface/hub",
    "TORCH_HOME": f"{WORKSPACE}/.cache/torch",
    "MPLCONFIGDIR": "/project/xdg-cache/matplotlib",
}


class BwrapPlanError(ValueError):
    """Raised when an emitted bwrap plan fails the launch contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BwrapPlanError(message)


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _as_list(value: Any, name: str) -> list[Any]:
    _require(isinstance(value, list), f"{name} must be a list")
    return value


def _mount_targets(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mounts: dict[str, dict[str, Any]] = {}
    for item in _as_list(plan.get("mounts"), "mounts"):
        _require(isinstance(item, dict), "mount entries must be objects")
        target = item.get("target")
        _require(bool(isinstance(target, str) and target), "mount target is required")
        mounts[target] = item
    return mounts


def _validate_required_mount(
    mounts: dict[str, dict[str, Any]],
    *,
    target: str,
    kind: str | None = None,
) -> dict[str, Any]:
    _require(target in mounts, f"required mount missing: {target}")
    mount = mounts[target]
    if kind is not None:
        _require(
            mount.get("kind") == kind,
            f"mount {target} expected kind {kind}, found {mount.get('kind')}",
        )
    return mount


def validate_bwrap_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate a rootfs bwrap plan and return a structured success report."""

    _require(plan.get("schema_version") == SCHEMA_VERSION, "unsupported schema_version")
    rootfs = _as_mapping(plan.get("rootfs"), "rootfs")
    rootfs_path = rootfs.get("path")
    _require(
        bool(isinstance(rootfs_path, str) and rootfs_path),
        "rootfs.path is required",
    )
    _require(plan.get("cwd") == WORKSPACE, f"cwd must be {WORKSPACE}")
    _require(
        plan.get("network_mode") in {"offline", "networked"}, "invalid network_mode"
    )
    _require(
        bool(isinstance(plan.get("inner_argv"), list) and plan["inner_argv"]),
        "inner_argv must be a non-empty list",
    )

    mounts = _mount_targets(plan)
    root_mount = _validate_required_mount(mounts, target="/", kind="bind")
    _require(
        root_mount.get("source") == rootfs_path,
        "rootfs mount source must match rootfs.path",
    )
    repo_mount = _validate_required_mount(mounts, target=WORKSPACE, kind="bind")
    _require(
        bool(isinstance(repo_mount.get("source"), str) and repo_mount["source"]),
        f"{WORKSPACE} mount source is required",
    )
    _validate_required_mount(mounts, target="/proc", kind="proc")
    _validate_required_mount(mounts, target="/tmp", kind="tmpfs")
    _validate_required_mount(mounts, target="/dev", kind="dev")

    environment = _as_mapping(plan.get("environment"), "environment")
    for key, expected in REQUIRED_ENV.items():
        _require(
            environment.get(key) == expected,
            f"{key} expected {expected!r}, found {environment.get(key)!r}",
        )
    rootfs_store_id = rootfs.get("store_id")
    _require(
        bool(isinstance(rootfs_store_id, str) and rootfs_store_id),
        "rootfs.store_id is required",
    )
    _require(
        environment.get("TORCHTITAN_ROOTFS_STORE_ID") == rootfs_store_id,
        "TORCHTITAN_ROOTFS_STORE_ID must match rootfs.store_id",
    )
    _require(
        "NVIDIA_VISIBLE_DEVICES" in environment,
        "NVIDIA_VISIBLE_DEVICES is required",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "bwrap_plan",
        "ok": True,
        "blockers": [],
        "rootfs": rootfs,
        "network_mode": plan["network_mode"],
        "mount_count": len(mounts),
        "inner_argv": plan["inner_argv"],
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise BwrapPlanError("plan JSON must be an object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a TorchTitan bwrap plan.")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    try:
        report = validate_bwrap_plan(_load_json(args.plan))
    except (BwrapPlanError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "phase": "bwrap_plan",
            "ok": False,
            "blockers": [{"phase": "bwrap_plan", "message": str(exc)}],
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
