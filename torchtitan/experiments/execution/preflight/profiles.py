# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Concrete composable execution profiles (roadmap Section 9).

Each profile is a bundle of doctor clauses built from a ProbeEnv snapshot of
the host/rootfs. ProbeEnv is injected so profiles are host-testable without a
real rootfs or GPU: production code calls ``probe_env()`` to snapshot the live
environment, and tests pass a fabricated ProbeEnv.

Profiles compose (roadmap Section 9): a run builds the profiles its stages
need. Because a clause for a package exists only in the profile that declares
that package, an optional package's absence blocks only its declaring profile.

Profiles implemented here:

- ``host_static``: docs/lint/parser/unit tests with no rootfs and no GPU;
- ``rootfs_cpu``: active bwrap boundary plus torch;
- ``vllm_1gpu``: rootfs_cpu prerequisites plus vllm and at least one GPU;
- ``reasoning``: a representative reasoning lane needing datasets/transformers.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass, field

from torchtitan.experiments.execution.preflight import doctor


@dataclass(frozen=True)
class ProbeEnv:
    """An injectable snapshot of execution prerequisites.

    Keeping this pure lets the profile clauses be evaluated deterministically in
    host unit tests. ``probe_env`` builds the live snapshot for real runs.
    """

    in_rootfs: bool
    available_packages: set[str] = field(default_factory=set)
    available_executables: set[str] = field(default_factory=set)
    gpu_count: int = 0


def probe_env(*, candidate_packages: set[str] | None = None) -> ProbeEnv:
    """Snapshot the live environment.

    ``candidate_packages`` bounds the import checks to the packages the profiles
    actually declare, so probing never imports an unrelated package.
    """

    candidates = candidate_packages or _ALL_DECLARED_PACKAGES
    available = {
        name for name in candidates if importlib.util.find_spec(name) is not None
    }
    return ProbeEnv(
        in_rootfs=os.environ.get("TORCHTITAN_IN_ROOTFS") == "1",
        available_packages=available,
        available_executables={
            name for name in _ALL_DECLARED_EXECUTABLES if shutil.which(name)
        },
        gpu_count=_visible_gpu_count(),
    )


def build(name: str, *, env: ProbeEnv) -> doctor.Profile:
    """Build one named profile from an environment snapshot."""

    builder = _PROFILE_BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            f"unknown profile {name!r}; known profiles are "
            f"{sorted(_PROFILE_BUILDERS)}"
        )
    return builder(env)


# Package clauses required by each profile. Declaring a package here is what
# makes its absence block this profile and only this profile.
def _package_clause(package: str, env: ProbeEnv) -> doctor.Clause:
    present = package in env.available_packages
    return doctor.Clause(
        name=f"package_{package}",
        group="packages",
        requirement=f"rootfs Python must import {package}",
        blocker_code=f"{package}_not_importable",
        probe=lambda: doctor.ClauseResult(
            status="pass" if present else "fail",
            details={"package": package, "available": present},
        ),
    )


def _rootfs_active_clause(env: ProbeEnv) -> doctor.Clause:
    return doctor.Clause(
        name="rootfs_active",
        group="rootfs",
        requirement="real execution must run inside scripts/rootfs/enter_rootfs.sh",
        blocker_code="rootfs_not_active",
        probe=lambda: doctor.ClauseResult(
            status="pass" if env.in_rootfs else "fail",
            details={"in_rootfs": env.in_rootfs},
        ),
    )


def _gpu_count_clause(minimum: int, env: ProbeEnv) -> doctor.Clause:
    enough = env.gpu_count >= minimum
    return doctor.Clause(
        name=f"gpu_count_ge_{minimum}",
        group="gpu",
        requirement=f"profile requires at least {minimum} visible GPU(s)",
        blocker_code="insufficient_gpus",
        probe=lambda: doctor.ClauseResult(
            status="pass" if enough else "fail",
            details={"gpu_count": env.gpu_count, "minimum": minimum},
        ),
    )


def _host_static(env: ProbeEnv) -> doctor.Profile:
    # host_static intentionally declares no rootfs, package, or GPU clause, so
    # it is ready on a bare host. Its one clause proves atomic output support.
    return doctor.Profile(
        name="host_static",
        clauses=[
            doctor.Clause(
                name="host_python",
                group="attempt",
                requirement="host Python can run docs/lint/parser/unit tests",
                probe=lambda: doctor.ClauseResult(status="pass", details={}),
            )
        ],
    )


def _rootfs_cpu(env: ProbeEnv) -> doctor.Profile:
    return doctor.Profile(
        name="rootfs_cpu",
        clauses=[_rootfs_active_clause(env), _package_clause("torch", env)],
    )


def _vllm_1gpu(env: ProbeEnv) -> doctor.Profile:
    return doctor.Profile(
        name="vllm_1gpu",
        clauses=[
            _rootfs_active_clause(env),
            _package_clause("torch", env),
            _package_clause("vllm", env),
            _gpu_count_clause(1, env),
        ],
    )


def _reasoning(env: ProbeEnv) -> doctor.Profile:
    return doctor.Profile(
        name="reasoning",
        clauses=[
            _rootfs_active_clause(env),
            _package_clause("torch", env),
            _package_clause("datasets", env),
            _package_clause("transformers", env),
            _gpu_count_clause(1, env),
        ],
    )


_PROFILE_BUILDERS = {
    "host_static": _host_static,
    "rootfs_cpu": _rootfs_cpu,
    "vllm_1gpu": _vllm_1gpu,
    "reasoning": _reasoning,
}

# Every package/executable a profile may declare, so probe_env bounds its
# import and which() checks to exactly these.
_ALL_DECLARED_PACKAGES = {"torch", "vllm", "datasets", "transformers"}
_ALL_DECLARED_EXECUTABLES: set[str] = set()


def _visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        entries = [item for item in visible.split(",") if item.strip() != ""]
        return len(entries)
    # No CUDA_VISIBLE_DEVICES set: fall back to torch if it is importable.
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    return 0
