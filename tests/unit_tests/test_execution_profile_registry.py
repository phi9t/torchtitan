# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wave F2 concrete profile registry tests (roadmap Section 9).

The registry builds the host_static, rootfs_cpu, vllm_1gpu, and reasoning
profiles from real probes. host_static must be ready on the host with no rootfs
and no GPU (host unit tests remain ungated). A package required only by
vllm_1gpu, when absent, must block only vllm_1gpu. These tests inject a fake
environment probe set so they need no rootfs or GPU.
"""

from __future__ import annotations

from torchtitan.experiments.execution.preflight import doctor, profiles


def _env(**overrides):
    base = profiles.ProbeEnv(
        in_rootfs=False,
        available_packages=set(),
        available_executables=set(),
        gpu_count=0,
    )
    return profiles.ProbeEnv(
        in_rootfs=overrides.get("in_rootfs", base.in_rootfs),
        available_packages=overrides.get("available_packages", base.available_packages),
        available_executables=overrides.get(
            "available_executables", base.available_executables
        ),
        gpu_count=overrides.get("gpu_count", base.gpu_count),
    )


def test_host_static_is_ready_without_rootfs_or_gpu():
    # Host unit tests remain ungated: host_static needs no rootfs and no GPU.
    profile = profiles.build("host_static", env=_env())
    assert profile.evaluate().readiness == "ready"


def test_rootfs_cpu_blocked_without_rootfs():
    profile = profiles.build("rootfs_cpu", env=_env(in_rootfs=False))
    report = profile.evaluate()
    assert report.readiness == "blocked"
    assert "rootfs_not_active" in report.blocker_codes


def test_rootfs_cpu_ready_inside_rootfs_with_torch():
    profile = profiles.build(
        "rootfs_cpu", env=_env(in_rootfs=True, available_packages={"torch"})
    )
    assert profile.evaluate().readiness == "ready"


def test_vllm_1gpu_blocked_when_vllm_absent_but_rootfs_cpu_ready():
    # vllm is declared only by vllm_1gpu. Absent vllm must block only vllm_1gpu.
    env = _env(in_rootfs=True, available_packages={"torch"}, gpu_count=1)
    rootfs_cpu = profiles.build("rootfs_cpu", env=env)
    vllm_1gpu = profiles.build("vllm_1gpu", env=env)
    composed = doctor.compose([rootfs_cpu, vllm_1gpu]).evaluate()
    per_profile = {r.name: r.readiness for r in composed.profiles}
    assert per_profile["rootfs_cpu"] == "ready"
    assert per_profile["vllm_1gpu"] == "blocked"
    assert "vllm_not_importable" in composed.blocker_codes


def test_vllm_1gpu_ready_with_vllm_and_gpu():
    env = _env(
        in_rootfs=True,
        available_packages={"torch", "vllm"},
        gpu_count=1,
    )
    assert profiles.build("vllm_1gpu", env=env).evaluate().readiness == "ready"


def test_reasoning_profile_requires_datasets_and_transformers():
    env = _env(in_rootfs=True, available_packages={"torch"}, gpu_count=1)
    report = profiles.build("reasoning", env=env).evaluate()
    assert report.readiness == "blocked"
    assert "datasets_not_importable" in report.blocker_codes


def test_unknown_profile_is_a_value_error():
    import pytest

    with pytest.raises(ValueError):
        profiles.build("no_such_profile", env=_env())
