# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "experiments/modded_nanogpt_b200/launch_nanogpt_rootfs.sh"
FULL_LAUNCHER = (
    REPO_ROOT / "experiments/modded_nanogpt_b200/launch_nanogpt_2gpu_full_rootfs.sh"
)


def _minimal_rootfs(tmp_path: Path) -> Path:
    rootfs = tmp_path / "rootfs"
    bash = rootfs / "bin" / "bash"
    bash.parent.mkdir(parents=True)
    bash.write_text("#!/usr/bin/env bash\n")
    bash.chmod(0o755)
    return rootfs


def test_launch_nanogpt_rootfs_rejects_arguments():
    proc = subprocess.run(
        ["bash", str(LAUNCHER), "--mode", "full"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 2
    assert "does not accept arguments" in proc.stdout


def test_launch_nanogpt_2gpu_full_rootfs_rejects_arguments():
    proc = subprocess.run(
        ["bash", str(FULL_LAUNCHER), "--mode", "smoke"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 2
    assert "does not accept arguments" in proc.stdout


def test_launch_nanogpt_rootfs_outer_path_emits_plan_for_launcher(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    result_root = REPO_ROOT / "experiments/modded_nanogpt_b200/results"
    before = {path.name for path in result_root.glob("nanogpt_rootfs_smoke_*")}

    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
            "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(tmp_path / "outer_plan.json"),
            "TORCHTITAN_ROOTFS_DIR": str(rootfs),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    after = {path.name for path in result_root.glob("nanogpt_rootfs_smoke_*")}
    created = sorted(after - before)
    assert proc.returncode == 0, proc.stdout
    assert len(created) == 1
    result_dir = result_root / created[0]
    plan = json.loads((result_dir / "rootfs_plan.json").read_text())
    assert plan["inner_argv"][:2] == ["/bin/bash", "-lc"]
    assert "experiments/modded_nanogpt_b200/launch_nanogpt_rootfs.sh" in (
        plan["inner_argv"][2]
    )
    assert f"MODDED_NANOGPT_ROOTFS_RUN_ID='{created[0]}'" in plan["inner_argv"][2]
    assert "rootfs prep: emitting bwrap launch plan" in (
        result_dir / "operator_launch.log"
    ).read_text()


def test_launch_nanogpt_2gpu_full_rootfs_outer_path_emits_plan_for_launcher(
    tmp_path: Path,
):
    rootfs = _minimal_rootfs(tmp_path)
    result_root = REPO_ROOT / "experiments/modded_nanogpt_b200/results"
    before = {path.name for path in result_root.glob("nanogpt_2gpu_full_*")}

    proc = subprocess.run(
        ["bash", str(FULL_LAUNCHER)],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
            "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(tmp_path / "outer_plan.json"),
            "TORCHTITAN_ROOTFS_DIR": str(rootfs),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    after = {path.name for path in result_root.glob("nanogpt_2gpu_full_*")}
    created = sorted(after - before)
    assert proc.returncode == 0, proc.stdout
    assert len(created) == 1
    result_dir = result_root / created[0]
    plan = json.loads((result_dir / "rootfs_plan.json").read_text())
    assert plan["inner_argv"][:2] == ["/bin/bash", "-lc"]
    assert "experiments/modded_nanogpt_b200/launch_nanogpt_2gpu_full_rootfs.sh" in (
        plan["inner_argv"][2]
    )
    assert f"MODDED_NANOGPT_2GPU_FULL_RUN_ID='{created[0]}'" in plan["inner_argv"][2]
    assert "rootfs prep: emitting bwrap launch plan" in (
        result_dir / "operator_launch.log"
    ).read_text()


def test_launch_nanogpt_rootfs_rejects_forged_rootfs_on_host(tmp_path: Path):
    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=tmp_path,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_IN_ROOTFS": "1",
            "CUDA_HOME": "/opt/cuda-synth",
            "CUDA_PATH": "/opt/cuda-synth",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 21
    assert "expected rootfs workspace /workspace/torchtitan" in proc.stdout
    assert "run_speedrun.sh --mode smoke" not in proc.stdout


def test_launch_nanogpt_2gpu_full_rootfs_rejects_forged_rootfs_on_host(
    tmp_path: Path,
):
    proc = subprocess.run(
        ["bash", str(FULL_LAUNCHER)],
        cwd=tmp_path,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_IN_ROOTFS": "1",
            "CUDA_HOME": "/opt/cuda-synth",
            "CUDA_PATH": "/opt/cuda-synth",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 21
    assert "expected rootfs workspace /workspace/torchtitan" in proc.stdout
    assert "run_speedrun.sh --mode full" not in proc.stdout


def test_launch_nanogpt_rootfs_uses_guarded_smoke_command_shape():
    text = LAUNCHER.read_text()

    assert "experiments/modded_nanogpt_b200/check_active_jobs.sh \\" in text
    assert "--active-jobs-output \"${ACTIVE_JOBS_JSON}\"" in text
    assert "experiments/modded_nanogpt_b200/run_speedrun.sh \\" in text
    assert "--mode smoke \\" in text
    assert "--lane B \\" in text
    assert "--attention-backend fa2" in text
    assert "--mlp-backend triton" in text
    assert "--launch-authorization" not in text
    assert "experiments/modded_nanogpt_b200/summarize.sh \\" in text
    assert "--results-root experiments/modded_nanogpt_b200/results" in text


def test_launch_nanogpt_2gpu_full_rootfs_uses_guarded_full_command_shape():
    text = FULL_LAUNCHER.read_text()

    assert FULL_LAUNCHER.stat().st_mode & 0o111
    assert "experiments/modded_nanogpt_b200/check_active_jobs.sh \\" in text
    assert "--active-jobs-output \"${ACTIVE_JOBS_JSON}\"" in text
    assert "experiments/modded_nanogpt_b200/run_speedrun.sh \\" in text
    assert "--mode full \\" in text
    assert "--lane B \\" in text
    assert "--verify-sha \\" in text
    assert "--launch-authorization=launch-full-b200 \\" in text
    assert "--attention-backend fa2" in text
    assert "--mlp-backend triton" in text
    assert "--skip-run" not in text
    assert "--nproc_per_node=2" not in text
    assert "full_manifest_refresh_20260816T111021Z/data_manifest.json" in text
    assert "experiments/modded_nanogpt_b200/summarize.sh \\" in text
    assert "--results-root experiments/modded_nanogpt_b200/results" in text
