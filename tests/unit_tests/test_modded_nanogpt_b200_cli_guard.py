# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("script", "wrapper", "args"),
    [
        (
            "experiments/modded_nanogpt_b200/parse_log.py",
            "experiments/modded_nanogpt_b200/parse_log.sh",
            [
                "--log",
                "missing.log",
                "--result-dir",
                "experiments/modded_nanogpt_b200/results/direct_parse_fixture",
                "--source",
                "experiments/modded_nanogpt_b200/sources/modded-nanogpt",
                "--data-manifest",
                "missing_manifest.json",
                "--preflight-report",
                "missing_preflight.json",
            ],
        ),
        (
            "experiments/modded_nanogpt_b200/summarize.py",
            "experiments/modded_nanogpt_b200/summarize.sh",
            [
                "--results-root",
                "experiments/modded_nanogpt_b200/results",
                "--output",
                "experiments/modded_nanogpt_b200/results/direct_summary_fixture/run_index.json",
            ],
        ),
        (
            "experiments/modded_nanogpt_b200/prepare_data.py",
            "experiments/modded_nanogpt_b200/prepare_data.sh",
            [
                "--source",
                "experiments/modded_nanogpt_b200/sources/modded-nanogpt",
                "--data-dir",
                "experiments/modded_nanogpt_b200/data/fineweb10B",
                "--output",
                "experiments/modded_nanogpt_b200/results/direct_data_fixture/data_manifest.json",
                "--skip-upstream-command",
            ],
        ),
        (
            "experiments/modded_nanogpt_b200/fetch_upstream.py",
            "experiments/modded_nanogpt_b200/fetch_upstream.sh",
            [
                "--source",
                "experiments/modded_nanogpt_b200/sources/direct_source_fixture",
                "--result-dir",
                "experiments/modded_nanogpt_b200/results/direct_source_fixture",
            ],
        ),
        (
            "experiments/modded_nanogpt_b200/preflight.py",
            "experiments/modded_nanogpt_b200/run_preflight.sh",
            [
                "--lane",
                "B",
                "--mode",
                "full",
                "--source",
                "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa",
                "--data-manifest",
                "experiments/modded_nanogpt_b200/results/full_manifest_refresh_20260816T000342Z/data_manifest.json",
                "--report",
                "experiments/modded_nanogpt_b200/results/direct_preflight_fixture/preflight_report.json",
                "--attention-backend",
                "fa2",
                "--mlp-backend",
                "triton",
                "--verify-sha",
            ],
        ),
        (
            "experiments/modded_nanogpt_b200/diagnose_mlp_backend.py",
            "experiments/modded_nanogpt_b200/diagnose_mlp_backend.sh",
            [
                "--source",
                "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa",
                "--backend",
                "triton",
                "--output",
                "experiments/modded_nanogpt_b200/results/direct_mlp_fixture/mlp_diagnostic.json",
            ],
        ),
        (
            "experiments/modded_nanogpt_b200/performance_probe.py",
            "experiments/modded_nanogpt_b200/run_performance_probe.sh",
            [
                "--probe-name",
                "direct_probe_fixture",
                "--probe-kind",
                "static_wrapper_preflight",
                "--result-dir",
                "experiments/modded_nanogpt_b200/results/direct_probe_fixture",
                "--source",
                "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa",
                "--data-manifest",
                "experiments/modded_nanogpt_b200/results/full_manifest_refresh_20260816T000342Z/data_manifest.json",
                "--attention-backend",
                "fa2",
                "--mlp-backend",
                "triton",
                "--run-id",
                "direct_probe_fixture",
                "--attempt-id",
                "direct_probe_fixture_attempt_001",
                "--gpu-ids",
                "0,1",
                "--world-size",
                "2",
            ],
        ),
        (
            "experiments/modded_nanogpt_b200/optimized_kernel_certifier.py",
            "experiments/modded_nanogpt_b200/certify_optimized_kernels.sh",
            [
                "--source",
                "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa",
                "--attention-backend",
                "fa2",
                "--mlp-backend",
                "triton",
                "--output",
                "experiments/modded_nanogpt_b200/results/direct_kernel_cert_fixture/runtime/optimized_kernel_report.json",
            ],
        ),
        (
            "experiments/modded_nanogpt_b200/run_experiment_matrix.py",
            "experiments/modded_nanogpt_b200/run_experiment_matrix.sh",
            [
                "--config",
                "experiments/modded_nanogpt_b200/configs/gpu_ladder_prerequisite.json",
                "--dry-run",
            ],
        ),
        (
            "experiments/modded_nanogpt_b200/cpu_smoke.py",
            "experiments/modded_nanogpt_b200/run_cpu_smoke.sh",
            [
                "--result-dir",
                "experiments/modded_nanogpt_b200/results/direct_cpu_smoke_fixture",
            ],
        ),
        (
            "experiments/modded_nanogpt_b200/cpu_stability.py",
            "experiments/modded_nanogpt_b200/run_cpu_stability.sh",
            [
                "--result-dir",
                "experiments/modded_nanogpt_b200/results/direct_cpu_stability_fixture",
            ],
        ),
    ],
)
def test_direct_python_cli_fails_closed_outside_rootfs(
    script: str,
    wrapper: str,
    args: list[str],
):
    env = dict(os.environ)
    env.pop("TORCHTITAN_IN_ROOTFS", None)
    proc = subprocess.run(
        [sys.executable, script, *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )

    assert proc.returncode == 21
    assert f"must run through {wrapper}" in proc.stdout
    assert "TORCHTITAN_IN_ROOTFS=1 is missing" in proc.stdout
    assert "Traceback" not in proc.stdout


def test_direct_python_cli_help_stays_available():
    proc = subprocess.run(
        [
            sys.executable,
            "experiments/modded_nanogpt_b200/diagnose_mlp_backend.py",
            "--help",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0
    assert "Run a bounded MLP backend diagnostic" in proc.stdout


def test_shell_wrapper_rejects_forged_rootfs_marker_outside_workspace(tmp_path: Path):
    fake_cuda = tmp_path / "cuda"
    fake_cuda_bin = fake_cuda / "bin"
    fake_cuda_lib = fake_cuda / "lib"
    fake_cuda_bin.mkdir(parents=True)
    fake_cuda_lib.mkdir()
    fake_nvcc = fake_cuda_bin / "nvcc"
    fake_nvcc.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_nvcc.chmod(0o755)
    (fake_cuda_lib / "libcudart.so.13").write_text("")

    env = {
        **os.environ,
        "TORCHTITAN_IN_ROOTFS": "1",
        "CUDA_HOME": str(fake_cuda),
        "CUDA_PATH": str(fake_cuda),
        "CC": "/bin/true",
        "CXX": "/bin/true",
    }
    proc = subprocess.run(
        [
            "bash",
            "/workspace/torchtitan/experiments/modded_nanogpt_b200/setup_flash_attention.sh",
            "--help",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=tmp_path,
    )

    assert proc.returncode == 21
    assert "expected rootfs workspace /workspace/torchtitan" in proc.stdout
    assert "pip install" not in proc.stdout


def test_flash_attention_setup_probes_health_before_install_commands():
    script = Path("experiments/modded_nanogpt_b200/setup_flash_attention.sh")
    text = script.read_text()

    probe_def = text.index("flash_attn_health_probe()")
    fast_path = text.index("flash_attn_health_probe; then")
    cuda_install = text.index('"${PYTHON_BIN}" -m pip install')
    flash_install = text.index("flash_attn_install=(")

    assert probe_def < fast_path < cuda_install < flash_install
    assert 'FLASH_ATTN_FORCE_REBUILD="${FLASH_ATTN_FORCE_REBUILD:-0}"' in text
    assert '"${FLASH_ATTN_FORCE_REBUILD}" != "1"' in text
    assert "flash-attn already healthy; skipping rebuild" in text
    assert "--force-reinstall" in text[fast_path:]
    assert "flash_attn_health_probe\n" in text[flash_install:]
