# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "experiments" / "modded_nanogpt_b200" / "runtime"


def _skip_unless_real_rootfs() -> None:
    if Path.cwd().resolve() != Path("/workspace/torchtitan"):
        pytest.skip("inner runtime sync behavior requires the real rootfs workspace")
    if not Path("/workspace/torchtitan/scripts/rootfs/enter_rootfs.sh").exists():
        pytest.skip("inner runtime sync behavior requires the rootfs sentinel")


def _skip_inside_real_rootfs() -> None:
    if Path.cwd().resolve() == Path("/workspace/torchtitan"):
        pytest.skip("forged host marker regression must run outside the rootfs")


def test_runtime_dependency_inputs_exclude_torch_stack_replacements():
    direct = [
        line.strip()
        for line in (RUNTIME_DIR / "requirements.direct.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    lock = [
        line.strip()
        for line in (RUNTIME_DIR / "requirements.lock").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    forbidden = (
        "torch",
        "triton",
        "nvidia-cuda-runtime",
        "nvidia-cublas",
        "nvidia-cudnn",
    )
    for package in forbidden:
        assert not any(
            line == package
            or line.startswith(f"{package}==")
            or line.startswith(f"{package}<")
            or line.startswith(f"{package}>")
            for line in direct + lock
        )

    expected_direct = (
        "numpy",
        "tqdm",
        "huggingface-hub",
        "datasets",
        "tiktoken",
        "typing-extensions",
        "setuptools",
        "kernels",
        "flash-attn==2.8.3.post1",
    )
    direct_text = "\n".join(direct)
    for package in expected_direct:
        assert package in direct_text


def test_direct_requirements_match_pyproject_dependencies():
    pyproject = tomllib.loads((RUNTIME_DIR / "pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]
    direct = [
        line.strip()
        for line in (RUNTIME_DIR / "requirements.direct.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert direct == dependencies


def test_sync_python_env_reenters_rootfs_from_host(tmp_path: Path):
    fake_rootfs = tmp_path / "enter_rootfs.sh"
    fake_rootfs.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$1"\n')
    fake_rootfs.chmod(0o755)
    marker = tmp_path / "marker"

    proc = subprocess.run(
        [
            "bash",
            str(RUNTIME_DIR / "sync_python_env.sh"),
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_ENTRYPOINT": str(fake_rootfs),
            "MODDED_NANOGPT_SYNC_MARKER": str(marker),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    assert (
        "experiments/modded_nanogpt_b200/runtime/sync_python_env.sh"
        in marker.read_text()
    )


def test_sync_python_env_rejects_forged_rootfs_marker_on_host(tmp_path: Path):
    _skip_inside_real_rootfs()
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'uv should not run on forged host marker\\n' >&2\n"
        "exit 99\n"
    )
    fake_uv.chmod(0o755)

    proc = subprocess.run(
        [
            "bash",
            str(RUNTIME_DIR / "sync_python_env.sh"),
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "TORCHTITAN_IN_ROOTFS": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 21
    assert "expected rootfs workspace /workspace/torchtitan" in proc.stdout
    assert "uv should not run" not in proc.stdout


def test_sync_python_env_requires_networked_fallback_for_direct_requirements(
    tmp_path: Path,
):
    _skip_unless_real_rootfs()
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "$MODDED_NANOGPT_UV_LOG"\n'
        "if [[ \"$1\" == 'venv' ]]; then "
        'mkdir -p "$5/bin"; '
        "printf '#!/usr/bin/env bash\\n' > \"$5/bin/python\"; "
        'chmod +x "$5/bin/python"; '
        "fi\n"
    )
    fake_uv.chmod(0o755)
    log = tmp_path / "uv.log"

    proc = subprocess.run(
        [
            "bash",
            str(RUNTIME_DIR / "sync_python_env.sh"),
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "TORCHTITAN_IN_ROOTFS": "1",
            "TORCHTITAN_ROOTFS_NETWORK": "offline",
            "MODDED_NANOGPT_UV_LOG": str(log),
            "MODDED_NANOGPT_RUNTIME_VENV": str(tmp_path / "venv"),
            "MODDED_NANOGPT_REQUIREMENTS_LOCK": str(tmp_path / "missing.lock"),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode != 0
    assert "networked" in proc.stdout.lower()
    assert not log.exists()


def test_sync_python_env_rejects_placeholder_lock_in_offline_mode(tmp_path: Path):
    _skip_unless_real_rootfs()
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'uv should not run for placeholder lock\\n' >&2\n"
        "exit 99\n"
    )
    fake_uv.chmod(0o755)
    placeholder_lock = tmp_path / "requirements.lock"
    placeholder_lock.write_text(
        "# Placeholder lock without hash-locked package entries.\n"
    )

    proc = subprocess.run(
        [
            "bash",
            str(RUNTIME_DIR / "sync_python_env.sh"),
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "TORCHTITAN_IN_ROOTFS": "1",
            "TORCHTITAN_ROOTFS_NETWORK": "offline",
            "MODDED_NANOGPT_RUNTIME_VENV": str(tmp_path / "venv"),
            "MODDED_NANOGPT_REQUIREMENTS_LOCK": str(placeholder_lock),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 21
    assert "hash-locked package entries" in proc.stdout
    assert "uv should not run" not in proc.stdout


def test_sync_python_env_uses_hash_locked_requirements_when_available(tmp_path: Path):
    _skip_unless_real_rootfs()
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "$MODDED_NANOGPT_UV_LOG"\n'
        "if [[ \"$1\" == 'venv' ]]; then "
        'mkdir -p "$5/bin"; '
        "printf '#!/usr/bin/env bash\\n' > \"$5/bin/python\"; "
        'chmod +x "$5/bin/python"; '
        "fi\n"
    )
    fake_uv.chmod(0o755)
    lock = tmp_path / "requirements.lock"
    lock.write_text("numpy==1.0 --hash=sha256:" + "0" * 64 + "\n")
    log = tmp_path / "uv.log"

    proc = subprocess.run(
        [
            "bash",
            str(RUNTIME_DIR / "sync_python_env.sh"),
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "TORCHTITAN_IN_ROOTFS": "1",
            "TORCHTITAN_ROOTFS_NETWORK": "offline",
            "MODDED_NANOGPT_UV_LOG": str(log),
            "MODDED_NANOGPT_RUNTIME_VENV": str(tmp_path / "venv"),
            "MODDED_NANOGPT_REQUIREMENTS_LOCK": str(lock),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    text = log.read_text()
    assert "--require-hashes" in text
    assert "--no-deps" in text
    assert str(lock) in text


def test_rootfs_guard_selects_managed_python_when_present(tmp_path: Path):
    venv = tmp_path / "venv"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/usr/bin/env bash\n")
    python.chmod(0o755)
    script = f"""
set -euo pipefail
source "{REPO_ROOT / "experiments/modded_nanogpt_b200/rootfs_guard.sh"}"
select_modded_nanogpt_python
"""
    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "MODDED_NANOGPT_RUNTIME_VENV": str(venv),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    assert proc.stdout.strip() == str(python)


def test_rootfs_wrappers_use_selected_python_for_first_party_entrypoints():
    exec_wrappers = (
        "experiments/modded_nanogpt_b200/certify_optimized_kernels.sh",
        "experiments/modded_nanogpt_b200/check_active_jobs.sh",
        "experiments/modded_nanogpt_b200/diagnose_mlp_backend.sh",
        "experiments/modded_nanogpt_b200/fetch_upstream.sh",
        "experiments/modded_nanogpt_b200/parse_log.sh",
        "experiments/modded_nanogpt_b200/prepare_data.sh",
        "experiments/modded_nanogpt_b200/run_cpu_smoke.sh",
        "experiments/modded_nanogpt_b200/run_cpu_stability.sh",
        "experiments/modded_nanogpt_b200/run_experiment_matrix.sh",
        "experiments/modded_nanogpt_b200/run_performance_probe.sh",
        "experiments/modded_nanogpt_b200/run_preflight.sh",
        "experiments/modded_nanogpt_b200/run_speedrun.sh",
        "experiments/modded_nanogpt_b200/summarize.sh",
    )
    for rel in exec_wrappers:
        text = (REPO_ROOT / rel).read_text()
        assert 'PYTHON_BIN="$(select_modded_nanogpt_python)"' in text
        assert 'exec "${PYTHON_BIN}"' in text

    setup_text = (
        REPO_ROOT / "experiments/modded_nanogpt_b200/setup_flash_attention.sh"
    ).read_text()
    assert 'PYTHON_BIN="$(select_modded_nanogpt_python)"' in setup_text
    assert '"${PYTHON_BIN}" - <<' in setup_text
    assert '"${PYTHON_BIN}" -m pip install' in setup_text


def test_rootfs_wrappers_do_not_use_bare_python_for_first_party_entrypoints():
    allowed_system_python = {
        # These bootstrap or verify the runtime/tool environment itself before
        # the managed runtime venv is guaranteed to exist.
        "experiments/modded_nanogpt_b200/runtime/sync_python_env.sh",
        "experiments/modded_nanogpt_b200/runtime/sync_tools.sh",
    }
    for path in (REPO_ROOT / "experiments/modded_nanogpt_b200").glob("*.sh"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in allowed_system_python:
            continue
        text = path.read_text()
        assert "exec python " not in text
        assert "exec python3 " not in text
        assert "python -m pip" not in text
        assert "python3 -m pip" not in text
        assert "python - <<" not in text
        assert "python3 - <<" not in text
