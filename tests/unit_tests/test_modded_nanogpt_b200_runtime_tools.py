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
BUILD_ROOTFS = REPO_ROOT / "scripts" / "rootfs" / "build_rootfs.sh"


def _skip_unless_real_rootfs() -> None:
    if Path.cwd().resolve() != Path("/workspace/torchtitan"):
        pytest.skip("inner runtime sync behavior requires the real rootfs workspace")
    if not Path("/workspace/torchtitan/scripts/rootfs/enter_rootfs.sh").exists():
        pytest.skip("inner runtime sync behavior requires the rootfs sentinel")


def _skip_inside_real_rootfs() -> None:
    if Path.cwd().resolve() == Path("/workspace/torchtitan"):
        pytest.skip("forged host marker regression must run outside the rootfs")


def test_mise_environment_paths_are_project_owned():
    sync = (RUNTIME_DIR / "sync_tools.sh").read_text()
    assert 'MISE_DATA_DIR="${MISE_DATA_DIR:-/project/mise/data}"' in sync
    assert 'MISE_CACHE_DIR="${MISE_CACHE_DIR:-/project/mise/cache}"' in sync
    assert 'MISE_CONFIG_DIR="${MISE_CONFIG_DIR:-${RUNTIME_DIR_IN_ROOTFS}}"' in sync
    assert "$HOME/.local" not in sync
    assert "$HOME/.cache" not in sync


def test_mise_config_pins_shellcheck_only():
    config = tomllib.loads((RUNTIME_DIR / "mise.toml").read_text())

    assert config == {"tools": {"shellcheck": "0.10.0"}}


def test_rootfs_build_installs_pinned_mise_without_attempt_time_curl():
    text = BUILD_ROOTFS.read_text()
    assert "MISE_VERSION=" in text
    assert "mise-v${MISE_VERSION}-linux-x64" in text
    assert "/usr/local/bin/mise" in text
    assert '"mise_version": "${MISE_VERSION}"' in text


def test_sync_tools_reenters_rootfs_from_host(tmp_path: Path):
    fake_rootfs = tmp_path / "enter_rootfs.sh"
    fake_rootfs.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$1"\n')
    fake_rootfs.chmod(0o755)
    marker = tmp_path / "marker"

    proc = subprocess.run(
        ["bash", str(RUNTIME_DIR / "sync_tools.sh")],
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
    assert "experiments/modded_nanogpt_b200/runtime/sync_tools.sh" in marker.read_text()


def test_sync_tools_rejects_forged_rootfs_marker_on_host(tmp_path: Path):
    _skip_inside_real_rootfs()
    fake_mise = tmp_path / "mise"
    fake_mise.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'mise should not run on forged host marker\\n' >&2\n"
        "exit 99\n"
    )
    fake_mise.chmod(0o755)

    proc = subprocess.run(
        ["bash", str(RUNTIME_DIR / "sync_tools.sh")],
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
    assert "mise should not run" not in proc.stdout


def test_sync_tools_runs_mise_trust_install_and_shellcheck_smoke(tmp_path: Path):
    _skip_unless_real_rootfs()
    fake_mise = tmp_path / "mise"
    fake_mise.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "$MODDED_NANOGPT_MISE_LOG"\n'
        "if [[ \"$1\" == 'exec' ]]; then printf 'ShellCheck - shell script analysis tool\\n'; fi\n"
    )
    fake_mise.chmod(0o755)
    log = tmp_path / "mise.log"
    report = tmp_path / "tool_env_report.json"

    proc = subprocess.run(
        ["bash", str(RUNTIME_DIR / "sync_tools.sh")],
        cwd=REPO_ROOT,
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "TORCHTITAN_IN_ROOTFS": "1",
            "MODDED_NANOGPT_MISE_LOG": str(log),
            "MODDED_NANOGPT_TOOL_REPORT": str(report),
            "MISE_DATA_DIR": str(tmp_path / "mise-state" / "data"),
            "MISE_CACHE_DIR": str(tmp_path / "mise-state" / "cache"),
            "MISE_INSTALL_PATH": str(tmp_path / "mise-state" / "installs"),
            "MISE_SHIMS_DIR": str(tmp_path / "mise-state" / "shims"),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    text = log.read_text()
    assert "trust experiments/modded_nanogpt_b200/runtime/mise.toml" in text
    assert "install --yes --cd experiments/modded_nanogpt_b200/runtime" in text
    assert (
        "exec --cd experiments/modded_nanogpt_b200/runtime -- shellcheck --version"
        in text
    )
    assert report.exists()
