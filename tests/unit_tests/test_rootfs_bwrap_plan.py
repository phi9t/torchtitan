# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host-side tests for TorchTitan bwrap plan emission and verification.

These tests do not enter bwrap. They pin the pre-launch contract artifact that
`scripts/rootfs/enter_rootfs.sh` must emit before a payload can be trusted.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.rootfs.verify_runtime_env import BwrapPlanError, validate_bwrap_plan


REPO_ROOT = Path(__file__).resolve().parents[2]
ENTER_ROOTFS = REPO_ROOT / "scripts" / "rootfs" / "enter_rootfs.sh"
ROOTFS_MARKER = ".torchtitan-rootfs-owner"


CANONICAL_PATH = ":".join(
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


def _canonical_environment(host_state: Path) -> dict[str, str]:
    return {
        "PATH": CANONICAL_PATH,
        "CUDA_HOME": "/opt/cuda-synth",
        "CUDA_PATH": "/opt/cuda-synth",
        "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:/opt/cuda-synth/lib64",
        "TORCHTITAN_IN_ROOTFS": "1",
        "TORCHTITAN_ROOTFS_ENV": "modded_nanogpt_b200",
        "TORCHTITAN_ROOTFS_STORE_ID": "legacy-rootfs",
        "TORCHTITAN_ROOTFS_HOST_STATE": str(host_state),
        "TORCHTITAN_ROOTFS_PROJECT": "/workspace/torchtitan",
        "TORCHTITAN_ROOTFS_LOG_DIR": "/project/logs",
        "HOME": "/project/home",
        "XDG_CACHE_HOME": "/project/xdg-cache",
        "UV_CACHE_DIR": "/project/uv-cache",
        "PIP_CACHE_DIR": "/project/pip-cache",
        "MISE_DATA_DIR": "/project/mise/data",
        "MISE_CACHE_DIR": "/project/mise/cache",
        "MISE_CONFIG_DIR": "/workspace/torchtitan/experiments/modded_nanogpt_b200/runtime",
        "PYTHON": "/project/venvs/b200-runtime/bin/python",
        "TMPDIR": "/project/tmp",
        "TEMP": "/project/tmp",
        "TMP": "/project/tmp",
        "HF_HOME": "/workspace/torchtitan/.cache/huggingface",
        "HF_HUB_CACHE": "/workspace/torchtitan/.cache/huggingface/hub",
        "TORCH_HOME": "/workspace/torchtitan/.cache/torch",
        "MPLCONFIGDIR": "/project/xdg-cache/matplotlib",
        "NVIDIA_VISIBLE_DEVICES": "all",
    }


def _minimal_rootfs(tmp_path: Path) -> Path:
    rootfs = tmp_path / "rootfs"
    bash = rootfs / "bin" / "bash"
    bash.parent.mkdir(parents=True)
    bash.write_text("#!/usr/bin/env bash\n")
    bash.chmod(0o755)
    return rootfs


def _minimal_store(tmp_path: Path, store_id: str = "rootfs-test") -> tuple[Path, Path]:
    store = tmp_path / "store"
    rootfs = store / "content" / store_id
    bash = rootfs / "bin" / "bash"
    bash.parent.mkdir(parents=True)
    bash.write_text("#!/usr/bin/env bash\n")
    bash.chmod(0o755)
    (rootfs / ROOTFS_MARKER).write_text("torchtitan-rootfs\n")
    (rootfs / "build_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "rootfs_build_manifest",
                "store_id": store_id,
                "mutable_rootfs_allowed": False,
            }
        )
        + "\n"
    )
    (store / "selected.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "rootfs_selection",
                "store_id": store_id,
            }
        )
        + "\n"
    )
    return store, rootfs


def test_enter_rootfs_emits_plan_without_running_bwrap(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    output = tmp_path / "bwrap_plan.json"

    proc = subprocess.run(
        [
            "bash",
            str(ENTER_ROOTFS),
            "--rootfs",
            str(rootfs),
            "--",
            "python",
            "-c",
            "print('should not run')",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
            "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(output),
            "CUDA_VISIBLE_DEVICES": "0,1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    assert "should not run" not in proc.stdout
    plan = json.loads(output.read_text())
    assert plan["schema_version"] == 1
    assert plan["entrypoint"].endswith("scripts/rootfs/enter_rootfs.sh")
    assert plan["rootfs"]["path"] == str(rootfs)
    assert plan["cwd"] == "/workspace/torchtitan"
    assert plan["network_mode"] == "offline"
    assert "--share-net" not in plan["bwrap_argv"]
    assert plan["inner_argv"] == ["python", "-c", "print('should not run')"]
    assert plan["environment"]["TORCHTITAN_IN_ROOTFS"] == "1"
    assert plan["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert plan["pid_mode"] == "isolated"
    assert "--unshare-all" in plan["bwrap_argv"]
    assert "/usr/local/cuda/bin" in plan["environment"]["PATH"].split(":")
    assert any(
        mount["target"] == "/workspace/torchtitan"
        and mount["source"] == str(REPO_ROOT)
        and mount["kind"] == "bind"
        for mount in plan["mounts"]
    )

    validated = validate_bwrap_plan(plan)
    assert validated["ok"] is True
    assert validated["blockers"] == []


def test_enter_rootfs_resolves_selected_store_root(tmp_path: Path):
    store, rootfs = _minimal_store(tmp_path)
    output = tmp_path / "bwrap_plan.json"

    proc = subprocess.run(
        [
            "bash",
            str(ENTER_ROOTFS),
            "--rootfs-store",
            str(store),
            "--",
            "/bin/true",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
            "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(output),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    plan = json.loads(output.read_text())
    assert plan["rootfs"]["path"] == str(rootfs.resolve())
    assert plan["rootfs"]["store_root"] == str(store.resolve())
    assert plan["rootfs"]["store_id"] == "rootfs-test"
    assert plan["rootfs"]["legacy"] is False
    assert plan["environment"]["TORCHTITAN_ROOTFS_STORE_ID"] == "rootfs-test"
    assert validate_bwrap_plan(plan)["ok"] is True


def test_enter_rootfs_rejects_legacy_rootfs_without_manifest(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    (rootfs / ROOTFS_MARKER).write_text("torchtitan-rootfs\n")
    output = tmp_path / "bwrap_plan.json"

    proc = subprocess.run(
        [
            "bash",
            str(ENTER_ROOTFS),
            "--rootfs",
            str(rootfs),
            "--",
            "/bin/true",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
            "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(output),
            "TORCHTITAN_ROOTFS_REQUIRE_MANIFEST": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode != 0
    assert "manifest" in proc.stdout.lower()
    assert not output.exists()


def test_enter_rootfs_accepts_nonmutable_legacy_manifest(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    (rootfs / ROOTFS_MARKER).write_text("torchtitan-rootfs\n")
    (rootfs / "build_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "rootfs_build_manifest",
                "store_id": "legacy-rootfs",
                "mutable_rootfs_allowed": False,
            }
        )
        + "\n"
    )
    output = tmp_path / "bwrap_plan.json"

    proc = subprocess.run(
        [
            "bash",
            str(ENTER_ROOTFS),
            "--rootfs",
            str(rootfs),
            "--",
            "/bin/true",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
            "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(output),
            "TORCHTITAN_ROOTFS_REQUIRE_MANIFEST": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    plan = json.loads(output.read_text())
    assert plan["rootfs"]["legacy"] is True
    assert plan["rootfs"]["store_id"] == "legacy-rootfs"
    assert plan["rootfs"]["mutable_rootfs_allowed"] is False
    assert validate_bwrap_plan(plan)["ok"] is True


def test_enter_rootfs_profile_mode_shares_pid_and_records_host_rootfs(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    output = tmp_path / "bwrap_plan.json"

    proc = subprocess.run(
        [
            "bash",
            str(ENTER_ROOTFS),
            "--rootfs",
            str(rootfs),
            "--profile",
            "--",
            "/bin/true",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
            "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(output),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    plan = json.loads(output.read_text())
    assert plan["profile_mode"] is True
    assert plan["pid_mode"] == "host"
    assert plan["environment"]["TORCHTITAN_ROOTFS_PROFILE"] == "1"
    assert plan["environment"]["TORCHTITAN_ROOTFS_HOST_ROOTFS"] == str(rootfs)
    assert "--unshare-pid" not in plan["bwrap_argv"]
    assert validate_bwrap_plan(plan)["ok"] is True


def test_enter_rootfs_share_pid_keeps_host_pid_namespace(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    output = tmp_path / "bwrap_plan.json"

    proc = subprocess.run(
        [
            "bash",
            str(ENTER_ROOTFS),
            "--rootfs",
            str(rootfs),
            "--share-pid",
            "--",
            "/bin/true",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
            "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(output),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    plan = json.loads(output.read_text())
    assert plan["pid_mode"] == "host"
    assert "--unshare-all" not in plan["bwrap_argv"]
    assert "--unshare-pid" not in plan["bwrap_argv"]
    assert plan["environment"]["TORCHTITAN_ROOTFS_SHARE_PID"] == "1"
    assert validate_bwrap_plan(plan)["ok"] is True


def test_enter_rootfs_binds_host_sysfs_when_present(tmp_path: Path):
    if not Path("/sys").is_dir():
        pytest.skip("host has no /sys")
    rootfs = _minimal_rootfs(tmp_path)
    output = tmp_path / "bwrap_plan.json"

    proc = subprocess.run(
        [
            "bash",
            str(ENTER_ROOTFS),
            "--rootfs",
            str(rootfs),
            "--",
            "/bin/true",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
            "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(output),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    mounts = {
        mount["target"]: mount for mount in json.loads(output.read_text())["mounts"]
    }
    assert mounts["/sys"]["kind"] == "ro-bind"
    assert mounts["/sys"]["source"] == "/sys"


def _emit_plan(tmp_path: Path, rootfs: Path, *mode_args: str) -> dict:
    output = tmp_path / f"bwrap_plan{'_'.join(mode_args).replace('-', '')}.json"
    proc = subprocess.run(
        [
            "bash",
            str(ENTER_ROOTFS),
            "--rootfs",
            str(rootfs),
            *mode_args,
            "--",
            "/bin/true",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
            "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(output),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.returncode == 0, proc.stdout
    return json.loads(output.read_text())


def test_enter_rootfs_privileged_mode_keeps_initial_user_namespace(tmp_path: Path):
    """bpf() checks capabilities against the initial user namespace.

    A capability minted inside a nested user namespace can never satisfy it, so
    privileged mode must not unshare the user namespace and must ask bwrap to
    retain the tracing capabilities instead.
    """
    plan = _emit_plan(tmp_path, _minimal_rootfs(tmp_path), "--privileged")

    assert plan["privileged_mode"] is True
    assert plan["pid_mode"] == "host"
    assert plan["environment"]["TORCHTITAN_ROOTFS_PRIVILEGED"] == "1"

    argv = plan["bwrap_argv"]
    assert "--unshare-user" not in argv
    assert "--unshare-user-try" not in argv
    assert "--unshare-all" not in argv
    assert "--unshare-pid" not in argv

    assert plan["capabilities"] == [
        "CAP_BPF",
        "CAP_PERFMON",
        "CAP_SYS_ADMIN",
        "CAP_SYS_PTRACE",
        "CAP_SYSLOG",
    ]
    added = {argv[i + 1] for i, arg in enumerate(argv) if arg == "--cap-add"}
    assert added == set(plan["capabilities"])


def test_enter_rootfs_privileged_mode_exposes_kernel_headers_and_sbin(tmp_path: Path):
    """BCC compiles each program against the running kernel at runtime."""
    plan = _emit_plan(tmp_path, _minimal_rootfs(tmp_path), "--privileged")

    mounts = {mount["target"]: mount for mount in plan["mounts"]}
    for target in ("/lib/modules", "/usr/src"):
        assert mounts[target]["kind"] == "ro-bind", f"{target} not ro-bound"

    # BCC's tools and bpftool install into sbin.
    path_entries = plan["environment"]["PATH"].split(":")
    assert "/usr/sbin" in path_entries
    assert "/sbin" in path_entries


def test_enter_rootfs_default_mode_grants_no_capabilities(tmp_path: Path):
    """Training must keep the unprivileged sandbox it has today."""
    plan = _emit_plan(tmp_path, _minimal_rootfs(tmp_path))

    assert plan["privileged_mode"] is False
    assert plan["capabilities"] == []
    assert "--cap-add" not in plan["bwrap_argv"]
    assert "--unshare-all" in plan["bwrap_argv"]
    assert "TORCHTITAN_ROOTFS_PRIVILEGED" not in plan["environment"]

    path_entries = plan["environment"]["PATH"].split(":")
    assert "/usr/sbin" not in path_entries
    assert "/sbin" not in path_entries


def test_enter_rootfs_fills_only_empty_rootfs_libraries(tmp_path: Path):
    """Host perf/bpftool need host libs, but must not shadow rootfs ones.

    The image ships zero-byte placeholders where host-only libraries such as
    libunwind belong, which is why the bound host perf reports "file too
    short" without this. A rootfs library that is actually populated has to be
    left alone.
    """
    if not Path("/usr/bin/perf").exists():
        pytest.skip("host has no perf")

    rootfs = _minimal_rootfs(tmp_path)
    # Mirror the merged-usr layout of the real image so that the /lib and
    # /usr/lib spellings of a library refer to the same file, as they do on
    # the host. ldd prints either one depending on LD_LIBRARY_PATH.
    if not (rootfs / "lib").exists():
        (rootfs / "usr" / "lib").mkdir(parents=True, exist_ok=True)
        (rootfs / "lib").symlink_to("usr/lib")

    # ldd prints the /lib or the /usr/lib spelling depending on
    # LD_LIBRARY_PATH, so query it under the same environment the plan run
    # uses. The script binds exactly the path ldd reports.
    host_libs = {
        line.split()[2]
        for line in subprocess.run(
            ["ldd", "/usr/bin/perf"],
            env={"PATH": "/usr/bin:/bin"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.splitlines()
        if "=> /" in line
    }
    if not host_libs:
        pytest.skip("host perf is not dynamically linked")

    populated = sorted(host_libs)[0]
    in_rootfs = rootfs / Path(populated).relative_to("/")
    in_rootfs.parent.mkdir(parents=True, exist_ok=True)
    in_rootfs.write_bytes(b"populated")

    plan = _emit_plan(tmp_path, rootfs, "--privileged")
    ro_bound = {
        mount["target"] for mount in plan["mounts"] if mount["kind"] == "ro-bind"
    }

    assert populated not in ro_bound, f"shadowed populated rootfs library {populated}"
    for lib in host_libs - {populated}:
        assert lib in ro_bound, f"{lib} missing from the plan"


def test_enter_rootfs_binds_host_cuda_toolkit_when_present(tmp_path: Path):
    if not Path("/usr/local/cuda/bin/nsys").is_file():
        pytest.skip("host has no /usr/local/cuda/bin/nsys")
    rootfs = _minimal_rootfs(tmp_path)
    output = tmp_path / "bwrap_plan.json"

    proc = subprocess.run(
        [
            "bash",
            str(ENTER_ROOTFS),
            "--rootfs",
            str(rootfs),
            "--",
            "/bin/true",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
            "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(output),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    mounts = {
        mount["target"]: mount for mount in json.loads(output.read_text())["mounts"]
    }
    assert mounts["/usr/local/cuda"]["kind"] == "ro-bind"
    assert Path(mounts["/usr/local/cuda"]["source"]).exists()


def test_enter_rootfs_can_emit_networked_plan(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    output = tmp_path / "bwrap_plan.json"

    proc = subprocess.run(
        [
            "bash",
            str(ENTER_ROOTFS),
            "--rootfs",
            str(rootfs),
            "--",
            "/bin/true",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
            "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(output),
            "TORCHTITAN_ROOTFS_NETWORK": "networked",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    plan = json.loads(output.read_text())
    assert plan["network_mode"] == "networked"
    assert "--share-net" in plan["bwrap_argv"]
    assert {"/etc/resolv.conf", "/etc/hosts"}.intersection(
        {mount["target"] for mount in plan["mounts"]}
    )
    assert validate_bwrap_plan(plan)["ok"] is True


def test_enter_rootfs_rejects_invalid_network_mode(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    output = tmp_path / "bwrap_plan.json"

    proc = subprocess.run(
        [
            "bash",
            str(ENTER_ROOTFS),
            "--rootfs",
            str(rootfs),
            "--",
            "/bin/true",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
            "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(output),
            "TORCHTITAN_ROOTFS_NETWORK": "internet",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode != 0
    assert "TORCHTITAN_ROOTFS_NETWORK" in proc.stdout
    assert not output.exists()


def test_enter_rootfs_plan_exports_canonical_runtime_environment(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    output = tmp_path / "bwrap_plan.json"
    host_state = REPO_ROOT / ".cache" / "pytest-rootfs-state"

    proc = subprocess.run(
        [
            "bash",
            str(ENTER_ROOTFS),
            "--rootfs",
            str(rootfs),
            "--",
            "/bin/true",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
            "TORCHTITAN_ROOTFS_PLAN_OUTPUT": str(output),
            "TORCHTITAN_ROOTFS_HOST_STATE": str(host_state),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    environment = json.loads(output.read_text())["environment"]
    expected = _canonical_environment(host_state)
    for key, value in expected.items():
        assert environment.get(key) == value
    assert "/project/venvs/b200-runtime/bin" in environment["PATH"].split(":")


def test_enter_rootfs_emit_plan_stdout_when_no_output_path(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    proc = subprocess.run(
        [
            "bash",
            str(ENTER_ROOTFS),
            "--rootfs",
            str(rootfs),
            "--",
            "/bin/true",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    plan = json.loads(proc.stdout)
    assert plan["inner_argv"] == ["/bin/true"]


def test_enter_rootfs_emit_plan_does_not_require_bwrap_on_path(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    python_link = fake_bin / "python3"
    python_link.symlink_to("/usr/bin/python3")
    dirname_link = fake_bin / "dirname"
    dirname_link.symlink_to("/usr/bin/dirname")
    pwd_link = fake_bin / "pwd"
    pwd_link.symlink_to("/usr/bin/pwd")
    bash_link = fake_bin / "bash"
    bash_link.symlink_to("/usr/bin/bash")
    cat_link = fake_bin / "cat"
    cat_link.symlink_to("/usr/bin/cat")
    mkdir_link = fake_bin / "mkdir"
    mkdir_link.symlink_to("/usr/bin/mkdir")

    proc = subprocess.run(
        [
            str(bash_link),
            str(ENTER_ROOTFS),
            "--rootfs",
            str(rootfs),
            "--",
            "/bin/true",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": str(fake_bin),
            "TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0, proc.stdout
    assert "bwrap not found" not in proc.stdout
    assert json.loads(proc.stdout)["inner_argv"] == ["/bin/true"]


def test_bwrap_plan_verifier_rejects_missing_workspace_mount(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    plan = {
        "schema_version": 1,
        "entrypoint": str(ENTER_ROOTFS),
        "rootfs": {"path": str(rootfs), "explicit": True},
        "cwd": "/workspace/torchtitan",
        "network_mode": "offline",
        "mounts": [
            {"kind": "bind", "source": str(rootfs), "target": "/", "writable": True}
        ],
        "devices": [],
        "driver_libraries": [],
        "environment": {
            "PATH": "/opt/cuda-synth/bin:/usr/local/bin:/usr/bin:/bin",
            "CUDA_HOME": "/opt/cuda-synth",
            "CUDA_PATH": "/opt/cuda-synth",
            "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:/opt/cuda-synth/lib64",
            "TORCHTITAN_IN_ROOTFS": "1",
            "HOME": "/root",
            "NVIDIA_VISIBLE_DEVICES": "all",
        },
        "inner_argv": ["/bin/true"],
        "bwrap_argv": [],
    }

    with pytest.raises(BwrapPlanError, match="/workspace/torchtitan"):
        validate_bwrap_plan(plan)


def test_bwrap_plan_verifier_rejects_rootfs_marker_drift(tmp_path: Path):
    rootfs = _minimal_rootfs(tmp_path)
    plan = {
        "schema_version": 1,
        "entrypoint": str(ENTER_ROOTFS),
        "rootfs": {"path": str(rootfs), "explicit": True},
        "cwd": "/workspace/torchtitan",
        "network_mode": "offline",
        "mounts": [
            {"kind": "bind", "source": str(rootfs), "target": "/", "writable": True},
            {"kind": "proc", "source": "procfs", "target": "/proc", "writable": False},
            {"kind": "tmpfs", "source": "tmpfs", "target": "/tmp", "writable": True},
            {"kind": "dev", "source": "devfs", "target": "/dev", "writable": True},
            {
                "kind": "bind",
                "source": str(REPO_ROOT),
                "target": "/workspace/torchtitan",
                "writable": True,
            },
        ],
        "devices": [],
        "driver_libraries": [],
        "environment": {
            **_canonical_environment(tmp_path / "state"),
            "TORCHTITAN_IN_ROOTFS": "0",
        },
        "inner_argv": ["/bin/true"],
        "bwrap_argv": [],
    }

    with pytest.raises(BwrapPlanError, match="TORCHTITAN_IN_ROOTFS"):
        validate_bwrap_plan(plan)
