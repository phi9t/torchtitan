# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Behavior tests for launcher-owned run evidence identity."""

import os
import subprocess
import textwrap
import uuid
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[2]
RUN_TRAIN = REPO_ROOT / "run_train.sh"
MULTINODE_TRAINER = REPO_ROOT / "multinode_trainer.slurm"
GENERATED_RUN_ID = "11111111-1111-4111-8111-111111111111"
GENERATED_ATTEMPT_ID = "22222222-2222-4222-8222-222222222222"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(textwrap.dedent(contents))
    path.chmod(0o755)


@pytest.fixture
def launcher_stubs(tmp_path: Path) -> tuple[Path, Path, Path]:
    stub_bin = tmp_path / "bin"
    capture_dir = tmp_path / "captures"
    state_dir = tmp_path / "state"
    stub_bin.mkdir()
    capture_dir.mkdir()
    state_dir.mkdir()

    _write_executable(
        stub_bin / "python3",
        f"""\
        #!/usr/bin/bash
        set -eu

        if [ "${{1:-}}" = "-c" ]; then
            counter_file="${{STUB_STATE_DIR:?}}/python3-generation-count"
            count=0
            if [ -f "$counter_file" ]; then
                read -r count < "$counter_file"
            fi
            count=$((count + 1))
            printf '%s\n' "$count" > "$counter_file"
            case "$count" in
                1) printf '%s\n' "{GENERATED_RUN_ID}" ;;
                2) printf '%s\n' "{GENERATED_ATTEMPT_ID}" ;;
                *) exit 91 ;;
            esac
            exit 0
        fi

        if [ "${{1:-}}" = "-m" ] && [ "${{2:-}}" = "torchtitan.train" ]; then
            printf '%s\\0' \
                "${{TORCHTITAN_RUN_ID-}}" \
                "${{TORCHTITAN_ATTEMPT_ID-}}" \
                "${{NGPU-}}" \
                "${{LOCAL_RANK-}}" \
                "${{TORCHELASTIC_RESTART_COUNT-}}" \
                > "${{STUB_CAPTURE_DIR:?}}/python-env"
            printf '%s\\0' "$@" > "${{STUB_CAPTURE_DIR:?}}/python-argv"
            exit 0
        fi

        exit 92
        """,
    )
    _write_executable(
        stub_bin / "torchrun",
        """\
        #!/usr/bin/bash
        set -eu

        printf '%s\\0' \
            "${TORCHTITAN_RUN_ID-}" \
            "${TORCHTITAN_ATTEMPT_ID-}" \
            > "${STUB_CAPTURE_DIR:?}/torchrun-env"
        printf '%s\\0' "$@" > "${STUB_CAPTURE_DIR:?}/torchrun-argv"
        """,
    )
    _write_executable(
        stub_bin / "scontrol",
        """\
        #!/usr/bin/bash
        set -eu

        [ "$#" -eq 3 ]
        [ "$1" = "show" ]
        [ "$2" = "hostnames" ]
        printf '%s\n' node-a node-b node-c node-d
        """,
    )
    _write_executable(
        stub_bin / "srun",
        """\
        #!/usr/bin/bash
        set -eu

        if [ "${1:-}" = "--nodes=1" ]; then
            printf '%s\n' "10.20.30.40"
            exit 0
        fi

        printf '%s\\0' \
            "${TORCHTITAN_RUN_ID-}" \
            "${TORCHTITAN_ATTEMPT_ID-}" \
            > "${STUB_CAPTURE_DIR:?}/srun-env"
        printf '%s\\0' "$@" > "${STUB_CAPTURE_DIR:?}/srun-argv"
        "$@"
        """,
    )
    _write_executable(
        stub_bin / "dcgmi",
        """\
        #!/usr/bin/bash
        set -eu

        printf '%s\\0' "$@" >> "${STUB_CAPTURE_DIR:?}/dcgmi-argv"
        """,
    )
    return stub_bin, capture_dir, state_dir


def _launcher_env(
    launcher_stubs: tuple[Path, Path, Path], **overrides: str
) -> dict[str, str]:
    stub_bin, capture_dir, state_dir = launcher_stubs
    env = os.environ.copy()
    for name in (
        "COMM_MODE",
        "CONFIG",
        "LOCAL_RANK",
        "LOG_RANK",
        "MODULE",
        "NGPU",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHTITAN_ATTEMPT_ID",
        "TORCHTITAN_RUN_ID",
    ):
        env.pop(name, None)
    env.update(
        {
            "PATH": f"{stub_bin}:/usr/bin:/bin",
            "SLURM_JOB_NODELIST": "node-[a-d]",
            "STUB_CAPTURE_DIR": str(capture_dir),
            "STUB_STATE_DIR": str(state_dir),
        }
    )
    env.update(overrides)
    return env


def _run_launcher(path: Path, env: dict[str, str], *args: str) -> None:
    subprocess.run(
        ["/usr/bin/bash", str(path), *args],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _nul_record(path: Path) -> list[str]:
    contents = path.read_bytes()
    assert contents.endswith(b"\0")
    return contents[:-1].decode().split("\0")


def _generation_count(state_dir: Path) -> int:
    count_file = state_dir / "python3-generation-count"
    return int(count_file.read_text()) if count_file.exists() else 0


def _local_torchrun_argv(ngpu: int, run_id: str) -> list[str]:
    return [
        f"--nproc_per_node={ngpu}",
        "--rdzv_id",
        run_id,
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint=localhost:0",
        "--local-ranks-filter",
        "0",
        "--role",
        "rank",
        "--tee",
        "3",
        "-m",
        "torchtitan.train",
        "--module",
        "llama3",
        "--config",
        "llama3_debugmodel",
        "--training.seq_len=128",
    ]


@pytest.mark.parametrize("ngpu", (2, 4, 8))
def test_local_launcher_shares_generated_identity_with_each_gpu_topology(
    launcher_stubs: tuple[Path, Path, Path], ngpu: int
) -> None:
    _, capture_dir, state_dir = launcher_stubs
    env = _launcher_env(launcher_stubs, NGPU=str(ngpu))

    _run_launcher(RUN_TRAIN, env, "--training.seq_len=128")

    assert _nul_record(capture_dir / "torchrun-env") == [
        GENERATED_RUN_ID,
        GENERATED_ATTEMPT_ID,
    ]
    argv = _nul_record(capture_dir / "torchrun-argv")
    assert argv == _local_torchrun_argv(ngpu, GENERATED_RUN_ID)
    assert argv.count("--rdzv_id") == 1
    assert uuid.UUID(GENERATED_RUN_ID).version == 4
    assert uuid.UUID(GENERATED_ATTEMPT_ID).version == 4
    assert GENERATED_RUN_ID != GENERATED_ATTEMPT_ID
    assert _generation_count(state_dir) == 2


@pytest.mark.parametrize(
    (
        "preset_run_id",
        "preset_attempt_id",
        "expected_run_id",
        "expected_attempt_id",
        "expected_generation_count",
    ),
    (
        ("platform-run", "platform-attempt", "platform-run", "platform-attempt", 0),
        (
            "platform-run",
            None,
            "platform-run",
            GENERATED_RUN_ID,
            1,
        ),
        (
            None,
            "platform-attempt",
            GENERATED_RUN_ID,
            "platform-attempt",
            1,
        ),
        ("", "", GENERATED_RUN_ID, GENERATED_ATTEMPT_ID, 2),
        ("", "platform-attempt", GENERATED_RUN_ID, "platform-attempt", 1),
        ("platform-run", "", "platform-run", GENERATED_RUN_ID, 1),
    ),
)
def test_local_launcher_preserves_overrides_and_generates_only_missing_ids(
    launcher_stubs: tuple[Path, Path, Path],
    preset_run_id: str | None,
    preset_attempt_id: str | None,
    expected_run_id: str,
    expected_attempt_id: str,
    expected_generation_count: int,
) -> None:
    _, capture_dir, state_dir = launcher_stubs
    overrides = {"NGPU": "2"}
    if preset_run_id is not None:
        overrides["TORCHTITAN_RUN_ID"] = preset_run_id
    if preset_attempt_id is not None:
        overrides["TORCHTITAN_ATTEMPT_ID"] = preset_attempt_id
    env = _launcher_env(launcher_stubs, **overrides)

    _run_launcher(RUN_TRAIN, env)

    assert _nul_record(capture_dir / "torchrun-env") == [
        expected_run_id,
        expected_attempt_id,
    ]
    assert _generation_count(state_dir) == expected_generation_count


@pytest.mark.parametrize("comm_mode", ("fake_backend", "local_tensor"))
def test_direct_communication_modes_receive_identity_without_torchrun(
    launcher_stubs: tuple[Path, Path, Path], comm_mode: str
) -> None:
    _, capture_dir, state_dir = launcher_stubs
    env = _launcher_env(launcher_stubs, COMM_MODE=comm_mode, NGPU="8")

    _run_launcher(RUN_TRAIN, env, "--training.seq_len=128")

    assert _nul_record(capture_dir / "python-env") == [
        GENERATED_RUN_ID,
        GENERATED_ATTEMPT_ID,
        "8",
        "0",
        "",
    ]
    assert _nul_record(capture_dir / "python-argv") == [
        "-m",
        "torchtitan.train",
        "--module",
        "llama3",
        "--config",
        "llama3_debugmodel",
        "--training.seq_len=128",
        f"--comm.mode={comm_mode}",
        "--training.steps",
        "1",
    ]
    assert not (capture_dir / "torchrun-argv").exists()
    assert _generation_count(state_dir) == 2


def test_multinode_launcher_propagates_identity_topology_and_rendezvous(
    launcher_stubs: tuple[Path, Path, Path],
) -> None:
    _, capture_dir, state_dir = launcher_stubs
    env = _launcher_env(launcher_stubs)

    _run_launcher(MULTINODE_TRAINER, env, "--training.seq_len=256")

    expected_identity = [GENERATED_RUN_ID, GENERATED_ATTEMPT_ID]
    assert _nul_record(capture_dir / "srun-env") == expected_identity
    assert _nul_record(capture_dir / "torchrun-env") == expected_identity
    expected_torchrun_argv = [
        "--nnodes",
        "4",
        "--nproc_per_node",
        "8",
        "--rdzv_id",
        GENERATED_RUN_ID,
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        "10.20.30.40:29500",
        "-m",
        "torchtitan.train",
        "--module",
        "llama3",
        "--config",
        "llama3_debugmodel",
        "--training.seq_len=256",
    ]
    assert _nul_record(capture_dir / "srun-argv") == [
        "torchrun",
        *expected_torchrun_argv,
    ]
    assert _nul_record(capture_dir / "torchrun-argv") == expected_torchrun_argv
    assert expected_torchrun_argv.count("--rdzv_id") == 1
    assert _nul_record(capture_dir / "dcgmi-argv") == [
        "profile",
        "--pause",
        "profile",
        "--resume",
    ]
    assert _generation_count(state_dir) == 2


def test_elastic_restart_does_not_modify_preserved_attempt_base(
    launcher_stubs: tuple[Path, Path, Path],
) -> None:
    _, capture_dir, state_dir = launcher_stubs
    env = _launcher_env(
        launcher_stubs,
        NGPU="4",
        TORCHELASTIC_RESTART_COUNT="7",
        TORCHTITAN_ATTEMPT_ID="platform-attempt-base",
        TORCHTITAN_RUN_ID="platform-run",
    )

    _run_launcher(RUN_TRAIN, env)

    assert _nul_record(capture_dir / "torchrun-env") == [
        "platform-run",
        "platform-attempt-base",
    ]
    assert _generation_count(state_dir) == 0
