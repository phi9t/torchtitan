# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY_STATIC = REPO_ROOT / "experiments/modded_nanogpt_b200/verify_static.py"


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def _init_repo(repo: Path) -> None:
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")


def _write(repo: Path, relative_path: str, text: str) -> Path:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _commit_all(repo: Path) -> None:
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-q", "-m", "initial")


def _run_verifier(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VERIFY_STATIC), *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_untracked_markdown_trailing_whitespace_is_detected(tmp_path: Path):
    _init_repo(tmp_path)
    _write(tmp_path, ".scratch/modded-nanogpt-b200/new_note.md", "bad line \n")

    proc = _run_verifier(tmp_path)

    assert proc.returncode == 1
    assert ".scratch/modded-nanogpt-b200/new_note.md:1: trailing whitespace" in (
        proc.stdout
    )


def test_ignored_generated_paths_are_excluded(tmp_path: Path):
    _init_repo(tmp_path)
    _write(tmp_path, ".scratch/trae-pytest-tmp/bad.md", "bad line \n")
    _write(tmp_path, "experiments/modded_nanogpt_b200/data/bad.py", "bad = \n")
    _write(tmp_path, "experiments/modded_nanogpt_b200/results/bad.sh", "if true\n")
    _write(tmp_path, "experiments/modded_nanogpt_b200/sources/bad.md", "bad \n")
    _write(tmp_path, "experiments/modded_nanogpt_b200/__pycache__/bad.py", "bad = \n")
    _write(tmp_path, "experiments/modded_nanogpt_b200/good.py", "value = 1\n")

    proc = _run_verifier(tmp_path)

    assert proc.returncode == 0
    assert "bad" not in proc.stdout


def test_clean_tracked_and_untracked_files_pass(tmp_path: Path):
    _init_repo(tmp_path)
    _write(tmp_path, "experiments/modded_nanogpt_b200/tracked.py", "value = 1\n")
    _write(tmp_path, "experiments/modded_nanogpt_b200/tracked.sh", "true\n")
    _commit_all(tmp_path)
    _write(tmp_path, "experiments/modded_nanogpt_b200/tracked.py", "value = 2\n")
    _write(tmp_path, "experiments/modded_nanogpt_b200/untracked.md", "clean\n")

    proc = _run_verifier(tmp_path)

    assert proc.returncode == 0
    assert "Static verification passed" in proc.stdout


def test_python_compile_and_shell_syntax_errors_are_detected(tmp_path: Path):
    _init_repo(tmp_path)
    _write(tmp_path, "experiments/modded_nanogpt_b200/bad.py", "def bad(:\n")
    _write(tmp_path, "experiments/modded_nanogpt_b200/bad.sh", "if true; then\n")

    proc = _run_verifier(tmp_path)

    assert proc.returncode == 1
    assert "experiments/modded_nanogpt_b200/bad.py: python compile error:" in (
        proc.stdout
    )
    assert "experiments/modded_nanogpt_b200/bad.sh: shell syntax error:" in (
        proc.stdout
    )


def test_list_and_json_modes_include_untracked_scoped_files(tmp_path: Path):
    _init_repo(tmp_path)
    _write(tmp_path, "experiments/modded_nanogpt_b200/untracked.py", "value = 1\n")
    _write(
        tmp_path,
        "tests/unit_tests/test_modded_nanogpt_b200_untracked.py",
        "def test_fixture():\n    assert True\n",
    )

    list_proc = _run_verifier(tmp_path, "--list-files")
    json_proc = _run_verifier(tmp_path, "--json")

    assert list_proc.returncode == 0
    assert "experiments/modded_nanogpt_b200/untracked.py" in list_proc.stdout
    assert "tests/unit_tests/test_modded_nanogpt_b200_untracked.py" in (
        list_proc.stdout
    )
    assert json_proc.returncode == 0
    data = json.loads(json_proc.stdout)
    assert "experiments/modded_nanogpt_b200/untracked.py" in data["files"]
    assert "tests/unit_tests/test_modded_nanogpt_b200_untracked.py" in data["files"]


def test_list_and_json_modes_include_staged_scoped_files(tmp_path: Path):
    _init_repo(tmp_path)
    staged_path = "experiments/modded_nanogpt_b200/staged.py"
    _write(tmp_path, staged_path, "value = 1\n")
    _run_git(tmp_path, "add", staged_path)

    list_proc = _run_verifier(tmp_path, "--list-files")
    json_proc = _run_verifier(tmp_path, "--json")

    assert list_proc.returncode == 0
    assert staged_path in list_proc.stdout
    assert json_proc.returncode == 0
    data = json.loads(json_proc.stdout)
    assert staged_path in data["files"]
