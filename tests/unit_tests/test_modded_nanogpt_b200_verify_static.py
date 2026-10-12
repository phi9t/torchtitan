# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY_STATIC = REPO_ROOT / "experiments/modded_nanogpt_b200/verify_static.py"
CRITICAL_HANDOFF_FILES = {
    ".gitignore",
    "AGENTS.md",
    ".claude/CLAUDE.md",
    ".scratch/modded-nanogpt-b200/completion_audit.md",
    ".scratch/modded-nanogpt-b200/execution_prompt.md",
    ".scratch/modded-nanogpt-b200/master_plan.md",
    ".scratch/modded-nanogpt-b200/spec.md",
    ".scratch/modded-nanogpt-b200/rootfs_runtime_env_spec.md",
    ".scratch/modded-nanogpt-b200/schema_matrix_spec.md",
    ".scratch/modded-nanogpt-b200/issues/14-nanogpt-performance-probe-ladder.md",
    ".scratch/modded-nanogpt-b200/issues/15-documentation-review-and-handoff.md",
    "experiments/modded_nanogpt_b200/preflight_checklist.md",
    "tests/unit_tests/test_modded_nanogpt_b200_tracker.py",
    "tests/unit_tests/test_modded_nanogpt_b200_verify_static.py",
}


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
    _write(tmp_path, ".gitignore", "a.out\n")
    _write(tmp_path, "a.out", "not review evidence \n")
    _write(tmp_path, ".scratch/trae-pytest-tmp/bad.md", "bad line \n")
    _write(tmp_path, "experiments/modded_nanogpt_b200/data/bad.py", "bad = \n")
    _write(tmp_path, "experiments/modded_nanogpt_b200/results/bad.sh", "if true\n")
    _write(tmp_path, "experiments/modded_nanogpt_b200/sources/bad.md", "bad \n")
    _write(tmp_path, "experiments/modded_nanogpt_b200/__pycache__/bad.py", "bad = \n")
    _write(tmp_path, "experiments/modded_nanogpt_b200/good.py", "value = 1\n")

    proc = _run_verifier(tmp_path)

    assert proc.returncode == 0
    assert "bad" not in proc.stdout
    assert "a.out" not in proc.stdout


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


def test_shell_entrypoint_mode_errors_are_detected(tmp_path: Path):
    _init_repo(tmp_path)
    _write(
        tmp_path,
        "experiments/modded_nanogpt_b200/run_something.sh",
        "#!/usr/bin/env bash\ntrue\n",
    )

    proc = _run_verifier(tmp_path)

    assert proc.returncode == 1
    assert (
        "experiments/modded_nanogpt_b200/run_something.sh: "
        "shell entrypoint is not executable"
    ) in proc.stdout


def test_sourced_shell_helper_mode_errors_are_detected(tmp_path: Path):
    _init_repo(tmp_path)
    helper = _write(
        tmp_path,
        "experiments/modded_nanogpt_b200/rootfs_guard.sh",
        "#!/usr/bin/env bash\ntrue\n",
    )
    helper.chmod(0o755)

    proc = _run_verifier(tmp_path)

    assert proc.returncode == 1
    assert (
        "experiments/modded_nanogpt_b200/rootfs_guard.sh: "
        "sourced shell helper should not be executable"
    ) in proc.stdout


def test_executable_shell_entrypoints_and_sourced_helpers_pass(tmp_path: Path):
    _init_repo(tmp_path)
    entrypoint = _write(
        tmp_path,
        "experiments/modded_nanogpt_b200/run_something.sh",
        "#!/usr/bin/env bash\ntrue\n",
    )
    entrypoint.chmod(0o755)
    _write(
        tmp_path,
        "experiments/modded_nanogpt_b200/rootfs_guard.sh",
        "#!/usr/bin/env bash\ntrue\n",
    )

    proc = _run_verifier(tmp_path)

    assert proc.returncode == 0
    assert "Static verification passed" in proc.stdout


def test_json_parse_errors_are_detected(tmp_path: Path):
    _init_repo(tmp_path)
    _write(
        tmp_path,
        "experiments/modded_nanogpt_b200/runtime/schemas/bad.schema.json",
        "{bad\n",
    )

    proc = _run_verifier(tmp_path)

    assert proc.returncode == 1
    assert (
        "experiments/modded_nanogpt_b200/runtime/schemas/bad.schema.json: "
        "json parse error:"
    ) in proc.stdout


def test_toml_parse_errors_are_detected(tmp_path: Path):
    _init_repo(tmp_path)
    _write(tmp_path, "pyproject.toml", "[tool.pyrefly\n")

    proc = _run_verifier(tmp_path)

    assert proc.returncode == 1
    assert "pyproject.toml: toml parse error:" in proc.stdout


def test_pyrefly_excludes_generated_rootfs_tree():
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    excludes = config["tool"]["pyrefly"]["project-excludes"]

    assert "scripts/rootfs/rootfs" in excludes


def test_list_and_json_modes_include_untracked_scoped_files(tmp_path: Path):
    _init_repo(tmp_path)
    _write(tmp_path, "AGENTS.md", "canonical\n")
    _write(tmp_path, ".claude/CLAUDE.md", "mirror\n")
    _write(
        tmp_path,
        ".scratch/modded-nanogpt-b200/completion_audit.md",
        "audit\n",
    )
    _write(tmp_path, "experiments/modded_nanogpt_b200/untracked.py", "value = 1\n")
    _write(
        tmp_path,
        "experiments/modded_nanogpt_b200/configs/untracked.json",
        "{}\n",
    )
    _write(
        tmp_path,
        "experiments/modded_nanogpt_b200/runtime/requirements.lock",
        "numpy==2.5.2\n",
    )
    _write(tmp_path, "docs/agents/agentic-engineering.md", "workflow\n")
    _write(tmp_path, "docs/agents/skill-orchestration.md", "skills\n")
    _write(
        tmp_path,
        "docs/research/2026-08-18-robotics-state-estimation-training-jobs.md",
        "research\n",
    )
    _write(tmp_path, "pyproject.toml", "[tool.pyrefly]\n")
    _write(tmp_path, "scripts/rootfs/untracked.sh", "true\n")
    _write(
        tmp_path,
        "tests/unit_tests/test_modded_nanogpt_b200_untracked.py",
        "def test_fixture():\n    assert True\n",
    )
    _write(
        tmp_path,
        "tests/unit_tests/test_execution_rootfs_selection_shell.py",
        "def test_execution_rootfs_selection_fixture():\n    assert True\n",
    )
    _write(
        tmp_path,
        "tests/unit_tests/test_rootfs_untracked.py",
        "def test_rootfs_fixture():\n    assert True\n",
    )

    list_proc = _run_verifier(tmp_path, "--list-files")
    json_proc = _run_verifier(tmp_path, "--json")

    assert list_proc.returncode == 0
    list_files = list_proc.stdout.splitlines()
    assert "AGENTS.md" in list_proc.stdout
    assert ".claude/CLAUDE.md" in list_proc.stdout
    assert ".scratch/modded-nanogpt-b200/completion_audit.md" in list_proc.stdout
    assert "experiments/modded_nanogpt_b200/untracked.py" in list_proc.stdout
    assert "experiments/modded_nanogpt_b200/configs/untracked.json" in (
        list_proc.stdout
    )
    assert "experiments/modded_nanogpt_b200/runtime/requirements.lock" in (
        list_proc.stdout
    )
    assert "docs/agents/agentic-engineering.md" in list_proc.stdout
    assert "docs/agents/skill-orchestration.md" in list_proc.stdout
    assert (
        "docs/research/2026-08-18-robotics-state-estimation-training-jobs.md"
        in list_proc.stdout
    )
    assert "pyproject.toml" in list_proc.stdout
    assert "scripts/rootfs/untracked.sh" in list_proc.stdout
    assert "tests/unit_tests/test_modded_nanogpt_b200_untracked.py" in (
        list_proc.stdout
    )
    assert "tests/unit_tests/test_execution_rootfs_selection_shell.py" in (
        list_proc.stdout
    )
    assert "tests/unit_tests/test_rootfs_untracked.py" in list_proc.stdout
    assert json_proc.returncode == 0
    data = json.loads(json_proc.stdout)
    assert data["ok"] is True
    assert data["errors"] == []
    assert data["files"] == list_files
    assert "AGENTS.md" in data["files"]
    assert ".claude/CLAUDE.md" in data["files"]
    assert ".scratch/modded-nanogpt-b200/completion_audit.md" in data["files"]
    assert "experiments/modded_nanogpt_b200/untracked.py" in data["files"]
    assert "experiments/modded_nanogpt_b200/configs/untracked.json" in data["files"]
    assert "experiments/modded_nanogpt_b200/runtime/requirements.lock" in data["files"]
    assert "docs/agents/agentic-engineering.md" in data["files"]
    assert "docs/agents/skill-orchestration.md" in data["files"]
    assert (
        "docs/research/2026-08-18-robotics-state-estimation-training-jobs.md"
        in data["files"]
    )
    assert "pyproject.toml" in data["files"]
    assert "scripts/rootfs/untracked.sh" in data["files"]
    assert "tests/unit_tests/test_modded_nanogpt_b200_untracked.py" in data["files"]
    assert "tests/unit_tests/test_execution_rootfs_selection_shell.py" in data["files"]
    assert "tests/unit_tests/test_rootfs_untracked.py" in data["files"]


def test_clean_tracked_critical_handoff_files_are_always_included(tmp_path: Path):
    _init_repo(tmp_path)
    for relative_path in CRITICAL_HANDOFF_FILES:
        _write(tmp_path, relative_path, "critical\n")
    _commit_all(tmp_path)

    list_proc = _run_verifier(tmp_path, "--list-files")
    json_proc = _run_verifier(tmp_path, "--json")

    assert list_proc.returncode == 0
    for relative_path in CRITICAL_HANDOFF_FILES:
        assert relative_path in list_proc.stdout
    assert json_proc.returncode == 0
    data = json.loads(json_proc.stdout)
    assert CRITICAL_HANDOFF_FILES <= set(data["files"])


def test_current_static_surface_includes_critical_handoff_files():
    json_proc = _run_verifier(REPO_ROOT, "--json")

    assert json_proc.returncode == 0
    data = json.loads(json_proc.stdout)
    assert data["ok"] is True
    assert data["errors"] == []
    assert CRITICAL_HANDOFF_FILES <= set(data["files"])


def test_current_list_and_json_modes_match_static_surface():
    list_proc = _run_verifier(REPO_ROOT, "--list-files")
    json_proc = _run_verifier(REPO_ROOT, "--json")

    assert list_proc.returncode == 0
    assert json_proc.returncode == 0
    data = json.loads(json_proc.stdout)
    assert data["ok"] is True
    assert data["errors"] == []
    assert data["files"] == list_proc.stdout.splitlines()


def test_list_and_json_modes_include_staged_scoped_files(tmp_path: Path):
    _init_repo(tmp_path)
    staged_path = "experiments/modded_nanogpt_b200/staged.py"
    _write(tmp_path, staged_path, "value = 1\n")
    _run_git(tmp_path, "add", staged_path)

    list_proc = _run_verifier(tmp_path, "--list-files")
    json_proc = _run_verifier(tmp_path, "--json")

    assert list_proc.returncode == 0
    list_files = list_proc.stdout.splitlines()
    assert staged_path in list_proc.stdout
    assert json_proc.returncode == 0
    data = json.loads(json_proc.stdout)
    assert data["ok"] is True
    assert data["errors"] == []
    assert data["files"] == list_files
    assert staged_path in data["files"]


def test_json_mode_reports_validation_errors(tmp_path: Path):
    _init_repo(tmp_path)
    bad_path = "experiments/modded_nanogpt_b200/bad.py"
    _write(tmp_path, bad_path, "def bad(:\n")

    json_proc = _run_verifier(tmp_path, "--json")

    assert json_proc.returncode == 1
    data = json.loads(json_proc.stdout)
    assert data["ok"] is False
    assert bad_path in data["files"]
    assert len(data["errors"]) == 1
    assert data["errors"][0].startswith(f"{bad_path}: python compile error:")
