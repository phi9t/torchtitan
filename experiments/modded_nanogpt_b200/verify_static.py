# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import os
import py_compile
import subprocess
import sys
import tomllib
from json import JSONDecodeError
from pathlib import Path
from tempfile import TemporaryDirectory


SCOPES = (
    "AGENTS.md",
    ".claude/CLAUDE.md",
    ".scratch/modded-nanogpt-b200",
    "docs/agents/agentic-engineering.md",
    "docs/agents/skill-orchestration.md",
    "docs/research/2026-08-18-robotics-state-estimation-training-jobs.md",
    "experiments/modded_nanogpt_b200",
    "pyproject.toml",
    "scripts/rootfs",
    "tests/unit_tests/test_modded_nanogpt_b200_*.py",
    "tests/unit_tests/test_execution_rootfs_selection_shell.py",
    "tests/unit_tests/test_rootfs_*.py",
)

ALWAYS_CHECK_FILES = (
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
)

CHECK_EXTENSIONS = {".py", ".sh", ".md", ".json", ".toml", ".txt", ".lock"}

SOURCED_SHELL_HELPERS = {
    "experiments/modded_nanogpt_b200/rootfs_guard.sh",
    "scripts/rootfs/rootfs_target.sh",
    "scripts/rootfs/runtime_env.sh",
}

EXCLUDED_PREFIXES = (
    ".scratch/trae-pytest-tmp/",
    "experiments/modded_nanogpt_b200/data/",
    "experiments/modded_nanogpt_b200/results/",
    "experiments/modded_nanogpt_b200/sources/",
)


def _run_git(args: list[str], repo_root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return [line for line in proc.stdout.splitlines() if line]


def _repo_root() -> Path:
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"git rev-parse --show-toplevel failed: {message}")
    return Path(proc.stdout.strip())


def _is_excluded(relative_path: str) -> bool:
    if relative_path.startswith(EXCLUDED_PREFIXES):
        return True
    return "__pycache__" in Path(relative_path).parts


def _is_checkable(relative_path: str) -> bool:
    if relative_path in ALWAYS_CHECK_FILES:
        return True
    return Path(relative_path).suffix in CHECK_EXTENSIONS


def list_candidate_files(repo_root: Path) -> list[str]:
    candidates: set[str] = set()
    candidates.update(
        path for path in ALWAYS_CHECK_FILES if (repo_root / path).exists()
    )
    candidates.update(_run_git(["diff", "--name-only", "--", *SCOPES], repo_root))
    candidates.update(
        _run_git(["diff", "--cached", "--name-only", "--", *SCOPES], repo_root)
    )
    candidates.update(
        _run_git(
            ["ls-files", "--others", "--exclude-standard", "--", *SCOPES], repo_root
        )
    )
    return sorted(
        path for path in candidates if _is_checkable(path) and not _is_excluded(path)
    )


def _check_text_file(repo_root: Path, relative_path: str) -> list[str]:
    path = repo_root / relative_path
    data = path.read_bytes()
    errors: list[str] = []
    for line_number, line in enumerate(data.splitlines(keepends=True), start=1):
        content = line.rstrip(b"\r\n")
        if content.endswith((b" ", b"\t")):
            errors.append(f"{relative_path}:{line_number}: trailing whitespace")
    if data and not data.endswith(b"\n"):
        errors.append(f"{relative_path}: missing final newline")
    return errors


def _check_python(repo_root: Path, relative_path: str) -> list[str]:
    path = repo_root / relative_path
    with TemporaryDirectory(prefix="modded-nanogpt-verify-static-") as cache_dir:
        try:
            py_compile.compile(
                str(path), cfile=str(Path(cache_dir) / "compiled.pyc"), doraise=True
            )
        except py_compile.PyCompileError as exc:
            return [f"{relative_path}: python compile error: {exc.msg}"]
    return []


def _check_shell(repo_root: Path, relative_path: str) -> list[str]:
    path = repo_root / relative_path
    proc = subprocess.run(
        ["bash", "-n", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    errors: list[str] = []
    if proc.returncode != 0:
        detail = proc.stdout.strip().replace("\n", " | ")
        errors.append(f"{relative_path}: shell syntax error: {detail}")

    lines = path.read_text(errors="replace").splitlines()
    first_line = lines[0] if lines else ""
    has_shebang = first_line.startswith("#!")
    is_executable = os.access(path, os.X_OK)
    if has_shebang and relative_path not in SOURCED_SHELL_HELPERS and not is_executable:
        errors.append(f"{relative_path}: shell entrypoint is not executable")
    if relative_path in SOURCED_SHELL_HELPERS and is_executable:
        errors.append(f"{relative_path}: sourced shell helper should not be executable")
    return errors


def _check_json(repo_root: Path, relative_path: str) -> list[str]:
    path = repo_root / relative_path
    try:
        json.loads(path.read_text())
    except JSONDecodeError as exc:
        return [f"{relative_path}: json parse error: {exc.msg}"]
    return []


def _check_toml(repo_root: Path, relative_path: str) -> list[str]:
    path = repo_root / relative_path
    try:
        tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        return [f"{relative_path}: toml parse error: {exc}"]
    return []


def check_files(repo_root: Path, files: list[str]) -> list[str]:
    errors: list[str] = []
    for relative_path in files:
        errors.extend(_check_text_file(repo_root, relative_path))
        suffix = Path(relative_path).suffix
        if suffix == ".py":
            errors.extend(_check_python(repo_root, relative_path))
        elif suffix == ".sh":
            errors.extend(_check_shell(repo_root, relative_path))
        elif suffix == ".json":
            errors.extend(_check_json(repo_root, relative_path))
        elif suffix == ".toml":
            errors.extend(_check_toml(repo_root, relative_path))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify static hygiene for Modded NanoGPT B200 harness files."
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="Print candidate files and skip validation.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable candidate and validation results.",
    )
    args = parser.parse_args(argv)

    try:
        repo_root = _repo_root()
        files = list_candidate_files(repo_root)
        if args.list_files:
            for relative_path in files:
                print(relative_path)
            return 0

        errors = check_files(repo_root, files)
    except RuntimeError as exc:
        print(f"verify_static.py: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {"ok": not errors, "files": files, "errors": errors},
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if not errors else 1

    if errors:
        for error in errors:
            print(error)
        return 1

    print(f"Static verification passed for {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
