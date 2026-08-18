# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import py_compile
import subprocess
import sys
from tempfile import TemporaryDirectory


SCOPES = (
    ".scratch/modded-nanogpt-b200",
    "experiments/modded_nanogpt_b200",
    "tests/unit_tests/test_modded_nanogpt_b200_*.py",
)

CHECK_EXTENSIONS = {".py", ".sh", ".md"}

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
    return Path(relative_path).suffix in CHECK_EXTENSIONS


def list_candidate_files(repo_root: Path) -> list[str]:
    candidates: set[str] = set()
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
    proc = subprocess.run(
        ["bash", "-n", str(repo_root / relative_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode == 0:
        return []
    detail = proc.stdout.strip().replace("\n", " | ")
    return [f"{relative_path}: shell syntax error: {detail}"]


def check_files(repo_root: Path, files: list[str]) -> list[str]:
    errors: list[str] = []
    for relative_path in files:
        errors.extend(_check_text_file(repo_root, relative_path))
        suffix = Path(relative_path).suffix
        if suffix == ".py":
            errors.extend(_check_python(repo_root, relative_path))
        elif suffix == ".sh":
            errors.extend(_check_shell(repo_root, relative_path))
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
        help="Print candidate files as JSON and skip validation.",
    )
    args = parser.parse_args(argv)

    try:
        repo_root = _repo_root()
        files = list_candidate_files(repo_root)
        if args.json:
            print(json.dumps({"files": files}, indent=2, sort_keys=True))
            return 0
        if args.list_files:
            for relative_path in files:
                print(relative_path)
            return 0

        errors = check_files(repo_root, files)
    except RuntimeError as exc:
        print(f"verify_static.py: {exc}", file=sys.stderr)
        return 2

    if errors:
        for error in errors:
            print(error)
        return 1

    print(f"Static verification passed for {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
