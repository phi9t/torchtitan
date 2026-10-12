# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Apply each manifest mutation and require the named test to fail.

The manifest records triples of (target file, textual mutation, expected
failing test). Five guards in this project were written in a shape that
could not fail -- a test that grepped a shell script instead of running it,
a skip detector whose regex matched zero lines of coloured output, a
coverage ledger keyed on a name a mutant also carried, an argument refusal
that asserted only a non-zero exit, and a label assertion that enshrined an
overclaim. Each was found by mutating the subject, re-running, and noticing
the test still passed. This module makes that standing: for every entry the
mutated tree must FAIL the named test and the pristine tree must PASS it.

Both directions matter. A mutant that still passes is a defect in the test.
A control that already fails means the entry proves nothing, so it is
reported as its own outcome rather than counted as a kill.

This runner is itself a guard, so it must be able to fail. Its own
polarity, including the report for a mutation that does NOT kill its test,
is exercised by tests/unit_tests/test_mutation_runner.py against a
synthetic tree. Nothing here is permanently red.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path


# Copied once per invocation and reverted between entries, so anything that
# is generated, versioned or an output directory is left out. Scratch trees
# live under outputs/, which is why outputs/ must not be copied into one.
_SKIPPED_TREE_NAMES = frozenset(
    {
        ".git",
        ".cache",
        "outputs",
        "__pycache__",
        ".pytest_cache",
        ".ipynb_checkpoints",
        ".mypy_cache",
        ".ruff_cache",
    }
)

# A mutated subject must make the named test FAIL, and pytest's exit code
# for "tests ran and failed" is 1. Any other non-zero code means something
# else broke, and reporting that as a kill would claim a failure mechanism
# that was never measured. Measured here rather than read off the exit-code
# table: a mutation that makes the named test's module uncollectable exits 4,
# pytest's usage error, which
# test_runner_reports_a_mutant_that_stopped_the_test_as_an_error pins.
_PYTEST_TESTS_FAILED = 1

_TEST_TIMEOUT_SECONDS = 1800

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
# Anchored per line, over output the caller has already stripped. The
# unstripped form of exactly this regex is the guard that matched zero lines
# of "\x1b[33mSKIPPED" and let a gate pass because it did nothing.
_SKIPPED_LINE = re.compile(r"^SKIPPED", re.MULTILINE)
_PASSED_COUNT = re.compile(r"(\d+) passed")

_REQUIRED_ENTRY_KEYS = frozenset({"id", "target", "test", "find", "replace", "why"})
_OPTIONAL_ENTRY_KEYS = frozenset({"occurrences"})

KILLED = "killed"
SURVIVED = "survived"
CONTROL_FAILED = "control_failed"
ERROR = "error"


@dataclasses.dataclass(frozen=True)
class MutationEntry:
    """One (target file, textual mutation, expected failing test) triple."""

    entry_id: str
    target: str
    test: str
    find: str
    replace: str
    occurrences: int
    why: str


@dataclasses.dataclass(frozen=True)
class EntryResult:
    """What the runner measured for one entry."""

    entry_id: str
    target: str
    test: str
    outcome: str
    detail: str = ""
    control_exit: int | None = None
    control_passed: int | None = None
    control_seconds: float | None = None
    mutant_exit: int | None = None
    mutant_seconds: float | None = None
    occurrences_found: int | None = None
    revert: str = "not_reached"
    output_tail: str = ""


@dataclasses.dataclass(frozen=True)
class ManifestResult:
    """The whole run: one EntryResult per selected entry, plus totals."""

    results: tuple[EntryResult, ...]
    entries_total: int
    manifest: str
    wall_seconds: float
    control_runs: int
    mutant_runs: int
    copy_seconds: float = 0.0

    @property
    def killed(self) -> int:
        """Return how many selected entries killed their named test.

        Returns:
            The number of entries whose outcome is KILLED.
        """

        return sum(1 for result in self.results if result.outcome == KILLED)

    @property
    def selected(self) -> int:
        """Return how many entries this run actually measured.

        Returns:
            The number of EntryResult records produced.
        """

        return len(self.results)

    @property
    def ok(self) -> bool:
        """Return whether every entry in the whole manifest killed.

        A subset selection is never ok: a partial run must not be readable
        as a full one.

        Returns:
            True when every manifest entry was selected and killed.
        """

        return (
            self.selected == self.entries_total
            and self.entries_total > 0
            and self.killed == self.entries_total
        )


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    """Hash a file by reading it, never from a recorded value.

    Args:
        path: File to read.

    Returns:
        Hex sha256 of the bytes currently on disk.
    """

    return _digest(path.read_bytes())


def load_manifest(manifest_path: Path) -> tuple[MutationEntry, ...]:
    """Parse the manifest, rejecting anything a typo could hide.

    Unknown keys are refused rather than ignored: a misspelled ``test`` key
    would otherwise leave an entry pointing at nothing while the file still
    parsed.

    Args:
        manifest_path: Path to the TOML manifest.

    Returns:
        The manifest entries in file order.

    Raises:
        ValueError: If the manifest is malformed, has unknown keys,
            duplicate ids, a no-op mutation, or an unsafe target path.
    """

    raw = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    unknown_sections = set(raw) - {"mutation"}
    if unknown_sections:
        raise ValueError(
            f"{manifest_path}: unknown top-level keys: " f"{sorted(unknown_sections)}"
        )
    tables = raw.get("mutation")
    if not isinstance(tables, list) or not tables:
        raise ValueError(f"{manifest_path}: expected a non-empty [[mutation]] array")

    entries: list[MutationEntry] = []
    seen_ids: set[str] = set()
    for index, table in enumerate(tables):
        where = f"{manifest_path}: [[mutation]] #{index + 1}"
        if not isinstance(table, dict):
            raise ValueError(f"{where}: is not a table")
        keys = set(table)
        missing = _REQUIRED_ENTRY_KEYS - keys
        if missing:
            raise ValueError(f"{where}: missing keys: {sorted(missing)}")
        unknown = keys - _REQUIRED_ENTRY_KEYS - _OPTIONAL_ENTRY_KEYS
        if unknown:
            raise ValueError(f"{where}: unknown keys: {sorted(unknown)}")
        for name in ("id", "target", "test", "find", "replace", "why"):
            if not isinstance(table[name], str):
                raise ValueError(f"{where}: {name} must be a string")
        entry_id = table["id"]
        if not entry_id or entry_id in seen_ids:
            raise ValueError(f"{where}: id must be unique and non-empty: {entry_id!r}")
        seen_ids.add(entry_id)
        target = table["target"]
        target_path = Path(target)
        if target_path.is_absolute() or ".." in target_path.parts:
            raise ValueError(f"{where}: target must be a relative in-tree path")
        test = table["test"]
        if "::" not in test:
            raise ValueError(f"{where}: test must be a pytest node id with '::'")
        find = table["find"]
        if not find:
            raise ValueError(f"{where}: find must not be empty")
        if find == table["replace"]:
            raise ValueError(f"{where}: find equals replace, so the entry is a no-op")
        occurrences = table.get("occurrences", 1)
        if not isinstance(occurrences, int) or isinstance(occurrences, bool):
            raise ValueError(f"{where}: occurrences must be an integer")
        if occurrences < 1:
            raise ValueError(f"{where}: occurrences must be at least 1")
        if not table["why"].strip():
            raise ValueError(f"{where}: why must say what the mutation reintroduces")
        entries.append(
            MutationEntry(
                entry_id=entry_id,
                target=target,
                test=test,
                find=find,
                replace=table["replace"],
                occurrences=occurrences,
                why=table["why"],
            )
        )
    return tuple(entries)


def _ignore_tree_names(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _SKIPPED_TREE_NAMES}


def stage_tree(source_root: Path, destination: Path) -> Path:
    """Copy the source tree once, so entries pay for one copy between them.

    Args:
        source_root: Tree to copy.
        destination: Directory to create as the copy.

    Returns:
        The staged tree root.
    """

    shutil.copytree(
        source_root,
        destination,
        symlinks=True,
        ignore=_ignore_tree_names,
    )
    return destination


@dataclasses.dataclass(frozen=True)
class _PytestRun:
    exit_code: int
    seconds: float
    passed: int | None
    skipped_lines: tuple[str, ...]
    output: str
    timed_out: bool


def _run_pytest(
    tree: Path,
    node_id: str,
    *,
    python_executable: str,
    timeout_seconds: int,
) -> _PytestRun:
    command = [
        python_executable,
        "-m",
        "pytest",
        "-q",
        "-rs",
        "--color=no",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        node_id,
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=tree,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as expired:
        elapsed = time.monotonic() - started
        partial = expired.output or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", "replace")
        return _PytestRun(
            exit_code=-1,
            seconds=elapsed,
            passed=None,
            skipped_lines=(),
            output=partial,
            timed_out=True,
        )
    elapsed = time.monotonic() - started
    # --color=no should already have removed the escapes; strip them anyway,
    # because the regex below is anchored and a surviving colour default
    # would make it match nothing at all.
    plain = _ANSI_ESCAPE.sub("", completed.stdout)
    passed_match = _PASSED_COUNT.search(plain)
    return _PytestRun(
        exit_code=completed.returncode,
        seconds=elapsed,
        passed=int(passed_match.group(1)) if passed_match else None,
        skipped_lines=tuple(
            line for line in plain.splitlines() if _SKIPPED_LINE.match(line)
        ),
        output=plain,
        timed_out=False,
    )


def _tail(text: str, limit: int = 2000) -> str:
    collapsed = text.strip()
    if len(collapsed) <= limit:
        return collapsed
    return "..." + collapsed[-limit:]


def run_entry(
    entry: MutationEntry,
    *,
    source_root: Path,
    tree: Path,
    control: _PytestRun | None,
    python_executable: str,
    timeout_seconds: int,
) -> tuple[EntryResult, _PytestRun | None]:
    """Measure one entry in both directions.

    Args:
        entry: The manifest entry.
        source_root: The pristine tree the staged copy came from.
        tree: The staged copy this entry mutates and then reverts.
        control: A cached unmutated run of the same node id, or None.
        python_executable: Interpreter used to run pytest.
        timeout_seconds: Per-pytest-run timeout.

    Returns:
        The entry's result and the control run for its node id, so a later
        entry naming the same test can reuse it.
    """

    source_file = source_root / entry.target
    staged_file = tree / entry.target
    if not source_file.is_file():
        return (
            EntryResult(
                entry_id=entry.entry_id,
                target=entry.target,
                test=entry.test,
                outcome=ERROR,
                detail=f"target does not exist: {entry.target}",
            ),
            control,
        )
    pristine = source_file.read_bytes()
    if not staged_file.is_file() or _file_digest(staged_file) != _digest(pristine):
        # Either the copy was not faithful, or an earlier entry's revert was
        # not exact. Both make every later measurement meaningless, so this
        # is an error rather than a survived entry.
        return (
            EntryResult(
                entry_id=entry.entry_id,
                target=entry.target,
                test=entry.test,
                outcome=ERROR,
                detail=(
                    "staged copy does not match the source tree before "
                    "mutation: either an earlier entry's revert was not "
                    "exact, or the source changed while this run was in "
                    "flight"
                ),
            ),
            control,
        )
    try:
        text = pristine.decode("utf-8")
    except UnicodeDecodeError as error:
        return (
            EntryResult(
                entry_id=entry.entry_id,
                target=entry.target,
                test=entry.test,
                outcome=ERROR,
                detail=f"target is not utf-8 text: {error}",
            ),
            control,
        )
    found = text.count(entry.find)
    if found != entry.occurrences:
        return (
            EntryResult(
                entry_id=entry.entry_id,
                target=entry.target,
                test=entry.test,
                outcome=ERROR,
                occurrences_found=found,
                detail=(
                    f"find appears {found} times, manifest declares "
                    f"{entry.occurrences}; the anchor drifted"
                ),
            ),
            control,
        )

    if control is None:
        control = _run_pytest(
            tree,
            entry.test,
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
    control_fields = {
        "control_exit": control.exit_code,
        "control_passed": control.passed,
        "control_seconds": round(control.seconds, 2),
        "occurrences_found": found,
    }
    control_problem = ""
    if control.timed_out:
        control_problem = f"control run timed out after {timeout_seconds}s"
    elif control.exit_code != 0:
        control_problem = f"control run exited {control.exit_code}, expected 0"
    elif control.skipped_lines:
        # A skipped control is the "antecedent never satisfied" shape: the
        # mutant would also skip, exit 0, and be reported as survived for a
        # reason that has nothing to do with the test.
        control_problem = f"control run skipped: {control.skipped_lines[0]}"
    elif not control.passed:
        control_problem = "control run reported no passing test"
    if control_problem:
        return (
            EntryResult(
                entry_id=entry.entry_id,
                target=entry.target,
                test=entry.test,
                outcome=CONTROL_FAILED,
                detail=control_problem,
                output_tail=_tail(control.output),
                **control_fields,
            ),
            control,
        )

    mutated = text.replace(entry.find, entry.replace)
    assert mutated != text, "load_manifest must reject a no-op mutation"
    mutated_bytes = mutated.encode("utf-8")
    staged_file.write_bytes(mutated_bytes)
    # Recomputed from disk rather than trusted: a write that silently did
    # not land would leave the pristine tree under test and report a
    # survived mutation that was never applied.
    assert _file_digest(staged_file) == _digest(
        mutated_bytes
    ), f"mutation did not land on disk: {staged_file}"
    try:
        mutant = _run_pytest(
            tree,
            entry.test,
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
    finally:
        staged_file.write_bytes(pristine)
    revert_digest = _file_digest(staged_file)
    source_digest = _file_digest(source_file)
    revert = "exact" if revert_digest == source_digest else "inexact"
    mutant_fields = {
        "mutant_exit": mutant.exit_code,
        "mutant_seconds": round(mutant.seconds, 2),
        "revert": revert,
    }
    if revert != "exact":
        return (
            EntryResult(
                entry_id=entry.entry_id,
                target=entry.target,
                test=entry.test,
                outcome=ERROR,
                detail=(
                    f"revert left {revert_digest[:12]} where the source has "
                    f"{source_digest[:12]}"
                ),
                **control_fields,
                **mutant_fields,
            ),
            control,
        )
    if mutant.timed_out:
        return (
            EntryResult(
                entry_id=entry.entry_id,
                target=entry.target,
                test=entry.test,
                outcome=ERROR,
                detail=f"mutant run timed out after {timeout_seconds}s",
                output_tail=_tail(mutant.output),
                **control_fields,
                **mutant_fields,
            ),
            control,
        )
    if mutant.skipped_lines:
        return (
            EntryResult(
                entry_id=entry.entry_id,
                target=entry.target,
                test=entry.test,
                outcome=ERROR,
                detail=f"mutant run skipped: {mutant.skipped_lines[0]}",
                output_tail=_tail(mutant.output),
                **control_fields,
                **mutant_fields,
            ),
            control,
        )
    if mutant.exit_code == 0:
        return (
            EntryResult(
                entry_id=entry.entry_id,
                target=entry.target,
                test=entry.test,
                outcome=SURVIVED,
                detail=(
                    "mutant_did_not_fail: the named test passes on the "
                    "mutated tree, so it does not check this mutation"
                ),
                output_tail=_tail(mutant.output),
                **control_fields,
                **mutant_fields,
            ),
            control,
        )
    if mutant.exit_code != _PYTEST_TESTS_FAILED:
        return (
            EntryResult(
                entry_id=entry.entry_id,
                target=entry.target,
                test=entry.test,
                outcome=ERROR,
                detail=(
                    f"mutant run exited {mutant.exit_code}, not "
                    f"{_PYTEST_TESTS_FAILED}; the test did not fail, "
                    "something else did"
                ),
                output_tail=_tail(mutant.output),
                **control_fields,
                **mutant_fields,
            ),
            control,
        )
    return (
        EntryResult(
            entry_id=entry.entry_id,
            target=entry.target,
            test=entry.test,
            outcome=KILLED,
            **control_fields,
            **mutant_fields,
        ),
        control,
    )


def run_manifest(
    *,
    source_root: Path,
    manifest_path: Path,
    scratch_root: Path,
    only: tuple[str, ...] = (),
    python_executable: str | None = None,
    timeout_seconds: int = _TEST_TIMEOUT_SECONDS,
    report: bool = True,
) -> ManifestResult:
    """Stage one copy of the tree and measure every selected entry on it.

    Args:
        source_root: The pristine tree to copy and compare digests against.
        manifest_path: Path to the TOML manifest.
        scratch_root: Directory to create the staged copy under. It must not
            be inside source_root's copied set.
        only: Entry ids to measure; empty means all of them.
        python_executable: Interpreter used to run pytest.
        timeout_seconds: Per-pytest-run timeout.
        report: Whether to print one QFV_MUT line per entry as it finishes.

    Returns:
        The measured result for the selected entries.

    Raises:
        ValueError: If `only` names an id the manifest does not define.
    """

    # A scratch root inside the copied set would make copytree recurse into
    # its own destination. outputs/ is excluded from the copy, which is why
    # the default lives there; anything else inside the tree is refused.
    source_root = source_root.resolve()
    scratch_root = scratch_root.resolve()
    if scratch_root == source_root or source_root in scratch_root.parents:
        parts = scratch_root.relative_to(source_root).parts
        if not parts or parts[0] not in _SKIPPED_TREE_NAMES:
            raise ValueError(
                f"scratch root {scratch_root} is inside the copied tree at "
                f"{source_root}; put it outside, or under one of "
                f"{sorted(_SKIPPED_TREE_NAMES)}"
            )

    entries = load_manifest(manifest_path)
    if only:
        by_id = {entry.entry_id: entry for entry in entries}
        unknown = sorted(set(only) - set(by_id))
        if unknown:
            raise ValueError(f"{manifest_path}: no such entry id: {unknown}")
        selected = tuple(entry for entry in entries if entry.entry_id in set(only))
    else:
        selected = entries

    scratch_root.mkdir(parents=True, exist_ok=True)
    # One copy for the whole manifest, reverted exactly between entries and
    # re-digested against the source before each one. Reported separately so
    # the per-entry timings are not read as including it.
    copy_started = time.monotonic()
    tree = stage_tree(source_root, scratch_root / "tree")
    copy_seconds = time.monotonic() - copy_started
    started = time.monotonic()
    controls: dict[str, _PytestRun] = {}
    results: list[EntryResult] = []
    control_runs = 0
    mutant_runs = 0
    for entry in selected:
        had_control = entry.test in controls
        result, control = run_entry(
            entry,
            source_root=source_root,
            tree=tree,
            control=controls.get(entry.test),
            python_executable=python_executable or sys.executable,
            timeout_seconds=timeout_seconds,
        )
        if control is not None:
            controls[entry.test] = control
            if not had_control:
                control_runs += 1
        if result.mutant_exit is not None:
            mutant_runs += 1
        results.append(result)
        if report:
            print(format_entry(result), flush=True)
    return ManifestResult(
        results=tuple(results),
        entries_total=len(entries),
        manifest=str(manifest_path),
        wall_seconds=time.monotonic() - started,
        control_runs=control_runs,
        mutant_runs=mutant_runs,
        copy_seconds=copy_seconds,
    )


def format_entry(result: EntryResult) -> str:
    """Render one entry as a single greppable result line.

    Args:
        result: The measured entry.

    Returns:
        A QFV_MUT line naming the entry, the target, the test, and both
        directions, so an aggregate result cannot hide which entry did what.
    """

    fields = [
        f"QFV_MUT entry={result.entry_id}",
        f"result={result.outcome}",
        f"target={result.target}",
        f"test={result.test}",
        f"control_exit={_render(result.control_exit)}",
        f"control_passed={_render(result.control_passed)}",
        f"mutant_exit={_render(result.mutant_exit)}",
        f"occurrences={_render(result.occurrences_found)}",
        f"revert={result.revert}",
        f"control_s={_render(result.control_seconds)}",
        f"mutant_s={_render(result.mutant_seconds)}",
    ]
    line = " ".join(fields)
    if result.detail:
        line += f" detail={result.detail.splitlines()[0]!r}"
    return line


def _render(value: object) -> str:
    return "none" if value is None else str(value)


def format_summary(result: ManifestResult) -> str:
    """Render the totals, with the counts the decision is made on.

    Args:
        result: The measured manifest run.

    Returns:
        A QFV_MUT_SUMMARY line.
    """

    return (
        f"QFV_MUT_SUMMARY killed={result.killed}/{result.entries_total}"
        f" entries_selected={result.selected}"
        f" entries_total={result.entries_total}"
        f" control_runs={result.control_runs}"
        f" mutant_runs={result.mutant_runs}"
        f" manifest={result.manifest}"
        f" tree_copy_s={result.copy_seconds:.1f}"
        f" wall_s={result.wall_seconds:.1f}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the manifest and exit non-zero unless every entry killed.

    Args:
        argv: Command line arguments, defaulting to sys.argv[1:].

    Returns:
        0 when every manifest entry was measured and killed, else 1.
    """

    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        prog="python -m tests.mutations.runner",
        description=(
            "Apply each manifest mutation to a scratch copy of the tree and "
            "require the named test to fail there and pass unmutated."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "tests" / "mutations" / "manifest.toml",
    )
    parser.add_argument("--source-root", type=Path, default=repo_root)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=None,
        help="Directory for the staged copy; default is a unique directory "
        "under outputs/.",
    )
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        metavar="ID",
        help="Measure only this entry id; repeatable. A subset run can "
        "never report success.",
    )
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--timeout", type=int, default=_TEST_TIMEOUT_SECONDS)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the manifest entries and exit without running anything.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for entry in load_manifest(args.manifest):
            print(f"{entry.entry_id}\t{entry.target}\t{entry.test}")
        return 0

    scratch_root = args.scratch_root
    if scratch_root is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        scratch_root = args.source_root / "outputs" / f"qfv-mutations-{stamp}"
    try:
        result = run_manifest(
            source_root=args.source_root,
            manifest_path=args.manifest,
            scratch_root=scratch_root,
            only=tuple(args.entry),
            timeout_seconds=args.timeout,
        )
    finally:
        if not args.keep_scratch and scratch_root.exists():
            shutil.rmtree(scratch_root, ignore_errors=True)

    print(format_summary(result), flush=True)
    if result.ok:
        return 0
    for entry_result in result.results:
        if entry_result.outcome != KILLED:
            print(
                f"error: entry {entry_result.entry_id} is {entry_result.outcome}:"
                f" {entry_result.detail}",
                file=sys.stderr,
            )
            if entry_result.output_tail:
                print(entry_result.output_tail, file=sys.stderr)
    if result.selected != result.entries_total:
        print(
            f"error: measured {result.selected} of {result.entries_total} "
            "entries, so this run cannot report success",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
