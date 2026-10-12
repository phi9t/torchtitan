# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Self-tests for the mutation runner, including its non-killing report.

The runner is a guard, so it must be a guard that can fail. A runner with an
inverted condition or a swallowed exit code would report killed=N/N forever
and be worse than nothing, because it would launder confidence. These tests
drive it over a synthetic tree whose expected outcome for each entry is
known, so the report for a mutation that does NOT kill its named test is
measured rather than assumed.

The synthetic entries live in throwaway manifests under tmp_path. Nothing
here adds a permanently-red entry to tests/mutations/manifest.toml, so a real
failure there is never confused with this file's deliberate one.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from tests.mutations import runner


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "tests" / "mutations" / "manifest.toml"

# A subject whose behaviour depends on a numeric floor, in the shape of the
# branching floor that a real test once asserted by grepping for "-ge 2".
_FLOOR_SCRIPT = """\
#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-0}" -ge 2 ]] || {
  echo "outdegree too low" >&2
  exit 1
}
echo ok
"""

# Four tests over that subject, one per outcome the runner must distinguish:
# a test that runs it (killed), a test that greps it (survived -- the canary),
# a test that is already red (control_failed), and a test that skips
# (control_failed, because a skipped control makes the mutant's exit 0
# meaningless).
_FLOOR_TESTS = """\
import subprocess
from pathlib import Path

import pytest


FLOOR = Path(__file__).resolve().parent / "subject" / "floor.sh"


def test_running_the_floor_refuses_one() -> None:
    done = subprocess.run(
        ["bash", str(FLOOR), "1"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode != 0
    assert "outdegree too low" in done.stderr


def test_reading_the_floor_mentions_a_threshold() -> None:
    assert "-ge" in FLOOR.read_text()


def test_that_fails_before_any_mutation() -> None:
    raise AssertionError("red before the manifest touches anything")


def test_that_skips_before_any_mutation() -> None:
    pytest.skip("nothing materialized")
"""

_FLOOR_MUTATION = """\
find = '[[ "${1:-0}" -ge 2 ]]'
replace = '[[ "${1:-0}" -ge 1 ]]'
"""

_SELF_MANIFEST = f"""\
[[mutation]]
id = "behavioural"
target = "subject/floor.sh"
test = "test_floor.py::test_running_the_floor_refuses_one"
{_FLOOR_MUTATION}why = "Lowering the floor must fail a test that runs the subject."

[[mutation]]
id = "text-only-canary"
target = "subject/floor.sh"
test = "test_floor.py::test_reading_the_floor_mentions_a_threshold"
{_FLOOR_MUTATION}why = "A test that greps the subject cannot see the floor move."

[[mutation]]
id = "already-red-control"
target = "subject/floor.sh"
test = "test_floor.py::test_that_fails_before_any_mutation"
{_FLOOR_MUTATION}why = "A control that already fails proves nothing either way."

[[mutation]]
id = "skipping-control"
target = "subject/floor.sh"
test = "test_floor.py::test_that_skips_before_any_mutation"
{_FLOOR_MUTATION}why = "A skipped control is an antecedent never satisfied."
"""


def _make_tree(root: Path) -> Path:
    """Write the synthetic subject and its four tests.

    Args:
        root: Directory to create as the synthetic source tree.

    Returns:
        The tree root.
    """

    (root / "subject").mkdir(parents=True)
    floor = root / "subject" / "floor.sh"
    floor.write_text(_FLOOR_SCRIPT)
    floor.chmod(0o755)
    (root / "test_floor.py").write_text(_FLOOR_TESTS)
    return root


def _write_manifest(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_runner_reports_a_mutation_that_does_not_kill_its_named_test(
    tmp_path: Path,
) -> None:
    """The canary: one mutation, four tests, four distinguishable outcomes.

    All four entries apply the SAME mutation to the SAME file, so the only
    thing that varies is the named test. That is what makes this a check on
    the runner rather than on the subject: an inverted condition would report
    the grep test as killed, a swallowed exit code would report the
    behavioural test as survived, and either would show up here.

    Args:
        tmp_path: pytest scratch directory.
    """

    source = _make_tree(tmp_path / "source")
    manifest = _write_manifest(tmp_path / "manifest.toml", _SELF_MANIFEST)
    pristine = (source / "subject" / "floor.sh").read_bytes()

    result = runner.run_manifest(
        source_root=source,
        manifest_path=manifest,
        scratch_root=tmp_path / "scratch",
        report=False,
    )

    outcomes = {entry.entry_id: entry.outcome for entry in result.results}
    assert outcomes == {
        "behavioural": runner.KILLED,
        "text-only-canary": runner.SURVIVED,
        "already-red-control": runner.CONTROL_FAILED,
        "skipping-control": runner.CONTROL_FAILED,
    }
    assert result.killed == 1
    assert result.entries_total == 4
    assert not result.ok

    by_id = {entry.entry_id: entry for entry in result.results}
    killed = by_id["behavioural"]
    assert killed.control_exit == 0
    assert killed.control_passed == 1
    assert killed.mutant_exit == 1
    assert killed.revert == "exact"

    canary = by_id["text-only-canary"]
    assert canary.control_exit == 0
    # The whole point: the mutant ran, the named test passed anyway, and the
    # runner says so in the direction that failed.
    assert canary.mutant_exit == 0
    assert "mutant_did_not_fail" in canary.detail
    assert canary.revert == "exact"
    line = runner.format_entry(canary)
    assert "entry=text-only-canary" in line
    assert "result=survived" in line
    assert "mutant_did_not_fail" in line

    # A skipped control must not be reported as a pass: the mutant would skip
    # too, exit 0, and look exactly like the canary for an unrelated reason.
    skipping = by_id["skipping-control"]
    assert skipping.mutant_exit is None
    assert "skipped" in skipping.detail

    # The revert must put the exact bytes back, recomputed from disk rather
    # than trusted, or every later entry measures a tree nobody described.
    staged = tmp_path / "scratch" / "tree" / "subject" / "floor.sh"
    assert staged.read_bytes() == pristine
    assert (
        hashlib.sha256(staged.read_bytes()).hexdigest()
        == hashlib.sha256((source / "subject" / "floor.sh").read_bytes()).hexdigest()
    )

    assert "killed=1/4" in runner.format_summary(result)


def test_runner_reports_success_only_when_every_entry_kills(
    tmp_path: Path,
) -> None:
    """The other polarity: a manifest of one killing entry must pass.

    Without this, a runner that reported everything as survived would satisfy
    the canary above while being useless.

    Args:
        tmp_path: pytest scratch directory.
    """

    source = _make_tree(tmp_path / "source")
    manifest = _write_manifest(
        tmp_path / "manifest.toml",
        _SELF_MANIFEST.split('\n[[mutation]]\nid = "text-only-canary"')[0],
    )

    result = runner.run_manifest(
        source_root=source,
        manifest_path=manifest,
        scratch_root=tmp_path / "scratch",
        report=False,
    )

    assert [entry.outcome for entry in result.results] == [runner.KILLED]
    assert result.ok
    assert "killed=1/1" in runner.format_summary(result)


def test_runner_main_exits_non_zero_and_names_the_surviving_entry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A swallowed exit code is the failure mode this test exists for.

    Args:
        tmp_path: pytest scratch directory.
        capsys: pytest stdout/stderr capture.
    """

    source = _make_tree(tmp_path / "source")
    manifest = _write_manifest(tmp_path / "manifest.toml", _SELF_MANIFEST)

    status = runner.main(
        [
            "--source-root",
            str(source),
            "--manifest",
            str(manifest),
            "--scratch-root",
            str(tmp_path / "scratch"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "QFV_MUT_SUMMARY killed=1/4" in captured.out
    assert "entry=text-only-canary result=survived" in captured.out
    assert "entry text-only-canary is survived" in captured.err
    # The scratch tree is removed unless --keep-scratch is passed.
    assert not (tmp_path / "scratch").exists()


def test_runner_reports_a_mutant_that_stopped_the_test_as_an_error(
    tmp_path: Path,
) -> None:
    """A mutant that stops the test from running is not a kill.

    pytest exits 1 when tests ran and failed. A mutation that makes the test
    file uncollectable exits differently, and counting that as a kill would
    claim a failure mechanism nobody measured -- the same class as the two
    module comments in this project that asserted "exits 75" for paths that
    exit 151. The code asserted below is the one measured here, not the one
    the documentation lists.

    Args:
        tmp_path: pytest scratch directory.
    """

    source = _make_tree(tmp_path / "source")
    manifest = _write_manifest(
        tmp_path / "manifest.toml",
        '[[mutation]]\nid = "breaks-collection"\ntarget = "test_floor.py"\n'
        'test = "test_floor.py::test_running_the_floor_refuses_one"\n'
        "find = 'import subprocess'\nreplace = 'import subprocess ('\n"
        'why = "an uncollectable module is not a failing test"\n',
    )

    result = runner.run_manifest(
        source_root=source,
        manifest_path=manifest,
        scratch_root=tmp_path / "scratch",
        report=False,
    )

    entry = result.results[0]
    assert entry.outcome == runner.ERROR, entry
    # Measured: 4, pytest's usage error, because the node id names a module
    # that cannot be imported. Not 2, which is what the exit-code table
    # would have suggested.
    assert entry.mutant_exit == 4, entry.detail
    assert "not 1" in entry.detail, entry.detail
    assert entry.revert == "exact"
    assert not result.ok


def test_runner_refuses_a_scratch_root_inside_the_copied_tree(
    tmp_path: Path,
) -> None:
    """copytree would otherwise recurse into its own destination.

    Args:
        tmp_path: pytest scratch directory.
    """

    source = _make_tree(tmp_path / "source")
    manifest = _write_manifest(tmp_path / "manifest.toml", _SELF_MANIFEST)

    with pytest.raises(ValueError, match="inside the copied tree"):
        runner.run_manifest(
            source_root=source,
            manifest_path=manifest,
            scratch_root=source / "subject" / "scratch",
            report=False,
        )

    # The excluded directories are fine, which is why the default lives there.
    result = runner.run_manifest(
        source_root=source,
        manifest_path=manifest,
        scratch_root=source / "outputs" / "scratch",
        only=("behavioural",),
        report=False,
    )
    assert [entry.outcome for entry in result.results] == [runner.KILLED]


def test_runner_refuses_to_report_success_for_a_subset(tmp_path: Path) -> None:
    """A focused run must never read as a full one.

    Args:
        tmp_path: pytest scratch directory.
    """

    source = _make_tree(tmp_path / "source")
    manifest = _write_manifest(tmp_path / "manifest.toml", _SELF_MANIFEST)

    result = runner.run_manifest(
        source_root=source,
        manifest_path=manifest,
        scratch_root=tmp_path / "scratch",
        only=("behavioural",),
        report=False,
    )

    assert [entry.outcome for entry in result.results] == [runner.KILLED]
    assert result.killed == 1
    assert result.entries_total == 4
    assert not result.ok
    assert "entries_selected=1 entries_total=4" in runner.format_summary(result)


def test_runner_reports_a_drifted_anchor_as_an_error_not_a_kill(
    tmp_path: Path,
) -> None:
    """An anchor that no longer matches must not pass for a mutation.

    A "replace what you find" rule would silently mutate nothing and report
    the entry as survived, or mutate two places and claim one. The occurrence
    count makes both an error.

    Args:
        tmp_path: pytest scratch directory.
    """

    source = _make_tree(tmp_path / "source")
    absent = _write_manifest(
        tmp_path / "absent.toml",
        '[[mutation]]\nid = "absent"\ntarget = "subject/floor.sh"\n'
        'test = "test_floor.py::test_running_the_floor_refuses_one"\n'
        'find = "no such text"\nreplace = "x"\nwhy = "anchor drift"\n',
    )
    doubled = _write_manifest(
        tmp_path / "doubled.toml",
        '[[mutation]]\nid = "doubled"\ntarget = "subject/floor.sh"\n'
        'test = "test_floor.py::test_running_the_floor_refuses_one"\n'
        'find = "set -euo pipefail"\nreplace = "set -u"\n'
        'occurrences = 2\nwhy = "declares two, file has one"\n',
    )
    missing = _write_manifest(
        tmp_path / "missing.toml",
        '[[mutation]]\nid = "missing"\ntarget = "subject/absent.sh"\n'
        'test = "test_floor.py::test_running_the_floor_refuses_one"\n'
        'find = "x"\nreplace = "y"\nwhy = "target is gone"\n',
    )

    for index, (manifest, fragment) in enumerate(
        (
            (absent, "find appears 0 times"),
            (doubled, "find appears 1 times"),
            (missing, "target does not exist"),
        )
    ):
        result = runner.run_manifest(
            source_root=source,
            manifest_path=manifest,
            scratch_root=tmp_path / f"scratch-{index}",
            report=False,
        )
        assert [entry.outcome for entry in result.results] == [runner.ERROR]
        assert fragment in result.results[0].detail
        assert not result.ok


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        pytest.param(
            '[[mutation]]\nid = "a"\ntarget = "t"\ntest = "f.py::t"\n'
            'find = "x"\nreplace = "y"\nwhy = "w"\ntpyo = "1"\n',
            "unknown keys",
            id="unknown_key",
        ),
        pytest.param(
            '[[mutation]]\nid = "a"\ntarget = "t"\ntest = "f.py::t"\n'
            'find = "x"\nreplace = "x"\nwhy = "w"\n',
            "find equals replace",
            id="no_op",
        ),
        pytest.param(
            '[[mutation]]\nid = "a"\ntarget = "t"\ntest = "f.py::t"\n'
            'find = "x"\nreplace = "y"\nwhy = "w"\n'
            '[[mutation]]\nid = "a"\ntarget = "t"\ntest = "f.py::t"\n'
            'find = "p"\nreplace = "q"\nwhy = "w"\n',
            "id must be unique",
            id="duplicate_id",
        ),
        pytest.param(
            '[[mutation]]\nid = "a"\ntarget = "../t"\ntest = "f.py::t"\n'
            'find = "x"\nreplace = "y"\nwhy = "w"\n',
            "relative in-tree path",
            id="escaping_target",
        ),
        pytest.param(
            '[[mutation]]\nid = "a"\ntarget = "t"\ntest = "no-node-id"\n'
            'find = "x"\nreplace = "y"\nwhy = "w"\n',
            "pytest node id",
            id="not_a_node_id",
        ),
        pytest.param(
            '[[mutation]]\nid = "a"\ntarget = "t"\ntest = "f.py::t"\n'
            'find = "x"\nreplace = "y"\nwhy = "  "\n',
            "why must say",
            id="empty_why",
        ),
        pytest.param(
            "mutations = []\n",
            "unknown top-level keys",
            id="wrong_section_name",
        ),
    ],
)
def test_manifest_refuses_a_shape_that_could_disable_an_entry(
    tmp_path: Path, body: str, fragment: str
) -> None:
    """A typo must be an error, not a quietly inert entry.

    Args:
        tmp_path: pytest scratch directory.
        body: Manifest text to reject.
        fragment: Expected substring of the diagnostic.
    """

    manifest = _write_manifest(tmp_path / "manifest.toml", body)

    with pytest.raises(ValueError, match=fragment):
        runner.load_manifest(manifest)


def test_shipped_manifest_parses_and_names_reachable_targets_and_tests() -> None:
    """The shipped manifest must load, and its anchors must still match.

    This is deliberately NOT a substitute for the mutation run: it proves the
    manifest is well formed and that every anchor occurs the declared number
    of times, which is what turns a drifted anchor into an error before the
    expensive stage starts.
    """

    entries = runner.load_manifest(MANIFEST)

    assert len(entries) >= 10, [entry.entry_id for entry in entries]
    for entry in entries:
        target = REPO_ROOT / entry.target
        assert target.is_file(), entry.target
        text = target.read_text()
        assert text.count(entry.find) == entry.occurrences, entry.entry_id
        node_path, _, _ = entry.test.partition("::")
        assert (REPO_ROOT / node_path).is_file(), entry.test

    # Every named test must be collectable, or the stage would report "no
    # tests ran" as a passing control for an entry that checks nothing.
    node_ids = sorted({entry.test for entry in entries})
    collected = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--collect-only",
            "--color=no",
            "-p",
            "no:cacheprovider",
            *node_ids,
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert collected.returncode == 0, collected.stdout
    # Each id must appear in the listing. Not a count: an unparametrized node
    # id selects every one of its cases, so the collected total is >= the
    # number of ids and a count would drift with any new parameter.
    for node_id in node_ids:
        assert node_id in collected.stdout, (node_id, collected.stdout)
