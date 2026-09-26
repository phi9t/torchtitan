# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Behavioral contracts for the opt-in formal toolchain surface."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_WRAPPER = REPO_ROOT / "scripts" / "run_formal_checks.sh"
CHECKER_CONTRACT = (
    REPO_ROOT
    / "experiments"
    / "qwen3_formal_verifier"
    / "formal"
    / "checker_contract.sh"
)
TLC_SMOKE_RUNNER = (
    REPO_ROOT / "experiments" / "qwen3_formal_verifier" / "formal" / "run_tlc_smoke.sh"
)
BAZELRC = REPO_ROOT / ".bazelrc"
MODULE_LOCK = REPO_ROOT / "MODULE.bazel.lock"


def _run_classifier(function: str, status: int, output: str, expected: str = ""):
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; "$2" "$3" "$4" "$5"',
            "classifier-test",
            str(CHECKER_CONTRACT),
            function,
            str(status),
            output,
            expected,
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_tlc_negative_accepts_only_the_named_invariant_violation() -> None:
    # A real TLC run always prints the search summary; the classifier requires
    # it so a crashed or truncated run cannot be read as a clean negative.
    # Shape taken from a real TLC run: an initial-state violation aborts before
    # any search summary is printed, so the completion signal is "Finished in".
    exact_violation = (
        "Error: Invariant SmokeInvariant is violated by the initial state:\n"
        'phase = "broken"\n'
        "Finished in 00s at (2026-09-25 09:29:29)\n"
    )

    accepted = _run_classifier(
        "formal_classify_tlc_negative", 12, exact_violation, "SmokeInvariant"
    )
    wrong_invariant = _run_classifier(
        "formal_classify_tlc_negative", 12, exact_violation, "DifferentInvariant"
    )
    parse_error = _run_classifier(
        "formal_classify_tlc_negative",
        12,
        "Parse Error\nError: Invariant SmokeInvariant is violated.\n",
        "SmokeInvariant",
    )
    additional_checker_error = _run_classifier(
        "formal_classify_tlc_negative",
        12,
        exact_violation + "Error: unrelated checker failure\n",
        "SmokeInvariant",
    )

    assert accepted.returncode == 0, accepted.stdout
    assert wrong_invariant.returncode != 0
    assert parse_error.returncode != 0
    assert additional_checker_error.returncode != 0


def test_tlc_outcomes_reject_timeout_and_missing_checker() -> None:
    for status in (124, 127):
        valid = _run_classifier(
            "formal_classify_tlc_valid",
            status,
            "Model checking completed. No error has been found.\n",
        )
        negative = _run_classifier(
            "formal_classify_tlc_negative",
            status,
            "Error: Invariant SmokeInvariant is violated by the initial state:\n",
            "SmokeInvariant",
        )

        assert valid.returncode != 0
        assert negative.returncode != 0


def test_tlc_smoke_keeps_generated_trace_products_out_of_runfiles(
    tmp_path: Path,
) -> None:
    runfiles = tmp_path / "runfiles"
    fixture_dir = (
        runfiles / "_main" / "experiments" / "qwen3_formal_verifier" / "formal"
    )
    fixture_dir.mkdir(parents=True)
    for filename in (
        "checker_contract.sh",
        "TlcSmokeGood.tla",
        "TlcSmokeGood.cfg",
        "TlcSmokeBad.tla",
        "TlcSmokeBad.cfg",
    ):
        (fixture_dir / filename).write_text(
            (TLC_SMOKE_RUNNER.parent / filename).read_text()
        )

    java_bin = tmp_path / "fake-java"
    java_bin.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "${1:-}" == "-version" ]]; then\n'
        "  printf 'openjdk version \"17.0.0\"\\n' >&2\n"
        "  exit 0\n"
        "fi\n"
        'printf "%s\\n" "$PWD" >> "${FAKE_JAVA_WORKDIRS}"\n'
        'printf "%s\\n" "$@" >> "${FAKE_JAVA_ARGUMENTS}"\n'
        "for ((index = 1; index <= $#; index++)); do\n"
        '  if [[ "${!index}" == "-metadir" ]]; then\n'
        "    next_index=$((index + 1))\n"
        '    mkdir -p "${!next_index}"\n'
        '    : > "${!next_index}/fake-state"\n'
        "  fi\n"
        "done\n"
        'for arg in "$@"; do\n'
        '  if [[ "${arg}" == TlcSmokeBad.tla || "${arg}" == */TlcSmokeBad.tla ]]; then\n'
        "    : > TlcSmokeBad_TTrace_fake.bin\n"
        "    printf 'Error: Invariant SmokeInvariant is violated by the initial state:\\n'\n"
        "    printf 'Finished in 00s at (2026-01-01 00:00:00)\\n'\n"
        "    exit 12\n"
        "  fi\n"
        "done\n"
        "printf 'Model checking completed. No error has been found.\\n'\n"
        "printf '12 states generated, 11 distinct states found, "
        "0 states left on queue.\\n'\n"
    )
    java_bin.chmod(0o755)
    tla_jar = tmp_path / "tla2tools.jar"
    tla_jar.touch()
    workdirs = tmp_path / "java-workdirs.txt"
    arguments = tmp_path / "java-arguments.txt"
    test_tmpdir = tmp_path / "test-tmpdir"
    test_tmpdir.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "FAKE_JAVA_WORKDIRS": str(workdirs),
            "FAKE_JAVA_ARGUMENTS": str(arguments),
            "TEST_SRCDIR": str(runfiles),
            "TEST_TMPDIR": str(test_tmpdir),
            "TEST_WORKSPACE": "_main",
        }
    )
    result = subprocess.run(
        ["bash", str(TLC_SMOKE_RUNNER), str(java_bin), str(tla_jar)],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode == 0, result.stdout
    assert not list(fixture_dir.glob("TlcSmoke*_TTrace_*.bin"))
    assert list(test_tmpdir.rglob("TlcSmokeBad_TTrace_fake.bin"))
    assert list(test_tmpdir.rglob("fake-state"))
    assert all(
        Path(workdir).is_relative_to(test_tmpdir)
        for workdir in workdirs.read_text().splitlines()
    )
    arguments_seen = arguments.read_text().splitlines()
    for filename in (
        "TlcSmokeGood.tla",
        "TlcSmokeGood.cfg",
        "TlcSmokeBad.tla",
        "TlcSmokeBad.cfg",
    ):
        assert str(fixture_dir / filename) in arguments_seen


def test_lean_valid_requires_kernel_success_without_nonstandard_axioms() -> None:
    clean = _run_classifier(
        "formal_classify_lean_valid",
        0,
        "'FormalSmoke.validStepReady' does not depend on any axioms\n",
        "FormalSmoke.validStepReady",
    )
    sorry = _run_classifier(
        "formal_classify_lean_valid",
        0,
        "'FormalSmoke.validStepReady' depends on axioms: [sorryAx]\n",
        "FormalSmoke.validStepReady",
    )
    project_axiom = _run_classifier(
        "formal_classify_lean_valid",
        0,
        "'FormalSmoke.validStepReady' depends on axioms: "
        "[FormalSmoke.uncheckedStep]\n",
        "FormalSmoke.validStepReady",
    )

    assert clean.returncode == 0, clean.stdout
    assert sorry.returncode != 0
    assert project_axiom.returncode != 0


def test_lean_negative_accepts_only_the_named_false_proposition() -> None:
    exact_rejection = (
        "error: Tactic `decide` proved that the proposition\n"
        "  FormalSmoke.ControlledInvalidProposition\n"
        "is false\n"
    )

    accepted = _run_classifier(
        "formal_classify_lean_negative",
        1,
        exact_rejection,
        "FormalSmoke.ControlledInvalidProposition",
    )
    wrong_proposition = _run_classifier(
        "formal_classify_lean_negative",
        1,
        exact_rejection,
        "FormalSmoke.DifferentProposition",
    )
    syntax_error = _run_classifier(
        "formal_classify_lean_negative",
        1,
        "unexpected token\n" + exact_rejection,
        "FormalSmoke.ControlledInvalidProposition",
    )
    additional_compiler_error = _run_classifier(
        "formal_classify_lean_negative",
        1,
        exact_rejection + "error: unrelated compiler failure\n",
        "FormalSmoke.ControlledInvalidProposition",
    )
    cross_associated_proposition = _run_classifier(
        "formal_classify_lean_negative",
        1,
        "error: Tactic `decide` proved that the proposition\n"
        "  FormalSmoke.DifferentProposition\n"
        "is false\n"
        "note: FormalSmoke.ControlledInvalidProposition\n",
        "FormalSmoke.ControlledInvalidProposition",
    )

    assert accepted.returncode == 0, accepted.stdout
    assert wrong_proposition.returncode != 0
    assert syntax_error.returncode != 0
    assert additional_compiler_error.returncode != 0
    assert cross_associated_proposition.returncode != 0


def test_lean_outcomes_reject_timeout_and_missing_toolchain() -> None:
    output = (
        "error: Tactic `decide` proved that the proposition\n"
        "  FormalSmoke.ControlledInvalidProposition\n"
        "is false\n"
    )
    for status in (124, 127):
        result = _run_classifier(
            "formal_classify_lean_negative",
            status,
            output,
            "FormalSmoke.ControlledInvalidProposition",
        )

        assert result.returncode != 0


def test_formal_wrapper_reenters_rootfs_with_requested_network_policy(
    tmp_path: Path,
) -> None:
    fake_entrypoint = tmp_path / "enter_rootfs.sh"
    invocation = tmp_path / "invocation.txt"
    fake_entrypoint.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "${TORCHTITAN_ROOTFS_NETWORK:-}" "$@" > "${FORMAL_TEST_LOG}"\n'
    )
    fake_entrypoint.chmod(0o755)
    cache = tmp_path / "formal-cache"
    rootfs = tmp_path / "rootfs"
    env = {
        "FORMAL_TEST_LOG": str(invocation),
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "TORCHTITAN_FORMAL_CACHE_HOST": str(cache),
        "TORCHTITAN_FORMAL_ROOTFS": str(rootfs),
        "TORCHTITAN_ROOTFS_ENTRYPOINT": str(fake_entrypoint),
    }

    result = subprocess.run(
        ["bash", str(FORMAL_WRAPPER), "--no-fetch"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode == 0, result.stdout
    assert invocation.read_text().splitlines() == [
        "offline",
        "--rootfs",
        str(rootfs),
        "--",
        "scripts/run_formal_checks.sh",
        "--no-fetch",
    ]
    assert cache.is_dir()


def test_formal_wrapper_fails_closed_without_rootfs_cache_contract(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_IN_ROOTFS": "1",
        }
    )
    env.pop("TORCHTITAN_FORMAL_CACHE", None)

    result = subprocess.run(
        ["bash", str(FORMAL_WRAPPER), "--no-fetch"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode != 0
    assert "formal cache" in result.stdout.lower()


def test_formal_wrapper_forces_checker_execution_on_no_fetch(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "formal-cache"
    registry = (
        cache
        / "bazel"
        / "vendor"
        / "_registries"
        / "bcr.bazel.build"
        / "bazel_registry.json"
    )
    registry.parent.mkdir(parents=True)
    registry.write_text("{}\n")
    bazel = cache / "tools" / "bazel" / "9.2.0" / "bazel"
    bazel.parent.mkdir(parents=True)
    invocation = tmp_path / "bazel-invocation.txt"
    bazel.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$@" > "${FORMAL_TEST_LOG}"\n'
    )
    bazel.chmod(0o755)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sha256sum = fake_bin / "sha256sum"
    fake_sha256sum.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_sha256sum.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "FORMAL_TEST_LOG": str(invocation),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "TORCHTITAN_FORMAL_CACHE": str(cache),
            "TORCHTITAN_FORMAL_REENTRY_NETWORK": "offline",
            "TORCHTITAN_IN_ROOTFS": "1",
            "TORCHTITAN_ROOTFS_NETWORK": "offline",
        }
    )

    result = subprocess.run(
        ["bash", str(FORMAL_WRAPPER), "--no-fetch"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode == 0, result.stdout
    args = invocation.read_text().splitlines()
    assert args[0].startswith("--output_user_root=")
    assert "test" in args
    assert "--repository_disable_download" in args
    assert "--cache_test_results=no" in args
    assert args[-1] == ("//experiments/qwen3_formal_verifier/formal:formal_smoke_tests")


def test_formal_wrapper_no_fetch_requires_materialized_vendor_registry(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "formal-cache"
    bazel = cache / "tools" / "bazel" / "9.2.0" / "bazel"
    bazel.parent.mkdir(parents=True)
    invocation = tmp_path / "bazel-invocation.txt"
    bazel.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$@" > "${FORMAL_TEST_LOG}"\n'
    )
    bazel.chmod(0o755)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sha256sum = fake_bin / "sha256sum"
    fake_sha256sum.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_sha256sum.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "FORMAL_TEST_LOG": str(invocation),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "TORCHTITAN_FORMAL_CACHE": str(cache),
            "TORCHTITAN_FORMAL_REENTRY_NETWORK": "offline",
            "TORCHTITAN_IN_ROOTFS": "1",
            "TORCHTITAN_ROOTFS_NETWORK": "offline",
        }
    )

    result = subprocess.run(
        ["bash", str(FORMAL_WRAPPER), "--no-fetch"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode != 0
    assert "run --networked first" in result.stdout
    assert not invocation.exists()


def test_formal_wrapper_passes_external_cache_to_every_networked_bazel_call(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "formal-cache"
    bazel = cache / "tools" / "bazel" / "9.2.0" / "bazel"
    bazel.parent.mkdir(parents=True)
    invocation = tmp_path / "bazel-invocations.txt"
    bazel.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "{ printf 'CALL\\n'; printf '%s\\n' \"$@\"; } >> \"${FORMAL_TEST_LOG}\"\n"
        "is_vendor=0\n"
        "vendor_dir=\n"
        'for arg in "$@"; do\n'
        '  [[ "${arg}" == vendor ]] && is_vendor=1\n'
        '  case "${arg}" in --vendor_dir=*) vendor_dir=${arg#*=} ;; esac\n'
        "done\n"
        "if [[ ${is_vendor} -eq 1 ]]; then\n"
        '  mkdir -p "${vendor_dir}/_registries/bcr.bazel.build"\n'
        '  : > "${vendor_dir}/_registries/bcr.bazel.build/bazel_registry.json"\n'
        "fi\n"
    )
    bazel.chmod(0o755)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sha256sum = fake_bin / "sha256sum"
    fake_sha256sum.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_sha256sum.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "FORMAL_TEST_LOG": str(invocation),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "TORCHTITAN_FORMAL_CACHE": str(cache),
            "TORCHTITAN_FORMAL_REENTRY_NETWORK": "networked",
            "TORCHTITAN_IN_ROOTFS": "1",
            "TORCHTITAN_ROOTFS_NETWORK": "networked",
        }
    )

    result = subprocess.run(
        ["bash", str(FORMAL_WRAPPER), "--networked"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode == 0, result.stdout
    calls = [
        call.splitlines() for call in invocation.read_text().split("CALL\n") if call
    ]
    assert len(calls) == 3
    expected_flags = {
        f"--repository_cache={cache}/bazel/repository-cache",
        f"--symlink_prefix={cache}/bazel/workspace-links/",
        f"--vendor_dir={cache}/bazel/vendor",
        "--config=formal",
    }
    for call in calls:
        assert expected_flags <= set(call)


def test_bazelrc_scopes_insula_only_paths_to_formal_config() -> None:
    for line in BAZELRC.read_text().splitlines():
        if "/project/formal-cache" not in line:
            continue
        scope = line.split(maxsplit=1)[0]
        assert scope.endswith(":formal"), line


def test_bazel_module_lock_is_enforced_and_portable() -> None:
    bazelrc = BAZELRC.read_text()
    assert "common --lockfile_mode=error" in bazelrc
    assert "--lockfile_mode=off" not in bazelrc

    lock = json.loads(MODULE_LOCK.read_text())
    serialized = json.dumps(lock, sort_keys=True)
    assert lock["lockFileVersion"] >= 1
    assert "rules_shell" in serialized
    assert "/project/formal-cache" not in serialized
    assert "/data02/" not in serialized
    assert "file:///" not in serialized


def test_bazel_module_lock_passes_repository_pii_guard() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_no_pii.py", "MODULE.bazel.lock"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode == 0, result.stdout


def test_formal_wrapper_rejects_no_fetch_from_existing_networked_rootfs(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "formal-cache"
    cache.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_FORMAL_CACHE": str(cache),
            "TORCHTITAN_IN_ROOTFS": "1",
            "TORCHTITAN_ROOTFS_NETWORK": "networked",
        }
    )
    env.pop("TORCHTITAN_FORMAL_REENTRY_NETWORK", None)

    result = subprocess.run(
        ["bash", str(FORMAL_WRAPPER), "--no-fetch"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode != 0
    assert "outside insula" in result.stdout.lower()


def test_formal_wrapper_rejects_inside_checkout_cache_in_existing_rootfs() -> None:
    env = os.environ.copy()
    env.update(
        {
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_FORMAL_CACHE": str(REPO_ROOT),
            "TORCHTITAN_FORMAL_REENTRY_NETWORK": "networked",
            "TORCHTITAN_IN_ROOTFS": "1",
            "TORCHTITAN_ROOTFS_NETWORK": "networked",
        }
    )

    result = subprocess.run(
        ["bash", str(FORMAL_WRAPPER), "--networked"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode != 0
    assert "outside the checkout" in result.stdout.lower()


@pytest.mark.parametrize("managed_dir", ("bazel", "tools", "locks"))
def test_formal_wrapper_rejects_managed_cache_symlinks(
    tmp_path: Path,
    managed_dir: str,
) -> None:
    cache = tmp_path / "formal-cache"
    cache.mkdir()
    (cache / managed_dir).symlink_to(REPO_ROOT, target_is_directory=True)
    env = os.environ.copy()
    env.update(
        {
            "PATH": "/usr/bin:/bin",
            "TORCHTITAN_FORMAL_CACHE": str(cache),
            "TORCHTITAN_FORMAL_REENTRY_NETWORK": "networked",
            "TORCHTITAN_IN_ROOTFS": "1",
            "TORCHTITAN_ROOTFS_NETWORK": "networked",
        }
    )

    result = subprocess.run(
        ["bash", str(FORMAL_WRAPPER), "--networked"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode != 0
    assert "symlink" in result.stdout.lower()


FORMAL_DIR = REPO_ROOT / "experiments" / "qwen3_formal_verifier" / "formal"
MODEL_RUNNER = FORMAL_DIR / "run_tlc_scout_a_model.sh"
BUILD_FILE = FORMAL_DIR / "BUILD.bazel"


def test_abstract_model_is_a_transition_system_not_a_cursor_walk() -> None:
    """The model must constrain behaviour, not replay one recorded sequence.

    ScoutAValid walks a cursor along a constant and evaluates predicates over
    it, so it explores one path. The point of ScoutAModel is that several
    actions are concurrently enabled and TLC explores every admitted
    interleaving, so an invariant that holds there holds for executions nobody
    observed.
    """

    model = (FORMAL_DIR / "ScoutAModel.tla").read_text()

    # Guarded actions over real state, not an index into a generated sequence.
    assert "VARIABLES" in model
    for action in (
        "StepStart",
        "ForwardComplete",
        "GradientReady",
        "GradientMissing",
        "BackwardComplete",
        "OptimizerMutate",
    ):
        assert f"{action} ==" in model, action
    # The model must not depend on the generated facts; that would make it a
    # replay of the observed trace again.
    assert "ScoutAFacts" not in model


def test_gradient_readiness_and_backward_completion_may_interleave() -> None:
    """Both orders are real executions, so the model must admit both."""

    model = (FORMAL_DIR / "ScoutAModel.tla").read_text()
    gradient_ready = model.split("GradientReady ==")[1].split("\n\n")[0]

    # Enabled while backward is started OR already completed -- that disjunction
    # is what creates the interleaving the safety argument has to survive.
    assert 'bwd \\in {"started", "completed"}' in gradient_ready


def test_load_bearing_guard_is_isolated_as_a_constant() -> None:
    """The negative model must relax exactly one guard and nothing else."""

    model = (FORMAL_DIR / "ScoutAModel.tla").read_text()
    assert "CONSTANT RequireReadyGradients" in model
    assert '(RequireReadyGradients => grad = "ready")' in model

    valid = (FORMAL_DIR / "ScoutAModel.cfg").read_text().splitlines()
    unsafe = (FORMAL_DIR / "ScoutAModelUnsafe.cfg").read_text().splitlines()

    # Computed, not asserted: the two configurations must differ in exactly one
    # line, and that line must be the constant. If the negative also dropped
    # invariants it would be checking less, not relaxing one guard, and any
    # violation it found would not be attributable to the guard.
    differing = [
        (left, right)
        for left, right in zip(valid, unsafe, strict=True)
        if left != right
    ]
    assert len(differing) == 1, differing
    assert differing[0] == (
        "CONSTANT RequireReadyGradients = TRUE",
        "CONSTANT RequireReadyGradients = FALSE",
    )


def test_refinement_bridge_constrains_the_model_to_the_observed_trace() -> None:
    refine = (FORMAL_DIR / "ScoutARefine.tla").read_text()

    assert "EXTENDS Naturals, Sequences, ScoutAModel, ScoutAFacts" in refine
    assert "Observed == EventKinds" in refine
    assert "IsPrefixOfObserved" in refine
    assert "ConstrainedNext ==" in refine
    # Reachability is proved by refutation, so the witness is a violation.
    assert "ObservedTraceIsNotAdmitted" in refine


def test_refinement_negative_is_rejected_for_the_right_reason() -> None:
    """Rejection alone is weak evidence; it must hit the guard under test.

    An earlier version swapped `gradient.ready` with `optimizer.mutated`, which
    put the mutation at a position where the optimizer had not started. The
    model refused it, but for an out-of-order lifecycle rather than unready
    gradients -- a different guard than the one being tested. The control now
    substitutes `gradient.missing`, so every earlier event stays admissible and
    the trace dies exactly at OptimizerMutate.
    """

    bad = (FORMAL_DIR / "ScoutARefineBad.tla").read_text()

    # Derived from the same facts the positive bridge uses, not hand-written.
    assert "ScoutAFacts" in bad
    assert "IndexOf(EventKinds" in bad
    assert '"step.started"' not in bad

    # One substitution, at the gradient event, mutation still after it.
    assert "CorruptionIsIsolated" in bad
    assert '"gradient.missing"' in bad
    assert "= {ReadyIndex}" in bad, "the corruption must be a single position"

    # Both halves must be checked, not merely defined.
    reject = (FORMAL_DIR / "ScoutARefineBad.cfg").read_text()
    reach = (FORMAL_DIR / "ScoutARefineBadReach.cfg").read_text()
    assert "CorruptionIsIsolated" in reject
    assert "CorruptionIsIsolated" in reach
    assert "ObservedTraceIsNotAdmitted" in reject
    assert "RejectionHappensBeforeTheMutation" in reach

    # The runner must require both outcomes, so a trace refused too early fails
    # the gate instead of counting as a successful negative control.
    runner = MODEL_RUNNER.read_text()
    assert "RejectionHappensBeforeTheMutation" in runner
    assert "wrong guard" in runner
    assert "rejected_at_mutation_guard" in runner


def test_refinement_configs_disable_deadlock_so_polarity_is_unambiguous() -> None:
    """Rejection must read as a clean completion, not as a checker error.

    Without this, a refused trace surfaces as TLC exit 11 (deadlock), which is
    indistinguishable from a genuine specification defect.
    """

    for name in ("ScoutARefine.cfg", "ScoutARefineBad.cfg"):
        assert "CHECK_DEADLOCK FALSE" in (FORMAL_DIR / name).read_text(), name


def test_model_runner_refuses_a_non_branching_specification() -> None:
    """Branching, not state count, separates a model from a replay.

    A cursor walking a recorded sequence has outdegree 1 everywhere however
    long the sequence is, so a threshold on state count would accept the very
    design this module replaces -- ScoutAValid reaches 12 states.
    """

    runner = MODEL_RUNNER.read_text()
    assert "max_outdegree" in runner
    assert "-ge 2" in runner, "the floor must be on branching"
    assert "replay, not a model" in runner
    # And the count must still be reported, so the evidence is inspectable.
    assert "distinct_states=%s max_outdegree=%s" in runner


def test_model_runner_does_not_clobber_errexit_around_expected_failures() -> None:
    """Regression: a stray `set -e` made the runner exit silently on exit 12.

    The expected TLC statuses here are non-zero, so a runner that re-enables
    errexit inside its helper dies with no diagnostic -- a gate step vanishing
    rather than reporting.
    """

    runner = MODEL_RUNNER.read_text()
    helper = runner.split("run_tlc() {")[1].split("\n}")[0]
    assert "set -e" not in helper
    assert "|| status=$?" in helper


def test_both_model_halves_are_wired_into_both_formal_suites() -> None:
    """The abstract/refinement split must not drop a check from the gate.

    The model checks are one target per input set: the abstract half reads only
    hand-written sources so it can run in tier 0, and the refinement half reads
    the generated facts. Both must stay in both sealed suites.
    """

    build = BUILD_FILE.read_text()
    _, sh_tests, suites = _parse_build_targets(build)

    halves = ("tlc_scout_a_model_abstract_test", "tlc_scout_a_model_refine_test")
    for half in halves:
        assert half in sh_tests, half
    for suite in ("scout_a_formal_tests", "scout_b_formal_tests"):
        members = {label.lstrip(":") for label in suites[suite]}
        for half in halves:
            assert half in members, (suite, half)


def test_tier0_lint_classifies_every_pre_commit_hook() -> None:
    """Tier-0 lint runs a subset, so every hook must be a deliberate choice.

    Tier 0 names the hooks it runs. A hook added to .pre-commit-config.yaml and
    not named here would be skipped by tier 0 forever with nothing noticing, so
    the union of the hooks it runs and the ones it deliberately defers to the
    gate has to account for the whole config.
    """

    config = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
    declared = set(re.findall(r"^\s*-\s*id:\s*(\S+)", config, flags=re.MULTILINE))

    runner = TIER0_RUNNER.read_text()
    loop = runner.split("for hook in \\\n")[1].split("do\n")[0]
    run_here = set(loop.replace("\\", " ").split())

    # Deferred to the gate: slow or network-bound, or a commit-time concern.
    # pyrefly-check is not skipped -- tier 0 invokes pyrefly directly on the
    # changed Python files instead of through pre-commit.
    deferred = {
        "no-commit-to-branch",
        "check-added-large-files",
        "lychee-link-checker",
        "pyrefly-check",
    }

    assert run_here <= declared, run_here - declared
    assert run_here | deferred == declared, declared - (run_here | deferred)
    assert "pyrefly" in runner


def test_every_suite_maps_to_a_declared_bazel_suite() -> None:
    """The wrapper's suite switch is the only thing selecting the gate's checks.

    Nothing validates the sealed transcript's result tokens, so if this switch
    resolved --suite scout-b to the trace-free tier-0 suite the gate would stop
    checking the facts and the refinement bridge without failing. Tier 0 exists
    to be weaker, which is exactly why the mapping needs pinning.
    """

    wrapper = FORMAL_WRAPPER.read_text()
    _, _, suites = _parse_build_targets(BUILD_FILE.read_text())

    mapping = dict(
        re.findall(
            r"^\s*([\w-]+)\)\s*formal_target=" r'"\$\{formal_package\}:(\w+)" ;;',
            wrapper,
            flags=re.MULTILINE,
        )
    )
    default_match = re.search(
        r'^formal_target="\$\{formal_package\}:(\w+)"$', wrapper, flags=re.MULTILINE
    )
    assert default_match is not None
    mapping["smoke"] = default_match.group(1)

    assert mapping == {
        "smoke": "formal_smoke_tests",
        "tier0": "tier0_formal_tests",
        "scout-a": "scout_a_formal_tests",
        "scout-b": "scout_b_formal_tests",
    }, mapping
    for suite in mapping.values():
        assert suite in suites, suite


def test_each_model_half_is_told_to_run_its_own_half() -> None:
    """Input coverage does not imply the check ran.

    Both halves invoke one runner and are distinguished only by a positional
    argument, so a target can carry the refinement inputs while being told to
    run the abstract half. That would silently stop the sealed gate checking
    that the observed trace is admitted, and an inputs-only test cannot see it.
    """

    args = _parse_sh_test_args(BUILD_FILE.read_text())

    for half in ("abstract", "refine"):
        target = f"tlc_scout_a_model_{half}_test"
        assert target in args, target
        assert args[target][-1] == half, (target, args[target])


def test_generated_checker_products_are_refused_from_source_identity(
    tmp_path: Path,
) -> None:
    """TLC writes state products beside the module unless given a metadir.

    An ad-hoc checker run inside the checkout therefore leaves artifacts that a
    later seal would attest to as source. This happened: four `.st`/`.fp` files
    under `formal/states/` were sealed as verified source before the guard
    existed. Failing closed beats gitignoring them, which would let the
    pollution persist invisibly.
    """

    from torchtitan.experiments.qwen3_formal_verifier.scout_a import (
        build_source_manifest,
    )

    states = tmp_path / "experiments" / "qwen3_formal_verifier" / "formal" / "states"
    states.mkdir(parents=True)
    (states / "TlcSmokeBad.st").write_text("state product\n")

    status = b"?? experiments/qwen3_formal_verifier/formal/states/TlcSmokeBad.st\0"
    with pytest.raises(ValueError, match="generated model-checker products"):
        build_source_manifest(tmp_path, "b" * 40, status)


@pytest.mark.parametrize(
    "path_text",
    [
        pytest.param("statements.py", id="name_contains_states"),
        pytest.param("build/tool.py", id="dir_named_build_outside_formal"),
        pytest.param("src/states/model.py", id="dir_named_states_outside_formal"),
        pytest.param("docs/MC_notes.md", id="mc_prefix_outside_formal"),
        pytest.param(
            "experiments/qwen3_formal_verifier/formal/ScoutAModel.tla",
            id="real_formal_source",
        ),
    ],
)
def test_generated_checker_product_guard_is_not_overbroad(
    tmp_path: Path,
    path_text: str,
) -> None:
    """Legitimate source must still be verified.

    A fail-closed rule that rejects real files would be a worse defect than the
    pollution it prevents, so the rule is scoped to the formal directory and
    matches artifact shapes rather than bare directory names.
    """

    from torchtitan.experiments.qwen3_formal_verifier.scout_a import (
        build_source_manifest,
    )

    target = tmp_path / path_text
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("value = 1\n")
    manifest = build_source_manifest(tmp_path, "b" * 40, f"?? {path_text}\0".encode())

    entries = cast(list[dict[str, Any]], manifest["entries"])
    assert [entry["path"] for entry in entries] == [path_text]


@pytest.mark.parametrize(
    "path_text",
    [
        pytest.param("experiments/qwen3_formal_verifier/formal/states/x/A.st", id="st"),
        pytest.param(
            "experiments/qwen3_formal_verifier/formal/states/x/A_0.fp", id="fp"
        ),
        pytest.param(
            "experiments/qwen3_formal_verifier/formal/TlcSmokeBad_TTrace_9.bin",
            id="ttrace_embedded_not_prefixed",
        ),
        pytest.param(
            "experiments/qwen3_formal_verifier/formal/.lake/build/x.olean",
            id="lean_build_product",
        ),
    ],
)
def test_generated_checker_products_are_refused(
    tmp_path: Path,
    path_text: str,
) -> None:
    from torchtitan.experiments.qwen3_formal_verifier.scout_a import (
        build_source_manifest,
    )

    target = tmp_path / path_text
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("generated\n")

    with pytest.raises(ValueError, match="generated model-checker products"):
        build_source_manifest(tmp_path, "b" * 40, f"?? {path_text}\0".encode())


def test_checker_contract_rejects_a_crash_after_the_expected_diagnostic() -> None:
    """A checker that printed the right line and then died is not a result."""

    crashed = (
        "Error: Invariant Foo is violated.\n"
        "Error: The behavior up to this point is:\n"
        "12 states generated, 11 distinct states found, 0 states left on queue.\n"
        "java.lang.OutOfMemoryError: Java heap space\n"
        "\tat tla2sany.semantic.Generator.generate(Generator.java:1)\n"
    )
    result = _run_classifier(
        "formal_classify_tlc_transition_negative", 12, crashed, "Foo"
    )
    assert result.returncode != 0, "a post-diagnostic crash must not classify"


def test_checker_contract_requires_a_completed_search() -> None:
    """Truncated output must not read as a completed model check."""

    truncated = "Model checking completed. No error has been found.\n"
    assert _run_classifier("formal_classify_tlc_valid", 0, truncated).returncode != 0

    complete = (
        "Model checking completed. No error has been found.\n"
        "25 states generated, 25 distinct states found, 0 states left on queue.\n"
    )
    assert _run_classifier("formal_classify_tlc_valid", 0, complete).returncode == 0


def test_initial_state_negative_rejects_a_post_diagnostic_crash() -> None:
    """The initial-state classifier needs the same crash resistance."""

    crashed = (
        "Error: Invariant SmokeInvariant is violated by the initial state:\n"
        "Finished in 00s at (2026-09-25 09:29:29)\n"
        "\tat java.base/java.util.HashMap.get(HashMap.java:1)\n"
    )
    assert (
        _run_classifier(
            "formal_classify_tlc_negative", 12, crashed, "SmokeInvariant"
        ).returncode
        != 0
    )


def test_runners_reject_an_empty_stage_log() -> None:
    """A stage that exits zero but captures nothing is not evidence.

    Sealing checks only that each log is a read-only regular file, so without
    this an empty sink would satisfy the bundle while proving no check ran.
    """

    for name in ("run_scout_a.sh", "run_scout_b.sh"):
        runner = (
            REPO_ROOT / "experiments" / "qwen3_formal_verifier" / name
        ).read_text()
        assert 'if [[ ! -s "${log}" ]]; then' in runner, name
        assert "produced an empty log" in runner, name


def test_runners_syntax_check_every_shell_file_not_just_the_first() -> None:
    """`bash -n a.sh b.sh` parses only a.sh; the rest become arguments.

    Verified directly: bash exits 0 for a good first file even when a later one
    has a syntax error, so the previous single call let shell defects seal as
    lint success.
    """

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.sh"
        bad = Path(tmp) / "bad.sh"
        good.write_text("echo ok\n")
        bad.write_text("if [ 1 ; then\n")

        both = subprocess.run(["bash", "-n", str(good), str(bad)], capture_output=True)
        only_bad = subprocess.run(["bash", "-n", str(bad)], capture_output=True)

    assert both.returncode == 0, "bash -n silently ignores files after the first"
    assert only_bad.returncode != 0, "the second file really is malformed"

    for name in ("run_scout_a.sh", "run_scout_b.sh"):
        runner = (
            REPO_ROOT / "experiments" / "qwen3_formal_verifier" / name
        ).read_text()
        assert 'bash -n "${shell_file}"' in runner, name
        assert 'bash -n "${shell_files[@]}"' not in runner, name


B_MODEL = FORMAL_DIR / "ScoutBModel.tla"
B_MODEL_RUNNER = FORMAL_DIR / "run_tlc_scout_b_model.sh"


def test_dpxtp_model_constrains_operation_not_only_communicator_class() -> None:
    """Regression: constraining only the class admits a program that cannot run.

    An earlier version required ranks to agree on the communicator CLASS at each
    position but not the operation. TLC immediately found a counterexample in
    which two members of one communicator issued all_gather and reduce_scatter
    at the same position, so that communicator could never start and the run
    hung. The model was wrong, not the protocol: NCCL requires members to agree
    on the operation, and a single program cannot diverge that way.
    """

    model = B_MODEL.read_text()
    schedule = model.split("UniformProgramScheduleOK(candidate) ==")[1]
    schedule = schedule.split("\n\n")[0]

    assert "CommClass" in schedule, "class agreement must still be required"
    assert ".op = " in schedule, "operation agreement must also be required"


def test_dpxtp_model_carries_streams_because_they_create_the_wait_edge() -> None:
    """Without streams this model has no deadlock at all.

    Every rank would eventually issue everything and every rendezvous would
    complete. The wait edge exists only because a CUDA stream executes in issue
    order, so a collective cannot start until earlier work on its stream is
    done. Removing AtStreamHead would silently make DeadlockFreedom vacuous.
    """

    model = B_MODEL.read_text()

    assert "StreamOfIssue" in model
    assert "AtStreamHead" in model
    # FSDP's two directional collectives ride dedicated streams; TP rides compute.
    assert '"rs"' in model and '"ag"' in model and '"compute"' in model
    # The stream-head condition must gate the rendezvous, not merely exist.
    ready = model.split("MemberReady(c, r) ==")[1].split("\n\n")[0]
    assert "AtStreamHead" in ready


def test_dpxtp_model_separates_communicators_that_share_a_rank_set() -> None:
    """mesh_batch, mesh_fsdp and mesh_loss_mesh all span {0,2} but are distinct.

    Treating them as one communicator would erase the cross-communicator wait
    cycle, which is the hazard the model exists to expose.
    """

    model = B_MODEL.read_text()
    wide = model.split("CommMembersWide ==")[1].split("\n\n")[0]

    # Two distinct communicators over the same rank set in the wide instance.
    assert wide.count("{0, 2}") >= 2, wide


def test_dpxtp_model_runner_requires_non_vacuity_and_branching() -> None:
    """Safety invariants are worthless if no state satisfies the guards."""

    runner = B_MODEL_RUNNER.read_text()

    assert "SCOUT_B_MODEL_NONVACUOUS" in runner
    assert "ModelNeverCompletes" in runner
    assert "vacuous" in runner
    assert "max_outdegree" in runner
    assert "-ge 2" in runner
    assert "replay, not a model" in runner


def test_dpxtp_negatives_are_distinct_results_not_one() -> None:
    """The two negatives say different things and must both be reported.

    Relaxing the program-schedule requirement deadlocks; additionally relaxing
    NCCL's matching guard instead runs a mismatched rendezvous. Reporting only
    one would lose the point that order agreement is necessary but not
    sufficient.
    """

    runner = B_MODEL_RUNNER.read_text()
    assert "SCOUT_B_MODEL_DIVERGENT" in runner
    assert "SCOUT_B_MODEL_UNGUARDED" in runner

    divergent = (FORMAL_DIR / "ScoutBModelDivergent.cfg").read_text()
    unguarded = (FORMAL_DIR / "ScoutBModelUnguarded.cfg").read_text()

    # Each negative config checks exactly one invariant: the shared classifier
    # requires exactly one Error line, so a config where two invariants are
    # violable would flake on schedule order.
    assert divergent.count("INVARIANT") == 1
    assert unguarded.count("INVARIANT") == 1
    assert "DeadlockFreedom" in divergent
    assert "RendezvousOpAgreement" in unguarded
    # The divergent case keeps NCCL's guard; the unguarded case drops it.
    assert "RequireMatchedIssueOrder = TRUE" in divergent
    assert "RequireMatchedIssueOrder = FALSE" in unguarded


def test_lint_path_file_does_not_collide_across_runs() -> None:
    """Regression: a derived temp name aborts the stage with an empty log.

    /project/tmp persists between runs, so a path derived from the stage
    identity already exists when the same output root is reused. The
    fail-closed existence check then aborted the lint stage before it produced
    any output, which surfaced only as a zero-byte log.
    """

    for name in ("run_scout_a.sh", "run_scout_b.sh"):
        runner = (
            REPO_ROOT / "experiments" / "qwen3_formal_verifier" / name
        ).read_text()
        assert "mktemp /project/tmp/lint-paths." in runner, name
        assert '"/project/tmp/${lint_identity}.paths"' not in runner, name


TIER0_RUNNER = (
    REPO_ROOT / "experiments" / "qwen3_formal_verifier" / "run_formal_tier0.sh"
)


def _parse_build_targets(
    build: str,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, list[str]]]:
    """Parse a BUILD file into filegroup srcs, sh_test inputs, and suites.

    Returns (filegroups, sh_tests, suites) where filegroups maps a filegroup
    name to its srcs, sh_tests maps a test name to its srcs plus data labels,
    and suites maps a test_suite name to its member labels. Deriving the sets
    from the BUILD text is the point: a test that pattern-matched an expected
    label list would pass while the targets behind those labels changed inputs.
    """

    filegroups: dict[str, set[str]] = {}
    sh_tests: dict[str, set[str]] = {}
    suites: dict[str, list[str]] = {}
    # Rule blocks start at column 0 and end at a lone ")" at column 0.
    for match in re.finditer(
        r"^(\w+)\(\n(.*?)^\)$", build, flags=re.DOTALL | re.MULTILINE
    ):
        rule, body = match.group(1), match.group(2)
        name_match = re.search(r'^\s*name = "([^"]+)",', body, flags=re.MULTILINE)
        if name_match is None:
            continue
        name = name_match.group(1)

        def list_attr(attr: str, body: str = body) -> list[str]:
            attr_match = re.search(
                rf"^\s*{attr} = \[([^\]]*)\]", body, flags=re.MULTILINE
            )
            if attr_match is None:
                return []
            return re.findall(r'"([^"]+)"', attr_match.group(1))

        if rule == "filegroup":
            filegroups[name] = set(list_attr("srcs"))
        elif rule == "sh_test":
            sh_tests[name] = set(list_attr("srcs")) | set(list_attr("data"))
        elif rule == "test_suite":
            suites[name] = list_attr("tests")
    return filegroups, sh_tests, suites


def _parse_sh_test_args(build: str) -> dict[str, list[str]]:
    """Map each sh_test name to its args list.

    Kept separate from _parse_build_targets so that helper's callers keep their
    three-tuple contract. Args matter because after the abstract/refinement
    split a target's inputs no longer determine which check it runs: both
    halves share one runner and are told apart by a positional argument.
    """

    args: dict[str, list[str]] = {}
    for match in re.finditer(
        r"^sh_test\(\n(.*?)^\)$", build, flags=re.DOTALL | re.MULTILINE
    ):
        body = match.group(1)
        name_match = re.search(r'^\s*name = "([^"]+)",', body, flags=re.MULTILINE)
        args_match = re.search(r"^\s*args = \[([^\]]*)\]", body, flags=re.MULTILINE)
        if name_match is None or args_match is None:
            continue
        args[name_match.group(1)] = re.findall(r'"([^"]+)"', args_match.group(1))
    return args


def _suite_input_files(build: str, suite: str) -> set[str]:
    """Resolve a test_suite to the package-local files its members read."""

    filegroups, sh_tests, suites = _parse_build_targets(build)
    assert suite in suites, f"{suite} is not declared in the BUILD file"
    files: set[str] = set()
    for member in suites[suite]:
        target = member.lstrip(":")
        assert target in sh_tests, f"{suite} refers to unknown target {target}"
        for label in sh_tests[target]:
            if label.startswith("@"):
                # External toolchain or archive, not a package source file.
                continue
            entry = label.lstrip(":")
            files |= filegroups.get(entry, {entry})
    return files


def test_tier0_formal_suite_reads_no_generated_facts_module() -> None:
    """Tier 0 must be runnable without an exported run.

    A *Facts module is generated from a sealed attempt bundle, so a tier-0
    target that read one would reintroduce the dependency the tier exists to
    remove. The forbidden set is derived from what the BUILD file actually
    wires up, not from a list kept in step by hand.
    """

    build = BUILD_FILE.read_text()
    tier0_files = _suite_input_files(build, "tier0_formal_tests")

    assert tier0_files, "the tier 0 suite resolved to no input files"
    facts = sorted(
        name
        for name in tier0_files
        if name.endswith("Facts.tla") or name.endswith("Facts.lean")
    )
    assert facts == [], facts

    # Guard against a vacuous check: generated facts modules must exist in the
    # package, so an empty result above means exclusion, not absence.
    assert list(FORMAL_DIR.glob("*Facts.tla"))
    assert list(FORMAL_DIR.glob("*Facts.lean"))


def test_tier0_targets_are_a_subset_of_the_sealed_scout_b_suite() -> None:
    """Tier 0 may only run checks the gate also runs.

    Otherwise a check could pass in iteration and never be sealed, which is the
    same hole as a tier-0 pass being presented as a gate pass.
    """

    build = BUILD_FILE.read_text()
    _, _, suites = _parse_build_targets(build)
    tier0 = {label.lstrip(":") for label in suites["tier0_formal_tests"]}
    sealed = {label.lstrip(":") for label in suites["scout_b_formal_tests"]}

    assert tier0, "the tier 0 suite declares no members"
    assert tier0 <= sealed, tier0 - sealed


# Generated from the Python trace IR for reference and read by no checker
# target. Every other formal source must be reachable from a sealed suite.
UNCHECKED_FORMAL_SOURCES = {"Qwen3StepTrace.lean", "Qwen3StepTrace.tla"}


def _formal_sources(*patterns: str) -> set[str]:
    return {
        path.name
        for pattern in patterns
        for path in FORMAL_DIR.glob(pattern)
        if path.suffix in {".tla", ".cfg", ".lean"}
    }


def test_sealed_suites_still_cover_every_checked_formal_source() -> None:
    """No-regression property for the abstract/refinement split.

    Splitting one target into two renames labels, so the durable invariant is
    over inputs, not target names: every formal source a checker is supposed to
    read must still be reachable from the sealed suite that owns it. A split
    that quietly left the refinement bridge out of the gate would surface here
    as an uncovered ScoutARefine module.
    """

    build = BUILD_FILE.read_text()

    sealed = _suite_input_files(build, "scout_b_formal_tests")
    every_source = _formal_sources("*")
    assert UNCHECKED_FORMAL_SOURCES < every_source, "stale exception list"
    uncovered = every_source - UNCHECKED_FORMAL_SOURCES - sealed
    assert uncovered == set(), sorted(uncovered)

    scout_a = _suite_input_files(build, "scout_a_formal_tests")
    scout_a_sources = _formal_sources(
        "ScoutA*", "ScoutLifecycle*", "TlcSmoke*", "LeanSmoke*"
    )
    assert scout_a_sources
    assert scout_a_sources <= scout_a, sorted(scout_a_sources - scout_a)


def test_every_scout_a_model_source_stays_in_both_sealed_suites() -> None:
    """Durable form of the no-regression property, independent of git history.

    Globbed from the package, so a new model or refinement configuration that
    nothing wires into the sealed suites fails here rather than sitting unused.
    """

    build = BUILD_FILE.read_text()
    model_sources = _formal_sources(
        "ScoutAModel*", "ScoutARefine*", "ScoutLifecycle.tla"
    )
    assert model_sources

    for suite in ("scout_a_formal_tests", "scout_b_formal_tests"):
        covered = _suite_input_files(build, suite)
        assert model_sources <= covered, (suite, sorted(model_sources - covered))


def _tier0_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    """Stub the formal wrapper and the rootfs entrypoint for a tier-0 run."""

    formal_log = tmp_path / "formal-checks.args"
    entrypoint_log = tmp_path / "entrypoint.args"
    formal_checks = tmp_path / "fake-run-formal-checks.sh"
    formal_checks.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$@" >> "${TIER0_FORMAL_LOG}"\n'
    )
    formal_checks.chmod(0o755)
    entrypoint = tmp_path / "fake-enter-rootfs.sh"
    entrypoint.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "${TORCHTITAN_ROOTFS_NETWORK:-}" >> "${TIER0_ENTRY_LOG}"\n'
    )
    entrypoint.chmod(0o755)
    # Deterministic changed-file set: whether this checkout happens to be dirty
    # must not decide whether the lint stage runs, or the assertions below would
    # pass or fail with the working tree rather than with the runner.
    git_stub = tmp_path / "fake-git.sh"
    git_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'case "$*" in\n'
        "  *ls-files*) : ;;\n"
        "  *diff*)\n"
        "    printf 'README.md\\0scripts/run_formal_checks.sh\\0'\n"
        "    ;;\n"
        '  *) echo "unexpected git invocation: $*" >&2; exit 2 ;;\n'
        "esac\n"
    )
    git_stub.chmod(0o755)
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "TIER0_FORMAL_LOG": str(formal_log),
            "TIER0_ENTRY_LOG": str(entrypoint_log),
            "TORCHTITAN_FORMAL_CACHE_HOST": str(tmp_path / "formal-cache"),
            "TORCHTITAN_TIER0_FORMAL_CHECKS": str(formal_checks),
            "TORCHTITAN_SCOUT_ROOTFS_ENTRYPOINT": str(entrypoint),
            "TORCHTITAN_SCOUT_ROOTFS": str(rootfs),
            "TORCHTITAN_SCOUT_GIT": str(git_stub),
        }
    )
    env.pop("TORCHTITAN_IN_ROOTFS", None)
    return env, formal_log, entrypoint_log


def _run_tier0(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(TIER0_RUNNER), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _outputs_snapshot() -> set[str]:
    outputs = REPO_ROOT / "outputs"
    if not outputs.exists():
        return set()
    return {str(path.relative_to(outputs)) for path in outputs.rglob("*")}


def test_tier0_runner_selects_the_tier0_suite_and_seals_no_bundle(
    tmp_path: Path,
) -> None:
    """The whole safety argument of the tier is that it produces no evidence."""

    env, formal_log, entrypoint_log = _tier0_environment(tmp_path)
    before = _outputs_snapshot()

    result = _run_tier0(env, "--no-fetch")

    assert result.returncode == 0, result.stdout
    assert formal_log.read_text().splitlines() == [
        "--no-fetch",
        "--suite",
        "tier0",
    ]
    # Lint runs networked for hook downloads; the pytest stage stays offline.
    assert entrypoint_log.read_text().splitlines() == ["networked", "offline"]
    assert "not-a-gate" in result.stdout
    assert "NOT a gate pass" in result.stdout
    assert _outputs_snapshot() == before


def test_tier0_runner_defaults_to_networked_and_honours_skips(
    tmp_path: Path,
) -> None:
    env, formal_log, entrypoint_log = _tier0_environment(tmp_path)

    result = _run_tier0(env, "--skip-lint", "--skip-pytest")

    assert result.returncode == 0, result.stdout
    assert formal_log.read_text().splitlines() == [
        "--networked",
        "--suite",
        "tier0",
    ]
    assert not entrypoint_log.exists()


@pytest.mark.parametrize(
    ("args", "diagnostic"),
    [
        # The gate's evidence-bundle arguments get their own diagnostic. Testing
        # only the exit status would pass if these fell into the catch-all, which
        # would drop the explanation a reader reaching for them needs.
        pytest.param(
            ["--attempt-id", "four-rank-dp2-tp2-v1"],
            "is a gate argument",
            id="attempt_id",
        ),
        pytest.param(
            ["--output-root", "outputs/x"], "is a gate argument", id="output_root"
        ),
        pytest.param(["--run-id", "qfv"], "is a gate argument", id="run_id"),
        pytest.param(
            ["--update-artifacts"], "is a gate argument", id="update_artifacts"
        ),
        pytest.param(
            ["--suite", "scout-b"], "unknown argument", id="unknown_suite_argument"
        ),
        pytest.param(
            ["--networked", "extra"], "unknown argument", id="stray_positional"
        ),
    ],
)
def test_tier0_runner_refuses_gate_and_unknown_arguments(
    tmp_path: Path, args: list[str], diagnostic: str
) -> None:
    """Refuse before running anything, so the refusal cannot be half a run."""

    env, formal_log, _ = _tier0_environment(tmp_path)

    result = _run_tier0(env, *args)

    assert result.returncode != 0
    assert "Usage:" in result.stdout
    assert diagnostic in result.stdout, result.stdout
    assert not formal_log.exists()


def test_tier0_runner_refuses_to_run_inside_the_rootfs(tmp_path: Path) -> None:
    """The formal wrapper needs a fresh sandbox it establishes itself."""

    env, formal_log, _ = _tier0_environment(tmp_path)
    env["TORCHTITAN_IN_ROOTFS"] = "1"

    result = _run_tier0(env)

    assert result.returncode != 0
    assert "outside Insula" in result.stdout
    assert not formal_log.exists()


def test_tier0_runner_help_states_it_is_not_a_gate(tmp_path: Path) -> None:
    env, formal_log, _ = _tier0_environment(tmp_path)

    result = _run_tier0(env, "--help")

    assert result.returncode == 0, result.stdout
    assert "not a gate" in result.stdout
    assert not formal_log.exists()
