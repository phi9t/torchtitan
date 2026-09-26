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


def test_tlc_constant_false_classifier_matches_the_real_checker_output() -> None:
    """A constant-level invariant is refuted by a different TLC outcome.

    StreamEdgeIsInert mentions no VARIABLES, so TLC evaluates it as a constant
    expression and reports a false one as "The invariant of X is equal to FALSE"
    with exit 151 -- not "violated by the initial state" with exit 12. The shape
    below is copied from a real run, and the classifier must accept exactly it:
    a looser one would let the stream-edge stage pass on a crashed run, and the
    state-predicate classifiers reject this output outright.
    """

    real = (
        "TLC2 Version 2.19 of 08 August 2024\n"
        "Starting... (2026-09-26 10:00:00)\n"
        "Error: The invariant of StreamEdgeIsInert is equal to FALSE\n"
        "Finished in 00s at (2026-09-26 10:00:00)\n"
    )

    accepted = _run_classifier(
        "formal_classify_tlc_constant_false", 151, real, "StreamEdgeIsInert"
    )
    assert accepted.returncode == 0, accepted.stdout

    # A different invariant name must not be accepted.
    wrong_name = _run_classifier(
        "formal_classify_tlc_constant_false", 151, real, "SomethingElse"
    )
    assert wrong_name.returncode != 0

    # Nor a different exit status, nor a run that died before finishing.
    wrong_status = _run_classifier(
        "formal_classify_tlc_constant_false", 12, real, "StreamEdgeIsInert"
    )
    assert wrong_status.returncode != 0

    truncated = _run_classifier(
        "formal_classify_tlc_constant_false",
        151,
        "Error: The invariant of StreamEdgeIsInert is equal to FALSE\n",
        "StreamEdgeIsInert",
    )
    assert truncated.returncode != 0, "no orderly finish line, so not a result"

    crashed = _run_classifier(
        "formal_classify_tlc_constant_false",
        151,
        real + 'Exception in thread "main" java.lang.NullPointerException\n',
        "StreamEdgeIsInert",
    )
    assert crashed.returncode != 0, "a crash after the diagnostic is not a result"

    # And the state-predicate classifiers must not accept it, or the stream-edge
    # stage could be wired to the wrong one and still pass.
    for function in (
        "formal_classify_tlc_negative",
        "formal_classify_tlc_transition_negative",
        "formal_classify_tlc_valid",
    ):
        mismatched = _run_classifier(function, 151, real, "StreamEdgeIsInert")
        assert mismatched.returncode != 0, function


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


def _tla_definition(module: str, name: str) -> str:
    """Return the body of a single TLA+ definition, up to the next blank line."""

    body = module.split(f"\n{name} ==", 1)[1]
    return body.split("\n\n", 1)[0]


def _tla_constants(module: str) -> set[str]:
    """Return the names declared in the module's CONSTANTS block."""

    block = module.split("\nCONSTANTS\n", 1)[1].split("\n\n", 1)[0]
    names = set()
    for raw in block.splitlines():
        line = raw.strip().rstrip(",")
        if not line or line.startswith("\\*"):
            continue
        names.add(line)
    return names


def _cfg_settings(cfg: str) -> dict[str, str]:
    """Parse a TLC cfg into {name: value} for scalar and override bindings."""

    settings: dict[str, str] = {}
    for raw in cfg.splitlines():
        line = raw.strip()
        if not line or line.startswith("\\*"):
            continue
        for separator in ("<-", "="):
            if separator in line:
                key, _, value = line.partition(separator)
                if key.strip() and value.strip():
                    settings[key.strip()] = value.strip()
                break
    return settings


def _cfg_invariants(cfg: str) -> list[str]:
    """Return the names in a cfg's INVARIANT / INVARIANTS block, in order.

    Parsed rather than substring-matched: an earlier version asserted
    `"StuckImpliesAllDone" in cfg`, which a `\\*` comment mentioning the name
    satisfies, so deleting the invariant from the block was caught by nothing.
    """

    names: list[str] = []
    collecting = False
    for raw in cfg.splitlines():
        line = raw.strip()
        if not line or line.startswith("\\*"):
            continue
        head = line.split()[0]
        if head in {"INVARIANT", "INVARIANTS"}:
            collecting = True
            names.extend(line.split()[1:])
            continue
        if head in {
            "CONSTANT",
            "CONSTANTS",
            "SPECIFICATION",
            "CONSTRAINT",
            "PROPERTY",
            "PROPERTIES",
            "CHECK_DEADLOCK",
            "SYMMETRY",
            "VIEW",
            "INIT",
            "NEXT",
        }:
            collecting = False
            continue
        if collecting:
            names.append(line)
    return names


B_MODEL_CFGS = (
    "ScoutBModel.cfg",
    "ScoutBModelReach.cfg",
    "ScoutBModelDivergent.cfg",
    "ScoutBModelWitness.cfg",
    "ScoutBModelOpMismatch.cfg",
    "ScoutBModelStreamShape.cfg",
    "ScoutBModelUnguarded.cfg",
)


def test_dpxtp_schedule_guard_agrees_on_communicator_identity_not_class() -> None:
    """The guard models one SPMD program, so it must fix the communicator.

    Retraction, recorded rather than quietly dropped. An earlier version made
    ranks agree on the communicator CLASS and the operation at each position,
    and claimed that modelled one program. It did not: two ranks could satisfy
    it while choosing different communicator INSTANCES of the same class, which
    is exactly the cross-communicator reordering freedom the model exists to
    constrain. The Small instance passed safety only because each rank held one
    communicator per class, so the class-only guard forced its choice.

    The guard now fixes the operation per position and fixes communicator
    IDENTITY wherever two ranks' communicators share a member -- which is
    precisely where a wait edge can form.

    Class agreement is gone, but not because it is inert. It WAS inert in the old
    module, where CommOps derived the admissible operations from the class, so
    requiring operations to agree already forced the class. Here CommOps is per
    communicator and gives mesh_fsdp all_reduce as well, so the conjunct does
    restrict: measured, adding a four-way role-agreement conjunct back changes no
    verdict and cuts the reachable graph from 38321 to 13583 distinct states. It
    is dropped because what it removes are not hazards -- it forbids rank 0
    issuing batch02 against rank 1 issuing loss13, which are disjoint and cannot
    wait on each other.
    """

    model = B_MODEL.read_text()

    ops_guard = _tla_definition(model, "UniformProgramOpsOK(candidate)")
    assert ".op = " in ops_guard, "operation agreement must be required"

    comms_guard = _tla_definition(model, "UniformProgramCommsOK(candidate)")
    assert "SameSite(" in comms_guard, "site agreement must be required"

    same_site = _tla_definition(model, "SameSite(c1, c2)")
    # Identical, or disjoint members. Anything else is two sites at once.
    assert "c1 = c2" in same_site
    assert "CommMembers[c1] \\cap CommMembers[c2] = {}" in same_site

    # The retracted conjunct must be gone, not merely unused: a CommClass
    # constant left declared would be bound by every cfg and read as a live
    # part of the guard. The header still names it, to record the retraction.
    assert "CommClass" not in _tla_constants(model)
    for name in B_MODEL_CFGS:
        assert "CommClass" not in (FORMAL_DIR / name).read_text(), name


def _parse_comm_members(model: str) -> dict[str, frozenset[int]]:
    """Reconstruct CommMembers2x2, including the unnamed OTHER branch."""

    body = _tla_definition(model, "CommMembers2x2")
    members: dict[str, frozenset[int]] = {}
    for comm, ranks in re.findall(r'c = "(\w+)"\s*->\s*\{([0-9, ]+)\}', body):
        members[comm] = frozenset(int(r) for r in ranks.split(","))
    declared = set(re.findall(r'"(\w+)"', _tla_definition(model, "CommIds2x2")))
    missing = declared - set(members)
    assert len(missing) == 1, f"unparsed CommMembers2x2 branches: {missing}"
    other = re.search(r"OTHER\s*->\s*\{([0-9, ]+)\}", body)
    assert other is not None
    members[missing.pop()] = frozenset(int(r) for r in other.group(1).split(","))
    return members


def _parse_comm_ops(model: str) -> dict[str, frozenset[str]]:
    """Reconstruct CommOps2x2, resolving the Ops alias and the OTHER branch."""

    all_ops = frozenset(re.findall(r'"(\w+)"', _tla_definition(model, "Ops")))
    assert all_ops, "Ops must be a set literal of operation names"

    body = _tla_definition(model, "CommOps2x2")

    def resolve(arm: str) -> frozenset[str]:
        arm = arm.strip()
        if arm == "Ops":
            return all_ops
        named = frozenset(re.findall(r'"(\w+)"', arm))
        assert named <= all_ops, f"unknown operations in CommOps2x2: {arm}"
        assert named, f"empty operation set in CommOps2x2: {arm}"
        return named

    ops: dict[str, frozenset[str]] = {}
    for comm, arm in re.findall(r'c = "(\w+)"\s*->\s*(Ops|\{[^}]*\})', body):
        ops[comm] = resolve(arm)
    other = re.search(r"OTHER\s*->\s*(Ops|\{[^}]*\})", body)
    assert other is not None
    default = resolve(other.group(1))
    for comm in _parse_comm_members(model):
        ops.setdefault(comm, default)
    return ops


def _parse_stream_of_issue(model: str) -> dict[str, str]:
    """Reconstruct StreamOfIssue as {operation: stream}, plus the OTHER arm.

    Returns a mapping with the literal operations it names and the key "OTHER"
    for the fall-through, so a caller can apply it to any operation. Asserts the
    definition is keyed on the operation at all: a stream map keyed on e.comm is
    refined by communicator identity, which is the design that made the
    stream-head conjunct redundant with per-communicator FIFO.
    """

    body = _tla_definition(model, "StreamOfIssue(e)")
    assert "e.op" in body, "StreamOfIssue must be keyed on the operation"
    assert "e.comm" not in body, (
        "StreamOfIssue keyed on the communicator makes AtStreamHead redundant "
        "with per-communicator FIFO and StuckByCircularWait unsatisfiable"
    )
    mapping = dict(re.findall(r'e\.op = "(\w+)"\s*->\s*"(\w+)"', body))
    other = re.search(r'OTHER\s*->\s*"(\w+)"', body)
    assert other is not None, "StreamOfIssue must have a fall-through arm"
    mapping["OTHER"] = other.group(1)
    return mapping


def test_dpxtp_instance_puts_distinct_communicators_on_one_rank_stream() -> None:
    """Derived from the three maps, not asserted: one rank, one stream, two
    communicators.

    A rank whose communicators never share a stream cannot have a
    cross-communicator wait edge at all, because per-communicator FIFO already
    orders everything else. That was true of every instance this module used to
    define, which is why no configuration could exhibit the hazard: the
    stream-head conjunct was exactly redundant by construction.

    An earlier version of this test claimed to reconstruct StreamOfIssue and did
    not: it collected every communicator a rank belonged to, never applied the
    stream map, and passed under three separate mutations -- a distinct single
    operation per communicator, StreamOfIssue keyed on e.comm, and AtStreamHead
    deleted from MemberReady. All three are now caught, the first two here and
    the third by the assertion at the end.
    """

    model = B_MODEL.read_text()
    members = _parse_comm_members(model)
    comm_ops = _parse_comm_ops(model)
    stream_of = _parse_stream_of_issue(model)

    def streams_of(comm: str) -> set[str]:
        return {stream_of.get(op, stream_of["OTHER"]) for op in comm_ops[comm]}

    # Two distinct communicators over the same rank set: without this, the
    # observed mesh_batch / mesh_fsdp / mesh_loss_mesh overlap is erased.
    by_member_set: dict[frozenset[int], set[str]] = {}
    for comm, ranks in members.items():
        by_member_set.setdefault(ranks, set()).add(comm)
    assert any(len(comms) >= 2 for comms in by_member_set.values()), members

    # The property the hazard needs: some rank holds two DISTINCT communicators
    # that can put an issue on one stream.
    overlaps: dict[tuple[int, str], set[str]] = {}
    for comm, ranks in members.items():
        for rank in ranks:
            for stream in streams_of(comm):
                overlaps.setdefault((rank, stream), set()).add(comm)
    shared = {key: comms for key, comms in overlaps.items() if len(comms) >= 2}
    assert shared, (
        "no rank can put two different communicators on one stream, so "
        f"AtStreamHead is redundant with per-communicator FIFO: {overlaps}"
    )

    # And the conjunct must still gate the rendezvous. Deleting it from
    # MemberReady leaves the instance shape above intact, so the shape check
    # alone cannot see it.
    ready = _tla_definition(model, "MemberReady(c, r)")
    assert "AtStreamHead" in ready, ready


def test_dpxtp_stream_edge_check_replaces_a_tautological_run() -> None:
    r"""Retraction: the RequireStreamOrder = FALSE rerun could not fail.

    With the stream edge deleted, AtStreamHead is TRUE, so MemberReady(c, r)
    reduces to CommCount(r, c) >= Front(c) and StartAllowed(c) to
    FullyPending(c) /\ OpsAgreeAtFront(c) -- which Stuck already denies for
    every communicator. StuckByCircularWait is therefore unsatisfiable by
    construction, at any bound on any instance, and the 1146243-state exhaustive
    search that was shipped here proved a two-line lemma rather than anything
    about this model. Corroborated two ways by the reviewer: deleting the
    balance conjunct changed nothing there, and StuckImpliesAllDone WAS violated
    under that cfg, so removing the stream edge removed the classification
    rather than the stuckness.

    What is contingent is the lemma's hypothesis, so that is what is checked:
    StreamEdgeIsInert must be violated on the shipped map, and must hold once
    StreamOfIssue is substituted with e.comm. Both halves are TLC runs in
    run_tlc_scout_b_model.sh, and both are instant because the cfg is pinned to
    the initial state.
    """

    assert not (
        FORMAL_DIR / "ScoutBModelStreams.cfg"
    ).exists(), "the tautological stream cfg must not come back"

    shape = (FORMAL_DIR / "ScoutBModelStreamShape.cfg").read_text()
    assert _cfg_invariants(shape) == ["StreamEdgeIsInert"], shape
    # Pinned to the initial state: StreamEdgeIsInert reads only the constants.
    assert "CONSTRAINT AtInitialState" in shape

    model = B_MODEL.read_text()
    initial = _tla_definition(model, "AtInitialState")
    assert "Len(issued[r]) = 0" in initial, initial

    runner = B_MODEL_RUNNER.read_text()
    stage = runner.split("# 6. The stream edge is load-bearing", 1)[1]
    stage = stage.split("# 7.", 1)[0]
    # The shipped map must fail the check...
    assert "formal_classify_tlc_constant_false" in stage
    assert "StreamEdgeIsInert" in stage
    # ...and the mutated map must pass it, or the check has no teeth.
    assert "StreamOfIssue(e) == e.comm" in stage
    assert "formal_classify_tlc_valid" in stage
    # Fail closed if the substitution silently stops applying.
    assert "the StreamOfIssue mutation did not apply" in stage
    assert "SCOUT_B_MODEL_STREAM_EDGE" in stage
    assert "SCOUT_B_MODEL_STREAM_EDGE_MUTANT" in stage


def test_dpxtp_witness_shape_is_checked_not_described() -> None:
    """ "DeadlockFreedom was violated" does not say which stuck state was found.

    This is the guard that would have caught the defect the review found: the
    first version of StuckByCircularWait required issue counts to be equal on
    every communicator in the state, so any unrelated imbalance elsewhere
    reclassified a real deadlock as StuckByBudget. A shape assertion that
    includes chain balance and a cycle of at least two communicators is
    satisfiable only by a witness that is genuinely local.
    """

    model = B_MODEL.read_text()

    shape = _tla_definition(model, "NoCrossCommunicatorCycleWitness")
    assert "Stuck" in shape
    assert "Len(issued[r]) = MaxIssues" in shape, "no rank may have stopped early"
    assert "CircularWaitAt(c)" in shape
    assert "Cardinality(BlockingClosure(c)) >= 2" in shape, "cross-communicator"
    assert "ChainIssuanceBalanced(BlockingClosure(c))" in shape

    witness = (FORMAL_DIR / "ScoutBModelWitness.cfg").read_text()
    assert _cfg_invariants(witness) == ["NoCrossCommunicatorCycleWitness"], witness

    # Same constants as the divergent cfg, or the shape is a statement about a
    # different model. The runner repeats this at run time.
    divergent = _cfg_settings((FORMAL_DIR / "ScoutBModelDivergent.cfg").read_text())
    differing = {
        key
        for key in set(divergent) | set(_cfg_settings(witness))
        if divergent.get(key) != _cfg_settings(witness).get(key)
    }
    assert differing == set(), differing

    runner = B_MODEL_RUNNER.read_text()
    assert "SCOUT_B_MODEL_WITNESS" in runner
    assert "expected_witness_diff" in runner


def test_dpxtp_model_runner_requires_non_vacuity_and_branching() -> None:
    """Safety invariants are worthless if no state satisfies the guards."""

    runner = B_MODEL_RUNNER.read_text()

    assert "SCOUT_B_MODEL_NONVACUOUS" in runner
    assert "ModelNeverCompletes" in runner
    assert "vacuous" in runner
    assert "max_outdegree" in runner
    assert "-ge 2" in runner
    assert "replay, not a model" in runner


def test_dpxtp_safety_token_reports_the_bound_it_was_checked_at() -> None:
    """A state count without a bound reads as a claim about the real run.

    The safety result is a statement about schedules of at most MaxIssues
    collectives per rank on a fixed instance. The token used to carry only
    distinct_states and max_outdegree, so a reader met the number without the
    qualifier, while the bound appeared in the ticket and the commit message
    only. The runner now derives the bound from the cfg and the module instead
    of printing a literal, so the reported bound cannot drift from the checked
    one.
    """

    runner = B_MODEL_RUNNER.read_text()

    assert "bound_max_issues_per_rank=%s" in runner
    assert "bound_ranks=%s" in runner
    assert "bound_communicators=%s" in runner
    assert "bound_issue_skew=unbounded" in runner
    assert "cfg_scalar ScoutBModel.cfg MaxIssues" in runner
    assert "set_size Ranks2x2" in runner
    assert "set_size CommIds2x2" in runner


def test_dpxtp_model_declares_no_inert_skew_bound() -> None:
    """MaxSkew was a tautology at MaxIssues = 2 and advertised a bound.

    The guard read `Len(issued[r]) + 1 <= Len(issued[q]) + MaxSkew` for every q.
    With MaxIssues = 2 the largest reachable left-hand side is 2 and the
    smallest right-hand side is 0 + 2, so the conjunct was true in every state,
    while a reader of the cfg saw `MaxSkew = 2` and believed rank skew was
    bounded. Dropped rather than tightened, because a real bound would have cut
    interleavings out of the safety claim; skew is now unbounded within
    MaxIssues, which is what the safety token says.
    """

    model = B_MODEL.read_text()
    assert "MaxSkew" not in _tla_constants(model)
    for name in B_MODEL_CFGS:
        assert "MaxSkew" not in (FORMAL_DIR / name).read_text(), name

    bounded = _tla_definition(model, "ModelBounded")
    assert "Len(issued[r]) <= MaxIssues" in bounded


def test_dpxtp_model_binds_every_instance_definition_it_declares() -> None:
    """A dead instance is worse than no instance: it reads as coverage.

    CommIdsWide, CommMembersWide and CommClassWide described the overlap that
    creates the wait cycle and were bound by no cfg, so the hazard they
    described was in nothing that was checked, while a test counting "{0, 2}"
    occurrences in that dead text reported otherwise.
    """

    model = B_MODEL.read_text()
    instance_definitions = {
        name
        for name in re.findall(r"^([A-Za-z]\w*) ==", model, re.MULTILINE)
        if re.search(r"(Small|Wide|2x2)$", name)
    }
    assert instance_definitions, "no instance definitions found"

    bound = set()
    for name in B_MODEL_CFGS:
        for value in _cfg_settings((FORMAL_DIR / name).read_text()).values():
            bound.add(value)
    assert instance_definitions <= bound, instance_definitions - bound


def test_dpxtp_hazard_classes_partition_the_stuck_states() -> None:
    """Stuck => AllDone cannot tell a wait cycle from the finite budget.

    AllDone requires every member of a communicator to have issued the same
    COUNT on it, so a rank spending its budget on one communicator while a peer
    spends it on another makes AllDone false forever; once both are out of
    budget, Stuck holds and `Stuck => AllDone` is violated by arithmetic. The
    module therefore classifies stuck states, and DeadlockFreedom is the
    circular-wait class directly.

    The balance conjunct is the load-bearing part: without it the class also
    fires on a chain that a larger budget would resolve, which was observed on
    a TLC witness before it was added.
    """

    model = B_MODEL.read_text()

    circular = _tla_definition(model, "StuckByCircularWait")
    assert "Stuck" in circular
    assert "CircularWaitAt(c)" in circular

    # The condition must be imposed on the blocking CLOSURE, not on all of
    # CommIds. The global form was the defect: a circular wait is local to its
    # chain, so any unrelated imbalance elsewhere in the state reclassified a
    # real deadlock as StuckByBudget. The review demonstrated it at MaxIssues=2
    # under ScoutBModelDivergent.cfg -- ranks 1 and 3 in a two-cycle over
    # fsdp13 and batch13, hung at any budget, discarded because rank 0 had
    # issued tp01 twice and rank 1 not at all.
    local = _tla_definition(model, "CircularWaitAt(c)")
    assert "FullyPending(c)" in local
    assert "OpsAgreeAtFront(c)" in local
    assert "BlockingClosure(c)" in local, local
    assert "IssuanceBalanced" not in local, (
        "a state-wide balance conjunct turns unrelated imbalance into a false "
        "negative for the hazard this invariant exists to catch"
    )

    mismatch = _tla_definition(model, "StuckByOpMismatch")
    assert "~StuckByCircularWait" in mismatch

    # The budget class is the remainder, so the three are exhaustive over
    # Stuck /\ ~AllDone by construction rather than by a case analysis that can
    # drift from the other two definitions.
    budget = _tla_definition(model, "StuckByBudget")
    assert "~StuckByCircularWait" in budget
    assert "~StuckByOpMismatch" in budget
    assert "~AllDone" in budget

    assert "DeadlockFreedom == ~StuckByCircularWait" in model
    assert "NoOpMismatchHang == ~StuckByOpMismatch" in model
    # The old, stronger statement is kept and still checked where it holds,
    # rather than being deleted along with its name. Checked by parsing the
    # INVARIANTS block: a mention in a comment must not satisfy this.
    assert "StuckImpliesAllDone == Stuck => AllDone" in model
    safety_invariants = _cfg_invariants((FORMAL_DIR / "ScoutBModel.cfg").read_text())
    assert "StuckImpliesAllDone" in safety_invariants, safety_invariants
    assert "DeadlockFreedom" in safety_invariants, safety_invariants
    assert "NoOpMismatchHang" not in safety_invariants, (
        "NoOpMismatchHang belongs to the op-mismatch negative; adding it here "
        "would make the safety cfg violate two invariants in one run"
    )


def test_dpxtp_negatives_are_distinct_results_not_one() -> None:
    """Four relaxations, four invariants, four result tokens.

    Each says something different: relaxing communicator-site agreement
    deadlocks; deleting the stream edge removes that deadlock; relaxing
    operation agreement hangs for an unrelated reason; additionally dropping
    NCCL's matching guard corrupts the rendezvous instead of hanging. Collapsing
    any two would lose the distinction between an ordering hazard and a
    divergent-program hazard.
    """

    runner = B_MODEL_RUNNER.read_text()
    for token in (
        "SCOUT_B_MODEL_DIVERGENT",
        "SCOUT_B_MODEL_WITNESS",
        "SCOUT_B_MODEL_OPMISMATCH",
        "SCOUT_B_MODEL_STREAM_EDGE",
        "SCOUT_B_MODEL_UNGUARDED",
    ):
        assert token in runner, token

    expected_invariant = {
        "ScoutBModelReach.cfg": "ModelNeverCompletes",
        "ScoutBModelDivergent.cfg": "DeadlockFreedom",
        "ScoutBModelWitness.cfg": "NoCrossCommunicatorCycleWitness",
        "ScoutBModelOpMismatch.cfg": "NoOpMismatchHang",
        "ScoutBModelStreamShape.cfg": "StreamEdgeIsInert",
        "ScoutBModelUnguarded.cfg": "RendezvousOpAgreement",
    }
    for name, invariant in expected_invariant.items():
        # Each negative config checks exactly one invariant: the shared
        # classifier requires exactly one Error line, so a config where two
        # invariants are violable would flake on schedule order. Parsed from
        # the block, so a comment naming an invariant proves nothing.
        assert _cfg_invariants((FORMAL_DIR / name).read_text()) == [invariant], name

    divergent = _cfg_settings((FORMAL_DIR / "ScoutBModelDivergent.cfg").read_text())
    opmismatch = _cfg_settings((FORMAL_DIR / "ScoutBModelOpMismatch.cfg").read_text())
    unguarded = _cfg_settings((FORMAL_DIR / "ScoutBModelUnguarded.cfg").read_text())

    # The ordering negative keeps operation agreement, which is what makes its
    # witness an ordering hazard; the mismatch negative is its mirror image.
    assert divergent["RequireUniformProgramOps"] == "TRUE"
    assert divergent["RequireUniformProgramComms"] == "FALSE"
    assert opmismatch["RequireUniformProgramOps"] == "FALSE"
    assert opmismatch["RequireUniformProgramComms"] == "TRUE"
    # Both keep NCCL's own guard; only the unguarded case drops it.
    assert divergent["RequireMatchedIssueOrder"] == "TRUE"
    assert opmismatch["RequireMatchedIssueOrder"] == "TRUE"
    assert unguarded["RequireMatchedIssueOrder"] == "FALSE"


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
