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


def _run_predicate(function: str, output: str):
    """Call a single-argument predicate, whose only argument is the output.

    formal_has_infrastructure_error, formal_completed_normally and
    formal_terminated_normally take the checker output as "$1"; the classify_*
    functions take (status, output). Passing one shape to the other silently
    tests nothing, so they get separate helpers.
    """

    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; "$2" "$3"',
            "predicate-test",
            str(CHECKER_CONTRACT),
            function,
            output,
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


def _tlc_toolchain() -> tuple[str, str]:
    """Locate the pinned TLC toolchain in the mounted formal cache.

    The tests below assert what the runner DOES, which means running real TLC.
    The toolchain is the same vendored JDK 17 and tla2tools 1.7.4 the Bazel
    targets use; run_formal_tier0.sh mounts the cache for the pytest stage so
    these do not silently degrade into skips.
    """

    cache = Path(os.environ.get("TORCHTITAN_FORMAL_CACHE", "/project/formal-cache"))
    vendor = cache / "bazel" / "vendor"
    java = sorted(vendor.glob("*remotejdk17*/bin/java"))
    jar = sorted(vendor.glob("*tla2tools*/file/tla2tools.jar"))
    if not java or not jar:
        pytest.skip(
            f"pinned TLC toolchain is not materialized under {vendor}; run "
            "scripts/run_formal_checks.sh --networked once to vendor it"
        )
    return str(java[0]), str(jar[0])


def _run_model_runner(
    tmp_path: Path,
    half: str,
    overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run run_tlc_scout_a_model.sh over a copy of the fixtures.

    `overrides` replaces the text of named fixture files, which is how a
    deliberately degenerate input is fed to the real checker. Each override
    must name a fixture that exists, so a renamed module fails the test rather
    than quietly adding a file the runner never stages.
    """

    java, jar = _tlc_toolchain()
    srcdir = tmp_path / "srcdir"
    fixtures = srcdir / "_main" / "experiments" / "qwen3_formal_verifier" / "formal"
    fixtures.mkdir(parents=True, exist_ok=True)
    for source in FORMAL_DIR.iterdir():
        if source.is_file():
            (fixtures / source.name).write_bytes(source.read_bytes())
    for name, text in (overrides or {}).items():
        target = fixtures / name
        assert target.is_file(), f"override names a missing fixture: {name}"
        target.write_text(text)
    test_tmpdir = tmp_path / "tmp" / half
    test_tmpdir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        ["bash", str(MODEL_RUNNER), java, jar, half],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "TEST_SRCDIR": str(srcdir),
            "TEST_WORKSPACE": "_main",
            "TEST_TMPDIR": str(test_tmpdir),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=900,
    )


def _run_tlc(
    tmp_path: Path,
    module: str,
    cfg_text: str,
    *,
    fixtures: tuple[str, ...] = (),
    files: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run real TLC over a staged directory; return (status, output).

    `fixtures` names shipped modules to stage as-is, so a probe can EXTEND the
    real ScoutLifecycle or ScoutARefineBad rather than a copy that could drift.
    `files` writes or replaces module text on top of them. `module` names the
    root module, whose .cfg is written from `cfg_text`.
    """

    java, jar = _tlc_toolchain()
    work = tmp_path / f"tlc-{module}"
    work.mkdir(parents=True, exist_ok=True)
    for name in fixtures:
        (work / name).write_bytes((FORMAL_DIR / name).read_bytes())
    for name, text in (files or {}).items():
        (work / name).write_text(text)
    assert (work / f"{module}.tla").is_file(), module
    (work / f"{module}.cfg").write_text(cfg_text)
    result = subprocess.run(
        [
            java,
            "-XX:+UseParallelGC",
            # Same thread stack the DPxTP refinement runner uses, and for the
            # same reason: ScoutBModel's SPMD-program guards contain a \A over
            # 1..issue-count inside an action, where TLC recurses once per
            # bound element, so a probe that replays tens of issues per rank
            # overflows the default 1 MB stack. Raising it here cannot weaken
            # any other probe.
            "-Xss32m",
            "-cp",
            jar,
            "tlc2.TLC",
            "-workers",
            "1",
            "-metadir",
            str(work / "states"),
            f"{module}.tla",
            "-config",
            f"{module}.cfg",
        ],
        cwd=work,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    return result.returncode, result.stdout


# A probe over ScoutLifecycle's own definitions. `Absent` applies the partial
# IndexOf outside its domain on purpose; `Guarded` and `FirstOccurrence` go
# through the guarded call sites and must not crash.
_LIFECYCLE_PROBE = """\
------------------------------ MODULE LifecycleProbe ------------------------------
EXTENDS Naturals, Sequences, ScoutLifecycle

VARIABLE cursor
vars == <<cursor>>
Init == cursor = 0
Next == UNCHANGED cursor
Spec == Init /\\ [][Next]_vars

Kinds == <<"a", "b", "a", "c">>

Absent == IndexOf(Kinds, "zzz") = 1

Guarded ==
  /\\ BeforeIfPresent(Kinds, "zzz", "c") = FALSE
  /\\ BeforeIfPresent(Kinds, "a", "zzz") = TRUE

FirstOccurrence ==
  /\\ IndexOf(Kinds, "a") = 1
  /\\ IndexOf(<<"b", "a", "a">>, "a") = 2
  /\\ BeforeIfPresent(Kinds, "c", "a") = FALSE

=============================================================================
"""


# A stand-in ScoutAFacts whose lifecycle never reaches gradient readiness. This
# is the input on which ScoutARefineBad's IndexOf lookups have no witness.
_FACTS_WITHOUT_GRADIENT_READY = """\
------------------------------ MODULE ScoutAFacts ------------------------------
EXTENDS Naturals, Sequences, FiniteSets, TLC

EventKinds ==
  << "step.started", "batch.observed", "forward.started", "forward.completed",
     "backward.started", "backward.completed", "optimizer.started",
     "step.completed" >>

=============================================================================
"""


def _substitute(text: str, old: str, new: str) -> str:
    """Replace `old` once, refusing to produce an unchanged mutant."""

    assert text.count(old) == 1, f"expected exactly one occurrence of {old!r}"
    return text.replace(old, new)


# A cursor walking one recorded sequence: outdegree 1 everywhere, while every
# safety invariant in ScoutAModel.cfg still holds. This is precisely the shape
# the outdegree floor exists to refuse and the shape a state-count threshold
# would accept, so it is the right degenerate input for that floor.
_NON_BRANCHING_MODEL = """\
-------------------------------- MODULE ScoutAModel --------------------------------
EXTENDS Naturals, Sequences, ScoutLifecycle

CONSTANT RequireReadyGradients

Recorded ==
  << "step.started", "batch.observed", "forward.started", "forward.completed",
     "backward.started", "gradient.ready", "backward.completed",
     "optimizer.started", "optimizer.mutated", "step.completed" >>

VARIABLES emitted, cursor

vars == <<emitted, cursor>>

Init ==
  /\\ emitted = <<>>
  /\\ cursor = 0

Advance ==
  /\\ cursor < Len(Recorded)
  /\\ cursor' = cursor + 1
  /\\ emitted' = Append(emitted, Recorded[cursor + 1])

Terminated ==
  /\\ cursor = Len(Recorded)
  /\\ UNCHANGED vars

Next == Advance \\/ Terminated

Spec == Init /\\ [][Next]_vars

TypeOK ==
  /\\ cursor \\in 0..Len(Recorded)
  /\\ emitted = Prefix(Recorded, cursor)

MutationRequiresReadyGradients ==
  HasKind(emitted, "optimizer.mutated") => HasKind(emitted, "gradient.ready")

MutationFollowsGradientEvidence ==
  BeforeIfPresent(emitted, "gradient.ready", "optimizer.mutated")

GradientResolutionUnique ==
  ~(HasKind(emitted, "gradient.ready") /\\ HasKind(emitted, "gradient.missing"))

ForwardBracketsBackward ==
  ForwardBeforeBackward(emitted, Len(emitted))

StepCompletionIsLast ==
  HasKind(emitted, "step.completed") =>
    emitted[Len(emitted)] = "step.completed"

=============================================================================
"""


def test_tlc_evaluation_error_is_classified_as_an_infrastructure_error(
    tmp_path: Path,
) -> None:
    """A CHOOSE with no witness abandons the search and must not classify.

    The output is produced by running TLC, not written here, because the whole
    point is what the real checker prints: alongside the evaluation error it
    ALSO prints `1 states generated, 1 distinct states found, 1 states left on
    queue.` and `Finished in 00s`, so formal_completed_normally and
    formal_terminated_normally both return TRUE. Before the evaluation-error
    class was added, the exit-status pin was the only thing rejecting it, which
    is single-layer defence where layered classification is advertised.
    """

    status, output = _run_tlc(
        tmp_path,
        "LifecycleProbe",
        "SPECIFICATION Spec\nINVARIANTS\n  Absent\n",
        fixtures=("ScoutLifecycle.tla",),
        files={"LifecycleProbe.tla": _LIFECYCLE_PROBE},
    )

    assert status == 75, output
    assert "Attempted to compute the value of an expression of form" in output
    assert "states left on queue" in output
    # The two weak signals really are TRUE on this output, which is why the
    # class is needed.
    assert _run_predicate("formal_completed_normally", output).returncode == 0
    assert _run_predicate("formal_terminated_normally", output).returncode == 0
    # And these are what reject it.
    assert (
        _run_predicate("formal_has_infrastructure_error", output).returncode == 0
    ), "the evaluation-error class must match"
    assert _run_classifier("formal_classify_tlc_valid", 0, output).returncode != 0
    assert (
        _run_classifier(
            "formal_classify_tlc_transition_negative", 12, output, "Absent"
        ).returncode
        != 0
    )


def test_guarded_index_call_sites_survive_an_absent_and_repeated_value(
    tmp_path: Path,
) -> None:
    """BeforeIfPresent must be total, and IndexOf must pin the first match.

    Two things are checked by running TLC: an absent value taken through
    BeforeIfPresent produces a clean FALSE rather than an evaluation error, and
    a repeated value resolves to its FIRST index.

    The first-index result is the one that matters for ticket 12. `CHOOSE` is
    deterministic but unspecified in TLA+, so before the `\\A earlier` conjunct
    the agreement between this module and Lean's left-to-right
    `occursBeforeIfPresent` rested on which witness TLC happens to return. The
    single-rank kind sequence has no repeats only because scout_a.py pins it to
    one literal; the DPxTP port repeats collective kinds once per rank.

    What this test cannot distinguish: the disjunctive form this replaced also
    evaluated safely under TLC 1.7.4, which short-circuits left to right. The
    guard makes the totality part of the semantics rather than of the
    evaluator. Where the guard is load-bearing on real input is
    ScoutARefineBad, covered by the next test.
    """

    for invariant in ("Guarded", "FirstOccurrence"):
        status, output = _run_tlc(
            tmp_path / invariant,
            "LifecycleProbe",
            f"SPECIFICATION Spec\nINVARIANTS\n  {invariant}\n",
            fixtures=("ScoutLifecycle.tla",),
            files={"LifecycleProbe.tla": _LIFECYCLE_PROBE},
        )
        assert status == 0, output
        assert "Attempted to compute the value" not in output, output
        assert (
            _run_classifier("formal_classify_tlc_valid", status, output).returncode == 0
        ), output


def test_index_of_pins_the_first_occurrence_in_the_specification() -> None:
    """Asserted textually, and for a reason worth stating.

    This is the one claim in this file that execution cannot witness. TLC's
    CHOOSE already returns the first witness it finds, so the unguarded
    definition evaluates to the same number, and no TLC run distinguishes the
    two. The difference is in TLA+: unguarded, `IndexOf` is satisfied by ANY
    index holding the value, so `BeforeIfPresent` agreeing with Lean's
    left-to-right `occursBeforeIfPresent` depended on TLC's choice rather than
    on the specification -- which defeats the purpose of maintaining two
    independent implementations. Ticket 12's DPxTP port makes the repeats real.
    """

    lifecycle = (FORMAL_DIR / "ScoutLifecycle.tla").read_text()
    body = lifecycle.split("IndexOf(sequence, value) ==", 1)[1].split("\n\n", 1)[0]

    assert "sequence[index] = value" in body, body
    assert "\\A earlier \\in 1..index - 1 : sequence[earlier] # value" in body, body


def test_refinement_control_names_its_missing_target_instead_of_crashing(
    tmp_path: Path,
) -> None:
    """The load-bearing case for the IndexOf guard, on a real module.

    ScoutARefineBad reads `IndexOf(EventKinds, "gradient.ready")` while it is
    constructing `Observed`, before any invariant can be evaluated, so an
    unguarded lookup over a trace without that event made TLC abandon the
    search with an evaluation error -- which the classifiers then read as a
    completed, orderly run and only the exit status rejected.

    Guarded, the same input produces a named refutation of
    `ControlEventsArePresent` instead. That invariant mentions no VARIABLES, so
    TLC reports it as a false constant expression.
    """

    status, output = _run_tlc(
        tmp_path,
        "ScoutARefineBad",
        (FORMAL_DIR / "ScoutARefineBad.cfg").read_text(),
        fixtures=(
            "ScoutLifecycle.tla",
            "ScoutAModel.tla",
            "ScoutARefineBad.tla",
        ),
        files={"ScoutAFacts.tla": _FACTS_WITHOUT_GRADIENT_READY},
    )

    assert "Attempted to compute the value" not in output, output
    assert (
        _run_classifier(
            "formal_classify_tlc_constant_false",
            status,
            output,
            "ControlEventsArePresent",
        ).returncode
        == 0
    ), (status, output)
    # And it is a refutation, not a crash dressed as one.
    assert _run_predicate("formal_has_infrastructure_error", output).returncode != 0


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


def test_refinement_half_admits_the_observed_trace_and_isolates_the_guard(
    tmp_path: Path,
) -> None:
    """Positive control for the refinement half, by running it.

    This is also the only place the guard-isolation check's success is
    observed, so a runner whose fifth stage never ran would fail here.
    """

    result = _run_model_runner(tmp_path, "refine")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "SCOUT_A_REFINEMENT result=admitted" in result.stdout
    assert "SCOUT_A_REFINEMENT_NEGATIVE result=rejected_at_mutation_guard" in (
        result.stdout
    )
    assert (
        "SCOUT_A_REFINEMENT_GUARD_ISOLATION relaxed=RequireReadyGradients"
        " result=admitted" in result.stdout
    ), result.stdout


def test_refinement_negative_is_rejected_for_the_right_reason(
    tmp_path: Path,
) -> None:
    """Rejection alone is weak evidence; it must hit the guard under test.

    An earlier version swapped `gradient.ready` with `optimizer.mutated`, which
    put the mutation at a position where the optimizer had not started. The
    model refused it, but for an out-of-order lifecycle rather than unready
    gradients -- a different guard than the one being tested. The control now
    substitutes `gradient.missing`, so every earlier event stays admissible and
    the trace dies exactly at OptimizerMutate.

    Run, not grepped. The degenerate input substitutes `forward.started`
    instead, which is inadmissible at the gradient position, so the trace is
    refused three events too early. `CorruptionIsIsolated` still holds -- one
    differing position, still before the mutation -- and the trace is still not
    admitted, so the first stage passes. Only the reach stage can catch it.
    """

    bad = (FORMAL_DIR / "ScoutARefineBad.tla").read_text()

    # Derived from the same facts the positive bridge uses, not hand-written.
    assert "ScoutAFacts" in bad
    assert "IndexOf(EventKinds" in bad
    assert '"step.started"' not in bad

    result = _run_model_runner(
        tmp_path,
        "refine",
        {
            "ScoutARefineBad.tla": _substitute(
                bad,
                'IF i = ReadyIndex THEN "gradient.missing"',
                'IF i = ReadyIndex THEN "forward.started"',
            )
        },
    )

    assert result.returncode != 0, result.stdout
    assert "refused before reaching the mutation guard" in result.stderr, result.stderr
    assert "wrong guard" in result.stderr
    # The weaker stages must have accepted it, or this proves nothing about the
    # reach stage.
    assert "corrupted trace was not refused" not in result.stderr
    assert "SCOUT_A_REFINEMENT result=admitted" in result.stdout
    assert "SCOUT_A_REFINEMENT_NEGATIVE" not in result.stdout


def test_relaxing_the_guard_must_admit_the_corrupted_trace(tmp_path: Path) -> None:
    """The decisive check: it attributes the refusal to one constant.

    Stages 3 and 4 only triangulate WHERE the refusal landed. Stage 5 reruns
    the same corrupted trace with RequireReadyGradients relaxed and requires it
    to become admitted, which is what makes the refusal that guard's doing.

    The degenerate input is the mistake that would hollow the stage out: a
    ScoutARefineBadRelaxed.cfg that forgot to relax the constant. The corrupted
    trace is then still refused, no violation is reported, and the stage must
    refuse to print its token.
    """

    relaxed = (FORMAL_DIR / "ScoutARefineBadRelaxed.cfg").read_text()

    result = _run_model_runner(
        tmp_path,
        "refine",
        {
            "ScoutARefineBadRelaxed.cfg": _substitute(
                relaxed,
                "CONSTANT RequireReadyGradients = FALSE",
                "CONSTANT RequireReadyGradients = TRUE",
            )
        },
    )

    assert result.returncode != 0, result.stdout
    assert "not attributable to that guard" in result.stderr, result.stderr
    # Everything before it still passed, so the failure is stage 5's alone.
    assert "SCOUT_A_REFINEMENT_NEGATIVE result=rejected_at_mutation_guard" in (
        result.stdout
    )
    assert "SCOUT_A_REFINEMENT_GUARD_ISOLATION" not in result.stdout


def test_refinement_reports_an_empty_observed_trace_as_vacuous(
    tmp_path: Path,
) -> None:
    """An empty Observed is admitted by every model, not refused by this one.

    Init already satisfies emitted = Observed, so TLC exits 0 with no
    violation -- byte-for-byte the outcome a REFUSED trace produces. The runner
    used to print "observed trace was not admitted by the abstract model" for
    it, which states the opposite of what happened. ObservedIsNonEmpty mentions
    no variables, so TLC refutes a false one as a constant expression with exit
    151 and its own message, giving the runner an outcome it can name.
    """

    refine = (FORMAL_DIR / "ScoutARefine.tla").read_text()

    result = _run_model_runner(
        tmp_path,
        "refine",
        {
            "ScoutARefine.tla": _substitute(
                refine, "Observed == EventKinds", "Observed == <<>>"
            )
        },
    )

    assert result.returncode != 0, result.stdout
    assert "vacuous" in result.stderr, result.stderr
    assert "the observed event-kind sequence is empty" in result.stderr
    assert "was not admitted by the abstract model" not in result.stderr


def test_refinement_negative_configurations_check_both_halves() -> None:
    """The two cfgs must differ only in which invariant they demand.

    Checked as configuration rather than by execution because the runs above
    already prove the outcomes; what this pins is that neither cfg quietly
    drops CorruptionIsIsolated or ControlEventsArePresent, which is what keeps
    a control that stopped being a control from passing.
    """

    reject = _cfg_invariants((FORMAL_DIR / "ScoutARefineBad.cfg").read_text())
    reach = _cfg_invariants((FORMAL_DIR / "ScoutARefineBadReach.cfg").read_text())
    relaxed = _cfg_invariants((FORMAL_DIR / "ScoutARefineBadRelaxed.cfg").read_text())

    assert reject == [
        "ControlEventsArePresent",
        "CorruptionIsIsolated",
        "ObservedTraceIsNotAdmitted",
        "MutationIsNeverEmitted",
    ], reject
    assert reach == [
        "ControlEventsArePresent",
        "CorruptionIsIsolated",
        "RejectionHappensBeforeTheMutation",
    ], reach
    assert relaxed == [
        "ControlEventsArePresent",
        "CorruptionIsIsolated",
        "ObservedTraceIsNotAdmitted",
    ], relaxed


def test_refinement_configs_disable_deadlock_so_polarity_is_unambiguous() -> None:
    """Rejection must read as a clean completion, not as a checker error.

    Without this, a refused trace surfaces as TLC exit 11 (deadlock), which is
    indistinguishable from a genuine specification defect.
    """

    for name in ("ScoutARefine.cfg", "ScoutARefineBad.cfg"):
        assert "CHECK_DEADLOCK FALSE" in (FORMAL_DIR / name).read_text(), name


def test_model_runner_accepts_the_shipped_abstract_model(tmp_path: Path) -> None:
    """Positive control for the abstract half, by running it.

    Without this, a runner that refused everything would satisfy the
    degenerate-input test below while checking nothing.
    """

    result = _run_model_runner(tmp_path, "abstract")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "SCOUT_A_MODEL_SAFETY result=success" in result.stdout
    assert "max_outdegree=3" in result.stdout, result.stdout
    assert (
        "SCOUT_A_MODEL_NEGATIVE invariant=MutationRequiresReadyGradients"
        " result=named_violation" in result.stdout
    )


def test_model_runner_refuses_a_non_branching_specification(tmp_path: Path) -> None:
    """Branching, not state count, separates a model from a replay.

    A cursor walking a recorded sequence has outdegree 1 everywhere however
    long the sequence is, so a threshold on state count would accept the very
    design this module replaces -- ScoutAValid reaches 12 states.

    This runs the real checker over a substituted ScoutAModel that is exactly
    such a cursor walk and still satisfies every invariant in ScoutAModel.cfg.
    Asserting the runner's text instead would keep passing if `-ge 2` were
    applied to a variable no TLC output ever populated, which is the shape
    that has burned this project twice.
    """

    result = _run_model_runner(
        tmp_path, "abstract", {"ScoutAModel.tla": _NON_BRANCHING_MODEL}
    )

    assert result.returncode != 0, result.stdout
    assert "max outdegree 1" in result.stderr, result.stderr
    assert "replay, not a model" in result.stderr, result.stderr
    # The refusal must happen at the floor, not by the safety check failing:
    # a degenerate model that also broke an invariant would prove nothing
    # about the floor.
    assert "abstract model safety check was not a clean success" not in result.stderr
    # And the check that follows must not have run.
    assert "SCOUT_A_MODEL_NEGATIVE" not in result.stdout


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
        build_source_manifest(tmp_path, "b" * 40, status, head_commit_paths=b"")


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
    manifest = build_source_manifest(
        tmp_path, "b" * 40, f"?? {path_text}\0".encode(), head_commit_paths=b""
    )

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
        build_source_manifest(
            tmp_path, "b" * 40, f"?? {path_text}\0".encode(), head_commit_paths=b""
        )


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


def _lint_stage_program() -> str:
    """Return the lint-stage program both runners execute.

    The stage body used to be pasted into each runner, so these contracts read
    the runner text. It now lives in one emitter in the runner library, which is
    also what the executing clean-tree test in test_qwen3_formal_scout_a.py
    runs, so the text is read from there instead.
    """

    library = (
        REPO_ROOT / "experiments" / "qwen3_formal_verifier" / "scout_a_runner_lib.sh"
    )
    result = subprocess.run(
        ["bash", "-c", f"source {library}; scout_lint_stage_program"],
        capture_output=True,
        check=True,
    )
    return result.stdout.decode()


def test_both_runners_use_the_shared_lint_stage_program() -> None:
    """Neither runner may keep a private copy of the lint stage body."""

    for name in ("run_scout_a.sh", "run_scout_b.sh"):
        runner = (
            REPO_ROOT / "experiments" / "qwen3_formal_verifier" / name
        ).read_text()
        assert 'bash -lc "$(scout_lint_stage_program)"' in runner, name
        assert "git init --bare" not in runner, name


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

    program = _lint_stage_program()
    assert 'bash -n "${shell_file}"' in program
    assert 'bash -n "${shell_files[@]}"' not in program


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
    "ScoutBModelLive.cfg",
    "ScoutBModelLiveDivergent.cfg",
    "ScoutBModelLiveUnconditional.cfg",
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


# Copied from a real run of ScoutBModelLiveDivergent.cfg (exit 13, 1599947
# states generated, 401719 distinct, 78116 left on queue, 02min 05s). The trace
# body is elided; the shape -- two Error lines, no invariant line, a lasso
# closed by stuttering, a search summary and an orderly finish -- is verbatim.
_REAL_LIVENESS_STUTTERING_LASSO = (
    "TLC2 Version 2.19 of 08 August 2024 (rev: 5a47802)\n"
    "Starting... (2026-09-26 15:10:26)\n"
    "Implied-temporal checking--satisfiability problem has 1 branches.\n"
    "Computing initial states...\n"
    "Error: Temporal properties were violated.\n"
    "\n"
    "Error: The following behavior constitutes a counter-example:\n"
    "\n"
    "State 1: <Initial predicate>\n"
    "/\\ running = {}\n"
    "\n"
    "State 10: Stuttering\n"
    "Finished checking temporal properties in 00s at 2026-09-26 15:12:31\n"
    "1599947 states generated, 401719 distinct states found, "
    "78116 states left on queue.\n"
    "Finished in 02min 05s at (2026-09-26 15:12:31)\n"
)

# The other lasso shape, from a real TLC run on a throwaway two-state flip-flop
# with WF and an unsatisfiable <>(x = 2): this model's variables only grow, so it
# cannot itself produce a cyclic lasso, and inventing one is exactly what this
# project has been burned by. Same exit status and same Error lines; only the
# terminator differs.
_REAL_LIVENESS_CYCLIC_LASSO = (
    "TLC2 Version 2.19 of 08 August 2024 (rev: 5a47802)\n"
    "Computing initial states...\n"
    "Error: Temporal properties were violated.\n"
    "\n"
    "Error: The following behavior constitutes a counter-example:\n"
    "\n"
    "State 1: <Initial predicate>\n"
    "x = 0\n"
    "\n"
    "State 2: <Flip line 6, col 13 to line 6, col 27 of module CycleLasso>\n"
    "x = 1\n"
    "\n"
    "Back to state 1: <Flip line 7, col 13 to line 7, col 27 of module "
    "CycleLasso>\n"
    "\n"
    "Finished checking temporal properties in 00s at 2026-09-26 15:20:19\n"
    "3 states generated, 2 distinct states found, 0 states left on queue.\n"
    "Finished in 00s at (2026-09-26 15:20:19)\n"
)


def test_tlc_liveness_negative_classifier_matches_the_real_checker_output() -> None:
    """A refuted temporal property is a fourth TLC outcome, and a lasso.

    The standing lesson of this ticket: a liveness counterexample is not an
    invariant violation. TLC exits 13, prints TWO Error lines, prints no
    "Invariant X is violated" line at all, and closes the behaviour with either
    a Stuttering line or a "Back to state" line. Both shapes below come from
    real runs -- the first from ScoutBModelLiveDivergent.cfg, the second from a
    throwaway flip-flop, because this model's state cannot cycle -- and the
    existing classifiers must reject both, or a liveness stage could be wired to
    one of them and pass on the wrong evidence.
    """

    for real in (
        _REAL_LIVENESS_STUTTERING_LASSO,
        _REAL_LIVENESS_CYCLIC_LASSO,
    ):
        accepted = _run_classifier("formal_classify_tlc_liveness_negative", 13, real)
        assert accepted.returncode == 0, accepted.stdout

        # The exit status TLC uses for a refuted invariant is 12, and for a
        # false constant invariant 151. Neither is this outcome.
        for status in (0, 12, 151, 124, 127):
            wrong_status = _run_classifier(
                "formal_classify_tlc_liveness_negative", status, real
            )
            assert wrong_status.returncode != 0, status

        # And the invariant classifiers must not accept a lasso. The transition
        # negative is the one the ticket predicted would be reached for: it
        # requires exactly one Error line, and a lasso prints two.
        for function, expected in (
            ("formal_classify_tlc_transition_negative", "DeadlockFreedom"),
            ("formal_classify_tlc_negative", "DeadlockFreedom"),
            ("formal_classify_tlc_constant_false", "StreamEdgeIsInert"),
            ("formal_classify_tlc_valid", ""),
        ):
            mismatched = _run_classifier(function, 13, real, expected)
            assert mismatched.returncode != 0, function

    real = _REAL_LIVENESS_STUTTERING_LASSO

    # A counterexample with no lasso terminator is a truncated report: TLC was
    # killed mid-trace, and the behaviour it printed is not closed.
    open_trace = _run_classifier(
        "formal_classify_tlc_liveness_negative",
        13,
        real.replace("State 10: Stuttering\n", ""),
    )
    assert open_trace.returncode != 0, "an unclosed behaviour is not a lasso"

    # The usual pair: no orderly finish, and a crash after the diagnostic.
    truncated = _run_classifier(
        "formal_classify_tlc_liveness_negative",
        13,
        real.replace("Finished in 02min 05s at (2026-09-26 15:12:31)\n", ""),
    )
    assert truncated.returncode != 0

    crashed = _run_classifier(
        "formal_classify_tlc_liveness_negative",
        13,
        real + 'Exception in thread "main" java.lang.NullPointerException\n',
    )
    assert crashed.returncode != 0

    # Any further Error line is another failure, not this result.
    extra_error = _run_classifier(
        "formal_classify_tlc_liveness_negative",
        13,
        real + "Error: unrelated checker failure\n",
    )
    assert extra_error.returncode != 0


def _run_cfg_property_check(cfg_file: Path, expected: str):
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; formal_cfg_declares_one_property "$2" "$3"',
            "cfg-property-test",
            str(CHECKER_CONTRACT),
            str(cfg_file),
            expected,
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_liveness_stage_is_pinned_to_its_property_by_its_configuration(
    tmp_path: Path,
) -> None:
    """TLC's liveness diagnostic names no property, so the cfg is the binding.

    "Error: Temporal properties were violated." is all TLC says, so a liveness
    stage pointed at the wrong cfg would classify exactly the same. The runner
    therefore asserts what its cfg declares, and this executes that assertion
    against the shipped configurations and against the shapes that must fail:
    two properties, a block-form list, and a commented mention only.
    """

    for name, prop in (
        ("ScoutBModelLive.cfg", "EveryIssuedCollectiveCompletes"),
        ("ScoutBModelLiveDivergent.cfg", "EveryIssuedCollectiveCompletes"),
        (
            "ScoutBModelLiveUnconditional.cfg",
            "EveryIssuedCollectiveCompletesUnconditionally",
        ),
    ):
        accepted = _run_cfg_property_check(FORMAL_DIR / name, prop)
        assert accepted.returncode == 0, (name, accepted.stdout)

        wrong = _run_cfg_property_check(FORMAL_DIR / name, "SomethingElse")
        assert wrong.returncode != 0, name

    # A safety cfg declares no temporal property at all, so a liveness stage
    # pointed at one must be refused rather than silently checking nothing.
    safety = _run_cfg_property_check(
        FORMAL_DIR / "ScoutBModel.cfg", "EveryIssuedCollectiveCompletes"
    )
    assert safety.returncode != 0

    two_properties = tmp_path / "TwoProperties.cfg"
    two_properties.write_text(
        "SPECIFICATION LiveSpec\n"
        "PROPERTY EveryIssuedCollectiveCompletes\n"
        "PROPERTY EveryIssuedCollectiveCompletesUnconditionally\n"
    )
    assert (
        _run_cfg_property_check(
            two_properties, "EveryIssuedCollectiveCompletes"
        ).returncode
        != 0
    ), "a second property would make the violated one ambiguous"

    block_form = tmp_path / "BlockForm.cfg"
    block_form.write_text(
        "SPECIFICATION LiveSpec\n"
        "PROPERTIES\n"
        "  EveryIssuedCollectiveCompletes\n"
        "  EveryIssuedCollectiveCompletesUnconditionally\n"
        "CHECK_DEADLOCK FALSE\n"
    )
    assert (
        _run_cfg_property_check(block_form, "EveryIssuedCollectiveCompletes").returncode
        != 0
    ), "a block-form second property must not hide from the check"

    commented = tmp_path / "Commented.cfg"
    commented.write_text(
        "\\* PROPERTY EveryIssuedCollectiveCompletes\n"
        "SPECIFICATION LiveSpec\n"
        "CHECK_DEADLOCK FALSE\n"
    )
    assert (
        _run_cfg_property_check(commented, "EveryIssuedCollectiveCompletes").returncode
        != 0
    ), "a commented mention is not a declaration"


def test_liveness_specification_is_fair_on_start_and_complete_only() -> None:
    """Issue must stay unfair, or the result is about the fairness assumption.

    Rank skew is exactly Issue's freedom. A fair Issue would keep every rank
    issuing forever and the liveness property would hold because of the
    assumption rather than because of the protocol. Start and Complete are fair
    per communicator rather than under one existential, so a behaviour cannot
    satisfy fairness by servicing one communicator forever and another never.
    """

    model = B_MODEL.read_text()

    fairness = _tla_definition(model, "Fairness")
    assert "WF_vars(Start(c))" in fairness
    assert "WF_vars(Complete(c))" in fairness
    assert "\\A c \\in CommIds" in fairness, "fairness must be per communicator"
    assert "Issue" not in fairness, "Issue must not be fair"
    assert "SF_" not in fairness, "weak fairness suffices; Start stays enabled"

    live_spec = _tla_definition(model, "LiveSpec")
    assert "Init" in live_spec
    assert "[][Next]_vars" in live_spec
    assert "Fairness" in live_spec

    # The safety specification must stay fairness-free, or every safety cfg
    # would quietly start checking a different specification.
    safety_spec = _tla_definition(model, "Spec")
    assert "WF_" not in safety_spec and "SF_" not in safety_spec

    # And each cfg must name the specification its claim needs.
    for name in B_MODEL_CFGS:
        specification = _cfg_settings((FORMAL_DIR / name).read_text())
        declared = [
            line.split()[1]
            for line in (FORMAL_DIR / name).read_text().splitlines()
            if line.strip().startswith("SPECIFICATION")
        ]
        assert len(declared) == 1, name
        expected = "LiveSpec" if "Live" in name else "Spec"
        assert declared == [expected], (name, declared)
        assert specification, name


def test_liveness_configurations_isolate_the_one_thing_each_changes() -> None:
    """Three liveness runs, and exactly one difference between each pair.

    The negative must differ from the positive only in the site guard, or its
    counterexample would not be attributable to order divergence; the
    unconditional run must differ only in the property, or its refutation would
    be a statement about a different configuration rather than about the
    unconditional reading. Both are parsed from the cfgs rather than described.
    """

    positive = _cfg_settings((FORMAL_DIR / "ScoutBModelLive.cfg").read_text())
    negative = _cfg_settings((FORMAL_DIR / "ScoutBModelLiveDivergent.cfg").read_text())
    unconditional = _cfg_settings(
        (FORMAL_DIR / "ScoutBModelLiveUnconditional.cfg").read_text()
    )
    safety = _cfg_settings((FORMAL_DIR / "ScoutBModel.cfg").read_text())
    divergent = _cfg_settings((FORMAL_DIR / "ScoutBModelDivergent.cfg").read_text())

    differing = {
        key
        for key in set(positive) | set(negative)
        if positive.get(key) != negative.get(key)
    }
    assert differing == {"RequireUniformProgramComms"}, differing
    assert positive["RequireUniformProgramComms"] == "TRUE"
    assert negative["RequireUniformProgramComms"] == "FALSE"

    assert unconditional == positive, "only the property may differ here"

    # The liveness pair must sit on the same constants as the safety pair it
    # mirrors, so the liveness result is about the checked model and not about a
    # quietly different instance or bound.
    for live, reference in ((positive, safety), (negative, divergent)):
        shared = {
            key: value
            for key, value in reference.items()
            if key in live and key != "SPECIFICATION"
        }
        assert {key: live[key] for key in shared} == shared, shared

    # A state constraint under liveness checking can turn a pruned state into a
    # terminal node and manufacture a stuttering lasso; TLC warns about exactly
    # this. ModelBounded would prune nothing here, so it buys nothing and is
    # left out rather than relied on.
    for name in (
        "ScoutBModelLive.cfg",
        "ScoutBModelLiveDivergent.cfg",
        "ScoutBModelLiveUnconditional.cfg",
    ):
        text = (FORMAL_DIR / name).read_text()
        declared = [
            line
            for line in text.splitlines()
            if line.strip().startswith(("CONSTRAINT", "ACTION_CONSTRAINT"))
        ]
        assert declared == [], (name, declared)

    # The positive run is what shows the second antecedent conjunct costs
    # nothing at full budget in the good configuration, so it must check it.
    assert _cfg_invariants((FORMAL_DIR / "ScoutBModelLive.cfg").read_text()) == [
        "FullBudgetRendezvousPopulated"
    ]


def test_liveness_property_is_conditioned_and_the_literal_one_is_refuted() -> None:
    """The ticket's literal property is false, and that is a checked claim.

    With Issue unfair a behaviour may simply stop issuing, so a collective one
    rank has issued and its peers have not never becomes fully pending, never
    starts, and never completes. That is a program that stopped, not a hang, so
    "every issued collective eventually completes" read unconditionally is false
    with every guard on. The module keeps both readings: the conditioned one is
    checked to hold, and the unconditional one is checked to fail, so the
    narrowing is justified by a run rather than asserted in a comment.
    """

    model = B_MODEL.read_text()

    conditioned = _tla_definition(model, "EveryIssuedCollectiveCompletes")
    assert "~>" in conditioned, "liveness, not an invariant"
    assert "AllIssued" in conditioned
    assert "EveryRendezvousPopulated" in conditioned

    unconditional = _tla_definition(
        model, "EveryIssuedCollectiveCompletesUnconditionally"
    )
    assert "~>" in unconditional
    assert "AllIssued" not in unconditional

    # The domain guard on the right-hand side is load-bearing: TLC evaluates
    # both sides of a leads-to in every state, and Completed(r, k) in the
    # initial state, where issued[r] is empty, is an evaluation error rather
    # than FALSE -- which formal_has_infrastructure_error would then classify as
    # a crashed run.
    guarded = _tla_definition(model, "IssuedAndCompleted(r, k)")
    assert "k \\in DOMAIN issued[r]" in guarded
    assert guarded.index("DOMAIN issued[r]") < guarded.index("Completed(r, k)")

    # AllIssued is about the program being fully issued, not about completion.
    all_issued = _tla_definition(model, "AllIssued")
    assert "Len(issued[r]) = MaxIssues" in all_issued

    # And the populated conjunct is what keeps the negative off the finite-budget
    # artifact: without it the first counterexample is a rendezvous short of an
    # issue that was never made, which is StuckByBudget rather than a hang.
    populated = _tla_definition(model, "EveryRendezvousPopulated")
    assert "FullyPending(c)" in populated
    assert "CommCount(r, c) >= Front(c)" in populated


B_REFINE = FORMAL_DIR / "ScoutBRefine.tla"
B_REFINE_RUNNER = FORMAL_DIR / "run_tlc_scout_b_refine.sh"
B_REFINE_CFGS = (
    "ScoutBRefineMapping.cfg",
    "ScoutBRefine.cfg",
    "ScoutBRefineUniformPermutation.cfg",
    "ScoutBRefineSingleRankPermutation.cfg",
    "ScoutBRefineSkew.cfg",
    "ScoutBRefineOverlap.cfg",
    "ScoutBRefineBad.cfg",
    "ScoutBRefineBadReach.cfg",
    "ScoutBRefineBadRelaxed.cfg",
    "ScoutBRefineBadUniform.cfg",
)

# The facts the refinement bridge must NOT read. Each of these carries the
# started/completed sub-order or the inferred stream and producer attribution:
# the observer appends a collective's three events contiguously, post hoc, and
# the raw-event projection drops observed_time_ns, so per-rank EVENT order
# asserts a full serialization of every collective before the next. That cannot
# have happened -- FSDP all-gathers overlap by construction -- and a bridge
# replaying it would PASS, because a serialized schedule satisfies every guard.
_EVENT_ORDER_FACTS = (
    "CollectiveEventIds",
    "CollectiveLifecycle",
    "CollectiveStream",
    "CollectiveProducers",
    "EventOrder",
    "EventKind",
    "EventIds",
)


def _refine_fixtures() -> tuple[str, ...]:
    return (
        "ScoutBModel.tla",
        "ScoutBFacts.tla",
        "ScoutDistributed.tla",
        "ScoutBRefine.tla",
    )


def _refine_constants(**overrides: str) -> str:
    """Render a constant block for a ScoutBRefine probe cfg."""

    settings = {
        "MaxIssues": "108",
        "RequireMatchedIssueOrder": "TRUE",
        "RequireUniformProgramOps": "TRUE",
        "RequireUniformProgramComms": "TRUE",
        "RequireStreamOrder": "TRUE",
        "GreedyReplay": "TRUE",
        "MaxReplaySkew": "108",
        "MaxOutstanding": "108",
        "TransposeMutation": "FALSE",
        "ReplayPermutation": '"observed"',
    }
    settings.update(overrides)
    lines = [
        "CONSTANT",
        "  Ranks <- Ranks2x2",
        "  CommIds <- CommIds2x2",
        "  CommMembers <- CommMembers2x2",
        "  CommOps <- CommOps2x2",
        "",
        "CONSTANT",
    ]
    lines += [f"  {name} = {value}" for name, value in settings.items()]
    return "\n".join(lines) + "\n"


def test_dpxtp_bridge_replays_the_issue_order_and_not_the_event_order() -> None:
    """The one trap: only the enqueued sub-order is evidence.

    Stated over the identifiers the bridge actually mentions, because the
    distinction is invisible in the result -- a bridge built on per-rank event
    order would assert a full serialization of every collective and would
    still pass every guard.
    """

    bridge = B_REFINE.read_text()
    facts = (FORMAL_DIR / "ScoutBFacts.tla").read_text()

    assert "CollectiveIssueOrder" in bridge
    for name in _EVENT_ORDER_FACTS:
        # Guard against a vacuous check: the fact must exist to be excluded.
        assert f"\n{name} ==" in facts, name
        assert name not in bridge, name


def test_dpxtp_bridge_configs_bind_every_constant_they_inherit() -> None:
    """A cfg that forgets a constant does not fail; TLC refuses to run.

    The bridge extends ScoutBModel, so each of its configurations has to bind
    that model's constants as well as its own. Derived from both CONSTANTS
    blocks rather than from a list kept in step by hand.
    """

    declared = _tla_constants(B_REFINE.read_text()) | _tla_constants(
        B_MODEL.read_text()
    )
    assert "MaxOutstanding" in declared

    for name in B_REFINE_CFGS:
        bound = set(_cfg_settings((FORMAL_DIR / name).read_text()))
        assert declared <= bound, (name, sorted(declared - bound))


def test_dpxtp_bridge_configurations_check_the_halves_they_are_for() -> None:
    """Each configuration's invariant list, in order, as the runner needs it.

    Order matters: a structural failure -- no transposable pair, a mapping that
    disagrees with the model, a replay that stopped replaying -- must be the
    reported one, not a refusal inferred from it. Checked as configuration
    because the runs prove the outcomes.
    """

    invariants = {
        name: _cfg_invariants((FORMAL_DIR / name).read_text()) for name in B_REFINE_CFGS
    }

    assert invariants["ScoutBRefineMapping.cfg"] == [
        "ModelInstanceIsNonEmpty",
        "ObservedRanksMatchTheModel",
        "ObservedIssueOrderIsDistinctWithinRank",
        "CommMappingIsWellFormed",
        "ReplayLengthsAgreeAcrossRanks",
        "ReplayBoundIsWithinTheObservedTrace",
        "TransposableIssuePairExists",
    ]
    assert invariants["ScoutBRefine.cfg"] == [
        "IssuedIsAReplayPrefix",
        "ReplaySkewIsWithinBound",
        "ReplayedRunIsNotAdmitted",
    ]
    assert invariants["ScoutBRefineSkew.cfg"] == [
        "IssuedIsAReplayPrefix",
        "ReplaySkewIsWithinBound",
        "ReplayOutstandingIsWithinBound",
        "TypeOK",
    ]
    assert invariants["ScoutBRefineOverlap.cfg"] == ["NoTwoCollectivesRunConcurrently"]
    assert invariants["ScoutBRefineBad.cfg"] == [
        "TransposableIssuePairExists",
        "MutationIsIsolated",
        "ReplayedRunIsNotAdmitted",
        "MismatchedCollectiveNeverRuns",
    ]
    assert invariants["ScoutBRefineBadReach.cfg"] == [
        "TransposableIssuePairExists",
        "MutationIsIsolated",
        "RejectionHappensBeforeTheRendezvous",
    ]
    assert invariants["ScoutBRefineBadRelaxed.cfg"] == [
        "TransposableIssuePairExists",
        "MutationIsIsolated",
        "ReplayedRunIsNotAdmitted",
    ]
    assert invariants["ScoutBRefineBadUniform.cfg"] == [
        "TransposableIssuePairExists",
        "MutationIsIsolated",
        "ReplayedRunIsNotAdmitted",
        "CorruptedColumnIsNeverFormed",
    ]
    assert invariants["ScoutBRefineUniformPermutation.cfg"] == [
        "UniformPermutationIsNotTheObservedOrder",
        "UniformPermutationPreservesAgreement",
        "IssuedIsAReplayPrefix",
        "ReplayedRunIsNotAdmitted",
    ]
    assert invariants["ScoutBRefineSingleRankPermutation.cfg"] == [
        "SingleRankPermutationIsNotTheObservedOrder",
        "SingleRankPermutationBreaksAgreement",
        "IssuedIsAReplayPrefix",
        "ReplayedRunIsNotAdmitted",
    ]

    # Only the confluence configuration turns deadlock detection on, because
    # only there is a dead end the result rather than the expectation.
    for name in B_REFINE_CFGS:
        text = (FORMAL_DIR / name).read_text()
        expected = "TRUE" if name == "ScoutBRefineSkew.cfg" else "FALSE"
        assert f"CHECK_DEADLOCK {expected}" in text, name


def test_dpxtp_bridge_configs_pin_the_guard_values_not_just_the_names() -> None:
    """A cfg binding every constant says nothing about what it binds them TO.

    The positive's whole point is that it runs with every guard on; the three
    isolation configurations deliberately relax one, and one deliberately
    relaxes two. Nothing else prevents the positive's guard set drifting to
    match the negatives', which would make the positive a weaker claim while
    every other test still passed.
    """

    guards = (
        "RequireMatchedIssueOrder",
        "RequireUniformProgramOps",
        "RequireUniformProgramComms",
        "RequireStreamOrder",
    )
    settings = {
        name: _cfg_settings((FORMAL_DIR / name).read_text()) for name in B_REFINE_CFGS
    }

    # Every guard on: the positive, its inputs, the confluence fragment, the
    # order-sensitivity pair, and the corruption run under the positive's set.
    for name in (
        "ScoutBRefineMapping.cfg",
        "ScoutBRefine.cfg",
        "ScoutBRefineUniformPermutation.cfg",
        "ScoutBRefineSingleRankPermutation.cfg",
        "ScoutBRefineSkew.cfg",
        "ScoutBRefineOverlap.cfg",
        "ScoutBRefineBadUniform.cfg",
    ):
        for guard in guards:
            assert settings[name][guard] == "TRUE", (name, guard)

    # The three that isolate the rendezvous guard relax the SPMD operation
    # guard, and only the relaxed one also flips NCCL's own guard.
    for name in (
        "ScoutBRefineBad.cfg",
        "ScoutBRefineBadReach.cfg",
        "ScoutBRefineBadRelaxed.cfg",
    ):
        assert settings[name]["RequireUniformProgramOps"] == "FALSE", name
        assert settings[name]["RequireUniformProgramComms"] == "TRUE", name
        assert settings[name]["RequireStreamOrder"] == "TRUE", name
    assert settings["ScoutBRefineBad.cfg"]["RequireMatchedIssueOrder"] == "TRUE"
    assert settings["ScoutBRefineBadReach.cfg"]["RequireMatchedIssueOrder"] == "TRUE"
    assert settings["ScoutBRefineBadRelaxed.cfg"]["RequireMatchedIssueOrder"] == "FALSE"

    # And which order each configuration replays, for the same reason.
    corrupted = {
        "ScoutBRefineBad.cfg",
        "ScoutBRefineBadReach.cfg",
        "ScoutBRefineBadRelaxed.cfg",
        "ScoutBRefineBadUniform.cfg",
    }
    for name in B_REFINE_CFGS:
        expected = "TRUE" if name in corrupted else "FALSE"
        assert settings[name]["TransposeMutation"] == expected, name
    permutations = {
        "ScoutBRefineUniformPermutation.cfg": '"uniform"',
        "ScoutBRefineSingleRankPermutation.cfg": '"single_rank"',
    }
    for name in B_REFINE_CFGS:
        expected = permutations.get(name, '"observed"')
        assert settings[name]["ReplayPermutation"] == expected, name


# Prints the derived mutation site so the shape of the negative control is read
# off the checker rather than off the module text.
_MUTATION_PROBE = """\
--------------------------- MODULE MutationProbe ---------------------------
EXTENDS ScoutBRefine

MutationSiteReport ==
  PrintT(<<"PROBE_MUTATION",
           MutationRank,
           MutationIndex,
           MutationComm,
           MutationFront,
           ObservedIssues[MutationRank][MutationIndex],
           ObservedIssues[MutationRank][MutationIndex + 1]>>)

=============================================================================
"""


def test_dpxtp_bridge_mutation_is_an_adjacent_same_communicator_transposition(
    tmp_path: Path,
) -> None:
    """The negative must corrupt the rendezvous, not the program's sites.

    Transposing across two DIFFERENT communicators would be refused because a
    member never reached a communicator's front -- the stream-head or
    unissued-communicator guard, not the rendezvous guard the control exists to
    exercise. So the derived site has to be adjacent, on one communicator, with
    the operations differing. Read off the real checker, pinned to the initial
    state because every value here is a constant.
    """

    cfg = (
        _refine_constants(TransposeMutation="TRUE")
        + "\nSPECIFICATION RSpec\n\nINVARIANTS\n"
        "  TransposableIssuePairExists\n"
        "  MutationIsIsolated\n"
        "  MutationSiteReport\n"
        "\nCONSTRAINT AtInitialState\nCHECK_DEADLOCK FALSE\n"
    )
    status, output = _run_tlc(
        tmp_path,
        "MutationProbe",
        cfg,
        fixtures=_refine_fixtures(),
        files={"MutationProbe.tla": _MUTATION_PROBE},
    )

    assert status == 0, output
    assert "Model checking completed. No error has been found." in output, output
    report = output.split('<< "PROBE_MUTATION",', 1)[1].split(">>", 1)[0]
    fields = [item.strip().rstrip(",") for item in report.strip().splitlines()]
    # rank, index, communicator, per-communicator position, then the two
    # records that get swapped.
    assert fields[0] == "0", fields
    assert fields[1] == "52", fields
    assert fields[2] == '"fsdp02"', fields
    swapped = " ".join(fields[4:])
    assert swapped.count('comm |-> "fsdp02"') == 2, swapped
    assert 'op |-> "reduce_scatter"' in swapped, swapped
    assert 'op |-> "all_gather"' in swapped, swapped


# The filtered form of the replay constraint, the way ScoutARefine writes it.
# Nothing else changes, so a state-count difference against ConstrainedNext
# would mean the forward form admits a different set of behaviours.
_FILTER_PROBE = """\
---------------------------- MODULE FilterProbe ----------------------------
EXTENDS ScoutBRefine

FilterNext ==
  /\\ Next
  /\\ IssuedIsAReplayPrefix'

FilterSpec == Init /\\ [][FilterNext]_vars

=============================================================================
"""


def test_dpxtp_bridge_forward_replay_equals_the_filtered_form(
    tmp_path: Path,
) -> None:
    """The bridge computes the next issue instead of filtering successors.

    ScoutARefine writes `Next /\\ IsPrefixOfObserved(emitted')`. Here that form
    evaluates the SPMD-program guards, which are linear in the issue count, for
    every rank/communicator/operation triple and discards almost all of them.
    The forward form is equivalent -- Issue appends exactly one record, so a
    successor that is still a prefix can only have appended the next observed
    record -- and this runs both over the same small prefix and requires the
    same reachable state count rather than arguing it.
    """

    constants = _refine_constants(MaxIssues="4", GreedyReplay="FALSE")
    tail = (
        "\nINVARIANTS\n  IssuedIsAReplayPrefix\n  TypeOK\n"
        "\nCONSTRAINT ModelBounded\nCHECK_DEADLOCK FALSE\n"
    )
    counts = {}
    for label, specification in (
        ("forward", "RSpec"),
        ("filtered", "FilterSpec"),
    ):
        status, output = _run_tlc(
            tmp_path / label,
            "FilterProbe",
            constants + f"\nSPECIFICATION {specification}\n" + tail,
            fixtures=_refine_fixtures(),
            files={"FilterProbe.tla": _FILTER_PROBE},
        )
        assert status == 0, output
        assert "No error has been found." in output, output
        # The final summary line, not a progress line: TLC prints the same
        # phrase every minute while it runs, and reading the first match
        # compares two partial counts taken at different times.
        summary = re.findall(
            r"([0-9]+) states generated, ([0-9]+) distinct states found,"
            r" [0-9]+ states left on queue",
            output,
        )
        assert summary, output
        counts[label] = int(summary[-1][1])

    assert counts["forward"] > 1, counts
    assert counts["forward"] == counts["filtered"], counts


# Issues only, with the skew bound to keep the search narrow. Used to show what
# a length-only admission test would accept.
_ISSUE_ONLY_PROBE = """\
-------------------------- MODULE IssueOnlyProbe --------------------------
EXTENDS ScoutBRefine

IssueOnlyNext ==
  \\E rank \\in Ranks :
    /\\ ReplayIssue(rank)
    /\\ ReplaySkewOK(issued')

IssueOnlySpec == Init /\\ [][IssueOnlyNext]_vars

\\* What a bridge that tested lengths rather than completion would call
\\* admission.
LengthOnlyAdmissionIsNotReached ==
  ~(\\A rank \\in Ranks : Len(issued[rank]) = MaxIssues)

=============================================================================
"""


def test_dpxtp_bridge_length_only_admission_would_accept_the_corruption(
    tmp_path: Path,
) -> None:
    """Why admission is AllDone and not "every rank reached its length".

    Issue's guard never reads doneOn, so issuing the whole sequence and running
    nothing satisfies a length-only test -- for the CORRUPTED issue order just
    as much as for the real one. That bridge would be vacuous and its negative
    control would pass. This runs the corrupted order under an issue-only spec
    and shows the length-only predicate is reachable; the shipped
    ScoutBRefineBad.cfg shows AllDone is not.
    """

    cfg = (
        _refine_constants(
            MaxIssues="54",
            MaxReplaySkew="1",
            GreedyReplay="FALSE",
            RequireUniformProgramOps="FALSE",
            TransposeMutation="TRUE",
        )
        + "\nSPECIFICATION IssueOnlySpec\n\nINVARIANTS\n"
        "  TransposableIssuePairExists\n"
        "  MutationIsIsolated\n"
        "  LengthOnlyAdmissionIsNotReached\n"
        "\nCHECK_DEADLOCK FALSE\n"
    )
    status, output = _run_tlc(
        tmp_path,
        "IssueOnlyProbe",
        cfg,
        fixtures=_refine_fixtures(),
        files={"IssueOnlyProbe.tla": _ISSUE_ONLY_PROBE},
    )

    assert status == 12, output
    assert (
        "Error: Invariant LengthOnlyAdmissionIsNotReached is violated." in output
    ), output


# A stand-in ScoutBFacts whose only collective belongs to rank 1. This is the
# input on which ScoutBIssueOrderInvalid's derived mutation target has no
# witness.
_B_FACTS_WITHOUT_RANK_ZERO = """\
------------------------------ MODULE ScoutBFacts ------------------------------
EXTENDS Naturals, Sequences, FiniteSets, TLC

CollectiveWorkIds == <<"work:r1:only">>
CollectiveRank == ("work:r1:only" :> 1)
CollectiveComm == ("work:r1:only" :> "dp_shard:1,3:mesh_fsdp")
CollectiveOperation == ("work:r1:only" :> "all_gather")
CollectiveMembers == ("work:r1:only" :> {1, 3})
CollectiveIssueOrder == ("work:r1:only" :> 7)

=============================================================================
"""


def test_issue_order_control_names_its_missing_target_instead_of_crashing(
    tmp_path: Path,
) -> None:
    """The derived mutation target must be reported, not crashed on.

    `MutatedWork` is a constant definition, so TLC folds it before checking
    anything; an unguarded CHOOSE over a trace where rank 0 issued no
    collective abandoned the search with an evaluation error and exit 75, which
    is not a checking result. The guard turns that into a named invariant, and
    because the invariant mentions no variables TLC refutes it as a false
    constant expression.
    """

    status, output = _run_tlc(
        tmp_path,
        "ScoutBIssueOrderInvalid",
        (FORMAL_DIR / "ScoutBIssueOrderInvalid.cfg").read_text(),
        fixtures=("ScoutDistributed.tla", "ScoutBIssueOrderInvalid.tla"),
        files={"ScoutBFacts.tla": _B_FACTS_WITHOUT_RANK_ZERO},
    )

    assert "Attempted to compute the value of an expression of form" not in output
    assert (
        _run_classifier(
            "formal_classify_tlc_constant_false",
            status,
            output,
            "Rank0HasACollective",
        ).returncode
        == 0
    ), (status, output)


def test_dpxtp_bridge_is_wired_into_the_sealed_scout_b_suite() -> None:
    """The bridge reads generated facts, so it belongs to the sealed suite only.

    Tier 0 is the trace-free subset and must not grow a dependency on an
    exported run; the gate must not be missing the bridge.
    """

    build = BUILD_FILE.read_text()
    _, sh_tests, suites = _parse_build_targets(build)

    assert "tlc_scout_b_refine_test" in sh_tests
    sealed = {label.lstrip(":") for label in suites["scout_b_formal_tests"]}
    tier0 = {label.lstrip(":") for label in suites["tier0_formal_tests"]}
    assert "tlc_scout_b_refine_test" in sealed
    assert "tlc_scout_b_refine_test" not in tier0

    bridge_sources = _formal_sources("ScoutBRefine*")
    assert len(bridge_sources) == 1 + len(B_REFINE_CFGS), sorted(bridge_sources)
    assert len(B_REFINE_CFGS) == 10, B_REFINE_CFGS
    covered = _suite_input_files(build, "scout_b_formal_tests")
    assert bridge_sources <= covered, sorted(bridge_sources - covered)


def test_lint_path_file_does_not_collide_across_runs() -> None:
    """Regression: a derived temp name aborts the stage with an empty log.

    /project/tmp persists between runs, so a path derived from the stage
    identity already exists when the same output root is reused. The
    fail-closed existence check then aborted the lint stage before it produced
    any output, which surfaced only as a zero-byte log.
    """

    program = _lint_stage_program()
    assert 'mktemp "${scratch_dir}/lint-paths.' in program
    assert '"${scratch_dir}/${lint_identity}.paths"' not in program


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
        'printf "%s %s\\n" "${TORCHTITAN_ROOTFS_NETWORK:-}" '
        '"${TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST:-none}" >> "${TIER0_ENTRY_LOG}"\n'
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
    # Popped, not ignored: if the caller's own cache variable leaked through,
    # the assertion that the PYTEST stage is the stage that gets the mount
    # would pass for every stage.
    env.pop("TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST", None)
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
    # The pytest stage additionally gets the formal cache mounted, because some
    # of the contracts in this file run the real TLC toolchain against a
    # degenerate input; without the mount they would skip and check nothing.
    cache = str(tmp_path / "formal-cache")
    assert entrypoint_log.read_text().splitlines() == [
        "networked none",
        f"offline {cache}",
    ]
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


# ---------------------------------------------------------------------------
# The quantified protocol theorems (tickets 13 and 14).
#
# Every assertion below about what a checker DOES runs the checker. The one
# source-level test here is a uniformity check over five runner scripts, and its
# docstring says why running is not the stronger option for that one.
# ---------------------------------------------------------------------------

PROTOCOL_RUNNER = FORMAL_DIR / "run_lean_scout_b_protocol.sh"
SCOUT_A_LEAN_RUNNER = FORMAL_DIR / "run_lean_scout_a.sh"

# The obligations that must each be reported separately. A missing one silently
# weakens the inductive-invariant claim, which is the whole reason the runner
# emits one token per theorem instead of one aggregate token.
PROTOCOL_OBLIGATIONS = ("initiation", "consecution", "sufficiency")


def _lean_toolchain() -> str:
    """Locate the pinned Lean 4.34.0 in the mounted formal cache."""

    cache = Path(os.environ.get("TORCHTITAN_FORMAL_CACHE", "/project/formal-cache"))
    vendor = cache / "bazel" / "vendor"
    lean = sorted(vendor.glob("*lean_4_34_0*/bin/lean"))
    if not lean:
        pytest.skip(
            f"pinned Lean toolchain is not materialized under {vendor}; run "
            "scripts/run_formal_checks.sh --networked once to vendor it"
        )
    return str(lean[0])


def _run_lean_runner(
    tmp_path: Path,
    runner: Path,
    overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Lean runner script over a copy of the fixtures with real Lean.

    `overrides` replaces the text of named fixture files, which is how a
    deliberately broken module is fed to the real checker. Each override must
    name a fixture that exists, so a renamed module fails the test rather than
    quietly adding a file the runner never stages.
    """

    lean = _lean_toolchain()
    srcdir = tmp_path / "srcdir"
    fixtures = srcdir / "_main" / "experiments" / "qwen3_formal_verifier" / "formal"
    fixtures.mkdir(parents=True, exist_ok=True)
    for source in FORMAL_DIR.iterdir():
        if source.is_file():
            (fixtures / source.name).write_bytes(source.read_bytes())
    for name, text in (overrides or {}).items():
        target = fixtures / name
        assert target.is_file(), f"override names a missing fixture: {name}"
        target.write_text(text)
    test_tmpdir = tmp_path / "tmp" / runner.stem
    test_tmpdir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        ["bash", str(runner), lean],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "TEST_SRCDIR": str(srcdir),
            "TEST_WORKSPACE": "_main",
            "TEST_TMPDIR": str(test_tmpdir),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=900,
    )


def test_protocol_runner_reports_each_obligation_as_its_own_theorem_token(
    tmp_path: Path,
) -> None:
    """Positive control, by running the real Lean checker.

    The three inductive obligations must each appear under their own token, and
    every protocol token must be labelled kind=theorem with bound=none: that is
    what distinguishes these results from the trace evaluations in the same
    sealed log. Asserting the runner's source text instead would keep passing if
    the loop that emits the tokens never executed.
    """

    result = _run_lean_runner(tmp_path, PROTOCOL_RUNNER)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "SCOUT_B_PROTOCOL_LEAN_TOOLCHAIN checker=lean version=4.34.0" in (
        result.stdout
    )
    for index, obligation in enumerate(PROTOCOL_OBLIGATIONS, start=1):
        expected = (
            "SCOUT_B_PROTOCOL_LEAN_THEOREM"
            f" theorem=Qwen3Formal.ScoutBProtocol.{obligation}"
            " kind=theorem scope=all-topologies-all-schedules bound=none"
            f" role=obligation-{index}-{obligation} axioms=[] exit=0"
        )
        assert expected in result.stdout, result.stdout

    # The general protocol theorem and the negative that shows its acyclicity
    # hypothesis is load-bearing.
    assert (
        "theorem=Qwen3Formal.ScoutBProtocol."
        "orderAgreementAndAcyclicWaitGraphExcludeBothHazards" in result.stdout
    )
    assert "theorem=Qwen3Formal.ScoutBProtocol.acyclicityIsLoadBearing" in result.stdout
    assert (
        "SCOUT_B_PROTOCOL_LEAN_NEGATIVE proposition=Qwen3Formal.ScoutBProtocol."
        "ControlledInvalidWaitGraphProposition kind=theorem-negative"
        " scope=fixed-instance bound=2 result=rejected exit=1" in result.stdout
    )
    # No token here claims a trace.
    assert "scope=observed-trace" not in result.stdout

    # And each token must carry the vocabulary its RESULT deserves, which is
    # not the same vocabulary for all of them. A blanket `bound=none`
    # assertion used to live here, and it made an overclaim pass a test:
    # `by decide` over one fixed topology whose maxIssues is 2 is not a
    # statement about all schedule lengths.
    theorem_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("SCOUT_B_PROTOCOL_LEAN_THEOREM")
    ]
    assert len(theorem_lines) >= 20, theorem_lines
    labels = {
        line.split("theorem=Qwen3Formal.ScoutBProtocol.")[1].split(" ")[0]: line
        for line in theorem_lines
    }
    general = (
        "idxOfOn_spec",
        "idxOfOn_isSome_of_le",
        "opAtOn_append_of_le",
        "initiation",
        "consecution",
        "sufficiency",
        "invOfReachable",
        "safetyOfReachable",
        "blockerOfStuckPending",
        "noChainInAcyclicRelation",
        "noCircularWaitOfAcyclicWaitGraph",
        "deadlockFreeOfAcyclicWaitGraph",
        "orderAgreementAndAcyclicWaitGraphExcludeBothHazards",
    )
    for name in general:
        assert (
            "kind=theorem scope=all-topologies-all-schedules bound=none" in labels[name]
        ), labels[name]
    # Fixed-instance results, with the instance's own bound reported.
    for name in (
        "cyclicWitnessIsStuck",
        "waitGraphWitnessIsNotVacuous",
        "cyclicWitnessIsWaitClosed",
        "cyclicWitnessHasTwoCycle",
        "cyclicWitnessAdmitsNoRankFunction",
    ):
        assert "kind=witness scope=fixed-instance bound=2" in labels[name], labels[name]
    assert (
        "kind=witness scope=fixed-instance bound=1"
        in labels["commFifoAloneIsNotInductive"]
    ), labels["commFifoAloneIsNotInductive"]
    assert (
        "kind=theorem-negative scope=fixed-instance bound=2"
        in labels["acyclicityIsLoadBearing"]
    ), labels["acyclicityIsLoadBearing"]
    # No general theorem may claim a fixed instance, and no fixed-instance
    # result may claim bound=none.
    for name in general:
        assert "fixed-instance" not in labels[name], labels[name]
    for name in labels:
        if name not in general:
            assert "bound=none" not in labels[name], labels[name]

    # Conditionality must be a field, not free text inside role=. A reader who
    # meets `deadlockFreeOfAcyclicWaitGraph ... bound=none` and nothing else
    # would read an unconditional deadlock-freedom claim.
    for name in (
        "noCircularWaitOfAcyclicWaitGraph",
        "deadlockFreeOfAcyclicWaitGraph",
        "orderAgreementAndAcyclicWaitGraphExcludeBothHazards",
    ):
        assert "conditional=acyclic-wait-for-graph" in labels[name], labels[name]
    assert "conditional=" not in labels["safetyOfReachable"], labels[
        "safetyOfReachable"
    ]


def test_protocol_runner_refuses_a_missing_obligation(tmp_path: Path) -> None:
    """One token per obligation must be enforced, not merely emitted.

    Removing `#print axioms consecution` leaves a module that still compiles
    and still contains the theorem, so a runner that only checked the exit
    status would report success with the consecution result absent. That is
    exactly the silent weakening the per-obligation tokens exist to prevent.
    """

    source = (FORMAL_DIR / "ScoutBInductiveInvariant.lean").read_text()
    assert "#print axioms consecution" in source
    result = _run_lean_runner(
        tmp_path,
        PROTOCOL_RUNNER,
        {
            "ScoutBInductiveInvariant.lean": source.replace(
                "#print axioms consecution\n", ""
            )
        },
    )

    assert result.returncode != 0, result.stdout
    assert "consecution" in result.stderr, result.stderr
    # Initiation precedes consecution in the loop, so its token must already
    # have been emitted: the refusal is specific, not a blanket failure.
    assert "role=obligation-1-initiation" in result.stdout, result.stdout
    assert "role=obligation-3-sufficiency" not in result.stdout, result.stdout


def test_protocol_runner_refuses_a_sorry_in_a_protocol_theorem(
    tmp_path: Path,
) -> None:
    """The axiom gate must actually cover these modules.

    `sufficiency` is the cheapest theorem to replace with `sorry`, and doing so
    makes Lean report `sorryAx` for it and for everything downstream. The runner
    must refuse rather than print an axioms=[] token, and it must not have been
    necessary to widen `formal_classify_lean_valid` to get there.
    """

    source = (FORMAL_DIR / "ScoutBInductiveInvariant.lean").read_text()
    needle = "  ⟨h.2.2.1, h.1, h.2.1, h.2.2.2⟩"
    assert needle in source
    result = _run_lean_runner(
        tmp_path,
        PROTOCOL_RUNNER,
        {"ScoutBInductiveInvariant.lean": source.replace(needle, "  sorry")},
    )

    assert result.returncode != 0, result.stdout
    assert "sufficiency" in result.stderr, result.stderr
    assert "role=obligation-3-sufficiency" not in result.stdout, result.stdout


def test_protocol_negative_must_be_refuted_not_compiled(tmp_path: Path) -> None:
    """A negative that compiles is a failure of the suite, not a success.

    Flipping the controlled proposition to the TRUE reading -- the witness IS
    stuck -- makes ScoutBProtocolInvalid.lean compile. The runner must then
    refuse, because `formal_classify_lean_negative` requires the `decide`
    diagnostic rather than merely a non-zero exit somewhere.
    """

    source = (FORMAL_DIR / "ScoutBProtocolInvalid.lean").read_text()
    needle = "stuckB cyclicTopology cyclicState = false"
    assert needle in source
    result = _run_lean_runner(
        tmp_path,
        PROTOCOL_RUNNER,
        {
            "ScoutBProtocolInvalid.lean": source.replace(
                needle, "stuckB cyclicTopology cyclicState = true"
            )
        },
    )

    assert result.returncode != 0, result.stdout
    assert "did not reject the named proposition" in result.stderr, result.stderr
    assert "SCOUT_B_PROTOCOL_LEAN_NEGATIVE" not in result.stdout, result.stdout


def test_safety_theorem_statement_does_not_mention_the_bound(
    tmp_path: Path,
) -> None:
    """Ticket 13's acceptance criterion, checked by asking Lean for the type.

    `MaxIssues` must be gone from the SAFETY CLAIM, not merely absent from a
    comment. The elaborated type of `safetyOfReachable` is the claim, so this
    prints it with the real Lean and asserts `maxIssues` does not occur in it,
    while asserting it DOES occur in the type of a definition that genuinely
    depends on the bound. Without that second half the test would pass against
    a Lean that printed nothing useful.
    """

    lean = _lean_toolchain()
    work = tmp_path / "statement"
    work.mkdir(parents=True, exist_ok=True)
    for name in (
        "ScoutBProtocol.lean",
        "ScoutBInductiveInvariant.lean",
        "ScoutBWaitGraph.lean",
    ):
        (work / name).write_text((FORMAL_DIR / name).read_text())
    (work / "Statement.lean").write_text(
        "import ScoutBWaitGraph\n"
        "open Qwen3Formal.ScoutBProtocol\n"
        "set_option pp.fullNames true\n"
        "#check @safetyOfReachable\n"
        "#print allDoneB\n"
        "#check @orderAgreementAndAcyclicWaitGraphExcludeBothHazards\n"
    )
    env = {**os.environ, "LEAN_PATH": "."}
    for name in (
        "ScoutBProtocol",
        "ScoutBInductiveInvariant",
        "ScoutBWaitGraph",
    ):
        built = subprocess.run(
            [lean, "-o", f"{name}.olean", f"{name}.lean"],
            cwd=work,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
        assert built.returncode == 0, built.stdout

    printed = subprocess.run(
        [lean, "Statement.lean"],
        cwd=work,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
    assert printed.returncode == 0, printed.stdout
    blocks = printed.stdout.split("safetyOfReachable")
    assert len(blocks) >= 2, printed.stdout
    safety_type, rest = blocks[1].split("allDoneB", 1)
    # The claim: the elaborated statement names Topology, State and Reachable
    # and does not name the bound.
    assert "maxIssues" not in safety_type, printed.stdout
    assert "Topology" in safety_type, printed.stdout
    assert "Reachable" in safety_type, printed.stdout
    # Non-vacuity: this Lean invocation does surface maxIssues where it occurs,
    # so the absence above is a property of the statement rather than of the
    # printer. allDoneB is a definition of this encoding that genuinely uses
    # the bound.
    assert "maxIssues" in rest, printed.stdout

    # The flagship theorem must deliver the whole of Safety, not just
    # RendezvousOpAgreement. That conjunct compares two Options and holds
    # vacuously where a running communicator is short of a member's issue;
    # RendezvousMembership, which travels inside Safety, is what forces both
    # sides to `some`. Projecting it out would make the theorem read stronger
    # than it is.
    flagship = printed.stdout.split(
        "orderAgreementAndAcyclicWaitGraphExcludeBothHazards"
    )[-1]
    assert "Safety" in flagship, printed.stdout
    assert "maxIssues" not in flagship, printed.stdout


def test_observed_trace_lean_tokens_are_labelled_as_evaluations(
    tmp_path: Path,
) -> None:
    """The relabelling must reach the sealed log, by running a real runner.

    Scout A's Lean runner is the cheap end-to-end witness for the evaluation
    label: its modules are small, so this asserts the emitted token text rather
    than the printf that produces it.
    """

    result = _run_lean_runner(tmp_path, SCOUT_A_LEAN_RUNNER)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "SCOUT_A_LEAN_VALID theorem=Qwen3Formal.ScoutA.validLifecycle"
        " kind=evaluation scope=observed-trace axioms=[] exit=0" in result.stdout
    )
    assert (
        "SCOUT_A_LEAN_NEGATIVE proposition=Qwen3Formal.ScoutA."
        "ControlledInvalidProposition kind=evaluation scope=observed-trace"
        " result=rejected exit=1" in result.stdout
    )
    assert "kind=theorem" not in result.stdout, result.stdout


def test_every_lean_token_declares_evaluation_or_theorem_exactly_once() -> None:
    """Uniformity of the label vocabulary across all Lean runners.

    The two behavioural tests above are what establish that each label reaches
    the sealed log; this one guards against a SIXTH token appearing later with
    no label at all, which no single run would reveal. Running every Lean runner
    to check that would cost the 92s Scout B trace check and prove nothing more
    about the label, so this one reads the printf lines.
    """

    # Two properties, neither keyed on `printf '`. Keying on that used to let a
    # token emitted by `echo`, or assembled in a variable, escape the very
    # check this test exists to make -- and the protocol runner now does take
    # its label from a variable, which the old form would have passed silently
    # rather than examined.
    #
    #   1. every `kind=` written anywhere in a runner, outside a comment, is one
    #      of the four recognised vocabularies -- so a fifth cannot appear
    #      unannounced;
    #   2. every line that emits a token either carries a vocabulary inline or
    #      interpolates one, so no token reaches the log unlabelled.
    #
    # What the emitted text actually says is asserted by the two tests above,
    # which run the runners.
    vocabularies = (
        "kind=evaluation scope=observed-trace",
        "kind=theorem scope=all-topologies-all-schedules bound=none",
        "kind=witness scope=fixed-instance bound=",
        "kind=theorem-negative",
        # The toolchain smoke is none of those: it proves the pinned Lean runs
        # and rejects a false proposition, over no trace and no protocol.
        "kind=smoke scope=toolchain",
    )
    token_name = re.compile(
        r"""['"]((?:SCOUT_[A-Z0-9_]*|LEAN)_"""
        r"""(?:VALID|NEGATIVE|MUTATION|THEOREM|TOOLCHAIN))\b"""
    )
    runners = sorted(FORMAL_DIR.glob("run_lean_*.sh"))
    assert len(runners) >= 3, runners
    labelled = 0
    interpolated = 0
    seen_names: set[str] = set()
    for runner in runners:
        for line in runner.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "kind=" in stripped:
                matched = [v for v in vocabularies if v in stripped]
                assert len(matched) == 1, (runner.name, stripped)
            match = token_name.search(stripped)
            if match is None:
                continue
            seen_names.add(match.group(1))
            if match.group(1).endswith("_TOOLCHAIN"):
                continue
            if "kind=" in stripped:
                labelled += 1
                continue
            # Label taken from a variable. The label strings themselves are
            # covered by property 1 above, and the emitted result by the
            # behavioural tests.
            assert "%s" in stripped, (runner.name, stripped)
            interpolated += 1
    # Guard against a vacuous pass if the emission shape ever changes: the
    # token names the suites actually print must all have been seen, and both
    # emission shapes must have been exercised.
    assert labelled >= 6, labelled
    assert interpolated >= 1, interpolated
    assert {
        "SCOUT_A_LEAN_VALID",
        "SCOUT_B_LEAN_VALID",
        "SCOUT_B_PROTOCOL_LEAN_THEOREM",
        "LEAN_VALID",
    } <= seen_names, sorted(seen_names)


def test_protocol_target_is_wired_into_tier0_and_the_sealed_suite() -> None:
    """The inductive check belongs in the CPU-only suite AND in the gate.

    Ticket 13 asks for it in the trace-free tier, and tier 0 may only run checks
    the gate also runs, so both memberships are asserted together.
    """

    build = BUILD_FILE.read_text()
    _, sh_tests, suites = _parse_build_targets(build)

    assert "lean_scout_b_protocol_test" in sh_tests
    for suite in ("tier0_formal_tests", "scout_b_formal_tests"):
        members = {label.lstrip(":") for label in suites[suite]}
        assert "lean_scout_b_protocol_test" in members, (suite, members)

    inputs = sh_tests["lean_scout_b_protocol_test"]
    assert {
        "ScoutBProtocol.lean",
        "ScoutBInductiveInvariant.lean",
        "ScoutBWaitGraph.lean",
        "ScoutBProtocolInvalid.lean",
    } <= {label.lstrip(":") for label in inputs}, inputs
    # It must not read a generated facts module: the theorems are about the
    # protocol, so a trace dependency here would be a category error as well as
    # a tier-0 violation.
    assert not [label for label in inputs if "Facts" in label], inputs


def test_protocol_modules_avoid_the_tactics_that_introduce_axioms() -> None:
    """Regression guard for the axiom discipline, at the source level.

    `omega` (propext, Quot.sound) and `simp` (propext) both compile fine and
    both make `#print axioms` non-empty, so a later edit reaching for either
    would fail the runner with a diagnostic about axioms rather than about the
    tactic. Naming the cause here makes that failure legible. The behavioural
    half is test_protocol_runner_refuses_a_sorry_in_a_protocol_theorem.
    """

    # Block comments first -- these files explain the discipline in prose, and
    # reading that prose as code is why the previous regexes had to be narrow.
    # Then `--` line comments. What is left is code, and the check is on
    # identifiers rather than on a handful of spellings: bare `simp` on its own
    # line, `simp_all` and `simp_arith` all escaped the earlier version.
    banned = re.compile(
        r"\b(?:omega|simp|simp_all|simp_arith|simp_rw|simpa|norm_num|aesop"
        r"|linarith|nlinarith|field_simp|funext|propext|sorry|admit)\b"
        r"|\bQuot\.sound\b|\bClassical\."
    )
    checked = 0
    for name in (
        "ScoutBProtocol.lean",
        "ScoutBInductiveInvariant.lean",
        "ScoutBWaitGraph.lean",
        "ScoutBProtocolInvalid.lean",
    ):
        text = re.sub(r"/-.*?-/", "", (FORMAL_DIR / name).read_text(), flags=re.S)
        for line in text.splitlines():
            code = line.split("--")[0]
            assert not banned.search(code), (name, line)
            checked += 1
    # Non-vacuity: stripping must not have eaten the files, and the regex must
    # fire on the things it is for.
    assert checked > 1000, checked
    assert banned.search("  exact (by omega)")
    assert banned.search("  simp")
    assert banned.search("  simp_all")
    assert banned.search("  exact Classical.em _")
    assert not banned.search("  exact ifPos rfl")


_TOPOLOGY_WITHOUT_COMMS_COMPLETE = """import ScoutBProtocol
open Qwen3Formal.ScoutBProtocol

def incompleteComms : Topology Unit Unit Unit Unit where
  ranks := [()]
  comms := [()]
  member := fun _ _ => true
  opsAll := [()]
  admissibleOp := fun _ _ => true
  streamOf := fun _ => ()
  maxIssues := 1
  requireMatchedIssueOrder := true
  requireUniformProgramOps := true
  requireUniformProgramComms := true
  requireStreamOrder := true
  membersInRanks := by
    intro _ r _
    cases r
    exact List.Mem.head _
  admissibleInOps := by
    intro _ op _
    cases op
    exact List.Mem.head _
"""


def test_topology_forces_the_comms_completeness_obligation(tmp_path: Path) -> None:
    """`stuckB` and `allDoneB` quantify only over `T.comms`, so a `comms` list
    that omitted a live communicator would make them quantify over too little.

    `Topology.commsComplete` turns that into a construction-time obligation
    rather than an unstated side condition. This checks it by asking the real
    Lean to elaborate a Topology literal that omits the field, which must fail
    and must name it. A grep for the field would pass against a field that
    Lean did not actually require.
    """

    lean = _lean_toolchain()
    work = tmp_path / "obligation"
    work.mkdir(parents=True, exist_ok=True)
    (work / "ScoutBProtocol.lean").write_text(
        (FORMAL_DIR / "ScoutBProtocol.lean").read_text()
    )
    env = {**os.environ, "LEAN_PATH": "."}
    built = subprocess.run(
        [lean, "-o", "ScoutBProtocol.olean", "ScoutBProtocol.lean"],
        cwd=work,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
    assert built.returncode == 0, built.stdout

    (work / "Incomplete.lean").write_text(_TOPOLOGY_WITHOUT_COMMS_COMPLETE)
    result = subprocess.run(
        [lean, "Incomplete.lean"],
        cwd=work,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
    )

    assert result.returncode != 0, result.stdout
    assert "commsComplete" in result.stdout, result.stdout
    # The other two carrier obligations must be satisfied by this literal, so
    # the refusal is specific to the field under test.
    assert "membersInRanks" not in result.stdout, result.stdout
    assert "admissibleInOps" not in result.stdout, result.stdout


def test_dpxtp_safety_token_names_every_invariant_it_checked() -> None:
    """Six invariants are checked in one TLC run; the token named none of them.

    `StuckImpliesAllDone` in particular reached the sealed log only inside an
    aggregate success line, so a reader could not tell it had been checked at
    all. The fix derives the list from the cfg rather than from a hand-kept
    string, and this runs that derivation -- the real `formal_cfg_invariants`
    from checker_contract.sh, over the real cfg -- and compares it against
    `_cfg_invariants`, the independent Python parse this file already had for
    the liveness contracts.

    The wiring of that value into the SAFETY token is asserted on the runner's
    source rather than by running it: the Scout B model runner takes about 290s
    for its eleven TLC checks, and both `tier0_formal_tests` and
    `scout_b_formal_tests` already execute it, so a unit test that repeated it
    would buy a fourth execution of the same code rather than new evidence.
    """

    cfg = FORMAL_DIR / "ScoutBModel.cfg"
    parsed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; formal_cfg_invariants "$2"',
            "invariants-test",
            str(CHECKER_CONTRACT),
            str(cfg),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert parsed.returncode == 0, parsed.stdout
    from_bash = parsed.stdout.strip().split(",")
    expected = _cfg_invariants(cfg.read_text())
    assert from_bash == expected, (from_bash, expected)
    assert "StuckImpliesAllDone" in from_bash, from_bash
    assert "DeadlockFreedom" in from_bash, from_bash
    assert len(from_bash) == 6, from_bash

    # A cfg that declares none must yield nothing rather than a stale list, so
    # an empty result cannot be mistaken for "all six".
    empty = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; formal_cfg_invariants "$2"',
            "invariants-test",
            str(CHECKER_CONTRACT),
            str(FORMAL_DIR / "ScoutBModelStreamShape.cfg"),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert empty.returncode == 0, empty.stdout
    assert empty.stdout.strip() != parsed.stdout.strip(), empty.stdout

    runner = (FORMAL_DIR / "run_tlc_scout_b_model.sh").read_text()
    assert 'formal_cfg_invariants "${fixture_dir}/ScoutBModel.cfg"' in runner
    assert "invariants=%s" in runner
    assert "${safety_invariants}" in runner


# Tiny hand-written placement facts, so the structural predicates are exercised
# on inputs they must reject as well as on the observed run they accept. The
# probe asserts the NEGATION of each predicate on a bad input, so one clean TLC
# run proves several rejections instead of stopping at the first.
_PLACEMENT_PROBE = """\
------------------------------ MODULE PlacementProbe ------------------------------
EXTENDS Naturals, Sequences, FiniteSets, TLC, ScoutDistributed

VARIABLE cursor
vars == <<cursor>>
Init == cursor = 0
Next == UNCHANGED cursor
Spec == Init /\\ [][Next]_vars

Degrees == ("dp_shard" :> 2) @@ ("tp" :> 2)
Names == <<"w">>
MeshAxes == ("w" :> <<"dp_shard", "tp">>)
Ids == <<"p0", "p1">>
Parameter == ("p0" :> "w") @@ ("p1" :> "w")
Axis == ("p0" :> "dp_shard") @@ ("p1" :> "tp")

\\* Good: 8 on dim 0 over dp_shard=2 composed with tp=2, local 2.
GoodKind == ("p0" :> "shard") @@ ("p1" :> "shard")
GoodStrided == ("p0" :> TRUE) @@ ("p1" :> FALSE)
GoodDim == ("p0" :> 0) @@ ("p1" :> 0)
GoodGlobal == ("w" :> <<8>>)
GoodLocal == ("w" :> <<2>>)

WellFormed(kind, strided, dim, global, local) ==
  ParameterPlacementWellFormed(
    Names, MeshAxes, global, local, Ids, Parameter, Axis, kind, strided, dim,
    Degrees)

Divides(kind, dim, global) ==
  ShardedDimDividesAxisDegree(
    Ids, Parameter, Axis, kind, dim, global, Degrees)

LocalShape(kind, dim, global, local) ==
  LocalShapeReflectsSharding(
    Names, global, local, Ids, Parameter, Axis, kind, dim, Degrees)

Strided(kind, strided, dim) ==
  StridedShardIsAnOuterComposedShard(
    MeshAxes, Ids, Parameter, Axis, kind, strided, dim)

\\* Not vacuous: the good input satisfies every predicate.
GoodInputIsAccepted ==
  /\\ WellFormed(GoodKind, GoodStrided, GoodDim, GoodGlobal, GoodLocal)
  /\\ Divides(GoodKind, GoodDim, GoodGlobal)
  /\\ LocalShape(GoodKind, GoodDim, GoodGlobal, GoodLocal)
  /\\ Strided(GoodKind, GoodStrided, GoodDim)
  /\\ NoPartialParameterPlacement(Ids, GoodKind)

\\* 3 is not divisible by the dp_shard degree of 2.
IndivisibleIsRejected ==
  ~Divides(GoodKind, GoodDim, ("w" :> <<3>>))

\\* An unreduced placement on a parameter the optimizer steps.
PartialIsRejected ==
  ~NoPartialParameterPlacement(Ids, ("p0" :> "partial") @@ ("p1" :> "shard"))

\\* The outer axis shards the same dim as the inner axis but is not strided.
MissingStridedFlagIsRejected ==
  ~Strided(GoodKind, ("p0" :> FALSE) @@ ("p1" :> FALSE), GoodDim)

\\* The inner axis carries the flag instead of the outer one.
InnerStridedFlagIsRejected ==
  ~Strided(GoodKind, ("p0" :> FALSE) @@ ("p1" :> TRUE), GoodDim)

\\* A dim nothing shards must keep its global size.
UnshardedDimMustNotShrinkIsRejected ==
  ~LocalShape(
     ("p0" :> "replicate") @@ ("p1" :> "replicate"),
     [x \\in {} |-> x], GoodGlobal, GoodLocal)

\\* A replicate placement carrying a shard dim breaks the restricted domain.
ReplicateWithADimIsRejected ==
  ~WellFormed(
     ("p0" :> "shard") @@ ("p1" :> "replicate"),
     GoodStrided, GoodDim, GoodGlobal, ("w" :> <<4>>))

=============================================================================
"""

_PLACEMENT_PROBE_CFG = """\
SPECIFICATION Spec

INVARIANTS
  GoodInputIsAccepted
  IndivisibleIsRejected
  PartialIsRejected
  MissingStridedFlagIsRejected
  InnerStridedFlagIsRejected
  UnshardedDimMustNotShrinkIsRejected
  ReplicateWithADimIsRejected
"""


def test_placement_predicates_reject_the_layouts_they_are_for(
    tmp_path: Path,
) -> None:
    """Run the real predicates over inputs they must refuse.

    The observed run satisfies all of them, which on its own says nothing about
    what they would catch. This applies the shipped ScoutDistributed
    definitions -- not a copy -- to tiny hand-written facts and asserts the
    NEGATION of each predicate on a bad layout, plus acceptance of a good one so
    the rejections are not vacuous.
    """

    status, output = _run_tlc(
        tmp_path,
        "PlacementProbe",
        _PLACEMENT_PROBE_CFG,
        fixtures=("ScoutDistributed.tla",),
        files={"PlacementProbe.tla": _PLACEMENT_PROBE},
    )

    assert (
        _run_classifier("formal_classify_tlc_valid", status, output).returncode == 0
    ), (status, output)


def test_partial_placement_negative_is_rejected_for_the_right_reason(
    tmp_path: Path,
) -> None:
    """The injected Partial must violate exactly the Partial invariant.

    Run over the shipped facts, not grepped: a control that tripped the
    well-formedness check instead would prove nothing about whether an
    unreduced placement is caught, so the module also asserts the structural
    invariant still holds under the override.
    """

    status, output = _run_tlc(
        tmp_path,
        "ScoutBPlacementPartialInvalid",
        (FORMAL_DIR / "ScoutBPlacementPartialInvalid.cfg").read_text(),
        fixtures=(
            "ScoutDistributed.tla",
            "ScoutBFacts.tla",
            "ScoutBPlacementPartialInvalid.tla",
        ),
    )

    assert (
        _run_classifier(
            "formal_classify_tlc_transition_negative",
            status,
            output,
            "ScoutBNoPartialAtOptimizer",
        ).returncode
        == 0
    ), (status, output)


_B_FACTS_WITHOUT_A_DP_SHARD_PLACEMENT = """\
------------------------------ MODULE ScoutBFacts ------------------------------
EXTENDS Naturals, Sequences, FiniteSets, TLC

RankSet == {0, 1, 2, 3}
EventIds == <<"event:r0:1">>
EventRank == ("event:r0:1" :> 0)
EventKind == ("event:r0:1" :> "step.started")
EventOrder == ("event:r0:1" :> 1)

ParameterNames == <<"w">>
ParameterMeshAxes == ("w" :> <<"dp_shard", "tp">>)
ParameterGlobalShape == ("w" :> <<8>>)
ParameterLocalShape == ("w" :> <<8>>)
PlacementIds == <<"p0", "p1">>
PlacementParameter == ("p0" :> "w") @@ ("p1" :> "w")
PlacementAxis == ("p0" :> "dp_shard") @@ ("p1" :> "tp")
PlacementKind == ("p0" :> "replicate") @@ ("p1" :> "replicate")
PlacementStrided == ("p0" :> FALSE) @@ ("p1" :> FALSE)
PlacementShardDim == [x \\in {} |-> x]
MeshAxisDegree == ("dp_shard" :> 2) @@ ("tp" :> 2)
PlacementSchemaDigest == "digest"
PlacementSchemaDigestByRank ==
  (0 :> "digest") @@ (1 :> "digest") @@ (2 :> "digest") @@ (3 :> "digest")

=============================================================================
"""


_PARTIAL_GUARD = """\
MutatedPlacement ==
  IF ThereIsADpShardPlacement
  THEN CHOOSE placement \\in DpShardPlacements :
         \\A other \\in DpShardPlacements :
           PlacementPosition[placement] <= PlacementPosition[other]
  ELSE \"\""""

_PARTIAL_UNGUARDED = """\
MutatedPlacement ==
  CHOOSE placement \\in DpShardPlacements :
    \\A other \\in DpShardPlacements :
      PlacementPosition[placement] <= PlacementPosition[other]"""

_ISSUE_ORDER_GUARD = """\
MutatedWork ==
  IF Rank0HasACollective
  THEN CHOOSE work \\in Rank0Works :
         \\A other \\in Rank0Works :
           CollectiveIssueOrder[work] <= CollectiveIssueOrder[other]
  ELSE \"\""""

_ISSUE_ORDER_UNGUARDED = """\
MutatedWork ==
  CHOOSE work \\in Rank0Works :
    \\A other \\in Rank0Works :
      CollectiveIssueOrder[work] <= CollectiveIssueOrder[other]"""


def test_partial_placement_control_names_its_missing_target_instead_of_crashing(
    tmp_path: Path,
) -> None:
    """A degenerate facts module must be reported, not crashed on.

    What does the work here is the CFG ORDER, not the CHOOSE guard: these are
    constant expressions, so TLC reports the first listed invariant that is
    FALSE and never forces the CHOOSE. An earlier version of this test claimed
    the guard and would have passed with the guard removed;
    test_the_partial_injection_guard_is_load_bearing covers the guard, and this
    one covers the sentinel reaching the classifier.
    """

    status, output = _run_tlc(
        tmp_path,
        "ScoutBPlacementPartialInvalid",
        (FORMAL_DIR / "ScoutBPlacementPartialInvalid.cfg").read_text(),
        fixtures=("ScoutDistributed.tla", "ScoutBPlacementPartialInvalid.tla"),
        files={"ScoutBFacts.tla": _B_FACTS_WITHOUT_A_DP_SHARD_PLACEMENT},
    )

    assert "Attempted to compute the value of an expression of form" not in output
    assert (
        _run_classifier(
            "formal_classify_tlc_constant_false",
            status,
            output,
            "ThereIsADpShardPlacement",
        ).returncode
        == 0
    ), (status, output)


_PAYLOAD_SIZE_GUARD = """\
MutatedWork ==
  IF ThereIsASizedWorkWithAPeer
  THEN CHOOSE work \\in ScalableWorks :
         \\A other \\in ScalableWorks :
           WorkPosition[work] <= WorkPosition[other]
  ELSE \"\""""

_PAYLOAD_SIZE_UNGUARDED = """\
MutatedWork ==
  CHOOSE work \\in ScalableWorks :
    \\A other \\in ScalableWorks :
      WorkPosition[work] <= WorkPosition[other]"""

_PAYLOAD_OPERATION_GUARD = """\
MutatedWork ==
  IF ThereIsAVolumeChangingCollective
  THEN CHOOSE work \\in VolumeChangingWorks :
         \\A other \\in VolumeChangingWorks :
           WorkPosition[work] <= WorkPosition[other]
  ELSE \"\""""

_PAYLOAD_OPERATION_UNGUARDED = """\
MutatedWork ==
  CHOOSE work \\in VolumeChangingWorks :
    \\A other \\in VolumeChangingWorks :
      WorkPosition[work] <= WorkPosition[other]"""


@pytest.mark.parametrize(
    ("module", "facts", "guard", "unguarded", "sentinel", "named"),
    [
        (
            "ScoutBPlacementPartialInvalid",
            "_B_FACTS_WITHOUT_A_DP_SHARD_PLACEMENT",
            _PARTIAL_GUARD,
            _PARTIAL_UNGUARDED,
            "ThereIsADpShardPlacement",
            "MutationIsIsolated",
        ),
        (
            "ScoutBIssueOrderInvalid",
            "_B_FACTS_WITHOUT_RANK_ZERO",
            _ISSUE_ORDER_GUARD,
            _ISSUE_ORDER_UNGUARDED,
            "Rank0HasACollective",
            "MutationIsIsolated",
        ),
        (
            "ScoutBPayloadSizeInvalid",
            "_B_FACTS_WITHOUT_AN_ELIGIBLE_PAYLOAD",
            _PAYLOAD_SIZE_GUARD,
            _PAYLOAD_SIZE_UNGUARDED,
            "ThereIsASizedWorkWithAPeer",
            "MutationIsIsolated",
        ),
        (
            "ScoutBPayloadOperationInvalid",
            "_B_FACTS_WITHOUT_AN_ELIGIBLE_PAYLOAD",
            _PAYLOAD_OPERATION_GUARD,
            _PAYLOAD_OPERATION_UNGUARDED,
            "ThereIsAVolumeChangingCollective",
            "MutationIsIsolated",
        ),
    ],
    ids=[
        "partial_placement",
        "issue_order",
        "payload_size",
        "payload_operation",
    ],
)
def test_the_derived_mutation_guard_is_load_bearing(
    tmp_path: Path,
    module: str,
    facts: str,
    guard: str,
    unguarded: str,
    sentinel: str,
    named: str,
) -> None:
    """Find the configuration where the ELSE sentinel actually decides the outcome.

    With the sentinel invariant listed first it does not: guarded and unguarded
    both report the sentinel FALSE, so a test over the shipped cfg cannot fail
    when the guard is deleted. Remove the sentinel from the cfg and the guard
    becomes decisive -- guarded gives a named invariant, unguarded abandons the
    search with an evaluation error. Both directions are asserted, so deleting
    the guard fails this test and so does a guard that no longer degrades to a
    named result.
    """

    facts_text = {
        "_B_FACTS_WITHOUT_A_DP_SHARD_PLACEMENT": _B_FACTS_WITHOUT_A_DP_SHARD_PLACEMENT,
        "_B_FACTS_WITHOUT_RANK_ZERO": _B_FACTS_WITHOUT_RANK_ZERO,
        "_B_FACTS_WITHOUT_AN_ELIGIBLE_PAYLOAD": (_B_FACTS_WITHOUT_AN_ELIGIBLE_PAYLOAD),
    }[facts]
    source = (FORMAL_DIR / f"{module}.tla").read_text()
    assert source.count(guard) == 1, "guard anchor drifted from the module"
    shipped_cfg = (FORMAL_DIR / f"{module}.cfg").read_text()
    # The cfg without its sentinel, so nothing is reported ahead of the CHOOSE.
    without_sentinel = "".join(
        line for line in shipped_cfg.splitlines(keepends=True) if sentinel not in line
    )
    assert sentinel not in without_sentinel

    guarded_status, guarded_output = _run_tlc(
        tmp_path / "guarded",
        module,
        without_sentinel,
        fixtures=("ScoutDistributed.tla", f"{module}.tla"),
        files={"ScoutBFacts.tla": facts_text},
    )
    assert (
        _run_classifier(
            "formal_classify_tlc_constant_false",
            guarded_status,
            guarded_output,
            named,
        ).returncode
        == 0
    ), (guarded_status, guarded_output)

    stripped_status, stripped_output = _run_tlc(
        tmp_path / "unguarded",
        module,
        without_sentinel,
        fixtures=("ScoutDistributed.tla",),
        files={
            "ScoutBFacts.tla": facts_text,
            f"{module}.tla": source.replace(guard, unguarded),
        },
    )
    assert "CHOOSE x \\in S: P, but no element of S satisfied P" in stripped_output, (
        stripped_status,
        stripped_output,
    )
    # _run_predicate, not _run_classifier: this one takes only the output, and
    # the file's own helper docstring warns that passing the wrong shape
    # silently tests nothing.
    assert (
        _run_predicate("formal_has_infrastructure_error", stripped_output).returncode
        == 0
    ), (stripped_status, stripped_output)
    assert stripped_status == 75, stripped_status


@pytest.mark.parametrize(
    ("module", "first", "last"),
    [
        (
            "ScoutBPlacementPartialInvalid",
            "ThereIsADpShardPlacement",
            "ScoutBNoPartialAtOptimizer",
        ),
        (
            "ScoutBIssueOrderInvalid",
            "Rank0HasACollective",
            "ScoutBPerCommunicatorIssueOrder",
        ),
        (
            "ScoutBPayloadSizeInvalid",
            "ThereIsASizedWorkWithAPeer",
            "ScoutBCollectivePayloadAgreement",
        ),
        (
            "ScoutBPayloadOperationInvalid",
            "ThereIsAVolumeChangingCollective",
            "ScoutBCollectivePayloadSizeRelation",
        ),
    ],
    ids=[
        "partial_placement",
        "issue_order",
        "payload_size",
        "payload_operation",
    ],
)
def test_placement_negative_configurations_pin_their_invariant_order(
    module: str, first: str, last: str
) -> None:
    """TLC reports only the first failing invariant, so the order is a contract.

    The sentinel has to be first or the missing-target control is reported
    against something else; the property under test has to be last or the
    isolation invariants are never evaluated and the negative goes green with an
    unproven mutation. Derived with the real formal_cfg_invariants, so this
    agrees with what the runners parse.
    """

    parsed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; formal_cfg_invariants "$2"',
            "invariants-test",
            str(CHECKER_CONTRACT),
            str(FORMAL_DIR / f"{module}.cfg"),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert parsed.returncode == 0, parsed.stdout
    listed = parsed.stdout.strip().split(",")

    assert listed[0] == first, listed
    assert listed[-1] == last, listed
    assert len(listed) >= 3, listed


def test_scout_b_tokens_name_the_invariants_they_checked() -> None:
    """A count is not a name; 19 invariants reached the log as none.

    The DPxTP model runner already derives its list with the shared
    formal_cfg_invariants, and this holds the observed-facts runner to the same
    convention, for the valid run and for both derived negatives -- where the
    survivor set is the whole attribution. Asserted on the runner source rather
    than by executing it: the three cfg derivations are exercised for real by
    test_placement_negative_configurations_pin_their_invariant_order and by the
    suite itself, so running the 11-second target again here would buy a repeat
    rather than new evidence.
    """

    runner = (FORMAL_DIR / "run_tlc_scout_b.sh").read_text()

    for cfg in (
        "ScoutBValid.cfg",
        "ScoutBIssueOrderInvalid.cfg",
        "ScoutBPlacementPartialInvalid.cfg",
    ):
        assert f'formal_cfg_invariants "${{fixture_dir}}/{cfg}"' in runner, cfg
    # The two payload negatives share one loop, so their cfg name is a variable.
    # The loop must still derive the list the same way, and must name both
    # modules, or a negative could run with its invariants unreported.
    assert (
        'formal_cfg_invariants "${fixture_dir}/${payload_module}.cfg"' in runner
    ), runner
    for module in ("ScoutBPayloadSizeInvalid", "ScoutBPayloadOperationInvalid"):
        assert module in runner, module
    for token in (
        "SCOUT_B_TLA_VALID",
        "SCOUT_B_TLA_ISSUE_ORDER_NEGATIVE",
        "SCOUT_B_TLA_PARTIAL_PLACEMENT_NEGATIVE",
        "SCOUT_B_TLA_%s_NEGATIVE",
    ):
        # Stripped, because the payload token is emitted inside a loop and is
        # therefore indented.
        line = next(
            candidate.strip()
            for candidate in runner.splitlines()
            if candidate.strip().startswith(f"printf '{token} ")
        )
        assert "invariants=%s" in line, line
    # Fails closed rather than printing an empty list, which would read as
    # "no invariants were checked" and classify green.
    # One per literal cfg plus one shared by the payload loop.
    assert runner.count("could not read the invariant list out of") == 4, runner


def test_every_scout_b_valid_definition_is_bound_as_an_invariant() -> None:
    """An invariant defined but never listed checks nothing.

    The placement work adds eight named invariants at once, which is exactly the
    situation where one silently fails to reach the configuration. Parsed from
    both files rather than searched for a known name, so a future addition is
    covered too.
    """

    module = (FORMAL_DIR / "ScoutBValid.tla").read_text()

    defined = {
        line.split(" ==")[0]
        for line in module.splitlines()
        if line.startswith("ScoutB") and " ==" in line
    }
    # The real derivation from checker_contract.sh over the real cfg, so this
    # agrees with what the runners report rather than with a second parser.
    parsed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; formal_cfg_invariants "$2"',
            "invariants-test",
            str(CHECKER_CONTRACT),
            str(FORMAL_DIR / "ScoutBValid.cfg"),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert parsed.returncode == 0, parsed.stdout
    listed = set(parsed.stdout.strip().split(","))

    assert len(defined) == 23, sorted(defined)
    assert defined == listed, sorted(defined ^ listed)
    for invariant in (
        "ScoutBPlacementValid",
        "ScoutBPlacementSchemaAgrees",
        "ScoutBMeshAxisDegree",
        "ScoutBParameterPlacementWellFormed",
        "ScoutBPlacementBooleansAgree",
        "ScoutBNoPartialAtOptimizer",
        "ScoutBShardedDimDividesAxisDegree",
        "ScoutBLocalShapeReflectsSharding",
        "ScoutBStridedShardComposition",
        "ScoutBCollectivePayloadWellFormed",
        "ScoutBPayloadCommKey",
        "ScoutBCollectivePayloadAgreement",
        "ScoutBCollectivePayloadSizeRelation",
    ):
        assert invariant in listed, invariant


# Tiny hand-written payload facts, so the payload predicates are exercised on
# inputs they must reject as well as on the observed run they accept. Two works
# of one two-member collective, plus a second collective on the same
# communicator, which is what the keying property needs to be non-trivial. The
# probe asserts the NEGATION of each predicate on a bad input, so one clean TLC
# run proves several rejections instead of stopping at the first.
_PAYLOAD_PROBE = """\
------------------------------ MODULE PayloadProbe ------------------------------
EXTENDS Naturals, Sequences, FiniteSets, TLC, ScoutDistributed

VARIABLE cursor
vars == <<cursor>>
Init == cursor = 0
Next == UNCHANGED cursor
Spec == Init /\\ [][Next]_vars

Works == <<"w0", "w1">>
Ids == ("w0" :> "tp:0,1:mesh_tp:seq1:all_gather")
         @@ ("w1" :> "tp:0,1:mesh_tp:seq1:all_gather")
Comm == ("w0" :> "tp:0,1:mesh_tp") @@ ("w1" :> "tp:0,1:mesh_tp")
Members == ("w0" :> {0, 1}) @@ ("w1" :> {0, 1})
Operation == ("w0" :> "all_gather") @@ ("w1" :> "all_gather")
Relation == ("w0" :> "output_is_member_count_times_input")
              @@ ("w1" :> "output_is_member_count_times_input")

\\* An all-gather of one 4x8 tensor per member into a 8x8 output: two members,
\\* so the output volume is twice the input volume. Plus a 0-dim tensor, which
\\* has no dimensions and one element, because the observed run can contain one.
InSizes == ("w0" :> << <<4, 8>>, <<>> >>) @@ ("w1" :> << <<4, 8>>, <<>> >>)
OutSizes == ("w0" :> << <<8, 8>>, <<2>> >>) @@ ("w1" :> << <<8, 8>>, <<2>> >>)
InDtypes == ("w0" :> <<"BFloat16", "BFloat16">>)
              @@ ("w1" :> <<"BFloat16", "BFloat16">>)
OutDtypes == ("w0" :> <<"BFloat16", "BFloat16">>)
               @@ ("w1" :> <<"BFloat16", "BFloat16">>)
InElements == ("w0" :> 33) @@ ("w1" :> 33)
OutElements == ("w0" :> 66) @@ ("w1" :> 66)

WellFormed(inSizes, outSizes, inDtypes, outDtypes, inElements, outElements) ==
  CollectivePayloadWellFormed(
    Works, Comm, inSizes, outSizes, inDtypes, outDtypes, inElements,
    outElements)

Agreement(inDtypes, outDtypes, inElements, outElements) ==
  CollectivePayloadAgreement(
    Works, Ids, inDtypes, outDtypes, inElements, outElements)

Relates(operation, members, relation, inElements, outElements) ==
  CollectivePayloadSizeRelationHolds(
    Works, operation, members, relation, inElements, outElements)

CommKey(comm, members) ==
  PayloadCommKeyIsCommunicatorIdentity(Works, comm, comm, members)

\\* Not vacuous: the good input satisfies every predicate. The element counts
\\* are 4*8 + 1 = 33 and 8*8 + 2 = 66, so the product of no dimensions is 1.
GoodInputIsAccepted ==
  /\\ WellFormed(
       InSizes, OutSizes, InDtypes, OutDtypes, InElements, OutElements)
  /\\ Agreement(InDtypes, OutDtypes, InElements, OutElements)
  /\\ Relates(Operation, Members, Relation, InElements, OutElements)
  /\\ CommKey(Comm, Members)
  /\\ ProductOfSizes(<<>>) = 1
  /\\ TotalElements(<< <<4, 8>>, <<>> >>) = 33

\\* An element count the shapes do not support. This is the exporter's own
\\* arithmetic being recomputed rather than trusted.
DerivedElementCountIsRecomputed ==
  ~WellFormed(
     InSizes, OutSizes, InDtypes, OutDtypes,
     ("w0" :> 34) @@ ("w1" :> 33), OutElements)

\\* One shape per tensor, one dtype name per tensor: a dtype list that does not
\\* pair up with the shapes is not payload identity.
UnpairedDtypeListIsRejected ==
  ~WellFormed(
     InSizes, OutSizes, ("w0" :> <<"BFloat16">>) @@ ("w1" :> <<"BFloat16">>),
     OutDtypes, InElements, OutElements)

\\* A zero dimension would make a volume relation hold vacuously.
ZeroDimensionIsRejected ==
  ~WellFormed(
     ("w0" :> << <<0, 8>>, <<>> >>) @@ ("w1" :> << <<4, 8>>, <<>> >>),
     OutSizes, InDtypes, OutDtypes, ("w0" :> 1) @@ ("w1" :> 33), OutElements)

\\* THE property: one member exchanging a different volume from its peer.
MemberVolumeMismatchIsRejected ==
  ~Agreement(
     InDtypes, OutDtypes, ("w0" :> 66) @@ ("w1" :> 33),
     ("w0" :> 132) @@ ("w1" :> 66))

\\* And one member exchanging a different dtype from its peer.
MemberDtypeMismatchIsRejected ==
  ~Agreement(
     ("w0" :> <<"Float", "BFloat16">>) @@ ("w1" :> <<"BFloat16", "BFloat16">>),
     OutDtypes, InElements, OutElements)

\\* NCCL requires one dtype for both buffers of a collective.
MixedInputOutputDtypeIsRejected ==
  ~Agreement(
     InDtypes,
     ("w0" :> <<"Float", "Float">>) @@ ("w1" :> <<"Float", "Float">>),
     InElements, OutElements)

\\* THE second property: the operation relabelled the way an exporter bug would,
\\* on every member, with the relation label moved to match. Ordering agreement
\\* cannot see this; the volume relation can.
MislabelledOperationIsRejected ==
  ~Relates(
     ("w0" :> "all_reduce") @@ ("w1" :> "all_reduce"),
     Members,
     ("w0" :> "input_equals_output") @@ ("w1" :> "input_equals_output"),
     InElements, OutElements)

\\* A label that disagrees with the operation it claims to describe.
RelationLabelMustMatchTheOperationIsRejected ==
  ~Relates(
     Operation, Members,
     ("w0" :> "input_equals_output") @@ ("w1" :> "input_equals_output"),
     InElements, OutElements)

\\* An operation the relation table does not know must fail, not pass vacuously.
UnknownOperationIsRejected ==
  /\\ ImpliedSizeRelation("reduce") \\notin PayloadSizeRelations
  /\\ ~Relates(
        ("w0" :> "reduce") @@ ("w1" :> "reduce"), Members, Relation,
        InElements, OutElements)

\\* The keying property: a per-rank runtime id puts communicators with different
\\* member sets under one key, which is the eight-versus-four confusion.
RuntimeIdKeyingIsRejected ==
  ~CommKey(
     ("w0" :> "2") @@ ("w1" :> "2"),
     ("w0" :> {0, 2}) @@ ("w1" :> {1, 3}))

=============================================================================
"""

_PAYLOAD_PROBE_CFG = """\
SPECIFICATION Spec

INVARIANTS
  GoodInputIsAccepted
  DerivedElementCountIsRecomputed
  UnpairedDtypeListIsRejected
  ZeroDimensionIsRejected
  MemberVolumeMismatchIsRejected
  MemberDtypeMismatchIsRejected
  MixedInputOutputDtypeIsRejected
  MislabelledOperationIsRejected
  RelationLabelMustMatchTheOperationIsRejected
  UnknownOperationIsRejected
  RuntimeIdKeyingIsRejected
"""


def test_payload_predicates_reject_the_payloads_they_are_for(
    tmp_path: Path,
) -> None:
    """Run the real payload predicates over inputs they must refuse.

    The observed run satisfies all of them, which on its own says nothing about
    what they would catch -- and the checked-in facts module cannot carry payload
    facts until a fresh four-rank run regenerates it. This applies the shipped
    ScoutDistributed definitions, not a copy, to tiny hand-written facts, and
    asserts the NEGATION of each predicate on a bad input plus acceptance of a
    good one so the rejections are not vacuous.
    """

    status, output = _run_tlc(
        tmp_path,
        "PayloadProbe",
        _PAYLOAD_PROBE_CFG,
        fixtures=("ScoutDistributed.tla",),
        files={"PayloadProbe.tla": _PAYLOAD_PROBE},
    )

    assert (
        _run_classifier("formal_classify_tlc_valid", status, output).returncode == 0
    ), (status, output)


# A stand-in ScoutBFacts with one single-member collective over a 0-dim tensor.
# This is the input on which both payload controls' derived mutation targets have
# no witness: no work has a peer, no shape can be scaled, and no operation
# changes volume.
_B_FACTS_WITHOUT_AN_ELIGIBLE_PAYLOAD = """\
------------------------------ MODULE ScoutBFacts ------------------------------
EXTENDS Naturals, Sequences, FiniteSets, TLC

CollectiveWorkIds == <<"work:r0:only">>
CollectiveRank == ("work:r0:only" :> 0)
CollectiveId == ("work:r0:only" :> "tp:0:mesh_tp:seq1:all_reduce")
CollectiveComm == ("work:r0:only" :> "tp:0:mesh_tp")
CollectiveOperation == ("work:r0:only" :> "all_reduce")
CollectiveMembers == ("work:r0:only" :> {0})
CollectiveIssueOrder == ("work:r0:only" :> 4)
CollectivePayloadComm == ("work:r0:only" :> "tp:0:mesh_tp")
CollectivePayloadSizeRelation == ("work:r0:only" :> "input_equals_output")
CollectivePayloadInputSizes == ("work:r0:only" :> << <<>> >>)
CollectivePayloadOutputSizes == ("work:r0:only" :> << <<>> >>)
CollectivePayloadInputDtypes == ("work:r0:only" :> <<"BFloat16">>)
CollectivePayloadOutputDtypes == ("work:r0:only" :> <<"BFloat16">>)
CollectivePayloadInputElements == ("work:r0:only" :> 1)
CollectivePayloadOutputElements == ("work:r0:only" :> 1)

=============================================================================
"""


@pytest.mark.parametrize(
    ("module", "sentinel"),
    [
        ("ScoutBPayloadSizeInvalid", "ThereIsASizedWorkWithAPeer"),
        ("ScoutBPayloadOperationInvalid", "ThereIsAVolumeChangingCollective"),
    ],
    ids=["payload_size", "payload_operation"],
)
def test_payload_controls_name_their_missing_target_instead_of_crashing(
    tmp_path: Path, module: str, sentinel: str
) -> None:
    """The derived mutation target must be reported, not crashed on.

    `MutatedWork` is a constant definition, so TLC folds it before checking
    anything; an unguarded CHOOSE over facts with no eligible work abandons the
    search with an evaluation error and exit 75, which is not a checking result.
    The sentinel listed first turns that into a named invariant, and because the
    invariant mentions no variables TLC refutes it as a false constant
    expression.
    """

    status, output = _run_tlc(
        tmp_path,
        module,
        (FORMAL_DIR / f"{module}.cfg").read_text(),
        fixtures=("ScoutDistributed.tla", f"{module}.tla"),
        files={"ScoutBFacts.tla": _B_FACTS_WITHOUT_AN_ELIGIBLE_PAYLOAD},
    )

    assert "Attempted to compute the value of an expression of form" not in output
    assert (
        _run_classifier(
            "formal_classify_tlc_constant_false", status, output, sentinel
        ).returncode
        == 0
    ), (status, output)
