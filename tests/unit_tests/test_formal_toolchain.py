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
