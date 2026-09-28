# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contracts for the observed single-rank Qwen3 formal-verifier scout."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from torchtitan.experiments.qwen3_formal_verifier.single_rank import (
    _source_identity_payload,
    _write_attempt_bundle,
    _write_immutable,
    artifact_sync_mismatches,
    build_single_rank_config,
    build_source_manifest,
    export_single_rank_lean_facts,
    export_single_rank_tla_facts,
    finalize_evidence_manifest,
    HEAD_COMMIT_PATHS_EVIDENCE_PATH,
    is_process_source_path,
    LINT_COVERAGE_DIRTY_TREE_BYTES,
    LINT_COVERAGE_HEAD_COMMIT_PATHS,
    normalize_raw_trace,
    QFV_SCHEMA,
    SINGLE_RANK_ATTEMPT_ID,
    SINGLE_RANK_REQUIRED_STAGES,
    SINGLE_RANK_RUN_ID,
    source_manifest_lint_paths,
    SOURCE_MANIFEST_SCHEMA,
    SOURCE_STATUS_EVIDENCE_PATH,
    validate_normalized_trace,
    verify_evidence_bundle,
    verify_source_manifest,
    write_source_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_DIR = REPO_ROOT / "experiments" / "qwen3_formal_verifier" / "formal"
FIXTURE = (
    REPO_ROOT
    / "experiments"
    / "qwen3_formal_verifier"
    / "fixtures"
    / "single_rank.normalized.json"
)
RUNNER = REPO_ROOT / "experiments" / "qwen3_formal_verifier" / "run_single_rank.sh"
RUNNER_LIBRARY = REPO_ROOT / "experiments" / "qwen3_formal_verifier" / "runner_lib.sh"
CHECKER_CONTRACT = FORMAL_DIR / "checker_contract.sh"
FORMAL_WRAPPER = REPO_ROOT / "scripts" / "run_formal_checks.sh"


def _raw_trace() -> dict[str, Any]:
    kinds = (
        ("step.started", "step"),
        ("batch.observed", "input"),
        ("forward.started", "forward"),
        ("forward.completed", "forward"),
        ("backward.started", "backward"),
        ("gradient.ready", "backward"),
        ("backward.completed", "backward"),
        ("optimizer.started", "optimizer"),
        ("optimizer.mutated", "optimizer"),
        ("step.completed", "step"),
    )
    events = []
    for source_order, (kind, phase) in enumerate(kinds, start=1):
        event_id = f"raw-r0-{source_order:06d}"
        predecessors = [] if source_order == 1 else [f"raw-r0-{source_order - 1:06d}"]
        observation: dict[str, object] = {"hook": f"test.{kind}"}
        if kind == "batch.observed":
            observation["batch_sha256"] = "3" * 64
        if kind == "optimizer.mutated":
            observation.update(
                {
                    "before_sha256": "4" * 64,
                    "after_sha256": "5" * 64,
                    "mutated": True,
                }
            )
        events.append(
            {
                "raw_event_id": event_id,
                "source_order": source_order,
                "kind": kind,
                "phase": phase,
                "step": 1,
                "causal_predecessors": predecessors,
                "observation": observation,
            }
        )
    return {
        "schema": "qwen3.formal.raw.v0",
        "identity": {
            "run_id": SINGLE_RANK_RUN_ID,
            "attempt_id": SINGLE_RANK_ATTEMPT_ID,
            "process_id": "rank-0",
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
        },
        "device": {
            "kind": "cuda",
            "logical_id": "cuda:0",
            "name": "NVIDIA B200",
        },
        "mesh": {
            "axes": ["dp_replicate", "dp_shard", "cp", "tp", "pp", "ep"],
            "degrees": [1, 1, 1, 1, 1, 1],
            "coordinate": [0, 0, 0, 0, 0, 0],
        },
        "profile": {
            "config_module": "qwen3",
            "config_name": "qwen3_debugmodel",
            "model_name": "qwen3",
            "model_flavor": "debugmodel",
            "optimizer": "AdamW",
            "dtype": "bfloat16",
            "local_batch_size": 1,
            "global_batch_size": 1,
            "sequence_length": 128,
            "tokens_per_optimizer_step": 128,
            "gradient_accumulation_steps": 1,
            "optimizer_steps": 1,
            "seed": 42,
            "deterministic": True,
            "dataset": "c4_test",
            "tokenizer_path": "tests/assets/tokenizer",
            "checkpoint_load": False,
            "checkpoint_save": False,
        },
        "lineage": {
            "config_sha256": "1" * 64,
            "model_init_sha256": "2" * 64,
            "first_batch_sha256": "3" * 64,
        },
        "events": events,
    }


def _write_source_captures(
    attempt: Path,
    *,
    status_bytes: bytes,
    head_commit_paths: bytes,
) -> tuple[Path, Path]:
    """Write the two host captures the source manifest is derived from."""

    status_path = attempt / SOURCE_STATUS_EVIDENCE_PATH
    _write_immutable(status_path, status_bytes, root=attempt)
    head_paths_path = attempt / HEAD_COMMIT_PATHS_EVIDENCE_PATH
    _write_immutable(head_paths_path, head_commit_paths, root=attempt)
    return status_path, head_paths_path


def _create_sealable_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status_bytes: bytes = b"",
    head_commit_paths: bytes = b"",
    source_files: dict[str, str] | None = None,
    empty_stage_log: str | None = None,
) -> Path:
    """Seal a bundle, optionally over a source tree with real status entries.

    ``source_files`` maps repository-relative paths to contents and must agree
    with ``status_bytes``; both default to an empty source tree.

    ``head_commit_paths`` is the host HEAD-commit path capture. Defaulting both
    it and ``status_bytes`` to empty seals a bundle whose lint coverage case is
    ``head_commit_paths`` with nothing to lint, which is a truthful description
    of an empty tree; the lint stage, not sealing, is where that fails.

    ``empty_stage_log`` names one required stage whose log is written zero-byte,
    which is what a stage that aborted before emitting anything leaves behind.
    """

    for relative_path, contents in (source_files or {}).items():
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
    attempt = tmp_path / SINGLE_RANK_RUN_ID / SINGLE_RANK_ATTEMPT_ID
    (attempt / "manifests").mkdir(parents=True)
    status_path, head_paths_path = _write_source_captures(
        attempt,
        status_bytes=status_bytes,
        head_commit_paths=head_commit_paths,
    )
    source_path = attempt / "manifests" / "source.json"
    write_source_manifest(
        status_path=status_path,
        head_paths_path=head_paths_path,
        output_path=source_path,
        repo_root=tmp_path,
        head="a" * 40,
        attempt_dir=attempt,
    )
    monkeypatch.setenv("QFV_SOURCE_MANIFEST", str(source_path))
    monkeypatch.setenv("QFV_EXPECTED_HEAD", "a" * 40)
    monkeypatch.setenv("QFV_SOURCE_ROOT", str(tmp_path))
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    raw = _raw_trace()
    normalized = normalize_raw_trace(raw)
    _write_attempt_bundle(attempt, raw, normalized, ["--fixture"])
    checker = attempt / "checker"
    checker.mkdir()
    journal_lines: list[str] = []
    for stage_name, relative_log in SINGLE_RANK_REQUIRED_STAGES:
        log = attempt / relative_log
        contents = "" if stage_name == empty_stage_log else f"{stage_name}=ok\n"
        _write_immutable(log, contents, root=attempt)
        journal_lines.append(f"{stage_name}\t0\t{relative_log}\tcommand:{stage_name}\n")
    _write_immutable(
        checker / "stages.tsv",
        "".join(journal_lines),
        root=attempt,
    )
    return attempt


def test_single_rank_profile_comes_from_qwen3_debugmodel_cli_overrides(
    tmp_path: Path,
) -> None:
    config, cli_args = build_single_rank_config(tmp_path)

    assert cli_args[:4] == [
        "--module",
        "qwen3",
        "--config",
        "qwen3_debugmodel",
    ]
    assert config.model_spec is not None
    assert config.model_spec.name == "qwen3"
    assert config.model_spec.flavor == "debugmodel"
    assert config.training.local_batch_size == 1
    assert config.training.global_batch_size == 1
    assert config.training.seq_len == 128
    assert config.training.steps == 1
    assert config.training.dtype == "bfloat16"
    assert config.optimizer.param_groups[0].optimizer_name == "AdamW"
    assert config.dataloader.dataset == "c4_test"
    assert config.hf_assets_path == "./tests/assets/tokenizer"
    assert config.parallelism.data_parallel_replicate_degree == 1
    assert config.parallelism.data_parallel_shard_degree == 1
    assert config.parallelism.tensor_parallel_degree == 1
    assert config.parallelism.pipeline_parallel_degree == 1
    assert config.parallelism.context_parallel_degree == 1
    assert config.parallelism.expert_parallel_degree == 1
    assert config.parallelism.enable_sequence_parallel is False
    assert config.checkpoint.enable is False
    assert config.checkpoint.initial_load_path is None
    assert config.checkpoint.initial_load_in_hf is False
    assert config.debug.seed == 42
    assert config.debug.deterministic is True
    assert config.debug.deterministic_warn_only is False


def test_normalization_is_deterministic_and_every_fact_has_raw_provenance() -> None:
    raw = _raw_trace()

    first = normalize_raw_trace(raw)
    second = normalize_raw_trace(deepcopy(raw))
    first_events = cast(list[dict[str, Any]], first["events"])
    raw_events = cast(list[dict[str, Any]], raw["events"])
    provenance = cast(dict[str, Any], first["provenance"])

    assert first == second
    assert first["schema"] == QFV_SCHEMA
    assert [event["kind"] for event in first_events] == [
        event["kind"] for event in raw_events
    ]
    assert all(event["provenance"]["raw_event_id"] for event in first_events)
    assert set(provenance) == {event["event_id"] for event in first_events}
    assert validate_normalized_trace(first) is None


def test_normalization_rejects_duplicate_raw_event_identity() -> None:
    raw = _raw_trace()
    events = cast(list[dict[str, Any]], raw["events"])
    events[1]["raw_event_id"] = events[0]["raw_event_id"]

    with pytest.raises(ValueError, match="duplicate raw event ID"):
        normalize_raw_trace(raw)


def test_validation_rejects_duplicate_normalized_event_identity() -> None:
    trace = normalize_raw_trace(_raw_trace())
    events = cast(list[dict[str, Any]], trace["events"])
    provenance = cast(dict[str, Any], trace["provenance"])
    duplicate_id = events[0]["event_id"]
    removed_id = events[1]["event_id"]
    events[1]["event_id"] = duplicate_id
    events[1]["provenance"] = deepcopy(events[0]["provenance"])
    provenance.pop(removed_id)
    events[2]["causal_predecessors"] = [duplicate_id]

    with pytest.raises(ValueError, match="duplicate normalized event ID"):
        validate_normalized_trace(trace)


def test_validation_rejects_invalid_normalized_event_id_shape() -> None:
    trace = normalize_raw_trace(_raw_trace())
    events = cast(list[dict[str, Any]], trace["events"])
    provenance = cast(dict[str, Any], trace["provenance"])
    original_id = events[0]["event_id"]
    forged_id = "forged-event-id"
    events[0]["event_id"] = forged_id
    provenance[forged_id] = provenance.pop(original_id)
    events[1]["causal_predecessors"] = [forged_id]

    with pytest.raises(ValueError, match="event ID"):
        validate_normalized_trace(trace)


def test_validation_rejects_event_and_root_identity_disagreement() -> None:
    trace = normalize_raw_trace(_raw_trace())
    events = cast(list[dict[str, Any]], trace["events"])
    events[3]["rank"] = 1

    with pytest.raises(ValueError, match="root process identity"):
        validate_normalized_trace(trace)


def test_validation_rejects_tampered_content_addressed_trace_id() -> None:
    trace = normalize_raw_trace(_raw_trace())
    trace["trace_id"] = f"sha256:{'0' * 64}"

    with pytest.raises(ValueError, match="trace_id"):
        validate_normalized_trace(trace)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_provenance", "provenance"),
        ("missing_gradient", "gradient.ready"),
        ("optimizer_not_mutated", "parameter mutation"),
        ("wrong_phase", "invalid phase"),
        ("future_causal_predecessor", "causal order"),
    ),
)
def test_normalized_trace_rejects_single_dimension_corruption(
    mutation: str,
    message: str,
) -> None:
    trace = normalize_raw_trace(_raw_trace())
    events = cast(list[dict[str, Any]], trace["events"])
    if mutation == "missing_provenance":
        events[4].pop("provenance")
    elif mutation == "missing_gradient":
        events[5]["kind"] = "gradient.missing"
    elif mutation == "optimizer_not_mutated":
        events[8]["observation"]["mutated"] = False
    elif mutation == "wrong_phase":
        events[3]["phase"] = "backward"
    elif mutation == "future_causal_predecessor":
        events[3]["causal_predecessors"] = [events[4]["event_id"]]

    with pytest.raises(ValueError, match=message):
        validate_normalized_trace(trace)


def test_exporters_emit_facts_and_identity_without_approving_propositions() -> None:
    trace = normalize_raw_trace(_raw_trace())

    tla = export_single_rank_tla_facts(trace)
    lean = export_single_rank_lean_facts(trace)

    assert f'RunId == "{SINGLE_RANK_RUN_ID}"' in tla
    assert f'AttemptId == "{SINGLE_RANK_ATTEMPT_ID}"' in tla
    assert '"gradient.ready"' in tla
    assert "EventPredecessors ==" in tla
    assert "Invariant" not in tla
    assert "THEOREM" not in tla
    assert f'def runId : String := "{SINGLE_RANK_RUN_ID}"' in lean
    assert "kind := .gradientReady" in lean
    assert "def observedEvents : List ObservedEvent" in lean
    assert "theorem" not in lean
    assert "axiom" not in lean


def test_checked_in_single_rank_artifacts_match_normalized_runtime_fixture() -> None:
    trace = json.loads(FIXTURE.read_text())

    assert artifact_sync_mismatches(trace, FORMAL_DIR, FIXTURE) == ()


def test_single_rank_tlc_negative_requires_named_transition_invariant() -> None:
    # A real TLC run always prints the search summary; the classifier now
    # requires it so a truncated or crashed run cannot be read as a result.
    output = (
        "Error: Invariant SingleRankGradientReadyBeforeOptimizer is violated.\n"
        "Error: The behavior up to this point is:\n"
        "12 states generated, 11 distinct states found, 0 states left on queue.\n"
    )

    accepted = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; formal_classify_tlc_transition_negative "$2" "$3" "$4"',
            "classifier-test",
            str(CHECKER_CONTRACT),
            "12",
            output,
            "SingleRankGradientReadyBeforeOptimizer",
        ],
        cwd=REPO_ROOT,
    )
    wrong_name = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; formal_classify_tlc_transition_negative "$2" "$3" "$4"',
            "classifier-test",
            str(CHECKER_CONTRACT),
            "12",
            output,
            "DifferentInvariant",
        ],
        cwd=REPO_ROOT,
    )

    assert accepted.returncode == 0
    assert wrong_name.returncode != 0


def test_formal_wrapper_routes_single_rank_suite_through_fresh_offline_rootfs(
    tmp_path: Path,
) -> None:
    fake_entrypoint = tmp_path / "enter_rootfs.sh"
    invocation = tmp_path / "invocation.txt"
    fake_entrypoint.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "${TORCHTITAN_ROOTFS_NETWORK:-}" "$@" > "${QFV_TEST_LOG}"\n'
    )
    fake_entrypoint.chmod(0o755)
    cache = tmp_path / "formal-cache"
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "QFV_TEST_LOG": str(invocation),
        "TORCHTITAN_FORMAL_CACHE_HOST": str(cache),
        "TORCHTITAN_ROOTFS_ENTRYPOINT": str(fake_entrypoint),
    }

    result = subprocess.run(
        [
            "bash",
            str(FORMAL_WRAPPER),
            "--no-fetch",
            "--suite",
            "single-rank",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode == 0, result.stdout
    assert invocation.read_text().splitlines() == [
        "offline",
        "--",
        "scripts/run_formal_checks.sh",
        "--no-fetch",
        "--suite",
        "single-rank",
    ]


def test_supported_single_rank_runner_is_insula_aware_and_documents_full_gate() -> None:
    result = subprocess.run(
        ["bash", str(RUNNER), "--help"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode == 0, result.stdout
    assert "real CUDA Qwen3 optimizer step" in result.stdout
    assert "focused pytest" in result.stdout
    assert "artifact sync" in result.stdout
    assert "networked formal check" in result.stdout
    assert "no-fetch formal check" in result.stdout


def test_supported_runner_stops_before_finalize_when_log_sink_fails(
    tmp_path: Path,
) -> None:
    fake_entrypoint = tmp_path / "enter_rootfs.sh"
    invocation = tmp_path / "rootfs-invocations.txt"
    fake_entrypoint.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$*" >> "${QFV_TEST_INVOCATIONS}"\n'
    )
    fake_entrypoint.chmod(0o755)
    failing_tee = tmp_path / "failing-tee.sh"
    failing_tee.write_text("#!/usr/bin/env bash\n" "cat >/dev/null\n" "exit 23\n")
    failing_tee.chmod(0o755)
    fake_git = tmp_path / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'case "$*" in\n'
        "  *rev-parse*) printf '%040d\\n' 0 ;;\n"
        "  *status*) exit 0 ;;\n"
        "  *diff*) exit 0 ;;\n"
        "esac\n"
    )
    fake_git.chmod(0o755)
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    output = REPO_ROOT / "outputs" / "qfv-scout-a-runner-tests" / tmp_path.name
    shutil.rmtree(output, ignore_errors=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "QFV_TEST_INVOCATIONS": str(invocation),
            "TORCHTITAN_FORMAL_CACHE_HOST": str(tmp_path / "formal-cache"),
            "TORCHTITAN_QFV_ROOTFS": str(rootfs),
            "TORCHTITAN_QFV_ROOTFS_ENTRYPOINT": str(fake_entrypoint),
            "TORCHTITAN_QFV_TEE": str(failing_tee),
            "TORCHTITAN_QFV_GIT": str(fake_git),
        }
    )
    env.pop("TORCHTITAN_IN_ROOTFS", None)
    try:
        result = subprocess.run(
            [
                "bash",
                str(RUNNER),
                "--output-root",
                str(output.relative_to(REPO_ROOT)),
                "--run-id",
                "sink-failure",
                "--attempt-id",
                "attempt-0",
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    finally:
        shutil.rmtree(output, ignore_errors=True)

    assert result.returncode != 0
    assert "log sink failed" in result.stdout
    assert invocation.is_file()
    assert "finalize" not in invocation.read_text()


def test_supported_runner_stops_before_finalize_when_stage_command_fails(
    tmp_path: Path,
) -> None:
    fake_entrypoint = tmp_path / "enter_rootfs.sh"
    invocation = tmp_path / "rootfs-invocations.txt"
    fake_entrypoint.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$*" >> "${QFV_TEST_INVOCATIONS}"\n'
        "exit 19\n"
    )
    fake_entrypoint.chmod(0o755)
    fake_git = tmp_path / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'case "$*" in\n'
        "  *rev-parse*) printf '%040d\\n' 0 ;;\n"
        "  *status*) exit 0 ;;\n"
        "  *diff*) exit 0 ;;\n"
        "esac\n"
    )
    fake_git.chmod(0o755)
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    output = REPO_ROOT / "outputs" / "qfv-scout-a-runner-tests" / tmp_path.name
    shutil.rmtree(output, ignore_errors=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "QFV_TEST_INVOCATIONS": str(invocation),
            "TORCHTITAN_FORMAL_CACHE_HOST": str(tmp_path / "formal-cache"),
            "TORCHTITAN_QFV_GIT": str(fake_git),
            "TORCHTITAN_QFV_ROOTFS": str(rootfs),
            "TORCHTITAN_QFV_ROOTFS_ENTRYPOINT": str(fake_entrypoint),
        }
    )
    env.pop("TORCHTITAN_IN_ROOTFS", None)
    try:
        result = subprocess.run(
            [
                "bash",
                str(RUNNER),
                "--output-root",
                str(output.relative_to(REPO_ROOT)),
                "--run-id",
                "command-failure",
                "--attempt-id",
                "attempt-0",
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    finally:
        shutil.rmtree(output, ignore_errors=True)

    assert result.returncode == 19
    assert "stage command failed with status 19" in result.stdout
    assert invocation.is_file()
    assert "finalize" not in invocation.read_text()


def test_supported_runner_rejects_nested_output_directory_symlink(
    tmp_path: Path,
) -> None:
    base = REPO_ROOT / "outputs" / "qfv-scout-a-runner-tests" / tmp_path.name
    escape = base / "escape"
    escape.mkdir(parents=True)
    linked_output = base / "linked-output"
    linked_output.symlink_to(escape, target_is_directory=True)
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "TORCHTITAN_FORMAL_CACHE_HOST": str(tmp_path / "formal-cache"),
            "TORCHTITAN_QFV_ROOTFS": str(rootfs),
        }
    )
    env.pop("TORCHTITAN_IN_ROOTFS", None)
    try:
        result = subprocess.run(
            [
                "bash",
                str(RUNNER),
                "--output-root",
                str(linked_output.relative_to(REPO_ROOT)),
                "--run-id",
                "symlink-run",
                "--attempt-id",
                "attempt-0",
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)

    assert result.returncode != 0
    assert "symlink" in result.stdout.lower()


def test_runner_log_creation_rejects_file_symlink(tmp_path: Path) -> None:
    approved_root = tmp_path / "attempt"
    checker = approved_root / "checker"
    checker.mkdir(parents=True)
    target = tmp_path / "outside.log"
    target.write_text("must remain intact\n")
    log = checker / "focused-pytest.log"
    log.symlink_to(target)

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; qfv_create_fresh_log "$2" "$3"',
            "runner-log-test",
            str(RUNNER_LIBRARY),
            str(approved_root),
            str(log),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode != 0
    assert "symlink" in result.stdout.lower()
    assert target.read_text() == "must remain intact\n"


def test_runner_directory_guard_rejects_symlink_in_conditional_helper_call(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(directory, target_is_directory=True)

    result = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$1"
runner_helper_call() {
  qfv_require_directory "$1" || return 1
  printf 'guard-returned-success\\n'
}
runner_helper_call "$2"
""",
            "runner-directory-test",
            str(RUNNER_LIBRARY),
            str(linked_root),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode != 0
    assert "symlink" in result.stdout.lower()
    assert "guard-returned-success" not in result.stdout


def test_immutable_writer_rejects_symlink_and_preserves_target(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    evidence_dir = attempt / "raw"
    evidence_dir.mkdir(parents=True)
    target = tmp_path / "outside.json"
    target.write_text("same bytes\n")
    evidence = evidence_dir / "observed.json"
    evidence.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        _write_immutable(evidence, "same bytes\n", root=attempt)

    assert target.read_text() == "same bytes\n"


def test_immutable_writer_rejects_equal_existing_file_with_wrong_mode(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    evidence_dir = attempt / "raw"
    evidence_dir.mkdir(parents=True)
    evidence = evidence_dir / "observed.json"
    evidence.write_text("same bytes\n")
    evidence.chmod(0o644)

    with pytest.raises(PermissionError, match="read-only mode 0444"):
        _write_immutable(evidence, "same bytes\n", root=attempt)


def test_source_manifest_is_deterministic_and_detects_executable_source_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    status = b"?? source.py\0"

    first = build_source_manifest(tmp_path, "b" * 40, status, head_commit_paths=b"")
    second = build_source_manifest(tmp_path, "b" * 40, status, head_commit_paths=b"")

    assert first == second
    source.write_text("value = 2\n")
    with pytest.raises(ValueError, match="no longer matches the current source tree"):
        build_source_manifest(
            tmp_path, "b" * 40, status, head_commit_paths=b"", expected=first
        )


@pytest.mark.parametrize(
    "path_text,expected",
    [
        (".scratch/notes.md", True),
        (".scratch/qwen3-formal-verifier/spec.md", True),
        (".superpowers/sdd/spec/task-03-report.md", True),
        (".scratchpad/notes.md", False),
        (".scratches/notes.md", False),
        ("scratch/notes.md", False),
        ("scripts/run.sh", False),
        ("a/.scratch/notes.md", False),
    ],
)
def test_process_source_path_predicate_is_component_exact(
    path_text: str,
    expected: bool,
) -> None:
    assert is_process_source_path(path_text) is expected


def test_process_document_bytes_do_not_change_source_identity(
    tmp_path: Path,
) -> None:
    """The regression for the ticket-03 post-seal tracker mutation."""

    (tmp_path / "source.py").write_text("value = 1\n")
    ticket_dir = tmp_path / ".scratch" / "qwen3-formal-verifier" / "issues"
    ticket_dir.mkdir(parents=True)
    ticket = ticket_dir / "03-device-mesh.md"
    ticket.write_text("**Status:** review-pending\n")
    status = (
        b"?? source.py\0?? .scratch/qwen3-formal-verifier/issues/03-device-mesh.md\0"
    )

    sealed = build_source_manifest(tmp_path, "b" * 40, status, head_commit_paths=b"")
    entries = cast(list[dict[str, Any]], sealed["entries"])
    assert [entry["path"] for entry in entries] == ["source.py"]
    section = cast(dict[str, Any], sealed["process_informational"])
    process = cast(list[dict[str, Any]], section["entries"])
    assert [entry["path"] for entry in process] == [
        ".scratch/qwen3-formal-verifier/issues/03-device-mesh.md"
    ]

    # Appending gate evidence and resolving the ticket must stay verifiable.
    ticket.write_text(
        "**Status:** resolved\n\n## Gate evidence\n\nevidence_id: sha256:abc\n"
    )
    rebuilt = build_source_manifest(
        tmp_path, "b" * 40, status, head_commit_paths=b"", expected=sealed
    )
    assert rebuilt["source_id"] == sealed["source_id"]


def test_process_entries_are_labelled_unverified(tmp_path: Path) -> None:
    (tmp_path / ".scratch").mkdir()
    (tmp_path / ".scratch" / "notes.md").write_text("notes\n")
    status = b"?? .scratch/notes.md\0"

    manifest = build_source_manifest(tmp_path, "b" * 40, status, head_commit_paths=b"")

    section = cast(dict[str, Any], manifest["process_informational"])
    assert section["note"]
    assert section["predicate"]
    record = cast(list[dict[str, Any]], section["entries"])[0]
    assert set(record) == {
        "status",
        "path",
        "kind",
        "mode",
        "size_bytes",
        "sha256_at_seal",
        "verified",
    }
    assert record["verified"] is False
    assert "sha256" not in record


def test_process_root_rejects_executable_extension(tmp_path: Path) -> None:
    (tmp_path / ".scratch").mkdir()
    (tmp_path / ".scratch" / "tool.py").write_text("value = 1\n")

    with pytest.raises(ValueError, match="may not carry executable"):
        build_source_manifest(
            tmp_path, "b" * 40, b"?? .scratch/tool.py\0", head_commit_paths=b""
        )


def test_source_identity_payload_key_set_is_closed(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_text("value = 1\n")
    manifest = build_source_manifest(
        tmp_path, "b" * 40, b"?? source.py\0", head_commit_paths=b""
    )

    payload = _source_identity_payload(manifest)

    assert set(payload) == {
        "schema",
        "head",
        "status_sha256",
        "entries",
        "source_id",
    }
    assert "process_informational" not in payload


def test_lint_paths_cover_verified_and_process_paths(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_text("value = 1\n")
    (tmp_path / ".scratch").mkdir()
    (tmp_path / ".scratch" / "notes.md").write_text("notes\n")
    status = b"?? source.py\0?? .scratch/notes.md\0"
    manifest = build_source_manifest(tmp_path, "b" * 40, status, head_commit_paths=b"")

    assert source_manifest_lint_paths(manifest) == (".scratch/notes.md", "source.py")

    with pytest.raises(ValueError, match="schema must be"):
        source_manifest_lint_paths({**manifest, "schema": "other.v0"})
    stripped = {
        key: manifest[key] for key in manifest if key != "process_informational"
    }
    with pytest.raises(ValueError, match="process_informational"):
        source_manifest_lint_paths(stripped)


@pytest.mark.parametrize(
    "mutate,expected",
    [
        pytest.param(
            lambda coverage: {**coverage, "case": LINT_COVERAGE_HEAD_COMMIT_PATHS},
            f"case must be {LINT_COVERAGE_DIRTY_TREE_BYTES}",
            id="claims_clean_while_dirty",
        ),
        pytest.param(
            lambda coverage: {**coverage, "head_commit_lint_paths": ["other.py"]},
            "must not list HEAD lint paths",
            id="lists_head_paths_while_dirty",
        ),
        pytest.param(
            lambda coverage: {**coverage, "head_commit_paths_sha256": "not-hex"},
            "hex head_commit_paths_sha256",
            id="non_hex_digest",
        ),
        pytest.param(
            lambda coverage: {**coverage, "num_head_commit_paths": -1},
            "must not be negative",
            id="negative_count",
        ),
        pytest.param(
            lambda coverage: {**coverage, "num_head_commit_paths": True},
            "must be an integer",
            id="boolean_count",
        ),
        pytest.param(
            lambda coverage: {k: v for k, v in coverage.items() if k != "case"},
            "unexpected key set",
            id="missing_case",
        ),
    ],
)
def test_forged_lint_coverage_is_rejected(
    mutate: Any,
    expected: str,
    tmp_path: Path,
) -> None:
    """The section states what lint covered, so a forged one must not read."""

    (tmp_path / "source.py").write_text("value = 1\n")
    manifest = build_source_manifest(
        tmp_path,
        "b" * 40,
        b"?? source.py\0",
        head_commit_paths=b"other.py\0",
    )
    forged = {**manifest, "lint_coverage": mutate(manifest["lint_coverage"])}

    with pytest.raises(ValueError, match=expected):
        source_manifest_lint_paths(forged)


def test_sealed_lint_coverage_must_match_the_recomputed_section(
    tmp_path: Path,
) -> None:
    """Outside source_id, so it is compared rather than hashed into identity."""

    (tmp_path / "source.py").write_text("value = 1\n")
    status = b"?? source.py\0"
    sealed = build_source_manifest(
        tmp_path,
        "b" * 40,
        status,
        head_commit_paths=b"other.py\0",
    )
    coverage = cast(dict[str, Any], sealed["lint_coverage"])
    forged = {
        **sealed,
        "lint_coverage": {**coverage, "head_commit_paths_sha256": "0" * 64},
    }

    # Identity is untouched by the forgery, which is exactly why the section
    # needs its own comparison.
    assert _source_identity_payload(forged) == _source_identity_payload(sealed)
    with pytest.raises(ValueError, match="lint coverage does not match"):
        build_source_manifest(
            tmp_path,
            "b" * 40,
            status,
            head_commit_paths=b"other.py\0",
            expected=forged,
        )


def test_head_commit_paths_stay_out_of_the_source_identity(
    tmp_path: Path,
) -> None:
    """Criterion: the fallback decides what to lint, not what is pinned."""

    (tmp_path / "source.py").write_text("value = 1\n")
    status = b"?? source.py\0"
    without = build_source_manifest(tmp_path, "b" * 40, status, head_commit_paths=b"")
    with_paths = build_source_manifest(
        tmp_path, "b" * 40, status, head_commit_paths=b"source.py\0other.py\0"
    )

    assert with_paths["source_id"] == without["source_id"]
    assert with_paths["entries"] == without["entries"]
    assert with_paths["lint_coverage"] != without["lint_coverage"]
    # Lint still covers only the dirty bytes while the tree is dirty.
    assert source_manifest_lint_paths(with_paths) == ("source.py",)


def test_duplicate_head_commit_paths_from_merge_diffs_collapse(
    tmp_path: Path,
) -> None:
    """``git diff-tree -m`` emits one diff per parent, so paths repeat."""

    (tmp_path / "module.py").write_text("value = 1\n")

    manifest = build_source_manifest(
        tmp_path,
        "b" * 40,
        b"",
        head_commit_paths=b"module.py\0module.py\0missing.py\0",
    )

    coverage = cast(dict[str, Any], manifest["lint_coverage"])
    assert coverage["case"] == LINT_COVERAGE_HEAD_COMMIT_PATHS
    assert coverage["num_head_commit_paths"] == 2
    # missing.py is reported by the commit but absent now, so it is dropped.
    assert coverage["head_commit_lint_paths"] == ["module.py"]
    assert source_manifest_lint_paths(manifest) == ("module.py",)


def _lint_stage_program() -> str:
    """Return the exact bash program the gate's lint stage runs."""

    result = subprocess.run(
        ["bash", "-c", f"source {RUNNER_LIBRARY}; qfv_lint_stage_program"],
        capture_output=True,
        check=True,
    )
    return result.stdout.decode()


def _write_stub_tool(directory: Path, name: str) -> None:
    """Install a PATH stub that echoes its argument list and succeeds.

    pre-commit and pyrefly are the two tools the lint stage shells out to. What
    is under test is the stage's own behaviour -- which paths it derives and
    which it hands on -- so the tools are stubbed and their argument lists are
    read back out of the stage's captured output.
    """

    script = directory / name
    script.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "{name}"\n'
        'for argument in "$@"; do printf " %s" "${argument}"; done\n'
        'printf "\\n"\n'
        "exit 0\n"
    )
    script.chmod(0o755)


def _run_lint_stage(
    *,
    repo_root: Path,
    attempt_dir: Path,
    scratch_dir: Path,
    head: str,
    stub_dir: Path,
    identity: str = "lint-stage-test",
) -> subprocess.CompletedProcess[bytes]:
    """Execute the gate's lint stage against a throwaway tree."""

    program = _lint_stage_program()
    environment = dict(os.environ)
    environment["PATH"] = f"{stub_dir}{os.pathsep}{environment['PATH']}"
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["TORCHTITAN_IN_ROOTFS"] = "1"
    return subprocess.run(
        [
            "bash",
            "-c",
            program,
            "single-rank-lint-stage-test",
            "single_rank",
            str(repo_root),
            str(scratch_dir),
            identity,
            str(attempt_dir / "manifests" / "source.json"),
            str(attempt_dir / SOURCE_STATUS_EVIDENCE_PATH),
            str(attempt_dir / HEAD_COMMIT_PATHS_EVIDENCE_PATH),
            head,
            str(attempt_dir),
        ],
        capture_output=True,
        env=environment,
    )


def _committed_source_tree(tmp_path: Path) -> tuple[Path, str, bytes, bytes]:
    """Build a real one-commit repository and capture the host Git evidence.

    Returns the repository root, HEAD, the porcelain status bytes, and the
    HEAD-commit path bytes -- the two captures the runners take on the host,
    taken here with the same commands.
    """

    repo_root = tmp_path / "repo"
    (repo_root / ".scratch").mkdir(parents=True)
    (repo_root / "module.py").write_text("value = 1\n")
    (repo_root / "script.sh").write_text("#!/usr/bin/env bash\necho ok\n")
    (repo_root / ".scratch" / "notes.md").write_text("notes\n")
    git = ["git", "-C", str(repo_root)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run(
        [
            *git,
            "-c",
            "user.name=scout",
            "-c",
            "user.email=scout@example.invalid",  # pii-allow: fixture identity
            "commit",
            "-q",
            "-m",
            "one",
        ],
        check=True,
    )
    head = (
        subprocess.run(
            [*git, "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .strip()
    )
    status_bytes = subprocess.run(
        [*git, "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        capture_output=True,
        check=True,
    ).stdout
    head_commit_paths = subprocess.run(
        [
            *git,
            "diff-tree",
            "--root",
            "-m",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-z",
            "HEAD",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return repo_root, head, status_bytes, head_commit_paths


def test_lint_stage_lints_the_head_commit_when_the_tree_is_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ticket-25 regression, run rather than read.

    Gating a committed state means gating a clean tree. The source manifest is
    derived from ``git status``, so a clean tree reported zero entries, the path
    list was empty, and the stage died on its own non-empty assertion with a
    ten-line log. This executes the stage's real program against a real clean
    one-commit repository.
    """

    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    repo_root, head, status_bytes, head_commit_paths = _committed_source_tree(tmp_path)
    assert status_bytes == b"", "the fixture repository must be clean"

    attempt = tmp_path / "attempt"
    (attempt / "manifests").mkdir(parents=True)
    status_path, head_paths_path = _write_source_captures(
        attempt,
        status_bytes=status_bytes,
        head_commit_paths=head_commit_paths,
    )
    write_source_manifest(
        status_path=status_path,
        head_paths_path=head_paths_path,
        output_path=attempt / "manifests" / "source.json",
        repo_root=repo_root,
        head=head,
        attempt_dir=attempt,
    )
    manifest = json.loads((attempt / "manifests" / "source.json").read_text())
    coverage = manifest["lint_coverage"]
    assert manifest["entries"] == []
    assert coverage["case"] == LINT_COVERAGE_HEAD_COMMIT_PATHS
    assert coverage["head_commit_lint_paths"] == [
        ".scratch/notes.md",
        "module.py",
        "script.sh",
    ]

    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    for tool in ("pre-commit", "pyrefly"):
        _write_stub_tool(stub_dir, tool)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    result = _run_lint_stage(
        repo_root=repo_root,
        attempt_dir=attempt,
        scratch_dir=scratch,
        head=head,
        stub_dir=stub_dir,
    )

    stdout = result.stdout.decode()
    stderr = result.stderr.decode()
    assert result.returncode == 0, stdout + stderr
    assert "QFV_LINT_STAGE result=success num_paths=3" in stdout
    assert (
        f"QFV_LINT_PATHS case={LINT_COVERAGE_HEAD_COMMIT_PATHS} num_paths=3"
    ) in stderr
    # Every declared hook ran, over exactly the HEAD-commit paths.
    hook_lines = [line for line in stdout.splitlines() if line.startswith("pre-commit")]
    assert len(hook_lines) == 14, hook_lines
    for line in hook_lines:
        assert line.endswith("--files .scratch/notes.md module.py script.sh"), line
    assert (
        "pyrefly check --remove-unused-ignores --summarize-errors module.py" in stdout
    )


def test_lint_stage_skips_pyrefly_for_test_modules_only(tmp_path: Path) -> None:
    """Honour the pyrefly exclude the project declares, without widening it.

    pyproject.toml puts **/tests/** in project-excludes, but passing explicit
    paths bypasses that, so the stage used to type-check files the project says
    to skip. Test modules are mock-heavy and carry hundreds of pre-existing
    errors -- test_checkpoint.py alone has 172 at HEAD -- so changing any test
    file failed the stage on errors it did not introduce.

    Both halves are asserted, because a skip that was too wide would be the
    worse bug: a non-test module must still reach pyrefly, and a shell file
    under tests/ must still reach bash -n.
    """

    repo_root = tmp_path / "repo"
    (repo_root / "tests" / "unit_tests").mkdir(parents=True)
    (repo_root / ".scratch").mkdir(parents=True)
    (repo_root / "module.py").write_text("value = 1\n")
    (repo_root / "tests" / "unit_tests" / "test_thing.py").write_text("value = 2\n")
    (repo_root / "tests" / "helper.sh").write_text("#!/usr/bin/env bash\necho ok\n")
    git = ["git", "-C", str(repo_root)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run(
        [
            *git,
            "-c",
            "user.name=verifier",
            "-c",
            "user.email=verifier@example.invalid",  # pii-allow: fixture identity
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    head_commit_paths = subprocess.run(
        [
            *git,
            "diff-tree",
            "--root",
            "-m",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-z",
            "HEAD",
        ],
        capture_output=True,
        check=True,
    ).stdout

    attempt = tmp_path / "attempt"
    (attempt / "manifests").mkdir(parents=True)
    status_path, head_paths_path = _write_source_captures(
        attempt,
        status_bytes=b"",
        head_commit_paths=head_commit_paths,
    )
    write_source_manifest(
        status_path=status_path,
        head_paths_path=head_paths_path,
        output_path=attempt / "manifests" / "source.json",
        repo_root=repo_root,
        head=head,
        attempt_dir=attempt,
    )
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    for tool in ("pre-commit", "pyrefly"):
        stub = stub_dir / tool
        stub.write_text(
            '#!/usr/bin/env bash\nprintf "%s %s\\n" "$(basename "$0")" "$*"\n'
        )
        stub.chmod(0o755)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    result = _run_lint_stage(
        repo_root=repo_root,
        attempt_dir=attempt,
        scratch_dir=scratch,
        head=head,
        stub_dir=stub_dir,
    )
    stdout = result.stdout.decode()
    assert result.returncode == 0, stdout

    pyrefly_lines = [ln for ln in stdout.splitlines() if ln.startswith("pyrefly ")]
    assert len(pyrefly_lines) == 1, stdout
    # The non-test module reaches pyrefly; the test module does not.
    assert "module.py" in pyrefly_lines[0], pyrefly_lines
    assert "test_thing.py" not in pyrefly_lines[0], pyrefly_lines
    # The skip must not swallow shell files under tests/, which still need
    # bash -n. pre-commit sees every file regardless.
    precommit = [ln for ln in stdout.splitlines() if ln.startswith("pre-commit ")]
    assert any("test_thing.py" in ln for ln in precommit), precommit
    assert any("tests/helper.sh" in ln for ln in precommit), precommit


def test_lint_stage_still_lints_the_dirty_tree_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dirty-tree path is unchanged: lint covers the reported paths only.

    Shown rather than asserted. ``module.py`` is committed and untouched, so it
    appears in the HEAD-commit capture but must not be linted while the tree is
    dirty -- the fallback is a fallback.
    """

    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    repo_root, head, _, head_commit_paths = _committed_source_tree(tmp_path)
    (repo_root / "script.sh").write_text("#!/usr/bin/env bash\necho changed\n")
    (repo_root / "added.py").write_text("value = 2\n")
    status_bytes = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        capture_output=True,
        check=True,
    ).stdout
    assert status_bytes != b""

    attempt = tmp_path / "attempt"
    (attempt / "manifests").mkdir(parents=True)
    status_path, head_paths_path = _write_source_captures(
        attempt,
        status_bytes=status_bytes,
        head_commit_paths=head_commit_paths,
    )
    write_source_manifest(
        status_path=status_path,
        head_paths_path=head_paths_path,
        output_path=attempt / "manifests" / "source.json",
        repo_root=repo_root,
        head=head,
        attempt_dir=attempt,
    )
    manifest = json.loads((attempt / "manifests" / "source.json").read_text())
    coverage = manifest["lint_coverage"]
    assert coverage["case"] == LINT_COVERAGE_DIRTY_TREE_BYTES
    assert coverage["head_commit_lint_paths"] == []
    assert coverage["num_head_commit_paths"] == 3
    assert [entry["path"] for entry in manifest["entries"]] == [
        "added.py",
        "script.sh",
    ]

    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    for tool in ("pre-commit", "pyrefly"):
        _write_stub_tool(stub_dir, tool)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    result = _run_lint_stage(
        repo_root=repo_root,
        attempt_dir=attempt,
        scratch_dir=scratch,
        head=head,
        stub_dir=stub_dir,
    )

    stdout = result.stdout.decode()
    assert result.returncode == 0, stdout + result.stderr.decode()
    assert "QFV_LINT_STAGE result=success num_paths=2" in stdout
    assert (
        f"QFV_LINT_PATHS case={LINT_COVERAGE_DIRTY_TREE_BYTES} num_paths=2"
    ) in result.stderr.decode()
    hook_lines = [line for line in stdout.splitlines() if line.startswith("pre-commit")]
    assert len(hook_lines) == 14, hook_lines
    for line in hook_lines:
        assert line.endswith("--files added.py script.sh"), line
    assert "module.py" not in stdout


def test_lint_stage_fails_with_a_diagnosis_when_nothing_is_lintable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty tree has nothing to lint, and must say so.

    This is the shape of the original failure -- no paths at all -- and the
    stage must now name the reason instead of exiting on a bare arithmetic
    assertion with an otherwise empty log.
    """

    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    attempt = tmp_path / "attempt"
    (attempt / "manifests").mkdir(parents=True)
    status_path, head_paths_path = _write_source_captures(
        attempt,
        status_bytes=b"",
        head_commit_paths=b"",
    )
    write_source_manifest(
        status_path=status_path,
        head_paths_path=head_paths_path,
        output_path=attempt / "manifests" / "source.json",
        repo_root=repo_root,
        head="b" * 40,
        attempt_dir=attempt,
    )
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    for tool in ("pre-commit", "pyrefly"):
        _write_stub_tool(stub_dir, tool)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    result = _run_lint_stage(
        repo_root=repo_root,
        attempt_dir=attempt,
        scratch_dir=scratch,
        head="b" * 40,
        stub_dir=stub_dir,
    )

    assert result.returncode != 0
    assert "the source manifest yields no lint paths" in result.stderr.decode()


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param("qwen3.formal.scout.source-manifest.v0", id="v0"),
        pytest.param("qwen3.formal.scout.source-manifest.v1", id="v1"),
    ],
)
def test_verify_source_manifest_rejects_superseded_schemas(
    schema: str,
    tmp_path: Path,
) -> None:
    """A v1 manifest records no lint coverage, so it is refused, not read."""

    attempt = tmp_path / "attempt"
    (attempt / "manifests").mkdir(parents=True)
    status_path, head_paths_path = _write_source_captures(
        attempt,
        status_bytes=b"",
        head_commit_paths=b"",
    )
    manifest_path = attempt / "manifests" / "source.json"
    _write_immutable(
        manifest_path,
        json.dumps(
            {
                "schema": schema,
                "head": "b" * 40,
                "status_sha256": "0" * 64,
                "entries": [],
                "source_id": "sha256:" + "0" * 64,
            }
        ),
        root=attempt,
    )

    with pytest.raises(ValueError, match="unsupported schema"):
        verify_source_manifest(
            manifest_path,
            status_path,
            head_paths_path=head_paths_path,
            repo_root=tmp_path,
            expected_head="b" * 40,
            attempt_dir=attempt,
        )


def test_verify_rejects_process_entry_that_fails_predicate(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    (attempt / "manifests").mkdir(parents=True)
    status_path, head_paths_path = _write_source_captures(
        attempt,
        status_bytes=b"",
        head_commit_paths=b"",
    )
    manifest_path = attempt / "manifests" / "source.json"
    _write_immutable(
        manifest_path,
        json.dumps(
            {
                "schema": SOURCE_MANIFEST_SCHEMA,
                "head": "b" * 40,
                "status_sha256": "0" * 64,
                "entries": [],
                "source_id": "sha256:" + "0" * 64,
                "process_informational": {
                    "note": "n",
                    "predicate": "p",
                    "entries": [
                        {
                            "status": "??",
                            "path": "scripts/run.sh",
                            "kind": "file",
                            "mode": "0644",
                            "size_bytes": 1,
                            "sha256_at_seal": "0" * 64,
                            "verified": False,
                        }
                    ],
                },
            }
        ),
        root=attempt,
    )

    with pytest.raises(ValueError, match="fails the process predicate"):
        verify_source_manifest(
            manifest_path,
            status_path,
            head_paths_path=head_paths_path,
            repo_root=tmp_path,
            expected_head="b" * 40,
            attempt_dir=attempt,
        )


def test_sealed_bundle_survives_process_document_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Informational process bytes may drift; executable source may not."""

    status = b"?? source.py\0?? .scratch/qwen3-formal-verifier/issues/03.md\0"
    attempt = _create_sealable_bundle(
        tmp_path,
        monkeypatch,
        status_bytes=status,
        source_files={
            "source.py": "value = 1\n",
            ".scratch/qwen3-formal-verifier/issues/03.md": "**Status:** claimed\n",
        },
    )
    finalize_evidence_manifest(
        attempt, expected_head="a" * 40, source_root=str(tmp_path)
    )

    ticket = tmp_path / ".scratch" / "qwen3-formal-verifier" / "issues" / "03.md"
    ticket.write_text("**Status:** resolved\n\n## Gate evidence\n\nid: sha256:abc\n")
    verify_evidence_bundle(attempt, expected_head="a" * 40, source_root=str(tmp_path))

    (tmp_path / "source.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="no longer matches the current source tree"):
        verify_evidence_bundle(
            attempt, expected_head="a" * 40, source_root=str(tmp_path)
        )


def test_sealed_process_section_is_byte_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Informational does not mean unprotected inside the bundle."""

    status = b"?? .scratch/notes.md\0"
    attempt = _create_sealable_bundle(
        tmp_path,
        monkeypatch,
        status_bytes=status,
        source_files={".scratch/notes.md": "notes\n"},
    )
    finalize_evidence_manifest(
        attempt, expected_head="a" * 40, source_root=str(tmp_path)
    )

    source_json = attempt / "manifests" / "source.json"
    manifest = json.loads(source_json.read_text())
    manifest["process_informational"]["entries"][0]["sha256_at_seal"] = "0" * 64
    source_json.chmod(0o644)
    source_json.write_text(json.dumps(manifest))
    source_json.chmod(0o444)

    with pytest.raises(ValueError, match="manifests/source.json"):
        verify_evidence_bundle(
            attempt, expected_head="a" * 40, source_root=str(tmp_path)
        )


def test_finalization_rejects_tampered_runtime_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = _create_sealable_bundle(tmp_path, monkeypatch)
    raw = attempt / "raw" / "observed.json"
    raw.chmod(0o644)
    raw.write_text(raw.read_text() + "tampered\n")
    raw.chmod(0o444)

    with pytest.raises(ValueError, match="raw/observed.json"):
        finalize_evidence_manifest(
            attempt,
            expected_head="a" * 40,
            source_root=tmp_path,
        )

    assert not (attempt / "manifests" / "evidence.json").exists()


def test_finalization_rejects_writable_checker_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = _create_sealable_bundle(tmp_path, monkeypatch)
    log = attempt / "checker" / "focused-pytest.log"
    log.chmod(0o644)

    with pytest.raises(PermissionError, match="read-only mode 0444"):
        finalize_evidence_manifest(
            attempt,
            expected_head="a" * 40,
            source_root=tmp_path,
        )


@pytest.mark.parametrize(
    "stage_name", [name for name, _ in SINGLE_RANK_REQUIRED_STAGES]
)
def test_sealing_refuses_a_zero_byte_stage_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_name: str,
) -> None:
    """Drive the real sealer with a zero-byte stage log for every stage.

    This is the experiment that exposed the hole: zero-byte stage logs were fed
    to the real sealer and every one was accepted, because the only non-empty
    check lived in the shell runner, outside the sealed contract.
    """

    attempt = _create_sealable_bundle(
        tmp_path,
        monkeypatch,
        empty_stage_log=stage_name,
    )

    with pytest.raises(ValueError, match=f"stage {stage_name} log is empty"):
        finalize_evidence_manifest(
            attempt,
            expected_head="a" * 40,
            source_root=tmp_path,
        )


def test_evidence_verification_never_creates_a_directory_in_the_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verification is read-only, so a missing parent is an error, not a mkdir.

    A third party verifying sealed evidence has a read-only copy. The managed
    path helper used to create missing parents for every caller, so
    verification mutated the very bundle it was verifying.
    """

    attempt = _create_sealable_bundle(tmp_path, monkeypatch)
    finalize_evidence_manifest(
        attempt,
        expected_head="a" * 40,
        source_root=tmp_path,
    )
    before = {
        str(path.relative_to(attempt)) for path in attempt.rglob("*") if path.is_dir()
    }
    shutil.rmtree(attempt / "generated")

    with pytest.raises(FileNotFoundError):
        verify_evidence_bundle(
            attempt,
            expected_head="a" * 40,
            source_root=tmp_path,
        )

    after = {
        str(path.relative_to(attempt)) for path in attempt.rglob("*") if path.is_dir()
    }
    assert "generated" not in after
    assert after == before - {"generated"}


def test_runner_library_skip_guard_refuses_only_undeclared_real_pytest_skips(
    tmp_path: Path,
) -> None:
    """Drive the shared guard against output from a real pytest run.

    Both directions are exercised, and the SKIPPED line is produced by pytest
    rather than hand-written -- the earlier guard was "verified in both
    directions" against hand-authored plain text while matching zero lines of
    real coloured output.
    """

    program = subprocess.run(
        ["bash", "-c", f"source {RUNNER_LIBRARY}; qfv_pytest_guard_program"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout
    workdir = tmp_path / "suite"
    workdir.mkdir()
    (workdir / "test_guard_demo.py").write_text(
        "import pytest\n"
        "\n"
        "\n"
        "def test_present() -> None:\n"
        "    assert True\n"
        "\n"
        "\n"
        "@pytest.mark.skip(reason='guard demo skip')\n"
        "def test_absent() -> None:\n"
        "    raise AssertionError('must not run')\n"
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    def run_guard(declared: list[str], *, force_colour: bool) -> Any:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-rs",
            "--color=no",
            "test_guard_demo.py",
        ]
        if force_colour:
            # pytest honours the last --color, so this reproduces a colour
            # default that survives --color=no. Only the ANSI strip can save
            # the guard here.
            command.append("--color=yes")
        return subprocess.run(
            [
                "bash",
                "-c",
                program,
                "guard-test",
                str(workdir),
                str(scratch),
                str(len(declared)),
                *declared,
                *command,
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    undeclared = run_guard([], force_colour=False)
    declared = run_guard(["guard demo skip"], force_colour=False)
    undeclared_coloured = run_guard([], force_colour=True)
    declared_coloured = run_guard(["guard demo skip"], force_colour=True)

    assert undeclared.returncode != 0, undeclared.stdout
    assert "undeclared skip in a pytest stage" in undeclared.stdout
    assert "guard demo skip" in undeclared.stdout
    assert declared.returncode == 0, declared.stdout
    assert "undeclared skip" not in declared.stdout
    # The coloured run must really carry escape bytes, or it proves nothing.
    assert "\x1b[" in undeclared_coloured.stdout
    assert undeclared_coloured.returncode != 0, undeclared_coloured.stdout
    assert "undeclared skip in a pytest stage" in undeclared_coloured.stdout
    assert "\x1b[" in declared_coloured.stdout
    assert declared_coloured.returncode == 0, declared_coloured.stdout
    assert "undeclared skip" not in declared_coloured.stdout


def test_runner_library_skip_guard_requires_the_flags_it_depends_on(
    tmp_path: Path,
) -> None:
    program = subprocess.run(
        ["bash", "-c", f"source {RUNNER_LIBRARY}; qfv_pytest_guard_program"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout

    results = {
        flag: subprocess.run(
            [
                "bash",
                "-c",
                program,
                "guard-test",
                str(tmp_path),
                str(tmp_path),
                "0",
                "pytest",
                "-q",
                flag,
                "tests/unit_tests/test_qwen3_formal_single_rank.py",
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for flag in ("-rs", "--color=no")
    }

    assert results["-rs"].returncode != 0
    assert "must pass --color=no" in results["-rs"].stdout
    assert results["--color=no"].returncode != 0
    assert "must pass -rs" in results["--color=no"].stdout


def test_single_rank_runner_routes_every_pytest_stage_through_the_skip_guard() -> None:
    runner_text = RUNNER.read_text()

    assert runner_text.count('bash -lc "$(qfv_pytest_guard_program)"') == 3
    invocations = [
        line
        for line in runner_text.splitlines()
        if "pytest -q" in line and not line.strip().startswith("#")
    ]
    assert len(invocations) == 3
    for line in invocations:
        assert "-rs" in line and "--color=no" in line, line


def test_source_recheck_fails_when_git_diff_check_fails(tmp_path: Path) -> None:
    """The recheck must not print result=success when its last step fails.

    The harness deliberately leaves errexit off, which is the condition the
    real stage runs under: qfv_run_logged does "set +e" before invoking the
    stage function, and errexit is not function-local in bash. Under that
    condition an unguarded "git diff --check" printed result=success regardless.
    """

    baseline = tmp_path / "baseline-status"
    baseline.write_bytes(b"")
    fake_git = tmp_path / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        "  *rev-parse*) printf '%040d\\n' 0 ;;\n"
        "  *status*) exit 0 ;;\n"
        '  *"diff --check"*) exit "${QFV_TEST_DIFF_STATUS}" ;;\n'
        "  *) exit 1 ;;\n"
        "esac\n"
    )
    fake_git.chmod(0o755)

    def run_recheck(diff_status: str) -> Any:
        env = os.environ.copy()
        env["QFV_TEST_DIFF_STATUS"] = diff_status
        return subprocess.run(
            [
                "bash",
                "-c",
                f"source {RUNNER_LIBRARY}; qfv_verify_source_identity "
                '"$1" "$2" "$3" "$4" "$5"',
                "recheck-test",
                str(fake_git),
                str(REPO_ROOT),
                "0" * 40,
                str(baseline),
                "SINGLE_RANK",
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    failing = run_recheck("1")
    passing = run_recheck("0")

    assert failing.returncode != 0, failing.stdout
    assert "git diff --check" in failing.stdout
    assert "SINGLE_RANK_SOURCE_RECHECK result=success" not in failing.stdout
    assert passing.returncode == 0, passing.stdout
    assert "SINGLE_RANK_SOURCE_RECHECK result=success head=" in passing.stdout


def test_source_recheck_names_the_paths_that_moved(tmp_path: Path) -> None:
    """A source-identity failure must name paths, not only report a mismatch.

    The bare message sent me looking for a checker bug when the real cause was
    an edit I had made to a ticket while the gate ran. The assertions below are
    on the paths, because a test that only checked for a non-zero exit would
    pass against the version that named nothing.
    """

    baseline = tmp_path / "baseline-status"
    baseline.write_bytes(b"M  kept.py\x00D  vanished.py\x00")
    fake_git = tmp_path / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        "  *rev-parse*) printf '%040d\\n' 0 ;;\n"
        '  *status*) printf "${QFV_TEST_STATUS}" ;;\n'
        '  *"diff --check"*) exit 0 ;;\n'
        "  *) exit 1 ;;\n"
        "esac\n"
    )
    fake_git.chmod(0o755)

    def run_recheck(status_payload: str) -> Any:
        env = os.environ.copy()
        env["QFV_TEST_STATUS"] = status_payload
        return subprocess.run(
            [
                "bash",
                "-c",
                f"source {RUNNER_LIBRARY}; qfv_verify_source_identity "
                '"$1" "$2" "$3" "$4" "$5"',
                "drift-test",
                str(fake_git),
                str(REPO_ROOT),
                "0" * 40,
                str(baseline),
                "SINGLE_RANK",
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    # "kept.py" is unchanged, "vanished.py" is gone, "appeared.md" is new.
    drifted = run_recheck(r"M  kept.py\0?? appeared.md\0")
    assert drifted.returncode != 0, drifted.stdout
    assert "dirty source identity changed" in drifted.stdout
    assert "appeared during the run: ?? appeared.md" in drifted.stdout
    assert "disappeared during the run: D  vanished.py" in drifted.stdout
    # The entry present in both snapshots must not be reported as moved.
    assert "kept.py" not in drifted.stdout.split("cause:")[0].replace(
        "appeared during the run: ?? appeared.md", ""
    ).replace("disappeared during the run: D  vanished.py", "")
    assert "Re-run without touching the tree." in drifted.stdout

    # An unchanged tree must produce no drift report at all.
    unchanged = run_recheck(r"M  kept.py\0D  vanished.py\0")
    assert unchanged.returncode == 0, unchanged.stdout
    assert "SINGLE_RANK_SOURCE_RECHECK result=success head=" in unchanged.stdout
    assert "appeared during the run" not in unchanged.stdout
    assert "disappeared during the run" not in unchanged.stdout


def test_complete_sealed_bundle_verification_detects_log_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = _create_sealable_bundle(tmp_path, monkeypatch)
    manifest = finalize_evidence_manifest(
        attempt,
        expected_head="a" * 40,
        source_root=tmp_path,
    )

    assert manifest.is_file()
    assert verify_evidence_bundle(
        attempt,
        expected_head="a" * 40,
        source_root=tmp_path,
    )["trace_id"]

    log = attempt / "checker" / "owning-pytest.log"
    log.chmod(0o644)
    log.write_text(log.read_text() + "tampered\n")
    log.chmod(0o444)
    with pytest.raises(ValueError, match="checker/owning-pytest.log"):
        verify_evidence_bundle(
            attempt,
            expected_head="a" * 40,
            source_root=tmp_path,
        )


def test_complete_sealed_bundle_rejects_unmanaged_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = _create_sealable_bundle(tmp_path, monkeypatch)
    finalize_evidence_manifest(
        attempt,
        expected_head="a" * 40,
        source_root=tmp_path,
    )
    unmanaged = attempt / "checker" / "unmanaged.log"
    unmanaged.write_text("not sealed\n")
    unmanaged.chmod(0o444)

    with pytest.raises(ValueError, match="unmanaged or missing files"):
        verify_evidence_bundle(
            attempt,
            expected_head="a" * 40,
            source_root=tmp_path,
        )


def _manifest_with_process(
    entries: list[dict[str, Any]],
    process_entries: list[dict[str, Any]],
    *,
    head_commit_lint_paths: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Hand-build a manifest so forged sections can be exercised.

    The lint-coverage case is derived from the entries exactly as production
    derives it, so a hand-built manifest is not accidentally self-inconsistent
    and the forged-section tests still fail for the reason they name.
    """

    dirty = {
        entry["path"]
        for entry in (*entries, *process_entries)
        if isinstance(entry, dict)
        and isinstance(entry.get("path"), str)
        and entry.get("kind") != "deleted"
    }
    case = LINT_COVERAGE_DIRTY_TREE_BYTES if dirty else LINT_COVERAGE_HEAD_COMMIT_PATHS
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "head": "b" * 40,
        "status_sha256": "0" * 64,
        "entries": entries,
        "source_id": "sha256:" + "0" * 64,
        "process_informational": {
            "note": "n",
            "predicate": "p",
            "entries": process_entries,
        },
        "lint_coverage": {
            "note": "n",
            "case": case,
            "head_commit_paths_sha256": "0" * 64,
            "num_head_commit_paths": len(head_commit_lint_paths),
            "head_commit_lint_paths": list(head_commit_lint_paths),
        },
    }


def _good_process_entry(path_text: str = ".scratch/notes.md") -> dict[str, Any]:
    return {
        "status": "??",
        "path": path_text,
        "kind": "file",
        "mode": "0644",
        "size_bytes": 6,
        "sha256_at_seal": "a" * 64,
        "verified": False,
    }


def test_lint_paths_accepts_a_well_formed_process_section() -> None:
    manifest = _manifest_with_process([], [_good_process_entry()])

    assert source_manifest_lint_paths(manifest) == (".scratch/notes.md",)


@pytest.mark.parametrize(
    "path_text,expected",
    [
        pytest.param("../outside.py", "escapes the checkout", id="parent"),
        pytest.param("/etc/passwd", "escapes the checkout", id="absolute"),
        pytest.param("a/../../b.py", "escapes the checkout", id="embedded_parent"),
        pytest.param("--force", "would parse as an option", id="option"),
        pytest.param("a\nb.py", "contains a delimiter", id="newline"),
    ],
)
def test_lint_paths_rejects_paths_that_must_not_reach_lint(
    path_text: str,
    expected: str,
) -> None:
    """The path list is handed to git add and pre-commit --files verbatim."""

    manifest = _manifest_with_process(
        [{"status": "??", "path": path_text, "kind": "file"}],
        [],
    )

    with pytest.raises(ValueError, match=expected):
        source_manifest_lint_paths(manifest)


@pytest.mark.parametrize(
    "mutate,expected",
    [
        pytest.param(
            lambda e: {**e, "verified": True},
            "verified=false",
            id="claims_verified",
        ),
        pytest.param(
            lambda e: {**e, "sha256_at_seal": "not-hex"},
            "hex sha256_at_seal",
            id="non_hex_digest",
        ),
        pytest.param(
            lambda e: {k: v for k, v in e.items() if k != "sha256_at_seal"},
            "unexpected key set",
            id="missing_digest",
        ),
        pytest.param(
            lambda e: {**e, "sha256": "b" * 64},
            "unexpected key set",
            id="smuggles_verified_sha256_key",
        ),
        pytest.param(
            lambda e: {**e, "kind": "symlink"},
            "may not be symlinks",
            id="symlink",
        ),
        pytest.param(
            lambda e: {**e, "mode": "0755"},
            "may not be executable",
            id="executable_mode",
        ),
        pytest.param(
            lambda e: {**e, "path": ".scratch/tool.PY"},
            "executable or formal source",
            id="uppercase_python_suffix",
        ),
        pytest.param(
            lambda e: {**e, "path": ".scratch/rules.bazel"},
            "executable or formal source",
            id="bazel_suffix",
        ),
        pytest.param(
            lambda e: {**e, "path": ".scratch/Makefile"},
            "prose documents",
            id="extensionless",
        ),
        pytest.param(
            lambda e: {**e, "path": ".scratch/model.cfg"},
            "prose documents",
            id="cfg_suffix",
        ),
        pytest.param(
            lambda e: {**e, "path": ".scratch/script.rb"},
            "prose documents",
            id="unlisted_interpreted_language",
        ),
        pytest.param(
            lambda e: {**e, "path": "scripts/run.sh"},
            "fails the process predicate",
            id="non_process_path",
        ),
        pytest.param(
            lambda e: {**e, "kind": "directory"},
            "unsupported kind",
            id="unsupported_kind",
        ),
    ],
)
def test_forged_process_entries_are_rejected(
    mutate: Any,
    expected: str,
) -> None:
    """A forged process section must never reach lint or verification."""

    manifest = _manifest_with_process([], [mutate(_good_process_entry())])

    with pytest.raises(ValueError, match=expected):
        source_manifest_lint_paths(manifest)


def test_process_entry_cannot_hide_a_present_file_as_deleted(
    tmp_path: Path,
) -> None:
    """The forgery that would drop a live document from lint coverage.

    A ``deleted`` record is skipped by lint. If the file is actually present,
    that record removes a real document from the linted set while leaving
    source verification untouched, because the process section is informational.
    """

    attempt = tmp_path / "attempt"
    (attempt / "manifests").mkdir(parents=True)
    status_path, head_paths_path = _write_source_captures(
        attempt,
        status_bytes=b"",
        head_commit_paths=b"",
    )
    (tmp_path / ".scratch").mkdir()
    (tmp_path / ".scratch" / "notes.md").write_text("present\n")

    forged = _manifest_with_process(
        [{"status": "??", "path": "source.py", "kind": "file"}],
        [
            {
                "status": "??",
                "path": ".scratch/notes.md",
                "kind": "deleted",
                "verified": False,
            }
        ],
    )
    manifest_path = attempt / "manifests" / "source.json"
    _write_immutable(manifest_path, json.dumps(forged), root=attempt)

    # Lint alone cannot see it, which is why verification must.
    assert source_manifest_lint_paths(forged) == ("source.py",)
    with pytest.raises(ValueError, match="recorded deleted but the file is present"):
        verify_source_manifest(
            manifest_path,
            status_path,
            head_paths_path=head_paths_path,
            repo_root=tmp_path,
            expected_head="b" * 40,
            attempt_dir=attempt,
        )


def test_malformed_manifest_raises_value_error_not_programmer_error() -> None:
    """Manifests come from disk, so malformed ones are invalid input."""

    with pytest.raises(ValueError, match="identity key set"):
        source_manifest_lint_paths(
            {
                "schema": SOURCE_MANIFEST_SCHEMA,
                "process_informational": {
                    "note": "n",
                    "predicate": "p",
                    "entries": [],
                },
            }
        )
    manifest = _manifest_with_process([{"path": ".scratch/x.md"}], [])
    with pytest.raises(ValueError, match="never reach the verified source entry list"):
        source_manifest_lint_paths(manifest)


def test_process_membership_changes_source_id_but_content_does_not(
    tmp_path: Path,
) -> None:
    """Pin both halves of the content-versus-membership boundary.

    Tracker text may drift without changing identity; adding a tracker path is a
    change to the attempt's path set and does change it, through status_sha256.
    """

    (tmp_path / "source.py").write_text("value = 1\n")
    scratch = tmp_path / ".scratch"
    scratch.mkdir()
    (scratch / "one.md").write_text("one\n")
    status_one = b"?? source.py\0?? .scratch/one.md\0"

    baseline = build_source_manifest(
        tmp_path, "b" * 40, status_one, head_commit_paths=b""
    )

    (scratch / "one.md").write_text("substantially rewritten tracker text\n")
    text_changed = build_source_manifest(
        tmp_path, "b" * 40, status_one, head_commit_paths=b""
    )
    assert text_changed["source_id"] == baseline["source_id"]

    (scratch / "two.md").write_text("two\n")
    status_two = b"?? source.py\0?? .scratch/one.md\0?? .scratch/two.md\0"
    path_added = build_source_manifest(
        tmp_path, "b" * 40, status_two, head_commit_paths=b""
    )
    assert path_added["source_id"] != baseline["source_id"]
    assert path_added["entries"] == baseline["entries"]


def test_process_roster_omission_is_rejected(tmp_path: Path) -> None:
    """Shape validation cannot catch an entry that simply is not there.

    Omitting a process entry would drop that document from lint coverage while
    source verification still passed, because process bytes are outside
    ``source_id``. Only a completeness check against the status closes it.
    """

    (tmp_path / "source.py").write_text("value = 1\n")
    scratch = tmp_path / ".scratch"
    scratch.mkdir()
    (scratch / "kept.md").write_text("kept\n")
    (scratch / "hidden.md").write_text("hidden\n")
    status = b"?? source.py\0?? .scratch/kept.md\0?? .scratch/hidden.md\0"

    sealed = build_source_manifest(tmp_path, "b" * 40, status, head_commit_paths=b"")
    section = cast(dict[str, Any], sealed["process_informational"])
    assert len(cast(list[dict[str, Any]], section["entries"])) == 2

    forged = json.loads(json.dumps(sealed))
    forged["process_informational"]["entries"] = [
        entry
        for entry in forged["process_informational"]["entries"]
        if entry["path"] != ".scratch/hidden.md"
    ]
    # The forgery is invisible to lint on its own -- it just returns less.
    assert ".scratch/hidden.md" not in source_manifest_lint_paths(forged)

    with pytest.raises(ValueError, match="process roster does not match"):
        build_source_manifest(
            tmp_path, "b" * 40, status, head_commit_paths=b"", expected=forged
        )


def test_process_roster_addition_is_rejected(tmp_path: Path) -> None:
    (tmp_path / ".scratch").mkdir()
    (tmp_path / ".scratch" / "real.md").write_text("real\n")
    status = b"?? .scratch/real.md\0"
    sealed = build_source_manifest(tmp_path, "b" * 40, status, head_commit_paths=b"")

    forged = json.loads(json.dumps(sealed))
    forged["process_informational"]["entries"].append(
        {
            "status": "??",
            "path": ".scratch/phantom.md",
            "kind": "file",
            "mode": "0644",
            "size_bytes": 1,
            "sha256_at_seal": "c" * 64,
            "verified": False,
        }
    )

    with pytest.raises(ValueError, match="process roster does not match"):
        build_source_manifest(
            tmp_path, "b" * 40, status, head_commit_paths=b"", expected=forged
        )


@pytest.mark.parametrize(
    "bad_manifest",
    [
        pytest.param("not a mapping", id="string"),
        pytest.param(["entries"], id="list"),
        pytest.param(None, id="none"),
    ],
)
def test_malformed_manifest_root_raises_value_error(bad_manifest: Any) -> None:
    """Malformed input must be ValueError, never AttributeError or TypeError."""

    with pytest.raises(ValueError, match="must be a mapping"):
        source_manifest_lint_paths(bad_manifest)


def test_deleting_a_captured_process_path_fails_verification(
    tmp_path: Path,
) -> None:
    """Removal is not symmetric with addition, and the README says so."""

    (tmp_path / ".scratch").mkdir()
    tracker = tmp_path / ".scratch" / "notes.md"
    tracker.write_text("notes\n")
    status = b"?? .scratch/notes.md\0"
    sealed = build_source_manifest(tmp_path, "b" * 40, status, head_commit_paths=b"")

    tracker.unlink()
    with pytest.raises(ValueError, match="missing without deletion status"):
        build_source_manifest(
            tmp_path, "b" * 40, status, head_commit_paths=b"", expected=sealed
        )
