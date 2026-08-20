# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
import subprocess

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ISSUE_DIR = REPO_ROOT / ".scratch" / "modded-nanogpt-b200" / "issues"
TRACKER_ROOT = REPO_ROOT / ".scratch" / "modded-nanogpt-b200"

EXPECTED_ISSUE_HEADERS = {
    "01-source-and-preflight.md": {
        "Status": "resolved",
        "Blocked by": "-",
        "Blocks": None,
    },
    "02-data-and-manifest.md": {
        "Status": "resolved",
        "Blocked by": "-",
        "Blocks": None,
    },
    "03-reproduction-wrapper-and-parser.md": {
        "Status": "resolved",
        "Blocked by": "-",
        "Blocks": None,
    },
    "04-b200-ablation-matrix.md": {
        "Status": "blocked",
        "Blocked by": (
            "Lane A or Lane B non-skip baseline; active Lane B path requires "
            "trusted user request containing `launch-full-b200`"
        ),
        "Blocks": None,
    },
    "05-harness-summary-and-run-index.md": {
        "Status": "resolved",
        "Blocked by": "-",
        "Blocks": None,
    },
    "06-lane-b-compatibility-baseline.md": {
        "Status": "blocked",
        "Blocked by": "trusted user request containing `launch-full-b200`",
        "Blocks": None,
    },
    "07-bwrap-rootfs-plan-verification.md": {
        "Status": "complete",
        "Blocked by": "-",
        "Blocks": None,
    },
    "08-optimized-kernel-certification.md": {
        "Status": "complete",
        "Blocked by": "-",
        "Blocks": None,
    },
    "09-two-gpu-trial-retro-and-next-iteration.md": {
        "Status": "blocked",
        "Blocked by": "trusted user request containing `launch-full-b200`",
        "Blocks": None,
    },
    "10-experiment-schema-foundation.md": {
        "Status": "complete",
        "Blocked by": "-",
        "Blocks": None,
    },
    "11-experiment-plan-materialization.md": {
        "Status": "complete",
        "Blocked by": "-",
        "Blocks": None,
    },
    "12-experiment-matrix-runner.md": {
        "Status": "complete",
        "Blocked by": "-",
        "Blocks": None,
    },
    "13-observability-profiles-and-rsi-report.md": {
        "Status": "complete",
        "Blocked by": "-",
        "Blocks": None,
    },
    "14-nanogpt-performance-probe-ladder.md": {
        "Status": "complete",
        "Blocked by": "-",
        "Blocks": "06, 09, 04",
    },
    "15-documentation-review-and-handoff.md": {
        "Status": "complete",
        "Blocked by": "-",
        "Blocks": None,
    },
}


def _approved_wrappers(text: str, start_marker: str, end_marker: str) -> list[str]:
    section = text.split(start_marker, 1)[1].split(end_marker, 1)[0]
    return re.findall(r"`(experiments/modded_nanogpt_b200/[^`]+\.sh)`", section)


def _header_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    current_key: str | None = None
    for line in path.read_text().splitlines()[1:8]:
        if line.startswith("## "):
            break
        if line.startswith("  ") and current_key is not None:
            fields[current_key] = f"{fields[current_key]} {line.strip()}"
            continue
        current_key = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in {"Type", "Status", "Blocked by", "Blocks"}:
            fields[key] = value.strip()
            current_key = key
    return fields


def test_tracker_issue_headers_preserve_current_blocked_state():
    issue_paths = sorted(ISSUE_DIR.glob("*.md"))
    assert issue_paths
    assert {path.name for path in issue_paths} == set(EXPECTED_ISSUE_HEADERS)

    observed_blocked: set[str] = set()
    for path in issue_paths:
        fields = _header_fields(path)
        expected = EXPECTED_ISSUE_HEADERS[path.name]

        assert fields.get("Type") == "task", path.name
        assert fields.get("Status") == expected["Status"], path.name
        assert fields.get("Blocked by") == expected["Blocked by"], path.name
        if expected["Blocks"] is None:
            assert "Blocks" not in fields, path.name
        else:
            assert fields.get("Blocks") == expected["Blocks"], path.name

        if fields["Status"] == "blocked":
            observed_blocked.add(path.name)
        else:
            assert fields["Blocked by"] == "-", path.name

    assert observed_blocked == {
        name
        for name, expected in EXPECTED_ISSUE_HEADERS.items()
        if expected["Status"] == "blocked"
    }


def test_active_handoff_docs_preserve_trusted_message_launch_authority():
    active_docs = [
        TRACKER_ROOT / "spec.md",
        TRACKER_ROOT / "master_plan.md",
        TRACKER_ROOT / "completion_audit.md",
        TRACKER_ROOT / "execution_prompt.md",
        TRACKER_ROOT / "rootfs_runtime_env_spec.md",
        ISSUE_DIR / "04-b200-ablation-matrix.md",
        ISSUE_DIR / "06-lane-b-compatibility-baseline.md",
        ISSUE_DIR / "09-two-gpu-trial-retro-and-next-iteration.md",
        ISSUE_DIR / "14-nanogpt-performance-probe-ladder.md",
        ISSUE_DIR / "15-documentation-review-and-handoff.md",
        REPO_ROOT / "experiments" / "modded_nanogpt_b200" / "preflight_checklist.md",
    ]
    stale_phrases = [
        "host-side tests",
        "same meaning as `training_launch_allowed`",
        "trusted request did not include `--launch-authorization=launch-full-b200`",
        "trusted request does not contain `--launch-authorization=launch-full-b200`",
    ]

    for path in active_docs:
        text = path.read_text()
        for phrase in stale_phrases:
            assert phrase not in text, path

    issue_09 = (ISSUE_DIR / "09-two-gpu-trial-retro-and-next-iteration.md").read_text()
    assert "trusted user request did\n  not contain `launch-full-b200`" in issue_09
    assert (
        "wrapper or CLI authorization flags are\n"
        "  lower-level launch markers, not primary authority"
    ) in issue_09


def test_issue_03_preserves_trusted_message_launch_authority():
    text = (ISSUE_DIR / "03-reproduction-wrapper-and-parser.md").read_text()

    assert (
        "Full-mode launches require the trusted user request itself to contain\n"
        "    `launch-full-b200`; only after that may the lower-level\n"
        "    `--launch-authorization=launch-full-b200` marker be supplied."
    ) in text
    assert (
        "Full-mode launches require explicit `--launch-authorization=launch-full-b200`\n"
        "    after preflight passes"
    ) not in text


def test_issue_06_authority_guard_preserves_trusted_message_boundary():
    text = (ISSUE_DIR / "06-lane-b-compatibility-baseline.md").read_text()

    assert (
        "runner stops before `torchrun` unless the trusted user request contains\n"
        "    `launch-full-b200` and the lower-level\n"
        "    `--launch-authorization=launch-full-b200` marker is supplied."
    ) in text
    assert (
        "runner stops before `torchrun` unless\n"
        "    `--launch-authorization=launch-full-b200` is present."
    ) not in text


def test_execution_prompt_guards_full_launch_template_before_token_snippet():
    text = (TRACKER_ROOT / "execution_prompt.md").read_text()
    section = text.split("Lane B full production candidate shape:", 1)[1].split(
        "For a prerequisite dry gate", 1
    )[0]

    warning = (
        "Use this non-skip template only after the trusted user request contains\n"
        "`launch-full-b200`; otherwise use the dry-gate form below without launch\n"
        "authorization."
    )
    token = "--launch-authorization=launch-full-b200"

    assert warning in section
    assert token in section
    assert section.index(warning) < section.index(token)


def test_rootfs_runtime_spec_preserves_skip_run_readiness_semantics():
    text = (TRACKER_ROOT / "rootfs_runtime_env_spec.md").read_text()

    assert "same meaning as `training_launch_allowed`" not in text
    assert (
        "A skip-run\n"
        "  artifact may record `ready_to_launch=true` as prerequisite evidence"
    ) in text
    assert (
        "still records `training_launched=false` and remains excluded from baseline\n"
        "  stats and non-skip launch-ready rows"
    ) in text


def test_execution_prompt_preserves_active_two_gpu_allocation():
    execution_prompt = (TRACKER_ROOT / "execution_prompt.md").read_text()
    spec = (TRACKER_ROOT / "spec.md").read_text()

    assert "  - 8x `NVIDIA B200`, compute capability `(10, 0)`" not in (
        execution_prompt
    )
    assert "- hardware: 8x `NVIDIA B200`, compute capability `(10, 0)`" not in spec
    assert (
        "the active RSI foundation launch allocation is exactly 2 visible B200 GPUs"
        in execution_prompt
    )
    assert (
        "the active RSI foundation launch allocation is exactly 2 visible\n"
        "  B200 GPUs" in spec
    )
    assert "Broader 8x\nreproduction" in execution_prompt


def test_issue_06_preserves_selected_triton_mlp_policy():
    text = (ISSUE_DIR / "06-lane-b-compatibility-baseline.md").read_text()

    assert "require Triton MLP and PyTorch MLP" not in text
    assert (
        "require the selected Triton MLP path to pass local forward/backward smoke\n"
        "  before the next full job"
    ) in text
    assert "keep PyTorch MLP diagnostic-only and blocked for full jobs" in text


def test_execution_prompt_uses_rootfs_data_preparation_helper():
    text = (TRACKER_ROOT / "execution_prompt.md").read_text()

    assert "\npython data/cached_fineweb10B.py 9\n" not in text
    assert "experiments/modded_nanogpt_b200/prepare_data.sh \\" in text
    assert "Do not run that upstream command directly from the host" in text


def test_active_docs_use_runtime_sync_for_python_and_tool_dependencies():
    execution_prompt = (TRACKER_ROOT / "execution_prompt.md").read_text()
    checklist = (
        REPO_ROOT / "experiments" / "modded_nanogpt_b200" / "preflight_checklist.md"
    ).read_text()

    assert "python -m pip install --break-system-packages" not in execution_prompt
    assert "python -m pip install --break-system-packages" not in checklist
    assert "experiments/modded_nanogpt_b200/runtime/sync_python_env.sh" in (
        execution_prompt
    )
    assert "experiments/modded_nanogpt_b200/runtime/sync_python_env.sh" in checklist
    assert "experiments/modded_nanogpt_b200/runtime/sync_tools.sh" in checklist
    assert "Regenerate the direct runtime lock" in checklist


def test_preflight_checklist_approved_wrappers_match_canonical_spec():
    spec = (TRACKER_ROOT / "spec.md").read_text()
    checklist = (
        REPO_ROOT / "experiments" / "modded_nanogpt_b200" / "preflight_checklist.md"
    ).read_text()

    spec_wrappers = _approved_wrappers(
        spec,
        "The current approved wrappers are:",
        "Every wrapper",
    )
    checklist_wrappers = _approved_wrappers(
        checklist,
        "The approved wrappers for this experiment are:",
        "If a new executable step",
    )

    assert checklist_wrappers == spec_wrappers


def test_canonical_approved_wrappers_exist_and_enforce_rootfs_boundary():
    spec = (TRACKER_ROOT / "spec.md").read_text()
    spec_wrappers = _approved_wrappers(
        spec,
        "The current approved wrappers are:",
        "Every wrapper",
    )

    for wrapper in spec_wrappers:
        path = REPO_ROOT / wrapper
        text = path.read_text()

        assert path.exists(), wrapper
        assert text.startswith("#!/usr/bin/env bash\n"), wrapper
        assert "enter_rootfs.sh" in text, wrapper
        assert 'source "${SCRIPT_DIR}/rootfs_guard.sh"' in text or (
            'source "${EXPERIMENT_DIR}/rootfs_guard.sh"' in text
        ), wrapper
        assert "require_modded_nanogpt_rootfs" in text, wrapper
        assert path.stat().st_mode & 0o111, wrapper


def test_sourced_shell_helpers_are_not_executable():
    helper_paths = [
        REPO_ROOT / "experiments" / "modded_nanogpt_b200" / "rootfs_guard.sh",
        REPO_ROOT / "scripts" / "rootfs" / "rootfs_target.sh",
        REPO_ROOT / "scripts" / "rootfs" / "runtime_env.sh",
    ]

    for path in helper_paths:
        assert path.exists(), path
        assert not (path.stat().st_mode & 0o111), path


def test_generated_nanogpt_artifacts_are_not_tracked():
    generated_patterns = [
        "experiments/modded_nanogpt_b200/results/**",
        "experiments/modded_nanogpt_b200/data/**",
        "experiments/modded_nanogpt_b200/sources/**",
        "experiments/modded_nanogpt_b200/__pycache__/**",
        ".scratch/trae-pytest-tmp/**",
    ]

    proc = subprocess.run(
        ["git", "ls-files", *generated_patterns],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert proc.stdout == ""


def test_default_compiler_output_is_ignored_and_static_checked():
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "\na.out\n" in gitignore

    proc = subprocess.run(
        ["git", "check-ignore", "-q", "a.out"],
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0

    static_verifier = (
        REPO_ROOT / "experiments" / "modded_nanogpt_b200" / "verify_static.py"
    ).read_text()
    assert '".gitignore",' in static_verifier


def test_active_handoff_docs_have_no_unresolved_placeholders():
    active_docs = [
        TRACKER_ROOT / "spec.md",
        TRACKER_ROOT / "master_plan.md",
        TRACKER_ROOT / "rootfs_runtime_env_spec.md",
        TRACKER_ROOT / "schema_matrix_spec.md",
        TRACKER_ROOT / "execution_prompt.md",
        REPO_ROOT / "experiments" / "modded_nanogpt_b200" / "preflight_checklist.md",
    ]
    active_docs.extend(sorted(ISSUE_DIR.glob("*.md")))
    unresolved_markers = ("TODO", "TBD", "FIXME", "XXX")

    for path in active_docs:
        text = path.read_text()
        for marker in unresolved_markers:
            assert marker not in text, path


def test_active_handoff_preserves_sequential_execution_boundary():
    master_plan = (TRACKER_ROOT / "master_plan.md").read_text()
    execution_prompt = (TRACKER_ROOT / "execution_prompt.md").read_text()
    checklist = (
        REPO_ROOT / "experiments" / "modded_nanogpt_b200" / "preflight_checklist.md"
    ).read_text()

    assert (
        "Run NanoGPT attempts sequentially only. A new attempt may start only after the\n"
        "  previous attempt has finished, artifacts have been summarized, and the\n"
        "  active-job scan proves no concurrent NanoGPT, `torchrun`, or data-prep work"
    ) in master_plan
    assert (
        "superseded until that first trial is stopped, parsed, summarized, and\n"
        "classified from preserved evidence."
    ) in execution_prompt
    assert (
        "Do not start another NanoGPT attempt until\n"
        "  the previous attempt has finished, been parsed, been summarized, and been\n"
        "  classified from preserved evidence."
    ) in checklist
    assert "10 full-job target" not in execution_prompt


def test_active_handoff_has_no_current_8x_or_10_run_requirement_drift():
    active_docs = [
        REPO_ROOT / ".claude" / "CLAUDE.md",
        TRACKER_ROOT / "spec.md",
        TRACKER_ROOT / "master_plan.md",
        TRACKER_ROOT / "schema_matrix_spec.md",
        TRACKER_ROOT / "execution_prompt.md",
        ISSUE_DIR / "01-source-and-preflight.md",
        ISSUE_DIR / "03-reproduction-wrapper-and-parser.md",
        ISSUE_DIR / "05-harness-summary-and-run-index.md",
        ISSUE_DIR / "06-lane-b-compatibility-baseline.md",
        ISSUE_DIR / "11-experiment-plan-materialization.md",
    ]
    stale_current_requirements = [
        "Full jobs require 8x B200",
        "Full jobs require rootfs sentinel evidence, NCCL, 8x B200",
        "full jobs require `8x B200`",
        "full jobs require 8 visible CUDA devices",
        "exactly 8x B200",
        "exactly 8 visible B200",
        "parser success for full attempts requires GPU inventory exactly 8x B200",
        "claim policy requires 8 GPUs",
        "dry gates do not count toward the 10 full-job target",
        "fewer than 10 full jobs",
        "10 full-job target",
    ]

    for path in active_docs:
        text = path.read_text()
        for phrase in stale_current_requirements:
            assert phrase not in text, f"{path}: {phrase}"

    claude = (REPO_ROOT / ".claude" / "CLAUDE.md").read_text()
    assert "the declared B200 allocation" in claude
    assert "exactly two visible B200 GPUs" in claude

    execution_prompt = (TRACKER_ROOT / "execution_prompt.md").read_text()
    assert "The older 10 full-job campaign target is\nhistorical" in execution_prompt
    assert (
        "A broader 8x B200 reproduction or production baseline still requires\n"
        "  explicit authorization"
    ) in execution_prompt


def test_claude_mirror_preserves_canonical_nanogpt_rootfs_guidance():
    agents = (REPO_ROOT / "AGENTS.md").read_text()
    claude = (REPO_ROOT / ".claude" / "CLAUDE.md").read_text()
    start = "## Repo-Local Experiment Discipline"
    end = "## Build and Test"

    assert claude == agents

    agents_section = agents[agents.index(start) : agents.index(end)]
    claude_section = claude[claude.index(start) : claude.index(end)]

    assert claude_section == agents_section
    assert "scripts/rootfs/enter_rootfs.sh" in claude_section
    assert "For `experiments/modded_nanogpt_b200`" in claude_section
    assert "the declared B200 allocation" in claude_section
    assert "exactly two visible B200 GPUs" in claude_section


def test_master_plan_distinguishes_complete_issue_tickets_from_open_launch_tasks():
    text = (TRACKER_ROOT / "master_plan.md").read_text()

    assert (
        "Issue tickets `10`-`15` and `08` are complete for the current non-launch\n"
        "foundation. Master-plan Tasks `10`-`14` remain launch- or baseline-dependent."
    ) in text
    assert "## Task 10: First Authorized B200-Compatible Full Baseline" in text
    assert (
        "Consumes: latest prerequisite evidence and a trusted user request containing\n"
        "  `launch-full-b200`."
    ) in text
    assert "## Task 12: Sequential Two-GPU Trial Repeatability" in text
    assert "Consumes: one successful B200-compatible two-GPU trial." in text
    assert "## Task 14: Optional Ablation Matrix" in text
    assert (
        "Consumes: accepted faithful-upstream or B200-compatible baseline artifact."
        in text
    )
    assert (
        "The next full launch still\n"
        "requires the trusted user request itself to contain `launch-full-b200`."
    ) in text
    assert "Unchecked task blocker map:" in text
    assert (
        "- Task 10 is blocked by the missing trusted `launch-full-b200` request."
        in text
    )
    assert (
        "- Task 11 is blocked until Task 10 produces a real full-run artifact." in text
    )
    assert (
        "- Task 12 is blocked until one B200-compatible two-GPU trial succeeds." in text
    )
    assert (
        "- Task 13 is blocked until a material FA3/B200 kernel input changes." in text
    )
    assert (
        "- Task 14 is blocked until an accepted faithful-upstream or B200-compatible\n"
        "  baseline artifact exists."
    ) in text


def test_two_gpu_convenience_launcher_preserves_authority_boundary():
    text = (
        REPO_ROOT
        / "experiments"
        / "modded_nanogpt_b200"
        / "launch_nanogpt_2gpu_full_rootfs.sh"
    ).read_text()

    assert 'AUTH_ENV_NAME="MODDED_NANOGPT_2GPU_FULL_LAUNCH_AUTHORIZATION"' in text
    assert 'FULL_LAUNCH_AUTHORIZATION_TOKEN="launch-full-b200"' in text
    assert (
        "set ${AUTH_ENV_NAME}=${FULL_LAUNCH_AUTHORIZATION_TOKEN} "
        "only after the trusted user request contains launch-full-b200"
    ) in text
    assert '--launch-authorization="${!AUTH_ENV_NAME}" \\' in text
    assert "--launch-authorization=launch-full-b200 \\" not in text


def test_spec_current_verification_summary_uses_latest_static_count():
    text = (TRACKER_ROOT / "spec.md").read_text()
    implementation_state = text.split("## Implementation State", 1)[1].split(
        "## Rootfs Contract", 1
    )[0]

    assert "Static verification passed for 114 file(s)" in implementation_state
    assert "Static verification passed for 101 file(s)" not in implementation_state
    assert "Static verification passed for 61 file(s)" not in implementation_state
    assert "Static verification passed for 60 file(s)" not in implementation_state
    assert "Static verification passed for 59 file(s)" not in implementation_state
    assert "Static verification passed for 41 file(s)" not in implementation_state
    assert "Static verification passed for 36 file(s)" not in implementation_state


def test_static_verifier_spec_preserves_json_validation_contract():
    text = (TRACKER_ROOT / "spec.md").read_text()
    static_spec = text.split("## Static Harness Verifier Spec", 1)[1].split(
        "## Current Known Runtime Facts", 1
    )[0]

    assert (
        "support `--json` for machine-readable candidate and check results with\n"
        "  top-level `ok`, `files`, and `errors`"
    ) in static_spec
    assert "make `--json` run validation" in static_spec
    assert "return `0` when `ok=true`" in static_spec
    assert (
        "return nonzero\n" "  while still printing JSON when validation fails"
    ) in static_spec


def test_master_plan_current_state_uses_latest_non_launch_verification_bundle():
    text = (TRACKER_ROOT / "master_plan.md").read_text()
    current_state = text.split("## Current State", 1)[1].split(
        "## Sequential Milestone Ladder", 1
    )[0]
    normalized_current_state = re.sub(r"\s+", " ", current_state)

    assert "`410 passed, 2 skipped`" in current_state
    assert "`410 passed, 2 skipped in 43.34s`" not in current_state
    assert "`49 passed`" in current_state
    assert "`48 passed`" not in current_state
    assert "`62 passed`" not in current_state
    assert "31-file static Python" in current_state
    assert "surface reports `0 errors`" in current_state
    assert "Static verification passed for 114 file(s)" in current_state
    assert "Static verification passed for 101 file(s)" not in current_state
    assert (
        "`SKIP=no-commit-to-branch pre-commit run --all-files`"
        in normalized_current_state
    )
    assert "Full `pre-commit run --all-files` remains unproven" not in (
        normalized_current_state
    )
    assert "`406 passed, 2 skipped`" not in current_state
    assert "`53 passed`" not in current_state
    assert "`97 passed`" not in current_state
    assert "`366 passed, 2 skipped`" not in current_state
    assert "`376 passed, 2 skipped`" not in current_state
    assert "`362 passed, 2 skipped`" not in current_state
    assert "`385 passed, 2 skipped`" not in current_state
    assert "`52 passed`" not in current_state
    assert "`332 passed, 2 skipped in 42.41s`" not in current_state
    assert "Static verification passed for 61 file(s)" not in current_state
    assert "Static verification passed for 100 file(s)" not in current_state
    assert "`166 passed, 2 skipped in 29.16s`" not in current_state
    assert "Static verification passed for 41 file(s)" not in current_state


def test_master_plan_task_15_uses_latest_final_non_launch_verification():
    text = (TRACKER_ROOT / "master_plan.md").read_text()
    task_15 = text.split("## Task 15: Documentation, Review, and Handoff", 1)[1]
    normalized_task_15 = re.sub(r"\s+", " ", task_15)

    assert "Current final non-launch verification:" in task_15
    assert "-> Static verification passed for 114 file(s)." in task_15
    assert "-> Static verification passed for 101 file(s)." not in task_15
    assert "-> 410 passed, 2 skipped" in task_15
    assert "-> 410 passed, 2 skipped in 43.34s" not in task_15
    assert "-> 49 passed" in task_15
    assert "-> 48 passed" not in task_15
    assert "-> 62 passed" not in task_15
    assert "test_modded_nanogpt_b200_tracker.py" in task_15
    assert "python experiments/modded_nanogpt_b200/verify_static.py" in task_15
    assert "31-file static Python surface" in task_15
    assert "text/lock hygiene for runtime dependency files" in normalized_task_15
    assert "`SKIP=no-commit-to-branch pre-commit run --all-files`" in normalized_task_15
    assert "Full `pre-commit run --all-files` remains unproven" not in (
        normalized_task_15
    )
    assert "-> 406 passed, 2 skipped" not in task_15
    assert "-> 53 passed" not in task_15
    assert "`97 passed`" not in task_15
    assert "-> 366 passed, 2 skipped" not in task_15
    assert "-> 376 passed, 2 skipped" not in task_15
    assert "-> 362 passed, 2 skipped" not in task_15
    assert "-> 385 passed, 2 skipped" not in task_15
    assert "-> 52 passed" not in task_15
    assert "-> Static verification passed for 61 file(s)." not in task_15
    assert "-> Static verification passed for 100 file(s)." not in task_15
    assert "-> 332 passed, 2 skipped in 42.41s" not in task_15
    assert "-> 43 passed in 0.26s" not in task_15
    assert "-> 213 passed, 2 skipped in 13.29s" not in task_15
    assert "-> Static verification passed for 38 file(s)." not in task_15
    assert "-> 393 passed, 2 skipped" not in task_15
    assert "Final authority-bound audit verification:" not in task_15
    assert "-> Static verification passed for 58 file(s)." not in task_15


def test_active_specs_use_latest_non_launch_verification_bundle():
    active_specs = [
        TRACKER_ROOT / "schema_matrix_spec.md",
        TRACKER_ROOT / "rootfs_runtime_env_spec.md",
        REPO_ROOT / "experiments" / "modded_nanogpt_b200" / "preflight_checklist.md",
    ]

    for path in active_specs:
        text = path.read_text()
        normalized_text = re.sub(r"\s+", " ", text)
        assert "`410 passed, 2 skipped`" in text, path
        assert "`410 passed, 2 skipped in 43.34s`" not in text, path
        assert "`49 passed`" in text, path
        assert "`48 passed`" not in text, path
        assert "`62 passed`" not in text, path
        assert "31-file static Python" in text, path
        assert "surface" in text, path
        assert "text/lock hygiene for runtime dependency files" in normalized_text, path
        assert "Static verification passed for 114 file(s)" in text, path
        assert "Static verification passed for 101 file(s)" not in text, path
        assert (
            "`SKIP=no-commit-to-branch pre-commit run --all-files`" in normalized_text
        ), path
        assert "Full `pre-commit run --all-files` remains unproven" not in (
            normalized_text
        ), path
        assert "`406 passed, 2 skipped`" not in text, path
        assert "`53 passed`" not in text, path
        assert "`97 passed`" not in text, path
        assert "`366 passed, 2 skipped`" not in text, path
        assert "`376 passed, 2 skipped`" not in text, path
        assert "`362 passed, 2 skipped`" not in text, path
        assert "`385 passed, 2 skipped`" not in text, path
        assert "`52 passed`" not in text, path
        assert "`332 passed, 2 skipped in 42.41s`" not in text, path
        assert "Static verification passed for 61 file(s)" not in text, path
        assert "Static verification passed for 100 file(s)" not in text, path
        assert "`284 passed, 2 skipped in 42.46s`" not in text, path
        assert "Static verification passed for 41 file(s)" not in text, path


def test_preflight_checklist_names_launch_readiness_sidecar_facts_precisely():
    text = (
        REPO_ROOT / "experiments" / "modded_nanogpt_b200" / "preflight_checklist.md"
    ).read_text()
    current_gate = text.split("The current Lane B full dry gate is:", 1)[1].split(
        "A full-job preflight must include NCCL", 1
    )[0]

    assert "- launch-readiness facts: 2x B200" in current_gate
    assert "- attempt facts: 2x B200" not in current_gate
    assert (
        "the ignored generated run index preserves\n  four prerequisite rows for history"
        in current_gate
    )
    assert "`launch_prerequisite_attempts=4`" in current_gate
    assert (
        "only the latest strict runtime-env row is the\n"
        "  current validated prerequisite handoff artifact" in current_gate
    )
    assert (
        "only\n  `lane_b_full_skiprun_runtime_env_refresh_20260819T131652Z` currently passes"
        in current_gate
    )
    assert (
        "Treat older prerequisite rows as\n  historical/superseded evidence"
        in current_gate
    )


def test_completion_audit_preserves_prompt_to_artifact_checklist_and_blocker():
    text = (TRACKER_ROOT / "completion_audit.md").read_text()
    checklist = text.split("## Prompt-to-Artifact Checklist", 1)[1].split(
        "## Completion Decision", 1
    )[0]
    normalized_checklist = re.sub(r"\s+", " ", checklist)
    decision = text.split("## Completion Decision", 1)[1].split("2026-08-19", 1)[0]

    for required_row in (
        "- Rootfs-only execution boundary:",
        "- Schema-governed attempt artifacts:",
        "- Source, data, runtime, hardware, kernel, and active-job gates:",
        "- Runtime Python and non-launch verification bundle:",
        "- Launch-readiness prerequisite artifact:",
        "- Baseline stats and final validation: not satisfied.",
        "- Sequential two-GPU launch policy: still binding.",
        "- Remaining blocked launch-ladder tasks in `master_plan.md`:",
    ):
        assert required_row in checklist

    assert "Updated: 2026-08-20" in text
    assert "`total_attempts=63`" in checklist
    assert "baseline_stats.count=0" in checklist
    assert "`launch_prerequisite_attempts.count=4`" in checklist
    assert "`len(launch_prerequisite_attempts)=1`" not in checklist
    assert "`launch_ready_attempts.count=0`" in checklist
    assert "Only the latest strict runtime-env row" in checklist
    assert "historical skip-run dry gates" in checklist
    assert "`410 passed, 2 skipped`" in checklist
    assert "`410 passed, 2 skipped in 43.34s`" not in checklist
    assert "`49 passed`" in checklist
    assert "`48 passed`" not in checklist
    assert "`62 passed`" not in checklist
    assert "Static verification passed for 114 file(s)" in checklist
    assert "Static verification passed for 101 file(s)" not in checklist
    assert (
        "`SKIP=no-commit-to-branch pre-commit run --all-files`" in normalized_checklist
    )
    assert "Full `pre-commit run --all-files` remains unproven" not in (
        normalized_checklist
    )
    assert "`406 passed, 2 skipped`" not in checklist
    assert "`53 passed`" not in checklist
    assert "Static verification passed for 36 file(s)" not in checklist
    assert "`385 passed, 2 skipped`" not in checklist
    assert "`52 passed`" not in checklist
    assert "Static verification passed for 100 file(s)" not in checklist
    assert "31-file static Python surface reports `0 errors`" in normalized_checklist
    assert "`97 passed`" not in normalized_checklist
    assert "MODDED_NANOGPT_RUNTIME_VENV" in checklist
    assert "zero non-skip launch-ready rows" in normalized_checklist
    assert "successful_b200_reproduction=false" in checklist
    assert "not complete" in decision
    assert "Do not call `update_goal`" in decision
    assert "no full attempt has reached final validation" in decision


def test_completion_audit_top_snapshot_uses_current_generated_run_index_counts():
    text = (TRACKER_ROOT / "completion_audit.md").read_text()
    snapshot = text.split("## Current Evidence Snapshot", 1)[1].split(
        "## Prompt-to-Artifact Checklist", 1
    )[0]

    assert "Current generated run-index state:" in snapshot
    assert "`total_attempts=63`" in snapshot
    assert "`baseline_stats.count=0`" in snapshot
    assert "`launch_prerequisite_attempts.count=4`" in snapshot
    assert "`launch_ready_attempts.count=0`" in snapshot
    assert "`len(launch_prerequisite_attempts)=1`" not in snapshot
    assert "`len(launch_ready_attempts)=0`" not in snapshot
    assert "one current\n  strict launch-prerequisite row" not in snapshot
    assert "one current strict launch-prerequisite row" not in snapshot
    assert "lane_b_full_skiprun_runtime_env_refresh_20260819T131652Z" in snapshot
    assert "historical skip-run dry gates" in snapshot
    assert "was read but not regenerated" in snapshot


def test_completion_audit_preserves_latest_rsi_checkpoint_evidence():
    text = (TRACKER_ROOT / "completion_audit.md").read_text()
    checkpoint = text.split(
        "2026-08-20 RSI foundation completion-audit checkpoint:", 1
    )[1]

    assert "Objective restated as\nconcrete deliverables" in checkpoint
    assert "continue sequential experiment execution" in checkpoint
    assert "launch-full-b200" in checkpoint
    assert "Fresh rootfs\n  verification reported `4 passed`" in checkpoint
    assert (
        "selected runtime Python coverage reported\n  `1 passed, 9 deselected`"
        in checkpoint
    )
    assert "Fresh rootfs\n  verification reported `43 passed`" in checkpoint
    assert "`static_json.file_count=97`" in checkpoint
    assert "`static_json.error_count=0`" in checkpoint
    assert "returned `ok=true` across sidecars" in checkpoint
    assert "returned `ok=true` and `active_job_count=0`" in checkpoint
    assert "`ready_to_launch=True`" in checkpoint
    assert "`skip_run=True`" in checkpoint
    assert "`training_launched=False`" in checkpoint
    assert "`runtime_verification.training_launch_allowed=True`" in checkpoint
    assert "`total_attempts=63`" in checkpoint
    assert "`baseline_stats.count=0`" in checkpoint
    assert "`launch_prerequisite_attempts.count=1`" in checkpoint
    assert "`launch_ready_attempts.count=0`" in checkpoint
    assert "`included_in_baseline_stats=False`" in checkpoint
    assert "`final_validation_reached=False`" in checkpoint
    assert (
        "the overall\nRSI performance foundation goal is still incomplete" in checkpoint
    )
    assert "Do not mark the goal\ncomplete, do not call `update_goal`" in checkpoint
    assert "No generated\nexperiment artifact" in checkpoint


def test_completion_audit_distinguishes_generated_and_rebuilt_run_index_state():
    text = (TRACKER_ROOT / "completion_audit.md").read_text()
    index_section = text.split("Current generated run index:", 1)[1].split(
        "- Parser baseline eligibility is now fail-closed", 1
    )[0]

    assert "- `len(launch_prerequisite_attempts)=4`" in index_section
    assert "Current rebuilt run-index classification:" in index_section
    assert "- `len(launch_prerequisite_attempts)=1`" in index_section
    assert "The older generated four-prerequisite index is preserved" in index_section
    assert (
        "generated index preserves four historical non-launch prerequisite rows"
        in index_section
    )
    assert (
        "refreshed index proves three non-launch prerequisite rows" not in index_section
    )

    checklist = text.split("## Prompt-to-Artifact Checklist", 1)[1].split(
        "## Current Missing Or Weak Requirements", 1
    )[0]
    assert (
        "The ignored generated `run_index.json` still preserves "
        "`total_attempts=63`, `baseline_stats.count=0`, four skip-run "
        "launch-prerequisite rows"
    ) in checklist
    assert (
        "The current rebuilt classifier admits only the strict runtime-env prerequisite row"
        in checklist
    )
    assert (
        "The rebuilt classifier now surfaces one strict launch-prerequisite row"
        in checklist
    )
    assert "| Current |" not in checklist


def test_completion_audit_tail_preserves_latest_guard_chronology():
    text = (TRACKER_ROOT / "completion_audit.md").read_text()
    tail = text.split("2026-08-20 completion-audit checkpoint and stop condition:", 1)[
        1
    ]
    headings = [
        "2026-08-20 issue-state audit:",
        "2026-08-20 executable issue-state guard refresh:",
        "2026-08-20 issue-dependency guard refresh:",
        "2026-08-20 audit-tail chronology guard refresh:",
        "2026-08-20 audit-tail chronology guard correction:",
        "2026-08-20 focused-count drift cleanup:",
        "2026-08-20 terminal-audit duplicate guard refresh:",
        "2026-08-20 static-surface critical handoff guard refresh:",
        "2026-08-20 broad non-launch completion audit refresh:",
        "2026-08-20 static-Python type surface refresh:",
        "2026-08-20 instruction-mirror static inclusion refresh:",
        "2026-08-20 broad non-launch suite refresh after mirror inclusion:",
        "2026-08-20 post-mirror static-Python type refresh:",
        "2026-08-20 critical ticket static-surface refresh:",
        "2026-08-20 execution-prompt static-surface refresh:",
        "2026-08-20 static-surface dirty-scope parity audit:",
        "2026-08-20 active-doc placeholder guard refresh:",
        "2026-08-20 all-issue placeholder guard refresh:",
        "2026-08-20 current static list-json parity refresh:",
        "2026-08-20 always-include critical handoff refresh:",
        "2026-08-20 broad non-launch suite refresh after critical handoff inclusion:",
        "2026-08-20 generated-artifact tracking guard refresh:",
        "2026-08-20 completion audit against live RSI foundation state:",
        "2026-08-20 final non-launch handoff sweep:",
        "2026-08-20 broad non-launch owner suite refresh after final sweep:",
        "2026-08-20 full static-Python surface refresh after broad suite:",
        "2026-08-20 broad suite membership audit:",
        "2026-08-20 current dirty-surface static coverage audit:",
        "2026-08-20 latest prerequisite artifact validator refresh:",
        "2026-08-20 whole tracked-diff hygiene refresh:",
        "2026-08-20 strict launch-prerequisite index refresh:",
        "2026-08-20 strict launch-prerequisite verification refresh:",
        "2026-08-20 static-surface and rebuilt-index consistency refresh:",
        "2026-08-20 Task 15 final-verification count consistency refresh:",
        "2026-08-20 generated-vs-rebuilt run-index audit cleanup:",
        "2026-08-20 focused verifier count consistency refresh:",
        "2026-08-20 broad owner-suite count consistency refresh:",
        "2026-08-20 broad collection and focused-count consistency refresh:",
        "2026-08-20 continuation blocked-state audit:",
        "2026-08-20 dirty-surface coverage continuation audit:",
        "2026-08-20 continued blocked-state audit after runtime-verifier review:",
        "2026-08-20 execution-prompt command-sidecar drift repair:",
        "2026-08-20 spec and checklist command-sidecar drift repair:",
        "2026-08-20 diagnostic performance-probe command-sidecar repair:",
        "2026-08-20 rootfs runtime spec command-env sidecar drift repair:",
        "2026-08-20 latest prerequisite artifact schema-validation refresh:",
        "2026-08-20 active placeholder and future-task audit:",
        "2026-08-20 live handoff hygiene refresh:",
        "2026-08-20 audit-heading integrity refresh:",
        "2026-08-20 current dirty review-surface coverage refresh:",
        "2026-08-20 continuation completion audit against current blockers:",
        "2026-08-20 broad non-launch owner-suite refresh after completion audit:",
        "2026-08-20 post-broad-suite dirty review-surface coverage audit:",
        "2026-08-20 instruction mirror recheck after completion-audit updates:",
        "2026-08-20 instruction mirror index-state audit:",
        "2026-08-20 launch-token boundary re-audit:",
        "2026-08-20 generated artifact tracking re-audit:",
        "2026-08-20 in-memory run-index completion-gate recheck:",
        "2026-08-20 shell entrypoint syntax and permission audit:",
        "2026-08-20 runtime environment manifest audit:",
        "2026-08-20 dirty review-surface static coverage closeout:",
        "2026-08-20 completion audit against objective:",
        "2026-08-20 broad non-launch owner-suite refresh after current-count",
        "2026-08-20 pre-commit offline gate audit:",
        "2026-08-20 post-residual-risk static-scope audit:",
        "2026-08-20 broad non-launch owner-suite refresh after static-scope audit:",
    ]

    positions = [tail.index(heading) for heading in headings]
    assert positions == sorted(positions)
    for heading in headings:
        assert text.count(f"\n{heading}") == 1, heading
    normalized_tail = re.sub(r"\s+", " ", tail.rstrip())
    assert normalized_tail.endswith(
        "No generated experiment artifact, run index, preflight, summarizer, "
        "GPU probe, non-dry matrix execution, package sync, staging, commit, "
        "cleanup, or full launch was run."
    )
    assert "accepted non-skip baseline" in tail
    assert "launch-full-b200" in tail


def test_completion_audit_records_current_objective_blocker():
    text = (TRACKER_ROOT / "completion_audit.md").read_text()
    audit = text.split("2026-08-20 completion audit against objective:", 1)[1]

    assert "five concrete deliverables" in audit
    assert "`baseline_stats.count=0`" in audit
    assert "`launch_prerequisite_attempts.count=1`" in audit
    assert "`launch_ready_attempts.count=0`" in audit
    assert "`latest_attempt_validator.ok=true`" in audit
    assert "`static.file_count=101`" in audit
    assert "Lane B arm `B0`, FA2 attention, Triton MLP" in audit
    assert "`skip_run=true`" in audit
    assert "`training_launched=false`" in audit
    assert "no accepted non-skip full B200 baseline exists" in audit
    assert "Do not call\n`update_goal`" in audit


def test_completion_audit_records_pre_commit_offline_blocker():
    text = (TRACKER_ROOT / "completion_audit.md").read_text()
    audit = text.split("2026-08-20 pre-commit offline gate audit:", 1)[1]

    assert "`pre-commit run\n--all-files`" in audit
    assert "Could not resolve host: github.com" in audit
    assert "cannot be claimed as passing offline" in audit
    assert "31-file static Python surface" in audit
    assert "`0 errors`" in audit
    assert "`62 passed in 1.75s`" in audit
    assert "Static verification passed for 101 file(s)" in audit
    assert "`baseline_stats.count=0`" in audit
    assert "`launch_ready_attempts.count=0`" in audit
    assert "`launch-full-b200`" in audit


def test_completion_audit_records_post_residual_risk_static_scope():
    text = (TRACKER_ROOT / "completion_audit.md").read_text()
    audit = text.split("2026-08-20 post-residual-risk static-scope audit:", 1)[1]

    assert "`review_relevant_dirty_count=100`" in audit
    assert "`static_candidate_count=101`" in audit
    assert "`missing_dirty_review_paths=[]`" in audit
    assert "`static_validation_error_count=0`" in audit
    assert "`ok=true`" in audit
    assert "`baseline_stats.count=0`" in audit
    assert "`launch_ready_attempts.count=0`" in audit
    assert "`launch-full-b200`" in audit


def test_completion_audit_has_no_duplicate_dated_headings():
    text = (TRACKER_ROOT / "completion_audit.md").read_text()
    headings = re.findall(r"^2026-08-20 .*:(?:$| )", text, flags=re.MULTILINE)
    duplicates = {heading for heading in headings if headings.count(heading) > 1}

    assert duplicates == set()


def test_completion_audit_tail_does_not_pin_stale_heading_counts():
    text = (TRACKER_ROOT / "completion_audit.md").read_text()
    latest_heading_audit = text.split("2026-08-20 audit-heading integrity refresh:", 1)[
        1
    ]

    assert "heading_count=" not in latest_heading_audit
    assert "duplicate_heading_count=" not in latest_heading_audit
    assert "no duplicate dated headings" in latest_heading_audit


def test_issue_15_marks_superseded_broad_verification_bundles_historical():
    text = (ISSUE_DIR / "15-documentation-review-and-handoff.md").read_text()

    assert "This is the current broad non-launch verification point" not in text
    assert "This is the current broad non-launch verification bundle" not in text
    assert (
        "superseded by the later environment/rootfs artifact\nvalidation bundle" in text
    )
    assert (
        "superseded by the stricter\nenvironment/rootfs artifact validation bundle"
        in text
    )


def test_issue_15_records_current_attempt_validator_cli():
    text = (ISSUE_DIR / "15-documentation-review-and-handoff.md").read_text()
    normalized = re.sub(r"\s+", " ", text)

    assert "--attempt-dir" not in text
    assert "--require-launch-prerequisite" not in text
    assert (
        "python experiments/modded_nanogpt_b200/runtime/validate_attempt_artifacts.py\n"
        "experiments/modded_nanogpt_b200/results/"
        "lane_b_full_skiprun_runtime_env_refresh_20260819T131652Z"
    ) in text
    assert "`ok=true`" in text
    assert "takes the result directory positionally" in normalized


def test_issue_15_records_latest_count_and_pyrefly_surface():
    text = (ISSUE_DIR / "15-documentation-review-and-handoff.md").read_text()
    latest_refresh = text.split(
        "2026-08-20 latest count and Python type-surface refresh:", 1
    )[1].split("2026-08-20 sequential kernel/probe verification:", 1)[0]
    normalized_text = re.sub(r"\s+", " ", text)
    normalized_code_text = re.sub(
        r"`pre-commit run\s+--all-files`",
        "`pre-commit run --all-files`",
        normalized_text,
    )

    assert "`410 passed, 2 skipped`" in latest_refresh
    assert "`410 passed, 2 skipped in 43.34s`" not in latest_refresh
    assert "`49 passed`" in latest_refresh
    assert "`48 passed`" not in latest_refresh
    assert "`62 passed`" not in normalized_code_text
    assert (
        "`SKIP=no-commit-to-branch pre-commit run --all-files`" in normalized_code_text
    )
    assert "full `pre-commit run --all-files` remains unproven" not in (
        normalized_code_text
    )
    assert "contains 114 files, including 43 Python files" in latest_refresh
    assert "contains 101 files, including 31 Python files" not in latest_refresh
    assert "prior 31-file Python surface" in latest_refresh
    assert "No errors found!" in text
    assert "Removed 0 unused\nerror suppression(s) in 0 file(s)" in text
    assert "`406 passed, 2 skipped`" not in text
    assert "`97 passed`" not in text
    assert "`367 passed, 2 skipped` as current" not in text


def test_completion_audit_records_current_broad_suite_collection_count():
    text = (TRACKER_ROOT / "completion_audit.md").read_text()
    membership_audit = text.split("2026-08-20 broad suite membership audit:", 1)[
        1
    ].split("2026-08-20 current dirty-surface static coverage audit:", 1)[0]

    assert "`386 passed, 2 skipped`" in membership_audit
    assert "collected 388 tests" in membership_audit
    assert "386 passed plus 2 skipped" in membership_audit
    assert "collected 383 tests" not in membership_audit
    assert "381 passed plus 2 skipped" not in membership_audit


def test_completion_audit_records_latest_broad_owner_suite_refresh():
    text = (TRACKER_ROOT / "completion_audit.md").read_text()
    latest_refresh = text.split(
        "2026-08-20 broad non-launch owner-suite refresh after layer checks:",
        1,
    )[1].split("2026-08-20 shell syntax and permission boundary audit:", 1)[0]

    assert "`410 passed, 2 skipped in 43.34s`" in latest_refresh
    assert "tests_unit_tests/test_modded_nanogpt_b200_*.py" not in latest_refresh
    assert "tests/unit_tests/test_modded_nanogpt_b200_*.py" in latest_refresh
    assert "tests/unit_tests/test_execution_rootfs_identity.py" in latest_refresh
    assert "baseline_stats.count=0" in latest_refresh
    assert "launch_ready_attempts.count=0" in latest_refresh
    assert "launch-full-b200" in latest_refresh


def test_issue_15_preserves_current_run_index_prerequisite_count():
    text = (ISSUE_DIR / "15-documentation-review-and-handoff.md").read_text()
    current_state = text.split("- Current rebuilt run-index facts:", 1)[1].split(
        "- Latest post-hardening refresh artifact:", 1
    )[0]
    normalized_current_state = re.sub(r"\s+", " ", current_state)

    assert "`total_attempts=63`" in current_state
    assert "`baseline_stats.count=0`" in current_state
    assert "`launch_prerequisite_attempts=4`" in current_state
    assert "`launch_ready_attempts=0`" in current_state
    assert "lane_b_full_skiprun_runtime_env_refresh_20260819T131652Z" in text
    assert (
        "Only the latest strict runtime-env row is the current validated "
        "prerequisite handoff artifact"
    ) in normalized_current_state
    assert "historical skip-run dry gates" in current_state


def test_schema_matrix_spec_distinguishes_matrix_arm_from_full_launch_arm():
    text = (TRACKER_ROOT / "schema_matrix_spec.md").read_text()

    assert "Every implemented v1 arm includes:" in text
    assert (
        "legacy artifact `arm` labels such as `B0` remain a\n"
        "  run-harness/full-launch field, not a required checked-in matrix arm field"
    ) in text
    assert (
        "identity: `name`, `experiment_kind`, `legacy_lane`, `arm`, `tags`" not in text
    )
    assert "two-GPU\n`g2_prerequisite` arm" in text
    assert "visible devices `0,1`" in text
    assert (
        "authorized single full-launch command still uses\n"
        "the legacy artifact arm `B0`" in text
    )
    assert "checked-in matrix config selects Lane B arm `B0`" not in text
    assert (
        "The checked-in `g2_prerequisite` arm is\n"
        "the current prerequisite-shaped materialization of that tier, so it is\n"
        "structurally claim-eligible in the dry matrix plan"
    ) in text
    assert (
        "Other `prerequisite`, `diagnostic`, and early `optimization_ablation` arms are"
        in text
    )
    assert (
        "Dry-run arms with `status=planned` are not failed executions and must be listed\n"
        "under `planned_arms` with `reason=dry_run`"
    ) in text
    assert "must not create\n`missing summary.json` observability warnings" in text


def test_claim_label_guidance_maps_semantic_categories_to_legacy_labels():
    master_plan = (TRACKER_ROOT / "master_plan.md").read_text()
    schema_matrix = (TRACKER_ROOT / "schema_matrix_spec.md").read_text()
    spec = (TRACKER_ROOT / "spec.md").read_text()
    execution_prompt = (TRACKER_ROOT / "execution_prompt.md").read_text()
    checklist = (
        REPO_ROOT / "experiments" / "modded_nanogpt_b200" / "preflight_checklist.md"
    ).read_text()

    assert (
        "Current v1 artifact schemas and parser/runner code still emit legacy\n"
        "`claim_label` strings for compatibility."
    ) in master_plan
    assert (
        "Interpret `B200 compatibility\n"
        "patchset` and `B200 systems-only` as the legacy labels for the\n"
        "`B200-compatible local setup` category"
    ) in master_plan
    assert (
        "The v1 materialized `claim_label` remains the legacy artifact string consumed by\n"
        "`run_speedrun.py`, `preflight.py`, `parse_log.py`, and `summarize.py`."
    ) in schema_matrix
    assert (
        "Consumers must interpret `claim_label` together with\n"
        "`claim_eligible`, mode, launch readiness, and run-index baseline inclusion"
    ) in schema_matrix
    assert (
        "a\n"
        "legacy compatibility label on a planned or ineligible arm is not a result claim"
    ) in schema_matrix
    assert (
        "A future compatibility migration may rename\n"
        "emitted labels, but this schema foundation must not silently break existing\n"
        "attempt artifacts."
    ) in schema_matrix
    for text in (spec, execution_prompt, checklist):
        assert "These `claim_label` values are" in text
        assert "v1 legacy emitted strings" in text
        assert "`B200 compatibility patchset` and `B200 systems-only`" in text
        assert "`B200-compatible local setup`" in text
        assert "schema compatibility migration" in text

    assert (
        "`claim_eligible` means only that the attempt is structurally allowed to support\n"
        "a claim."
    ) in spec
    assert (
        "`claim_eligible` means only that the attempt or materialized plan is\n"
        "structurally allowed to support a claim."
    ) in execution_prompt
    assert "claim_eligible` may become true only when" not in execution_prompt


def test_execution_prompt_uses_current_command_sidecar_names():
    text = (TRACKER_ROOT / "execution_prompt.md").read_text()
    metadata = text.split("## Phase 3: Prepare Result Directory and Metadata", 1)[
        1
    ].split("## Phase 4: Source Preparation", 1)[0]
    telemetry = text.split("## Phase 8: Telemetry Capture Plan", 1)[1].split(
        "## Phase 9: Launch and Babysitting", 1
    )[0]

    for section in (metadata, telemetry):
        assert "command.env.json" in section
        assert "command.argv.json" in section
        assert "command.env`" not in section
        assert "command.argv`" not in section

    assert telemetry.count("- `attempt.json`") == 1


def test_spec_and_checklist_use_current_command_sidecar_names():
    spec = (TRACKER_ROOT / "spec.md").read_text()
    checklist = (
        REPO_ROOT / "experiments" / "modded_nanogpt_b200" / "preflight_checklist.md"
    ).read_text()

    bundle_layout = spec.split(
        "Every run attempt has one immutable result directory:", 1
    )[1].split("No-go conditions:", 1)[0]
    env_gate = checklist.split("## Phase 5: Environment Variable Gate", 1)[1].split(
        "Lane A required variables:", 1
    )[0]

    assert "command.env.json" in bundle_layout
    assert "command.argv.json" in bundle_layout
    assert "command.env\n" not in bundle_layout
    assert "command.argv" not in bundle_layout.replace("command.argv.json", "")

    assert "command.env.json" in env_gate
    assert "command.env`" not in env_gate
    assert 'result_dir / "command.env.json"' in env_gate
    assert 'record["environment"]["DATA_PATH"]' in env_gate


def test_performance_probe_issue_uses_current_command_sidecar_names():
    issue = (ISSUE_DIR / "14-nanogpt-performance-probe-ladder.md").read_text()
    artifact_contract = issue.split("## Artifact Contract", 1)[1].split(
        "## Acceptance Criteria", 1
    )[0]

    assert "command.argv.json" in artifact_contract
    assert "command.env.json" in artifact_contract
    assert "`command.argv`" not in artifact_contract
    assert "`command.env`" not in artifact_contract


def test_rootfs_runtime_spec_requires_structured_command_env_sidecar():
    spec = (TRACKER_ROOT / "rootfs_runtime_env_spec.md").read_text()
    full_attempt = spec.split("### 9. Full Attempt", 1)[1].split(
        "### 10. Summarization", 1
    )[0]

    assert "`command.env.json` with redacted environment metadata" in full_attempt
    assert "must not replace `command.env.json`" in full_attempt
    assert (
        "`command.env` without secret or authorization-token values" not in full_attempt
    )
