# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wave F4 tests for the Countdown run_common lifecycle migration.

These verify that the shared countdown_run_stage helper drives the typed
begin/stage/finish lifecycle while preserving the legacy per-stage row content
as stage-event extras. They run on the host with no GPU and no vLLM: the stage
commands are stubs (true/false), TORCHTITAN_IN_ROOTFS=1 skips the bwrap
re-exec, and the typed stage adapter runs an arbitrary argv.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COUNTDOWN_DIR = REPO_ROOT / "experiments" / "countdown_search_distill"
RUN_COMMON = COUNTDOWN_DIR / "run_common.sh"


def _run_common_script(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    """Source run_common.sh outside the rootfs and run a stage scenario.

    TORCHTITAN_IN_ROOTFS=1 keeps countdown_enter_rootfs_if_needed from
    re-entering bwrap; the helpers under test do not need the rootfs. The
    results root is redirected under tmp_path so the typed attempt bundle is
    isolated.
    """

    results_root = tmp_path / "results"
    script = f"""
set -euo pipefail
export TORCHTITAN_IN_ROOTFS=1
export PYTHONPATH="{REPO_ROOT}:${{PYTHONPATH:-}}"
export TORCHTITAN_COUNTDOWN_RESULTS_ROOT="{results_root}"
export TORCHTITAN_COUNTDOWN_DATA_ROOT="{tmp_path / 'data'}"
export RUN_ID="fixed-run-01"
export TORCHTITAN_COUNTDOWN_ATTEMPT_ID="attempt-01"
source "{RUN_COMMON}"
countdown_setup_env
{body}
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def _bundle_dir(tmp_path: Path) -> Path:
    return tmp_path / "results" / "runs" / "fixed-run-01" / "attempt-01"


def _events(tmp_path: Path) -> list[dict]:
    path = _bundle_dir(tmp_path) / "processes" / "coordinator" / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_countdown_runners_syntax():
    scripts = [RUN_COMMON] + sorted(COUNTDOWN_DIR.glob("run_*.sh"))
    for script in scripts:
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{script.name}: {result.stderr}"


def test_countdown_run_stage_drives_typed_stage(tmp_path):
    result = _run_common_script(
        tmp_path,
        """
countdown_begin_attempt full
countdown_run_stage stub-stage true
""",
    )
    assert result.returncode == 0, result.stderr

    events = _events(tmp_path)
    kinds = {(e["stage_id"], e["kind"]) for e in events}
    assert ("stub-stage", "stage_started") in kinds
    assert ("stub-stage", "stage_succeeded") in kinds

    # The legacy row content survives as stage-event extras.
    started = next(
        e
        for e in events
        if e["stage_id"] == "stub-stage" and e["kind"] == "stage_started"
    )
    assert started["argv"][-1:] == ["true"] or "true" in started["argv"]


def test_countdown_run_stage_propagates_failure(tmp_path):
    result = _run_common_script(
        tmp_path,
        """
countdown_begin_attempt full
countdown_run_stage fail-stage false
""",
    )
    assert result.returncode != 0

    events = _events(tmp_path)
    kinds = {(e["stage_id"], e["kind"]) for e in events}
    assert ("fail-stage", "stage_failed") in kinds


def test_countdown_begin_attempt_idempotent(tmp_path):
    result = _run_common_script(
        tmp_path,
        """
countdown_begin_attempt full
countdown_begin_attempt full
countdown_run_stage stub-stage true
""",
    )
    assert result.returncode == 0, result.stderr

    manifest = _bundle_dir(tmp_path) / "manifest.json"
    payload = json.loads(manifest.read_text())
    assert payload["kind"] == "attempt_manifest"
    assert payload["run"]["family"] == "countdown"


def test_countdown_stage_status_attached(tmp_path):
    status_dir = tmp_path / "status"
    result = _run_common_script(
        tmp_path,
        f"""
export TORCHTITAN_COUNTDOWN_STAGE_STATUS_DIR="{status_dir}"
mkdir -p "{status_dir}"
countdown_begin_attempt full
countdown_run_stage stub-stage bash -c 'echo "{{\\"selected\\": true}}" > "$TORCHTITAN_COUNTDOWN_STAGE_STATUS"'
""",
    )
    assert result.returncode == 0, result.stderr

    events = _events(tmp_path)
    terminal = next(
        e
        for e in events
        if e["stage_id"] == "stub-stage" and e["kind"] == "stage_succeeded"
    )
    assert terminal.get("stage_status") == {"selected": True}
