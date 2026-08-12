#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

COUNTDOWN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${COUNTDOWN_DIR}/../.." && pwd)"

countdown_enter_rootfs_if_needed() {
  local script_path="$1"
  shift
  if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
    exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "experiments/countdown_search_distill/${script_path}" "$@"
  fi
}

countdown_setup_env() {
  cd "${REPO_ROOT}"
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  export HF_HOME="${REPO_ROOT}/.cache/huggingface"
  export HF_HUB_CACHE="${HF_HOME}/hub"
  export TORCHTITAN_COUNTDOWN_ROOT="${REPO_ROOT}/experiments/countdown_search_distill"
  export TORCHTITAN_COUNTDOWN_DATA_ROOT="${TORCHTITAN_COUNTDOWN_DATA_ROOT:-${TORCHTITAN_COUNTDOWN_ROOT}/data}"
  export TORCHTITAN_COUNTDOWN_RESULTS_ROOT="${TORCHTITAN_COUNTDOWN_RESULTS_ROOT:-${TORCHTITAN_COUNTDOWN_ROOT}/results}"
  export TORCHTITAN_COUNTDOWN_RUN_ID="${TORCHTITAN_COUNTDOWN_RUN_ID:-${RUN_ID:-current}}"
  export TORCHTITAN_COUNTDOWN_MANIFEST="${TORCHTITAN_COUNTDOWN_MANIFEST:-${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}/manifests/current.jsonl}"
  mkdir -p \
    "${TORCHTITAN_COUNTDOWN_DATA_ROOT}" \
    "${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}" \
    "${TORCHTITAN_COUNTDOWN_ROOT}/reports" \
    "${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}/manifests" \
    "${HF_HOME}"
}

countdown_run_stage() {
  local stage="$1"
  shift
  local status_path=""
  if [[ -n "${TORCHTITAN_COUNTDOWN_STAGE_STATUS_DIR:-}" ]]; then
    status_path="${TORCHTITAN_COUNTDOWN_STAGE_STATUS_DIR}/${stage}.json"
    rm -f "${status_path}"
    export TORCHTITAN_COUNTDOWN_STAGE_STATUS="${status_path}"
  else
    unset TORCHTITAN_COUNTDOWN_STAGE_STATUS
  fi
  local start_epoch
  local start_iso
  local end_epoch
  local end_iso
  local status
  start_epoch="$(date -u +%s)"
  start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set +e
  "$@"
  status=$?
  set -e
  end_epoch="$(date -u +%s)"
  end_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  STAGE="${stage}" \
    START_ISO="${start_iso}" \
    END_ISO="${end_iso}" \
    DURATION_SECONDS="$((end_epoch - start_epoch))" \
    STATUS="${status}" \
    RUN_ID="${TORCHTITAN_COUNTDOWN_RUN_ID}" \
    MODE="${MODE:-}" \
    ROOTFS_ACTIVE="${TORCHTITAN_IN_ROOTFS:-0}" \
    STAGE_STATUS_PATH="${status_path}" \
    COMMAND_JSON="$(printf '%s\n' "$@" | python -c 'import json, sys; print(json.dumps([line.rstrip("\n") for line in sys.stdin]))')" \
    MANIFEST="${TORCHTITAN_COUNTDOWN_MANIFEST}" \
    python - <<'PY'
import json
import os
from pathlib import Path

row = {
    "stage": os.environ["STAGE"],
    "run_id": os.environ["RUN_ID"],
    "mode": os.environ["MODE"] or None,
    "rootfs_active": os.environ["ROOTFS_ACTIVE"] == "1",
    "start_time": os.environ["START_ISO"],
    "end_time": os.environ["END_ISO"],
    "duration_seconds": int(os.environ["DURATION_SECONDS"]),
    "return_code": int(os.environ["STATUS"]),
    "command": json.loads(os.environ["COMMAND_JSON"]),
}
stage_status_path = os.environ["STAGE_STATUS_PATH"]
if stage_status_path and Path(stage_status_path).is_file():
    row["stage_status"] = json.loads(Path(stage_status_path).read_text())
manifest = Path(os.environ["MANIFEST"])
manifest.parent.mkdir(parents=True, exist_ok=True)
with manifest.open("a") as f:
    f.write(json.dumps(row, sort_keys=True) + "\n")
PY
  if [[ "${status}" -ne 0 ]]; then
    return "${status}"
  fi
}
