#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCAFFOLD_TO_POLICY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCAFFOLD_TO_POLICY_DIR}/../.." && pwd)"

scaffold_enter_rootfs_if_needed() {
  local script_path="$1"
  shift
  if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
    exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "experiments/scaffold_to_policy/${script_path}" "$@"
  fi
}

scaffold_setup_env() {
  cd "${REPO_ROOT}"
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export VLLM_USE_FLASHINFER_SAMPLER="${SCAFFOLD_TO_POLICY_VLLM_USE_FLASHINFER_SAMPLER:-0}"
  mkdir -p "${HF_HOME}"
}

scaffold_setup_run_manifest() {
  export TORCHTITAN_SCAFFOLD_TO_POLICY_RUN_ID="${RUN_ID}"
  export TORCHTITAN_SCAFFOLD_TO_POLICY_DATA_ROOT="${DATA_ROOT:-}"
  export TORCHTITAN_SCAFFOLD_TO_POLICY_RESULTS_ROOT="${RESULTS_ROOT}"
  export TORCHTITAN_SCAFFOLD_TO_POLICY_MANIFEST="${TORCHTITAN_SCAFFOLD_TO_POLICY_MANIFEST:-${RESULTS_ROOT}/manifests/${RUN_ID}.jsonl}"
  mkdir -p "${RESULTS_ROOT}/manifests"
  : > "${TORCHTITAN_SCAFFOLD_TO_POLICY_MANIFEST}"
}

scaffold_run_stage() {
  local stage="$1"
  shift
  local status_path=""
  if [[ -n "${TORCHTITAN_SCAFFOLD_TO_POLICY_STAGE_STATUS_DIR:-}" ]]; then
    status_path="${TORCHTITAN_SCAFFOLD_TO_POLICY_STAGE_STATUS_DIR}/${stage}.json"
    rm -f "${status_path}"
    export TORCHTITAN_SCAFFOLD_TO_POLICY_STAGE_STATUS="${status_path}"
  else
    unset TORCHTITAN_SCAFFOLD_TO_POLICY_STAGE_STATUS
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
    RUN_ID="${TORCHTITAN_SCAFFOLD_TO_POLICY_RUN_ID}" \
    ROOTFS_ACTIVE="${TORCHTITAN_IN_ROOTFS:-0}" \
    DATA_ROOT="${TORCHTITAN_SCAFFOLD_TO_POLICY_DATA_ROOT:-}" \
    RESULTS_ROOT="${TORCHTITAN_SCAFFOLD_TO_POLICY_RESULTS_ROOT}" \
    STAGE_STATUS_PATH="${status_path}" \
    COMMAND_JSON="$(printf '%s\n' "$@" | python -c 'import json, sys; print(json.dumps([line.rstrip("\n") for line in sys.stdin]))')" \
    MANIFEST="${TORCHTITAN_SCAFFOLD_TO_POLICY_MANIFEST}" \
    python - <<'PY'
import json
import os
from pathlib import Path

row = {
    "stage": os.environ["STAGE"],
    "run_id": os.environ["RUN_ID"],
    "rootfs_active": os.environ["ROOTFS_ACTIVE"] == "1",
    "start_time": os.environ["START_ISO"],
    "end_time": os.environ["END_ISO"],
    "duration_seconds": int(os.environ["DURATION_SECONDS"]),
    "return_code": int(os.environ["STATUS"]),
    "command": json.loads(os.environ["COMMAND_JSON"]),
    "data_root": os.environ["DATA_ROOT"] or None,
    "results_root": os.environ["RESULTS_ROOT"],
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

scaffold_write_stage_failure_marker() {
  local stage="$1"
  local failure_type="$2"
  local output="$3"
  STAGE="${stage}" \
    FAILURE_TYPE="${failure_type}" \
    OUTPUT="${output}" \
    RUN_ID="${TORCHTITAN_SCAFFOLD_TO_POLICY_RUN_ID}" \
    ROOTFS_ACTIVE="${TORCHTITAN_IN_ROOTFS:-0}" \
    MANIFEST="${TORCHTITAN_SCAFFOLD_TO_POLICY_MANIFEST}" \
    python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

marker = {
    "schema_version": 1,
    "kind": "stage_failure",
    "failure_type": os.environ["FAILURE_TYPE"],
    "stage": os.environ["STAGE"],
    "run_id": os.environ["RUN_ID"],
    "rootfs_active": os.environ["ROOTFS_ACTIVE"] == "1",
    "manifest": os.environ["MANIFEST"],
    "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
output = Path(os.environ["OUTPUT"])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
PY
}
