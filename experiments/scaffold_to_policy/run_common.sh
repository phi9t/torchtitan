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
  # Append-only: a manifest is scientific evidence, so a repeated setup (a nested
  # doctor call, a retry, or a re-run under the same RUN_ID) must never erase
  # prior rows. Create the file if it does not exist, then record a distinct
  # attempt marker so downstream tools can separate one operational execution
  # from another within the same run.
  local stage_invocation_id
  stage_invocation_id="$(scaffold_new_stage_invocation_id)"
  export TORCHTITAN_SCAFFOLD_TO_POLICY_STAGE_INVOCATION_ID="${stage_invocation_id}"
  RUN_ID="${RUN_ID}" \
    STAGE_INVOCATION_ID="${stage_invocation_id}" \
    MANIFEST="${TORCHTITAN_SCAFFOLD_TO_POLICY_MANIFEST}" \
    python - <<'PY'
import json
import os
from pathlib import Path

marker = {
    "kind": "manifest_attempt",
    "run_id": os.environ["RUN_ID"],
    "stage_invocation_id": os.environ["STAGE_INVOCATION_ID"],
}
manifest = Path(os.environ["MANIFEST"])
manifest.parent.mkdir(parents=True, exist_ok=True)
with manifest.open("a") as f:
    f.write(json.dumps(marker, sort_keys=True) + "\n")
PY
}

scaffold_new_stage_invocation_id() {
  # A stage_invocation_id identifies one actual stage launch within an attempt.
  # Combine a UTC timestamp with process and random entropy so repeated calls in
  # the same second still produce distinct identifiers.
  printf 'inv-%s-%s-%s\n' \
    "$(date -u +%Y%m%dT%H%M%SZ)" \
    "$$" \
    "${RANDOM}${RANDOM}"
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
    STAGE_INVOCATION_ID="${TORCHTITAN_SCAFFOLD_TO_POLICY_STAGE_INVOCATION_ID:-}" \
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
    "work_status": "executed",
    "start_time": os.environ["START_ISO"],
    "end_time": os.environ["END_ISO"],
    "duration_seconds": int(os.environ["DURATION_SECONDS"]),
    "return_code": int(os.environ["STATUS"]),
    "command": json.loads(os.environ["COMMAND_JSON"]),
    "data_root": os.environ["DATA_ROOT"] or None,
    "results_root": os.environ["RESULTS_ROOT"],
}
stage_invocation_id = os.environ.get("STAGE_INVOCATION_ID", "")
if stage_invocation_id:
    row["stage_invocation_id"] = stage_invocation_id
stage_status_path = os.environ["STAGE_STATUS_PATH"]
if stage_status_path and Path(stage_status_path).is_file():
    stage_status = json.loads(Path(stage_status_path).read_text())
    row["stage_status"] = stage_status
    if isinstance(stage_status, dict) and isinstance(stage_status.get("work_status"), str):
        row["work_status"] = stage_status["work_status"]
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

scaffold_capture_vllm_runtime_metadata() {
  local output="$1"
  scaffold_run_stage capture_runtime_metadata python -m torchtitan.experiments.scaffold_to_policy.cli capture-runtime-metadata \
    --output "${output}" \
    --run-id "${RUN_ID}" \
    --model "${MODEL}" \
    --num-rollouts "${NUM_ROLLOUTS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --prompt-variant "${PROMPT_VARIANT}" \
    --max-model-len "${MAX_MODEL_LEN:-${SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN:-2048}}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --attention-backend "${SCAFFOLD_TO_POLICY_VLLM_ATTENTION_BACKEND:-TRITON_ATTN}" \
    --use-flashinfer-sampler "${SCAFFOLD_TO_POLICY_VLLM_USE_FLASHINFER_SAMPLER:-0}"
}
