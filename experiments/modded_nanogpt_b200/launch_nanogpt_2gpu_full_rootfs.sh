#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  printf 'error: %s does not accept arguments\n' "$0" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_REL="experiments/modded_nanogpt_b200/launch_nanogpt_2gpu_full_rootfs.sh"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" == "1" && -n "${MODDED_NANOGPT_2GPU_FULL_RUN_ID:-}" ]]; then
  RUN_ID="${MODDED_NANOGPT_2GPU_FULL_RUN_ID}"
else
  RUN_ID="nanogpt_2gpu_full_$(date -u +%Y%m%dT%H%M%SZ)"
  export MODDED_NANOGPT_2GPU_FULL_RUN_ID="${RUN_ID}"
fi
ATTEMPT_ID="${RUN_ID}_attempt_001"
RESULT_DIR="${REPO_ROOT}/experiments/modded_nanogpt_b200/results/${RUN_ID}"
SOURCE="experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa"
DATA_MANIFEST="experiments/modded_nanogpt_b200/results/full_manifest_refresh_20260816T111021Z/data_manifest.json"
OPERATOR_LOG="${RESULT_DIR}/operator_launch.log"
TEARDOWN_JSON="${RESULT_DIR}/teardown.json"
ROOTFS_PLAN="${RESULT_DIR}/rootfs_plan.json"
ACTIVE_JOBS_JSON="${RESULT_DIR}/active_jobs_prelaunch.json"

mkdir -p "${RESULT_DIR}"

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${message}" | tee -a "${OPERATOR_LOG}"
}

write_teardown() {
  local exit_code="$1"
  local phase="$2"
  cat >"${TEARDOWN_JSON}" <<EOF
{
  "schema_version": 1,
  "run_id": "${RUN_ID}",
  "attempt_id": "${ATTEMPT_ID}",
  "phase": "${phase}",
  "exit_code": ${exit_code},
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "result_dir": "${RESULT_DIR}",
  "operator_log": "${OPERATOR_LOG}",
  "rootfs_plan": "${ROOTFS_PLAN}",
  "active_jobs_prelaunch": "${ACTIVE_JOBS_JSON}",
  "run_log": "${RESULT_DIR}/run.log",
  "summary": "${RESULT_DIR}/summary.json",
  "launch_readiness": "${RESULT_DIR}/launch_readiness.json"
}
EOF
}

on_exit() {
  local exit_code="$?"
  set +e
  if [[ "${TORCHTITAN_IN_ROOTFS:-0}" == "1" ]]; then
    log "teardown: rootfs payload finished with exit_code=${exit_code}"
  else
    log "teardown: host launcher finished with exit_code=${exit_code}"
  fi
  write_teardown "${exit_code}" "launcher_exit"
}
trap on_exit EXIT

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  log "rootfs prep: emitting bwrap launch plan"
  TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY=1 \
    TORCHTITAN_ROOTFS_PLAN_OUTPUT="${ROOTFS_PLAN}" \
    "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- \
      /bin/bash -lc "MODDED_NANOGPT_2GPU_FULL_RUN_ID='${RUN_ID}' exec '${SCRIPT_REL}'"
  log "rootfs prep: wrote ${ROOTFS_PLAN}"

  if [[ -f "${REPO_ROOT}/scripts/rootfs/verify_runtime_env.py" ]]; then
    log "rootfs prep: verifying emitted bwrap plan"
    python3 "${REPO_ROOT}/scripts/rootfs/verify_runtime_env.py" \
      --plan "${ROOTFS_PLAN}" \
      --report "${RESULT_DIR}/rootfs_plan_verification.json" \
      >>"${OPERATOR_LOG}" 2>&1
    log "rootfs prep: plan verifier passed"
  fi

  log "rootfs prep: entering bwrap runtime"
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- \
    /bin/bash -lc "MODDED_NANOGPT_2GPU_FULL_RUN_ID='${RUN_ID}' exec '${SCRIPT_REL}'"
fi

source "${SCRIPT_DIR}/rootfs_guard.sh"
require_modded_nanogpt_rootfs "${SCRIPT_REL}"

cd "${REPO_ROOT}"

log "rootfs runtime: workspace=$(pwd -P)"
log "rootfs runtime: source=${SOURCE}"
log "rootfs runtime: data_manifest=${DATA_MANIFEST}"
log "model bringup: checking active nanoGPT jobs before launch"
experiments/modded_nanogpt_b200/check_active_jobs.sh \
  --active-jobs-output "${ACTIVE_JOBS_JSON}" \
  >>"${OPERATOR_LOG}" 2>&1
log "model bringup: active-job scan wrote ${ACTIVE_JOBS_JSON}"

if [[ ! -d "${SOURCE}" ]]; then
  log "model bringup: missing source directory ${SOURCE}"
  exit 21
fi
if [[ ! -f "${DATA_MANIFEST}" ]]; then
  log "model bringup: missing data manifest ${DATA_MANIFEST}"
  exit 21
fi

log "training: starting guarded Lane B 2-GPU full run through run_speedrun.sh"
set +e
experiments/modded_nanogpt_b200/run_speedrun.sh \
  --mode full \
  --lane B \
  --run-id "${RUN_ID}" \
  --attempt-id "${ATTEMPT_ID}" \
  --arm B0 \
  --source "${SOURCE}" \
  --data-manifest "${DATA_MANIFEST}" \
  --attention-backend fa2 \
  --mlp-backend triton \
  --verify-sha \
  --launch-authorization=launch-full-b200 \
  --result-dir "${RESULT_DIR}" \
  >>"${OPERATOR_LOG}" 2>&1
launch_code="$?"
set -e
log "training: run_speedrun.sh finished with exit_code=${launch_code}"

log "teardown: refreshing run index"
experiments/modded_nanogpt_b200/summarize.sh \
  --results-root experiments/modded_nanogpt_b200/results \
  --output experiments/modded_nanogpt_b200/results/run_index.json \
  >>"${OPERATOR_LOG}" 2>&1 || log "teardown: run-index refresh failed"

if [[ -f "${RESULT_DIR}/summary.json" ]]; then
  log "losses: summary artifact available at ${RESULT_DIR}/summary.json"
fi
if [[ -f "${RESULT_DIR}/run.log" ]]; then
  log "training progress: raw run log available at ${RESULT_DIR}/run.log"
fi

exit "${launch_code}"
