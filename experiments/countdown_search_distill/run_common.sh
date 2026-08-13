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
  export TORCHTITAN_COUNTDOWN_ATTEMPT_ID="${TORCHTITAN_COUNTDOWN_ATTEMPT_ID:-attempt-01}"
  export TORCHTITAN_COUNTDOWN_MANIFEST="${TORCHTITAN_COUNTDOWN_MANIFEST:-${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}/manifests/current.jsonl}"
  mkdir -p \
    "${TORCHTITAN_COUNTDOWN_DATA_ROOT}" \
    "${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}" \
    "${TORCHTITAN_COUNTDOWN_ROOT}/reports" \
    "${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}/manifests" \
    "${HF_HOME}"
}

# The typed lifecycle locator addresses one attempt bundle under
# <results-root>/runs/<run-id>/<attempt-id>/. Every begin/stage/finish call
# shares it.
countdown_lifecycle_locator() {
  printf '%s\n' \
    --results-root "${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}" \
    --run-id "${TORCHTITAN_COUNTDOWN_RUN_ID}" \
    --attempt-id "${TORCHTITAN_COUNTDOWN_ATTEMPT_ID}"
}

# Freeze one Countdown run declaration into an attempt bundle. Idempotent for a
# fixed (run_id, attempt_id): a re-begin over an existing manifest is a no-op so
# a sub-runner invoked standalone can begin without clobbering a pilot attempt.
countdown_begin_attempt() {
  local mode="${1:-${MODE:-full}}"
  local bundle_dir="${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}/runs/${TORCHTITAN_COUNTDOWN_RUN_ID}/${TORCHTITAN_COUNTDOWN_ATTEMPT_ID}"
  if [[ -f "${bundle_dir}/manifest.json" ]]; then
    return 0
  fi
  local fields_file
  fields_file="$(mktemp)"
  MODE="${mode}" ARMS="${ARMS:-}" RUN_ID="${TORCHTITAN_COUNTDOWN_RUN_ID}" \
    python - >"${fields_file}" <<'PY'
import json
import os

print(
    json.dumps(
        {
            "mode": os.environ.get("MODE") or None,
            "arms": os.environ.get("ARMS") or None,
            "run_id": os.environ["RUN_ID"],
        },
        sort_keys=True,
    )
)
PY
  local locator
  mapfile -t locator < <(countdown_lifecycle_locator)
  python -m torchtitan.experiments.execution begin \
    "${locator[@]}" \
    --family countdown \
    --task search_distill \
    --lane "${mode}" \
    --fields "${fields_file}"
  rm -f "${fields_file}"
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

  # An attempt must exist before any stage; a sub-runner invoked standalone
  # begins its own idempotently. The typed stage records argv, timing, and the
  # terminal outcome on the coordinator event stream, so the legacy manifest row
  # content is preserved as stage-event extras rather than a parallel JSONL.
  countdown_begin_attempt "${MODE:-full}"

  local kind
  local adapter
  kind="$(countdown_stage_kind "${stage}")"
  adapter="$(countdown_stage_adapter "${stage}")"

  local locator
  mapfile -t locator < <(countdown_lifecycle_locator)

  local mode_json
  if [[ -n "${MODE:-}" ]]; then
    mode_json="\"${MODE}\""
  else
    mode_json="null"
  fi
  local rootfs_json
  if [[ "${TORCHTITAN_IN_ROOTFS:-0}" == "1" ]]; then
    rootfs_json="true"
  else
    rootfs_json="false"
  fi
  local extra_args=(
    --stage-extra "mode=${mode_json}"
    --stage-extra "rootfs_active=${rootfs_json}"
  )
  if [[ -n "${status_path}" ]]; then
    extra_args+=(--stage-extra-file "stage_status=${status_path}")
  fi

  local status
  set +e
  python -m torchtitan.experiments.execution stage \
    "${locator[@]}" \
    --stage-id "${stage}" \
    --name "${stage}" \
    --kind "${kind}" \
    --adapter "${adapter}" \
    "${extra_args[@]}" \
    -- "$@"
  status=$?
  set -e
  if [[ "${status}" -ne 0 ]]; then
    return "${status}"
  fi
}

# Map a Countdown stage name onto a typed stage kind (models.STAGE_KINDS).
countdown_stage_kind() {
  case "$1" in
    preflight) echo preflight ;;
    calibration|calibration_sweep|collect) echo generate ;;
    validate_splits) echo verify ;;
    train_*) echo train ;;
    base_eval_*|eval_adapters) echo evaluate ;;
    export_adapters) echo export ;;
    build_report_input) echo report ;;
    *) echo generate ;;
  esac
}

# Map a Countdown stage name onto a typed execution adapter
# (models.EXECUTION_ADAPTERS). Generation and evaluation drive vLLM; training
# drives torchrun SFT; gate, validation, and report stages need no model
# forward and run on the rootfs CPU adapter.
countdown_stage_adapter() {
  case "$1" in
    calibration|calibration_sweep|collect|base_eval_*|eval_adapters) echo rootfs_vllm ;;
    train_*) echo rootfs_torchrun_sft ;;
    export_adapters) echo rootfs_cpu ;;
    preflight|validate_splits|build_report_input) echo rootfs_cpu ;;
    *) echo rootfs_cpu ;;
  esac
}
