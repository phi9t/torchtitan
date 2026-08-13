#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"
countdown_enter_rootfs_if_needed "run_full_pilot.sh" "$@"
countdown_setup_env

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${MODE:-full}}"
export RUN_ID
export TORCHTITAN_COUNTDOWN_RUN_ID="${RUN_ID}"
export TORCHTITAN_COUNTDOWN_MANIFEST="${TORCHTITAN_COUNTDOWN_MANIFEST:-${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}/manifests/${RUN_ID}.jsonl}"
export TORCHTITAN_COUNTDOWN_STAGE_STATUS_DIR="${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}/manifests/${RUN_ID}.stage_status"
: > "${TORCHTITAN_COUNTDOWN_MANIFEST}"
rm -rf "${TORCHTITAN_COUNTDOWN_STAGE_STATUS_DIR}"
mkdir -p "${TORCHTITAN_COUNTDOWN_STAGE_STATUS_DIR}"
REQUESTED_ARMS="${ARMS:-}"

countdown_run_stage preflight "${SCRIPT_DIR}/run_preflight.sh"

if [[ "${MODE:-full}" != "smoke" ]]; then
  countdown_run_stage calibration_sweep env MODE="${MODE:-full}" "${SCRIPT_DIR}/run_calibration_sweep.sh"
fi
countdown_run_stage calibration env MODE="${MODE:-full}" "${SCRIPT_DIR}/run_calibration.sh"
countdown_run_stage collect env MODE="${MODE:-full}" "${SCRIPT_DIR}/run_collect.sh"
if [[ "${MODE:-full}" != "smoke" ]]; then
  countdown_run_stage validate_splits "${SCRIPT_DIR}/run_validate_splits.sh"
fi

case "${MODE:-full}" in
  smoke)
    ARMS=(debug_smoke)
    ;;
  reduced)
    python -m torchtitan.experiments.countdown_search_distill.cli preflight-reduced \
      --calibration-decision "${TORCHTITAN_COUNTDOWN_DATA_ROOT}/calibration/decision.json" \
      --train-evaluations "${TORCHTITAN_COUNTDOWN_DATA_ROOT}/train/evaluations.jsonl" \
      --dev-evaluations "${TORCHTITAN_COUNTDOWN_DATA_ROOT}/dev/evaluations.jsonl" \
      --decision "${TORCHTITAN_COUNTDOWN_DATA_ROOT}/reduced_preflight.json"
    ARMS=(raw hindsight curriculum)
    TRAIN_RESULT_ROOT="${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}/train/reduced"
    ;;
  full)
    python -m torchtitan.experiments.countdown_search_distill.cli preflight-reduced \
      --calibration-decision "${TORCHTITAN_COUNTDOWN_DATA_ROOT}/calibration/decision.json" \
      --train-evaluations "${TORCHTITAN_COUNTDOWN_DATA_ROOT}/train/evaluations.jsonl" \
      --dev-evaluations "${TORCHTITAN_COUNTDOWN_DATA_ROOT}/dev/evaluations.jsonl" \
      --decision "${TORCHTITAN_COUNTDOWN_DATA_ROOT}/full_preflight.json"
    ARMS=(raw clean formatting hindsight curriculum)
    TRAIN_RESULT_ROOT="${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}/train/full"
    ;;
  *) echo "unknown MODE=${MODE}" >&2; exit 2 ;;
esac
if [[ -n "${REQUESTED_ARMS}" ]]; then
  read -r -a ARMS <<< "${REQUESTED_ARMS}"
fi

for arm in "${ARMS[@]}"; do
  countdown_run_stage "train_${arm}" env ARM="${arm}" TRAIN_RESULT_ROOT="${TRAIN_RESULT_ROOT:-}" "${SCRIPT_DIR}/run_train.sh"
done

for split in dev iid_test ood_test; do
  countdown_run_stage "base_eval_${split}" env SPLIT="${split}" "${SCRIPT_DIR}/run_eval.sh"
done

if [[ "${MODE:-full}" != "smoke" ]]; then
  countdown_run_stage export_adapters env MODE="${MODE:-full}" ARMS="${ARMS[*]}" "${SCRIPT_DIR}/run_export_adapters.sh"
  countdown_run_stage eval_adapters env MODE="${MODE:-full}" ARMS="${ARMS[*]}" "${SCRIPT_DIR}/run_eval_adapters.sh"
  countdown_run_stage build_report_input \
    python -m torchtitan.experiments.countdown_search_distill.cli build-report-input \
      --experiment-root "${TORCHTITAN_COUNTDOWN_ROOT}" \
      --mode "${MODE:-full}" \
      --run-id "${RUN_ID}" \
      --attempt-id "${TORCHTITAN_COUNTDOWN_ATTEMPT_ID}" \
      --manifest "${TORCHTITAN_COUNTDOWN_MANIFEST}" \
      --arms "${ARMS[@]}" \
      --data-root "${TORCHTITAN_COUNTDOWN_DATA_ROOT}" \
      --results-root "${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}" \
      --output "${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
fi
