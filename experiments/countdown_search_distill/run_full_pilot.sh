#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"
countdown_enter_rootfs_if_needed "run_full_pilot.sh" "$@"
countdown_setup_env

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${MODE:-full}}"
export TORCHTITAN_COUNTDOWN_MANIFEST="${TORCHTITAN_COUNTDOWN_MANIFEST:-${TORCHTITAN_COUNTDOWN_ROOT}/results/manifests/${RUN_ID}.jsonl}"
: > "${TORCHTITAN_COUNTDOWN_MANIFEST}"

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
      --calibration-decision "${TORCHTITAN_COUNTDOWN_ROOT}/data/calibration/decision.json" \
      --train-evaluations "${TORCHTITAN_COUNTDOWN_ROOT}/data/train/evaluations.jsonl" \
      --dev-evaluations "${TORCHTITAN_COUNTDOWN_ROOT}/data/dev/evaluations.jsonl" \
      --decision "${TORCHTITAN_COUNTDOWN_ROOT}/data/reduced_preflight.json"
    ARMS=(raw hindsight curriculum)
    TRAIN_RESULT_ROOT="${TORCHTITAN_COUNTDOWN_ROOT}/results/train/reduced"
    ;;
  full)
    python -m torchtitan.experiments.countdown_search_distill.cli preflight-reduced \
      --calibration-decision "${TORCHTITAN_COUNTDOWN_ROOT}/data/calibration/decision.json" \
      --train-evaluations "${TORCHTITAN_COUNTDOWN_ROOT}/data/train/evaluations.jsonl" \
      --dev-evaluations "${TORCHTITAN_COUNTDOWN_ROOT}/data/dev/evaluations.jsonl" \
      --decision "${TORCHTITAN_COUNTDOWN_ROOT}/data/full_preflight.json"
    ARMS=(raw clean hindsight curriculum)
    TRAIN_RESULT_ROOT="${TORCHTITAN_COUNTDOWN_ROOT}/results/train/full"
    ;;
  *) echo "unknown MODE=${MODE}" >&2; exit 2 ;;
esac

for arm in "${ARMS[@]}"; do
  countdown_run_stage "train_${arm}" env ARM="${arm}" TRAIN_RESULT_ROOT="${TRAIN_RESULT_ROOT:-}" "${SCRIPT_DIR}/run_train.sh"
done

for split in dev iid_test ood_test; do
  countdown_run_stage "base_eval_${split}" env SPLIT="${split}" "${SCRIPT_DIR}/run_eval.sh"
done

if [[ "${MODE:-full}" != "smoke" ]]; then
  countdown_run_stage export_adapters env MODE="${MODE:-full}" "${SCRIPT_DIR}/run_export_adapters.sh"
  countdown_run_stage eval_adapters env MODE="${MODE:-full}" "${SCRIPT_DIR}/run_eval_adapters.sh"
fi
