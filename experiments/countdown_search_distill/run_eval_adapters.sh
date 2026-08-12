#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"
countdown_enter_rootfs_if_needed "run_eval_adapters.sh" "$@"
countdown_setup_env

MODE="${MODE:-full}"
ADAPTER_RESULT_ROOT="${ADAPTER_RESULT_ROOT:-${TORCHTITAN_COUNTDOWN_ROOT}/results/adapters/${MODE}}"
EVAL_RESULT_ROOT="${EVAL_RESULT_ROOT:-${TORCHTITAN_COUNTDOWN_ROOT}/results/eval/adapters/${MODE}}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-32}"
FORCE="${FORCE:-0}"
REQUESTED_ARMS="${ARMS:-}"
STAGE_EVENTS=()

case "${MODE}" in
  reduced)
    ARMS=(raw hindsight curriculum)
    DEFAULT_DEV_SIZE=300
    DEFAULT_IID_SIZE=300
    DEFAULT_OOD_SIZE=200
    ;;
  full)
    ARMS=(raw clean formatting hindsight curriculum)
    DEFAULT_DEV_SIZE=500
    DEFAULT_IID_SIZE=1000
    DEFAULT_OOD_SIZE=500
    ;;
  *) echo "adapter eval is only defined for MODE=reduced or MODE=full, got ${MODE}" >&2; exit 2 ;;
esac
if [[ -n "${REQUESTED_ARMS}" ]]; then
  read -r -a ARMS <<< "${REQUESTED_ARMS}"
fi

if [[ -n "${SPLITS:-}" ]]; then
  read -r -a SPLIT_LIST <<< "${SPLITS}"
else
  SPLIT_LIST=(dev iid_test ood_test)
fi
EXPECTED_PROBLEMS=()
for split in "${SPLIT_LIST[@]}"; do
  case "${split}" in
    dev) EXPECTED_PROBLEMS+=("dev=${DEV_SIZE:-${DEFAULT_DEV_SIZE}}") ;;
    iid_test) EXPECTED_PROBLEMS+=("iid_test=${IID_SIZE:-${DEFAULT_IID_SIZE}}") ;;
    ood_test) EXPECTED_PROBLEMS+=("ood_test=${OOD_SIZE:-${DEFAULT_OOD_SIZE}}") ;;
    *) echo "unknown split=${split}" >&2; exit 2 ;;
  esac
done

for split in "${SPLIT_LIST[@]}"; do
  for arm in "${ARMS[@]}"; do
    summary="${EVAL_RESULT_ROOT}/${split}/${arm}/summary.json"
    if [[ "${FORCE}" != "1" && -f "${summary}" ]]; then
      echo "skip eval split=${split} arm=${arm}: ${summary} exists"
      STAGE_EVENTS+=("${split}/${arm}:reused:${summary}")
      continue
    fi
    SPLIT="${split}" \
      NUM_ROLLOUTS="${NUM_ROLLOUTS}" \
      LORA_ADAPTER="${ADAPTER_RESULT_ROOT}/${arm}" \
      LORA_NAME="${arm}" \
      RESULT_DIR="${EVAL_RESULT_ROOT}/${split}/${arm}" \
      "${SCRIPT_DIR}/run_eval.sh" "$@"
    STAGE_EVENTS+=("${split}/${arm}:fresh:${summary}")
  done
done

python -m torchtitan.experiments.countdown_search_distill.cli validate-eval-matrix \
  --eval-root "${EVAL_RESULT_ROOT}" \
  --decision "${EVAL_RESULT_ROOT}/adapter_matrix_${MODE}.json" \
  --splits "${SPLIT_LIST[@]}" \
  --arms "${ARMS[@]}" \
  --expected-problems "${EXPECTED_PROBLEMS[@]}" \
  --num-rollouts "${NUM_ROLLOUTS}"

if [[ -n "${TORCHTITAN_COUNTDOWN_STAGE_STATUS:-}" ]]; then
  STAGE_EVENTS_JSON="$(
    printf '%s\n' "${STAGE_EVENTS[@]}" |
      python -c 'import json, sys
events = []
for line in sys.stdin:
    line = line.rstrip("\n")
    if not line:
        continue
    name, status, path = line.split(":", 2)
    split, arm = name.split("/", 1)
    events.append({"split": split, "arm": arm, "status": status, "summary": path})
print(json.dumps({"artifacts": events}, sort_keys=True))'
  )"
  mkdir -p "$(dirname -- "${TORCHTITAN_COUNTDOWN_STAGE_STATUS}")"
  printf '%s\n' "${STAGE_EVENTS_JSON}" > "${TORCHTITAN_COUNTDOWN_STAGE_STATUS}"
fi
