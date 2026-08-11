#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"
countdown_enter_rootfs_if_needed "run_collect.sh" "$@"
countdown_setup_env

MODE="${MODE:-full}"
MODEL="${MODEL:-./assets/hf/Qwen3-1.7B}"
case "${MODE}" in
  smoke)
    TRAIN_SIZE="${TRAIN_SIZE:-100}"
    DEV_SIZE="${DEV_SIZE:-50}"
    IID_SIZE="${IID_SIZE:-100}"
    OOD_SIZE="${OOD_SIZE:-50}"
    ;;
  reduced)
    TRAIN_SIZE="${TRAIN_SIZE:-1000}"
    DEV_SIZE="${DEV_SIZE:-300}"
    IID_SIZE="${IID_SIZE:-300}"
    OOD_SIZE="${OOD_SIZE:-200}"
    ;;
  full)
    TRAIN_SIZE="${TRAIN_SIZE:-2000}"
    DEV_SIZE="${DEV_SIZE:-500}"
    IID_SIZE="${IID_SIZE:-1000}"
    OOD_SIZE="${OOD_SIZE:-500}"
    ;;
  *) echo "unknown MODE=${MODE}" >&2; exit 2 ;;
esac

REGIME_PATH="${REGIME_PATH:-${TORCHTITAN_COUNTDOWN_ROOT}/data/calibration_regime.json}"
if [[ "${MODE}" != "smoke" ]]; then
  if [[ ! -f "${REGIME_PATH}" ]]; then
    echo "missing selected calibration regime: ${REGIME_PATH}" >&2
    echo "run experiments/countdown_search_distill/run_calibration_sweep.sh first" >&2
    exit 2
  fi
  read -r NUM_NUMBERS TARGET_MIN TARGET_MAX MIN_SOLUTION_DEPTH REQUIRE_ALL_NUMBERS PROMPT_VARIANT < <(
    REGIME_PATH="${REGIME_PATH}" python - <<'PY'
import json
import os
from pathlib import Path

regime = json.loads(Path(os.environ["REGIME_PATH"]).read_text())
print(
    regime["num_numbers"],
    regime["target_min"],
    regime["target_max"],
    regime["min_solution_depth"],
    int(regime["require_all_numbers"]),
    regime.get("prompt_variant", "default"),
)
PY
  )
else
  NUM_NUMBERS="${NUM_NUMBERS:-6}"
  TARGET_MIN="${TARGET_MIN:-100}"
  TARGET_MAX="${TARGET_MAX:-499}"
  MIN_SOLUTION_DEPTH="${MIN_SOLUTION_DEPTH:-3}"
  REQUIRE_ALL_NUMBERS="${REQUIRE_ALL_NUMBERS:-1}"
  PROMPT_VARIANT="${PROMPT_VARIANT:-default}"
fi
OOD_TARGET_MIN="${OOD_TARGET_MIN:-$((TARGET_MAX + 1))}"
OOD_TARGET_MAX="${OOD_TARGET_MAX:-$((TARGET_MAX + TARGET_MAX - TARGET_MIN + 1))}"

generate_and_eval() {
  local split="$1"
  local size="$2"
  local target_min="$3"
  local target_max="$4"
  shift 4
  local exclude_paths=("$@")
  local out="${TORCHTITAN_COUNTDOWN_ROOT}/data/${split}"
  mkdir -p "${out}"
  GENERATE_ARGS=(
    --output "${out}/problems.jsonl"
    --num-problems "${size}"
    --seed "${SEED:-42}"
    --num-numbers "${NUM_NUMBERS}"
    --target-min "${target_min}"
    --target-max "${target_max}"
    --min-solution-depth "${MIN_SOLUTION_DEPTH}"
  )
  if (( ${#exclude_paths[@]} > 0 )); then
    GENERATE_ARGS+=(--exclude-problems "${exclude_paths[@]}")
  fi
  if [[ "${REQUIRE_ALL_NUMBERS}" == "1" ]]; then
    GENERATE_ARGS+=(--require-all-numbers)
  else
    GENERATE_ARGS+=(--no-require-all-numbers)
  fi
  python -m torchtitan.experiments.countdown_search_distill.cli generate-pool \
    "${GENERATE_ARGS[@]}"
  python -m torchtitan.experiments.countdown_search_distill.cli evaluate-vllm \
    --problems "${out}/problems.jsonl" \
    --model "${MODEL}" \
    --output "${out}/evaluations.jsonl" \
    --rollouts-output "${out}/rollouts.jsonl" \
    --annotations "${out}/annotations.jsonl" \
    --summary "${out}/summary.json" \
    --num-rollouts "${NUM_ROLLOUTS:-32}" \
    --temperature "${TEMPERATURE:-0.8}" \
    --top-p "${TOP_P:-0.95}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-192}" \
    --prompt-variant "${PROMPT_VARIANT}"
}

generate_and_eval train "${TRAIN_SIZE}" "${TARGET_MIN}" "${TARGET_MAX}"
generate_and_eval dev "${DEV_SIZE}" "${TARGET_MIN}" "${TARGET_MAX}" \
  "${TORCHTITAN_COUNTDOWN_ROOT}/data/train/problems.jsonl"
generate_and_eval iid_test "${IID_SIZE}" "${TARGET_MIN}" "${TARGET_MAX}" \
  "${TORCHTITAN_COUNTDOWN_ROOT}/data/train/problems.jsonl" \
  "${TORCHTITAN_COUNTDOWN_ROOT}/data/dev/problems.jsonl"
generate_and_eval ood_test "${OOD_SIZE}" "${OOD_TARGET_MIN}" "${OOD_TARGET_MAX}" \
  "${TORCHTITAN_COUNTDOWN_ROOT}/data/train/problems.jsonl" \
  "${TORCHTITAN_COUNTDOWN_ROOT}/data/dev/problems.jsonl" \
  "${TORCHTITAN_COUNTDOWN_ROOT}/data/iid_test/problems.jsonl"

DATASET_ARGS=(
  --evaluations "${TORCHTITAN_COUNTDOWN_ROOT}/data/train/evaluations.jsonl"
  --output-dir "${TORCHTITAN_COUNTDOWN_ROOT}/data/train"
  --matched-only
)
if [[ "${MODE}" == "smoke" ]]; then
  DATASET_ARGS+=(--canonical-raw-fallback)
fi
python -m torchtitan.experiments.countdown_search_distill.cli build-datasets \
  "${DATASET_ARGS[@]}"
