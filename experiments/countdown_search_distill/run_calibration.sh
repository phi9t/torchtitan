#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"
countdown_enter_rootfs_if_needed "run_calibration.sh" "$@"
countdown_setup_env

case "${MODE:-full}" in
  smoke) SIZE="${SIZE:-50}" ;;
  reduced) SIZE="${SIZE:-200}" ;;
  full) SIZE="${SIZE:-500}" ;;
  *) echo "unknown MODE=${MODE}" >&2; exit 2 ;;
esac
MODEL="${MODEL:-./assets/hf/Qwen3-1.7B}"
OUT="${TORCHTITAN_COUNTDOWN_DATA_ROOT}/calibration"
mkdir -p "${OUT}"

REGIME_PATH="${REGIME_PATH:-${TORCHTITAN_COUNTDOWN_DATA_ROOT}/calibration_regime.json}"
if [[ "${MODE:-full}" != "smoke" && -f "${REGIME_PATH}" ]]; then
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

GENERATE_ARGS=(
  --output "${OUT}/problems.jsonl"
  --num-problems "${SIZE}"
  --seed "${SEED:-42}"
  --num-numbers "${NUM_NUMBERS}"
  --target-min "${TARGET_MIN}"
  --target-max "${TARGET_MAX}"
  --min-solution-depth "${MIN_SOLUTION_DEPTH}"
)
if [[ "${REQUIRE_ALL_NUMBERS}" == "1" ]]; then
  GENERATE_ARGS+=(--require-all-numbers)
else
  GENERATE_ARGS+=(--no-require-all-numbers)
fi

python -m torchtitan.experiments.countdown_search_distill.cli generate-pool \
  "${GENERATE_ARGS[@]}"

python -m torchtitan.experiments.countdown_search_distill.cli evaluate-vllm \
  --problems "${OUT}/problems.jsonl" \
  --model "${MODEL}" \
  --output "${OUT}/evaluations.jsonl" \
  --rollouts-output "${OUT}/rollouts.jsonl" \
  --annotations "${OUT}/annotations.jsonl" \
  --summary "${OUT}/summary.json" \
  --num-rollouts "${NUM_ROLLOUTS:-32}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --top-p "${TOP_P:-0.95}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-192}" \
  --prompt-variant "${PROMPT_VARIANT}"

python -m torchtitan.experiments.countdown_search_distill.cli calibrate \
  --evaluations "${OUT}/evaluations.jsonl" \
  --summary "${OUT}/calibration_summary.json" \
  --decision "${OUT}/decision.json" \
  --pass1-min "${PASS1_MIN:-0.0}" \
  --pass1-max "${PASS1_MAX:-0.20}" \
  --pass32-min "${PASS32_MIN:-0.35}" \
  --pass32-max "${PASS32_MAX:-0.70}"
