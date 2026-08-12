#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"
countdown_enter_rootfs_if_needed "run_calibration_sweep.sh" "$@"
countdown_setup_env

MODEL="${MODEL:-./assets/hf/Qwen3-1.7B}"
SWEEP_ROOT="${TORCHTITAN_COUNTDOWN_DATA_ROOT}/calibration_sweep"
mkdir -p "${SWEEP_ROOT}"

case "${MODE:-full}" in
  smoke)
    SIZE="${SIZE:-20}"
    NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
    REQUIRE_SELECTION="${REQUIRE_SELECTION:-0}"
    CANDIDATES=(
      "4n_10_99_d2_all:4:10:99:2:1"
    )
    ;;
  reduced)
    SIZE="${SIZE:-200}"
    NUM_ROLLOUTS="${NUM_ROLLOUTS:-32}"
    REQUIRE_SELECTION="${REQUIRE_SELECTION:-1}"
    CANDIDATES=(
      "3n_10_50_d2_all:3:10:50:2:1"
      "3n_10_99_d2_all:3:10:99:2:1"
      "4n_10_99_d2_all:4:10:99:2:1"
      "4n_25_199_d2_all:4:25:199:2:1"
      "5n_50_299_d3_all:5:50:299:3:1"
      "6n_100_499_d3_all:6:100:499:3:1"
    )
    ;;
  full)
    SIZE="${SIZE:-500}"
    NUM_ROLLOUTS="${NUM_ROLLOUTS:-32}"
    REQUIRE_SELECTION="${REQUIRE_SELECTION:-1}"
    CANDIDATES=(
      "3n_10_50_d2_all:3:10:50:2:1"
      "3n_10_99_d2_all:3:10:99:2:1"
      "4n_10_99_d2_all:4:10:99:2:1"
      "4n_25_199_d2_all:4:25:199:2:1"
      "5n_50_299_d3_all:5:50:299:3:1"
      "6n_100_499_d3_all:6:100:499:3:1"
    )
    ;;
  *) echo "unknown MODE=${MODE:-full}" >&2; exit 2 ;;
esac

CANDIDATES_JSONL="${SWEEP_ROOT}/candidates.jsonl"
: > "${CANDIDATES_JSONL}"

for candidate in "${CANDIDATES[@]}"; do
  IFS=: read -r name num_numbers target_min target_max min_depth require_all <<< "${candidate}"
  OUT="${SWEEP_ROOT}/${name}"
  mkdir -p "${OUT}"
  GENERATE_ARGS=(
    --output "${OUT}/problems.jsonl"
    --num-problems "${SIZE}"
    --seed "${SEED:-42}"
    --num-numbers "${num_numbers}"
    --target-min "${target_min}"
    --target-max "${target_max}"
    --min-solution-depth "${min_depth}"
  )
  if [[ "${require_all}" == "1" ]]; then
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
    --num-rollouts "${NUM_ROLLOUTS}" \
    --temperature "${TEMPERATURE:-0.8}" \
    --top-p "${TOP_P:-0.95}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-192}" \
    --prompt-variant "${PROMPT_VARIANT:-default}"
  python - <<PY
import json
from pathlib import Path

row = {
    "name": "${name}",
    "summary": str(Path("${OUT}/summary.json")),
    "problems": str(Path("${OUT}/problems.jsonl")),
    "evaluations": str(Path("${OUT}/evaluations.jsonl")),
    "num_numbers": int("${num_numbers}"),
    "target_min": int("${target_min}"),
    "target_max": int("${target_max}"),
    "min_solution_depth": int("${min_depth}"),
    "require_all_numbers": "${require_all}" == "1",
    "prompt_variant": "${PROMPT_VARIANT:-default}",
    "num_rollouts": int("${NUM_ROLLOUTS}"),
}
with Path("${CANDIDATES_JSONL}").open("a") as f:
    f.write(json.dumps(row, sort_keys=True) + "\\n")
PY
done

SELECTION_ARGS=()
if [[ "${REQUIRE_SELECTION}" == "1" ]]; then
  SELECTION_ARGS+=(--require-selection)
else
  SELECTION_ARGS+=(--no-require-selection)
fi
python -m torchtitan.experiments.countdown_search_distill.cli select-sweep \
  --candidates "${CANDIDATES_JSONL}" \
  --output "${TORCHTITAN_COUNTDOWN_DATA_ROOT}/calibration_regime.json" \
  --decision "${SWEEP_ROOT}/decision.json" \
  --pass1-min "${PASS1_MIN:-0.0}" \
  --pass1-max "${PASS1_MAX:-0.20}" \
  --pass32-min "${PASS32_MIN:-0.35}" \
  --pass32-max "${PASS32_MAX:-0.70}" \
  "${SELECTION_ARGS[@]}"
