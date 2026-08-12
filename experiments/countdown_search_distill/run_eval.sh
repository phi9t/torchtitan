#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"
countdown_enter_rootfs_if_needed "run_eval.sh" "$@"
countdown_setup_env

SPLIT="${SPLIT:-iid_test}"
MODEL="${MODEL:-./assets/hf/Qwen3-1.7B}"
DATA_DIR="${TORCHTITAN_COUNTDOWN_DATA_ROOT}/${SPLIT}"
if [[ -n "${LORA_ADAPTER:-}" ]]; then
  RESULT_DIR="${RESULT_DIR:-${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}/eval/${SPLIT}/${LORA_NAME:-countdown_adapter}}"
else
  RESULT_DIR="${RESULT_DIR:-${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}/eval/${SPLIT}/base}"
fi
mkdir -p "${RESULT_DIR}"

REGIME_PATH="${REGIME_PATH:-${TORCHTITAN_COUNTDOWN_DATA_ROOT}/calibration_regime.json}"
if [[ -f "${REGIME_PATH}" ]]; then
  PROMPT_VARIANT="$(
    REGIME_PATH="${REGIME_PATH}" python - <<'PY'
import json
import os
from pathlib import Path

regime = json.loads(Path(os.environ["REGIME_PATH"]).read_text())
print(regime.get("prompt_variant", "default"))
PY
  )"
else
  PROMPT_VARIANT="${PROMPT_VARIANT:-default}"
fi

LORA_ARGS=()
if [[ -n "${LORA_ADAPTER:-}" ]]; then
  LORA_ARGS=(
    --lora-adapter "${LORA_ADAPTER}"
    --lora-name "${LORA_NAME:-countdown_adapter}"
    --lora-id "${LORA_ID:-1}"
  )
fi

python -m torchtitan.experiments.countdown_search_distill.cli evaluate-vllm \
  --problems "${DATA_DIR}/problems.jsonl" \
  --model "${MODEL}" \
  --output "${RESULT_DIR}/evaluations.jsonl" \
  --rollouts-output "${RESULT_DIR}/rollouts.jsonl" \
  --annotations "${RESULT_DIR}/annotations.jsonl" \
  --summary "${RESULT_DIR}/summary.json" \
  --num-rollouts "${NUM_ROLLOUTS:-32}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --top-p "${TOP_P:-0.95}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-192}" \
  --prompt-variant "${PROMPT_VARIANT}" \
  "${LORA_ARGS[@]}"
