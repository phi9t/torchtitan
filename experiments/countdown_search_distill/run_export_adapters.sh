#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"
countdown_enter_rootfs_if_needed "run_export_adapters.sh" "$@"
countdown_setup_env

MODE="${MODE:-full}"
MODEL="${MODEL:-./assets/hf/Qwen3-1.7B}"
TRAIN_RESULT_ROOT="${TRAIN_RESULT_ROOT:-${TORCHTITAN_COUNTDOWN_ROOT}/results/train/${MODE}}"
ADAPTER_RESULT_ROOT="${ADAPTER_RESULT_ROOT:-${TORCHTITAN_COUNTDOWN_ROOT}/results/adapters/${MODE}}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-step-94}"
FORCE="${FORCE:-0}"

case "${MODE}" in
  reduced) ARMS=(raw hindsight curriculum) ;;
  full) ARMS=(raw clean hindsight curriculum) ;;
  *) echo "adapter export is only defined for MODE=reduced or MODE=full, got ${MODE}" >&2; exit 2 ;;
esac

for arm in "${ARMS[@]}"; do
  checkpoint="${TRAIN_RESULT_ROOT}/${arm}/checkpoint/${CHECKPOINT_STEP}"
  output="${ADAPTER_RESULT_ROOT}/${arm}"
  summary="${output}/export_summary.json"
  if [[ "${FORCE}" != "1" && -f "${summary}" && -f "${output}/adapter_config.json" && -f "${output}/adapter_model.safetensors" ]]; then
    echo "skip export arm=${arm}: ${summary} exists"
    continue
  fi
  python -m torchtitan.experiments.countdown_search_distill.cli export-lora \
    --checkpoint "${checkpoint}" \
    --output "${output}" \
    --base-model-name-or-path "${MODEL}" \
    "$@"
done
