#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"
countdown_enter_rootfs_if_needed "run_train.sh" "$@"
countdown_setup_env

ARM="${ARM:-raw}"
NGPU="${NGPU:-1}"
TRAIN_ARGS=()

case "${ARM}" in
  raw) CONFIG="qwen3_1_7b_countdown_lora_raw" ;;
  clean) CONFIG="qwen3_1_7b_countdown_lora_clean" ;;
  formatting) CONFIG="qwen3_1_7b_countdown_lora_formatting" ;;
  hindsight) CONFIG="qwen3_1_7b_countdown_lora_hindsight" ;;
  curriculum) CONFIG="qwen3_1_7b_countdown_lora_curriculum" ;;
  debug_smoke) CONFIG="qwen3_debugmodel_countdown_lora_smoke" ;;
  *) echo "unknown ARM=${ARM}" >&2; exit 2 ;;
esac

if [[ -n "${TRAIN_RESULT_ROOT:-}" ]]; then
  TRAIN_ARGS+=(--dump-folder "${TRAIN_RESULT_ROOT}/${ARM}")
fi

torchrun --standalone --nproc_per_node="${NGPU}" torchtitan/train.py \
  --module qwen3 \
  --config "${CONFIG}" \
  "${TRAIN_ARGS[@]}" \
  "$@"
