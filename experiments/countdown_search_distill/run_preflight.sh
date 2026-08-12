#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"
countdown_enter_rootfs_if_needed "run_preflight.sh" "$@"
countdown_setup_env

MODEL="${MODEL:-./assets/hf/Qwen3-1.7B}"
DECISION="${DECISION:-${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}/runtime_preflight.json}"

python -m torchtitan.experiments.countdown_search_distill.cli preflight-runtime \
  --model "${MODEL}" \
  --decision "${DECISION}" \
  "$@"
