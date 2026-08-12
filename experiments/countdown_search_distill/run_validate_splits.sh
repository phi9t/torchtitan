#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"
countdown_enter_rootfs_if_needed "run_validate_splits.sh" "$@"
countdown_setup_env

SPLIT_REGISTRY="${SPLIT_REGISTRY:-${TORCHTITAN_COUNTDOWN_DATA_ROOT}/split_registry.json}"

python -m torchtitan.experiments.countdown_search_distill.cli validate-splits \
  --split \
    "train=${TORCHTITAN_COUNTDOWN_DATA_ROOT}/train/problems.jsonl" \
    "dev=${TORCHTITAN_COUNTDOWN_DATA_ROOT}/dev/problems.jsonl" \
    "iid_test=${TORCHTITAN_COUNTDOWN_DATA_ROOT}/iid_test/problems.jsonl" \
    "ood_test=${TORCHTITAN_COUNTDOWN_DATA_ROOT}/ood_test/problems.jsonl" \
  --output "${SPLIT_REGISTRY}" \
  "$@"
