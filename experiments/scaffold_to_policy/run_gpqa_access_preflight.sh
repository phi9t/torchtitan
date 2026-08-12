#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"

scaffold_enter_rootfs_if_needed "run_gpqa_access_preflight.sh" "$@"
scaffold_setup_env

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-gpqa-access-preflight}"
DATASET="${DATASET:-Idavidrein/gpqa}"
DATASET_SUBSET="${DATASET_SUBSET:-gpqa_diamond}"
DATASET_REVISION="${DATASET_REVISION:-main}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train}"
DATA_ROOT="${DATA_ROOT:-experiments/scaffold_to_policy/data/gpqa_access_preflight}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/gpqa_access_preflight}"
DEV_PROBLEMS="${DEV_PROBLEMS:-1}"
OOD_PROBLEMS="${OOD_PROBLEMS:-1}"
DEV_OFFSET="${DEV_OFFSET:-0}"
OOD_OFFSET="${OOD_OFFSET:-64}"
OFFLINE="${OFFLINE:-0}"
DEV_RAW_CACHE="${DEV_RAW_CACHE:-}"
OOD_RAW_CACHE="${OOD_RAW_CACHE:-}"

mkdir -p "${DATA_ROOT}" "${RESULTS_ROOT}/manifests" "${HF_HOME}"
scaffold_setup_run_manifest

offline_args=()
case "${OFFLINE}" in
  1|true|TRUE|yes|YES)
    offline_args=(--offline)
    ;;
esac

cache_args=()
if [[ -n "${DEV_RAW_CACHE}" ]]; then
  cache_args+=(--dev-raw-cache "${DEV_RAW_CACHE}")
fi
if [[ -n "${OOD_RAW_CACHE}" ]]; then
  cache_args+=(--ood-raw-cache "${OOD_RAW_CACHE}")
fi

scaffold_run_stage preflight_gpqa_access python -m torchtitan.experiments.scaffold_to_policy.cli preflight-gpqa-access \
  --output "${RESULTS_ROOT}/manifests/gpqa_access_preflight_${RUN_ID}.json" \
  --dataset "${DATASET}" \
  --subset "${DATASET_SUBSET}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${DATASET_REVISION}" \
  --dev-limit "${DEV_PROBLEMS}" \
  --ood-limit "${OOD_PROBLEMS}" \
  --dev-offset "${DEV_OFFSET}" \
  --ood-offset "${OOD_OFFSET}" \
  "${cache_args[@]}" \
  "${offline_args[@]}" \
  --no-require-selected

echo "wrote ${RESULTS_ROOT}/manifests/gpqa_access_preflight_${RUN_ID}.json"
