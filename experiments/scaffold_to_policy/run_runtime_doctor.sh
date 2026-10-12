#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"

scaffold_enter_rootfs_if_needed "run_runtime_doctor.sh" "$@"
scaffold_setup_env

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-runtime-doctor}"
MODEL="${MODEL:-./assets/hf/Qwen3-1.7B}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/runtime_doctor}"
MIN_GPUS="${MIN_GPUS:-1}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.05}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"

mkdir -p "${RESULTS_ROOT}/manifests"
scaffold_setup_run_manifest

package_args=()
if [[ -n "${REQUIRED_PACKAGES:-}" ]]; then
  IFS=',' read -r -a packages <<< "${REQUIRED_PACKAGES}"
  for package in "${packages[@]}"; do
    package_args+=(--required-package "${package}")
  done
fi

executable_args=()
if [[ -n "${REQUIRED_EXECUTABLES:-}" ]]; then
  IFS=',' read -r -a executables <<< "${REQUIRED_EXECUTABLES}"
  for executable in "${executables[@]}"; do
    executable_args+=(--required-executable "${executable}")
  done
fi

scaffold_run_stage runtime_doctor python -m torchtitan.experiments.scaffold_to_policy.cli doctor-runtime-contract \
  --output "${RESULTS_ROOT}/manifests/runtime_doctor_${RUN_ID}.json" \
  --run-id "${RUN_ID}" \
  --model "${MODEL}" \
  --min-gpus "${MIN_GPUS}" \
  --device-index "${DEVICE_INDEX}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  "${package_args[@]}" \
  "${executable_args[@]}"

echo "wrote ${RESULTS_ROOT}/manifests/runtime_doctor_${RUN_ID}.json"
