#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"

scaffold_enter_rootfs_if_needed "run_livecodebench_public_vllm_smoke.sh" "$@"
scaffold_setup_env

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-livecodebench-public-vllm-smoke}"
MODEL="${MODEL:-./assets/hf/Qwen3-1.7B}"
DATASET="${DATASET:-livecodebench/code_generation}"
DATASET_SUBSET="${DATASET_SUBSET:-}"
DATASET_REVISION="${DATASET_REVISION:-main}"
SOURCE_SPLIT="${SOURCE_SPLIT:-test}"
DATA_ROOT="${DATA_ROOT:-experiments/scaffold_to_policy/data/livecodebench_public_vllm_smoke}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/livecodebench_public_vllm_smoke}"
DEV_PROBLEMS="${DEV_PROBLEMS:-2}"
OOD_PROBLEMS="${OOD_PROBLEMS:-2}"
DEV_OFFSET="${DEV_OFFSET:-0}"
OOD_OFFSET="${OOD_OFFSET:-32}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
PROMPT_VARIANT="${PROMPT_VARIANT:-chat}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.95}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-10}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

mkdir -p "${DATA_ROOT}" "${RESULTS_ROOT}/eval" "${RESULTS_ROOT}/manifests" "${HF_HOME}"
scaffold_setup_run_manifest

subset_args=()
if [[ -n "${DATASET_SUBSET}" ]]; then
  subset_args=(--subset "${DATASET_SUBSET}")
fi

scaffold_run_stage import_dev python -m torchtitan.experiments.scaffold_to_policy.cli import-livecodebench-split \
  --dataset "${DATASET}" \
  "${subset_args[@]}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${DATASET_REVISION}" \
  --limit "${DEV_PROBLEMS}" \
  --offset "${DEV_OFFSET}" \
  --output "${DATA_ROOT}/dev.jsonl" \
  --provenance "${DATA_ROOT}/dev_provenance.json"

scaffold_run_stage import_ood_test python -m torchtitan.experiments.scaffold_to_policy.cli import-livecodebench-split \
  --dataset "${DATASET}" \
  "${subset_args[@]}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${DATASET_REVISION}" \
  --limit "${OOD_PROBLEMS}" \
  --offset "${OOD_OFFSET}" \
  --output "${DATA_ROOT}/ood_test.jsonl" \
  --provenance "${DATA_ROOT}/ood_test_provenance.json"

scaffold_run_stage validate_splits python -m torchtitan.experiments.scaffold_to_policy.cli validate-contest-code-splits \
  --split \
    "dev=${DATA_ROOT}/dev.jsonl" \
    "ood_test=${DATA_ROOT}/ood_test.jsonl" \
  --output "${DATA_ROOT}/split_registry.json"

scaffold_run_stage preflight_gpu_memory python -m torchtitan.experiments.scaffold_to_policy.cli preflight-vllm-gpu-memory \
  --output "${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --no-require-selected
if ! scaffold_run_stage require_gpu_memory_selected python - <<'PY' "${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json"
import json
import sys

payload = json.loads(open(sys.argv[1]).read())
raise SystemExit(0 if payload.get("selected") else 1)
PY
then
  scaffold_run_stage write_blocker_report_input python -m torchtitan.experiments.scaffold_to_policy.cli write-blocker-report-input \
    --results-root "${RESULTS_ROOT}" \
    --run-id "${RUN_ID}" \
    --task contest_code \
    --lane coding \
    --blocker-type vllm_gpu_memory_preflight \
    --artifact "gpu_memory=${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json" \
    --limitation "LiveCodeBench coding run stopped before model execution because vLLM GPU memory preflight failed." \
    --limitation "No benchmark task execution or model score was produced." \
    --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
  echo "wrote LiveCodeBench blocker ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
  exit 0
fi

if ! scaffold_run_stage evaluate_splits python -m torchtitan.experiments.scaffold_to_policy.cli evaluate-contest-code-vllm-splits \
  --problems \
    "dev=${DATA_ROOT}/dev.jsonl" \
    "ood_test=${DATA_ROOT}/ood_test.jsonl" \
  --model "${MODEL}" \
  --output \
    "dev=${RESULTS_ROOT}/eval/dev_evaluations.jsonl" \
    "ood_test=${RESULTS_ROOT}/eval/ood_test_evaluations.jsonl" \
  --summary \
    "dev=${RESULTS_ROOT}/eval/dev_summary.json" \
    "ood_test=${RESULTS_ROOT}/eval/ood_test_summary.json" \
  --num-rollouts "${NUM_ROLLOUTS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --prompt-variant "${PROMPT_VARIANT}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --timeout-seconds "${TIMEOUT_SECONDS}"; then
  FAILURE_MARKER="${RESULTS_ROOT}/eval/vllm_runtime_failure.json"
  scaffold_write_stage_failure_marker evaluate_splits vllm_runtime_failure "${FAILURE_MARKER}"
  scaffold_run_stage write_runtime_blocker_report_input python -m torchtitan.experiments.scaffold_to_policy.cli write-blocker-report-input \
    --results-root "${RESULTS_ROOT}" \
    --run-id "${RUN_ID}" \
    --task contest_code \
    --lane coding \
    --blocker-type vllm_runtime_failure \
    --artifact \
      "gpu_memory=${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json" \
      "stage_failure=${FAILURE_MARKER}" \
    --limitation "LiveCodeBench coding run stopped during model execution because vLLM failed at runtime." \
    --limitation "No complete benchmark task execution or model score was produced." \
    --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
  echo "wrote LiveCodeBench runtime blocker ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
  exit 0
fi

scaffold_run_stage build_report_input python -m torchtitan.experiments.scaffold_to_policy.cli build-contest-code-report-input \
  --data-root "${DATA_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --split-registry "${DATA_ROOT}/split_registry.json" \
  --summary \
    "dev=${RESULTS_ROOT}/eval/dev_summary.json" \
    "ood_test=${RESULTS_ROOT}/eval/ood_test_summary.json" \
  --scaffold-budget "${NUM_ROLLOUTS}" \
  --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"

echo "wrote ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
