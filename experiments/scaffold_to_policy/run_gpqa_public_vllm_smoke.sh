#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"

scaffold_enter_rootfs_if_needed "run_gpqa_public_vllm_smoke.sh" "$@"
scaffold_setup_env

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-gpqa-public-vllm-smoke}"
MODEL="${MODEL:-./assets/hf/Qwen3-1.7B}"
DATASET="${DATASET:-Idavidrein/gpqa}"
DATASET_SUBSET="${DATASET_SUBSET:-gpqa_diamond}"
DATASET_REVISION="${DATASET_REVISION:-main}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train}"
DATA_ROOT="${DATA_ROOT:-experiments/scaffold_to_policy/data/gpqa_public_vllm_smoke}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/gpqa_public_vllm_smoke}"
DEV_PROBLEMS="${DEV_PROBLEMS:-4}"
OOD_PROBLEMS="${OOD_PROBLEMS:-4}"
DEV_OFFSET="${DEV_OFFSET:-0}"
OOD_OFFSET="${OOD_OFFSET:-64}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
PROMPT_VARIANT="${PROMPT_VARIANT:-chat}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.95}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
OFFLINE="${OFFLINE:-0}"
DEV_RAW_CACHE="${DEV_RAW_CACHE:-}"
OOD_RAW_CACHE="${OOD_RAW_CACHE:-}"
RUNTIME_METADATA="${RESULTS_ROOT}/manifests/runtime_${RUN_ID}.json"

mkdir -p "${DATA_ROOT}" "${RESULTS_ROOT}/eval" "${RESULTS_ROOT}/manifests" "${HF_HOME}"
scaffold_setup_run_manifest
scaffold_capture_vllm_runtime_metadata "${RUNTIME_METADATA}"

offline_args=()
case "${OFFLINE}" in
  1|true|TRUE|yes|YES)
    offline_args=(--offline)
    ;;
esac

dev_cache_args=()
if [[ -n "${DEV_RAW_CACHE}" ]]; then
  dev_cache_args=(--raw-cache "${DEV_RAW_CACHE}")
fi

ood_cache_args=()
if [[ -n "${OOD_RAW_CACHE}" ]]; then
  ood_cache_args=(--raw-cache "${OOD_RAW_CACHE}")
fi

if ! scaffold_run_stage import_dev python -m torchtitan.experiments.scaffold_to_policy.cli import-gpqa-split \
  --dataset "${DATASET}" \
  --subset "${DATASET_SUBSET}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${DATASET_REVISION}" \
  --limit "${DEV_PROBLEMS}" \
  --offset "${DEV_OFFSET}" \
  --output "${DATA_ROOT}/dev.jsonl" \
  --provenance "${DATA_ROOT}/dev_provenance.json" \
  "${dev_cache_args[@]}" \
  "${offline_args[@]}"; then
  FAILURE_MARKER="${RESULTS_ROOT}/eval/gpqa_import_failure.json"
  scaffold_write_stage_failure_marker import_dev gpqa_import_blocker "${FAILURE_MARKER}"
  scaffold_run_stage write_import_blocker_report_input python -m torchtitan.experiments.scaffold_to_policy.cli write-blocker-report-input \
    --results-root "${RESULTS_ROOT}" \
    --run-id "${RUN_ID}" \
    --task multiple_choice \
    --lane reasoning \
    --blocker-type gpqa_import_blocker \
    --artifact "import_failure=${FAILURE_MARKER}" \
    --runtime "${RUNTIME_METADATA}" \
    --limitation "GPQA Diamond import failed, commonly because the dataset is gated and HF_TOKEN is not configured inside the rootfs." \
    --limitation "No benchmark task execution or model score was produced." \
    --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
  echo "wrote GPQA blocker ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
  exit 0
fi

scaffold_run_stage import_ood_test python -m torchtitan.experiments.scaffold_to_policy.cli import-gpqa-split \
  --dataset "${DATASET}" \
  --subset "${DATASET_SUBSET}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${DATASET_REVISION}" \
  --limit "${OOD_PROBLEMS}" \
  --offset "${OOD_OFFSET}" \
  --output "${DATA_ROOT}/ood_test.jsonl" \
  --provenance "${DATA_ROOT}/ood_test_provenance.json" \
  "${ood_cache_args[@]}" \
  "${offline_args[@]}"

scaffold_run_stage validate_splits python -m torchtitan.experiments.scaffold_to_policy.cli validate-multiple-choice-splits \
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
  scaffold_run_stage write_gpu_blocker_report_input python -m torchtitan.experiments.scaffold_to_policy.cli write-blocker-report-input \
    --results-root "${RESULTS_ROOT}" \
    --run-id "${RUN_ID}" \
    --task multiple_choice \
    --lane reasoning \
    --blocker-type vllm_gpu_memory_preflight \
    --artifact "gpu_memory=${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json" \
    --runtime "${RUNTIME_METADATA}" \
    --limitation "GPQA Diamond run stopped before model execution because vLLM GPU memory preflight failed." \
    --limitation "No benchmark task execution or model score was produced." \
    --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
  echo "wrote GPQA GPU blocker ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
  exit 0
fi

for split in dev ood_test; do
  if ! scaffold_run_stage "evaluate_${split}" python -m torchtitan.experiments.scaffold_to_policy.cli evaluate-multiple-choice-vllm \
    --problems "${DATA_ROOT}/${split}.jsonl" \
    --model "${MODEL}" \
    --output "${RESULTS_ROOT}/eval/${split}_evaluations.jsonl" \
    --summary "${RESULTS_ROOT}/eval/${split}_summary.json" \
    --num-rollouts "${NUM_ROLLOUTS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --prompt-variant "${PROMPT_VARIANT}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"; then
    FAILURE_MARKER="${RESULTS_ROOT}/eval/${split}_vllm_runtime_failure.json"
    scaffold_write_stage_failure_marker "evaluate_${split}" vllm_runtime_failure "${FAILURE_MARKER}"
    scaffold_run_stage write_runtime_blocker_report_input python -m torchtitan.experiments.scaffold_to_policy.cli write-blocker-report-input \
      --results-root "${RESULTS_ROOT}" \
      --run-id "${RUN_ID}" \
      --task multiple_choice \
      --lane reasoning \
      --blocker-type vllm_runtime_failure \
      --artifact \
        "gpu_memory=${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json" \
        "stage_failure=${FAILURE_MARKER}" \
      --runtime "${RUNTIME_METADATA}" \
      --limitation "GPQA Diamond run stopped during ${split} model execution because vLLM failed at runtime." \
      --limitation "No complete benchmark task execution or model score was produced." \
      --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
    echo "wrote GPQA runtime blocker ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
    exit 0
  fi
done

scaffold_run_stage build_report_input python -m torchtitan.experiments.scaffold_to_policy.cli build-multiple-choice-report-input \
  --data-root "${DATA_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --split-registry "${DATA_ROOT}/split_registry.json" \
  --summary \
    "dev=${RESULTS_ROOT}/eval/dev_summary.json" \
    "ood_test=${RESULTS_ROOT}/eval/ood_test_summary.json" \
  --scaffold-budget "${NUM_ROLLOUTS}" \
  --runtime "${RUNTIME_METADATA}" \
  --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"

echo "wrote ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
