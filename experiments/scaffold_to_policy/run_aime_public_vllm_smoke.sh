#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "experiments/scaffold_to_policy/run_aime_public_vllm_smoke.sh" "$@"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export VLLM_USE_FLASHINFER_SAMPLER="${SCAFFOLD_TO_POLICY_VLLM_USE_FLASHINFER_SAMPLER:-0}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-aime-public-vllm-smoke}"
MODEL="${MODEL:-./assets/hf/Qwen3-1.7B}"
DATASET="${DATASET:-HuggingFaceH4/aime_2024}"
DATASET_SUBSET="${DATASET_SUBSET:-}"
DATASET_REVISION="${DATASET_REVISION:-main}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train}"
DATA_ROOT="${DATA_ROOT:-experiments/scaffold_to_policy/data/aime_public_vllm_smoke}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/aime_public_vllm_smoke}"
DEV_PROBLEMS="${DEV_PROBLEMS:-8}"
OOD_PROBLEMS="${OOD_PROBLEMS:-8}"
DEV_OFFSET="${DEV_OFFSET:-0}"
OOD_OFFSET="${OOD_OFFSET:-16}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
PROMPT_VARIANT="${PROMPT_VARIANT:-chat}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

mkdir -p "${DATA_ROOT}" "${RESULTS_ROOT}/eval" "${RESULTS_ROOT}/manifests" "${HF_HOME}"

subset_args=()
if [[ -n "${DATASET_SUBSET}" ]]; then
  subset_args=(--subset "${DATASET_SUBSET}")
fi

python -m torchtitan.experiments.scaffold_to_policy.cli import-aime-split \
  --dataset "${DATASET}" \
  "${subset_args[@]}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${DATASET_REVISION}" \
  --limit "${DEV_PROBLEMS}" \
  --offset "${DEV_OFFSET}" \
  --output "${DATA_ROOT}/dev.jsonl" \
  --provenance "${DATA_ROOT}/dev_provenance.json"

python -m torchtitan.experiments.scaffold_to_policy.cli import-aime-split \
  --dataset "${DATASET}" \
  "${subset_args[@]}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${DATASET_REVISION}" \
  --limit "${OOD_PROBLEMS}" \
  --offset "${OOD_OFFSET}" \
  --output "${DATA_ROOT}/ood_test.jsonl" \
  --provenance "${DATA_ROOT}/ood_test_provenance.json"

python -m torchtitan.experiments.scaffold_to_policy.cli validate-math-style-splits \
  --split \
    "dev=${DATA_ROOT}/dev.jsonl" \
    "ood_test=${DATA_ROOT}/ood_test.jsonl" \
  --output "${DATA_ROOT}/split_registry.json"

python -m torchtitan.experiments.scaffold_to_policy.cli preflight-vllm-gpu-memory \
  --output "${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"

for split in dev ood_test; do
  python -m torchtitan.experiments.scaffold_to_policy.cli evaluate-math-style-vllm \
    --problems "${DATA_ROOT}/${split}.jsonl" \
    --model "${MODEL}" \
    --output "${RESULTS_ROOT}/eval/${split}_evaluations.jsonl" \
    --summary "${RESULTS_ROOT}/eval/${split}_summary.json" \
    --num-rollouts "${NUM_ROLLOUTS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --prompt-variant "${PROMPT_VARIANT}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
done

python -m torchtitan.experiments.scaffold_to_policy.cli build-math-style-report-input \
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
