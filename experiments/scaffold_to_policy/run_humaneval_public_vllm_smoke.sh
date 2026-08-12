#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "experiments/scaffold_to_policy/run_humaneval_public_vllm_smoke.sh" "$@"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export VLLM_USE_FLASHINFER_SAMPLER="${SCAFFOLD_TO_POLICY_VLLM_USE_FLASHINFER_SAMPLER:-0}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-humaneval-public-vllm-smoke}"
MODEL="${MODEL:-./assets/hf/Qwen3-1.7B}"
DATASET="${DATASET:-openai/openai_humaneval}"
DATASET_SUBSET="${DATASET_SUBSET:-}"
DATASET_REVISION="${DATASET_REVISION:-7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544}"
SOURCE_SPLIT="${SOURCE_SPLIT:-test}"
DATA_ROOT="${DATA_ROOT:-experiments/scaffold_to_policy/data/humaneval_public_vllm_smoke}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/humaneval_public_vllm_smoke}"
DEV_PROBLEMS="${DEV_PROBLEMS:-4}"
OOD_PROBLEMS="${OOD_PROBLEMS:-4}"
DEV_OFFSET="${DEV_OFFSET:-0}"
OOD_OFFSET="${OOD_OFFSET:-64}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
PROMPT_VARIANT="${PROMPT_VARIANT:-chat}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.95}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-5}"

mkdir -p "${DATA_ROOT}" "${RESULTS_ROOT}/eval" "${RESULTS_ROOT}/manifests" "${HF_HOME}"

subset_args=()
if [[ -n "${DATASET_SUBSET}" ]]; then
  subset_args=(--subset "${DATASET_SUBSET}")
fi

python -m torchtitan.experiments.scaffold_to_policy.cli import-humaneval-split \
  --dataset "${DATASET}" \
  "${subset_args[@]}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${DATASET_REVISION}" \
  --limit "${DEV_PROBLEMS}" \
  --offset "${DEV_OFFSET}" \
  --output "${DATA_ROOT}/dev.jsonl" \
  --provenance "${DATA_ROOT}/dev_provenance.json"

python -m torchtitan.experiments.scaffold_to_policy.cli import-humaneval-split \
  --dataset "${DATASET}" \
  "${subset_args[@]}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${DATASET_REVISION}" \
  --limit "${OOD_PROBLEMS}" \
  --offset "${OOD_OFFSET}" \
  --output "${DATA_ROOT}/ood_test.jsonl" \
  --provenance "${DATA_ROOT}/ood_test_provenance.json"

python -m torchtitan.experiments.scaffold_to_policy.cli validate-coding-style-splits \
  --split \
    "dev=${DATA_ROOT}/dev.jsonl" \
    "ood_test=${DATA_ROOT}/ood_test.jsonl" \
  --output "${DATA_ROOT}/split_registry.json"

for split in dev ood_test; do
  python -m torchtitan.experiments.scaffold_to_policy.cli evaluate-coding-style-vllm \
    --problems "${DATA_ROOT}/${split}.jsonl" \
    --model "${MODEL}" \
    --output "${RESULTS_ROOT}/eval/${split}_evaluations.jsonl" \
    --summary "${RESULTS_ROOT}/eval/${split}_summary.json" \
    --num-rollouts "${NUM_ROLLOUTS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --prompt-variant "${PROMPT_VARIANT}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --timeout-seconds "${TIMEOUT_SECONDS}"
done

python -m torchtitan.experiments.scaffold_to_policy.cli build-coding-style-report-input \
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
