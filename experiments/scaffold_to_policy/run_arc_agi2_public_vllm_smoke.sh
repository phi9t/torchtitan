#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "experiments/scaffold_to_policy/run_arc_agi2_public_vllm_smoke.sh" "$@"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export VLLM_USE_FLASHINFER_SAMPLER="${SCAFFOLD_TO_POLICY_VLLM_USE_FLASHINFER_SAMPLER:-0}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-arc-agi2-public-vllm-smoke}"
MODEL="${MODEL:-./assets/hf/Qwen3-1.7B}"
ARC_REPO_URL="${ARC_REPO_URL:-https://github.com/arcprize/ARC-AGI-2.git}"
ARC_REVISION="${ARC_REVISION:-f3283f727488ad98fe575ea6a5ac981e4a188e49}"
SOURCE_SPLIT="${SOURCE_SPLIT:-training}"
DATA_ROOT="${DATA_ROOT:-experiments/scaffold_to_policy/data/arc_agi2_public_vllm_smoke}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/arc_agi2_public_vllm_smoke}"
DEV_PROBLEMS="${DEV_PROBLEMS:-2}"
OOD_PROBLEMS="${OOD_PROBLEMS:-2}"
DEV_OFFSET="${DEV_OFFSET:-0}"
OOD_OFFSET="${OOD_OFFSET:-64}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
PROMPT_VARIANT="${PROMPT_VARIANT:-chat}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.95}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${SCAFFOLD_TO_POLICY_VLLM_GPU_MEMORY_UTILIZATION:-}}"
gpu_memory_args=()
if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  gpu_memory_args=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi

mkdir -p "${DATA_ROOT}/src" "${RESULTS_ROOT}/eval" "${RESULTS_ROOT}/manifests" "${HF_HOME}"

ARC_SRC="${DATA_ROOT}/src/ARC-AGI-2"
if [[ ! -d "${ARC_SRC}/.git" ]]; then
  rm -rf "${ARC_SRC}"
  git clone "${ARC_REPO_URL}" "${ARC_SRC}"
fi
git -C "${ARC_SRC}" fetch --depth 1 origin "${ARC_REVISION}"
git -C "${ARC_SRC}" checkout --detach "${ARC_REVISION}"

python -m torchtitan.experiments.scaffold_to_policy.cli import-arc-grid-split \
  --task-dir "${ARC_SRC}/data" \
  --repo-url "${ARC_REPO_URL}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${ARC_REVISION}" \
  --limit "${DEV_PROBLEMS}" \
  --offset "${DEV_OFFSET}" \
  --output "${DATA_ROOT}/dev.jsonl" \
  --provenance "${DATA_ROOT}/dev_provenance.json"

python -m torchtitan.experiments.scaffold_to_policy.cli import-arc-grid-split \
  --task-dir "${ARC_SRC}/data" \
  --repo-url "${ARC_REPO_URL}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${ARC_REVISION}" \
  --limit "${OOD_PROBLEMS}" \
  --offset "${OOD_OFFSET}" \
  --output "${DATA_ROOT}/ood_test.jsonl" \
  --provenance "${DATA_ROOT}/ood_test_provenance.json"

python -m torchtitan.experiments.scaffold_to_policy.cli validate-arc-grid-splits \
  --split \
    "dev=${DATA_ROOT}/dev.jsonl" \
    "ood_test=${DATA_ROOT}/ood_test.jsonl" \
  --output "${DATA_ROOT}/split_registry.json"

for split in dev ood_test; do
  python -m torchtitan.experiments.scaffold_to_policy.cli evaluate-arc-grid-vllm \
    --problems "${DATA_ROOT}/${split}.jsonl" \
    --model "${MODEL}" \
    --output "${RESULTS_ROOT}/eval/${split}_evaluations.jsonl" \
    --summary "${RESULTS_ROOT}/eval/${split}_summary.json" \
    --num-rollouts "${NUM_ROLLOUTS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --prompt-variant "${PROMPT_VARIANT}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    "${gpu_memory_args[@]}"
done

python -m torchtitan.experiments.scaffold_to_policy.cli build-arc-grid-report-input \
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
