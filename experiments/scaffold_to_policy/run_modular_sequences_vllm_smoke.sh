#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "experiments/scaffold_to_policy/run_modular_sequences_vllm_smoke.sh" "$@"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER="${SCAFFOLD_TO_POLICY_VLLM_USE_FLASHINFER_SAMPLER:-0}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-modular-sequences-vllm-smoke}"
MODEL="${MODEL:-./assets/hf/Qwen3-1.7B}"
DATA_ROOT="${DATA_ROOT:-experiments/scaffold_to_policy/data/modular_sequences_vllm_smoke}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/modular_sequences_vllm_smoke}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
MIN_STEPS="${MIN_STEPS:-10}"
MAX_STEPS="${MAX_STEPS:-16}"
MIN_MODULUS="${MIN_MODULUS:-997}"
MAX_MODULUS="${MAX_MODULUS:-9973}"
PROMPT_VARIANT="${PROMPT_VARIANT:-${SCAFFOLD_TO_POLICY_PROMPT_VARIANT:-chat}}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"

mkdir -p "${DATA_ROOT}" "${RESULTS_ROOT}/eval" "${RESULTS_ROOT}/manifests"

python -m torchtitan.experiments.scaffold_to_policy.cli generate-modular-sequences \
  --seed 3100 \
  --num-problems 16 \
  --min-steps "${MIN_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --min-modulus "${MIN_MODULUS}" \
  --max-modulus "${MAX_MODULUS}" \
  --output "${DATA_ROOT}/train.jsonl"

python -m torchtitan.experiments.scaffold_to_policy.cli generate-modular-sequences \
  --seed 3200 \
  --num-problems 8 \
  --min-steps "${MIN_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --min-modulus "${MIN_MODULUS}" \
  --max-modulus "${MAX_MODULUS}" \
  --output "${DATA_ROOT}/dev.jsonl"

python -m torchtitan.experiments.scaffold_to_policy.cli generate-modular-sequences \
  --seed 3300 \
  --num-problems 8 \
  --min-steps "${MIN_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --min-modulus "${MIN_MODULUS}" \
  --max-modulus "${MAX_MODULUS}" \
  --output "${DATA_ROOT}/ood_test.jsonl"

python -m torchtitan.experiments.scaffold_to_policy.cli validate-modular-splits \
  --split \
    "train=${DATA_ROOT}/train.jsonl" \
    "dev=${DATA_ROOT}/dev.jsonl" \
    "ood_test=${DATA_ROOT}/ood_test.jsonl" \
  --output "${DATA_ROOT}/split_registry.json"

for split in dev ood_test; do
  python -m torchtitan.experiments.scaffold_to_policy.cli evaluate-modular-vllm \
    --problems "${DATA_ROOT}/${split}.jsonl" \
    --model "${MODEL}" \
    --output "${RESULTS_ROOT}/eval/${split}_evaluations.jsonl" \
    --summary "${RESULTS_ROOT}/eval/${split}_summary.json" \
    --num-rollouts "${NUM_ROLLOUTS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --prompt-variant "${PROMPT_VARIANT}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}"
done

python -m torchtitan.experiments.scaffold_to_policy.cli build-modular-report-input \
  --data-root "${DATA_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --split-registry "${DATA_ROOT}/split_registry.json" \
  --summary \
    "dev=${RESULTS_ROOT}/eval/dev_summary.json" \
    "ood_test=${RESULTS_ROOT}/eval/ood_test_summary.json" \
  --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"

echo "wrote ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
