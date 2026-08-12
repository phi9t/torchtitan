#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "experiments/scaffold_to_policy/run_modular_sequences_transfer_smoke.sh" "$@"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export VLLM_USE_FLASHINFER_SAMPLER="${SCAFFOLD_TO_POLICY_VLLM_USE_FLASHINFER_SAMPLER:-0}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-modular-sequences-transfer-smoke}"
MODEL="${MODEL:-./assets/hf/Qwen3-1.7B}"
DATA_ROOT="${DATA_ROOT:-experiments/scaffold_to_policy/data/modular_sequences_transfer_smoke}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/modular_sequences_transfer_smoke}"
TRAIN_PROBLEMS="${TRAIN_PROBLEMS:-32}"
DEV_PROBLEMS="${DEV_PROBLEMS:-8}"
OOD_PROBLEMS="${OOD_PROBLEMS:-8}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MIN_STEPS="${MIN_STEPS:-3}"
MAX_STEPS="${MAX_STEPS:-5}"
MIN_MODULUS="${MIN_MODULUS:-37}"
MAX_MODULUS="${MAX_MODULUS:-257}"
PROMPT_VARIANT="${PROMPT_VARIANT:-chat}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
NGPU="${NGPU:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-step-${TRAIN_STEPS}}"
MIN_TRAIN_EXAMPLES="${MIN_TRAIN_EXAMPLES:-8}"

mkdir -p \
  "${DATA_ROOT}/train" \
  "${RESULTS_ROOT}/eval/base" \
  "${RESULTS_ROOT}/eval/adapters/raw" \
  "${RESULTS_ROOT}/adapters/raw" \
  "${RESULTS_ROOT}/manifests" \
  "${HF_HOME}"

MANIFEST="${RESULTS_ROOT}/manifests/${RUN_ID}.jsonl"
: > "${MANIFEST}"

record_stage() {
  local stage="$1"
  local status="$2"
  local artifact="$3"
  STAGE="${stage}" STATUS="${status}" ARTIFACT="${artifact}" RUN_ID="${RUN_ID}" MANIFEST="${MANIFEST}" python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

row = {
    "artifact": os.environ["ARTIFACT"],
    "run_id": os.environ["RUN_ID"],
    "rootfs_active": os.environ.get("TORCHTITAN_IN_ROOTFS") == "1",
    "stage": os.environ["STAGE"],
    "status": os.environ["STATUS"],
    "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
manifest = Path(os.environ["MANIFEST"])
manifest.parent.mkdir(parents=True, exist_ok=True)
with manifest.open("a") as f:
    f.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

python -m torchtitan.experiments.scaffold_to_policy.cli generate-modular-sequences \
  --seed 4100 \
  --num-problems "${TRAIN_PROBLEMS}" \
  --min-steps "${MIN_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --min-modulus "${MIN_MODULUS}" \
  --max-modulus "${MAX_MODULUS}" \
  --output "${DATA_ROOT}/train/problems.jsonl"
python -m torchtitan.experiments.scaffold_to_policy.cli generate-modular-sequences \
  --seed 4200 \
  --num-problems "${DEV_PROBLEMS}" \
  --min-steps "${MIN_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --min-modulus "${MIN_MODULUS}" \
  --max-modulus "${MAX_MODULUS}" \
  --output "${DATA_ROOT}/dev.jsonl"
python -m torchtitan.experiments.scaffold_to_policy.cli generate-modular-sequences \
  --seed 4300 \
  --num-problems "${OOD_PROBLEMS}" \
  --min-steps "${MIN_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --min-modulus "${MIN_MODULUS}" \
  --max-modulus "${MAX_MODULUS}" \
  --output "${DATA_ROOT}/ood_test.jsonl"
record_stage generate_splits fresh "${DATA_ROOT}/split_registry.json"

python -m torchtitan.experiments.scaffold_to_policy.cli validate-modular-splits \
  --split \
    "train=${DATA_ROOT}/train/problems.jsonl" \
    "dev=${DATA_ROOT}/dev.jsonl" \
    "ood_test=${DATA_ROOT}/ood_test.jsonl" \
  --output "${DATA_ROOT}/split_registry.json"

python -m torchtitan.experiments.scaffold_to_policy.cli evaluate-modular-vllm \
  --problems "${DATA_ROOT}/train/problems.jsonl" \
  --model "${MODEL}" \
  --output "${RESULTS_ROOT}/eval/base/train_evaluations.jsonl" \
  --summary "${RESULTS_ROOT}/eval/base/train_summary.json" \
  --num-rollouts "${NUM_ROLLOUTS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --prompt-variant "${PROMPT_VARIANT}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}"
record_stage collect_train fresh "${RESULTS_ROOT}/eval/base/train_summary.json"

python -m torchtitan.experiments.scaffold_to_policy.cli build-modular-dataset \
  --evaluations "${RESULTS_ROOT}/eval/base/train_evaluations.jsonl" \
  --output "${DATA_ROOT}/train/modular_sequences_raw.jsonl" \
  --condition raw \
  --min-examples "${MIN_TRAIN_EXAMPLES}"
record_stage build_dataset fresh "${DATA_ROOT}/train/modular_sequences_raw.jsonl"

for split in dev ood_test; do
  python -m torchtitan.experiments.scaffold_to_policy.cli evaluate-modular-vllm \
    --problems "${DATA_ROOT}/${split}.jsonl" \
    --model "${MODEL}" \
    --output "${RESULTS_ROOT}/eval/base/${split}_evaluations.jsonl" \
    --summary "${RESULTS_ROOT}/eval/base/${split}_summary.json" \
    --num-rollouts "${EVAL_ROLLOUTS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --prompt-variant "${PROMPT_VARIANT}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}"
done
record_stage eval_base fresh "${RESULTS_ROOT}/eval/base/dev_summary.json"

export TORCHTITAN_SCAFFOLD_TO_POLICY_DATA_ROOT="${DATA_ROOT}"
export TORCHTITAN_SCAFFOLD_TO_POLICY_RESULTS_ROOT="${RESULTS_ROOT}"
export TORCHTITAN_SCAFFOLD_TO_POLICY_STEPS="${TRAIN_STEPS}"
export TORCHTITAN_SCAFFOLD_TO_POLICY_LORA_RANK="${LORA_RANK}"
export TORCHTITAN_SCAFFOLD_TO_POLICY_LORA_ALPHA="${LORA_ALPHA}"

torchrun --standalone --nproc_per_node="${NGPU}" torchtitan/train.py \
  --module qwen3 \
  --config qwen3_1_7b_modular_sequences_lora_raw
record_stage train_raw fresh "${RESULTS_ROOT}/train/modular_sequences_raw/checkpoint/${CHECKPOINT_STEP}"

python -m torchtitan.experiments.countdown_search_distill.cli export-lora \
  --checkpoint "${RESULTS_ROOT}/train/modular_sequences_raw/checkpoint/${CHECKPOINT_STEP}" \
  --output "${RESULTS_ROOT}/adapters/raw" \
  --base-model-name-or-path "${MODEL}" \
  --rank "${LORA_RANK}" \
  --alpha "${LORA_ALPHA}"
record_stage export_raw fresh "${RESULTS_ROOT}/adapters/raw/export_summary.json"

for split in dev ood_test; do
  python -m torchtitan.experiments.scaffold_to_policy.cli evaluate-modular-vllm \
    --problems "${DATA_ROOT}/${split}.jsonl" \
    --model "${MODEL}" \
    --output "${RESULTS_ROOT}/eval/adapters/raw/${split}_evaluations.jsonl" \
    --summary "${RESULTS_ROOT}/eval/adapters/raw/${split}_summary.json" \
    --num-rollouts "${EVAL_ROLLOUTS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --prompt-variant "${PROMPT_VARIANT}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --lora-adapter "${RESULTS_ROOT}/adapters/raw" \
    --lora-name raw \
    --max-lora-rank "${LORA_RANK}"
done
record_stage eval_adapter_raw fresh "${RESULTS_ROOT}/eval/adapters/raw/dev_summary.json"

python -m torchtitan.experiments.scaffold_to_policy.cli build-modular-report-input \
  --data-root "${DATA_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --split-registry "${DATA_ROOT}/split_registry.json" \
  --summary \
    "dev=${RESULTS_ROOT}/eval/base/dev_summary.json" \
    "ood_test=${RESULTS_ROOT}/eval/base/ood_test_summary.json" \
    "adapter_raw_dev=${RESULTS_ROOT}/eval/adapters/raw/dev_summary.json" \
    "adapter_raw_ood_test=${RESULTS_ROOT}/eval/adapters/raw/ood_test_summary.json" \
  --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json" \
  --no-require-selected
record_stage build_report_input fresh "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"

echo "wrote ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
