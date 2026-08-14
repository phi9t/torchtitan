#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Wave F4 runner: synthetic reasoning transfer smoke driven through the typed
# begin/stage/finish lifecycle and the vllm_1gpu profile doctor. This is the
# second runner migrated off the prototype JSONL manifest and the first on the
# real GPU path (vLLM generation + torchrun LoRA SFT + export + adapter eval).
# It stays a thin compatibility entrypoint with unchanged scientific conditions
# and defaults. See
# experiments/scaffold_to_policy/f4_modular_sequences_transfer_migration_spec.md.

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
ATTEMPT_ID="${ATTEMPT_ID:-attempt-01}"
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

CLI="python -m torchtitan.experiments.scaffold_to_policy.cli"
LIFECYCLE="python -m torchtitan.experiments.execution"
LOCATOR=(--results-root "${RESULTS_ROOT}" --run-id "${RUN_ID}" --attempt-id "${ATTEMPT_ID}")

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" == "1" ]]; then
  ROOTFS_JSON=true
else
  ROOTFS_JSON=false
fi

# One stage wrapper mirroring run_common.sh: it records rootfs provenance as a
# stage-event extra (replacing the legacy JSONL row) and propagates the wrapped
# command's return code. set -euo pipefail aborts the run on a nonzero stage.
run_stage() {
  local stage_id="$1"
  local kind="$2"
  local adapter="$3"
  shift 3
  ${LIFECYCLE} stage "${LOCATOR[@]}" \
    --stage-id "${stage_id}" --name "${stage_id}" \
    --kind "${kind}" --adapter "${adapter}" \
    --stage-extra "rootfs_active=${ROOTFS_JSON}" \
    -- "$@"
}

# vllm_1gpu composes the rootfs-active clause, torch, vllm, and at least one
# visible GPU. --require-ready aborts before any work if the environment is
# unexpectedly blocked; it never fabricates an attempt.
${LIFECYCLE} preflight \
  --profile vllm_1gpu \
  --require-ready \
  --output "${RESULTS_ROOT}/manifests/preflight_${RUN_ID}.json"

# Freeze the declaration. --fields records the scientific knobs so the
# declaration digest changes if any seed, split size, sampling knob, training
# step, or LoRA hyperparameter changes.
FIELDS_FILE="${RESULTS_ROOT}/manifests/fields_${RUN_ID}.json"
RUN_ID="${RUN_ID}" MODEL="${MODEL}" \
  TRAIN_PROBLEMS="${TRAIN_PROBLEMS}" DEV_PROBLEMS="${DEV_PROBLEMS}" OOD_PROBLEMS="${OOD_PROBLEMS}" \
  NUM_ROLLOUTS="${NUM_ROLLOUTS}" EVAL_ROLLOUTS="${EVAL_ROLLOUTS}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  MIN_STEPS="${MIN_STEPS}" MAX_STEPS="${MAX_STEPS}" MIN_MODULUS="${MIN_MODULUS}" MAX_MODULUS="${MAX_MODULUS}" \
  PROMPT_VARIANT="${PROMPT_VARIANT}" TEMPERATURE="${TEMPERATURE}" TOP_P="${TOP_P}" \
  TRAIN_STEPS="${TRAIN_STEPS}" LORA_RANK="${LORA_RANK}" LORA_ALPHA="${LORA_ALPHA}" \
  python - >"${FIELDS_FILE}" <<'PY'
import json
import os

def _int(name):
    return int(os.environ[name])

def _float(name):
    return float(os.environ[name])

print(
    json.dumps(
        {
            "model": os.environ["MODEL"],
            "arm": "raw",
            "splits": {
                "train": {"seed": 4100, "num_problems": _int("TRAIN_PROBLEMS")},
                "dev": {"seed": 4200, "num_problems": _int("DEV_PROBLEMS")},
                "ood_test": {"seed": 4300, "num_problems": _int("OOD_PROBLEMS")},
            },
            "problem": {
                "min_steps": _int("MIN_STEPS"),
                "max_steps": _int("MAX_STEPS"),
                "min_modulus": _int("MIN_MODULUS"),
                "max_modulus": _int("MAX_MODULUS"),
            },
            "sampling": {
                "prompt_variant": os.environ["PROMPT_VARIANT"],
                "temperature": _float("TEMPERATURE"),
                "top_p": _float("TOP_P"),
                "max_new_tokens": _int("MAX_NEW_TOKENS"),
            },
            "rollouts": {
                "collect": _int("NUM_ROLLOUTS"),
                "eval": _int("EVAL_ROLLOUTS"),
            },
            "training": {
                "steps": _int("TRAIN_STEPS"),
                "lora_rank": _int("LORA_RANK"),
                "lora_alpha": _int("LORA_ALPHA"),
            },
            "run_id": os.environ["RUN_ID"],
        },
        sort_keys=True,
    )
)
PY

${LIFECYCLE} begin "${LOCATOR[@]}" \
  --family reasoning \
  --task modular_sequences_transfer \
  --lane gpu_smoke \
  --fields "${FIELDS_FILE}"

# Synthetic problem enumeration is CPU-only, so the split generation stages run
# on the rootfs_cpu adapter. Seeds and counts are unchanged.
run_stage gen-train generate rootfs_cpu \
  ${CLI} generate-modular-sequences \
  --seed 4100 --num-problems "${TRAIN_PROBLEMS}" \
  --min-steps "${MIN_STEPS}" --max-steps "${MAX_STEPS}" \
  --min-modulus "${MIN_MODULUS}" --max-modulus "${MAX_MODULUS}" \
  --output "${DATA_ROOT}/train/problems.jsonl"

run_stage gen-dev generate rootfs_cpu \
  ${CLI} generate-modular-sequences \
  --seed 4200 --num-problems "${DEV_PROBLEMS}" \
  --min-steps "${MIN_STEPS}" --max-steps "${MAX_STEPS}" \
  --min-modulus "${MIN_MODULUS}" --max-modulus "${MAX_MODULUS}" \
  --output "${DATA_ROOT}/dev.jsonl"

run_stage gen-ood_test generate rootfs_cpu \
  ${CLI} generate-modular-sequences \
  --seed 4300 --num-problems "${OOD_PROBLEMS}" \
  --min-steps "${MIN_STEPS}" --max-steps "${MAX_STEPS}" \
  --min-modulus "${MIN_MODULUS}" --max-modulus "${MAX_MODULUS}" \
  --output "${DATA_ROOT}/ood_test.jsonl"

run_stage validate-splits verify rootfs_cpu \
  ${CLI} validate-modular-splits \
  --split \
    "train=${DATA_ROOT}/train/problems.jsonl" \
    "dev=${DATA_ROOT}/dev.jsonl" \
    "ood_test=${DATA_ROOT}/ood_test.jsonl" \
  --output "${DATA_ROOT}/split_registry.json"

# Base rollout collection on train drives vLLM.
run_stage collect-train generate rootfs_vllm \
  ${CLI} evaluate-modular-vllm \
  --problems "${DATA_ROOT}/train/problems.jsonl" \
  --model "${MODEL}" \
  --output "${RESULTS_ROOT}/eval/base/train_evaluations.jsonl" \
  --summary "${RESULTS_ROOT}/eval/base/train_summary.json" \
  --num-rollouts "${NUM_ROLLOUTS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --prompt-variant "${PROMPT_VARIANT}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}"

run_stage build-dataset prepare rootfs_cpu \
  ${CLI} build-modular-dataset \
  --evaluations "${RESULTS_ROOT}/eval/base/train_evaluations.jsonl" \
  --output "${DATA_ROOT}/train/modular_sequences_raw.jsonl" \
  --condition raw \
  --min-examples "${MIN_TRAIN_EXAMPLES}"

for split in dev ood_test; do
  run_stage "base-eval-${split}" evaluate rootfs_vllm \
    ${CLI} evaluate-modular-vllm \
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

export TORCHTITAN_SCAFFOLD_TO_POLICY_DATA_ROOT="${DATA_ROOT}"
export TORCHTITAN_SCAFFOLD_TO_POLICY_RESULTS_ROOT="${RESULTS_ROOT}"
export TORCHTITAN_SCAFFOLD_TO_POLICY_STEPS="${TRAIN_STEPS}"
export TORCHTITAN_SCAFFOLD_TO_POLICY_LORA_RANK="${LORA_RANK}"
export TORCHTITAN_SCAFFOLD_TO_POLICY_LORA_ALPHA="${LORA_ALPHA}"

run_stage train-raw train rootfs_torchrun_sft \
  torchrun --standalone --nproc_per_node="${NGPU}" torchtitan/train.py \
  --module qwen3 \
  --config qwen3_1_7b_modular_sequences_lora_raw

run_stage export-raw export rootfs_cpu \
  python -m torchtitan.experiments.countdown_search_distill.cli export-lora \
  --checkpoint "${RESULTS_ROOT}/train/modular_sequences_raw/checkpoint/${CHECKPOINT_STEP}" \
  --output "${RESULTS_ROOT}/adapters/raw" \
  --base-model-name-or-path "${MODEL}" \
  --rank "${LORA_RANK}" \
  --alpha "${LORA_ALPHA}"

for split in dev ood_test; do
  run_stage "eval-adapter-raw-${split}" evaluate rootfs_vllm \
    ${CLI} evaluate-modular-vllm \
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

REPORT_INPUT="${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
run_stage build-report-input report rootfs_cpu \
  ${CLI} build-modular-report-input \
  --data-root "${DATA_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --split-registry "${DATA_ROOT}/split_registry.json" \
  --summary \
    "dev=${RESULTS_ROOT}/eval/base/dev_summary.json" \
    "ood_test=${RESULTS_ROOT}/eval/base/ood_test_summary.json" \
    "adapter_raw_dev=${RESULTS_ROOT}/eval/adapters/raw/dev_summary.json" \
    "adapter_raw_ood_test=${RESULTS_ROOT}/eval/adapters/raw/ood_test_summary.json" \
  --evaluation \
    "dev=${RESULTS_ROOT}/eval/base/dev_evaluations.jsonl" \
    "ood_test=${RESULTS_ROOT}/eval/base/ood_test_evaluations.jsonl" \
    "adapter_raw_dev=${RESULTS_ROOT}/eval/adapters/raw/dev_evaluations.jsonl" \
    "adapter_raw_ood_test=${RESULTS_ROOT}/eval/adapters/raw/ood_test_evaluations.jsonl" \
  --execution-preflight "${RESULTS_ROOT}/manifests/preflight_${RUN_ID}.json" \
  --output "${REPORT_INPUT}" \
  --no-require-selected

# Every eval condition is a real vLLM measurement on Qwen3-1.7B. This is a
# smoke-sized transfer probe with no calibration/promotion gate, so promotion is
# not_evaluated for all conditions. Built from the condition list so a new
# condition cannot silently drop one.
EVALUATIONS_FILE="${RESULTS_ROOT}/manifests/evaluations_${RUN_ID}.json"
python - >"${EVALUATIONS_FILE}" <<'PY'
import json

conditions = ["base_dev", "base_ood_test", "adapter_raw_dev", "adapter_raw_ood_test"]
print(
    json.dumps(
        {
            key: {
                "execution_outcome": "completed",
                "measurement": "real",
                "promotion": "not_evaluated",
            }
            for key in conditions
        },
        sort_keys=True,
    )
)
PY

${LIFECYCLE} finish "${LOCATOR[@]}" \
  --attempt-outcome completed \
  --report-input "${REPORT_INPUT}" \
  --evaluations "${EVALUATIONS_FILE}"

echo "wrote attempt bundle ${RESULTS_ROOT}/runs/${RUN_ID}/${ATTEMPT_ID}"
