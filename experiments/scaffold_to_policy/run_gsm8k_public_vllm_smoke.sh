#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Wave F4 public reasoning reference: GSM8K public-dataset vLLM smoke driven
# through the typed begin/stage/finish lifecycle and a composed
# vllm_1gpu+reasoning profile doctor. Thin compatibility entrypoint with
# unchanged dataset revision, offsets, sampling, and rollout budget. See
# experiments/scaffold_to_policy/f4_public_vllm_runner_migration_spec.md.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "experiments/scaffold_to_policy/run_gsm8k_public_vllm_smoke.sh" "$@"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export VLLM_USE_FLASHINFER_SAMPLER="${SCAFFOLD_TO_POLICY_VLLM_USE_FLASHINFER_SAMPLER:-0}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-gsm8k-public-vllm-smoke}"
ATTEMPT_ID="${ATTEMPT_ID:-attempt-01}"
MODEL="${MODEL:-./assets/hf/Qwen3-1.7B}"
DATASET="${DATASET:-openai/gsm8k}"
DATASET_SUBSET="${DATASET_SUBSET:-main}"
DATASET_REVISION="${DATASET_REVISION:-740312add88f781978c0658806c59bc2815b9866}"
SOURCE_SPLIT="${SOURCE_SPLIT:-test}"
DATA_ROOT="${DATA_ROOT:-experiments/scaffold_to_policy/data/gsm8k_public_vllm_smoke}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/gsm8k_public_vllm_smoke}"
DEV_PROBLEMS="${DEV_PROBLEMS:-8}"
OOD_PROBLEMS="${OOD_PROBLEMS:-8}"
DEV_OFFSET="${DEV_OFFSET:-0}"
OOD_OFFSET="${OOD_OFFSET:-256}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
PROMPT_VARIANT="${PROMPT_VARIANT:-chat}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"

mkdir -p "${DATA_ROOT}" "${RESULTS_ROOT}/eval" "${RESULTS_ROOT}/manifests" "${HF_HOME}"

CLI="python -m torchtitan.experiments.scaffold_to_policy.cli"
LIFECYCLE="python -m torchtitan.experiments.execution"
LOCATOR=(--results-root "${RESULTS_ROOT}" --run-id "${RUN_ID}" --attempt-id "${ATTEMPT_ID}")

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" == "1" ]]; then
  ROOTFS_JSON=true
else
  ROOTFS_JSON=false
fi

# One stage wrapper: records rootfs provenance as a stage-event extra and
# propagates the wrapped command's return code. set -euo pipefail aborts the run
# on a nonzero stage before finish.
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

# Compose the dataset-import prerequisites (reasoning) with the vLLM eval
# prerequisites (vllm_1gpu). A missing package in either blocks before any work.
${LIFECYCLE} preflight \
  --profile vllm_1gpu \
  --profile reasoning \
  --require-ready \
  --output "${RESULTS_ROOT}/manifests/preflight_${RUN_ID}.json"

# Freeze the declaration. --fields records the pinned dataset identity and the
# scientific knobs so the declaration digest changes if any of them change.
FIELDS_FILE="${RESULTS_ROOT}/manifests/fields_${RUN_ID}.json"
RUN_ID="${RUN_ID}" MODEL="${MODEL}" DATASET="${DATASET}" DATASET_SUBSET="${DATASET_SUBSET}" \
  DATASET_REVISION="${DATASET_REVISION}" SOURCE_SPLIT="${SOURCE_SPLIT}" \
  DEV_PROBLEMS="${DEV_PROBLEMS}" OOD_PROBLEMS="${OOD_PROBLEMS}" \
  DEV_OFFSET="${DEV_OFFSET}" OOD_OFFSET="${OOD_OFFSET}" \
  NUM_ROLLOUTS="${NUM_ROLLOUTS}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  PROMPT_VARIANT="${PROMPT_VARIANT}" TEMPERATURE="${TEMPERATURE}" TOP_P="${TOP_P}" \
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
            "dataset": {
                "name": os.environ["DATASET"],
                "subset": os.environ["DATASET_SUBSET"] or None,
                "revision": os.environ["DATASET_REVISION"],
                "source_split": os.environ["SOURCE_SPLIT"],
            },
            "splits": {
                "dev": {"limit": _int("DEV_PROBLEMS"), "offset": _int("DEV_OFFSET")},
                "ood_test": {"limit": _int("OOD_PROBLEMS"), "offset": _int("OOD_OFFSET")},
            },
            "sampling": {
                "prompt_variant": os.environ["PROMPT_VARIANT"],
                "temperature": _float("TEMPERATURE"),
                "top_p": _float("TOP_P"),
                "max_new_tokens": _int("MAX_NEW_TOKENS"),
            },
            "rollouts": {"eval": _int("NUM_ROLLOUTS")},
            "run_id": os.environ["RUN_ID"],
        },
        sort_keys=True,
    )
)
PY

${LIFECYCLE} begin "${LOCATOR[@]}" \
  --family reasoning \
  --task gsm8k \
  --lane public_smoke \
  --fields "${FIELDS_FILE}"

run_stage import-dev acquire rootfs_cpu \
  ${CLI} import-gsm8k-split \
  --dataset "${DATASET}" \
  --subset "${DATASET_SUBSET}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${DATASET_REVISION}" \
  --limit "${DEV_PROBLEMS}" \
  --offset "${DEV_OFFSET}" \
  --output "${DATA_ROOT}/dev.jsonl" \
  --provenance "${DATA_ROOT}/dev_provenance.json"

run_stage import-ood acquire rootfs_cpu \
  ${CLI} import-gsm8k-split \
  --dataset "${DATASET}" \
  --subset "${DATASET_SUBSET}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${DATASET_REVISION}" \
  --limit "${OOD_PROBLEMS}" \
  --offset "${OOD_OFFSET}" \
  --output "${DATA_ROOT}/ood_test.jsonl" \
  --provenance "${DATA_ROOT}/ood_test_provenance.json"

run_stage validate-splits verify rootfs_cpu \
  ${CLI} validate-gsm-style-splits \
  --split \
    "dev=${DATA_ROOT}/dev.jsonl" \
    "ood_test=${DATA_ROOT}/ood_test.jsonl" \
  --output "${DATA_ROOT}/split_registry.json"

for split in dev ood_test; do
  run_stage "base-eval-${split}" evaluate rootfs_vllm \
    ${CLI} evaluate-gsm-style-vllm \
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

REPORT_INPUT="${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
run_stage build-report-input report rootfs_cpu \
  ${CLI} build-gsm-style-report-input \
  --data-root "${DATA_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --split-registry "${DATA_ROOT}/split_registry.json" \
  --summary \
    "dev=${RESULTS_ROOT}/eval/dev_summary.json" \
    "ood_test=${RESULTS_ROOT}/eval/ood_test_summary.json" \
  --scaffold-budget "${NUM_ROLLOUTS}" \
  --output "${REPORT_INPUT}"

# Real vLLM base-model evaluation on both splits; no trained arm and no
# calibration gate, so promotion is not_evaluated. Built from the split list so
# a new split cannot silently drop a condition.
EVALUATIONS_FILE="${RESULTS_ROOT}/manifests/evaluations_${RUN_ID}.json"
python - >"${EVALUATIONS_FILE}" <<'PY'
import json

splits = ["dev", "ood_test"]
print(
    json.dumps(
        {
            key: {
                "execution_outcome": "completed",
                "measurement": "real",
                "promotion": "not_evaluated",
            }
            for key in splits
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
