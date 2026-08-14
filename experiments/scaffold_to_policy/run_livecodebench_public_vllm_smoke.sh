#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Wave F4 public coding runner: LiveCodeBench public-dataset vLLM smoke driven
# through the typed begin/stage/finish lifecycle. Preserves the prototype
# runner's graceful blocker behavior for GPU-memory preflight and vLLM runtime
# failures by finishing the attempt as blocked with not_run conditions.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "experiments/scaffold_to_policy/run_livecodebench_public_vllm_smoke.sh" "$@"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export VLLM_USE_FLASHINFER_SAMPLER="${SCAFFOLD_TO_POLICY_VLLM_USE_FLASHINFER_SAMPLER:-0}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-livecodebench-public-vllm-smoke}"
ATTEMPT_ID="${ATTEMPT_ID:-attempt-01}"
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

CLI="python -m torchtitan.experiments.scaffold_to_policy.cli"
LIFECYCLE="python -m torchtitan.experiments.execution"
LOCATOR=(--results-root "${RESULTS_ROOT}" --run-id "${RUN_ID}" --attempt-id "${ATTEMPT_ID}")

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" == "1" ]]; then
  ROOTFS_JSON=true
else
  ROOTFS_JSON=false
fi

subset_args=()
if [[ -n "${DATASET_SUBSET}" ]]; then
  subset_args=(--subset "${DATASET_SUBSET}")
fi

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

write_evaluations() {
  local execution_outcome="$1"
  local measurement="$2"
  local output="$3"
  EXECUTION_OUTCOME="${execution_outcome}" MEASUREMENT="${measurement}" \
    python - >"${output}" <<'PY'
import json
import os

splits = ["dev", "ood_test"]
print(
    json.dumps(
        {
            split: {
                "execution_outcome": os.environ["EXECUTION_OUTCOME"],
                "measurement": os.environ["MEASUREMENT"],
                "promotion": "not_evaluated",
            }
            for split in splits
        },
        sort_keys=True,
    )
)
PY
}

finish_with_status() {
  local attempt_outcome="$1"
  local measurement="$2"
  local report_input="$3"
  local evaluations_file="${RESULTS_ROOT}/manifests/evaluations_${RUN_ID}.json"
  write_evaluations "${attempt_outcome}" "${measurement}" "${evaluations_file}"
  ${LIFECYCLE} finish "${LOCATOR[@]}" \
    --attempt-outcome "${attempt_outcome}" \
    --report-input "${report_input}" \
    --evaluations "${evaluations_file}"
}

write_failure_marker() {
  local stage="$1"
  local failure_type="$2"
  local output="$3"
  STAGE="${stage}" FAILURE_TYPE="${failure_type}" OUTPUT="${output}" RUN_ID="${RUN_ID}" \
    ROOTFS_ACTIVE="${ROOTFS_JSON}" python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

marker = {
    "schema_version": 1,
    "kind": "stage_failure",
    "failure_type": os.environ["FAILURE_TYPE"],
    "stage": os.environ["STAGE"],
    "run_id": os.environ["RUN_ID"],
    "rootfs_active": os.environ["ROOTFS_ACTIVE"] == "true",
    "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
output = Path(os.environ["OUTPUT"])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
PY
}

${LIFECYCLE} preflight \
  --profile vllm_1gpu \
  --profile reasoning \
  --require-ready \
  --output "${RESULTS_ROOT}/manifests/preflight_${RUN_ID}.json"

FIELDS_FILE="${RESULTS_ROOT}/manifests/fields_${RUN_ID}.json"
RUN_ID="${RUN_ID}" MODEL="${MODEL}" DATASET="${DATASET}" DATASET_SUBSET="${DATASET_SUBSET}" \
  DATASET_REVISION="${DATASET_REVISION}" SOURCE_SPLIT="${SOURCE_SPLIT}" \
  DEV_PROBLEMS="${DEV_PROBLEMS}" OOD_PROBLEMS="${OOD_PROBLEMS}" \
  DEV_OFFSET="${DEV_OFFSET}" OOD_OFFSET="${OOD_OFFSET}" \
  NUM_ROLLOUTS="${NUM_ROLLOUTS}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  PROMPT_VARIANT="${PROMPT_VARIANT}" TEMPERATURE="${TEMPERATURE}" TOP_P="${TOP_P}" \
  TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
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
                "gpu_memory_utilization": _float("GPU_MEMORY_UTILIZATION"),
            },
            "rollouts": {"eval": _int("NUM_ROLLOUTS")},
            "code_execution": {"timeout_seconds": _int("TIMEOUT_SECONDS")},
            "run_id": os.environ["RUN_ID"],
        },
        sort_keys=True,
    )
)
PY

${LIFECYCLE} begin "${LOCATOR[@]}" \
  --family coding \
  --task livecodebench \
  --lane public_smoke \
  --fields "${FIELDS_FILE}"

run_stage import-dev acquire rootfs_cpu \
  ${CLI} import-livecodebench-split \
  --dataset "${DATASET}" \
  "${subset_args[@]}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${DATASET_REVISION}" \
  --limit "${DEV_PROBLEMS}" \
  --offset "${DEV_OFFSET}" \
  --output "${DATA_ROOT}/dev.jsonl" \
  --provenance "${DATA_ROOT}/dev_provenance.json"

run_stage import-ood acquire rootfs_cpu \
  ${CLI} import-livecodebench-split \
  --dataset "${DATASET}" \
  "${subset_args[@]}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${DATASET_REVISION}" \
  --limit "${OOD_PROBLEMS}" \
  --offset "${OOD_OFFSET}" \
  --output "${DATA_ROOT}/ood_test.jsonl" \
  --provenance "${DATA_ROOT}/ood_test_provenance.json"

run_stage validate-splits verify rootfs_cpu \
  ${CLI} validate-contest-code-splits \
  --split \
    "dev=${DATA_ROOT}/dev.jsonl" \
    "ood_test=${DATA_ROOT}/ood_test.jsonl" \
  --output "${DATA_ROOT}/split_registry.json"

run_stage preflight-gpu-memory preflight rootfs_vllm \
  ${CLI} preflight-vllm-gpu-memory \
  --output "${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --no-require-selected
if ! run_stage require-gpu-memory-selected verify rootfs_cpu python - "${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1]).read())
raise SystemExit(0 if payload.get("selected") else 1)
PY
then
  REPORT_INPUT="${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
  run_stage write-blocker-report-input report rootfs_cpu \
    ${CLI} write-blocker-report-input \
    --results-root "${RESULTS_ROOT}" \
    --run-id "${RUN_ID}" \
    --task contest_code \
    --lane coding \
    --blocker-type vllm_gpu_memory_preflight \
    --artifact "gpu_memory=${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json" \
    --limitation "LiveCodeBench coding run stopped before model execution because vLLM GPU memory preflight failed." \
    --limitation "No benchmark task execution or model score was produced." \
    --output "${REPORT_INPUT}"
  finish_with_status blocked not_run "${REPORT_INPUT}"
  echo "wrote blocked attempt bundle ${RESULTS_ROOT}/runs/${RUN_ID}/${ATTEMPT_ID}"
  exit 0
fi

if ! run_stage evaluate-splits evaluate rootfs_vllm \
  ${CLI} evaluate-contest-code-vllm-splits \
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
  write_failure_marker evaluate-splits vllm_runtime_failure "${FAILURE_MARKER}"
  REPORT_INPUT="${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
  run_stage write-runtime-blocker-report-input report rootfs_cpu \
    ${CLI} write-blocker-report-input \
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
    --output "${REPORT_INPUT}"
  finish_with_status blocked not_run "${REPORT_INPUT}"
  echo "wrote blocked attempt bundle ${RESULTS_ROOT}/runs/${RUN_ID}/${ATTEMPT_ID}"
  exit 0
fi

REPORT_INPUT="${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
run_stage build-report-input report rootfs_cpu \
  ${CLI} build-contest-code-report-input \
  --data-root "${DATA_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --split-registry "${DATA_ROOT}/split_registry.json" \
  --summary \
    "dev=${RESULTS_ROOT}/eval/dev_summary.json" \
    "ood_test=${RESULTS_ROOT}/eval/ood_test_summary.json" \
  --scaffold-budget "${NUM_ROLLOUTS}" \
  --execution-preflight "${RESULTS_ROOT}/manifests/preflight_${RUN_ID}.json" \
  --output "${REPORT_INPUT}"

finish_with_status completed real "${REPORT_INPUT}"

echo "wrote attempt bundle ${RESULTS_ROOT}/runs/${RUN_ID}/${ATTEMPT_ID}"
