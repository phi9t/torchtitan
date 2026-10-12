#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Wave F4 public structured-reasoning runner: ARC-AGI-2 public repository vLLM
# smoke driven through the typed begin/stage/finish lifecycle. Source,
# prompt-preflight, GPU-memory, and vLLM runtime blockers finish the attempt as
# blocked with not_run split statuses.

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
ATTEMPT_ID="${ATTEMPT_ID:-attempt-01}"
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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN:-4096}}"
PROMPT_VARIANT="${PROMPT_VARIANT:-chat}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.95}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${SCAFFOLD_TO_POLICY_VLLM_GPU_MEMORY_UTILIZATION:-}}"

mkdir -p "${DATA_ROOT}/src" "${RESULTS_ROOT}/eval" "${RESULTS_ROOT}/manifests" "${HF_HOME}"

CLI="python -m torchtitan.experiments.scaffold_to_policy.cli"
LIFECYCLE="python -m torchtitan.experiments.execution"
LOCATOR=(--results-root "${RESULTS_ROOT}" --run-id "${RUN_ID}" --attempt-id "${ATTEMPT_ID}")

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" == "1" ]]; then
  ROOTFS_JSON=true
else
  ROOTFS_JSON=false
fi

gpu_memory_args=()
if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  gpu_memory_args=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
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

finish_blocked() {
  local blocker_type="$1"
  local limitation="$2"
  shift 2
  local report_input="${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
  run_stage "write-${blocker_type}-report-input" report rootfs_cpu \
    ${CLI} write-blocker-report-input \
    --results-root "${RESULTS_ROOT}" \
    --run-id "${RUN_ID}" \
    --task arc_grid \
    --lane reasoning \
    --blocker-type "${blocker_type}" \
    "$@" \
    --limitation "${limitation}" \
    --limitation "No benchmark task execution or model score was produced." \
    --output "${report_input}"
  finish_with_status blocked not_run "${report_input}"
  echo "wrote blocked attempt bundle ${RESULTS_ROOT}/runs/${RUN_ID}/${ATTEMPT_ID}"
}

${LIFECYCLE} preflight \
  --profile vllm_1gpu \
  --profile reasoning \
  --require-ready \
  --output "${RESULTS_ROOT}/manifests/preflight_${RUN_ID}.json"

FIELDS_FILE="${RESULTS_ROOT}/manifests/fields_${RUN_ID}.json"
RUN_ID="${RUN_ID}" MODEL="${MODEL}" ARC_REPO_URL="${ARC_REPO_URL}" \
  ARC_REVISION="${ARC_REVISION}" SOURCE_SPLIT="${SOURCE_SPLIT}" \
  DEV_PROBLEMS="${DEV_PROBLEMS}" OOD_PROBLEMS="${OOD_PROBLEMS}" \
  DEV_OFFSET="${DEV_OFFSET}" OOD_OFFSET="${OOD_OFFSET}" \
  NUM_ROLLOUTS="${NUM_ROLLOUTS}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" PROMPT_VARIANT="${PROMPT_VARIANT}" \
  TEMPERATURE="${TEMPERATURE}" TOP_P="${TOP_P}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" python - >"${FIELDS_FILE}" <<'PY'
import json
import os

def _int(name):
    return int(os.environ[name])

def _float(name):
    value = os.environ[name]
    return None if value == "" else float(value)

print(
    json.dumps(
        {
            "model": os.environ["MODEL"],
            "dataset": {
                "repo_url": os.environ["ARC_REPO_URL"],
                "revision": os.environ["ARC_REVISION"],
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
                "max_model_len": _int("MAX_MODEL_LEN"),
                "gpu_memory_utilization": _float("GPU_MEMORY_UTILIZATION"),
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
  --task arc_agi2 \
  --lane public_smoke \
  --fields "${FIELDS_FILE}"

ARC_SRC="${DATA_ROOT}/src/ARC-AGI-2"
if [[ ! -d "${ARC_SRC}/.git" ]]; then
  rm -rf "${ARC_SRC}"
  if ! run_stage clone-arc-repo acquire rootfs_cpu git clone "${ARC_REPO_URL}" "${ARC_SRC}"; then
    FAILURE_MARKER="${RESULTS_ROOT}/eval/arc_source_setup_failure.json"
    write_failure_marker clone-arc-repo arc_source_setup_failure "${FAILURE_MARKER}"
    finish_blocked arc_source_setup \
      "ARC-AGI-2 exact-grid run stopped before import because source repository clone failed." \
      --artifact "stage_failure=${FAILURE_MARKER}"
    exit 0
  fi
fi
if ! run_stage fetch-arc-revision acquire rootfs_cpu git -C "${ARC_SRC}" fetch --depth 1 origin "${ARC_REVISION}"; then
  FAILURE_MARKER="${RESULTS_ROOT}/eval/arc_source_setup_failure.json"
  write_failure_marker fetch-arc-revision arc_source_setup_failure "${FAILURE_MARKER}"
  finish_blocked arc_source_setup \
    "ARC-AGI-2 exact-grid run stopped before import because source repository fetch failed." \
    --artifact "stage_failure=${FAILURE_MARKER}"
  exit 0
fi
if ! run_stage checkout-arc-revision acquire rootfs_cpu git -C "${ARC_SRC}" checkout --detach "${ARC_REVISION}"; then
  FAILURE_MARKER="${RESULTS_ROOT}/eval/arc_source_setup_failure.json"
  write_failure_marker checkout-arc-revision arc_source_setup_failure "${FAILURE_MARKER}"
  finish_blocked arc_source_setup \
    "ARC-AGI-2 exact-grid run stopped before import because source repository checkout failed." \
    --artifact "stage_failure=${FAILURE_MARKER}"
  exit 0
fi

run_stage import-dev acquire rootfs_cpu ${CLI} import-arc-grid-split \
  --task-dir "${ARC_SRC}/data" \
  --repo-url "${ARC_REPO_URL}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${ARC_REVISION}" \
  --limit "${DEV_PROBLEMS}" \
  --offset "${DEV_OFFSET}" \
  --output "${DATA_ROOT}/dev.jsonl" \
  --provenance "${DATA_ROOT}/dev_provenance.json"

run_stage import-ood acquire rootfs_cpu ${CLI} import-arc-grid-split \
  --task-dir "${ARC_SRC}/data" \
  --repo-url "${ARC_REPO_URL}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${ARC_REVISION}" \
  --limit "${OOD_PROBLEMS}" \
  --offset "${OOD_OFFSET}" \
  --output "${DATA_ROOT}/ood_test.jsonl" \
  --provenance "${DATA_ROOT}/ood_test_provenance.json"

run_stage validate-splits verify rootfs_cpu ${CLI} validate-arc-grid-splits \
  --split \
    "dev=${DATA_ROOT}/dev.jsonl" \
    "ood_test=${DATA_ROOT}/ood_test.jsonl" \
  --output "${DATA_ROOT}/split_registry.json"

for split in dev ood_test; do
  if ! run_stage "preflight-${split}-prompts" preflight rootfs_cpu ${CLI} preflight-arc-grid-prompts \
    --problems "${DATA_ROOT}/${split}.jsonl" \
    --model "${MODEL}" \
    --output "${RESULTS_ROOT}/eval/${split}_prompt_preflight.json" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --prompt-variant "${PROMPT_VARIANT}"; then
    FAILURE_MARKER="${RESULTS_ROOT}/eval/${split}_prompt_preflight_failure.json"
    write_failure_marker "preflight-${split}-prompts" arc_prompt_preflight_failure "${FAILURE_MARKER}"
    finish_blocked arc_prompt_preflight \
      "ARC-AGI-2 exact-grid run stopped before model execution because ${split} prompt preflight failed." \
      --artifact \
        "prompt=${RESULTS_ROOT}/eval/${split}_prompt_preflight.json" \
        "stage_failure=${FAILURE_MARKER}"
    exit 0
  fi
done

memory_preflight_args=()
if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  memory_preflight_args=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi
run_stage preflight-gpu-memory preflight rootfs_vllm ${CLI} preflight-vllm-gpu-memory \
  --output "${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json" \
  "${memory_preflight_args[@]}" \
  --no-require-selected
if ! run_stage require-gpu-memory-selected verify rootfs_cpu python - "${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1]).read())
raise SystemExit(0 if payload.get("selected") else 1)
PY
then
  finish_blocked vllm_gpu_memory_preflight \
    "ARC-AGI-2 exact-grid run stopped before model execution because vLLM GPU memory preflight failed." \
    --artifact \
      "dev_prompt=${RESULTS_ROOT}/eval/dev_prompt_preflight.json" \
      "ood_test_prompt=${RESULTS_ROOT}/eval/ood_test_prompt_preflight.json" \
      "gpu_memory=${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json"
  exit 0
fi

for split in dev ood_test; do
  if ! run_stage "base-eval-${split}" evaluate rootfs_vllm ${CLI} evaluate-arc-grid-vllm \
    --problems "${DATA_ROOT}/${split}.jsonl" \
    --model "${MODEL}" \
    --output "${RESULTS_ROOT}/eval/${split}_evaluations.jsonl" \
    --summary "${RESULTS_ROOT}/eval/${split}_summary.json" \
    --num-rollouts "${NUM_ROLLOUTS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --prompt-variant "${PROMPT_VARIANT}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    "${gpu_memory_args[@]}"; then
    FAILURE_MARKER="${RESULTS_ROOT}/eval/${split}_vllm_runtime_failure.json"
    write_failure_marker "base-eval-${split}" vllm_runtime_failure "${FAILURE_MARKER}"
    finish_blocked vllm_runtime_failure \
      "ARC-AGI-2 exact-grid run stopped during ${split} model execution because vLLM failed at runtime." \
      --artifact \
        "dev_prompt=${RESULTS_ROOT}/eval/dev_prompt_preflight.json" \
        "ood_test_prompt=${RESULTS_ROOT}/eval/ood_test_prompt_preflight.json" \
        "gpu_memory=${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json" \
        "stage_failure=${FAILURE_MARKER}"
    exit 0
  fi
done

REPORT_INPUT="${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
run_stage build-report-input report rootfs_cpu ${CLI} build-arc-grid-report-input \
  --data-root "${DATA_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --split-registry "${DATA_ROOT}/split_registry.json" \
  --summary \
    "dev=${RESULTS_ROOT}/eval/dev_summary.json" \
    "ood_test=${RESULTS_ROOT}/eval/ood_test_summary.json" \
  --preflight \
    "dev=${RESULTS_ROOT}/eval/dev_prompt_preflight.json" \
    "ood_test=${RESULTS_ROOT}/eval/ood_test_prompt_preflight.json" \
  --scaffold-budget "${NUM_ROLLOUTS}" \
  --execution-preflight "${RESULTS_ROOT}/manifests/preflight_${RUN_ID}.json" \
  --output "${REPORT_INPUT}"

finish_with_status completed real "${REPORT_INPUT}"

echo "wrote attempt bundle ${RESULTS_ROOT}/runs/${RUN_ID}/${ATTEMPT_ID}"
