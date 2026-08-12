#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"

scaffold_enter_rootfs_if_needed "run_arc_agi2_public_vllm_smoke.sh" "$@"
scaffold_setup_env

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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN:-4096}}"
PROMPT_VARIANT="${PROMPT_VARIANT:-chat}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.95}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${SCAFFOLD_TO_POLICY_VLLM_GPU_MEMORY_UTILIZATION:-}}"
gpu_memory_args=()
if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  gpu_memory_args=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi

mkdir -p "${DATA_ROOT}/src" "${RESULTS_ROOT}/eval" "${RESULTS_ROOT}/manifests" "${HF_HOME}"
scaffold_setup_run_manifest

ARC_SRC="${DATA_ROOT}/src/ARC-AGI-2"
if [[ ! -d "${ARC_SRC}/.git" ]]; then
  rm -rf "${ARC_SRC}"
  scaffold_run_stage clone_arc_repo git clone "${ARC_REPO_URL}" "${ARC_SRC}"
fi
scaffold_run_stage fetch_arc_revision git -C "${ARC_SRC}" fetch --depth 1 origin "${ARC_REVISION}"
scaffold_run_stage checkout_arc_revision git -C "${ARC_SRC}" checkout --detach "${ARC_REVISION}"

scaffold_run_stage import_dev python -m torchtitan.experiments.scaffold_to_policy.cli import-arc-grid-split \
  --task-dir "${ARC_SRC}/data" \
  --repo-url "${ARC_REPO_URL}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${ARC_REVISION}" \
  --limit "${DEV_PROBLEMS}" \
  --offset "${DEV_OFFSET}" \
  --output "${DATA_ROOT}/dev.jsonl" \
  --provenance "${DATA_ROOT}/dev_provenance.json"

scaffold_run_stage import_ood_test python -m torchtitan.experiments.scaffold_to_policy.cli import-arc-grid-split \
  --task-dir "${ARC_SRC}/data" \
  --repo-url "${ARC_REPO_URL}" \
  --source-split "${SOURCE_SPLIT}" \
  --revision "${ARC_REVISION}" \
  --limit "${OOD_PROBLEMS}" \
  --offset "${OOD_OFFSET}" \
  --output "${DATA_ROOT}/ood_test.jsonl" \
  --provenance "${DATA_ROOT}/ood_test_provenance.json"

scaffold_run_stage validate_splits python -m torchtitan.experiments.scaffold_to_policy.cli validate-arc-grid-splits \
  --split \
    "dev=${DATA_ROOT}/dev.jsonl" \
    "ood_test=${DATA_ROOT}/ood_test.jsonl" \
  --output "${DATA_ROOT}/split_registry.json"

for split in dev ood_test; do
  scaffold_run_stage "preflight_${split}_prompts" python -m torchtitan.experiments.scaffold_to_policy.cli preflight-arc-grid-prompts \
    --problems "${DATA_ROOT}/${split}.jsonl" \
    --model "${MODEL}" \
    --output "${RESULTS_ROOT}/eval/${split}_prompt_preflight.json" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --prompt-variant "${PROMPT_VARIANT}"
done

memory_preflight_args=()
if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  memory_preflight_args=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi
scaffold_run_stage preflight_gpu_memory python -m torchtitan.experiments.scaffold_to_policy.cli preflight-vllm-gpu-memory \
  --output "${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json" \
  "${memory_preflight_args[@]}" \
  --no-require-selected
if ! scaffold_run_stage require_gpu_memory_selected python - <<'PY' "${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json"
import json
import sys

payload = json.loads(open(sys.argv[1]).read())
raise SystemExit(0 if payload.get("selected") else 1)
PY
then
  scaffold_run_stage write_blocker_report_input python -m torchtitan.experiments.scaffold_to_policy.cli write-blocker-report-input \
    --results-root "${RESULTS_ROOT}" \
    --run-id "${RUN_ID}" \
    --task arc_grid \
    --lane reasoning \
    --blocker-type vllm_gpu_memory_preflight \
    --artifact \
      "dev_prompt=${RESULTS_ROOT}/eval/dev_prompt_preflight.json" \
      "ood_test_prompt=${RESULTS_ROOT}/eval/ood_test_prompt_preflight.json" \
      "gpu_memory=${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json" \
    --limitation "ARC-AGI-2 exact-grid run stopped before model execution because vLLM GPU memory preflight failed." \
    --limitation "Prompt preflights may be present, but no model score was produced." \
    --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
  echo "wrote ARC-AGI-2 blocker ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
  exit 0
fi

for split in dev ood_test; do
  if ! scaffold_run_stage "evaluate_${split}" python -m torchtitan.experiments.scaffold_to_policy.cli evaluate-arc-grid-vllm \
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
    scaffold_write_stage_failure_marker "evaluate_${split}" vllm_runtime_failure "${FAILURE_MARKER}"
    scaffold_run_stage write_runtime_blocker_report_input python -m torchtitan.experiments.scaffold_to_policy.cli write-blocker-report-input \
      --results-root "${RESULTS_ROOT}" \
      --run-id "${RUN_ID}" \
      --task arc_grid \
      --lane reasoning \
      --blocker-type vllm_runtime_failure \
      --artifact \
        "dev_prompt=${RESULTS_ROOT}/eval/dev_prompt_preflight.json" \
        "ood_test_prompt=${RESULTS_ROOT}/eval/ood_test_prompt_preflight.json" \
        "gpu_memory=${RESULTS_ROOT}/eval/vllm_gpu_memory_preflight.json" \
        "stage_failure=${FAILURE_MARKER}" \
      --limitation "ARC-AGI-2 exact-grid run stopped during ${split} model execution because vLLM failed at runtime." \
      --limitation "Prompt preflights may be present, but no complete model score was produced." \
      --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
    echo "wrote ARC-AGI-2 runtime blocker ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
    exit 0
  fi
done

scaffold_run_stage build_report_input python -m torchtitan.experiments.scaffold_to_policy.cli build-arc-grid-report-input \
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
  --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"

echo "wrote ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
