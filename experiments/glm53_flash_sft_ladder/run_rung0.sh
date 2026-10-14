#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
rootfs_repo_root="/workspace/torchtitan"
rootfs_script_dir="${rootfs_repo_root}/experiments/glm53_flash_sft_ladder"

RUN_ID="${RUN_ID:-rung0-$(date -u +%Y%m%dT%H%M%SZ)}"
ATTEMPT_ID="${ATTEMPT_ID:-attempt-01}"
HOST_CACHE_ROOT="${GLM_LADDER_CACHE:-/data05/${USER:-user}/glm53-ladder-cache}"
ROOTFS_CACHE_ROOT="${ROOTFS_CACHE_ROOT:-${rootfs_script_dir}/cache/rung0-inputs}"
RESULT_ROOT="${RESULT_ROOT:-${script_dir}/results/${RUN_ID}/${ATTEMPT_ID}}"
ROOTFS_RESULT_ROOT="${ROOTFS_RESULT_ROOT:-${rootfs_script_dir}/results/${RUN_ID}/${ATTEMPT_ID}}"
DATASET_URI="${DATASET_URI:-hdfs://harunava/user/tiger/waystone/torchtitan/datasets/countdown-sft/v1}"
QWEN3_0_6B_URI="${QWEN3_0_6B_URI:-hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca}"
QWEN3_1_7B_URI="${QWEN3_1_7B_URI:-hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"

HOST_DATA_ROOT="${HOST_CACHE_ROOT}/datasets/countdown-sft/v1"
HOST_QWEN3_0_6B_ROOT="${HOST_CACHE_ROOT}/model-weights/Qwen/Qwen3-0.6B"
HOST_QWEN3_1_7B_ROOT="${HOST_CACHE_ROOT}/model-weights/Qwen/Qwen3-1.7B"
DATA_ROOT="${ROOTFS_CACHE_ROOT}/datasets/countdown-sft/v1"
QWEN3_0_6B_ROOT="${ROOTFS_CACHE_ROOT}/model-weights/Qwen/Qwen3-0.6B"
QWEN3_1_7B_ROOT="${ROOTFS_CACHE_ROOT}/model-weights/Qwen/Qwen3-1.7B"
HOST_MIRROR_ROOT="${script_dir}/cache/rung0-inputs"
HOST_DATA_ROOTFS_MIRROR="${HOST_MIRROR_ROOT}/datasets/countdown-sft/v1"
HOST_QWEN3_0_6B_ROOTFS_MIRROR="${HOST_MIRROR_ROOT}/model-weights/Qwen/Qwen3-0.6B"
HOST_QWEN3_1_7B_ROOTFS_MIRROR="${HOST_MIRROR_ROOT}/model-weights/Qwen/Qwen3-1.7B"
TRAIN_DATA_FILE="${DATA_ROOT}/train/formatting.jsonl"
DEV_PROBLEMS="${DATA_ROOT}/dev/problems.jsonl"
RUNG_LOG_DIR="${RESULT_ROOT}/logs"
mkdir -p "${RESULT_ROOT}" "${RUNG_LOG_DIR}"

usage() {
  cat >&2 <<'USAGE'
Usage: run_rung0.sh [all|stage|dry-run|parity|train|export|eval|publish]

Host-side orchestrator for T02 rung 0. Python, torchrun, and vLLM stages
re-enter the rootfs. Waystone stage/publish stays on the host.
USAGE
}

stage_inputs() {
  "${script_dir}/stage.sh" "${DATASET_URI}" "${HOST_DATA_ROOT}" | tee "${RUNG_LOG_DIR}/stage_dataset.log"
  "${script_dir}/stage.sh" "${QWEN3_0_6B_URI}" "${HOST_QWEN3_0_6B_ROOT}" | tee "${RUNG_LOG_DIR}/stage_qwen3_0_6b.log"
  "${script_dir}/stage.sh" "${QWEN3_1_7B_URI}" "${HOST_QWEN3_1_7B_ROOT}" | tee "${RUNG_LOG_DIR}/stage_qwen3_1_7b.log"
  mirror_inputs
  verify_hdfs_inputs
}

mirror_one() {
  local source="$1" target="$2"
  if [[ ! -d "${source}" ]]; then
    printf 'missing staged input directory: %s\n' "${source}" >&2
    exit 1
  fi
  mkdir -p "${target}"
  rsync -a --delete "${source}/" "${target}/"
}

mirror_inputs() {
  mirror_one "${HOST_DATA_ROOT}" "${HOST_DATA_ROOTFS_MIRROR}"
  mirror_one "${HOST_QWEN3_0_6B_ROOT}" "${HOST_QWEN3_0_6B_ROOTFS_MIRROR}"
  mirror_one "${HOST_QWEN3_1_7B_ROOT}" "${HOST_QWEN3_1_7B_ROOTFS_MIRROR}"
}

verify_hdfs_inputs() {
  "${script_dir}/lib/hdfs_strict.sh" --quick \
    "${HOST_DATA_ROOTFS_MIRROR}" \
    "${HOST_QWEN3_0_6B_ROOTFS_MIRROR}" \
    "${HOST_QWEN3_1_7B_ROOTFS_MIRROR}"
}

ensure_inputs_visible() {
  if [[ ! -f "${HOST_DATA_ROOTFS_MIRROR}/train/formatting.jsonl" \
    || ! -f "${HOST_DATA_ROOTFS_MIRROR}/MANIFEST.jsonl" \
    || ! -f "${HOST_DATA_ROOTFS_MIRROR}/.staged.json" \
    || ! -d "${HOST_QWEN3_0_6B_ROOTFS_MIRROR}" \
    || ! -f "${HOST_QWEN3_0_6B_ROOTFS_MIRROR}/MANIFEST.jsonl" \
    || ! -f "${HOST_QWEN3_0_6B_ROOTFS_MIRROR}/.staged.json" \
    || ! -d "${HOST_QWEN3_1_7B_ROOTFS_MIRROR}" \
    || ! -f "${HOST_QWEN3_1_7B_ROOTFS_MIRROR}/MANIFEST.jsonl" \
    || ! -f "${HOST_QWEN3_1_7B_ROOTFS_MIRROR}/.staged.json" ]]; then
    mirror_inputs
  fi
  verify_hdfs_inputs
}

rootfs_train() {
  TORCHTITAN_ROOTFS_ENV=glm53_train "${repo_root}/scripts/rootfs/enter_rootfs.sh" -- bash -lc \
    "unset CC CXX; cd /workspace/torchtitan && $*"
}

rootfs_default() {
  "${repo_root}/scripts/rootfs/enter_rootfs.sh" -- bash -lc \
    "unset CC CXX; cd /workspace/torchtitan && $*"
}

train_one() {
  local name="$1" config="$2" seed="$3" ngpu="$4" devices="$5" steps="${6:-94}" resume_step="${7:-}" stop_after_step="${8:-}"
  local out="${RESULT_ROOT}/train/${name}"
  local rootfs_out="${ROOTFS_RESULT_ROOT}/train/${name}"
  mkdir -p "${out}"
  local resume_args=()
  if [[ -n "${resume_step}" ]]; then
    resume_args=(--resume-step "${resume_step}")
  fi
  local train_module="torchtitan.train"
  local stop_args=()
  if [[ -n "${stop_after_step}" ]]; then
    train_module="torchtitan.experiments.glm53_flash_sft_ladder.train_until"
    stop_args=(--stop-after-step "${stop_after_step}")
  fi
  GLM53_LADDER_DATA_FILE="${TRAIN_DATA_FILE}" \
  GLM53_LADDER_DUMP_ROOT="${ROOTFS_RESULT_ROOT}/train" \
  GLM53_LADDER_DUMP_FOLDER="${rootfs_out}" \
  GLM53_LADDER_HF_ASSETS_PATH_0_6B="${QWEN3_0_6B_ROOT}" \
  GLM53_LADDER_HF_ASSETS_PATH_1_7B="${QWEN3_1_7B_ROOT}" \
  GLM53_LADDER_SEED="${seed}" \
  GLM53_LADDER_DETERMINISTIC="${GLM53_LADDER_DETERMINISTIC:-0}" \
  GLM53_LADDER_STEPS="${steps}" \
  GLM53_LADDER_CHECKPOINT_INTERVAL=31 \
  CUDA_VISIBLE_DEVICES="${devices}" \
  TORCHTITAN_ROOTFS_ENV=glm53_train \
  "${repo_root}/scripts/rootfs/enter_rootfs.sh" -- bash -lc \
    "unset CC CXX; cd /workspace/torchtitan && torchrun --nproc_per_node=${ngpu} --rdzv_id '${RUN_ID}-${name}' --rdzv_backend c10d --rdzv_endpoint='${TORCHTITAN_RDZV_ENDPOINT:-127.0.0.1:0}' --local_addr='${TORCHTITAN_LOCAL_ADDR:-127.0.0.1}' --local-ranks-filter 0 --role rank --tee 3 -m ${train_module} ${stop_args[*]} --module torchtitan.experiments.glm53_flash_sft_ladder.config_registry --config ${config} --output-dir ${rootfs_out} ${resume_args[*]}" \
    2>&1 | tee "${RUNG_LOG_DIR}/train_${name}.log"
}

dry_run_configs() {
  ensure_inputs_visible
  train_one dry_qwen3_0_6b_lora qwen3_0_6b_countdown_lora 42 1 "${DRY_RUN_DEVICES:-0}" "${DRY_RUN_STEPS:-2}"
  train_one dry_qwen3_1_7b_lora qwen3_1_7b_countdown_lora 42 1 "${DRY_RUN_DEVICES:-0}" "${DRY_RUN_STEPS:-2}"
  train_one dry_qwen3_0_6b_full qwen3_0_6b_countdown_full 42 1 "${DRY_RUN_DEVICES:-0}" "${DRY_RUN_STEPS:-2}"
}

run_parity() {
  ensure_inputs_visible
  rootfs_train "python -m torchtitan.experiments.glm53_flash_sft_ladder.parity --model-flavor 0.6B --hf-assets-path '${QWEN3_0_6B_ROOT}' --output '${ROOTFS_RESULT_ROOT}/parity/qwen3_0_6b.json'" \
    2>&1 | tee "${RUNG_LOG_DIR}/parity_qwen3_0_6b.log"
}

train_gate_runs() {
  ensure_inputs_visible
  train_one qwen3_0_6b_lora_seed42_ngpu2 qwen3_0_6b_countdown_lora 42 2 "${GPU_PAIR_0:-0,1}" &
  train_one qwen3_0_6b_full_seed42_ngpu2 qwen3_0_6b_countdown_full 42 2 "${GPU_PAIR_1:-2,3}" &
  train_one qwen3_0_6b_lora_seed43_ngpu2 qwen3_0_6b_countdown_lora 43 2 "${GPU_PAIR_2:-4,5}" &
  train_one qwen3_1_7b_lora_seed42_ngpu2 qwen3_1_7b_countdown_lora 42 2 "${GPU_PAIR_3:-6,7}" &
  wait
  train_one qwen3_1_7b_lora_seed43_ngpu2 qwen3_1_7b_countdown_lora 43 2 "${GPU_PAIR_0:-0,1}"
  train_one qwen3_0_6b_lora_seed42_ngpu4 qwen3_0_6b_countdown_lora 42 4 "${GPU_QUAD:-0,1,2,3}"
  train_one qwen3_0_6b_lora_seed42_ngpu8 qwen3_0_6b_countdown_lora 42 8 "${GPU_ALL:-0,1,2,3,4,5,6,7}"
}

resume_gate() {
  ensure_inputs_visible
  GLM53_LADDER_DETERMINISTIC=1 train_one resume_full_a qwen3_0_6b_countdown_lora 42 2 "${GPU_PAIR_0:-0,1}"
  GLM53_LADDER_DETERMINISTIC=1 train_one resume_stop62 qwen3_0_6b_countdown_lora 42 2 "${GPU_PAIR_1:-2,3}" 94 "" 62
  "${script_dir}/publish.sh" "${RESULT_ROOT}/train/resume_stop62/checkpoint/step-62" \
    "torchtitan/checkpoints/glm53-ladder/${RUN_ID}/${ATTEMPT_ID}/resume_stop62_step62" \
    | tee "${RUNG_LOG_DIR}/publish_resume_step62.log"
  rm -rf "${RESULT_ROOT}/train/resume_stop62/checkpoint/step-62"
  "${script_dir}/stage.sh" \
    "hdfs://harunava/user/tiger/waystone/torchtitan/checkpoints/glm53-ladder/${RUN_ID}/${ATTEMPT_ID}/resume_stop62_step62" \
    "${RESULT_ROOT}/train/resume_stop62/checkpoint/step-62" \
    | tee "${RUNG_LOG_DIR}/stage_resume_step62.log"
  "${script_dir}/lib/hdfs_strict.sh" --quick "${RESULT_ROOT}/train/resume_stop62/checkpoint/step-62"
  if [[ -d "${RESULT_ROOT}/train/resume_stop62/run_evidence" ]]; then
    rm -rf "${RESULT_ROOT}/train/resume_stop62/run_evidence_stop_after_62"
    mv "${RESULT_ROOT}/train/resume_stop62/run_evidence" \
      "${RESULT_ROOT}/train/resume_stop62/run_evidence_stop_after_62"
  fi
  GLM53_LADDER_DETERMINISTIC=1 train_one resume_stop62 qwen3_0_6b_countdown_lora 42 2 "${GPU_PAIR_1:-2,3}" 94 62
}

export_one() {
  local train_name="$1" model_root="$2" adapter_name="$3"
  local checkpoint="${ROOTFS_RESULT_ROOT}/train/${train_name}/checkpoint/step-94"
  local output="${ROOTFS_RESULT_ROOT}/adapters/${adapter_name}"
  rootfs_train "python -m torchtitan.experiments.countdown_search_distill.cli export-lora --checkpoint '${checkpoint}' --output '${output}' --base-model-name-or-path '${model_root}' --rank 16 --alpha 64" \
    2>&1 | tee "${RUNG_LOG_DIR}/export_${adapter_name}.log"
}

export_adapters() {
  ensure_inputs_visible
  export_one qwen3_0_6b_lora_seed42_ngpu2 "${QWEN3_0_6B_ROOT}" qwen3_0_6b_seed42
  export_one qwen3_0_6b_lora_seed43_ngpu2 "${QWEN3_0_6B_ROOT}" qwen3_0_6b_seed43
  export_one qwen3_1_7b_lora_seed42_ngpu2 "${QWEN3_1_7B_ROOT}" qwen3_1_7b_seed42
  export_one qwen3_1_7b_lora_seed43_ngpu2 "${QWEN3_1_7B_ROOT}" qwen3_1_7b_seed43
}

eval_one() {
  local name="$1" model_root="$2" adapter="${3:-}"
  local result_dir="${ROOTFS_RESULT_ROOT}/eval/dev/${name}"
  local lora_env=()
  if [[ -n "${adapter}" ]]; then
    lora_env=(LORA_ADAPTER="${adapter}" LORA_NAME="${name}" LORA_ID=1)
  fi
  TORCHTITAN_COUNTDOWN_DATA_ROOT="${DATA_ROOT}" \
  TORCHTITAN_COUNTDOWN_RESULTS_ROOT="${ROOTFS_RESULT_ROOT}/countdown_results" \
  SPLIT=dev \
  MODEL="${model_root}" \
  RESULT_DIR="${result_dir}" \
  NUM_ROLLOUTS="${NUM_ROLLOUTS:-32}" \
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}" \
  VLLM_CACHE_ROOT="${ROOTFS_RESULT_ROOT}/vllm_cache/${name}/vllm" \
  XDG_CACHE_HOME="${ROOTFS_RESULT_ROOT}/vllm_cache/${name}/xdg" \
  TORCHINDUCTOR_CACHE_DIR="${ROOTFS_RESULT_ROOT}/vllm_cache/${name}/torchinductor" \
  env "${lora_env[@]}" \
  "${repo_root}/experiments/countdown_search_distill/run_eval.sh" \
    2>&1 | tee "${RUNG_LOG_DIR}/eval_${name}.log"
}

eval_gate() {
  ensure_inputs_visible
  eval_one qwen3_0_6b_base "${QWEN3_0_6B_ROOT}"
  eval_one qwen3_1_7b_base "${QWEN3_1_7B_ROOT}"
  eval_one qwen3_0_6b_seed42 "${QWEN3_0_6B_ROOT}" "${ROOTFS_RESULT_ROOT}/adapters/qwen3_0_6b_seed42"
  eval_one qwen3_0_6b_seed43 "${QWEN3_0_6B_ROOT}" "${ROOTFS_RESULT_ROOT}/adapters/qwen3_0_6b_seed43"
  eval_one qwen3_1_7b_seed42 "${QWEN3_1_7B_ROOT}" "${ROOTFS_RESULT_ROOT}/adapters/qwen3_1_7b_seed42"
  eval_one qwen3_1_7b_seed43 "${QWEN3_1_7B_ROOT}" "${ROOTFS_RESULT_ROOT}/adapters/qwen3_1_7b_seed43"
}

write_manifest() {
  local manifest="${RESULT_ROOT}/run_manifest.json"
  jq -n \
    --arg run_id "${RUN_ID}" \
    --arg attempt_id "${ATTEMPT_ID}" \
    --arg dataset_uri "${DATASET_URI}" \
    --arg qwen3_0_6b_uri "${QWEN3_0_6B_URI}" \
    --arg qwen3_1_7b_uri "${QWEN3_1_7B_URI}" \
    --slurpfile dataset "${HOST_DATA_ROOTFS_MIRROR}/.staged.json" \
    --slurpfile q06 "${HOST_QWEN3_0_6B_ROOTFS_MIRROR}/.staged.json" \
    --slurpfile q17 "${HOST_QWEN3_1_7B_ROOTFS_MIRROR}/.staged.json" \
    '{
      run_id: $run_id,
      attempt_id: $attempt_id,
      claim_labels: ["smoke", "deterministic_comparison", "replicated_eval"],
      staged_inputs: {
        dataset: {uri: $dataset_uri, staged: $dataset[0]},
        qwen3_0_6b: {uri: $qwen3_0_6b_uri, staged: $q06[0]},
        qwen3_1_7b: {uri: $qwen3_1_7b_uri, staged: $q17[0]}
      },
      training: {
        tokens_per_step: 32768,
        seq_len: 512,
        steps: 94,
        checkpoint_interval: 31,
        lora: {rank: 16, alpha: 64, targets: ["wqkv", "wo", "w13", "w2"]}
      },
      environment_class: "shared_gpu"
    }' > "${manifest}"
}

publish_outputs() {
  write_manifest
  local publish_root="${script_dir}/cache/publish-bundles/glm53-ladder/${RUN_ID}/${ATTEMPT_ID}"
  rm -rf "${publish_root}"
  mkdir -p "${publish_root}"
  cp "${RESULT_ROOT}/run_manifest.json" "${publish_root}/run_manifest.json"
  if [[ -d "${RESULT_ROOT}/parity" ]]; then
    cp -R "${RESULT_ROOT}/parity" "${publish_root}/parity"
  fi
  if [[ -d "${RUNG_LOG_DIR}" ]]; then
    cp -R "${RUNG_LOG_DIR}" "${publish_root}/logs"
  fi
  if [[ -d "${RESULT_ROOT}/train" ]]; then
    mkdir -p "${publish_root}/train"
    rsync -a --exclude checkpoint "${RESULT_ROOT}/train/" "${publish_root}/train/"
  fi
  "${script_dir}/publish.sh" "${RESULT_ROOT}/adapters" \
    "torchtitan/runs/glm53-ladder/${RUN_ID}/${ATTEMPT_ID}/adapters" \
    | tee "${RUNG_LOG_DIR}/publish_adapters.log"
  "${script_dir}/publish.sh" "${RESULT_ROOT}/eval" \
    "torchtitan/runs/glm53-ladder/${RUN_ID}/${ATTEMPT_ID}/eval" \
    | tee "${RUNG_LOG_DIR}/publish_eval.log"
  "${script_dir}/publish.sh" "${RUNG_LOG_DIR}" \
    "torchtitan/runs/glm53-ladder/${RUN_ID}/${ATTEMPT_ID}/logs" \
    | tee "${RUNG_LOG_DIR}/publish_logs.log"
  "${script_dir}/publish.sh" "${publish_root}" \
    "torchtitan/runs/glm53-ladder/${RUN_ID}/${ATTEMPT_ID}/bundle" \
    | tee "${RUNG_LOG_DIR}/publish_bundle.log"
}

cmd="${1:-all}"
case "${cmd}" in
  all)
    stage_inputs
    dry_run_configs
    run_parity
    train_gate_runs
    resume_gate
    export_adapters
    eval_gate
    publish_outputs
    ;;
  stage) stage_inputs ;;
  mirror) mirror_inputs; verify_hdfs_inputs ;;
  dry-run) dry_run_configs ;;
  parity) run_parity ;;
  train) train_gate_runs ;;
  resume) resume_gate ;;
  export) export_adapters ;;
  eval) eval_gate ;;
  publish) publish_outputs ;;
  --help|-h) usage ;;
  *) usage; exit 2 ;;
esac
