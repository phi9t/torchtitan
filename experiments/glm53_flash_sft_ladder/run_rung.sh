#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
rootfs_repo_root="/workspace/torchtitan"
rootfs_script_dir="${rootfs_repo_root}/experiments/glm53_flash_sft_ladder"

usage() {
  cat >&2 <<'USAGE'
Usage: MODEL_KEY=<key> run_rung.sh [all|stage|mirror|dry-run|parity|train|resume|export|router-stats|eval|publish|prune]

Model keys: qwen3_0_6b, qwen3_1_7b, qwen3_8b, qwen3_30b_a3b.
Python, torchrun, and vLLM stages re-enter the rootfs. Waystone
stage/publish stays on the host.
USAGE
}

MODEL_KEY="${MODEL_KEY:-qwen3_8b}"
LORA_ARM="${LORA_ARM:-${GLM53_LADDER_LORA_ARM:-A}}"
DATASET_URI="${DATASET_URI:-hdfs://harunava/user/tiger/waystone/torchtitan/datasets/countdown-sft/v1}"
HOST_CACHE_ROOT="${GLM_LADDER_CACHE:-/data05/${USER:-user}/glm53-ladder-cache}"

case "${MODEL_KEY}" in
  qwen3_0_6b)
    MODEL_FLAVOR="0.6B"
    MODEL_CONFIG="qwen3_0_6b_countdown_lora"
    MODEL_DIR_NAME="Qwen3-0.6B"
    MODEL_URI="${MODEL_URI:-${QWEN3_0_6B_URI:-hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca}}"
    MODEL_ASSETS_ENV="GLM53_LADDER_HF_ASSETS_PATH_0_6B"
    DEFAULT_TRAIN_NGPU=2
    ;;
  qwen3_1_7b)
    MODEL_FLAVOR="1.7B"
    MODEL_CONFIG="qwen3_1_7b_countdown_lora"
    MODEL_DIR_NAME="Qwen3-1.7B"
    MODEL_URI="${MODEL_URI:-${QWEN3_1_7B_URI:-hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}}"
    MODEL_ASSETS_ENV="GLM53_LADDER_HF_ASSETS_PATH_1_7B"
    DEFAULT_TRAIN_NGPU=2
    ;;
  qwen3_8b)
    MODEL_FLAVOR="8B"
    MODEL_CONFIG="qwen3_8b_countdown_lora"
    MODEL_DIR_NAME="Qwen3-8B"
    MODEL_URI="${MODEL_URI:-${QWEN3_8B_URI:-hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218}}"
    MODEL_ASSETS_ENV="GLM53_LADDER_HF_ASSETS_PATH_8B"
    DEFAULT_TRAIN_NGPU=2
    ;;
  qwen3_30b_a3b)
    MODEL_FLAVOR="30B-A3B"
    MODEL_CONFIG="qwen3_30b_a3b_countdown_lora"
    MODEL_DIR_NAME="Qwen3-30B-A3B"
    MODEL_URI="${MODEL_URI:-${QWEN3_30B_A3B_URI:-hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-30B-A3B/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}}"
    MODEL_ASSETS_ENV="GLM53_LADDER_HF_ASSETS_PATH_30B_A3B"
    DEFAULT_TRAIN_NGPU=4
    ;;
  *)
    printf 'unknown MODEL_KEY: %s\n' "${MODEL_KEY}" >&2
    usage
    exit 2
    ;;
esac

if [[ -z "${MODEL_URI}" ]]; then
  printf 'MODEL_URI is required for MODEL_KEY=%s\n' "${MODEL_KEY}" >&2
  exit 2
fi

if [[ "${MODEL_KEY}" == "qwen3_30b_a3b" ]]; then
  case "${LORA_ARM}" in
    A|B) ;;
    *)
      printf 'LORA_ARM must be A or B for MODEL_KEY=%s, got %s\n' "${MODEL_KEY}" "${LORA_ARM}" >&2
      exit 2
      ;;
  esac
  RUN_VARIANT="${MODEL_KEY}_arm${LORA_ARM}"
  DEFAULT_TRAIN_SEEDS="$([[ "${LORA_ARM}" == "A" ]] && printf '42 43' || printf '42')"
  LORA_TARGETS_JSON="$([[ "${LORA_ARM}" == "A" ]] && printf '["wqkv","wo"]' || printf '["wqkv","wo","routed_experts.w13","routed_experts.w2"]')"
else
  RUN_VARIANT="${MODEL_KEY}"
  DEFAULT_TRAIN_SEEDS="42 43"
  LORA_TARGETS_JSON='["wqkv","wo","w13","w2"]'
fi
TRAIN_SEEDS="${TRAIN_SEEDS:-${DEFAULT_TRAIN_SEEDS}}"
read -r -a TRAIN_SEEDS_ARRAY <<< "${TRAIN_SEEDS}"

RUN_ID="${RUN_ID:-${MODEL_KEY}-$(date -u +%Y%m%dT%H%M%SZ)}"
ATTEMPT_ID="${ATTEMPT_ID:-attempt-01}"
ROOTFS_CACHE_ROOT="${ROOTFS_CACHE_ROOT:-${rootfs_script_dir}/cache/${MODEL_KEY}-inputs}"
RESULT_ROOT="${RESULT_ROOT:-${script_dir}/results/${RUN_ID}/${ATTEMPT_ID}}"
ROOTFS_RESULT_ROOT="${ROOTFS_RESULT_ROOT:-${rootfs_script_dir}/results/${RUN_ID}/${ATTEMPT_ID}}"
HOST_DATA_ROOT="${HOST_CACHE_ROOT}/datasets/countdown-sft/v1"
HOST_MODEL_ROOT="${HOST_CACHE_ROOT}/model-weights/Qwen/${MODEL_DIR_NAME}"
DATA_ROOT="${ROOTFS_CACHE_ROOT}/datasets/countdown-sft/v1"
MODEL_ROOT="${ROOTFS_CACHE_ROOT}/model-weights/Qwen/${MODEL_DIR_NAME}"
HOST_MIRROR_ROOT="${script_dir}/cache/${MODEL_KEY}-inputs"
HOST_DATA_ROOTFS_MIRROR="${HOST_MIRROR_ROOT}/datasets/countdown-sft/v1"
HOST_MODEL_ROOTFS_MIRROR="${HOST_MIRROR_ROOT}/model-weights/Qwen/${MODEL_DIR_NAME}"
TRAIN_DATA_FILE="${DATA_ROOT}/train/formatting.jsonl"
DEV_PROBLEMS="${DATA_ROOT}/dev/problems.jsonl"
RUNG_LOG_DIR="${RESULT_ROOT}/logs"
TRAIN_NGPU="${TRAIN_NGPU:-${DEFAULT_TRAIN_NGPU}}"
RESUME_NGPU="${RESUME_NGPU:-${TRAIN_NGPU}}"
ENVIRONMENT_CLASS="${ENVIRONMENT_CLASS:-shared_gpu}"
mkdir -p "${RESULT_ROOT}" "${RUNG_LOG_DIR}"

join_devices() {
  local count="$1"
  local result=""
  local index
  for ((index = 0; index < count; index += 1)); do
    if [[ -n "${result}" ]]; then
      result+=","
    fi
    result+="${index}"
  done
  printf '%s\n' "${result}"
}

default_devices() {
  local env_name="$1" count="$2"
  local fallback="$3"
  local value="${!env_name:-}"
  if [[ -n "${value}" ]]; then
    printf '%s\n' "${value}"
    return
  fi
  if [[ -n "${fallback}" ]]; then
    printf '%s\n' "${fallback}"
    return
  fi
  join_devices "${count}"
}

stage_inputs() {
  "${script_dir}/stage.sh" "${DATASET_URI}" "${HOST_DATA_ROOT}" | tee "${RUNG_LOG_DIR}/stage_dataset.log"
  "${script_dir}/stage.sh" "${MODEL_URI}" "${HOST_MODEL_ROOT}" | tee "${RUNG_LOG_DIR}/stage_${MODEL_KEY}.log"
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
  mirror_one "${HOST_MODEL_ROOT}" "${HOST_MODEL_ROOTFS_MIRROR}"
}

verify_hdfs_inputs() {
  "${script_dir}/lib/hdfs_strict.sh" --quick \
    "${HOST_DATA_ROOTFS_MIRROR}" \
    "${HOST_MODEL_ROOTFS_MIRROR}"
}

ensure_inputs_visible() {
  if [[ ! -f "${HOST_DATA_ROOTFS_MIRROR}/train/formatting.jsonl" \
    || ! -f "${HOST_DATA_ROOTFS_MIRROR}/MANIFEST.jsonl" \
    || ! -f "${HOST_DATA_ROOTFS_MIRROR}/.staged.json" \
    || ! -d "${HOST_MODEL_ROOTFS_MIRROR}" \
    || ! -f "${HOST_MODEL_ROOTFS_MIRROR}/MANIFEST.jsonl" \
    || ! -f "${HOST_MODEL_ROOTFS_MIRROR}/.staged.json" ]]; then
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
  local name="$1" seed="$2" ngpu="$3" devices="$4" steps="${5:-94}" resume_step="${6:-}" stop_after_step="${7:-}"
  local out="${RESULT_ROOT}/train/${name}"
  local rootfs_out="${ROOTFS_RESULT_ROOT}/train/${name}"
  local guard_log="${RUNG_LOG_DIR}/host_ram_guard_${name}.log"
  local guard_pattern="${RUN_ID}-${name}"
  local guard_pid=""
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
  GLM53_HOST_RAM_GUARD_PATTERN="${guard_pattern}" \
    "${script_dir}/host_ram_guard.sh" "${GLM53_HOST_RAM_FLOOR_GB:-800}" "${guard_log}" "${guard_pattern}" &
  guard_pid="$!"
  trap 'if [[ -n "${guard_pid}" ]]; then kill "${guard_pid}" 2>/dev/null || true; wait "${guard_pid}" 2>/dev/null || true; fi' RETURN
  env \
    GLM53_LADDER_DATA_FILE="${TRAIN_DATA_FILE}" \
    GLM53_LADDER_DUMP_ROOT="${ROOTFS_RESULT_ROOT}/train" \
    GLM53_LADDER_DUMP_FOLDER="${rootfs_out}" \
    "${MODEL_ASSETS_ENV}=${MODEL_ROOT}" \
    GLM53_LADDER_LORA_ARM="${LORA_ARM}" \
    GLM53_LADDER_EP_DEGREE="${GLM53_LADDER_EP_DEGREE:-4}" \
    GLM53_LADDER_CPU_OFFLOAD="${GLM53_LADDER_CPU_OFFLOAD:-0}" \
    GLM53_LADDER_SEED="${seed}" \
    GLM53_LADDER_DETERMINISTIC="${GLM53_LADDER_DETERMINISTIC:-0}" \
    GLM53_LADDER_STEPS="${steps}" \
    GLM53_LADDER_CHECKPOINT_INTERVAL=31 \
    GLM53_LADDER_KEEP_LATEST_K="${GLM53_LADDER_KEEP_LATEST_K:-2}" \
    CUDA_VISIBLE_DEVICES="${devices}" \
    TORCHTITAN_ROOTFS_ENV=glm53_train \
    "${repo_root}/scripts/rootfs/enter_rootfs.sh" -- bash -lc \
      "unset CC CXX; cd /workspace/torchtitan && torchrun --nproc_per_node=${ngpu} --rdzv_id '${RUN_ID}-${name}' --rdzv_backend c10d --rdzv_endpoint='${TORCHTITAN_RDZV_ENDPOINT:-127.0.0.1:0}' --local_addr='${TORCHTITAN_LOCAL_ADDR:-127.0.0.1}' --local-ranks-filter 0 --role rank --tee 3 -m ${train_module} ${stop_args[*]} --module torchtitan.experiments.glm53_flash_sft_ladder.config_registry --config ${MODEL_CONFIG} --output-dir ${rootfs_out} ${resume_args[*]}" \
      2>&1 | tee "${RUNG_LOG_DIR}/train_${name}.log"
  kill "${guard_pid}" 2>/dev/null || true
  wait "${guard_pid}" 2>/dev/null || true
  trap - RETURN
}

prune_train_checkpoints() {
  local train_name="$1" keep_step="${2:-}"
  local checkpoint_dir="${RESULT_ROOT}/train/${train_name}/checkpoint"
  if [[ ! -d "${checkpoint_dir}" ]]; then
    return
  fi
  if [[ -z "${keep_step}" ]]; then
    rm -rf "${checkpoint_dir}"
    return
  fi
  find "${checkpoint_dir}" -mindepth 1 -maxdepth 1 -type d -name 'step-*' ! -name "step-${keep_step}" -print -exec rm -rf {} +
}

dry_run_configs() {
  ensure_inputs_visible
  local dry_ngpu dry_devices
  if [[ "${MODEL_KEY}" == "qwen3_30b_a3b" ]]; then
    dry_ngpu="${DRY_RUN_NGPU:-${TRAIN_NGPU}}"
  else
    dry_ngpu="${DRY_RUN_NGPU:-1}"
  fi
  dry_devices="$(default_devices DRY_RUN_DEVICES "${dry_ngpu}" "")"
  train_one "dry_${RUN_VARIANT}_lora" 42 "${dry_ngpu}" "${dry_devices}" "${DRY_RUN_STEPS:-2}"
  prune_train_checkpoints "dry_${RUN_VARIANT}_lora"
}

run_parity() {
  ensure_inputs_visible
  CUDA_VISIBLE_DEVICES="${PARITY_DEVICES:-0}" rootfs_train "python -m torchtitan.experiments.glm53_flash_sft_ladder.parity --model-flavor '${MODEL_FLAVOR}' --hf-assets-path '${MODEL_ROOT}' --output '${ROOTFS_RESULT_ROOT}/parity/${MODEL_KEY}.json'" \
    2>&1 | tee "${RUNG_LOG_DIR}/parity_${MODEL_KEY}.log"
}

train_gate_runs() {
  ensure_inputs_visible
  local seed devices fallback index name
  index=0
  if [[ "${TRAIN_PARALLEL:-0}" == "1" ]]; then
    for seed in "${TRAIN_SEEDS_ARRAY[@]}"; do
      fallback=""
      if [[ "${index}" == "0" ]]; then
        fallback="${GPU_PAIR_0:-}"
      elif [[ "${index}" == "1" ]]; then
        fallback="${GPU_PAIR_1:-}"
      fi
      devices="$(default_devices "TRAIN_DEVICES_SEED${seed}" "${TRAIN_NGPU}" "${fallback}")"
      name="${RUN_VARIANT}_lora_seed${seed}_ngpu${TRAIN_NGPU}"
      train_one "${name}" "${seed}" "${TRAIN_NGPU}" "${devices}" &
      index=$((index + 1))
    done
    wait
  else
    for seed in "${TRAIN_SEEDS_ARRAY[@]}"; do
      fallback=""
      if [[ "${index}" == "0" ]]; then
        fallback="${GPU_PAIR_0:-}"
      elif [[ "${index}" == "1" ]]; then
        fallback="${GPU_PAIR_1:-}"
      fi
      devices="$(default_devices "TRAIN_DEVICES_SEED${seed}" "${TRAIN_NGPU}" "${fallback}")"
      name="${RUN_VARIANT}_lora_seed${seed}_ngpu${TRAIN_NGPU}"
      train_one "${name}" "${seed}" "${TRAIN_NGPU}" "${devices}"
      prune_train_checkpoints "${name}" 94
      index=$((index + 1))
    done
  fi
  for seed in "${TRAIN_SEEDS_ARRAY[@]}"; do
    prune_train_checkpoints "${RUN_VARIANT}_lora_seed${seed}_ngpu${TRAIN_NGPU}" 94
  done
}

resume_gate() {
  ensure_inputs_visible
  local resume_a_devices resume_b_devices resume_full_name resume_stop_name
  resume_full_name="resume_full_a_${RUN_VARIANT}"
  resume_stop_name="resume_stop62_${RUN_VARIANT}"
  resume_a_devices="$(default_devices RESUME_DEVICES_A "${RESUME_NGPU}" "${GPU_PAIR_0:-}")"
  resume_b_devices="$(default_devices RESUME_DEVICES_B "${RESUME_NGPU}" "${GPU_PAIR_1:-}")"
  GLM53_LADDER_DETERMINISTIC=1 train_one "${resume_full_name}" 42 "${RESUME_NGPU}" "${resume_a_devices}"
  prune_train_checkpoints "${resume_full_name}"
  GLM53_LADDER_DETERMINISTIC=1 train_one "${resume_stop_name}" 42 "${RESUME_NGPU}" "${resume_b_devices}" 94 "" 62
  prune_train_checkpoints "${resume_stop_name}" 62
  "${script_dir}/publish.sh" "${RESULT_ROOT}/train/${resume_stop_name}/checkpoint/step-62" \
    "torchtitan/checkpoints/glm53-ladder/${RUN_ID}/${ATTEMPT_ID}/resume_stop62_${RUN_VARIANT}_step62" \
    | tee "${RUNG_LOG_DIR}/publish_resume_step62.log"
  rm -rf "${RESULT_ROOT}/train/${resume_stop_name}/checkpoint/step-62"
  "${script_dir}/stage.sh" \
    "hdfs://harunava/user/tiger/waystone/torchtitan/checkpoints/glm53-ladder/${RUN_ID}/${ATTEMPT_ID}/resume_stop62_${RUN_VARIANT}_step62" \
    "${RESULT_ROOT}/train/${resume_stop_name}/checkpoint/step-62" \
    | tee "${RUNG_LOG_DIR}/stage_resume_step62.log"
  "${script_dir}/lib/hdfs_strict.sh" --quick "${RESULT_ROOT}/train/${resume_stop_name}/checkpoint/step-62"
  if [[ -d "${RESULT_ROOT}/train/${resume_stop_name}/run_evidence" ]]; then
    rm -rf "${RESULT_ROOT}/train/${resume_stop_name}/run_evidence_stop_after_62"
    mv "${RESULT_ROOT}/train/${resume_stop_name}/run_evidence" \
      "${RESULT_ROOT}/train/${resume_stop_name}/run_evidence_stop_after_62"
  fi
  GLM53_LADDER_DETERMINISTIC=1 train_one "${resume_stop_name}" 42 "${RESUME_NGPU}" "${resume_b_devices}" 94 62
  prune_train_checkpoints "${resume_stop_name}"
  rootfs_train "python -m torchtitan.experiments.glm53_flash_sft_ladder.compare_resume_loss --full '${ROOTFS_RESULT_ROOT}/train/${resume_full_name}' --resumed '${ROOTFS_RESULT_ROOT}/train/${resume_stop_name}' --output-json '${ROOTFS_RESULT_ROOT}/logs/resume_loss_compare.json' --output-md '${ROOTFS_RESULT_ROOT}/logs/resume_loss_compare.md'" \
    2>&1 | tee "${RUNG_LOG_DIR}/compare_resume_loss.log"
}

export_one() {
  local train_name="$1" adapter_name="$2"
  local checkpoint="${ROOTFS_RESULT_ROOT}/train/${train_name}/checkpoint/step-94"
  local output="${ROOTFS_RESULT_ROOT}/adapters/${adapter_name}"
  rootfs_train "python -m torchtitan.experiments.countdown_search_distill.cli export-lora --checkpoint '${checkpoint}' --output '${output}' --base-model-name-or-path '${MODEL_ROOT}' --rank 16 --alpha 64" \
    2>&1 | tee "${RUNG_LOG_DIR}/export_${adapter_name}.log"
}

export_adapters() {
  ensure_inputs_visible
  local seed
  for seed in "${TRAIN_SEEDS_ARRAY[@]}"; do
    export_one "${RUN_VARIANT}_lora_seed${seed}_ngpu${TRAIN_NGPU}" "${RUN_VARIANT}_seed${seed}"
  done
}

eval_one() {
  local name="$1" adapter="${2:-}"
  local result_dir="${ROOTFS_RESULT_ROOT}/eval/dev/${name}"
  local lora_env=""
  if [[ -n "${adapter}" ]]; then
    lora_env="LORA_ADAPTER='${adapter}' LORA_NAME='${name}' LORA_ID=1 "
  fi
  rootfs_default \
    "TORCHTITAN_COUNTDOWN_DATA_ROOT='${DATA_ROOT}' \
TORCHTITAN_COUNTDOWN_RESULTS_ROOT='${ROOTFS_RESULT_ROOT}/countdown_results' \
SPLIT=dev \
MODEL='${MODEL_ROOT}' \
RESULT_DIR='${result_dir}' \
NUM_ROLLOUTS='${NUM_ROLLOUTS:-32}' \
VLLM_GPU_MEMORY_UTILIZATION='${VLLM_GPU_MEMORY_UTILIZATION:-0.35}' \
VLLM_MOE_BACKEND='${VLLM_MOE_BACKEND:-}' \
VLLM_TENSOR_PARALLEL_SIZE='${VLLM_TENSOR_PARALLEL_SIZE:-}' \
VLLM_CACHE_ROOT='${ROOTFS_RESULT_ROOT}/vllm_cache/${name}/vllm' \
XDG_CACHE_HOME='${ROOTFS_RESULT_ROOT}/vllm_cache/${name}/xdg' \
TORCHINDUCTOR_CACHE_DIR='${ROOTFS_RESULT_ROOT}/vllm_cache/${name}/torchinductor' \
CUDA_VISIBLE_DEVICES='${EVAL_DEVICES:-0}' \
${lora_env}experiments/countdown_search_distill/run_eval.sh" \
    2>&1 | tee "${RUNG_LOG_DIR}/eval_${name}.log"
}

eval_gate() {
  ensure_inputs_visible
  eval_one "${MODEL_KEY}_base"
  local seed
  for seed in "${TRAIN_SEEDS_ARRAY[@]}"; do
    eval_one "${RUN_VARIANT}_seed${seed}" "${ROOTFS_RESULT_ROOT}/adapters/${RUN_VARIANT}_seed${seed}"
  done
}

router_stats_gate() {
  ensure_inputs_visible
  local seed checkpoint adapter_arg=()
  rootfs_default "CUDA_VISIBLE_DEVICES='${ROUTER_STATS_DEVICES:-0}' python -m torchtitan.experiments.glm53_flash_sft_ladder.router_stats --hf-assets-path '${MODEL_ROOT}' --dev-problems '${DEV_PROBLEMS}' --output '${ROOTFS_RESULT_ROOT}/router_stats/${MODEL_KEY}_base.json' --model-id '${MODEL_KEY}_base'" \
    2>&1 | tee "${RUNG_LOG_DIR}/router_stats_base.log"
  seed="${TRAIN_SEEDS_ARRAY[0]}"
  checkpoint="${ROOTFS_RESULT_ROOT}/adapters/${RUN_VARIANT}_seed${seed}"
  if [[ -d "${RESULT_ROOT}/adapters/${RUN_VARIANT}_seed${seed}" ]]; then
    adapter_arg=(--adapter-path "${checkpoint}" --model-id "${RUN_VARIANT}_seed${seed}")
    rootfs_default "CUDA_VISIBLE_DEVICES='${ROUTER_STATS_DEVICES:-0}' python -m torchtitan.experiments.glm53_flash_sft_ladder.router_stats --hf-assets-path '${MODEL_ROOT}' --dev-problems '${DEV_PROBLEMS}' --output '${ROOTFS_RESULT_ROOT}/router_stats/${RUN_VARIANT}_seed${seed}.json' ${adapter_arg[*]}" \
      2>&1 | tee "${RUNG_LOG_DIR}/router_stats_${RUN_VARIANT}_seed${seed}.log"
  fi
}

write_manifest() {
  local manifest="${RESULT_ROOT}/run_manifest.json"
  local seeds_json lora_arms_json
  seeds_json="$(printf '%s\n' "${TRAIN_SEEDS_ARRAY[@]}" | jq -Rsc 'split("\n")[:-1] | map(tonumber)')"
  if [[ "${MODEL_KEY}" == "qwen3_30b_a3b" ]]; then
    lora_arms_json='{
      "A": {"seeds": [42, 43], "targets": ["wqkv", "wo"]},
      "B": {"seeds": [42], "targets": ["wqkv", "wo", "routed_experts.w13", "routed_experts.w2"]}
    }'
  else
    lora_arms_json="$(jq -nc --arg arm "${LORA_ARM}" --argjson seeds "${seeds_json}" --argjson targets "${LORA_TARGETS_JSON}" \
      '{"\($arm)": {seeds: $seeds, targets: $targets}}')"
  fi
  jq -n \
    --arg run_id "${RUN_ID}" \
    --arg attempt_id "${ATTEMPT_ID}" \
    --arg model_key "${MODEL_KEY}" \
    --arg run_variant "${RUN_VARIANT}" \
    --arg lora_arm "${LORA_ARM}" \
    --arg model_flavor "${MODEL_FLAVOR}" \
    --arg model_config "${MODEL_CONFIG}" \
    --arg dataset_uri "${DATASET_URI}" \
    --arg model_uri "${MODEL_URI}" \
    --argjson train_ngpu "${TRAIN_NGPU}" \
    --argjson resume_ngpu "${RESUME_NGPU}" \
    --argjson seeds "${seeds_json}" \
    --argjson lora_targets "${LORA_TARGETS_JSON}" \
    --argjson lora_arms "${lora_arms_json}" \
    --argjson expert_parallel_degree "${GLM53_LADDER_EP_DEGREE:-4}" \
    --argjson enable_cpu_offload "${GLM53_LADDER_CPU_OFFLOAD:-0}" \
    --arg environment_class "${ENVIRONMENT_CLASS}" \
    --slurpfile dataset "${HOST_DATA_ROOTFS_MIRROR}/.staged.json" \
    --slurpfile model "${HOST_MODEL_ROOTFS_MIRROR}/.staged.json" \
    '{
      run_id: $run_id,
      attempt_id: $attempt_id,
      model_key: $model_key,
      run_variant: $run_variant,
      lora_arm: $lora_arm,
      model_flavor: $model_flavor,
      model_config: $model_config,
      claim_labels: ["smoke", "deterministic_comparison", "replicated_eval"],
      staged_inputs: {
        dataset: {uri: $dataset_uri, staged: $dataset[0]},
        model: {uri: $model_uri, staged: $model[0]}
      },
      training: {
        seeds: $seeds,
        train_ngpu: $train_ngpu,
        resume_ngpu: $resume_ngpu,
        tokens_per_step: 32768,
        seq_len: 512,
        steps: 94,
        checkpoint_interval: 31,
        lora: {rank: 16, alpha: 64, targets: $lora_targets},
        lora_arms: $lora_arms,
        expert_parallel_degree: $expert_parallel_degree,
        enable_cpu_offload: ($enable_cpu_offload != 0)
      },
      eval: {
        split: "dev",
        num_problems: 500,
        num_rollouts: 32
      },
      environment_class: $environment_class
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
  if [[ -d "${RESULT_ROOT}/router_stats" ]]; then
    cp -R "${RESULT_ROOT}/router_stats" "${publish_root}/router_stats"
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
  if [[ -d "${RESULT_ROOT}/router_stats" ]]; then
    "${script_dir}/publish.sh" "${RESULT_ROOT}/router_stats" \
      "torchtitan/runs/glm53-ladder/${RUN_ID}/${ATTEMPT_ID}/router_stats" \
      | tee "${RUNG_LOG_DIR}/publish_router_stats.log"
  fi
  "${script_dir}/publish.sh" "${RUNG_LOG_DIR}" \
    "torchtitan/runs/glm53-ladder/${RUN_ID}/${ATTEMPT_ID}/logs" \
    | tee "${RUNG_LOG_DIR}/publish_logs.log"
  "${script_dir}/publish.sh" "${publish_root}" \
    "torchtitan/runs/glm53-ladder/${RUN_ID}/${ATTEMPT_ID}/bundle" \
    | tee "${RUNG_LOG_DIR}/publish_bundle.log"
}

prune_checkpoints() {
  if [[ -d "${RESULT_ROOT}/train" ]]; then
    find "${RESULT_ROOT}/train" -mindepth 2 -maxdepth 2 -type d -name checkpoint -print -exec rm -rf {} +
  fi
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
    router_stats_gate
    for seed in "${TRAIN_SEEDS_ARRAY[@]}"; do
      prune_train_checkpoints "${RUN_VARIANT}_lora_seed${seed}_ngpu${TRAIN_NGPU}"
    done
    eval_gate
    publish_outputs
    prune_checkpoints
    ;;
  stage) stage_inputs ;;
  mirror) mirror_inputs; verify_hdfs_inputs ;;
  dry-run) dry_run_configs ;;
  parity) run_parity ;;
  train) train_gate_runs ;;
  resume) resume_gate ;;
  export) export_adapters ;;
  router-stats) router_stats_gate ;;
  eval) eval_gate ;;
  publish) publish_outputs ;;
  prune) prune_checkpoints ;;
  --help|-h) usage ;;
  *) usage; exit 2 ;;
esac
