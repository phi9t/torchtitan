#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
waystone_loaded=0

ensure_waystone_loaded() {
  if (( waystone_loaded == 0 )); then
    # shellcheck source=experiments/glm53_flash_sft_ladder/lib/waystone.sh
    source "${repo_root}/experiments/glm53_flash_sft_ladder/lib/waystone.sh"
    waystone_loaded=1
  fi
}

usage() {
  cat >&2 <<'USAGE'
Usage:
  run_rung.sh [L1|L2|L3|L4]
  run_rung.sh --print-params

Runs one GLM-5.3 architecture-ladder rung through TorchTitan. Configuration is
owned by GLM53_ARCH_LADDER_* environment variables. One default-mode GPU with no
existing compute process is selected or verified before every launch.
USAGE
}

config_for_rung() {
  case "$1" in
    L1) printf 'glm53_arch_ladder_l1_qwen3_dense\n' ;;
    L2) printf 'glm53_arch_ladder_l2_qwen3_moe\n' ;;
    L3) printf 'glm53_arch_ladder_l3_kimi_k3\n' ;;
    L4) printf 'glm53_arch_ladder_l4_glm53_flash\n' ;;
    *) return 1 ;;
  esac
}

run_print_params() {
  (
    cd "$repo_root"
    TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc \
      'unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH="$PWD:${PYTHONPATH:-}" && python -m torchtitan.experiments.glm53_arch_ladder.inspect'
  )
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--print-params" ]]; then
  run_print_params
  exit 0
fi

rung="${1:-${GLM53_ARCH_LADDER_RUNG:-L1}}"
rung="${rung^^}"
config_name="$(config_for_rung "$rung")" || {
  printf 'unknown rung: %s\n' "$rung" >&2
  usage
  exit 2
}
export GLM53_ARCH_LADDER_RUNG="$rung"

require_nvidia_smi() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'nvidia-smi is required for S07 GPU safety checks\n' >&2
    exit 1
  fi
}

gpu_compute_mode() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,compute_mode --format=csv,noheader,nounits \
    | awk -F, -v gpu="$gpu" '$1 ~ "^[[:space:]]*" gpu "[[:space:]]*$" {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}'
}

gpu_free_mib() {
  local gpu="$1"
  nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits \
    | awk 'NR == 1 {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1}'
}

gpu_compute_pids() {
  local gpu="$1"
  nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1}'
}

is_allowed_glm53_flash_rung4_pid() {
  local pid="$1" cmd
  [[ "${GLM53_ARCH_LADDER_ALLOW_GLM53_FLASH_RUNG4_CORUN:-1}" == "1" ]] || return 1
  cmd="$(ps -o cmd= -p "$pid" 2>/dev/null || true)"
  [[ "$cmd" == *"torchtitan.experiments.glm53_flash.config_registry"* ]] || return 1
  [[ "$cmd" == *"glm53_flash_countdown_lora"* ]] || return 1
  [[ "$cmd" == *"t08d-rung4-glm53"* ]] || return 1
  return 0
}

gpu_has_blocking_compute_apps() {
  local gpu="$1" pid
  if [[ "${GLM53_ARCH_LADDER_ALLOW_CORUN:-0}" == "1" ]]; then
    return 1
  fi
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    if ! is_allowed_glm53_flash_rung4_pid "$pid"; then
      return 0
    fi
  done < <(gpu_compute_pids "$gpu")
  return 1
}

is_safe_gpu() {
  local gpu="$1" mode free_mib cap_mib
  mode="$(gpu_compute_mode "$gpu")"
  [[ "$mode" == "Default" ]] || return 1
  gpu_has_blocking_compute_apps "$gpu" && return 1
  free_mib="$(gpu_free_mib "$gpu")"
  cap_mib="$(awk -v gib="${GLM53_ARCH_LADDER_GPU_MEMORY_CAP_GIB:-24}" 'BEGIN {printf "%d", gib * 1024}')"
  (( free_mib >= cap_mib )) || return 1
  return 0
}

select_safe_gpu() {
  local requested gpu
  requested="${GLM53_ARCH_LADDER_GPU:-${CUDA_VISIBLE_DEVICES:-}}"
  if [[ -n "$requested" ]]; then
    if [[ "$requested" == *,* ]]; then
      printf 'S07 requires one GPU per run; got CUDA_VISIBLE_DEVICES=%s\n' "$requested" >&2
      exit 1
    fi
    if ! is_safe_gpu "$requested"; then
      printf 'GPU %s is not safe for S07 (must be Default mode, have cap-sized free memory, and contain no blocking compute process)\n' "$requested" >&2
      nvidia-smi --query-gpu=index,compute_mode --format=csv,noheader,nounits >&2 || true
      nvidia-smi -i "$requested" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits >&2 || true
      exit 1
    fi
    printf '%s\n' "$requested"
    return 0
  fi

  while IFS=, read -r gpu _mode; do
    gpu="${gpu//[[:space:]]/}"
    if is_safe_gpu "$gpu"; then
      printf '%s\n' "$gpu"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,compute_mode --format=csv,noheader,nounits)

  printf 'no safe Default GPU with enough free memory and no blocking process is available for S07\n' >&2
  nvidia-smi --query-gpu=index,compute_mode --format=csv,noheader,nounits >&2 || true
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits >&2 || true
  exit 1
}

enforce_concurrency_limit() {
  local matches concurrent
  matches="$(pgrep -af 'torchrun .*torchtitan.experiments.glm53_arch_ladder.config_registry' || true)"
  concurrent="$(printf '%s\n' "$matches" | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')"
  if (( concurrent >= 3 )); then
    printf 'S07 allows at most 3 concurrent ladder runs; found %s\n' "$concurrent" >&2
    printf '%s\n' "$matches" >&2
    exit 1
  fi
}

require_nvidia_smi
enforce_concurrency_limit
selected_gpu="$(select_safe_gpu)"
export CUDA_VISIBLE_DEVICES="$selected_gpu"
export NGPU=1
export GLM53_ARCH_LADDER_ENVIRONMENT_CLASS="${GLM53_ARCH_LADDER_ENVIRONMENT_CLASS:-shared_gpu}"
export GLM53_ARCH_LADDER_GPU_MEMORY_CAP_GIB="${GLM53_ARCH_LADDER_GPU_MEMORY_CAP_GIB:-24}"
if [[ "$rung" == "L4" ]]; then
  export GLM53_FLASH_EXPERTS_IMPL="${GLM53_FLASH_EXPERTS_IMPL:-grouped}"
  export GLM53_ARCH_LADDER_VAL_TOKENS="${GLM53_ARCH_LADDER_VAL_TOKENS:-262144}"
fi

data_rev="${GLM53_ARCH_LADDER_DATA_REV:-ecbb586-fineweb10b-gpt2-binonly-v1}"
data_rel="datasets/fineweb10B-gpt2/${data_rev}"
staged_data_rel="experiments/glm53_arch_speedrun/results/data/fineweb10B-gpt2/${data_rev}"
host_staged_data_dir="${GLM53_ARCH_LADDER_HOST_DATA_DIR:-${repo_root}/${staged_data_rel}}"
rootfs_staged_data_dir="${GLM53_ARCH_LADDER_ROOTFS_DATA_DIR:-${staged_data_rel}}"
hdfs_data_uri="${GLM53_ARCH_LADDER_DATA_HDFS_URI:-}"
export GLM53_ARCH_LADDER_DATA_DIR="$rootfs_staged_data_dir"

resolve_hdfs_data_uri() {
  ensure_waystone_loaded
  if [[ -z "$hdfs_data_uri" ]]; then
    hdfs_data_uri="$(ws_child_uri "$data_rel")"
  fi
}

stage_dataset_from_hdfs() {
  resolve_hdfs_data_uri
  "${repo_root}/experiments/glm53_flash_sft_ladder/stage.sh" \
    --parallel "${GLM53_ARCH_LADDER_STAGE_PARALLEL:-2}" \
    "$hdfs_data_uri" "$host_staged_data_dir"
}

verify_hdfs_strict_inputs() {
  "${repo_root}/experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh" \
    --quick "$host_staged_data_dir"
}

resolve_hdfs_data_uri
if ! hdfs_strict_output="$(verify_hdfs_strict_inputs 2>&1)"; then
  printf '%s\n' "$hdfs_strict_output" >&2
  stage_dataset_from_hdfs
  hdfs_strict_output="$(verify_hdfs_strict_inputs)"
fi
staged_source_uri="$(jq -r '.source_uri' "${host_staged_data_dir}/.staged.json")"
staged_manifest_md5="$(jq -r '.manifest_md5' "${host_staged_data_dir}/.staged.json")"

run_id="${TORCHTITAN_RUN_ID:-s07-${rung,,}-seed${GLM53_ARCH_LADDER_SEED:-0}-$(date -u +%Y%m%dT%H%M%SZ)}"
attempt_id="${TORCHTITAN_ATTEMPT_ID:-attempt-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
export TORCHTITAN_RUN_ID="$run_id" TORCHTITAN_ATTEMPT_ID="$attempt_id"

mode="${GLM53_ARCH_LADDER_MODE:-science}"
claim_label="${GLM53_ARCH_LADDER_CLAIM_LABEL:-representative_small}"
evidence_tier="${GLM53_ARCH_LADDER_EVIDENCE_TIER:-tier0}"
environment_class="${GLM53_ARCH_LADDER_ENVIRONMENT_CLASS:-shared_gpu}"

dump_rel="experiments/glm53_arch_ladder/results/${run_id}"
host_dump_folder="${GLM53_ARCH_LADDER_HOST_DUMP_FOLDER:-${repo_root}/${dump_rel}}"
rootfs_dump_folder="${GLM53_ARCH_LADDER_ROOTFS_DUMP_FOLDER:-${dump_rel}}"
log_dir="${repo_root}/experiments/glm53_arch_ladder/logs"
mkdir -p "$host_dump_folder" "$log_dir"
export GLM53_ARCH_LADDER_DUMP_FOLDER="$rootfs_dump_folder"
printf '%s\n' "$hdfs_strict_output" > "${host_dump_folder}/hdfs_strict_inputs.txt"

metadata_file="${host_dump_folder}/attempt.json"
cat > "$metadata_file" <<EOF
{
  "lane": "glm53_arch_ladder",
  "mode": "${mode}",
  "rung": "${rung}",
  "claim_label": "${claim_label}",
  "evidence_tier": "${evidence_tier}",
  "run_id": "${run_id}",
  "attempt_id": "${attempt_id}",
  "environment_class": "${environment_class}",
  "seq_len": ${GLM53_ARCH_LADDER_SEQ_LEN:-4096},
  "local_batch_size": ${GLM53_ARCH_LADDER_LOCAL_BATCH_SIZE:-0},
  "tokens_per_step": ${GLM53_ARCH_LADDER_TOKENS_PER_STEP:-262144},
  "token_budget": ${GLM53_ARCH_LADDER_TOKEN_BUDGET:-268435456},
  "val_tokens": ${GLM53_ARCH_LADDER_VAL_TOKENS:-10485760},
  "data_mode": "${GLM53_ARCH_LADDER_DATA_MODE:-fineweb}",
  "fixed_bank_microbatches": ${GLM53_ARCH_LADDER_FIXED_BANK_MICROBATCHES:-1},
  "muon_lr": ${GLM53_ARCH_LADDER_MUON_LR:-0.02},
  "adamw_lr": ${GLM53_ARCH_LADDER_ADAMW_LR:-0.0008},
  "weight_decay": ${GLM53_ARCH_LADDER_WEIGHT_DECAY:-0.1},
  "moe_matrix_batches_on_muon": ${GLM53_ARCH_LADDER_MOE_MATRIX_BATCHES_ON_MUON:-1},
  "moe_router_on_muon": ${GLM53_ARCH_LADDER_MOE_ROUTER_ON_MUON:-1},
  "gpu_memory_cap_gib": ${GLM53_ARCH_LADDER_GPU_MEMORY_CAP_GIB:-24},
  "selected_gpu": "${selected_gpu}",
  "allow_corun": ${GLM53_ARCH_LADDER_ALLOW_CORUN:-0},
  "data_hdfs_uri": "${staged_source_uri}",
  "data_manifest_md5": "${staged_manifest_md5}",
  "data_host_dir": "${host_staged_data_dir}",
  "data_rootfs_dir": "${rootfs_staged_data_dir}",
  "glm53_flash_experts_impl": "${GLM53_FLASH_EXPERTS_IMPL:-}",
  "ngpu": 1
}
EOF

launch_cmd=(
  "${repo_root}/scripts/rootfs/enter_rootfs.sh"
  --
  bash
  -lc
  "unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH=\"\$PWD:\${PYTHONPATH:-}\" && LOG_RANK=\${LOG_RANK:-0} NGPU=1 MODULE=torchtitan.experiments.glm53_arch_ladder.config_registry CONFIG=${config_name} ./run_train.sh"
)

printf '%q ' CUDA_VISIBLE_DEVICES="$selected_gpu" TORCHTITAN_ROOTFS_ENV=glm53_train GLM53_FLASH_EXPERTS_IMPL="${GLM53_FLASH_EXPERTS_IMPL:-}" "${launch_cmd[@]}" > "${host_dump_folder}/launch_command.txt"
printf '\n' >> "${host_dump_folder}/launch_command.txt"

log_file="${log_dir}/${run_id}.log"
(
  cd "$repo_root"
  CUDA_VISIBLE_DEVICES="$selected_gpu" TORCHTITAN_ROOTFS_ENV=glm53_train "${launch_cmd[@]}"
) 2>&1 | tee "$log_file"

cp "$log_file" "${host_dump_folder}/train.log"
cp "${host_staged_data_dir}/MANIFEST.jsonl" "${host_dump_folder}/DATA_MANIFEST.jsonl"
cp "${host_staged_data_dir}/.staged.json" "${host_dump_folder}/DATA_STAGED.json"

write_metrics_json() {
  local src_log="$1" out_json="$2" line clean step loss memory tps
  local train_step train_loss train_memory train_tps val_step val_loss val_memory val_tps
  local resolved_microbatch_tokens resolved_tokens_per_step resolved_grad_accum resolved_seq_len resolved_local_batch_size
  train_step="" train_loss="" train_memory="" train_tps=""
  val_step="" val_loss="" val_memory="" val_tps=""
  resolved_microbatch_tokens="" resolved_tokens_per_step="" resolved_grad_accum="" resolved_seq_len="" resolved_local_batch_size=""
  while IFS= read -r line; do
    clean="$(printf '%s\n' "$line" | sed -E 's/\x1B\[[0-9;]*[A-Za-z]//g')"
    if [[ "$clean" =~ initialized[[:space:]]with[[:space:]]([0-9]+)[[:space:]]tokens[[:space:]]per[[:space:]]DP[[:space:]]rank,[[:space:]]([0-9]+)[[:space:]]tokens[[:space:]]per[[:space:]]train[[:space:]]step,[[:space:]]gradient[[:space:]]accumulation[[:space:]]steps[[:space:]]([0-9]+),[[:space:]]maximum[[:space:]]context[[:space:]]length[[:space:]]([0-9]+) ]]; then
      resolved_microbatch_tokens="${BASH_REMATCH[1]}"
      resolved_tokens_per_step="${BASH_REMATCH[2]}"
      resolved_grad_accum="${BASH_REMATCH[3]}"
      resolved_seq_len="${BASH_REMATCH[4]}"
      if (( resolved_seq_len > 0 )); then
        resolved_local_batch_size=$((resolved_microbatch_tokens / resolved_seq_len))
      fi
    fi
    if [[ "$clean" =~ step:[[:space:]]*([0-9]+).*loss:[[:space:]]*([0-9.]+).*memory:[[:space:]]*([0-9.]+)GiB.*tps:[[:space:]]*([0-9,]+) ]]; then
      step="${BASH_REMATCH[1]}"
      loss="${BASH_REMATCH[2]}"
      memory="${BASH_REMATCH[3]}"
      tps="${BASH_REMATCH[4]//,/}"
      if [[ "$clean" == *"validate step:"* ]]; then
        val_step="$step"; val_loss="$loss"; val_memory="$memory"; val_tps="$tps"
      else
        train_step="$step"; train_loss="$loss"; train_memory="$memory"; train_tps="$tps"
      fi
    fi
  done < "$src_log"
  jq -n \
    --arg run_id "$run_id" \
    --arg attempt_id "$attempt_id" \
    --arg rung "$rung" \
    --arg environment_class "$environment_class" \
    --arg selected_gpu "$selected_gpu" \
    --argjson seq_len "${GLM53_ARCH_LADDER_SEQ_LEN:-4096}" \
    --argjson tokens_per_step "${GLM53_ARCH_LADDER_TOKENS_PER_STEP:-262144}" \
    --argjson token_budget "${GLM53_ARCH_LADDER_TOKEN_BUDGET:-268435456}" \
    --arg train_step "$train_step" \
    --arg train_loss "$train_loss" \
    --arg train_memory "$train_memory" \
    --arg train_tps "$train_tps" \
    --arg val_step "$val_step" \
    --arg val_loss "$val_loss" \
    --arg val_memory "$val_memory" \
    --arg val_tps "$val_tps" \
    --arg resolved_microbatch_tokens "$resolved_microbatch_tokens" \
    --arg resolved_tokens_per_step "$resolved_tokens_per_step" \
    --arg resolved_grad_accum "$resolved_grad_accum" \
    --arg resolved_seq_len "$resolved_seq_len" \
    --arg resolved_local_batch_size "$resolved_local_batch_size" \
    'def maybe_num($value):
      if ($value | length) > 0 then ($value | tonumber) else null end;
    {
      run_id:$run_id,
      attempt_id:$attempt_id,
      rung:$rung,
      environment_class:$environment_class,
      selected_gpu:$selected_gpu,
      seq_len:$seq_len,
      tokens_per_step:$tokens_per_step,
      token_budget:$token_budget,
      resolved_seq_len:maybe_num($resolved_seq_len),
      resolved_local_batch_size:maybe_num($resolved_local_batch_size),
      resolved_microbatch_tokens:maybe_num($resolved_microbatch_tokens),
      resolved_tokens_per_step:maybe_num($resolved_tokens_per_step),
      gradient_accumulation_steps:maybe_num($resolved_grad_accum),
      final_train_step:maybe_num($train_step),
      final_train_loss:maybe_num($train_loss),
      tokens_per_second:maybe_num($train_tps),
      peak_memory_gib:maybe_num($train_memory),
      final_val_step:maybe_num($val_step),
      final_val_ce:maybe_num($val_loss),
      validation_tokens_per_second:maybe_num($val_tps),
      validation_peak_memory_gib:maybe_num($val_memory)
    }' > "$out_json"
}

write_metrics_json "$log_file" "${host_dump_folder}/metrics.json"

metrics_extract_log="${host_dump_folder}/metrics_extract.log"
if (
  cd "$repo_root"
  TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc \
    "unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH=\"\$PWD:\${PYTHONPATH:-}\" && python -m torchtitan.experiments.glm53_arch_ladder.extract_metrics '${rootfs_dump_folder}' --write"
) > "$metrics_extract_log" 2>&1; then
  :
else
  printf 'metrics extraction failed; keeping log-derived metrics.json. See %s\n' "$metrics_extract_log" >&2
fi

run_rel="runs/glm53-arch-ladder/${run_id}"
if [[ "${GLM53_ARCH_LADDER_SKIP_PUBLISH:-0}" != "1" ]]; then
  publish_tmp="$(mktemp)"
  "${repo_root}/experiments/glm53_flash_sft_ladder/publish.sh" \
    --parallel "${GLM53_ARCH_LADDER_PUBLISH_PARALLEL:-2}" \
    "$host_dump_folder" "$run_rel" | tee "$publish_tmp"
  cp "$publish_tmp" "${host_dump_folder}/publish.txt"
  rm -f "$publish_tmp"
fi
