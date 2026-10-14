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
  run_arm.sh [B0|A1|A2|A3|A4|A5|A6|G]
  run_arm.sh run-cell --cell-json PATH --output-dir PATH

Runs one GLM-5.3 architecture-speedrun arm through TorchTitan. Configuration is
owned by environment variables with GLM53_SPEEDRUN_* names. The default is B0.
The run-cell mode is used by the queue scheduler and translates the generated
cell JSON into the same environment contract.
USAGE
}

run_cell=0
cell_json_path=""
cell_output_dir=""

case "${1:-}" in
  --help|-h)
    usage
    exit 0
    ;;
  run-cell)
    run_cell=1
    shift
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --cell-json)
          [[ $# -ge 2 ]] || { usage; exit 2; }
          cell_json_path="$2"
          shift 2
          ;;
        --output-dir)
          [[ $# -ge 2 ]] || { usage; exit 2; }
          cell_output_dir="$2"
          shift 2
          ;;
        *)
          printf 'unknown run-cell argument: %s\n' "$1" >&2
          usage
          exit 2
          ;;
      esac
    done
    [[ -n "$cell_json_path" && -f "$cell_json_path" ]] || {
      printf 'run-cell requires --cell-json PATH\n' >&2
      exit 2
    }
    [[ -n "$cell_output_dir" ]] || {
      printf 'run-cell requires --output-dir PATH\n' >&2
      exit 2
    }
    ;;
esac

env_name_for_override() {
  case "$1" in
    debug.seed) printf 'GLM53_SPEEDRUN_SEED\n' ;;
    training.seq_len) printf 'GLM53_SPEEDRUN_SEQ_LEN\n' ;;
    training.token_budget) printf 'GLM53_SPEEDRUN_TOKEN_BUDGET\n' ;;
    training.tokens_per_step) printf 'GLM53_SPEEDRUN_TOKENS_PER_STEP\n' ;;
    model.elements.rope_enabled) printf 'GLM53_SPEEDRUN_ROPE_ENABLED\n' ;;
    model.elements.mhc_enabled) printf 'GLM53_SPEEDRUN_MHC_ENABLED\n' ;;
    model.elements.mhc_streams) printf 'GLM53_SPEEDRUN_MHC_STREAMS\n' ;;
    model.elements.mhc_sinkhorn_iters) printf 'GLM53_SPEEDRUN_MHC_SINKHORN_ITERS\n' ;;
    model.elements.kda_enabled) printf 'GLM53_SPEEDRUN_KDA_ENABLED\n' ;;
    model.elements.kda_hybrid_3_to_1) printf 'GLM53_SPEEDRUN_KDA_HYBRID_3_TO_1\n' ;;
    model.elements.sparse_attention_enabled) printf 'GLM53_SPEEDRUN_SPARSE_ATTENTION_ENABLED\n' ;;
    model.elements.sparse_top_k) printf 'GLM53_SPEEDRUN_SPARSE_TOP_K\n' ;;
    model.elements.index_pool_size) printf 'GLM53_SPEEDRUN_INDEX_POOL_SIZE\n' ;;
    model.elements.indexer_aux_loss_weight) printf 'GLM53_SPEEDRUN_INDEXER_AUX_LOSS_WEIGHT\n' ;;
    model.elements.mtp_enabled) printf 'GLM53_SPEEDRUN_MTP_ENABLED\n' ;;
    model.elements.mtp_aux_loss_weight) printf 'GLM53_SPEEDRUN_MTP_AUX_LOSS_WEIGHT\n' ;;
    *) return 1 ;;
  esac
}

export_cell_overrides() {
  local key value env_name
  while IFS=$'\t' read -r key value; do
    [[ -n "$key" ]] || continue
    if env_name="$(env_name_for_override "$key")"; then
      printf -v "$env_name" '%s' "$value"
      export "$env_name"
    fi
  done < <(jq -r '.config_overrides // {} | to_entries[] | [.key, (.value|tostring)] | @tsv' "$cell_json_path")
}

if (( run_cell == 1 )); then
  export_cell_overrides
  arm="$(jq -r '.arm' "$cell_json_path")"
  export GLM53_SPEEDRUN_CELL_ID="$(jq -r '.cell_id' "$cell_json_path")"
  export GLM53_SPEEDRUN_MODE="ablation_$(jq -r '.slice' "$cell_json_path")"
  export GLM53_SPEEDRUN_CLAIM_LABEL="$(jq -r '.claim_label // "representative_small+replicated_eval"' "$cell_json_path")"
  export GLM53_SPEEDRUN_EVIDENCE_TIER="$(jq -r '.evidence_tier // "ablation_matrix"' "$cell_json_path")"
  export GLM53_SPEEDRUN_ENVIRONMENT_CLASS="$(jq -r '.resources.environment_class // "shared_gpu"' "$cell_json_path")"
  export GLM53_SPEEDRUN_GPU_MEMORY_CAP_GIB="$(jq -r '.resources.gpu_memory_cap_gib // 24' "$cell_json_path")"
  export TORCHTITAN_RUN_ID="${TORCHTITAN_RUN_ID:-${GLM53_ARCH_RUN_ID:-$(jq -r '.cell_id' "$cell_json_path")}}"
  export TORCHTITAN_ATTEMPT_ID="${TORCHTITAN_ATTEMPT_ID:-${GLM53_ARCH_ATTEMPT_ID:-attempt-001}}"
  export GLM53_SPEEDRUN_HOST_DUMP_FOLDER="$cell_output_dir"
  case "$cell_output_dir" in
    "$repo_root"/*)
      export GLM53_SPEEDRUN_ROOTFS_DUMP_FOLDER="${cell_output_dir#"$repo_root"/}"
      ;;
    /*)
      printf 'run-cell --output-dir must be under the repo root, got %s\n' "$cell_output_dir" >&2
      exit 2
      ;;
    *)
      export GLM53_SPEEDRUN_ROOTFS_DUMP_FOLDER="$cell_output_dir"
      ;;
  esac
  export GLM53_SPEEDRUN_SKIP_PUBLISH=1
else
  arm="${1:-${GLM53_SPEEDRUN_ARM:-B0}}"
fi
export GLM53_SPEEDRUN_ARM="$arm"

data_rev="${GLM53_SPEEDRUN_DATA_REV:-ecbb586-fineweb10b-gpt2-binonly-v1}"
data_rel="datasets/fineweb10B-gpt2/${data_rev}"
staged_data_rel="experiments/glm53_arch_speedrun/results/data/fineweb10B-gpt2/${data_rev}"
host_staged_data_dir="${GLM53_SPEEDRUN_HOST_DATA_DIR:-${repo_root}/${staged_data_rel}}"
rootfs_staged_data_dir="${GLM53_SPEEDRUN_ROOTFS_DATA_DIR:-${staged_data_rel}}"
source_data_dir="${GLM53_SPEEDRUN_SOURCE_DATA_DIR:-${host_staged_data_dir}}"
hdfs_data_uri="${GLM53_SPEEDRUN_DATA_HDFS_URI:-}"
export GLM53_SPEEDRUN_DATA_DIR="$rootfs_staged_data_dir"
staged_from_hdfs=0
data_manifest_md5=""
data_file_count=""

publish_dataset_if_needed() {
  ensure_waystone_loaded
  if [[ -z "$hdfs_data_uri" ]]; then
    hdfs_data_uri="$(ws_child_uri "$data_rel")"
  fi
  if ws_exists "$(ws_join_uri "$hdfs_data_uri" "_COMPLETE")"; then
    printf 'DATA_HDFS_URI=%s\n' "$hdfs_data_uri"
    return 0
  fi
  if [[ ! -d "$source_data_dir" ]]; then
    printf 'FineWeb source data not found: %s\n' "$source_data_dir" >&2
    printf 'Set GLM53_SPEEDRUN_SOURCE_DATA_DIR to a clean bin-only shard directory or run data staging first.\n' >&2
    exit 1
  fi
  "${repo_root}/experiments/glm53_flash_sft_ladder/publish.sh" \
    --parallel "${GLM53_SPEEDRUN_PUBLISH_PARALLEL:-2}" \
    "$source_data_dir" "$data_rel"
}

stage_dataset_from_hdfs() {
  local manifest_tmp rel size md5 hdfs_file local_file
  ensure_waystone_loaded
  if [[ -z "$hdfs_data_uri" ]]; then
    hdfs_data_uri="$(ws_child_uri "$data_rel")"
  fi
  mkdir -p "$host_staged_data_dir"
  manifest_tmp="$(mktemp)"
  "${WAYSTONE_BIN}" --error-format json \
    --command-timeout-secs "$WAYSTONE_METADATA_TIMEOUT_SECS" \
    cat "$(ws_join_uri "$hdfs_data_uri" "MANIFEST.jsonl")" > "$manifest_tmp"
  while IFS= read -r line; do
    rel="$(jq -r '.file' <<<"$line")"
    size="$(jq -r '.size' <<<"$line")"
    md5="$(jq -r '.md5' <<<"$line")"
    hdfs_file="$(ws_join_uri "$hdfs_data_uri" "$rel")"
    local_file="${host_staged_data_dir}/${rel}"
    if [[ -f "$local_file" ]] && [[ "$(ws_file_size_bytes "$local_file")" == "$size" ]] && [[ "$(ws_md5_file "$local_file")" == "$md5" ]]; then
      continue
    fi
    WS_EXPECTED_SIZE="$size" WS_EXPECTED_MD5="$md5" ws_get_file "$hdfs_file" "$local_file"
  done < "$manifest_tmp"
  cp "$manifest_tmp" "${host_staged_data_dir}/MANIFEST.jsonl"
  rm -f "$manifest_tmp"
  staged_from_hdfs=1
}

write_staged_sidecar() {
  ensure_waystone_loaded
  data_manifest_md5="$(ws_md5_file "${host_staged_data_dir}/MANIFEST.jsonl")"
  data_file_count="$(wc -l < "${host_staged_data_dir}/MANIFEST.jsonl" | tr -d '[:space:]')"
  jq -n \
    --arg source_uri "$hdfs_data_uri" \
    --arg manifest_md5 "$data_manifest_md5" \
    --arg staged_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson file_count "$data_file_count" \
    '{source_uri:$source_uri, manifest_md5:$manifest_md5, file_count:$file_count, staged_at:$staged_at}' \
    > "${host_staged_data_dir}/.staged.json"
}

verify_hdfs_input_cache() {
  if [[ "${GLM53_SPEEDRUN_HDFS_STRICT:-1}" != "1" ]]; then
    return 0
  fi
  case "$hdfs_data_uri" in
    hdfs://*) ;;
    *)
      printf 'HDFS strict input mode requires GLM53_SPEEDRUN_DATA_HDFS_URI or a staged HDFS cache, got: %s\n' "$hdfs_data_uri" >&2
      exit 1
      ;;
  esac
  if (( staged_from_hdfs == 1 )); then
    write_staged_sidecar
  fi
  if [[ ! -f "${host_staged_data_dir}/.staged.json" ]]; then
    printf 'HDFS strict input mode requires %s/.staged.json\n' "$host_staged_data_dir" >&2
    exit 1
  fi
  "${repo_root}/experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh" \
    --quick "$host_staged_data_dir"
  ensure_waystone_loaded
  data_manifest_md5="$(ws_md5_file "${host_staged_data_dir}/MANIFEST.jsonl")"
  data_file_count="$(wc -l < "${host_staged_data_dir}/MANIFEST.jsonl" | tr -d '[:space:]')"
}

use_existing_hdfs_cache=0
if [[ -f "${host_staged_data_dir}/.staged.json" && -f "${host_staged_data_dir}/MANIFEST.jsonl" ]]; then
  staged_source_uri="$(jq -r '.source_uri' "${host_staged_data_dir}/.staged.json")"
  if [[ -z "$hdfs_data_uri" ]]; then
    hdfs_data_uri="$staged_source_uri"
    use_existing_hdfs_cache=1
  elif [[ "$hdfs_data_uri" == "$staged_source_uri" ]]; then
    use_existing_hdfs_cache=1
  fi
fi

if (( use_existing_hdfs_cache == 1 )); then
  :
elif [[ "${GLM53_SPEEDRUN_SKIP_PUBLISH:-0}" == "1" && -f "${host_staged_data_dir}/MANIFEST.jsonl" ]]; then
  if [[ -z "$hdfs_data_uri" && -f "${host_staged_data_dir}/.staged.json" ]]; then
    hdfs_data_uri="$(jq -r '.source_uri' "${host_staged_data_dir}/.staged.json")"
  fi
  : "${hdfs_data_uri:=local:${staged_data_rel}}"
else
  publish_dataset_if_needed
  stage_dataset_from_hdfs
fi
verify_hdfs_input_cache

run_id="${TORCHTITAN_RUN_ID:-glm53-speedrun-$(date -u +%Y%m%dT%H%M%SZ)-${arm,,}-seed${GLM53_SPEEDRUN_SEED:-0}-seq${GLM53_SPEEDRUN_SEQ_LEN:-1024}}"
attempt_id="${TORCHTITAN_ATTEMPT_ID:-attempt-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
export TORCHTITAN_RUN_ID="$run_id" TORCHTITAN_ATTEMPT_ID="$attempt_id"

mode="${GLM53_SPEEDRUN_MODE:-science}"
claim_label="${GLM53_SPEEDRUN_CLAIM_LABEL:-smoke}"
evidence_tier="${GLM53_SPEEDRUN_EVIDENCE_TIER:-tier0}"
environment_class="${GLM53_SPEEDRUN_ENVIRONMENT_CLASS:-shared_gpu}"
ngpu="${NGPU:-1}"

dump_rel="experiments/glm53_arch_speedrun/results/${run_id}"
host_dump_folder="${GLM53_SPEEDRUN_HOST_DUMP_FOLDER:-${repo_root}/${dump_rel}}"
rootfs_dump_folder="${GLM53_SPEEDRUN_ROOTFS_DUMP_FOLDER:-${dump_rel}}"
log_dir="${repo_root}/experiments/glm53_arch_speedrun/logs"
mkdir -p "$host_dump_folder" "$log_dir"
export GLM53_SPEEDRUN_DUMP_FOLDER="$rootfs_dump_folder"

metadata_file="${host_dump_folder}/attempt.json"
cat > "$metadata_file" <<EOF
{
  "lane": "glm53_arch_speedrun",
  "mode": "${mode}",
  "arm": "${arm}",
  "cell_id": "${GLM53_SPEEDRUN_CELL_ID:-}",
  "claim_label": "${claim_label}",
  "evidence_tier": "${evidence_tier}",
  "run_id": "${run_id}",
  "attempt_id": "${attempt_id}",
  "environment_class": "${environment_class}",
  "seq_len": ${GLM53_SPEEDRUN_SEQ_LEN:-1024},
  "local_batch_size": ${GLM53_SPEEDRUN_LOCAL_BATCH_SIZE:-0},
  "tokens_per_step": ${GLM53_SPEEDRUN_TOKENS_PER_STEP:-262144},
  "token_budget": ${GLM53_SPEEDRUN_TOKEN_BUDGET:-268435456},
  "gpu_memory_cap_gib": ${GLM53_SPEEDRUN_GPU_MEMORY_CAP_GIB:-24},
  "data_hdfs_uri": "${hdfs_data_uri}",
  "data_manifest_md5": "${data_manifest_md5}",
  "data_file_count": ${data_file_count:-0},
  "data_host_dir": "${host_staged_data_dir}",
  "data_rootfs_dir": "${rootfs_staged_data_dir}",
  "ngpu": ${ngpu}
}
EOF

launch_cmd=(
  "${repo_root}/scripts/rootfs/enter_rootfs.sh"
  --
  bash
  -lc
  "unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH=\"\$PWD:\${PYTHONPATH:-}\" && LOG_RANK=\${LOG_RANK:-0} NGPU=${ngpu} MODULE=torchtitan.experiments.glm53_arch_speedrun.config_registry CONFIG=glm53_arch_speedrun_b0 ./run_train.sh"
)

printf '%q ' TORCHTITAN_ROOTFS_ENV=glm53_train "${launch_cmd[@]}" > "${host_dump_folder}/launch_command.txt"
printf '\n' >> "${host_dump_folder}/launch_command.txt"

log_file="${log_dir}/${run_id}.log"
(
  cd "$repo_root"
  TORCHTITAN_ROOTFS_ENV=glm53_train "${launch_cmd[@]}"
) 2>&1 | tee "$log_file"

cp "$log_file" "${host_dump_folder}/train.log"
cp "${host_staged_data_dir}/MANIFEST.jsonl" "${host_dump_folder}/DATA_MANIFEST.jsonl"

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
    --arg arm "$arm" \
    --arg environment_class "$environment_class" \
    --argjson seq_len "${GLM53_SPEEDRUN_SEQ_LEN:-1024}" \
    --argjson tokens_per_step "${GLM53_SPEEDRUN_TOKENS_PER_STEP:-262144}" \
    --argjson token_budget "${GLM53_SPEEDRUN_TOKEN_BUDGET:-268435456}" \
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
      arm:$arm,
      environment_class:$environment_class,
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

run_rel="runs/glm53-arch-speedrun/${run_id}"
if [[ "${GLM53_SPEEDRUN_SKIP_PUBLISH:-0}" != "1" ]]; then
  "${repo_root}/experiments/glm53_flash_sft_ladder/publish.sh" \
    --parallel "${GLM53_SPEEDRUN_PUBLISH_PARALLEL:-2}" \
    "$host_dump_folder" "$run_rel" | tee "${host_dump_folder}/publish.txt"
fi
