#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

# shellcheck source=experiments/glm53_flash_sft_ladder/lib/waystone.sh
source "${repo_root}/experiments/glm53_flash_sft_ladder/lib/waystone.sh"

usage() {
  cat >&2 <<'USAGE'
Usage: run_arm.sh [B0]

Runs one GLM-5.3 architecture-speedrun arm through TorchTitan. Configuration is
owned by environment variables with GLM53_SPEEDRUN_* names. The default is B0.
USAGE
}

case "${1:-}" in
  --help|-h)
    usage
    exit 0
    ;;
esac

arm="${1:-${GLM53_SPEEDRUN_ARM:-B0}}"
if [[ "$arm" != "B0" ]]; then
  printf 'S01 runner supports B0 only; got %s\n' "$arm" >&2
  exit 2
fi
export GLM53_SPEEDRUN_ARM="$arm"

data_rev="${GLM53_SPEEDRUN_DATA_REV:-ecbb586-fineweb10b-gpt2-binonly-v1}"
data_rel="datasets/fineweb10B-gpt2/${data_rev}"
hdfs_data_uri="$(ws_child_uri "$data_rel")"
staged_data_rel="experiments/glm53_arch_speedrun/results/data/fineweb10B-gpt2/${data_rev}"
host_staged_data_dir="${GLM53_SPEEDRUN_HOST_DATA_DIR:-${repo_root}/${staged_data_rel}}"
rootfs_staged_data_dir="${GLM53_SPEEDRUN_ROOTFS_DATA_DIR:-${staged_data_rel}}"
source_data_dir="${GLM53_SPEEDRUN_SOURCE_DATA_DIR:-${host_staged_data_dir}}"
export GLM53_SPEEDRUN_DATA_DIR="$rootfs_staged_data_dir"

publish_dataset_if_needed() {
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
}

publish_dataset_if_needed
stage_dataset_from_hdfs

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
  "claim_label": "${claim_label}",
  "evidence_tier": "${evidence_tier}",
  "run_id": "${run_id}",
  "attempt_id": "${attempt_id}",
  "environment_class": "${environment_class}",
  "data_hdfs_uri": "${hdfs_data_uri}",
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

run_rel="runs/glm53-arch-speedrun/${run_id}"
"${repo_root}/experiments/glm53_flash_sft_ladder/publish.sh" \
  --parallel "${GLM53_SPEEDRUN_PUBLISH_PARALLEL:-2}" \
  "$host_dump_folder" "$run_rel" | tee "${host_dump_folder}/publish.txt"
