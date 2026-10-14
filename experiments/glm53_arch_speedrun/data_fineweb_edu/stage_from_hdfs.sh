#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

# shellcheck source=experiments/glm53_flash_sft_ladder/lib/waystone.sh
source "${repo_root}/experiments/glm53_flash_sft_ladder/lib/waystone.sh"

usage() {
  cat >&2 <<'USAGE'
Usage: stage_from_hdfs.sh <hdfs-uri> <local-dir>

Downloads every file listed in MANIFEST.jsonl from an HDFS dataset directory
with size and MD5 verification, then writes MANIFEST.jsonl and _COMPLETE locally.
USAGE
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

hdfs_uri="${1%/}"
local_dir="${2%/}"
mkdir -p "${local_dir}"

manifest_tmp="$(mktemp)"
"${WAYSTONE_BIN}" --error-format json \
  --command-timeout-secs "${WAYSTONE_METADATA_TIMEOUT_SECS}" \
  cat "$(ws_join_uri "${hdfs_uri}" "MANIFEST.jsonl")" > "${manifest_tmp}"
manifest_md5="$(ws_md5_file "${manifest_tmp}")"
file_count="$(grep -c . "${manifest_tmp}")"

while IFS= read -r line; do
  [[ -n "${line}" ]] || continue
  rel="$(jq -r '.file' <<<"${line}")"
  size="$(jq -r '.size' <<<"${line}")"
  md5="$(jq -r '.md5' <<<"${line}")"
  hdfs_file="$(ws_join_uri "${hdfs_uri}" "${rel}")"
  local_file="${local_dir}/${rel}"
  if [[ -f "${local_file}" ]] \
    && [[ "$(ws_file_size_bytes "${local_file}")" == "${size}" ]] \
    && [[ "$(ws_md5_file "${local_file}")" == "${md5}" ]]; then
    continue
  fi
  WS_EXPECTED_SIZE="${size}" WS_EXPECTED_MD5="${md5}" \
    ws_get_file "${hdfs_file}" "${local_file}"
done < "${manifest_tmp}"

cp "${manifest_tmp}" "${local_dir}/MANIFEST.jsonl"
rm -f "${manifest_tmp}"
"${WAYSTONE_BIN}" --error-format json \
  --command-timeout-secs "${WAYSTONE_METADATA_TIMEOUT_SECS}" \
  cat "$(ws_join_uri "${hdfs_uri}" "_COMPLETE")" > "${local_dir}/_COMPLETE"
jq -n \
  --arg source_uri "${hdfs_uri}" \
  --arg manifest_md5 "${manifest_md5}" \
  --arg staged_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson file_count "${file_count}" \
  '{source_uri:$source_uri, manifest_md5:$manifest_md5, file_count:$file_count, staged_at:$staged_at}' \
  > "${local_dir}/.staged.json"

printf 'LOCAL_DIR=%s\n' "${local_dir}"

