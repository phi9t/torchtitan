#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# variant/ is generated from pinned upstream + variant.patch (not committed).
if [[ ! -f "${SCRIPT_DIR}/variant/train_gpt.py" ]]; then
  "${SCRIPT_DIR}/materialize_variant.sh"
fi
SCRIPT_REL="experiments/attn_geometry_speedrun/run_arm.sh"
DATA_REV="${ATTN_DATA_REV:-ecbb586-fineweb10b-gpt2-binonly-v1}"
DATA_REL="experiments/glm53_arch_speedrun/results/data/fineweb10B-gpt2/${DATA_REV}"
HOST_DATA_DIR="${ATTN_HOST_DATA_DIR:-${REPO_ROOT}/${DATA_REL}}"
ROOTFS_DATA_DIR="${ATTN_ROOTFS_DATA_DIR:-/workspace/torchtitan/${DATA_REL}}"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  HDFS_STRICT_OUTPUT="$("${REPO_ROOT}/experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh" \
    --quick "${HOST_DATA_DIR}")"
  DATA_HDFS_URI="$(jq -er '.source_uri' "${HOST_DATA_DIR}/.staged.json")"
  DATA_MANIFEST_MD5="$(jq -er '.manifest_md5' "${HOST_DATA_DIR}/.staged.json")"
  export ATTN_DATA_HDFS_URI="${DATA_HDFS_URI}"
  export ATTN_DATA_MANIFEST_MD5="${DATA_MANIFEST_MD5}"
  export ATTN_DATA_HOST_DIR="${HOST_DATA_DIR}"
  export ATTN_DATA_ROOTFS_DIR="${ROOTFS_DATA_DIR}"
  export ATTN_HDFS_STRICT_OUTPUT="${HDFS_STRICT_OUTPUT}"
  export DATA_PATH="${DATA_PATH:-${ROOTFS_DATA_DIR}}"
  export ATTN_TRAIN_FILES="${ATTN_TRAIN_FILES:-${ROOTFS_DATA_DIR}/fineweb_train_*.bin}"
  export ATTN_VAL_FILES="${ATTN_VAL_FILES:-${ROOTFS_DATA_DIR}/fineweb_val_*.bin}"
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "${SCRIPT_REL}" "$@"
fi

cd "${REPO_ROOT}"
exec python experiments/attn_geometry_speedrun/run_attempt.py "$@"
