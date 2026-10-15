#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -ne 1 ]]; then
  echo "usage: experiments/attn_geometry_speedrun/check_data_waystone.sh <local-staged-cache-dir>" >&2
  exit 2
fi

"${REPO_ROOT}/experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh" --quick "$1"
