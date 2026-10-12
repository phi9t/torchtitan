#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="$(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | awk -F, '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $2 " " $1}' \
            | sort -nr \
            | head -n 1 \
            | awk '{print $2}'
    )"
    export CUDA_VISIBLE_DEVICES
fi

if [ "${NGPU:-1}" != "1" ]; then
    echo "ERROR: DiT synthetic v1 supports only NGPU=1." >&2
    exit 1
fi

# Use fake_backend for the single-rank proof run. It still executes the CUDA
# training loop and DCP checkpointing path, while avoiding NCCL object
# collectives that are unnecessary for a one-rank experiment on this host.
NGPU=1 LOCAL_RANK=0 python3 -m torchtitan.train \
    --module dit \
    --config dit_debug_synthetic \
    --comm.mode=fake_backend \
    "$@"
