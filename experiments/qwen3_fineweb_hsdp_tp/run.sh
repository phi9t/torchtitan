#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Offline launch wrapper for the hermetic Qwen3 FineWeb HSDP + TP experiment.
# Prefetch once online:  python experiments/qwen3_fineweb_hsdp_tp/prefetch_fineweb.py
# Then run this offline:  experiments/qwen3_fineweb_hsdp_tp/run.sh

set -ex

# Paths below are repo-root-relative; cd there regardless of invocation dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_FILE="experiments/qwen3_fineweb_hsdp_tp/data/fineweb_test/data.json"
if [ ! -f "${DATA_FILE}" ]; then
    echo "ERROR: ${DATA_FILE} not found." >&2
    echo "Run the one-time online prefetch first:" >&2
    echo "  python experiments/qwen3_fineweb_hsdp_tp/prefetch_fineweb.py" >&2
    exit 1
fi

# Force fully offline data/model access during training.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# datasets writes a small on-disk cache when materializing the local JSON
# dataset. Default HF_HOME to a repo-local, writable dir if the caller has not
# set one (the system default cache may not be writable).
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"

# torch.compile / Triton shell out to a C compiler to build the CUDA driver
# shim. Point it at the system gcc if the caller has not set one: a gcc that
# lacks /usr/include/x86_64-linux-gnu on its search path (e.g. a nix-profile
# gcc) cannot find the multiarch pyconfig.h that Python.h includes.
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

# dp_replicate(2) x dp_shard(2) x tp(2) = 8 = NGPU
NGPU=8 MODULE=qwen3 CONFIG=qwen3_debugmodel_fineweb ./run_train.sh \
    --parallelism.data_parallel_replicate_degree=2 \
    --parallelism.data_parallel_shard_degree=2 \
    --parallelism.tensor_parallel_degree=2 \
    "$@"
