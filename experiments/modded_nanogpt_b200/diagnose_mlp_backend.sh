#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_REL="experiments/modded_nanogpt_b200/diagnose_mlp_backend.sh"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "${SCRIPT_REL}" "$@"
fi

source "${SCRIPT_DIR}/rootfs_guard.sh"
require_modded_nanogpt_rootfs "${SCRIPT_REL}"

cd "${REPO_ROOT}"

export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export CUDA_HOME="${CUDA_HOME:-/opt/cuda-synth}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${REPO_ROOT}/experiments/modded_nanogpt_b200/results/.torchinductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${REPO_ROOT}/experiments/modded_nanogpt_b200/results/.triton_cache}"

mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

exec python experiments/modded_nanogpt_b200/diagnose_mlp_backend.py "$@"
