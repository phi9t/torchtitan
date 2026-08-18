#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_REL="experiments/modded_nanogpt_b200/setup_flash_attention.sh"

FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3.post1}"
CUDA_WHEEL_VERSION="${CUDA_WHEEL_VERSION:-13.2.86}"
MAX_JOBS="${MAX_JOBS:-8}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
FLASH_ATTN_SOURCE_BUILD="${FLASH_ATTN_SOURCE_BUILD:-1}"
FLASH_ATTN_FORCE_REBUILD="${FLASH_ATTN_FORCE_REBUILD:-0}"
EXPECTED_TORCH_VERSION="${EXPECTED_TORCH_VERSION:-2.13.0+cu132}"
EXPECTED_CUDA_RUNTIME="${EXPECTED_CUDA_RUNTIME:-13.2}"
EXPECTED_TRITON_VERSION="${EXPECTED_TRITON_VERSION:-3.7.1}"
export FLASH_ATTN_VERSION CUDA_WHEEL_VERSION
export EXPECTED_TORCH_VERSION EXPECTED_CUDA_RUNTIME EXPECTED_TRITON_VERSION

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
export MAX_JOBS
export TORCH_CUDA_ARCH_LIST

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

[[ -x "${CC}" ]] || die "C compiler not found or not executable: ${CC}"
[[ -x "${CXX}" ]] || die "C++ compiler not found or not executable: ${CXX}"
[[ -d "${CUDA_HOME}" ]] || die "CUDA_HOME does not exist: ${CUDA_HOME}"
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || die "nvcc not found at ${CUDA_HOME}/bin/nvcc"
[[ -f "${CUDA_HOME}/lib/libcudart.so.13" ]] || die "missing ${CUDA_HOME}/lib/libcudart.so.13"

flash_attn_health_probe() {
  python - <<'PY'
import importlib.metadata as metadata
import os
import sys

import torch
import triton
from flash_attn.flash_attn_interface import flash_attn_varlen_func

expected = {
    "nvidia-cuda-nvcc": os.environ["CUDA_WHEEL_VERSION"],
    "nvidia-cuda-crt": os.environ["CUDA_WHEEL_VERSION"],
    "nvidia-cuda-cccl": os.environ["CUDA_WHEEL_VERSION"],
    "nvidia-nvvm": os.environ["CUDA_WHEEL_VERSION"],
    "flash-attn": os.environ["FLASH_ATTN_VERSION"],
}
for package, version in expected.items():
    actual = metadata.version(package)
    if actual != version:
        raise SystemExit(f"{package} version mismatch: expected {version}, found {actual}")

if torch.__version__ != os.environ["EXPECTED_TORCH_VERSION"]:
    raise SystemExit(
        f"torch version mismatch: expected {os.environ['EXPECTED_TORCH_VERSION']}, found {torch.__version__}"
    )
if torch.version.cuda != os.environ["EXPECTED_CUDA_RUNTIME"]:
    raise SystemExit(
        f"CUDA runtime mismatch: expected {os.environ['EXPECTED_CUDA_RUNTIME']}, found {torch.version.cuda}"
    )
if triton.__version__ != os.environ["EXPECTED_TRITON_VERSION"]:
    raise SystemExit(
        f"triton version mismatch: expected {os.environ['EXPECTED_TRITON_VERSION']}, found {triton.__version__}"
    )
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
if torch.cuda.get_device_capability(0) < (10, 0):
    raise SystemExit(f"expected Blackwell/B200 capability >= (10, 0), found {torch.cuda.get_device_capability(0)}")

cu = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
q = torch.randn(256, 6, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
y = flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q=cu,
    cu_seqlens_k=cu,
    max_seqlen_q=128,
    max_seqlen_k=128,
    causal=True,
    softmax_scale=0.1,
    window_size=(64, 0),
)
torch.cuda.synchronize()
if y.shape != q.shape:
    raise SystemExit(f"wrong FA2 smoke output shape: expected {q.shape}, found {y.shape}")
if y.dtype is not torch.bfloat16:
    raise SystemExit(f"wrong FA2 smoke output dtype: expected torch.bfloat16, found {y.dtype}")

print(
    "flash-attn ok:",
    {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "flash_attn": metadata.version("flash-attn"),
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "output_shape": tuple(y.shape),
        "output_dtype": str(y.dtype),
    },
)
PY
}

if [[ "${FLASH_ATTN_FORCE_REBUILD}" != "1" ]] && flash_attn_health_probe; then
  printf 'flash-attn already healthy; skipping rebuild\n'
  exit 0
fi

python -m pip install \
  --break-system-packages \
  --no-deps \
  --force-reinstall \
  "nvidia-cuda-nvcc==${CUDA_WHEEL_VERSION}" \
  "nvidia-cuda-crt==${CUDA_WHEEL_VERSION}" \
  "nvidia-cuda-cccl==${CUDA_WHEEL_VERSION}" \
  "nvidia-nvvm==${CUDA_WHEEL_VERSION}"

# flash-attn's extension build links with -lcudart. The CUDA 13 wheel provides
# libcudart.so.13, so keep the unversioned soname bridge in the CUDA synth root.
ln -sf libcudart.so.13 "${CUDA_HOME}/lib/libcudart.so"

flash_attn_install=(
  python -m pip install
  --break-system-packages \
  --no-deps \
  --no-build-isolation \
  --force-reinstall \
)
if [[ "${FLASH_ATTN_SOURCE_BUILD}" == "1" ]]; then
  flash_attn_install+=(--no-binary flash-attn --no-cache-dir)
fi
flash_attn_install+=("flash-attn==${FLASH_ATTN_VERSION}")
"${flash_attn_install[@]}"

flash_attn_health_probe
