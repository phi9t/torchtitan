#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Shared canonical runtime environment for TorchTitan bwrap rootfs entrypoints.

ROOTFS_RUNTIME_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOTFS_RUNTIME_REPO_ROOT="$(cd -- "${ROOTFS_RUNTIME_SCRIPT_DIR}/../.." && pwd)"
ROOTFS_RUNTIME_ENV_NAME="${TORCHTITAN_ROOTFS_ENV:-modded_nanogpt_b200}"
ROOTFS_RUNTIME_STORE_ID="${TORCHTITAN_ROOTFS_STORE_ID:-legacy-rootfs}"
ROOTFS_RUNTIME_PROJECT="/workspace/torchtitan"
ROOTFS_RUNTIME_CONFIG="${ROOTFS_RUNTIME_PROJECT}/experiments/modded_nanogpt_b200/runtime"
ROOTFS_RUNTIME_VENV="/project/venvs/b200-runtime"
ROOTFS_RUNTIME_PATH="${ROOTFS_RUNTIME_VENV}/bin:/project/mise/data/shims:/usr/local/cuda/bin:/opt/cuda-synth/bin:/usr/local/bin:/usr/bin:/bin"
ROOTFS_RUNTIME_STATE_ROOT="${TORCHTITAN_ROOTFS_HOST_STATE:-${ROOTFS_RUNTIME_REPO_ROOT}/.cache/torchtitan-rootfs/${ROOTFS_RUNTIME_ENV_NAME}}"
ROOTFS_RUNTIME_STATE_DIRS=(home xdg-cache uv-cache pip-cache mise venvs wheels downloads scratch tmp logs)

rootfs_runtime_die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

rootfs_runtime_json_escape() {
  python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$1"
}

rootfs_runtime_resolve_state_root() {
  local requested="$1"
  if [[ -z "${requested}" ]]; then
    rootfs_runtime_die "runtime state root is empty"
  fi

  local parent basename
  parent="$(dirname -- "${requested}")"
  basename="${requested##*/}"
  if [[ -z "${basename}" || "${basename}" == "." || "${basename}" == ".." ]]; then
    rootfs_runtime_die "runtime state root has invalid basename: ${requested}"
  fi
  if [[ ! -d "${parent}" ]]; then
    mkdir -p "${parent}"
  fi

  local canonical_parent
  canonical_parent="$(cd -P -- "${parent}" && pwd)" \
    || rootfs_runtime_die "cannot resolve runtime state parent: ${parent}"
  local canonical="${canonical_parent}/${basename}"
  case "${canonical}/" in
    "${ROOTFS_RUNTIME_REPO_ROOT}/.cache/"*) ;;
    *) rootfs_runtime_die "runtime state root must be under ${ROOTFS_RUNTIME_REPO_ROOT}/.cache: ${canonical}" ;;
  esac
  if [[ -L "${canonical}" ]]; then
    rootfs_runtime_die "runtime state root must not be a symlink: ${canonical}"
  fi
  printf '%s\n' "${canonical}"
}

rootfs_runtime_init() {
  ROOTFS_RUNTIME_STATE_ROOT="$(rootfs_runtime_resolve_state_root "${ROOTFS_RUNTIME_STATE_ROOT}")"
  for dir in "${ROOTFS_RUNTIME_STATE_DIRS[@]}"; do
    mkdir -p "${ROOTFS_RUNTIME_STATE_ROOT}/${dir}"
  done
}

rootfs_runtime_set_store_id() {
  local store_id="$1"
  if [[ -z "${store_id}" ]]; then
    rootfs_runtime_die "rootfs runtime store ID is empty"
  fi
  ROOTFS_RUNTIME_STORE_ID="${store_id}"
}

rootfs_runtime_path_for() {
  local key="$1"
  case "${key}" in
    home) printf '/project/home\n' ;;
    xdg-cache) printf '/project/xdg-cache\n' ;;
    uv-cache) printf '/project/uv-cache\n' ;;
    pip-cache) printf '/project/pip-cache\n' ;;
    mise) printf '/project/mise\n' ;;
    venvs) printf '/project/venvs\n' ;;
    wheels) printf '/project/wheels\n' ;;
    downloads) printf '/project/downloads\n' ;;
    scratch) printf '/project/scratch\n' ;;
    tmp) printf '/project/tmp\n' ;;
    logs) printf '/project/logs\n' ;;
    *) rootfs_runtime_die "unknown runtime state directory: ${key}" ;;
  esac
}

rootfs_runtime_env_json() {
  python3 - "$ROOTFS_RUNTIME_STATE_ROOT" <<'PY'
import json
import sys

host_state = sys.argv[1]
workspace = "/workspace/torchtitan"
env = {
    "PATH": "/project/venvs/b200-runtime/bin:/project/mise/data/shims:/usr/local/cuda/bin:/opt/cuda-synth/bin:/usr/local/bin:/usr/bin:/bin",
    "CUDA_HOME": "/opt/cuda-synth",
    "CUDA_PATH": "/opt/cuda-synth",
    "TORCHTITAN_IN_ROOTFS": "1",
    "TORCHTITAN_ROOTFS_ENV": "modded_nanogpt_b200",
    "TORCHTITAN_ROOTFS_STORE_ID": "legacy-rootfs",
    "TORCHTITAN_ROOTFS_HOST_STATE": host_state,
    "TORCHTITAN_ROOTFS_PROJECT": workspace,
    "TORCHTITAN_ROOTFS_LOG_DIR": "/project/logs",
    "HOME": "/project/home",
    "XDG_CACHE_HOME": "/project/xdg-cache",
    "UV_CACHE_DIR": "/project/uv-cache",
    "PIP_CACHE_DIR": "/project/pip-cache",
    "MISE_DATA_DIR": "/project/mise/data",
    "MISE_CACHE_DIR": "/project/mise/cache",
    "MISE_CONFIG_DIR": f"{workspace}/experiments/modded_nanogpt_b200/runtime",
    "PYTHON": "/project/venvs/b200-runtime/bin/python",
    "TMPDIR": "/project/tmp",
    "TEMP": "/project/tmp",
    "TMP": "/project/tmp",
    "HF_HOME": f"{workspace}/.cache/huggingface",
    "HF_HUB_CACHE": f"{workspace}/.cache/huggingface/hub",
    "TORCH_HOME": f"{workspace}/.cache/torch",
    "MPLCONFIGDIR": "/project/xdg-cache/matplotlib",
}
print(json.dumps(env, sort_keys=True))
PY
}

rootfs_runtime_add_bwrap_binds() {
  local -n target_args="$1"
  for dir in "${ROOTFS_RUNTIME_STATE_DIRS[@]}"; do
    target_args+=(--bind "${ROOTFS_RUNTIME_STATE_ROOT}/${dir}" "$(rootfs_runtime_path_for "${dir}")")
  done
}

rootfs_runtime_add_bwrap_env() {
  local -n target_args="$1"
  target_args+=(
    --setenv PATH "${ROOTFS_RUNTIME_PATH}"
    --setenv CUDA_HOME /opt/cuda-synth
    --setenv CUDA_PATH /opt/cuda-synth
    --setenv TORCHTITAN_IN_ROOTFS 1
    --setenv TORCHTITAN_ROOTFS_ENV "${ROOTFS_RUNTIME_ENV_NAME}"
    --setenv TORCHTITAN_ROOTFS_STORE_ID "${ROOTFS_RUNTIME_STORE_ID}"
    --setenv TORCHTITAN_ROOTFS_HOST_STATE "${ROOTFS_RUNTIME_STATE_ROOT}"
    --setenv TORCHTITAN_ROOTFS_PROJECT "${ROOTFS_RUNTIME_PROJECT}"
    --setenv TORCHTITAN_ROOTFS_LOG_DIR /project/logs
    --setenv HOME /project/home
    --setenv XDG_CACHE_HOME /project/xdg-cache
    --setenv UV_CACHE_DIR /project/uv-cache
    --setenv PIP_CACHE_DIR /project/pip-cache
    --setenv MISE_DATA_DIR /project/mise/data
    --setenv MISE_CACHE_DIR /project/mise/cache
    --setenv MISE_CONFIG_DIR "${ROOTFS_RUNTIME_CONFIG}"
    --setenv PYTHON "${ROOTFS_RUNTIME_VENV}/bin/python"
    --setenv TMPDIR /project/tmp
    --setenv TEMP /project/tmp
    --setenv TMP /project/tmp
    --setenv HF_HOME "${ROOTFS_RUNTIME_PROJECT}/.cache/huggingface"
    --setenv HF_HUB_CACHE "${ROOTFS_RUNTIME_PROJECT}/.cache/huggingface/hub"
    --setenv TORCH_HOME "${ROOTFS_RUNTIME_PROJECT}/.cache/torch"
    --setenv MPLCONFIGDIR /project/xdg-cache/matplotlib
  )
}
