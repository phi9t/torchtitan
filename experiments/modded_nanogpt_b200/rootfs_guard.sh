#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

require_modded_nanogpt_rootfs() {
  local wrapper="${1:?wrapper path is required}"
  if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
    printf 'error: must run through %s; TORCHTITAN_IN_ROOTFS=1 is missing\n' "${wrapper}" >&2
    return 21
  fi
  if [[ "$(pwd -P)" != "/workspace/torchtitan" ]]; then
    printf 'error: expected rootfs workspace /workspace/torchtitan, found %s\n' "$(pwd -P)" >&2
    return 21
  fi
  if [[ ! -f "/workspace/torchtitan/scripts/rootfs/enter_rootfs.sh" ]]; then
    printf 'error: rootfs workspace sentinel is missing: /workspace/torchtitan/scripts/rootfs/enter_rootfs.sh\n' >&2
    return 21
  fi
  return 0
}

select_modded_nanogpt_python() {
  local venv="${MODDED_NANOGPT_RUNTIME_VENV:-/project/venvs/b200-runtime}"
  local managed_python="${venv}/bin/python"
  if [[ -x "${managed_python}" ]]; then
    printf '%s\n' "${managed_python}"
    return 0
  fi
  if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
    printf '%s\n' "${PYTHON}"
    return 0
  fi
  command -v python
}
