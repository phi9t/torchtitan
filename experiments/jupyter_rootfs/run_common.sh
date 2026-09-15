#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Shared configuration and helpers for the rootfs-hosted Jupyter workflow.
# This is the single source of truth for the repo root, the loopback port,
# and the log/state directory layout. Host-side orchestrators source it for
# path resolution; rootfs entrypoints source it for the same paths and the
# re-exec guard, mirroring experiments/countdown_search_distill/run_common.sh.

set -euo pipefail

JUPYTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${JUPYTER_DIR}/../.." && pwd)"

# Re-enter the bwrap rootfs for any Python/torch work. Host callers that only
# orchestrate (mint tokens, manage tmux, inspect files) do not call this; the
# server entrypoint does, so its Jupyter server and kernels run inside the
# sandbox. TORCHTITAN_ROOTFS_NETWORK=networked lets the loopback server bind.
jupyter_enter_rootfs_if_needed() {
  local script_path="$1"
  shift
  if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
    exec env TORCHTITAN_ROOTFS_NETWORK="${TORCHTITAN_ROOTFS_NETWORK:-networked}" \
      "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- \
      "experiments/jupyter_rootfs/${script_path}" "$@"
  fi
}

# Shared env and directory layout. Kept identical on the host and inside the
# rootfs so the token/pid/log paths resolve to the same repo-relative files.
jupyter_setup_env() {
  # Loopback only. The server never binds a routable interface.
  export JUPYTER_ROOTFS_HOST="${JUPYTER_ROOTFS_HOST:-127.0.0.1}"
  export JUPYTER_ROOTFS_PORT="${JUPYTER_ROOTFS_PORT:-${PORT:-8899}}"
  export JUPYTER_ROOTFS_SESSION="${JUPYTER_ROOTFS_SESSION:-${SESSION:-jupyter-rootfs}}"
  # The kernelspec registered inside the rootfs venv; verify.sh asserts against
  # it. Its argv pins .venv-rootfs/bin/python, which is why sys.prefix ends in
  # .venv-rootfs when a kernel launches.
  export JUPYTER_ROOTFS_KERNEL="${JUPYTER_ROOTFS_KERNEL:-torchtitan-rootfs}"

  # State lives under the git-ignored .cache tree. On the host these paths are
  # under the checkout; inside the rootfs the same checkout is mounted at
  # /workspace/torchtitan, so a relative path resolves to the same files.
  export JUPYTER_ROOTFS_STATE_DIR="${REPO_ROOT}/.cache/jupyter-rootfs"
  export JUPYTER_ROOTFS_LOG_DIR="${JUPYTER_ROOTFS_STATE_DIR}/logs"
  export JUPYTER_ROOTFS_TOKEN_FILE="${JUPYTER_ROOTFS_STATE_DIR}/token.txt"
  export JUPYTER_ROOTFS_URL_FILE="${JUPYTER_ROOTFS_STATE_DIR}/url.txt"
  export JUPYTER_ROOTFS_SERVER_LOG="${JUPYTER_ROOTFS_LOG_DIR}/server.log"
  export JUPYTER_ROOTFS_ROOT_DIR="${JUPYTER_ROOTFS_ROOT_DIR:-/workspace/torchtitan}"

  mkdir -p "${JUPYTER_ROOTFS_STATE_DIR}" "${JUPYTER_ROOTFS_LOG_DIR}"
}

# Loud failure per repo error discipline. Prefix with a marker the callers and
# tests can grep for.
jupyter_die() {
  printf '\033[1;31mjupyter-rootfs error: %s\033[0m\n' "$*" >&2
  exit 1
}
