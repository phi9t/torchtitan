#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Rootfs-side server entrypoint. run_server.sh invokes this inside a tmux
# session; the re-exec guard in run_common.sh re-enters the bwrap rootfs before
# any Jupyter work, so this asserts TORCHTITAN_IN_ROOTFS=1. It ensures the
# server dependencies are present in .venv-rootfs (installing only when
# missing), then execs `jupyter lab` bound to loopback with token-only auth.

set -euo pipefail

# shellcheck source=experiments/jupyter_rootfs/run_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_common.sh"

# Re-enter the rootfs if launched from the host. After this returns we are
# inside bwrap with TORCHTITAN_IN_ROOTFS=1.
jupyter_enter_rootfs_if_needed "server_entry.sh" "$@"

jupyter_setup_env

[[ "${TORCHTITAN_IN_ROOTFS:-0}" == "1" ]] \
  || jupyter_die "server_entry.sh must run inside the rootfs (TORCHTITAN_IN_ROOTFS=1)"

VENV_PY="${REPO_ROOT}/.venv-rootfs/bin/python"
[[ -x "${VENV_PY}" ]] || jupyter_die "rootfs venv python not found at ${VENV_PY}"

TOKEN="${JUPYTER_ROOTFS_TOKEN:-}"
[[ -n "${TOKEN}" ]] || jupyter_die "JUPYTER_ROOTFS_TOKEN not set; launch via run_server.sh"

# Ensure the server stack is present in .venv-rootfs. Install idempotently only
# when jupyter_server/jupyterlab are missing, pinning the recorded
# requirements. This makes the environment reproducible rather than a one-off
# interactive install.
if ! "${VENV_PY}" -c 'import jupyter_server, jupyterlab' >/dev/null 2>&1; then
  printf 'jupyter-rootfs: installing server deps into .venv-rootfs\n' >&2
  "${VENV_PY}" -m uv pip install \
    -r "${JUPYTER_DIR}/requirements.txt" 2>&1 \
    || "${VENV_PY}" -m pip install -r "${JUPYTER_DIR}/requirements.txt"
fi

# Register/refresh the rootfs kernelspec so a fresh checkout has the
# torchtitan-rootfs kernel verify.sh expects. Idempotent.
if ! "${VENV_PY}" -m jupyter kernelspec list 2>/dev/null \
    | grep -q "${JUPYTER_ROOTFS_KERNEL}"; then
  "${VENV_PY}" -m ipykernel install --user \
    --name "${JUPYTER_ROOTFS_KERNEL}" \
    --display-name "TorchTitan (rootfs, torch cu130)"
fi

printf 'jupyter-rootfs: starting jupyter lab on %s:%s\n' \
  "${JUPYTER_ROOTFS_HOST}" "${JUPYTER_ROOTFS_PORT}" | tee -a "${JUPYTER_ROOTFS_SERVER_LOG}"

# Loopback-only bind, token-only identity, no password fallback, no browser.
# stdout/stderr tee to the shared log for host-side inspection.
exec "${VENV_PY}" -m jupyter lab \
  --ServerApp.ip="${JUPYTER_ROOTFS_HOST}" \
  --ServerApp.port="${JUPYTER_ROOTFS_PORT}" \
  --IdentityProvider.token="${TOKEN}" \
  --ServerApp.open_browser=False \
  --ServerApp.password='' \
  --ServerApp.root_dir="${JUPYTER_ROOTFS_ROOT_DIR}" \
  2>&1 | tee -a "${JUPYTER_ROOTFS_SERVER_LOG}"
