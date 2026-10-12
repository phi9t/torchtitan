#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Idempotent teardown for the rootfs-hosted Jupyter server. Reaps live kernels
# via the server REST API (so the server shuts them down cleanly), then kills
# the tmux session hosting the server. Safe to run when nothing is running.
#
# This is host-side orchestration only: it talks to the loopback REST endpoint
# and manages tmux. No Python business logic runs outside the rootfs; the small
# python3 helper here only issues HTTP requests to reap kernels.

set -euo pipefail

# shellcheck source=experiments/jupyter_rootfs/run_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_common.sh"
jupyter_setup_env

# Reap kernels through the REST API while the server is still up. Best effort:
# if the server is already gone this is a no-op.
if [[ -f "${JUPYTER_ROOTFS_TOKEN_FILE}" ]] && command -v python3 >/dev/null; then
  TOKEN="$(cat "${JUPYTER_ROOTFS_TOKEN_FILE}")"
  JUPYTER_ROOTFS_TOKEN="${TOKEN}" python3 - \
    "${JUPYTER_ROOTFS_HOST}" "${JUPYTER_ROOTFS_PORT}" <<'PY' || true
import json, os, sys, urllib.request

host, port = sys.argv[1], sys.argv[2]
token = os.environ["JUPYTER_ROOTFS_TOKEN"]
base = f"http://{host}:{port}/api"
headers = {"Authorization": f"token {token}"}


def request(method, path):
    req = urllib.request.Request(f"{base}{path}", method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=5) as resp:
        body = resp.read()
        return json.loads(body) if body else None


try:
    kernels = request("GET", "/kernels") or []
except Exception as exc:  # server likely already down
    print(f"jupyter-rootfs: no live server to reap ({exc})", file=sys.stderr)
    sys.exit(0)

for kernel in kernels:
    kid = kernel.get("id")
    if not kid:
        continue
    try:
        request("DELETE", f"/kernels/{kid}")
        print(f"jupyter-rootfs: reaped kernel {kid}", file=sys.stderr)
    except Exception as exc:
        print(f"jupyter-rootfs: failed to reap kernel {kid}: {exc}", file=sys.stderr)
PY
fi

# Kill the tmux session hosting the server. Idempotent.
if tmux has-session -t "${JUPYTER_ROOTFS_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${JUPYTER_ROOTFS_SESSION}"
  printf 'jupyter-rootfs: killed tmux session %s\n' "${JUPYTER_ROOTFS_SESSION}"
else
  printf 'jupyter-rootfs: no tmux session %s to kill\n' "${JUPYTER_ROOTFS_SESSION}"
fi

# The per-launch token is no longer valid once the server is down.
rm -f "${JUPYTER_ROOTFS_TOKEN_FILE}" "${JUPYTER_ROOTFS_URL_FILE}"
printf 'jupyter-rootfs: teardown complete\n'
