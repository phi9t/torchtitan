#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Host-side orchestrator for the rootfs-hosted Jupyter server. This script does
# no Python/torch work itself: it mints a random token, launches the server
# inside a dedicated tmux session, and prints the loopback URL. The server
# process re-enters the bwrap rootfs via server_entry.sh (which sources
# run_common.sh), so all Jupyter and kernel work runs inside the sandbox.
#
# Usage:
#   experiments/jupyter_rootfs/run_server.sh
#   PORT=8901 SESSION=jlab experiments/jupyter_rootfs/run_server.sh
#
# Preflight refuses to start (loud fail) if the tmux session already exists or
# the port is already bound, so a second launch cannot silently collide with a
# live server.

set -euo pipefail

# shellcheck source=experiments/jupyter_rootfs/run_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_common.sh"
jupyter_setup_env

command -v tmux >/dev/null || jupyter_die "tmux not found on host"
command -v python3 >/dev/null || jupyter_die "python3 not found on host for token minting"

# Preflight: an existing session or a bound port is a loud error, not a silent
# reuse. Kill an existing session with stop_server.sh first.
if tmux has-session -t "${JUPYTER_ROOTFS_SESSION}" 2>/dev/null; then
  jupyter_die "tmux session '${JUPYTER_ROOTFS_SESSION}' already exists; run stop_server.sh first"
fi
if python3 - "$JUPYTER_ROOTFS_HOST" "$JUPYTER_ROOTFS_PORT" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.connect((host, port))
    sys.exit(0)  # something is listening
except OSError:
    sys.exit(1)  # free
finally:
    s.close()
PY
then
  jupyter_die "port ${JUPYTER_ROOTFS_HOST}:${JUPYTER_ROOTFS_PORT} is already bound"
fi

# Mint a fresh random token per launch and store it 0600. The predictable
# timestamp token from the ad-hoc proof is replaced by secrets.token_urlsafe.
# The token still appears in the tmux command line; that is a documented
# tradeoff (see README.org "Security posture").
umask 077
python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "${JUPYTER_ROOTFS_TOKEN_FILE}"
chmod 0600 "${JUPYTER_ROOTFS_TOKEN_FILE}"
TOKEN="$(cat "${JUPYTER_ROOTFS_TOKEN_FILE}")"

URL="http://${JUPYTER_ROOTFS_HOST}:${JUPYTER_ROOTFS_PORT}/?token=${TOKEN}"
printf '%s\n' "${URL}" > "${JUPYTER_ROOTFS_URL_FILE}"
chmod 0600 "${JUPYTER_ROOTFS_URL_FILE}"

# Launch the server inside a detached tmux session. The tmux command re-enters
# the rootfs through server_entry.sh (guarded by run_common.sh), so the server
# itself runs under bwrap with TORCHTITAN_IN_ROOTFS=1.
tmux new-session -d -s "${JUPYTER_ROOTFS_SESSION}" -c "${REPO_ROOT}" \
  "JUPYTER_ROOTFS_TOKEN='${TOKEN}' \
   JUPYTER_ROOTFS_PORT='${JUPYTER_ROOTFS_PORT}' \
   JUPYTER_ROOTFS_HOST='${JUPYTER_ROOTFS_HOST}' \
   JUPYTER_ROOTFS_ROOT_DIR='${JUPYTER_ROOTFS_ROOT_DIR}' \
   experiments/jupyter_rootfs/server_entry.sh"

printf 'jupyter-rootfs: server launching in tmux session %s\n' "${JUPYTER_ROOTFS_SESSION}"
printf 'jupyter-rootfs: url      %s\n' "${URL}"
printf 'jupyter-rootfs: token    %s\n' "${JUPYTER_ROOTFS_TOKEN_FILE}"
printf 'jupyter-rootfs: log      %s\n' "${JUPYTER_ROOTFS_SERVER_LOG}"
printf 'jupyter-rootfs: attach   experiments/jupyter_rootfs/verify.sh (headless self-check)\n'
printf 'jupyter-rootfs: stop     experiments/jupyter_rootfs/stop_server.sh\n'
