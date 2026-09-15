#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Headless drift self-check for the rootfs-hosted Jupyter server. Resolves the
# Doom Emacs binary via ~/devx/activate-doom-emacs.sh, then runs
# verify_attach.el under `emacs --batch'. That elisp drives the same
# jupyter-run-server-repl codepath M-x uses, launches the pinned kernel, and
# asserts the kernel is on GPU inside the rootfs. Any drift exits non-zero.
#
# This is a client-side attach: emacs-jupyter and the REST/websocket traffic
# run on the host, but the kernel it talks to runs inside the rootfs. That is
# the whole point of the check -- prove the Emacs->server->rootfs-kernel path.
#
# Usage:
#   experiments/jupyter_rootfs/verify.sh
#   JUPYTER_VERIFY_MIN_GPUS=8 experiments/jupyter_rootfs/verify.sh

set -euo pipefail

# shellcheck source=experiments/jupyter_rootfs/run_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_common.sh"
jupyter_setup_env

[[ -f "${JUPYTER_ROOTFS_TOKEN_FILE}" ]] \
  || jupyter_die "no token at ${JUPYTER_ROOTFS_TOKEN_FILE}; is the server running? (run_server.sh)"
TOKEN="$(cat "${JUPYTER_ROOTFS_TOKEN_FILE}")"

# Resolve the Doom Emacs binary. activate-doom-emacs.sh exports $EMACS.
# shellcheck source=/dev/null
source "${HOME}/devx/activate-doom-emacs.sh" >/dev/null 2>&1 \
  || jupyter_die "cannot source ~/devx/activate-doom-emacs.sh"
[[ -x "${EMACS:-}" ]] || jupyter_die "Doom Emacs binary not found via \$EMACS"

# Allow a wrong-port smoke test: JUPYTER_VERIFY_URL overrides the derived URL
# so a deliberate mispoint confirms the script exits non-zero on drift.
VERIFY_URL="${JUPYTER_VERIFY_URL:-http://${JUPYTER_ROOTFS_HOST}:${JUPYTER_ROOTFS_PORT}}"

printf 'jupyter-rootfs: verifying %s (kernel %s, min-gpus %s)\n' \
  "${VERIFY_URL}" "${JUPYTER_ROOTFS_KERNEL}" "${JUPYTER_VERIFY_MIN_GPUS:-1}"

JUPYTER_VERIFY_URL="${VERIFY_URL}" \
JUPYTER_VERIFY_TOKEN="${TOKEN}" \
JUPYTER_VERIFY_KERNEL="${JUPYTER_ROOTFS_KERNEL}" \
JUPYTER_VERIFY_MIN_GPUS="${JUPYTER_VERIFY_MIN_GPUS:-1}" \
  timeout 240 "${EMACS}" --batch -l "${JUPYTER_DIR}/verify_attach.el" 2>&1 \
  | grep -aE "PASS|FAIL|VERIFY-OK|VERIFY-FAIL"
# Propagate emacs's exit code, not grep's.
exit "${PIPESTATUS[0]}"
