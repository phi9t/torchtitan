#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Headless drift self-check for the rootfs-hosted Jupyter server. The wrapper
# re-enters the bwrap rootfs with the repo-owned literate Emacs profile, then
# runs verify_attach.el under `emacs --batch'. That elisp launches the pinned
# kernel through emacs-jupyter and asserts the kernel is on GPU inside the
# rootfs. Any drift exits non-zero.
#
# Usage:
#   experiments/jupyter_rootfs/verify.sh
#   JUPYTER_VERIFY_MIN_GPUS=8 experiments/jupyter_rootfs/verify.sh

set -euo pipefail

# shellcheck source=experiments/jupyter_rootfs/run_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_common.sh"
jupyter_enter_rootfs_with_emacs_if_needed "verify.sh" "$@"
jupyter_setup_env

[[ -f "${JUPYTER_ROOTFS_TOKEN_FILE}" ]] \
  || jupyter_die "no token at ${JUPYTER_ROOTFS_TOKEN_FILE}; is the server running? (run_server.sh)"
TOKEN="$(cat "${JUPYTER_ROOTFS_TOKEN_FILE}")"

# Resolve the Doom Emacs binary. activate-doom-emacs.sh exports $EMACS.
# shellcheck source=/dev/null
source "${TORCHTITAN_ROOTFS_HOST_HOME:-${HOME}}/devx/activate-doom-emacs.sh" >/dev/null 2>&1 \
  || jupyter_die "cannot source devx/activate-doom-emacs.sh"
[[ -x "${EMACS:-}" ]] || jupyter_die "Doom Emacs binary not found via \$EMACS"
EMACS_PROFILE="${TORCHTITAN_EMACS_PROFILE:-${JUPYTER_DIR}/emacs_profile}"
[[ -f "${EMACS_PROFILE}/init.el" ]] || jupyter_die "Emacs profile missing init.el: ${EMACS_PROFILE}"
if [[ "${TORCHTITAN_IN_ROOTFS:-0}" == "1" ]]; then
  export TMPDIR=/project/tmp
  export TEMP=/project/tmp
  export TMP=/project/tmp
fi

# Allow a wrong-port smoke test: JUPYTER_VERIFY_URL overrides the derived URL
# so a deliberate mispoint confirms the script exits non-zero on drift.
VERIFY_URL="${JUPYTER_VERIFY_URL:-http://${JUPYTER_ROOTFS_HOST}:${JUPYTER_ROOTFS_PORT}}"

printf 'jupyter-rootfs: verifying %s (kernel %s, min-gpus %s)\n' \
  "${VERIFY_URL}" "${JUPYTER_ROOTFS_KERNEL}" "${JUPYTER_VERIFY_MIN_GPUS:-1}"

jupyter_reap_disconnected_kernels "${TOKEN}"

JUPYTER_VERIFY_URL="${VERIFY_URL}" \
JUPYTER_VERIFY_TOKEN="${TOKEN}" \
JUPYTER_VERIFY_KERNEL="${JUPYTER_ROOTFS_KERNEL}" \
JUPYTER_VERIFY_MIN_GPUS="${JUPYTER_VERIFY_MIN_GPUS:-1}" \
  timeout "${JUPYTER_VERIFY_TIMEOUT:-360}" "${EMACS}" --batch \
    --init-directory "${EMACS_PROFILE}" \
    -l "${EMACS_PROFILE}/init.el" \
    -l "${JUPYTER_DIR}/verify_attach.el" 2>&1 \
  | grep -aE "PASS|FAIL|VERIFY-OK|VERIFY-FAIL"
# Propagate emacs's exit code, not grep's.
exit "${PIPESTATUS[0]}"
