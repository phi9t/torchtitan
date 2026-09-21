#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Execute an Org Babel literate notebook through the rootfs-hosted Jupyter
# server. The verifier re-enters bwrap with the repo-owned literate Emacs
# profile, and every jupyter-python block executes in the torchtitan-rootfs
# kernel inside bwrap.

set -euo pipefail

# shellcheck source=experiments/jupyter_rootfs/run_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_common.sh"
jupyter_enter_rootfs_with_emacs_if_needed "verify_org_babel.sh" "$@"
jupyter_setup_env

ORG_FILE="${1:-experiments/modded_nanogpt_b200/hack.org}"
if [[ "${ORG_FILE}" != /* ]]; then
  ORG_FILE="${REPO_ROOT}/${ORG_FILE}"
fi
[[ -f "${ORG_FILE}" ]] || jupyter_die "org file not found: ${ORG_FILE}"

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

VERIFY_URL="${JUPYTER_VERIFY_URL:-http://${JUPYTER_ROOTFS_HOST}:${JUPYTER_ROOTFS_PORT}}"
VERIFY_ORG_SESSION="${JUPYTER_VERIFY_ORG_SESSION:-torchtitan-literate-$$}"
printf 'jupyter-rootfs: verifying org babel %s via %s (kernel %s)\n' \
  "${ORG_FILE}" "${VERIFY_URL}" "${JUPYTER_ROOTFS_KERNEL}"

jupyter_reap_disconnected_kernels "${TOKEN}"

JUPYTER_VERIFY_URL="${VERIFY_URL}" \
JUPYTER_VERIFY_TOKEN="${TOKEN}" \
JUPYTER_VERIFY_KERNEL="${JUPYTER_ROOTFS_KERNEL}" \
JUPYTER_VERIFY_ORG_SESSION="${VERIFY_ORG_SESSION}" \
JUPYTER_VERIFY_ORG_FILE="${ORG_FILE}" \
  timeout "${JUPYTER_VERIFY_TIMEOUT:-360}" "${EMACS}" --batch \
    --init-directory "${EMACS_PROFILE}" \
    -l "${EMACS_PROFILE}/init.el" \
    -l "${JUPYTER_DIR}/verify_org_babel.el" 2>&1 \
  | grep -aE "PASS|FAIL|VERIFY-OK|VERIFY-FAIL"
exit "${PIPESTATUS[0]}"
