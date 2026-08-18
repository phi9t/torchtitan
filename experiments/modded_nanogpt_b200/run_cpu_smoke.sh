#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_REL="experiments/modded_nanogpt_b200/run_cpu_smoke.sh"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "${SCRIPT_REL}" "$@"
fi

source "${SCRIPT_DIR}/rootfs_guard.sh"
require_modded_nanogpt_rootfs "${SCRIPT_REL}"

cd "${REPO_ROOT}"
exec python experiments/modded_nanogpt_b200/cpu_smoke.py "$@"
