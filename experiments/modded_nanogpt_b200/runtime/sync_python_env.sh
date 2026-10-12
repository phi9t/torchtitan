#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${EXPERIMENT_DIR}/../.." && pwd)"
SCRIPT_REL="experiments/modded_nanogpt_b200/runtime/sync_python_env.sh"
ROOTFS_ENTRYPOINT="${TORCHTITAN_ROOTFS_ENTRYPOINT:-${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 21
}

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  if [[ -n "${MODDED_NANOGPT_SYNC_MARKER:-}" ]]; then
    exec "${ROOTFS_ENTRYPOINT}" "${MODDED_NANOGPT_SYNC_MARKER}" -- "${SCRIPT_REL}" "$@"
  fi
  exec "${ROOTFS_ENTRYPOINT}" -- "${SCRIPT_REL}" "$@"
fi

source "${EXPERIMENT_DIR}/rootfs_guard.sh"
require_modded_nanogpt_rootfs "${SCRIPT_REL}"

cd "${REPO_ROOT}"

command -v uv >/dev/null || die "uv is required inside the rootfs"

VENV="${MODDED_NANOGPT_RUNTIME_VENV:-/project/venvs/b200-runtime}"
LOCK="${MODDED_NANOGPT_REQUIREMENTS_LOCK:-${SCRIPT_DIR}/requirements.lock}"
DIRECT="${MODDED_NANOGPT_REQUIREMENTS_DIRECT:-${SCRIPT_DIR}/requirements.direct.txt}"
BASE_PYTHON="${MODDED_NANOGPT_BASE_PYTHON:-$(command -v python)}"

lock_has_hash_entries() {
  [[ -f "$1" ]] || return 1
  awk '
    /^[[:space:]]*($|#)/ { next }
    /--hash=sha256:/ { found = 1 }
    END { exit found ? 0 : 1 }
  ' "$1"
}

if ! lock_has_hash_entries "${LOCK}" && [[ "${TORCHTITAN_ROOTFS_NETWORK:-offline}" != "networked" ]]; then
  die "requirements.lock has no hash-locked package entries; direct requirements fallback requires TORCHTITAN_ROOTFS_NETWORK=networked"
fi

uv venv --python "${BASE_PYTHON}" --system-site-packages "${VENV}"

if lock_has_hash_entries "${LOCK}"; then
  uv pip install \
    --python "${VENV}/bin/python" \
    --no-deps \
    --require-hashes \
    -r "${LOCK}"
else
  uv pip install \
    --python "${VENV}/bin/python" \
    -r "${DIRECT}"
fi

cat > "${VENV}/python_env_report.json" <<EOF
{
  "schema_version": 1,
  "kind": "python_env_report",
  "ok": true,
  "venv": "${VENV}",
  "lock": "${LOCK}",
  "network_mode": "${TORCHTITAN_ROOTFS_NETWORK:-offline}"
}
EOF
python3 -m experiments.modded_nanogpt_b200.runtime.schema_validation \
  --schema python_env_report \
  --input "${VENV}/python_env_report.json"
