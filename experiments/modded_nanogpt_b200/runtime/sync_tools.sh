#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${EXPERIMENT_DIR}/../.." && pwd)"
SCRIPT_REL="experiments/modded_nanogpt_b200/runtime/sync_tools.sh"
RUNTIME_DIR_IN_ROOTFS="experiments/modded_nanogpt_b200/runtime"
ROOTFS_ENTRYPOINT="${TORCHTITAN_ROOTFS_ENTRYPOINT:-${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 21
}

json_escape() {
  python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$1"
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

command -v mise >/dev/null || die "mise is required inside the rootfs; rebuild the rootfs with scripts/rootfs/build_rootfs.sh"

export MISE_DATA_DIR="${MISE_DATA_DIR:-/project/mise/data}"
export MISE_CACHE_DIR="${MISE_CACHE_DIR:-/project/mise/cache}"
export MISE_CONFIG_DIR="${MISE_CONFIG_DIR:-${RUNTIME_DIR_IN_ROOTFS}}"
export MISE_INSTALL_PATH="${MISE_INSTALL_PATH:-/project/mise/installs}"
export MISE_SHIMS_DIR="${MISE_SHIMS_DIR:-/project/mise/data/shims}"

mkdir -p "${MISE_DATA_DIR}" "${MISE_CACHE_DIR}" "${MISE_INSTALL_PATH}" "${MISE_SHIMS_DIR}"

mise trust "${RUNTIME_DIR_IN_ROOTFS}/mise.toml"
mise install --yes --cd "${RUNTIME_DIR_IN_ROOTFS}"
shellcheck_output="$(mise exec --cd "${RUNTIME_DIR_IN_ROOTFS}" -- shellcheck --version)"
printf '%s\n' "${shellcheck_output}"
shellcheck_json="$(json_escape "${shellcheck_output}")"

report="${MODDED_NANOGPT_TOOL_REPORT:-/project/mise/tool_env_report.json}"
mkdir -p "$(dirname -- "${report}")"
cat > "${report}" <<EOF
{
  "schema_version": 1,
  "kind": "tool_env_report",
  "ok": true,
  "mise_data_dir": "${MISE_DATA_DIR}",
  "mise_cache_dir": "${MISE_CACHE_DIR}",
  "mise_config_dir": "${MISE_CONFIG_DIR}",
  "mise_install_path": "${MISE_INSTALL_PATH}",
  "mise_shims_dir": "${MISE_SHIMS_DIR}",
  "shellcheck_version": ${shellcheck_json}
}
EOF
python3 -m experiments.modded_nanogpt_b200.runtime.schema_validation \
  --schema tool_env_report \
  --input "${report}"
