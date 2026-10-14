#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${EXPERIMENT_DIR}/../.." && pwd)"
SCRIPT_REL="experiments/glm53_flash_sft_ladder/runtime/sync_python_env.sh"
ROOTFS_ENTRYPOINT="${TORCHTITAN_ROOTFS_ENTRYPOINT:-${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh}"
ROOTFS_ARGS=()

die() {
  printf 'error: %s\n' "$*" >&2
  exit 21
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rootfs-store)
      ROOTFS_ARGS+=(--rootfs-store "${2:?--rootfs-store requires a value}")
      shift 2
      ;;
    --rootfs-store=*)
      ROOTFS_ARGS+=(--rootfs-store "${1#*=}")
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  export TORCHTITAN_ROOTFS_ENV=glm53_ep
  : "${TORCHTITAN_ROOTFS_NETWORK:=networked}"
  export TORCHTITAN_ROOTFS_NETWORK
  exec "${ROOTFS_ENTRYPOINT}" "${ROOTFS_ARGS[@]}" -- "${SCRIPT_REL}"
fi

if [[ "${TORCHTITAN_ROOTFS_ENV:-}" != "glm53_ep" ]]; then
  die "expected TORCHTITAN_ROOTFS_ENV=glm53_ep, found ${TORCHTITAN_ROOTFS_ENV:-unset}"
fi
if [[ "$(pwd -P)" != "/workspace/torchtitan" ]]; then
  die "expected rootfs workspace /workspace/torchtitan, found $(pwd -P)"
fi
if [[ ! -f "/workspace/torchtitan/scripts/rootfs/enter_rootfs.sh" ]]; then
  die "rootfs workspace sentinel is missing"
fi

cd "${REPO_ROOT}"

command -v uv >/dev/null || die "uv is required inside the rootfs"

unset CC CXX CUDAHOSTCXX
export MAX_JOBS="${MAX_JOBS:-16}"
export FLASH_ATTENTION_SKIP_CUDA_BUILD="${FLASH_ATTENTION_SKIP_CUDA_BUILD:-TRUE}"

VENV="${GLM53_EP_RUNTIME_VENV:-/project/venvs/b200-runtime}"
LOCK="${GLM53_EP_REQUIREMENTS_LOCK:-${SCRIPT_DIR}/requirements.lock}"
DIRECT="${GLM53_EP_REQUIREMENTS_DIRECT:-${SCRIPT_DIR}/requirements.direct.txt}"
BASE_PYTHON="${GLM53_EP_BASE_PYTHON:-$(command -v python)}"

lock_has_hash_entries() {
  [[ -f "$1" ]] || return 1
  awk '
    /--hash=sha256:/ { found = 1 }
    /^[[:space:]]*($|#|--)/ { next }
    END { exit found ? 0 : 1 }
  ' "$1"
}

lock_has_unhashed_requirements() {
  awk '
    /--hash=sha256:/ { hashed = 1 }
    /^[[:space:]]*($|#|--)/ { next }
    /^[^[:space:]]/ {
      if (seen && !hashed) {
        bad = 1
      }
      seen = 1
      hashed = 0
    }
    END {
      if (seen && !hashed) {
        bad = 1
      }
      exit bad ? 0 : 1
    }
  ' "$1"
}

[[ -f "${LOCK}" ]] || die "requirements lock is missing: ${LOCK}"
lock_has_hash_entries "${LOCK}" || die "requirements.lock has no hash-locked package entries"
if lock_has_unhashed_requirements "${LOCK}"; then
  die "requirements.lock contains package entries without --hash"
fi

uv venv --clear --python "${BASE_PYTHON}" --system-site-packages "${VENV}"
BOOTSTRAP_REQUIREMENTS="$(mktemp)"
trap 'rm -f "${BOOTSTRAP_REQUIREMENTS}"' EXIT
grep -E '^(torch|triton) @ ' "${DIRECT}" > "${BOOTSTRAP_REQUIREMENTS}" \
  || die "direct requirements must pin torch and triton bootstrap wheels"
uv pip install \
  --python "${VENV}/bin/python" \
  --index-strategy unsafe-best-match \
  --no-deps \
  --require-hashes \
  -r "${BOOTSTRAP_REQUIREMENTS}"
uv pip install \
  --python "${VENV}/bin/python" \
  --index-strategy unsafe-best-match \
  --no-build-isolation-package flash-attn \
  --require-hashes \
  -r "${LOCK}"

"${VENV}/bin/python" - <<'PY' "${VENV}" "${LOCK}" "${DIRECT}"
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict[str, object]:
    manifest_path = Path("/build_manifest.json")
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {"manifest_error": "invalid json"}
    return payload if isinstance(payload, dict) else {"manifest_error": "not object"}


venv = Path(sys.argv[1])
lock = Path(sys.argv[2])
direct = Path(sys.argv[3])
manifest = load_manifest()
report = {
    "schema_version": 1,
    "kind": "glm53_ep_python_env_report",
    "ok": True,
    "env_name": os.environ.get("TORCHTITAN_ROOTFS_ENV"),
    "venv": str(venv),
    "python": str(venv / "bin" / "python"),
    "lock": str(lock),
    "lock_sha256": file_sha256(lock),
    "direct": str(direct),
    "direct_sha256": file_sha256(direct),
    "network_mode": os.environ.get("TORCHTITAN_ROOTFS_NETWORK", "offline"),
    "flash_attention_skip_cuda_build": os.environ.get(
        "FLASH_ATTENTION_SKIP_CUDA_BUILD"
    ),
    "rootfs_store_id": os.environ.get("TORCHTITAN_ROOTFS_STORE_ID"),
    "rootfs_manifest": manifest,
    "rootfs_digest": manifest.get("exported_rootfs_digest"),
}
(venv / "python_env_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(report, indent=2, sort_keys=True))
PY
