#!/usr/bin/env bash
# Build and publish GLM-5.3 L2 native wheels.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
lock_file="${script_dir}/packages.lock.json"
scratch_root="${GLM53_NATIVE_SCRATCH:-${repo_root}/.scratch/glm53-native}"

json_field() {
  python - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
value = payload
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

package_json() {
  python - "$lock_file" "$1" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
for item in payload["packages"]:
    if item["name"] == sys.argv[2]:
        print(json.dumps(item, sort_keys=True))
        raise SystemExit(0)
raise SystemExit(1)
PY
}

rootfs_store="${GLM53_NATIVE_ROOTFS_STORE:-$(json_field "$lock_file" rootfs_store)}"
rootfs_env="${GLM53_NATIVE_ROOTFS_ENV:-$(json_field "$lock_file" default_rootfs_env)}"

usage() {
  cat >&2 <<'USAGE'
Usage: build_native.sh <pkg>|all [--dev] [--no-publish]

Builds native packages declared in packages.lock.json against the glm53_ep
rootfs env. Normal builds publish wheel + receipt to HDFS through Waystone.
--dev installs an editable source tree inside the env and never publishes.
USAGE
}

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

native_publish_tree() {
  local local_dir="$1"
  local relpath="$2"
  # shellcheck source=experiments/glm53_flash_sft_ladder/lib/waystone.sh
  source "${repo_root}/experiments/glm53_flash_sft_ladder/lib/waystone.sh"
  local hdfs_dir tmp_root manifest complete_file
  hdfs_dir="$(ws_child_uri "$relpath")"
  tmp_root="${scratch_root}/publish-tmp/$(date -u +%Y%m%dT%H%M%SZ)-$$"
  mkdir -p "$tmp_root"
  manifest="${tmp_root}/MANIFEST.jsonl"
  complete_file="${tmp_root}/_COMPLETE"
  python - "$local_dir" "$manifest" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
rows = []
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(root).as_posix()
    if rel in {"MANIFEST.jsonl", "_COMPLETE", "publish.log"}:
        continue
    data = path.read_bytes()
    rows.append({"file": rel, "size": path.stat().st_size, "md5": hashlib.md5(data).hexdigest()})
manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
PY
  while IFS= read -r rel; do
    [[ -n "$rel" ]] || continue
    ws_put_file "${local_dir%/}/$rel" "$(ws_join_uri "$hdfs_dir" "$rel")"
  done < <(python - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

for line in Path(sys.argv[1]).read_text().splitlines():
    if line:
        print(json.loads(line)["file"])
PY
)
  {
    printf 'source_dir=%s\n' "$local_dir"
    printf 'relpath=%s\n' "$relpath"
    printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$complete_file"
  ws_put_file "$manifest" "$(ws_join_uri "$hdfs_dir" "MANIFEST.jsonl")"
  ws_put_file "$complete_file" "$(ws_join_uri "$hdfs_dir" "_COMPLETE")"
  printf 'HDFS_URI=%s\n' "$hdfs_dir"
  printf 'FILES=%s\n' "$(wc -l < "$manifest" | tr -d '[:space:]')"
  printf 'LOCAL_MANIFEST=%s\n' "$manifest"
}

package_exists() {
  local pkg="$1"
  package_json "$pkg" >/dev/null
}

package_names() {
  python - "$lock_file" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
for item in payload["packages"]:
    print(item["name"])
PY
}

dev=0
publish=1
if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi
pkg="$1"
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      dev=1
      publish=0
      shift
      ;;
    --no-publish)
      publish=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      printf 'unknown argument: %s\n' "$1" >&2
      usage
      exit 2
      ;;
    *)
      printf 'unexpected argument: %s\n' "$1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$pkg" == "all" ]]; then
  while IFS= read -r one_pkg; do
    args=("$one_pkg")
    (( dev == 1 )) && args+=(--dev)
    (( publish == 0 && dev == 0 )) && args+=(--no-publish)
    "${BASH_SOURCE[0]}" "${args[@]}"
  done < <(package_names)
  exit 0
fi

if ! package_exists "$pkg"; then
  printf 'unknown native package: %s\n' "$pkg" >&2
  package_names >&2
  exit 2
fi

latest_dir="${scratch_root}/latest"
mkdir -p "$latest_dir"

if [[ "${GLM53_NATIVE_IN_ROOTFS:-0}" != "1" ]]; then
  rootfs_args=(
    TORCHTITAN_ROOTFS_NETWORK=networked
    TORCHTITAN_ROOTFS_ENV="$rootfs_env"
    "${repo_root}/scripts/rootfs/enter_rootfs.sh"
    --rootfs-store "$rootfs_store"
    --
    bash -lc
  )
  inner_args=("$pkg")
  (( dev == 1 )) && inner_args+=(--dev)
  (( publish == 0 && dev == 0 )) && inner_args+=(--no-publish)
  inner_cmd=$(
    printf 'unset CC CXX CUDAHOSTCXX; cd /workspace/torchtitan && GLM53_NATIVE_IN_ROOTFS=1 %q' \
      "experiments/glm53_native/build_native.sh"
    printf ' %q' "${inner_args[@]}"
  )
  env "${rootfs_args[@]}" "$inner_cmd"

  env_file="${latest_dir}/${pkg}.env"
  if [[ ! -f "$env_file" ]]; then
    printf 'build did not write expected env file: %s\n' "$env_file" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$env_file"
  rootfs_project="/workspace/torchtitan"
  case "${PUBLISH_DIR:-}" in
    "$rootfs_project"/*) PUBLISH_DIR="${repo_root}/${PUBLISH_DIR#"$rootfs_project"/}" ;;
  esac
  case "${RECEIPT:-}" in
    "$rootfs_project"/*) RECEIPT="${repo_root}/${RECEIPT#"$rootfs_project"/}" ;;
  esac
  printf 'BUILD_KEY=%s\n' "$BUILD_KEY"
  printf 'RECEIPT=%s\n' "$RECEIPT"
  if (( dev == 1 || publish == 0 )); then
    printf 'PUBLISH_SKIPPED=1\n'
    exit 0
  fi
  publish_log="${scratch_root}/publish/${pkg}/${BUILD_KEY}/publish.log"
  mkdir -p "$(dirname "$publish_log")"
  native_publish_tree "$PUBLISH_DIR" "native-wheels/${pkg}/${BUILD_KEY}" | tee "$publish_log"
  hdfs_uri="$(awk -F= '/^HDFS_URI=/{print $2}' "$publish_log" | tail -1)"
  printf 'HDFS_URI=%s\n' "$hdfs_uri"
  exit 0
fi

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  printf 'inner native build must run inside the TorchTitan rootfs\n' >&2
  exit 1
fi
if [[ "${TORCHTITAN_ROOTFS_ENV:-}" != "$rootfs_env" ]]; then
  printf 'native build expected TORCHTITAN_ROOTFS_ENV=%s, found %s\n' \
    "$rootfs_env" "${TORCHTITAN_ROOTFS_ENV:-}" >&2
  exit 1
fi

require_tool git
require_tool sha256sum

work_dir="${scratch_root}/work/${pkg}"
rm -rf "$work_dir"
mkdir -p "$work_dir"

pkg_file="${work_dir}/package.json"
package_json "$pkg" > "$pkg_file"
source_type="$(json_field "$pkg_file" source.type)"
patch_dir_rel="$(python - "$pkg_file" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(payload.get("patch_dir", ""))
PY
)"
patch_dir="${script_dir}/${patch_dir_rel}"
source_dir="${work_dir}/src"

case "$source_type" in
  git)
    repo="$(json_field "$pkg_file" source.repo)"
    commit="$(json_field "$pkg_file" source.commit)"
    git clone "$repo" "$source_dir"
    git -C "$source_dir" checkout "$commit"
    if [[ "$(python - "$pkg_file" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print("true" if payload["source"].get("submodules", False) else "false")
PY
)" == "true" ]]; then
      git -C "$source_dir" submodule update --init --recursive
    fi
    ;;
  path)
    rel_path="$(json_field "$pkg_file" source.path)"
    mkdir -p "$source_dir"
    cp -a "${repo_root}/${rel_path}/." "$source_dir/"
    ;;
  *)
    printf 'unsupported source.type for %s: %s\n' "$pkg" "$source_type" >&2
    exit 1
    ;;
esac

if [[ -d "$patch_dir" ]]; then
  shopt -s nullglob
  for patch in "$patch_dir"/*.patch; do
    git -C "$source_dir" apply "$patch"
  done
  shopt -u nullglob
fi

build_arg_exports=()
while IFS=$'\t' read -r key value; do
  build_arg_exports+=("$key=$value")
done < <(python - "$pkg_file" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
for key, value in (payload.get("build_args") or {}).items():
    print(f"{key}\t{value}")
PY
)

ccache_dir="${GLM53_NATIVE_CCACHE_DIR:-/project/xdg-cache/ccache/glm53_native/${pkg}}"
mkdir -p "$ccache_dir"
export CCACHE_DIR="$ccache_dir"
export CMAKE_CUDA_COMPILER_LAUNCHER="${CMAKE_CUDA_COMPILER_LAUNCHER:-ccache}"
export CMAKE_CXX_COMPILER_LAUNCHER="${CMAKE_CXX_COMPILER_LAUNCHER:-ccache}"
export CUDA_HOME="${CUDA_HOME:-/opt/cuda-synth}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
for item in "${build_arg_exports[@]}"; do
  export "$item"
done

metadata_json="${work_dir}/build-metadata.json"
python - "$lock_file" "$pkg" "$source_dir" "$patch_dir" > "$metadata_json" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

lock_path = Path(sys.argv[1])
pkg_name = sys.argv[2]
source_dir = Path(sys.argv[3])
patch_dir = Path(sys.argv[4])

lock = json.loads(lock_path.read_text())
pkg = next(item for item in lock["packages"] if item["name"] == pkg_name)


def tree_hash(root: Path, *, patch_files_only: bool = False) -> tuple[str, list[str]]:
    digest = hashlib.sha256()
    files: list[str] = []
    if not root.exists():
        return hashlib.sha256(b"").hexdigest(), files
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        parts = set(path.relative_to(root).parts)
        if parts & {".git", "__pycache__", "build", "dist", ".eggs"}:
            continue
        if rel.endswith(".pyc") or ".egg-info/" in rel or rel.endswith(".egg-info"):
            continue
        if patch_files_only and not rel.endswith(".patch"):
            continue
        data = path.read_bytes()
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(str(path.stat().st_mode & 0o777).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).hexdigest().encode())
        digest.update(b"\0")
        files.append(rel)
    return digest.hexdigest(), files


def run_text(argv: list[str]) -> str:
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


source_hash, source_files = tree_hash(source_dir)
patch_hash, patch_files = tree_hash(patch_dir, patch_files_only=True)
build_args = {key: str(value) for key, value in (pkg.get("build_args") or {}).items()}
for key in build_args:
    build_args[key] = os.environ.get(key, build_args[key])

payload = {
    "schema_version": 1,
    "package": pkg,
    "source_tree_hash": source_hash,
    "source_file_count": len(source_files),
    "patches_hash": patch_hash,
    "patch_files": patch_files,
    "torch": {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
    },
    "cuda": {
        "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
        "nvcc": run_text([os.environ.get("CUDA_HOME", "/opt/cuda-synth") + "/bin/nvcc", "--version"]),
    },
    "arch": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
    "build_args": build_args,
    "rootfs": {
        "env": os.environ.get("TORCHTITAN_ROOTFS_ENV", ""),
        "store_id": os.environ.get("TORCHTITAN_ROOTFS_STORE_ID", ""),
    },
}
key_material = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
payload["build_key"] = hashlib.sha256(key_material).hexdigest()
print(json.dumps(payload, indent=2, sort_keys=True))
PY

build_key="$(json_field "$metadata_json" build_key)"

if (( dev == 1 )); then
  python -m pip install --no-build-isolation --no-deps -e "$source_dir"
  receipt_dir="${scratch_root}/dev/${pkg}/${build_key}"
  mkdir -p "$receipt_dir"
  receipt="${receipt_dir}/${pkg}-${build_key}.json"
  python - "$metadata_json" "$receipt" "$ccache_dir" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text())
metadata.update(
    {
        "mode": "dev",
        "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "ccache_dir": sys.argv[3],
        "publishable": False,
    }
)
Path(sys.argv[2]).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY
  {
    printf 'BUILD_KEY=%s\n' "$build_key"
    printf 'RECEIPT=%s\n' "$receipt"
    printf 'PUBLISH_DIR=%s\n' "$receipt_dir"
  } > "${latest_dir}/${pkg}.env"
  exit 0
fi

dist_dir="${work_dir}/dist"
rm -rf "$dist_dir"
mkdir -p "$dist_dir"
python -m pip wheel --no-deps --no-build-isolation --wheel-dir "$dist_dir" "$source_dir"

mapfile -t wheels < <(find "$dist_dir" -maxdepth 1 -type f -name '*.whl' | sort)
if [[ "${#wheels[@]}" -ne 1 ]]; then
  printf 'expected exactly one wheel for %s, found %s\n' "$pkg" "${#wheels[@]}" >&2
  exit 1
fi
wheel="${wheels[0]}"
publish_dir="${scratch_root}/publish/${pkg}/${build_key}"
rm -rf "$publish_dir"
mkdir -p "$publish_dir"
cp "$wheel" "$publish_dir/"
wheel_name="$(basename "$wheel")"
receipt="${publish_dir}/${pkg}-${build_key}.json"
python - "$metadata_json" "$publish_dir/$wheel_name" "$receipt" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text())
wheel = Path(sys.argv[2])
receipt = Path(sys.argv[3])
data = wheel.read_bytes()
metadata.update(
    {
        "mode": "wheel",
        "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "publishable": True,
        "hdfs_relpath": f"native-wheels/{metadata['package']['name']}/{metadata['build_key']}",
        "wheel": {
            "filename": wheel.name,
            "size": wheel.stat().st_size,
            "md5": hashlib.md5(data).hexdigest(),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
    }
)
receipt.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

{
  printf 'BUILD_KEY=%s\n' "$build_key"
  printf 'RECEIPT=%s\n' "$receipt"
  printf 'PUBLISH_DIR=%s\n' "$publish_dir"
} > "${latest_dir}/${pkg}.env"
