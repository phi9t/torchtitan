#!/usr/bin/env bash
# Stage and install a published GLM-5.3 native wheel into glm53_ep.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
lock_file="${script_dir}/packages.lock.json"
scratch_root="${GLM53_NATIVE_SCRATCH:-${repo_root}/.scratch/glm53-native}"
keys_file_default="/project/venvs/b200-runtime/glm53_native_build_keys.json"

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

rootfs_store="${GLM53_NATIVE_ROOTFS_STORE:-$(json_field "$lock_file" rootfs_store)}"
rootfs_env="${GLM53_NATIVE_ROOTFS_ENV:-$(json_field "$lock_file" default_rootfs_env)}"

usage() {
  cat >&2 <<'USAGE'
Usage: install_native.sh <pkg> --key <build-key> [--hdfs-uri URI] [--parallel N]

Stages a completed native wheel artifact tree from HDFS, verifies it with
hdfs_strict.sh, installs the wheel into glm53_ep, runs the package import check,
and records the installed build key for training attempt metadata.
USAGE
}

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

native_stage_tree() {
  local hdfs_dir="$1"
  local local_dir="$2"
  # shellcheck source=experiments/glm53_flash_sft_ladder/lib/waystone.sh
  source "${repo_root}/experiments/glm53_flash_sft_ladder/lib/waystone.sh"
  local tmp_root manifest_local manifest_md5 file_count
  tmp_root="${scratch_root}/stage-tmp/$(date -u +%Y%m%dT%H%M%SZ)-$$"
  mkdir -p "$tmp_root" "$local_dir"
  if ! ws_exists "$(ws_join_uri "$hdfs_dir" "_COMPLETE")"; then
    printf 'missing _COMPLETE marker under %s\n' "$hdfs_dir" >&2
    exit 1
  fi
  manifest_local="${tmp_root}/MANIFEST.jsonl"
  ws_get_file "$(ws_join_uri "$hdfs_dir" "MANIFEST.jsonl")" "$manifest_local"
  manifest_md5="$(ws_md5_file "$manifest_local")"
  file_count="$(wc -l < "$manifest_local" | tr -d '[:space:]')"
  while IFS=$'\t' read -r rel size md5; do
    [[ -n "$rel" ]] || continue
    WS_EXPECTED_SIZE="$size" WS_EXPECTED_MD5="$md5" \
      ws_get_file "$(ws_join_uri "$hdfs_dir" "$rel")" "${local_dir%/}/$rel"
  done < <(python - "$manifest_local" <<'PY'
import json
import sys
from pathlib import Path

for line in Path(sys.argv[1]).read_text().splitlines():
    if not line:
        continue
    row = json.loads(line)
    print(f"{row['file']}\t{row['size']}\t{row['md5']}")
PY
)
  cp "$manifest_local" "${local_dir%/}/MANIFEST.jsonl"
  python - "$local_dir/.staged.json" "$hdfs_dir" "$manifest_md5" "$file_count" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "source_uri": sys.argv[2],
    "manifest_md5": sys.argv[3],
    "file_count": int(sys.argv[4]),
    "staged_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  native_hdfs_strict "$local_dir"
  printf 'STAGED_DIR=%s\n' "$local_dir"
  printf 'SOURCE_URI=%s\n' "$hdfs_dir"
  printf 'FILES=%s\n' "$file_count"
  printf 'STAGED_JSON=%s\n' "${local_dir%/}/.staged.json"
}

native_hdfs_strict() {
  local dir="$1"
  python - "$dir" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
staged = json.loads((root / ".staged.json").read_text())
manifest = root / "MANIFEST.jsonl"
manifest_md5 = hashlib.md5(manifest.read_bytes()).hexdigest()
if manifest_md5.lower() != staged["manifest_md5"].lower():
    raise SystemExit(f"manifest md5 mismatch for {root}: {manifest_md5} != {staged['manifest_md5']}")
count = 0
for line in manifest.read_text().splitlines():
    if not line:
        continue
    row = json.loads(line)
    rel = row["file"]
    if rel.startswith("/") or ".." in Path(rel).parts:
        raise SystemExit(f"unsafe manifest path: {rel}")
    path = root / rel
    data = path.read_bytes()
    if len(data) != int(row["size"]):
        raise SystemExit(f"size mismatch for {path}")
    md5 = hashlib.md5(data).hexdigest()
    if md5.lower() != row["md5"].lower():
        raise SystemExit(f"md5 mismatch for {path}: {md5} != {row['md5']}")
    count += 1
if count != int(staged.get("file_count", count)):
    raise SystemExit(f"file count mismatch for {root}")
print(f"HDFS_STRICT_DIR={root} SOURCE_URI={staged['source_uri']} MANIFEST_MD5={manifest_md5} FILES={count} MODE=full")
PY
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi
pkg="$1"
shift
key=""
hdfs_uri=""
parallel=2
while [[ $# -gt 0 ]]; do
  case "$1" in
    --key)
      key="${2:?--key requires a value}"
      shift 2
      ;;
    --hdfs-uri)
      hdfs_uri="${2:?--hdfs-uri requires a value}"
      shift 2
      ;;
    --parallel)
      parallel="${2:?--parallel requires a value}"
      shift 2
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

if ! package_json "$pkg" >/dev/null; then
  printf 'unknown native package: %s\n' "$pkg" >&2
  package_names >&2
  exit 2
fi
if [[ -z "$key" ]]; then
  printf '--key is required\n' >&2
  usage
  exit 2
fi

if [[ "${GLM53_NATIVE_INSTALL_IN_ROOTFS:-0}" != "1" ]]; then
  if [[ -z "$hdfs_uri" ]]; then
    # shellcheck source=experiments/glm53_flash_sft_ladder/lib/waystone.sh
    source "${repo_root}/experiments/glm53_flash_sft_ladder/lib/waystone.sh"
    hdfs_uri="$(ws_child_uri "native-wheels/${pkg}/${key}")"
  fi
  stage_dir="${scratch_root}/stage/${pkg}/${key}"
  native_stage_tree "$hdfs_uri" "$stage_dir"
  stage_dir_inner="$stage_dir"
  case "$stage_dir" in
    "$repo_root"/*) stage_dir_inner="/workspace/torchtitan/${stage_dir#"$repo_root"/}" ;;
  esac
  inner_cmd=$(
    printf 'unset CC CXX CUDAHOSTCXX; cd /workspace/torchtitan && GLM53_NATIVE_INSTALL_IN_ROOTFS=1 GLM53_NATIVE_STAGE_DIR=%q %q %q --key %q' \
      "$stage_dir_inner" "experiments/glm53_native/install_native.sh" "$pkg" "$key"
  )
  TORCHTITAN_ROOTFS_ENV="$rootfs_env" "${repo_root}/scripts/rootfs/enter_rootfs.sh" \
    --rootfs-store "$rootfs_store" -- bash -lc "$inner_cmd"
  exit 0
fi

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  printf 'inner native install must run inside the TorchTitan rootfs\n' >&2
  exit 1
fi
if [[ "${TORCHTITAN_ROOTFS_ENV:-}" != "$rootfs_env" ]]; then
  printf 'native install expected TORCHTITAN_ROOTFS_ENV=%s, found %s\n' \
    "$rootfs_env" "${TORCHTITAN_ROOTFS_ENV:-}" >&2
  exit 1
fi

stage_dir="${GLM53_NATIVE_STAGE_DIR:?GLM53_NATIVE_STAGE_DIR is required inside rootfs}"
mapfile -t wheels < <(find "$stage_dir" -maxdepth 1 -type f -name '*.whl' | sort)
if [[ "${#wheels[@]}" -ne 1 ]]; then
  printf 'expected exactly one wheel under %s, found %s\n' "$stage_dir" "${#wheels[@]}" >&2
  exit 1
fi
receipt="$(find "$stage_dir" -maxdepth 1 -type f -name "${pkg}-${key}.json" | sort | head -1)"
if [[ -z "$receipt" ]]; then
  printf 'missing receipt %s-%s.json under %s\n' "$pkg" "$key" "$stage_dir" >&2
  exit 1
fi

python -m pip install --no-deps --force-reinstall "${wheels[0]}"
pkg_file="${stage_dir}/package-${pkg}.json"
package_json "$pkg" > "$pkg_file"
verify_import="$(json_field "$pkg_file" verify_import)"
python -c "$verify_import"

keys_file="${GLM53_NATIVE_KEYS_FILE:-$keys_file_default}"
python - "$keys_file" "$pkg" "$key" "$receipt" "$stage_dir/.staged.json" <<'PY'
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

keys_path = Path(sys.argv[1])
pkg = sys.argv[2]
key = sys.argv[3]
receipt_path = Path(sys.argv[4])
staged_path = Path(sys.argv[5])
receipt = json.loads(receipt_path.read_text())
staged = json.loads(staged_path.read_text())
if keys_path.exists():
    payload = json.loads(keys_path.read_text())
else:
    payload = {
        "schema_version": 1,
        "rootfs_env": os.environ.get("TORCHTITAN_ROOTFS_ENV", ""),
        "rootfs_store_id": os.environ.get("TORCHTITAN_ROOTFS_STORE_ID", ""),
        "packages": {},
    }
payload["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
payload["packages"][pkg] = {
    "build_key": key,
    "receipt": receipt,
    "source_uri": staged.get("source_uri"),
    "manifest_md5": staged.get("manifest_md5"),
    "installed_at": payload["updated_at"],
}
keys_path.parent.mkdir(parents=True, exist_ok=True)
tmp = keys_path.with_suffix(keys_path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(keys_path)
print(json.dumps(payload["packages"][pkg], indent=2, sort_keys=True))
PY
