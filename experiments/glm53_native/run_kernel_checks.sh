#!/usr/bin/env bash
# Run GLM kernel convention checks inside glm53_ep.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
lock_file="${script_dir}/packages.lock.json"
json_output="${GLM53_KERNEL_CHECK_JSON:-${repo_root}/.scratch/glm53-native/kernel-checks.json}"

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

rootfs_store="${GLM53_NATIVE_ROOTFS_STORE:-$(json_field "$lock_file" rootfs_store)}"
rootfs_env="${GLM53_NATIVE_ROOTFS_ENV:-$(json_field "$lock_file" default_rootfs_env)}"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  json_output_inner="$json_output"
  case "$json_output" in
    "$repo_root"/*) json_output_inner="/workspace/torchtitan/${json_output#"$repo_root"/}" ;;
  esac
  inner_cmd=$(
    printf 'unset CC CXX CUDAHOSTCXX; cd /workspace/torchtitan && GLM53_KERNEL_CHECK_JSON=%q %q %q' \
      "$json_output_inner" "experiments/glm53_native/run_kernel_checks.sh" "$@"
  )
  TORCHTITAN_ROOTFS_ENV="$rootfs_env" "${repo_root}/scripts/rootfs/enter_rootfs.sh" \
    --rootfs-store "$rootfs_store" -- bash -lc "$inner_cmd"
  exit 0
fi

python -m torchtitan.experiments.glm53_kernels.checks \
  --json-output "${GLM53_KERNEL_CHECK_JSON:-/workspace/torchtitan/.scratch/glm53-native/kernel-checks.json}" \
  "$@"

