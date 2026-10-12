#!/usr/bin/env bash
# Install memray / Scalene / Fil into the rootfs interpreter that Trainer uses.
# Rebuilds wipe /usr/local; rerun this after a new rootfs export.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  export TORCHTITAN_ROOTFS_NETWORK=networked
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- \
    /bin/bash -lc \
    'cd /workspace/torchtitan && exec .claude/skills/attaching-live-training/scripts/install_python_profilers.sh'
fi

python -m pip install --break-system-packages memray scalene filprofiler
python -c 'import memray, scalene, filprofiler; print("memray", memray.__version__)'
command -v memray
command -v scalene
command -v fil-profile
