#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_REL="experiments/falcon/run.sh"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- \
    /bin/bash -lc \
    'cd /workspace/torchtitan && exec experiments/falcon/run.sh "$@"' \
    "${SCRIPT_REL}" "$@"
fi

cd "${REPO_ROOT}"

command="${1:-overfit}"
if [[ "${command}" == "overfit" ]]; then
  shift || true
  exec python -m experiments.falcon.overfit "$@"
fi

if [[ "${command}" == "train" ]]; then
  shift || true
  export MODULE="${MODULE:-falcon}"
  export CONFIG="${CONFIG:-falcon_tiny_overfit}"
  export NGPU="${NGPU:-1}"
  export COMM_MODE="${COMM_MODE:-fake_backend}"
  exec ./run_train.sh "$@"
fi

if [[ "${command}" == "perf" ]]; then
  shift || true
  exec python -m experiments.falcon.perf_base "$@"
fi

if [[ "${command}" == "nsys" ]]; then
  shift || true
  nsys_bin="$(command -v nsys || true)"
  if [[ -z "${nsys_bin}" ]]; then
    echo "nsys is not on PATH inside the rootfs; bind /usr/local/cuda into enter_rootfs.sh" >&2
    exit 2
  fi
  out="${NSYS_OUTPUT:-experiments/falcon/results/tooling/nsys_$(date -u +%Y%m%dT%H%M%SZ)}"
  mkdir -p "$(dirname -- "$out")"
  export TMPDIR="${TMPDIR:-/project/tmp}"
  exec "${nsys_bin}" profile \
    --output "$out" \
    --force-overwrite true \
    --trace cuda,nvtx,osrt \
    --sample cpu \
    -- python -m experiments.falcon.perf_base "$@"
fi

echo "unknown Falcon command: ${command}" >&2
echo "usage: experiments/falcon/run.sh [overfit|train|perf|nsys]" >&2
exit 2
