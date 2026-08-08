#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOTFS="$REPO_ROOT/scripts/rootfs/rootfs"

usage() {
  cat <<'EOF'
Usage: scripts/rootfs/enter_rootfs.sh [options] [-- <command> [args...]]

Enter the TorchTitan bwrap rootfs with this checkout mounted read-write at
/workspace/torchtitan. With no command, opens an interactive bash shell.

Options:
  --rootfs DIR    Rootfs directory to enter (default: scripts/rootfs/rootfs).
  -h, --help      Show this help and exit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rootfs)
      ROOTFS="${2:?--rootfs requires a value}"
      shift 2
      ;;
    --rootfs=*)
      ROOTFS="${1#*=}"
      shift
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

die() {
  printf '\033[1;31merror: %s\033[0m\n' "$*" >&2
  exit 1
}

command -v bwrap >/dev/null || die "bwrap not found on host"

if [[ ! -x "$ROOTFS/bin/bash" ]]; then
  printf '\033[1m== rootfs not found at %s; building it ==\033[0m\n' "$ROOTFS" >&2
  "$REPO_ROOT/scripts/rootfs/build_rootfs.sh" --dest "$ROOTFS"
  [[ -x "$ROOTFS/bin/bash" ]] || die "rootfs build did not produce a usable rootfs at $ROOTFS"
fi

REPO_MNT=/workspace/torchtitan
HOST_LIBDIR=/usr/lib/x86_64-linux-gnu

bwrap_args=(
  --bind "$ROOTFS" /
  --proc /proc
  --tmpfs /tmp
  --dev /dev
  --bind "$REPO_ROOT" "$REPO_MNT"
  --unshare-all --share-net
  --die-with-parent
  --chdir "$REPO_MNT"
)

for f in /etc/resolv.conf /etc/hosts; do
  [[ -e "$f" ]] && bwrap_args+=(--ro-bind "$f" "$f")
done

shopt -s nullglob
for dev in /dev/nvidia* /dev/nvidia-caps; do
  bwrap_args+=(--dev-bind "$dev" "$dev")
done
for lib in "$HOST_LIBDIR"/libcuda.so* "$HOST_LIBDIR"/libnvidia-*.so*; do
  bwrap_args+=(--ro-bind "$lib" "$lib")
done
if [[ -x /usr/bin/nvidia-smi ]]; then
  bwrap_args+=(--ro-bind /usr/bin/nvidia-smi /usr/bin/nvidia-smi)
fi
shopt -u nullglob

bwrap_args+=(
  --setenv PATH "/opt/cuda-synth/bin:/usr/local/bin:/usr/bin:/bin"
  --setenv CUDA_HOME /opt/cuda-synth
  --setenv CUDA_PATH /opt/cuda-synth
  --setenv LD_LIBRARY_PATH "$HOST_LIBDIR:/opt/cuda-synth/lib64"
  --setenv TORCHTITAN_IN_ROOTFS 1
  --setenv HOME /root
)

if [[ "${CUDA_VISIBLE_DEVICES+set}" == set ]]; then
  bwrap_args+=(--setenv CUDA_VISIBLE_DEVICES "$CUDA_VISIBLE_DEVICES")
fi
bwrap_args+=(--setenv NVIDIA_VISIBLE_DEVICES "${NVIDIA_VISIBLE_DEVICES:-all}")

if [[ $# -eq 0 ]]; then
  exec bwrap "${bwrap_args[@]}" /bin/bash -l
else
  exec bwrap "${bwrap_args[@]}" "$@"
fi
