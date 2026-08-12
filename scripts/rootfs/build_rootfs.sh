#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOTFS_DIR="$REPO_ROOT/scripts/rootfs"

# shellcheck source=scripts/rootfs/rootfs_target.sh
source "$ROOTFS_DIR/rootfs_target.sh"

TAG="torchtitan-rootfs:local"
DEST="$ROOTFS_DIR/rootfs"
REBUILD=0
BASE_IMAGE="ghcr.io/pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime"

usage() {
  cat <<'EOF'
Usage: scripts/rootfs/build_rootfs.sh [options]

Build and export the TorchTitan bwrap rootfs. Host-side work is limited to
shell, docker, and tar; Python setup happens later inside the rootfs.

Options:
  --rebuild       Force a docker rebuild even if the image tag exists.
  --tag TAG       Docker image tag to build/use (default: torchtitan-rootfs:local).
  --dest DIR      Directory to export the flattened rootfs into
                  (default: scripts/rootfs/rootfs).
  -h, --help      Show this help and exit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild)
      REBUILD=1
      shift
      ;;
    --tag)
      TAG="${2:?--tag requires a value}"
      shift 2
      ;;
    --tag=*)
      TAG="${1#*=}"
      shift
      ;;
    --dest)
      DEST="${2:?--dest requires a value}"
      shift 2
      ;;
    --dest=*)
      DEST="${1#*=}"
      shift
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

log() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() {
  printf '\033[1;31merror: %s\033[0m\n' "$*" >&2
  exit 1
}

command -v docker >/dev/null || die "docker not found on host"

# Resolve the destination through the shared fail-closed resolver before any
# Docker work or filesystem mutation. An arbitrary --dest that is not in the
# allowlist, is symlinked, or has a symlinked parent is refused here rather
# than becoming a deletion target below.
DEST="$(rootfs_resolve_managed_dest "$DEST")"

image_exists() { docker image inspect "$TAG" >/dev/null 2>&1; }

if image_exists && [[ "$REBUILD" -eq 0 ]]; then
  log "image $TAG already exists (use --rebuild to force)"
else
  log "building $TAG from $BASE_IMAGE"
  docker build -t "$TAG" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -f - "$ROOTFS_DIR" <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        bash build-essential g++ clang libclang-dev llvm-dev \
        git curl ca-certificates rsync pkg-config \
        libaio-dev libibverbs-dev librdmacm-dev libnuma-dev \
        libssl-dev libffi-dev libgl1 libglib2.0-0 \
        ninja-build cmake && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN pip install --break-system-packages --no-cache-dir \
        nvidia-cuda-nvcc nvidia-cuda-cccl
RUN set -euo pipefail; \
    site="$(python -c 'import site;print(site.getsitepackages()[0])')"; \
    cu="$site/nvidia/cu13"; \
    test -x "$cu/bin/nvcc" || { echo "nvcc missing at $cu/bin/nvcc" >&2; exit 1; }; \
    test -f "$cu/include/cuda_runtime.h" || { echo "cuda headers missing" >&2; exit 1; }; \
    ln -sfn lib "$cu/lib64"; \
    ln -sfn "$cu" /opt/cuda-synth; \
    /opt/cuda-synth/bin/nvcc --version

ENV CUDA_HOME=/opt/cuda-synth CUDA_PATH=/opt/cuda-synth
ENV PATH=/opt/cuda-synth/bin:/usr/local/bin:/usr/bin:/bin
DOCKERFILE
fi

log "exporting $TAG to $DEST"
mkdir -p "$(dirname "$DEST")"
STAGE="$(mktemp -d "${DEST%/*}/.rootfs.stage.XXXXXX")"
cid="$(docker create "$TAG")"
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true; rm -rf "$STAGE"' EXIT
docker export "$cid" | tar -C "$STAGE" -xf -
[[ -x "$STAGE/bin/bash" ]] || die "export produced an unusable rootfs (no bin/bash)"
# Mark the staged tree as builder-owned before it can replace the destination,
# so a later rebuild can prove the target is a managed rootfs.
rootfs_write_ownership_marker "$STAGE"
# Only replace an existing destination that carries a valid ownership marker;
# an unrelated or ambiguous existing directory is never removed.
if [[ -e "$DEST" ]]; then
  rootfs_assert_removable "$DEST"
  rm -rf "$DEST"
fi
mv "$STAGE" "$DEST"
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT

log "rootfs ready: $DEST"
