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
STORE_ROOT=""
REBUILD=0
BASE_IMAGE="ghcr.io/pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime"
MISE_VERSION="2025.8.16"
TEST_STAGED_ROOTFS=0

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
  --store DIR     Publish the exported rootfs under DIR/content/<store_id> and
                  atomically select it with DIR/selected.json.
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
    --store)
      STORE_ROOT="${2:?--store requires a value}"
      shift 2
      ;;
    --store=*)
      STORE_ROOT="${1#*=}"
      shift
      ;;
    --test-staged-rootfs)
      TEST_STAGED_ROOTFS=1
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

json_escape() {
  python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$1"
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

write_build_manifest() {
  local rootfs="$1"
  local store_id="$2"
  local image_id="$3"
  local exported_digest="$4"
  local base_image_json tag_json store_id_json image_json digest_json mise_json
  base_image_json="$(json_escape "${BASE_IMAGE}")"
  tag_json="$(json_escape "${TAG}")"
  store_id_json="$(json_escape "${store_id}")"
  image_json="$(json_escape "${image_id}")"
  digest_json="$(json_escape "${exported_digest}")"
  mise_json="$(json_escape "${MISE_VERSION}")"
  cat > "$(rootfs_manifest_path "${rootfs}")" <<EOF
{
  "schema_version": 1,
  "kind": "rootfs_build_manifest",
  "store_id": ${store_id_json},
  "base_image": ${base_image_json},
  "image_tag": ${tag_json},
  "image_id": ${image_json},
  "exported_rootfs_digest": ${digest_json},
  "mise_version": "${MISE_VERSION}",
  "mutable_rootfs_allowed": false
}
EOF
}

create_test_staged_rootfs() {
  local stage="$1"
  mkdir -p "${stage}/bin"
  cat > "${stage}/bin/bash" <<'EOF'
#!/usr/bin/env bash
EOF
  chmod 0755 "${stage}/bin/bash"
  mkdir -p "${stage}/usr/local/bin"
  cat > "${stage}/usr/local/bin/mise" <<'EOF'
#!/usr/bin/env bash
printf 'mise test stub\n'
EOF
  chmod 0755 "${stage}/usr/local/bin/mise"
}

publish_store_entry() {
  local stage="$1"
  local store_root="$2"
  local store_id="$3"
  if [[ -e "${store_root}" && ! -d "${store_root}" ]]; then
    die "rootfs store exists and is not a directory: ${store_root}"
  fi
  if [[ -L "${store_root}" ]]; then
    die "rootfs store must not be a symlink: ${store_root}"
  fi
  mkdir -p "${store_root}/${ROOTFS_STORE_CONTENT_DIRNAME}"
  local target="${store_root}/${ROOTFS_STORE_CONTENT_DIRNAME}/${store_id}"
  if [[ -e "${target}" ]]; then
    rootfs_assert_removable "${target}"
    rm -rf "${target}"
  fi
  mv "${stage}" "${target}"
  local selection_tmp
  selection_tmp="$(mktemp "${store_root}/.${ROOTFS_STORE_SELECTION_FILENAME}.XXXXXX")"
  cat > "${selection_tmp}" <<EOF
{
  "schema_version": 1,
  "kind": "rootfs_selection",
  "store_id": "$(printf '%s' "${store_id}")"
}
EOF
  mv "${selection_tmp}" "${store_root}/${ROOTFS_STORE_SELECTION_FILENAME}"
  printf '%s\n' "${target}"
}

if [[ "${TEST_STAGED_ROOTFS}" -eq 0 ]]; then
  command -v docker >/dev/null || die "docker not found on host"
fi

# Resolve the destination through the shared fail-closed resolver before any
# Docker work or filesystem mutation. An arbitrary --dest that is not in the
# allowlist, is symlinked, or has a symlinked parent is refused here rather
# than becoming a deletion target below.
if [[ -z "${STORE_ROOT}" ]]; then
  DEST="$(rootfs_resolve_managed_dest "$DEST")"
else
  STORE_PARENT="$(dirname -- "${STORE_ROOT}")"
  [[ -d "${STORE_PARENT}" ]] || die "rootfs store parent is not an existing directory: ${STORE_PARENT}"
  STORE_ROOT="$(cd -P -- "${STORE_PARENT}" && pwd)/$(basename -- "${STORE_ROOT}")"
fi

image_exists() { docker image inspect "$TAG" >/dev/null 2>&1; }

if [[ "${TEST_STAGED_ROOTFS}" -eq 1 ]]; then
  log "using test staged rootfs"
elif image_exists && [[ "$REBUILD" -eq 0 ]]; then
  log "image $TAG already exists (use --rebuild to force)"
else
  log "building $TAG from $BASE_IMAGE"
  docker build -t "$TAG" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg MISE_VERSION="$MISE_VERSION" \
    -f - "$ROOTFS_DIR" <<'DOCKERFILE'
ARG BASE_IMAGE
ARG MISE_VERSION
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

RUN curl -fsSL -o /tmp/mise.tar.gz \
        "https://github.com/jdx/mise/releases/download/v${MISE_VERSION}/mise-v${MISE_VERSION}-linux-x64.tar.gz" && \
    tar -C /usr/local/bin -xzf /tmp/mise.tar.gz mise && \
    rm -f /tmp/mise.tar.gz && \
    /usr/local/bin/mise --version

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

if [[ -n "${STORE_ROOT}" ]]; then
  log "exporting $TAG to managed store $STORE_ROOT"
  mkdir -p "${STORE_ROOT}"
  STAGE="$(mktemp -d "${STORE_ROOT}/.rootfs.stage.XXXXXX")"
else
  log "exporting $TAG to $DEST"
  mkdir -p "$(dirname "$DEST")"
  STAGE="$(mktemp -d "${DEST%/*}/.rootfs.stage.XXXXXX")"
fi
cid=""
trap '[[ -n "${cid:-}" ]] && docker rm -f "$cid" >/dev/null 2>&1 || true; rm -rf "$STAGE"' EXIT
if [[ "${TEST_STAGED_ROOTFS}" -eq 1 ]]; then
  create_test_staged_rootfs "$STAGE"
else
  cid="$(docker create "$TAG")"
  docker export "$cid" | tar -C "$STAGE" -xf -
fi
[[ -x "$STAGE/bin/bash" ]] || die "export produced an unusable rootfs (no bin/bash)"
# Mark the staged tree as builder-owned before it can replace the destination,
# so a later rebuild can prove the target is a managed rootfs.
rootfs_write_ownership_marker "$STAGE"
STORE_ID="rootfs-$(sha256_file "$STAGE/bin/bash")"
IMAGE_ID="${TAG}"
EXPORTED_DIGEST="sha256:$(sha256_file "$STAGE/bin/bash")"
write_build_manifest "$STAGE" "$STORE_ID" "$IMAGE_ID" "$EXPORTED_DIGEST"
if [[ -n "${STORE_ROOT}" ]]; then
  TARGET="$(publish_store_entry "$STAGE" "$STORE_ROOT" "$STORE_ID")"
  trap '[[ -n "${cid:-}" ]] && docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
  log "rootfs ready: $TARGET"
  exit 0
fi
# Only replace an existing destination that carries a valid ownership marker;
# an unrelated or ambiguous existing directory is never removed.
if [[ -e "$DEST" ]]; then
  rootfs_assert_removable "$DEST"
  rm -rf "$DEST"
fi
mv "$STAGE" "$DEST"
trap '[[ -n "${cid:-}" ]] && docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT

log "rootfs ready: $DEST"
