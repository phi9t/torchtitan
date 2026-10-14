#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOTFS_DIR="$REPO_ROOT/scripts/rootfs"

# shellcheck source=scripts/rootfs/rootfs_target.sh
source "$ROOTFS_DIR/rootfs_target.sh"

DEFAULT_TAG="torchtitan-rootfs:local"
TAG="${DEFAULT_TAG}"
TAG_EXPLICIT=0
DEST="$ROOTFS_DIR/rootfs"
STORE_ROOT=""
REBUILD=0
BASE_IMAGE="ghcr.io/pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime"
MISE_VERSION="2025.8.16"
TEST_STAGED_ROOTFS=0
VARIANT="default"
NATIVE_LOCK="$ROOTFS_DIR/native_variant.lock"

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
  --variant NAME  Rootfs recipe variant: default or native. The native variant
                  adds pinned ccache, CUDA linker libs, and CUTLASS headers for
                  building C++/CUDA extensions. Defaults to default.
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
      TAG_EXPLICIT=1
      shift 2
      ;;
    --tag=*)
      TAG="${1#*=}"
      TAG_EXPLICIT=1
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
    --variant)
      VARIANT="${2:?--variant requires a value}"
      shift 2
      ;;
    --variant=*)
      VARIANT="${1#*=}"
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

case "${VARIANT}" in
  default|native) ;;
  *) die "unknown rootfs variant: ${VARIANT}" ;;
esac

if [[ "${VARIANT}" == "native" ]]; then
  [[ -f "${NATIVE_LOCK}" ]] || die "native rootfs lock is missing: ${NATIVE_LOCK}"
  # shellcheck source=scripts/rootfs/native_variant.lock
  source "${NATIVE_LOCK}"
  if [[ "${TAG_EXPLICIT}" -eq 0 ]]; then
    TAG="torchtitan-rootfs:native"
  fi
fi

json_escape() {
  python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$1"
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

sha256_rootfs_tree() {
  tar --sort=name \
    --mtime='UTC 1970-01-01' \
    --owner=0 \
    --group=0 \
    --numeric-owner \
    -C "$1" \
    -cf - . | sha256sum | awk '{print $1}'
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
  if [[ "${VARIANT}" == "native" ]]; then
    local variant_json native_lock_json native_lock_sha_json cuda_pip_json ccache_json cutlass_version_json cutlass_url_json cutlass_sha_json
    variant_json="$(json_escape "${VARIANT}")"
    native_lock_json="$(json_escape "${NATIVE_LOCK}")"
    native_lock_sha_json="$(json_escape "$(sha256_file "${NATIVE_LOCK}")")"
    cuda_pip_json="$(json_escape "${ROOTFS_NATIVE_CUDA_PIP_PACKAGES}")"
    ccache_json="$(json_escape "${ROOTFS_NATIVE_CCACHE_APT_VERSION}")"
    cutlass_version_json="$(json_escape "${ROOTFS_NATIVE_CUTLASS_VERSION}")"
    cutlass_url_json="$(json_escape "${ROOTFS_NATIVE_CUTLASS_URL}")"
    cutlass_sha_json="$(json_escape "${ROOTFS_NATIVE_CUTLASS_SHA256}")"
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
  "mutable_rootfs_allowed": false,
  "variant": ${variant_json},
  "native_lock": {
    "path": ${native_lock_json},
    "sha256": ${native_lock_sha_json}
  },
  "native_packages": {
    "apt": {
      "ccache": ${ccache_json}
    },
    "pip": ${cuda_pip_json},
    "cutlass": {
      "version": ${cutlass_version_json},
      "url": ${cutlass_url_json},
      "sha256": ${cutlass_sha_json}
    }
  }
}
EOF
    return
  fi
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

build_default_image() {
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
}

build_native_image() {
  docker build -t "$TAG" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg MISE_VERSION="$MISE_VERSION" \
    --build-arg ROOTFS_NATIVE_CCACHE_APT_VERSION="$ROOTFS_NATIVE_CCACHE_APT_VERSION" \
    --build-arg ROOTFS_NATIVE_CUDA_PIP_PACKAGES="$ROOTFS_NATIVE_CUDA_PIP_PACKAGES" \
    --build-arg ROOTFS_NATIVE_CUTLASS_URL="$ROOTFS_NATIVE_CUTLASS_URL" \
    --build-arg ROOTFS_NATIVE_CUTLASS_SHA256="$ROOTFS_NATIVE_CUTLASS_SHA256" \
    -f - "$ROOTFS_DIR" <<'DOCKERFILE'
ARG BASE_IMAGE
ARG MISE_VERSION
ARG ROOTFS_NATIVE_CCACHE_APT_VERSION
ARG ROOTFS_NATIVE_CUDA_PIP_PACKAGES
ARG ROOTFS_NATIVE_CUTLASS_URL
ARG ROOTFS_NATIVE_CUTLASS_SHA256
FROM ${BASE_IMAGE}
ARG MISE_VERSION
ARG ROOTFS_NATIVE_CCACHE_APT_VERSION
ARG ROOTFS_NATIVE_CUDA_PIP_PACKAGES
ARG ROOTFS_NATIVE_CUTLASS_URL
ARG ROOTFS_NATIVE_CUTLASS_SHA256
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        bash build-essential g++ clang libclang-dev llvm-dev \
        git curl ca-certificates rsync pkg-config \
        libaio-dev libibverbs-dev librdmacm-dev libnuma-dev \
        libssl-dev libffi-dev libgl1 libglib2.0-0 \
        ninja-build cmake \
        ccache=${ROOTFS_NATIVE_CCACHE_APT_VERSION} && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN curl -fsSL -o /tmp/mise.tar.gz \
        "https://github.com/jdx/mise/releases/download/v${MISE_VERSION}/mise-v${MISE_VERSION}-linux-x64.tar.gz" && \
    mkdir -p /tmp/mise-extract && \
    tar -C /tmp/mise-extract -xzf /tmp/mise.tar.gz mise && \
    install -m 0755 /tmp/mise-extract/mise/bin/mise /usr/local/bin/mise && \
    rm -rf /tmp/mise.tar.gz /tmp/mise-extract && \
    /usr/local/bin/mise --version

RUN pip install --break-system-packages --no-cache-dir \
        ${ROOTFS_NATIVE_CUDA_PIP_PACKAGES}
RUN set -euo pipefail; \
    site="$(python -c 'import site;print(site.getsitepackages()[0])')"; \
    cu="$site/nvidia/cu13"; \
    test -x "$cu/bin/nvcc" || { echo "nvcc missing at $cu/bin/nvcc" >&2; exit 1; }; \
    test -x "$cu/bin/cuobjdump" || { echo "cuobjdump missing at $cu/bin/cuobjdump" >&2; exit 1; }; \
    test -x "$cu/bin/ptxas" || { echo "ptxas missing at $cu/bin/ptxas" >&2; exit 1; }; \
    test -x "$cu/bin/nvlink" || { echo "nvlink missing at $cu/bin/nvlink" >&2; exit 1; }; \
    test -x "$cu/bin/fatbinary" || { echo "fatbinary missing at $cu/bin/fatbinary" >&2; exit 1; }; \
    test -x "$cu/nvvm/bin/cicc" || { echo "cicc missing at $cu/nvvm/bin/cicc" >&2; exit 1; }; \
    test -f "$cu/include/cuda_runtime.h" || { echo "cuda headers missing" >&2; exit 1; }; \
    test -f "$cu/include/cublas_v2.h" || { echo "cuBLAS headers missing" >&2; exit 1; }; \
    test -f "$cu/include/nvrtc.h" || { echo "NVRTC headers missing" >&2; exit 1; }; \
    ln -sfn lib "$cu/lib64"; \
    ln -sfn "$cu" /opt/cuda-synth; \
    for soname in libcublas.so libcublasLt.so libcudart.so libnvrtc.so; do \
      target="$(find "$cu/lib" -maxdepth 1 -type f -name "${soname}.*" | sort -V | tail -n 1)"; \
      test -n "$target" || { echo "${soname} versioned library missing" >&2; exit 1; }; \
      ln -sfn "$(basename "$target")" "$cu/lib/${soname}"; \
    done; \
    python -c 'import importlib.metadata as m, sys; expected={"nvidia-cuda-nvcc":"13.2.86","nvidia-cuda-cuobjdump":"13.2.86","nvidia-nvvm":"13.2.86","nvidia-cuda-crt":"13.2.86","nvidia-cuda-cccl":"13.2.86","nvidia-cuda-runtime":"13.2.86","nvidia-cuda-nvrtc":"13.2.86","nvidia-cublas":"13.2.2.2"}; mismatches=[f"{name}: {m.version(name)} != {want}" for name, want in expected.items() if m.version(name) != want]; sys.exit("\n".join(mismatches) if mismatches else 0)'; \
    export PATH="/opt/cuda-synth/bin:/usr/lib/ccache:/usr/local/bin:/usr/bin:/bin"; \
    for tool in nvcc cuobjdump ptxas nvlink fatbinary; do \
      first="$(command -v "$tool")"; \
      test "$first" = "/opt/cuda-synth/bin/$tool" || { echo "$tool resolves to $first" >&2; exit 1; }; \
    done; \
    for tool in nvcc cuobjdump ptxas nvlink fatbinary; do \
      "/opt/cuda-synth/bin/$tool" --version | grep -E '13\.2' >/dev/null \
        || { echo "$tool is not CUDA 13.2" >&2; exit 1; }; \
    done; \
    test -x /opt/cuda-synth/nvvm/bin/cicc || { echo "cicc missing under /opt/cuda-synth" >&2; exit 1; }; \
    printf '%s\n' \
      'extern "C" __global__ void torchtitan_sm100_probe(float *x) {' \
      '  x[threadIdx.x] = x[threadIdx.x] + 1.0f;' \
      '}' > /tmp/torchtitan-sm100-probe.cu; \
    /opt/cuda-synth/bin/nvcc -arch=sm_100 -cubin /tmp/torchtitan-sm100-probe.cu -o /tmp/torchtitan-sm100-probe.cubin; \
    rm -f /tmp/torchtitan-sm100-probe.cu /tmp/torchtitan-sm100-probe.cubin

RUN curl -fsSL -o /tmp/cutlass.tar.gz "${ROOTFS_NATIVE_CUTLASS_URL}" && \
    printf '%s  /tmp/cutlass.tar.gz\n' "${ROOTFS_NATIVE_CUTLASS_SHA256}" | sha256sum -c - && \
    mkdir -p /opt/cutlass && \
    tar -C /opt/cutlass --strip-components=1 -xzf /tmp/cutlass.tar.gz && \
    rm -f /tmp/cutlass.tar.gz && \
    test -f /opt/cutlass/include/cutlass/cutlass.h && \
    test -f /opt/cutlass/include/cute/arch/tmem_capacity_sm100.hpp && \
    site="$(python -c 'import site;print(site.getsitepackages()[0])')" && \
    ln -sfn /opt/cutlass/include/cutlass "$site/nvidia/cu13/include/cutlass" && \
    ln -sfn /opt/cutlass/include/cute "$site/nvidia/cu13/include/cute" && \
    test -f "$site/nvidia/cu13/include/cutlass/cutlass.h" && \
    test -f "$site/nvidia/cu13/include/cute/arch/tmem_capacity_sm100.hpp"

ENV CUDA_HOME=/opt/cuda-synth CUDA_PATH=/opt/cuda-synth
ENV PATH=/opt/cuda-synth/bin:/usr/lib/ccache:/usr/local/bin:/usr/bin:/bin
DOCKERFILE
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
  log "building $TAG ($VARIANT variant) from $BASE_IMAGE"
  if [[ "${VARIANT}" == "native" ]]; then
    build_native_image
  else
    build_default_image
  fi
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
ROOTFS_TREE_DIGEST="$(sha256_rootfs_tree "$STAGE")"
STORE_ID="rootfs-${ROOTFS_TREE_DIGEST}"
IMAGE_ID="${TAG}"
EXPORTED_DIGEST="sha256:${ROOTFS_TREE_DIGEST}"
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
