#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOTFS="$REPO_ROOT/scripts/rootfs/rootfs"
ROOTFS_DEFAULT="$ROOTFS"
ROOTFS_EXPLICIT=0

# shellcheck source=scripts/rootfs/rootfs_target.sh
source "$REPO_ROOT/scripts/rootfs/rootfs_target.sh"

usage() {
  cat <<'EOF'
Usage: scripts/rootfs/enter_rootfs.sh [options] [-- <command> [args...]]

Enter the TorchTitan bwrap rootfs with this checkout mounted read-write at
/workspace/torchtitan. With no command, opens an interactive bash shell.

Options:
  --rootfs DIR    Rootfs directory to enter (default: scripts/rootfs/rootfs).
  -h, --help      Show this help and exit.

Environment:
  TORCHTITAN_ROOTFS_BIND_DOCKER=1
                  Bind the host Docker CLI and socket into the rootfs. This is
                  intended only for external harness probes that explicitly
                  require Docker, such as Harbor/Terminal-Bench.
                  Also binds this checkout at its host path so Docker daemon
                  bind mounts see the same paths as processes inside rootfs.
  TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY=1
                  Emit the resolved bwrap plan as JSON and exit before running
                  bwrap. Writes to TORCHTITAN_ROOTFS_PLAN_OUTPUT when set,
                  otherwise stdout.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rootfs)
      ROOTFS="${2:?--rootfs requires a value}"
      ROOTFS_EXPLICIT=1
      shift 2
      ;;
    --rootfs=*)
      ROOTFS="${1#*=}"
      ROOTFS_EXPLICIT=1
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

json_escape() {
  python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$1"
}

json_array() {
  python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@"
}

emit_bwrap_plan() {
  local output="${TORCHTITAN_ROOTFS_PLAN_OUTPUT:-}"
  local inner_argv_json
  inner_argv_json="$(json_array "$@")"
  local bwrap_argv_json
  bwrap_argv_json="$(json_array "${bwrap_args[@]}")"
  local entrypoint_json rootfs_json repo_root_json host_libdir_json cwd_json
  entrypoint_json="$(json_escape "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh")"
  rootfs_json="$(json_escape "$ROOTFS")"
  repo_root_json="$(json_escape "$REPO_ROOT")"
  host_libdir_json="$(json_escape "$HOST_LIBDIR")"
  cwd_json="$(json_escape "$REPO_MNT")"

  local plan
  plan="$(cat <<EOF
{
  "schema_version": 1,
  "entrypoint": ${entrypoint_json},
  "rootfs": {
    "path": ${rootfs_json},
    "explicit": $([[ "$ROOTFS_EXPLICIT" -eq 1 ]] && printf 'true' || printf 'false')
  },
  "cwd": ${cwd_json},
  "network_mode": "shared",
  "mounts": [
EOF
)"
  local first=1
  append_mount() {
    local item="$1"
    if [[ "$first" -eq 0 ]]; then
      plan+=$',\n'
    fi
    plan+="    ${item}"
    first=0
  }
  append_mount "{\"kind\":\"bind\",\"source\":${rootfs_json},\"target\":\"/\",\"writable\":true}"
  append_mount "{\"kind\":\"proc\",\"source\":\"procfs\",\"target\":\"/proc\",\"writable\":false}"
  append_mount "{\"kind\":\"tmpfs\",\"source\":\"tmpfs\",\"target\":\"/tmp\",\"writable\":true}"
  append_mount "{\"kind\":\"dev\",\"source\":\"devfs\",\"target\":\"/dev\",\"writable\":true}"
  append_mount "{\"kind\":\"bind\",\"source\":${repo_root_json},\"target\":\"${REPO_MNT}\",\"writable\":true}"
  for f in /etc/resolv.conf /etc/hosts; do
    if [[ -e "$f" ]]; then
      local escaped
      escaped="$(json_escape "$f")"
      append_mount "{\"kind\":\"ro-bind\",\"source\":${escaped},\"target\":${escaped},\"writable\":false}"
    fi
  done
  local dev_json_entries=()
  local lib_json_entries=()
  for dev in "${rootfs_plan_devices[@]}"; do
    local escaped_dev
    escaped_dev="$(json_escape "$dev")"
    append_mount "{\"kind\":\"dev-bind\",\"source\":${escaped_dev},\"target\":${escaped_dev},\"writable\":true}"
    dev_json_entries+=("$dev")
  done
  for lib in "${rootfs_plan_driver_libraries[@]}"; do
    local escaped_lib
    escaped_lib="$(json_escape "$lib")"
    append_mount "{\"kind\":\"ro-bind\",\"source\":${escaped_lib},\"target\":${escaped_lib},\"writable\":false}"
    lib_json_entries+=("$lib")
  done
  if [[ -n "${rootfs_plan_nvidia_smi:-}" ]]; then
    local nvidia_smi_json
    nvidia_smi_json="$(json_escape "$rootfs_plan_nvidia_smi")"
    append_mount "{\"kind\":\"ro-bind\",\"source\":${nvidia_smi_json},\"target\":${nvidia_smi_json},\"writable\":false}"
  fi
  if [[ "${TORCHTITAN_ROOTFS_BIND_DOCKER:-0}" == "1" ]]; then
    append_mount "{\"kind\":\"dir\",\"source\":\"dir\",\"target\":\"/run\",\"writable\":true}"
    append_mount "{\"kind\":\"bind\",\"source\":\"/var/run/docker.sock\",\"target\":\"/run/docker.sock\",\"writable\":true}"
    append_mount "{\"kind\":\"bind\",\"source\":${repo_root_json},\"target\":${repo_root_json},\"writable\":true}"
  fi

  local devices_json driver_libraries_json nvidia_visible_json ld_json path_json cuda_visible_json
  devices_json="$(json_array "${dev_json_entries[@]}")"
  driver_libraries_json="$(json_array "${lib_json_entries[@]}")"
  nvidia_visible_json="$(json_escape "${NVIDIA_VISIBLE_DEVICES:-all}")"
  ld_json="$(json_escape "${HOST_LIBDIR}:/opt/cuda-synth/lib64")"
  path_json="$(json_escape "/opt/cuda-synth/bin:/usr/local/bin:/usr/bin:/bin")"
  plan+=$'\n'
  plan+="  ],
  \"devices\": ${devices_json},
  \"driver_libraries\": ${driver_libraries_json},
  \"environment\": {
    \"PATH\": ${path_json},
    \"CUDA_HOME\": \"/opt/cuda-synth\",
    \"CUDA_PATH\": \"/opt/cuda-synth\",
    \"LD_LIBRARY_PATH\": ${ld_json},
    \"TORCHTITAN_IN_ROOTFS\": \"1\",
    \"HOME\": \"/root\",
    \"NVIDIA_VISIBLE_DEVICES\": ${nvidia_visible_json}"
  if [[ "${CUDA_VISIBLE_DEVICES+set}" == set ]]; then
    cuda_visible_json="$(json_escape "$CUDA_VISIBLE_DEVICES")"
    plan+=",
    \"CUDA_VISIBLE_DEVICES\": ${cuda_visible_json}"
  fi
  if [[ "${TORCHTITAN_ROOTFS_BIND_DOCKER:-0}" == "1" ]]; then
    plan+=",
    \"TORCHTITAN_ROOTFS_BIND_DOCKER\": \"1\",
    \"TORCHTITAN_ROOTFS_HOST_REPO_ROOT\": ${repo_root_json}"
  fi
  plan+="
  },
  \"inner_argv\": ${inner_argv_json},
  \"bwrap_argv\": ${bwrap_argv_json},
  \"host\": {
    \"repo_root\": ${repo_root_json},
    \"host_libdir\": ${host_libdir_json}
  }
}
"
  if [[ -n "$output" ]]; then
    mkdir -p "$(dirname -- "$output")"
    printf '%s' "$plan" > "$output"
  else
    printf '%s' "$plan"
  fi
}

if [[ ! -x "$ROOTFS/bin/bash" ]]; then
  # Fail-closed construction (runtime_preflight_roadmap.md Section 8.2, Wave
  # F0): implicit build is allowed only for the canonical default rootfs. A
  # missing custom --rootfs is an error, not an invitation to build, so a typo
  # cannot silently trigger Docker work or a deletion at an arbitrary path.
  if [[ "$ROOTFS_EXPLICIT" -eq 1 || "$ROOTFS" != "$ROOTFS_DEFAULT" ]]; then
    die "rootfs not found at $ROOTFS; refusing implicit build of a custom --rootfs (build it explicitly with scripts/rootfs/build_rootfs.sh)"
  fi
  printf '\033[1m== rootfs not found at %s; building default ==\033[0m\n' "$ROOTFS" >&2
  "$REPO_ROOT/scripts/rootfs/build_rootfs.sh" --dest "$ROOTFS"
  [[ -x "$ROOTFS/bin/bash" ]] || die "rootfs build did not produce a usable rootfs at $ROOTFS"
fi

REPO_MNT=/workspace/torchtitan
HOST_LIBDIR=/usr/lib/x86_64-linux-gnu
rootfs_plan_devices=()
rootfs_plan_driver_libraries=()
rootfs_plan_nvidia_smi=""

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
  rootfs_plan_devices+=("$dev")
done
for lib in "$HOST_LIBDIR"/libcuda.so* "$HOST_LIBDIR"/libnvidia-*.so*; do
  bwrap_args+=(--ro-bind "$lib" "$lib")
  rootfs_plan_driver_libraries+=("$lib")
done
if [[ -x /usr/bin/nvidia-smi ]]; then
  bwrap_args+=(--ro-bind /usr/bin/nvidia-smi /usr/bin/nvidia-smi)
  rootfs_plan_nvidia_smi="/usr/bin/nvidia-smi"
fi
shopt -u nullglob

if [[ "${TORCHTITAN_ROOTFS_BIND_DOCKER:-0}" == "1" ]]; then
  [[ -x /usr/bin/docker ]] || die "TORCHTITAN_ROOTFS_BIND_DOCKER=1 but /usr/bin/docker is missing"
  [[ -S /var/run/docker.sock ]] || die "TORCHTITAN_ROOTFS_BIND_DOCKER=1 but /var/run/docker.sock is missing"
  bwrap_args+=(
    --ro-bind /usr/bin/docker /usr/bin/docker
    --dir /run
    --bind /var/run/docker.sock /run/docker.sock
    --bind "$REPO_ROOT" "$REPO_ROOT"
  )
  if [[ -d /usr/libexec/docker/cli-plugins ]]; then
    bwrap_args+=(--ro-bind /usr/libexec/docker/cli-plugins /usr/libexec/docker/cli-plugins)
  fi
fi

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
if [[ "${TORCHTITAN_ROOTFS_BIND_DOCKER:-0}" == "1" ]]; then
  bwrap_args+=(--setenv TORCHTITAN_ROOTFS_BIND_DOCKER 1)
  bwrap_args+=(--setenv TORCHTITAN_ROOTFS_HOST_REPO_ROOT "$REPO_ROOT")
fi
bwrap_args+=(--setenv NVIDIA_VISIBLE_DEVICES "${NVIDIA_VISIBLE_DEVICES:-all}")

if [[ "${TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY:-0}" == "1" ]]; then
  if [[ $# -eq 0 ]]; then
    emit_bwrap_plan /bin/bash -l
  else
    emit_bwrap_plan "$@"
  fi
  exit 0
fi

command -v bwrap >/dev/null || die "bwrap not found on host"

if [[ $# -eq 0 ]]; then
  exec bwrap "${bwrap_args[@]}" /bin/bash -l
else
  exec bwrap "${bwrap_args[@]}" "$@"
fi
