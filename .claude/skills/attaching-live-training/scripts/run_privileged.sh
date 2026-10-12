#!/usr/bin/env bash
# Run a command in the rootfs filesystem with host kernel privileges.
# Used for bpftrace / BCC / perf tracepoints, which cannot load BPF
# in an unprivileged bwrap user namespace.
#
# Must be invoked from inside enter_rootfs.sh --profile (docker socket
# bound, TORCHTITAN_ROOTFS_HOST_ROOTFS set).
set -euo pipefail

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  echo "run_privileged.sh must run inside enter_rootfs.sh --profile" >&2
  exit 2
fi

# Under enter_rootfs.sh --privileged the sandbox already holds CAP_BPF in the
# initial user namespace, so the tool runs directly and no escalation hop is
# needed. Callers can pass the same command line in either mode.
if [[ "${TORCHTITAN_ROOTFS_PRIVILEGED:-0}" == "1" ]]; then
  if [[ "${1:-}" == "--host" ]]; then
    shift
  fi
  exec "$@"
fi

if [[ -z "${TORCHTITAN_ROOTFS_HOST_ROOTFS:-}" ]]; then
  echo "TORCHTITAN_ROOTFS_HOST_ROOTFS is unset; enter with --profile" >&2
  exit 2
fi
if ! command -v docker >/dev/null; then
  echo "docker is not bound into the rootfs; enter with --profile" >&2
  exit 2
fi

# --host runs the host binary via chroot /host (needed for the
# distro perf/bpftool that link host libunwind/libbfd).
if [[ "${1:-}" == "--host" ]]; then
  shift
  exec docker run --rm --privileged --pid=host --net=host \
    -v /:/host \
    alpine:3.20 \
    chroot /host "$@"
fi

host_rootfs="${TORCHTITAN_ROOTFS_HOST_ROOTFS}"
repo_root="${TORCHTITAN_ROOTFS_HOST_REPO_ROOT:?TORCHTITAN_ROOTFS_HOST_REPO_ROOT unset}"

# Overlay host kernel interfaces and the host perf/bpftool binaries
# (the image copies are Ubuntu wrappers that look for the wrong kernel).
exec docker run --rm --privileged --pid=host --net=host \
  -v /:/host \
  -v /sys:"/host${host_rootfs}/sys" \
  -v /proc:"/host${host_rootfs}/proc" \
  -v /dev:"/host${host_rootfs}/dev" \
  -v /lib/modules:"/host${host_rootfs}/lib/modules" \
  -v /usr/src:"/host${host_rootfs}/usr/src" \
  -v /usr/bin/perf:"/host${host_rootfs}/usr/bin/perf" \
  -v /usr/sbin/bpftool:"/host${host_rootfs}/usr/sbin/bpftool" \
  -v "${repo_root}:/host${host_rootfs}/workspace/torchtitan" \
  alpine:3.20 \
  chroot "/host${host_rootfs}" "$@"
