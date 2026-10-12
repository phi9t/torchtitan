#!/usr/bin/env bash
# Install bpftrace + BCC into the writable rootfs tree (survives bwrap,
# wiped by a rootfs rebuild). Uses host sudo chroot + apt.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
ROOTFS="${1:-$REPO_ROOT/scripts/rootfs/rootfs}"

if [[ ! -x "$ROOTFS/usr/bin/apt-get" ]]; then
  echo "rootfs apt-get missing at $ROOTFS" >&2
  exit 2
fi

sudo -n cp /etc/resolv.conf "$ROOTFS/etc/resolv.conf"
sudo -n chroot "$ROOTFS" /bin/bash -lc '
  set -euo pipefail
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y --no-install-recommends \
    bpfcc-tools bpftrace python3-bpfcc pciutils linux-tools-common gdb
  command -v bpftrace
  bpftrace --version
'
