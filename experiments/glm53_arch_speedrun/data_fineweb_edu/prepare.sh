#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
script_rel="experiments/glm53_arch_speedrun/data_fineweb_edu/prepare.sh"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${repo_root}/scripts/rootfs/enter_rootfs.sh" -- "${script_rel}" "$@"
fi

cd "${repo_root}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
exec python -m torchtitan.experiments.glm53_arch_speedrun.data_prep.fineweb_edu "$@"

