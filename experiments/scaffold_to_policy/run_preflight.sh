#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# run_preflight.sh is a QUERY over composable execution profiles (roadmap
# Section 9-11), not a manifest initializer. It reports readiness and stable
# blocker codes for the profiles a run's stages need and never writes the
# attempt journal. A caller uses its exit code to decide whether to proceed.
#
# Usage:
#   PROFILES=rootfs_cpu,vllm_1gpu experiments/scaffold_to_policy/run_preflight.sh
#   PROFILES=host_static REQUIRE_READY=0 experiments/scaffold_to_policy/run_preflight.sh
#
# host_static needs no rootfs and no GPU, so a host readiness query does not
# re-enter the rootfs. Any other profile requires the rootfs, so the script
# re-enters bwrap before probing.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"

PROFILES="${PROFILES:-host_static}"
REQUIRE_READY="${REQUIRE_READY:-1}"
OUTPUT="${OUTPUT:-}"

# Only host_static is safe to query without the rootfs. If any other profile is
# requested, re-enter the rootfs so its clauses probe the real environment.
needs_rootfs=0
IFS=',' read -r -a requested_profiles <<< "${PROFILES}"
for profile in "${requested_profiles[@]}"; do
  if [[ "${profile}" != "host_static" ]]; then
    needs_rootfs=1
  fi
done

if [[ "${needs_rootfs}" == "1" ]]; then
  scaffold_enter_rootfs_if_needed "run_preflight.sh" "$@"
fi
scaffold_setup_env

profile_args=()
for profile in "${requested_profiles[@]}"; do
  profile_args+=(--profile "${profile}")
done

require_ready_arg=()
if [[ "${REQUIRE_READY}" == "1" ]]; then
  require_ready_arg+=(--require-ready)
fi

output_arg=()
if [[ -n "${OUTPUT}" ]]; then
  output_arg+=(--output "${OUTPUT}")
fi

python -m torchtitan.experiments.execution preflight \
  "${profile_args[@]}" \
  "${output_arg[@]}" \
  "${require_ready_arg[@]}"
