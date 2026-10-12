#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "experiments/scaffold_to_policy/run_external_harness_dry_run_smoke.sh" "$@"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-external-harness-dry-run-smoke}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/external_harness_dry_run_smoke}"
HARBOR_TERMINAL_TASK_SUBSET="${HARBOR_TERMINAL_TASK_SUBSET:-terminal-bench-dry-run}"
TAU2_TASK_SUBSET="${TAU2_TASK_SUBSET:-tau2-dry-run}"

mkdir -p "${RESULTS_ROOT}/raw" "${RESULTS_ROOT}/ingested" "${RESULTS_ROOT}/manifests"

python -m torchtitan.experiments.scaffold_to_policy.cli write-external-harness-smoke \
  --harness-family harbor_terminal \
  --run-id "${RUN_ID}" \
  --task-subset "${HARBOR_TERMINAL_TASK_SUBSET}" \
  --output "${RESULTS_ROOT}/raw/harbor_terminal.json"

python -m torchtitan.experiments.scaffold_to_policy.cli ingest-external-harness-smoke \
  --raw-result "${RESULTS_ROOT}/raw/harbor_terminal.json" \
  --results-root "${RESULTS_ROOT}" \
  --output "${RESULTS_ROOT}/ingested/harbor_terminal.json"

python -m torchtitan.experiments.scaffold_to_policy.cli write-external-harness-smoke \
  --harness-family tau2 \
  --run-id "${RUN_ID}" \
  --task-subset "${TAU2_TASK_SUBSET}" \
  --output "${RESULTS_ROOT}/raw/tau2.json"

python -m torchtitan.experiments.scaffold_to_policy.cli ingest-external-harness-smoke \
  --raw-result "${RESULTS_ROOT}/raw/tau2.json" \
  --results-root "${RESULTS_ROOT}" \
  --output "${RESULTS_ROOT}/ingested/tau2.json"

python -m torchtitan.experiments.scaffold_to_policy.cli build-external-harness-report-input \
  --results-root "${RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --ingested \
    "harbor_terminal=${RESULTS_ROOT}/ingested/harbor_terminal.json" \
    "tau2=${RESULTS_ROOT}/ingested/tau2.json" \
  --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"

echo "wrote ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
