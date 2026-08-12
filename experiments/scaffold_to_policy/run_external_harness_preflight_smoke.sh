#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "experiments/scaffold_to_policy/run_external_harness_preflight_smoke.sh" "$@"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-external-harness-preflight-smoke}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/external_harness_preflight_smoke}"
HARBOR_TERMINAL_VENV_DIR="${HARBOR_TERMINAL_VENV_DIR:-${RESULTS_ROOT}/.venv-harbor-terminal-preflight}"
TAU2_VENV_DIR="${TAU2_VENV_DIR:-${RESULTS_ROOT}/.venv-tau2-preflight}"
HARBOR_VERSION="${HARBOR_VERSION:-0.21.0}"
TERMINAL_BENCH_VERSION="${TERMINAL_BENCH_VERSION:-0.2.18}"
TAU2_REVISION="${TAU2_REVISION:-668d3bcd135c02aa3438f987ef45735b7c163ee3}"

mkdir -p "${RESULTS_ROOT}/raw" "${RESULTS_ROOT}/ingested" "${RESULTS_ROOT}/manifests"

if [[ "${RECREATE_VENV:-1}" == "1" ]]; then
  rm -rf "${HARBOR_TERMINAL_VENV_DIR}" "${TAU2_VENV_DIR}"
fi

if [[ ! -x "${HARBOR_TERMINAL_VENV_DIR}/bin/python" ]]; then
  python -m virtualenv "${HARBOR_TERMINAL_VENV_DIR}"
fi

"${HARBOR_TERMINAL_VENV_DIR}/bin/python" -m pip install -q --upgrade pip
"${HARBOR_TERMINAL_VENV_DIR}/bin/python" -m pip install -q \
  "harbor==${HARBOR_VERSION}" \
  "terminal-bench==${TERMINAL_BENCH_VERSION}"

if [[ ! -x "${TAU2_VENV_DIR}/bin/python" ]]; then
  python -m virtualenv "${TAU2_VENV_DIR}"
fi

"${TAU2_VENV_DIR}/bin/python" -m pip install -q --upgrade pip
"${TAU2_VENV_DIR}/bin/python" -m pip install -q \
  "git+https://github.com/sierra-research/tau2-bench.git@${TAU2_REVISION}"

"${HARBOR_TERMINAL_VENV_DIR}/bin/python" -m torchtitan.experiments.scaffold_to_policy.cli write-external-harness-preflight \
  --harness-family harbor_terminal \
  --run-id "${RUN_ID}" \
  --task-subset "terminal-bench-preflight" \
  --cli-name harbor \
  --cli-name terminal-bench \
  --cli-name tb \
  --output "${RESULTS_ROOT}/raw/harbor_terminal_preflight.json"

"${HARBOR_TERMINAL_VENV_DIR}/bin/python" -m torchtitan.experiments.scaffold_to_policy.cli ingest-external-harness-smoke \
  --raw-result "${RESULTS_ROOT}/raw/harbor_terminal_preflight.json" \
  --results-root "${RESULTS_ROOT}" \
  --output "${RESULTS_ROOT}/ingested/harbor_terminal_preflight.json"

"${TAU2_VENV_DIR}/bin/python" -m torchtitan.experiments.scaffold_to_policy.cli write-external-harness-preflight \
  --harness-family tau2 \
  --run-id "${RUN_ID}" \
  --task-subset "tau2-preflight" \
  --cli-name tau2 \
  --output "${RESULTS_ROOT}/raw/tau2_preflight.json"

"${TAU2_VENV_DIR}/bin/python" -m torchtitan.experiments.scaffold_to_policy.cli ingest-external-harness-smoke \
  --raw-result "${RESULTS_ROOT}/raw/tau2_preflight.json" \
  --results-root "${RESULTS_ROOT}" \
  --output "${RESULTS_ROOT}/ingested/tau2_preflight.json"

"${HARBOR_TERMINAL_VENV_DIR}/bin/python" -m torchtitan.experiments.scaffold_to_policy.cli build-external-harness-report-input \
  --results-root "${RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --ingested \
    "harbor_terminal=${RESULTS_ROOT}/ingested/harbor_terminal_preflight.json" \
    "tau2=${RESULTS_ROOT}/ingested/tau2_preflight.json" \
  --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"

echo "wrote ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
