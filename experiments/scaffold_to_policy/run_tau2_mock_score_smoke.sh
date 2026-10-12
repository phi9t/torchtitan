#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "experiments/scaffold_to_policy/run_tau2_mock_score_smoke.sh" "$@"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-tau2-mock-score-smoke}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/tau2_mock_score_smoke}"
VENV_DIR="${VENV_DIR:-${RESULTS_ROOT}/.venv-tau2-score-smoke}"
TAU2_REVISION="${TAU2_REVISION:-668d3bcd135c02aa3438f987ef45735b7c163ee3}"
TAU2_REPO_DIR="${TAU2_REPO_DIR:-${RESULTS_ROOT}/src/tau2-bench}"

mkdir -p "${RESULTS_ROOT}/raw" "${RESULTS_ROOT}/ingested" "${RESULTS_ROOT}/manifests" "${RESULTS_ROOT}/src"

if [[ "${RECREATE_VENV:-1}" == "1" ]]; then
  rm -rf "${VENV_DIR}"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python -m virtualenv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install -q --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -q \
  "git+https://github.com/sierra-research/tau2-bench.git@${TAU2_REVISION}"

if [[ ! -d "${TAU2_REPO_DIR}/.git" ]]; then
  rm -rf "${TAU2_REPO_DIR}"
  git clone --filter=blob:none https://github.com/sierra-research/tau2-bench.git "${TAU2_REPO_DIR}"
fi
git -C "${TAU2_REPO_DIR}" fetch --quiet origin "${TAU2_REVISION}"
git -C "${TAU2_REPO_DIR}" checkout --quiet "${TAU2_REVISION}"

export TAU2_DATA_DIR="${TAU2_REPO_DIR}/data"

"${VENV_DIR}/bin/python" -m torchtitan.experiments.scaffold_to_policy.cli write-tau2-mock-score-smoke \
  --run-id "${RUN_ID}" \
  --task-id "create_task_1" \
  --evaluation-type "all_ignore_basis" \
  --output "${RESULTS_ROOT}/raw/tau2_mock_score.json"

"${VENV_DIR}/bin/python" -m torchtitan.experiments.scaffold_to_policy.cli ingest-external-harness-smoke \
  --raw-result "${RESULTS_ROOT}/raw/tau2_mock_score.json" \
  --results-root "${RESULTS_ROOT}" \
  --output "${RESULTS_ROOT}/ingested/tau2_mock_score.json"

"${VENV_DIR}/bin/python" -m torchtitan.experiments.scaffold_to_policy.cli build-external-harness-report-input \
  --results-root "${RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --ingested "tau2=${RESULTS_ROOT}/ingested/tau2_mock_score.json" \
  --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"

echo "wrote ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
