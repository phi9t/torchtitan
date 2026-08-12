#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  TORCHTITAN_ROOTFS_BIND_DOCKER=1 exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "experiments/scaffold_to_policy/run_terminal_bench_oracle_probe.sh" "$@"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-terminal-bench-oracle-probe}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/terminal_bench_oracle_probe}"
VENV_DIR="${VENV_DIR:-${RESULTS_ROOT}/.venv-harbor-terminal-probe}"
TERMINAL_BENCH_REVISION="${TERMINAL_BENCH_REVISION:-7131e4375048a0e408a8fb404b5f499d726b695b}"
TERMINAL_BENCH_REPO_DIR="${TERMINAL_BENCH_REPO_DIR:-${RESULTS_ROOT}/src/terminal-bench-2-1}"
HARBOR_VERSION="${HARBOR_VERSION:-0.21.0}"
TERMINAL_BENCH_VERSION="${TERMINAL_BENCH_VERSION:-0.2.18}"
TASK_ID="${TASK_ID:-headless-terminal}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-180}"

mkdir -p "${RESULTS_ROOT}/raw" "${RESULTS_ROOT}/ingested" "${RESULTS_ROOT}/manifests" "${RESULTS_ROOT}/runs" "${RESULTS_ROOT}/src"

if [[ "${RECREATE_VENV:-1}" == "1" ]]; then
  rm -rf "${VENV_DIR}"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python -m virtualenv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install -q --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -q \
  "harbor==${HARBOR_VERSION}" \
  "terminal-bench==${TERMINAL_BENCH_VERSION}"

if [[ ! -d "${TERMINAL_BENCH_REPO_DIR}/.git" ]]; then
  rm -rf "${TERMINAL_BENCH_REPO_DIR}"
  git clone --filter=blob:none https://github.com/harbor-framework/terminal-bench-2-1.git "${TERMINAL_BENCH_REPO_DIR}"
fi
git -C "${TERMINAL_BENCH_REPO_DIR}" fetch --quiet origin "${TERMINAL_BENCH_REVISION}"
git -C "${TERMINAL_BENCH_REPO_DIR}" checkout --quiet "${TERMINAL_BENCH_REVISION}"

"${VENV_DIR}/bin/python" -m torchtitan.experiments.scaffold_to_policy.cli write-terminal-bench-result-smoke \
  --run-id "${RUN_ID}" \
  --task-id "${TASK_ID}" \
  --output "${RESULTS_ROOT}/raw/terminal_bench_result_smoke.json"

"${VENV_DIR}/bin/python" -m torchtitan.experiments.scaffold_to_policy.cli ingest-external-harness-smoke \
  --raw-result "${RESULTS_ROOT}/raw/terminal_bench_result_smoke.json" \
  --results-root "${RESULTS_ROOT}" \
  --output "${RESULTS_ROOT}/ingested/terminal_bench_result_smoke.json"

"${VENV_DIR}/bin/python" -m torchtitan.experiments.scaffold_to_policy.cli write-terminal-bench-execution-probe \
  --run-id "${RUN_ID}" \
  --task-id "${TASK_ID}" \
  --cwd "${REPO_ROOT}" \
  --harbor-result-json "${RESULTS_ROOT}/runs/${RUN_ID}/result.json" \
  --timeout-seconds "${TIMEOUT_SECONDS}" \
  --output "${RESULTS_ROOT}/raw/terminal_bench_execution_probe.json" \
  "${VENV_DIR}/bin/harbor" run \
    --path "${TERMINAL_BENCH_REPO_DIR}/tasks/${TASK_ID}" \
    --jobs-dir "${RESULTS_ROOT}/runs" \
    --job-name "${RUN_ID}" \
    --env docker \
    --agent oracle \
    --n-concurrent 1 \
    --n-attempts 1 \
    --max-retries 0 \
    --yes \
    --no-delete \
    --agent-timeout-multiplier 1 \
    --verifier-timeout-multiplier 1

"${VENV_DIR}/bin/python" -m torchtitan.experiments.scaffold_to_policy.cli ingest-external-harness-smoke \
  --raw-result "${RESULTS_ROOT}/raw/terminal_bench_execution_probe.json" \
  --results-root "${RESULTS_ROOT}" \
  --output "${RESULTS_ROOT}/ingested/terminal_bench_execution_probe.json"

"${VENV_DIR}/bin/python" -m torchtitan.experiments.scaffold_to_policy.cli build-external-harness-report-input \
  --results-root "${RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --ingested \
    "terminal_bench_result=${RESULTS_ROOT}/ingested/terminal_bench_result_smoke.json" \
    "terminal_bench_execution=${RESULTS_ROOT}/ingested/terminal_bench_execution_probe.json" \
  --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json" \
  --no-require-selected

echo "wrote ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
