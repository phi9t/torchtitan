#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "experiments/scaffold_to_policy/run_tau2_execution_probe.sh" "$@"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-tau2-execution-probe}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/tau2_execution_probe}"
VENV_DIR="${VENV_DIR:-${RESULTS_ROOT}/.venv-tau2-execution-probe}"
TAU2_REVISION="${TAU2_REVISION:-668d3bcd135c02aa3438f987ef45735b7c163ee3}"
TAU2_REPO_DIR="${TAU2_REPO_DIR:-${RESULTS_ROOT}/src/tau2-bench}"
TAU2_SAVE_NAME="${TAU2_SAVE_NAME:-${RUN_ID}}"
TAU2_TASK_ID="${TAU2_TASK_ID:-create_task_1}"
TAU2_REGISTER_TORCHTITAN_AGENT="${TAU2_REGISTER_TORCHTITAN_AGENT:-1}"
TAU2_AGENT="${TAU2_AGENT:-torchtitan_mock_oracle_agent}"
TAU2_AGENT_LLM="${TAU2_AGENT_LLM:-fake}"
TAU2_USER="${TAU2_USER:-torchtitan_static_user}"
TAU2_USER_LLM="${TAU2_USER_LLM:-fake}"
TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-4}"
TAU2_MAX_ERRORS="${TAU2_MAX_ERRORS:-1}"
TAU2_TIMEOUT="${TAU2_TIMEOUT:-20}"
PROBE_TIMEOUT_SECONDS="${PROBE_TIMEOUT_SECONDS:-120}"

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

TAU2_WRAPPER="${RESULTS_ROOT}/tau2_registered_cli.py"
cat > "${TAU2_WRAPPER}" <<'PY'
import sys

if __name__ == "__main__":
    import torchtitan.experiments.scaffold_to_policy.tau2_probe_agent
    from tau2.cli import main

    sys.exit(main())
PY

if [[ "${TAU2_REGISTER_TORCHTITAN_AGENT}" == "1" ]]; then
  TAU2_CLI_COMMAND=("${VENV_DIR}/bin/python" "${TAU2_WRAPPER}")
else
  TAU2_CLI_COMMAND=("${VENV_DIR}/bin/tau2")
fi

if [[ ! -d "${TAU2_REPO_DIR}/.git" ]]; then
  rm -rf "${TAU2_REPO_DIR}"
  git clone --filter=blob:none https://github.com/sierra-research/tau2-bench.git "${TAU2_REPO_DIR}"
fi
git -C "${TAU2_REPO_DIR}" fetch --quiet origin "${TAU2_REVISION}"
git -C "${TAU2_REPO_DIR}" checkout --quiet "${TAU2_REVISION}"

export TAU2_DATA_DIR="${TAU2_REPO_DIR}/data"
"${TAU2_CLI_COMMAND[@]}" check-data

TAU2_RESULTS_JSON="${TAU2_DATA_DIR}/simulations/${TAU2_SAVE_NAME}/results.json"
rm -rf "${TAU2_DATA_DIR}/simulations/${TAU2_SAVE_NAME}"

"${VENV_DIR}/bin/python" -m torchtitan.experiments.scaffold_to_policy.cli write-tau2-execution-probe \
  --run-id "${RUN_ID}" \
  --task-id "${TAU2_TASK_ID}" \
  --cwd "${REPO_ROOT}" \
  --results-json "${TAU2_RESULTS_JSON}" \
  --timeout-seconds "${PROBE_TIMEOUT_SECONDS}" \
  --output "${RESULTS_ROOT}/raw/tau2_execution_probe.json" \
  "${TAU2_CLI_COMMAND[@]}" run \
    --domain mock \
    --task-set-name mock \
    --task-ids "${TAU2_TASK_ID}" \
    --num-trials 1 \
    --agent "${TAU2_AGENT}" \
    --agent-llm "${TAU2_AGENT_LLM}" \
    --user "${TAU2_USER}" \
    --user-llm "${TAU2_USER_LLM}" \
    --max-steps "${TAU2_MAX_STEPS}" \
    --max-errors "${TAU2_MAX_ERRORS}" \
    --timeout "${TAU2_TIMEOUT}" \
    --max-concurrency 1 \
    --max-retries 0 \
    --log-level DEBUG \
    --save-to "${TAU2_SAVE_NAME}"

"${VENV_DIR}/bin/python" -m torchtitan.experiments.scaffold_to_policy.cli ingest-external-harness-smoke \
  --raw-result "${RESULTS_ROOT}/raw/tau2_execution_probe.json" \
  --results-root "${RESULTS_ROOT}" \
  --output "${RESULTS_ROOT}/ingested/tau2_execution_probe.json"

"${VENV_DIR}/bin/python" -m torchtitan.experiments.scaffold_to_policy.cli build-external-harness-report-input \
  --results-root "${RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --ingested "tau2=${RESULTS_ROOT}/ingested/tau2_execution_probe.json" \
  --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json" \
  --no-require-selected

echo "wrote ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
