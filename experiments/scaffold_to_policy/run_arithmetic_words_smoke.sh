#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Reference runner for wave F4: arithmetic-words smoke driven through the typed
# begin/stage/finish lifecycle and the host_static profile doctor. This is the
# first runner migrated off the prototype manifest; it stays a thin
# compatibility entrypoint with unchanged scientific conditions and defaults.
# See experiments/scaffold_to_policy/f4_reference_runner_migration_spec.md.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "experiments/scaffold_to_policy/run_arithmetic_words_smoke.sh" "$@"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-arithmetic-words-smoke}"
ATTEMPT_ID="${ATTEMPT_ID:-attempt-01}"
DATA_ROOT="${DATA_ROOT:-experiments/scaffold_to_policy/data/arithmetic_words_smoke}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/arithmetic_words_smoke}"

CLI="python -m torchtitan.experiments.scaffold_to_policy.cli"
LIFECYCLE="python -m torchtitan.experiments.execution"
LOCATOR=(--results-root "${RESULTS_ROOT}" --run-id "${RUN_ID}" --attempt-id "${ATTEMPT_ID}")

mkdir -p "${DATA_ROOT}" "${RESULTS_ROOT}/eval" "${RESULTS_ROOT}/manifests"

# host_static is ungated (no rootfs/package/GPU clause), so it is ready on the
# host under the rootfs re-exec guard. --require-ready aborts before any work if
# the environment is unexpectedly blocked; it never fabricates an attempt.
${LIFECYCLE} preflight \
  --profile host_static \
  --require-ready \
  --output "${RESULTS_ROOT}/manifests/preflight_${RUN_ID}.json"

# Freeze the declaration. --fields records the scientific knobs so the
# declaration digest changes if any seed, split size, or rollout budget changes.
FIELDS_FILE="${RESULTS_ROOT}/manifests/fields_${RUN_ID}.json"
cat >"${FIELDS_FILE}" <<JSON
{
  "splits": {
    "train": {"seed": 100, "num_problems": 12},
    "dev": {"seed": 200, "num_problems": 8},
    "ood_test": {"seed": 300, "num_problems": 8}
  },
  "eval_splits": ["dev", "ood_test"],
  "max_rollouts": 3
}
JSON

${LIFECYCLE} begin "${LOCATOR[@]}" \
  --family reasoning \
  --task arithmetic_words \
  --lane host_smoke \
  --fields "${FIELDS_FILE}"

${LIFECYCLE} stage "${LOCATOR[@]}" \
  --stage-id gen-train --name generate-train --kind generate --adapter host_test \
  -- ${CLI} generate-arithmetic-words \
  --seed 100 --num-problems 12 --output "${DATA_ROOT}/train.jsonl"

${LIFECYCLE} stage "${LOCATOR[@]}" \
  --stage-id gen-dev --name generate-dev --kind generate --adapter host_test \
  -- ${CLI} generate-arithmetic-words \
  --seed 200 --num-problems 8 --output "${DATA_ROOT}/dev.jsonl"

${LIFECYCLE} stage "${LOCATOR[@]}" \
  --stage-id gen-ood_test --name generate-ood_test --kind generate --adapter host_test \
  -- ${CLI} generate-arithmetic-words \
  --seed 300 --num-problems 8 --output "${DATA_ROOT}/ood_test.jsonl"

${LIFECYCLE} stage "${LOCATOR[@]}" \
  --stage-id validate-splits --name validate-splits --kind verify --adapter host_test \
  -- ${CLI} validate-arithmetic-splits \
  --split \
    "train=${DATA_ROOT}/train.jsonl" \
    "dev=${DATA_ROOT}/dev.jsonl" \
    "ood_test=${DATA_ROOT}/ood_test.jsonl" \
  --output "${DATA_ROOT}/split_registry.json"

for split in dev ood_test; do
  ${LIFECYCLE} stage "${LOCATOR[@]}" \
    --stage-id "write-fixture-${split}" --name "write-fixture-${split}" \
    --kind generate --adapter host_test \
    -- ${CLI} write-arithmetic-fixture \
    --problems "${DATA_ROOT}/${split}.jsonl" \
    --output "${RESULTS_ROOT}/eval/${split}_fixture_rollouts.jsonl"

  ${LIFECYCLE} stage "${LOCATOR[@]}" \
    --stage-id "evaluate-${split}" --name "evaluate-${split}" \
    --kind evaluate --adapter host_test \
    -- ${CLI} evaluate-arithmetic-fixture \
    --problems "${DATA_ROOT}/${split}.jsonl" \
    --rollouts "${RESULTS_ROOT}/eval/${split}_fixture_rollouts.jsonl" \
    --output "${RESULTS_ROOT}/eval/${split}_evaluations.jsonl" \
    --summary "${RESULTS_ROOT}/eval/${split}_summary.json" \
    --max-rollouts 3
done

REPORT_INPUT="${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
${LIFECYCLE} stage "${LOCATOR[@]}" \
  --stage-id build-report-input --name build-report-input --kind report --adapter host_test \
  -- ${CLI} build-arithmetic-report-input \
  --data-root "${DATA_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --split-registry "${DATA_ROOT}/split_registry.json" \
  --summary \
    "dev=${RESULTS_ROOT}/eval/dev_summary.json" \
    "ood_test=${RESULTS_ROOT}/eval/ood_test_summary.json" \
  --output "${REPORT_INPUT}"

# Fixture rollouts are never a real measurement: each eval split is a fixture
# condition so the derived run gate reports has_real_measurement=false.
EVALUATIONS_FILE="${RESULTS_ROOT}/manifests/evaluations_${RUN_ID}.json"
cat >"${EVALUATIONS_FILE}" <<JSON
{
  "dev": {"execution_outcome": "completed", "measurement": "fixture", "promotion": "not_evaluated"},
  "ood_test": {"execution_outcome": "completed", "measurement": "fixture", "promotion": "not_evaluated"}
}
JSON

${LIFECYCLE} finish "${LOCATOR[@]}" \
  --attempt-outcome completed \
  --report-input "${REPORT_INPUT}" \
  --evaluations "${EVALUATIONS_FILE}"

echo "wrote attempt bundle ${RESULTS_ROOT}/runs/${RUN_ID}/${ATTEMPT_ID}"
