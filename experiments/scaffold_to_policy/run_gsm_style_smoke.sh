#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "experiments/scaffold_to_policy/run_gsm_style_smoke.sh" "$@"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-gsm-style-smoke}"
DATA_ROOT="${DATA_ROOT:-experiments/scaffold_to_policy/data/gsm_style_smoke}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/scaffold_to_policy/results/gsm_style_smoke}"
FIXTURE_ROOT="${FIXTURE_ROOT:-experiments/scaffold_to_policy/fixtures}"

mkdir -p "${DATA_ROOT}" "${RESULTS_ROOT}/eval" "${RESULTS_ROOT}/manifests"

python -m torchtitan.experiments.scaffold_to_policy.cli prepare-gsm-style-split \
  --input "${FIXTURE_ROOT}/gsm_style_train.jsonl" \
  --output "${DATA_ROOT}/train.jsonl"

python -m torchtitan.experiments.scaffold_to_policy.cli prepare-gsm-style-split \
  --input "${FIXTURE_ROOT}/gsm_style_dev.jsonl" \
  --output "${DATA_ROOT}/dev.jsonl"

python -m torchtitan.experiments.scaffold_to_policy.cli prepare-gsm-style-split \
  --input "${FIXTURE_ROOT}/gsm_style_ood_test.jsonl" \
  --output "${DATA_ROOT}/ood_test.jsonl"

python -m torchtitan.experiments.scaffold_to_policy.cli validate-gsm-style-splits \
  --split \
    "train=${DATA_ROOT}/train.jsonl" \
    "dev=${DATA_ROOT}/dev.jsonl" \
    "ood_test=${DATA_ROOT}/ood_test.jsonl" \
  --output "${DATA_ROOT}/split_registry.json"

for split in dev ood_test; do
  python -m torchtitan.experiments.scaffold_to_policy.cli write-gsm-style-fixture \
    --problems "${DATA_ROOT}/${split}.jsonl" \
    --output "${RESULTS_ROOT}/eval/${split}_fixture_rollouts.jsonl"

  python -m torchtitan.experiments.scaffold_to_policy.cli evaluate-gsm-style-fixture \
    --problems "${DATA_ROOT}/${split}.jsonl" \
    --rollouts "${RESULTS_ROOT}/eval/${split}_fixture_rollouts.jsonl" \
    --output "${RESULTS_ROOT}/eval/${split}_evaluations.jsonl" \
    --summary "${RESULTS_ROOT}/eval/${split}_summary.json" \
    --max-rollouts 4
done

python -m torchtitan.experiments.scaffold_to_policy.cli build-gsm-style-report-input \
  --data-root "${DATA_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --split-registry "${DATA_ROOT}/split_registry.json" \
  --summary \
    "dev=${RESULTS_ROOT}/eval/dev_summary.json" \
    "ood_test=${RESULTS_ROOT}/eval/ood_test_summary.json" \
  --scaffold-budget 4 \
  --output "${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"

echo "wrote ${RESULTS_ROOT}/manifests/report_input_${RUN_ID}.json"
