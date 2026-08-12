#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_common.sh"
countdown_enter_rootfs_if_needed "run_replication_sweep.sh" "$@"
countdown_setup_env

MODE="${MODE:-full}"
ARMS="${ARMS:-clean}"
SEEDS="${SEEDS:-43}"
TRAIN_SIZES="${TRAIN_SIZES:-2000}"
LORA_RANKS="${LORA_RANKS:-32}"
NGPU="${NGPU:-8}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-step-94}"
FORCE="${FORCE:-0}"

case "${MODE}" in
  full|reduced) ;;
  *) echo "replication sweep supports MODE=full or MODE=reduced, got ${MODE}" >&2; exit 2 ;;
esac

read -r -a ARM_LIST <<< "${ARMS}"
read -r -a SEED_LIST <<< "${SEEDS}"
read -r -a TRAIN_SIZE_LIST <<< "${TRAIN_SIZES}"
read -r -a LORA_RANK_LIST <<< "${LORA_RANKS}"

SWEEP_ROOT="${TORCHTITAN_COUNTDOWN_ROOT}/sweeps/replication"
mkdir -p "${SWEEP_ROOT}"
SUMMARY="${SWEEP_ROOT}/replication_sweep_${MODE}.jsonl"
: > "${SUMMARY}"

for seed in "${SEED_LIST[@]}"; do
  for train_size in "${TRAIN_SIZE_LIST[@]}"; do
    for lora_rank in "${LORA_RANK_LIST[@]}"; do
      label="seed${seed}_train${train_size}_rank${lora_rank}"
      run_id="${RUN_ID_PREFIX:-$(date -u +%Y%m%dT%H%M%SZ)-replication}-${label}"
      run_root="${TORCHTITAN_COUNTDOWN_ROOT}/sweeps/replication/${label}"
      data_root="${run_root}/data"
      results_root="${run_root}/results"
      mkdir -p "${data_root}" "${results_root}"

      echo "running replication label=${label} mode=${MODE} arms=${ARM_LIST[*]}"
      TORCHTITAN_COUNTDOWN_DATA_ROOT="${data_root}" \
      TORCHTITAN_COUNTDOWN_RESULTS_ROOT="${results_root}" \
      TORCHTITAN_COUNTDOWN_LORA_RANK="${lora_rank}" \
      SEED="${seed}" \
      TRAIN_SIZE="${train_size}" \
      MODE="${MODE}" \
      ARMS="${ARM_LIST[*]}" \
      NGPU="${NGPU}" \
      CHECKPOINT_STEP="${CHECKPOINT_STEP}" \
      FORCE="${FORCE}" \
      RUN_ID="${run_id}" \
        "${SCRIPT_DIR}/run_full_pilot.sh" "$@"

      report_input="${results_root}/manifests/report_input_${run_id}.json"
      python - <<PY
import json
from pathlib import Path

row = {
    "label": "${label}",
    "run_id": "${run_id}",
    "mode": "${MODE}",
    "seed": int("${seed}"),
    "train_size": int("${train_size}"),
    "lora_rank": int("${lora_rank}"),
    "data_root": "${data_root}",
    "results_root": "${results_root}",
    "report_input": "${report_input}",
}
row["arms"] = "${ARM_LIST[*]}".split()
with Path("${SUMMARY}").open("a") as f:
    f.write(json.dumps(row, sort_keys=True) + "\\n")
PY
    done
  done
done

echo "wrote ${SUMMARY}"
