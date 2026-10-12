#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Mesh-sweep + ablation harness for the Qwen3 FineWeb HSDP+TP experiment.
#
# Runs several 8-GPU parallelism configurations (pure FSDP, HSDP, HSDP+TP,
# TP-heavy) plus reshard / async-TP / sequence-parallel ablations, all against
# CONFIG=qwen3_debugmodel_fineweb with deterministic debug options, tees each
# run to results/<name>.log, then extracts a comparison table to
# results/summary.md.
#
# Prerequisite (one-time, online):
#   python experiments/qwen3_fineweb_hsdp_tp/prefetch_fineweb.py
#
# Usage (offline, 8x GPU node):
#   experiments/qwen3_fineweb_hsdp_tp/sweep.sh
#
# The env exports below mirror run.sh so the sweep is self-contained offline.

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_FILE="experiments/qwen3_fineweb_hsdp_tp/data/fineweb_test/data.json"
if [ ! -f "${DATA_FILE}" ]; then
    echo "ERROR: ${DATA_FILE} not found." >&2
    echo "Run the one-time online prefetch first:" >&2
    echo "  python experiments/qwen3_fineweb_hsdp_tp/prefetch_fineweb.py" >&2
    exit 1
fi

RESULTS_DIR="experiments/qwen3_fineweb_hsdp_tp/results"
mkdir -p "${RESULTS_DIR}"

# --- Offline + toolchain env, identical to run.sh -------------------------
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

# Activate the project virtualenv if present and not already active: torch and
# torchrun live there, not in the system Python.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# Deterministic debug options shared by every config (AGENTS.md: never use
# --debug.deterministic_warn_only). 10 steps so warmup does not dominate.
# Checkpointing is disabled: the sweep compares mesh shapes and each shape has
# a different data-parallel layout, so loading/saving a checkpoint written
# under one mesh into another fails on mismatched dataloader dp_rank keys.
# The sweep is about parallelism metrics, not checkpoint I/O.
COMMON_OPTS=(
    --debug.seed=42
    --debug.deterministic
    --training.steps=10
    --checkpoint.no-enable
)

# Each entry: "name|<extra run_train.sh args...>". The four mesh shapes all
# multiply to 8 = NGPU; the ablations reuse the landed HSDP+TP shape.
CONFIGS=(
    "fsdp8|--parallelism.data_parallel_shard_degree=8"
    "hsdp_2x4|--parallelism.data_parallel_replicate_degree=2 --parallelism.data_parallel_shard_degree=4"
    "hsdp_tp_2x2x2|--parallelism.data_parallel_replicate_degree=2 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2"
    "tp_heavy_1x2x4|--parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=4"
    "hsdp_tp_reshard_always|--parallelism.data_parallel_replicate_degree=2 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --parallelism.fsdp_reshard_after_forward=always"
    "hsdp_tp_reshard_never|--parallelism.data_parallel_replicate_degree=2 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --parallelism.fsdp_reshard_after_forward=never"
    "hsdp_tp_async_tp|--parallelism.data_parallel_replicate_degree=2 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --parallelism.enable_async_tensor_parallel --compile.enable"
    "hsdp_tp_no_seq_parallel|--parallelism.data_parallel_replicate_degree=2 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --parallelism.no-enable-sequence-parallel"
)

run_one() {
    local name="$1"
    shift
    local log="${RESULTS_DIR}/${name}.log"
    echo ">>> [${name}] launching: $*"
    # Use PIPESTATUS so a training failure is recorded even though tee succeeds.
    # Do not abort the whole sweep on a single config failure; note it and move on.
    set +e
    # shellcheck disable=SC2086
    NGPU=8 MODULE=qwen3 CONFIG=qwen3_debugmodel_fineweb ./run_train.sh \
        "${COMMON_OPTS[@]}" $* 2>&1 | tee "${log}"
    local status=${PIPESTATUS[0]}
    set -e
    if [ "${status}" -ne 0 ]; then
        echo ">>> [${name}] FAILED with exit ${status} (see ${log})"
    else
        echo ">>> [${name}] done -> ${log}"
    fi
}

for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"
    args="${entry#*|}"
    run_one "${name}" ${args}
done

# --- Build results/summary.md --------------------------------------------
python experiments/qwen3_fineweb_hsdp_tp/summarize_sweep.py \
    --results-dir "${RESULTS_DIR}" \
    --names "$(printf '%s ' "${CONFIGS[@]%%|*}")"

echo "Sweep complete. Summary at ${RESULTS_DIR}/summary.md"
