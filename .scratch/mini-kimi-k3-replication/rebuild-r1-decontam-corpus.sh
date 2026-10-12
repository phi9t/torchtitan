#!/usr/bin/env bash
set -euo pipefail

cd /workspace/torchtitan

WORK=/tmp/mini-kimi-k3-r1-decontam-work-20260823
MATERIALIZE_REPORT=experiments/mini_kimi_k3/reports/stage1-local-materialize-r1-decontam.json
MANIFEST=experiments/mini_kimi_k3/data/manifest-r1-decontam.json
TOKENS_DIR=experiments/mini_kimi_k3/data/tokens-r1-decontam
REPORTS_DIR=experiments/mini_kimi_k3/reports/r1-local-provenance-decontam
TOKENIZER_SHA=e1de82950ca55dec3b6dc1c5d5e78d06ddadbd4682515e8571e24635f50348ad
DECONTAM_REPORT=experiments/mini_kimi_k3/reports/decontamination.json
DECONTAM_SHA=83821495d825b86a958464b515f40747271f93bc1dce9bcebde961d1bce267de
DECONTAM_INDEX=experiments/mini_kimi_k3/data/decontamination/index.npz

export HF_HOME="$WORK/hf-home"
export HF_HUB_CACHE="$WORK/hf-cache"
export TRANSFORMERS_CACHE="$WORK/transformers-cache"
export XDG_CACHE_HOME="$WORK/xdg-cache"

df -h /tmp /workspace/torchtitan
rm -rf "$WORK"
mkdir -p "$WORK" "$TOKENS_DIR" "$REPORTS_DIR"

experiments/mini_kimi_k3/run.sh materialize-stage1-local \
  --source-resolution-report experiments/mini_kimi_k3/reports/stage1-source-resolution.json \
  --corpus-plan-report experiments/mini_kimi_k3/reports/r1-corpus-plan.json \
  --output-root "$WORK" \
  --report "$MATERIALIZE_REPORT" \
  --allow-download

run_prepare() {
  local source="$1"
  local dataset_id="$2"
  local snapshot="$3"
  local license_label="$4"
  local max_tokens="$5"
  local manifest_path="$WORK/${source}-inputs.json"

  experiments/mini_kimi_k3/run.sh prepare-stage1-local \
    --manifest "$MANIFEST" \
    --tokens-dir "$TOKENS_DIR" \
    --reports-dir "$REPORTS_DIR" \
    --source "$source" \
    --dataset-id "$dataset_id" \
    --snapshot "$snapshot" \
    --license "$license_label" \
    --input-manifest "$manifest_path" \
    --input-format parquet-text \
    --tokenizer-sha256 "$TOKENIZER_SHA" \
    --tokenizer-asset-report experiments/mini_kimi_k3/assets/kimi-k3-tokenizer.json \
    --decontamination-report "$DECONTAM_REPORT" \
    --decontamination-sha256 "$DECONTAM_SHA" \
    --decontamination-index "$DECONTAM_INDEX" \
    --decontamination-min-matches 1 \
    --shard-tokens 100000000 \
    --max-tokens "$max_tokens" \
    --prefix "${source}-r1-decontam"

  rm -rf "$WORK/files/$source" "$manifest_path"
  df -h /tmp /workspace/torchtitan
}

run_prepare code-python local/code-python-r1-local Python-all local-source-metadata-pending 612500000
run_prepare cosmopedia local/cosmopedia-r1-local train local-source-metadata-pending 150000000
run_prepare finemath local/finemath-r1-local finemath-4plus local-source-metadata-pending 262500000
run_prepare fineweb-edu local/fineweb-edu-r1-local sample-100BT local-source-metadata-pending 2660000000
run_prepare open-web-math local/open-web-math-r1-local train local-source-metadata-pending 175000000
run_prepare web-diverse local/web-diverse-r1-local sample-100BT local-source-metadata-pending 1140000000

rm -rf "$WORK"

experiments/mini_kimi_k3/run.sh audit-r1-corpus-plan \
  --manifest "$MANIFEST" \
  --tokens-dir "$TOKENS_DIR" \
  --corpus-plan experiments/mini_kimi_k3/reports/r1-corpus-plan.json \
  --report experiments/mini_kimi_k3/reports/r1-corpus-plan-audit-decontam.json

df -h /tmp /workspace/torchtitan
