#!/usr/bin/env bash
# T03: ingest every ladder model to HDFS and keep a staged local copy on /data05.
# Order follows the ladder so earlier rungs unblock first.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cache="${GLM_LADDER_CACHE:-/data05/philip.yang/glm53-ladder-cache}/model-weights"
out="${here}/results/t03"
mkdir -p "$out"
ingest() {
  local repo="$1" rev="$2" par="$3"
  local log="${out}/$(tr '/' '_' <<<"$repo").log"
  echo "[t03] $(date -u +%FT%TZ) start ${repo}@${rev}"
  if "${here}/hf_ingest.sh" --repo "$repo" --revision "$rev" --parallel "$par" \
      --keep-local "${cache}/${repo}" > "$log" 2>&1; then
    echo "[t03] $(date -u +%FT%TZ) done ${repo}: $(grep -E 'HDFS_URI|HF_DOWNLOAD_MBPS|HDFS_PUT_MBPS' "$log" | tr '\n' ' ')"
  else
    echo "[t03] $(date -u +%FT%TZ) FAILED ${repo} (see ${log})"
    return 1
  fi
}
ingest Qwen/Qwen3-8B main 4
ingest Qwen/Qwen3-30B-A3B main 6
# Pinned revision from docs/research/2026-10-14-glm-5.3-flash-architecture-and-finetuning.md
ingest zai-org/GLM-5.3-Flash-BF16 a5b45eb41df6402735dedc900be14a42e8d5e538 8
echo "[t03] ALL_DONE"
