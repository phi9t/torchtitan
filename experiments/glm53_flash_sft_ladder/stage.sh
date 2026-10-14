#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=experiments/glm53_flash_sft_ladder/lib/waystone.sh
source "${script_dir}/lib/waystone.sh"

usage() {
  cat >&2 <<'USAGE'
Usage: stage.sh [--parallel N] <hdfs-uri-dir> <local-dir>

Stages a completed Waystone artifact tree from HDFS into a local cache
directory, verifying every file against MANIFEST.jsonl.
USAGE
}

parallel=2
while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      parallel="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      printf 'unknown argument: %s\n' "$1" >&2
      usage
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi
if ! [[ "$parallel" =~ ^[0-9]+$ ]] || (( parallel < 1 )); then
  printf 'parallel must be a positive integer: %s\n' "$parallel" >&2
  exit 2
fi

ws_require jq
ws_require flock
ws_require awk

hdfs_dir="${1%/}"
local_dir="$2"
cache_root="${GLM_LADDER_CACHE:-/data05/philip.yang/glm53-ladder-cache}"
tmp_root="${cache_root}/stage-tmp/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$tmp_root" "$local_dir"

manifest_local="${tmp_root}/MANIFEST.jsonl"
metrics_jsonl="${tmp_root}/metrics.jsonl"
lock_file="${tmp_root}/stage.lock"
: > "$metrics_jsonl"
: > "$lock_file"

if ! ws_exists "$(ws_join_uri "$hdfs_dir" "_COMPLETE")"; then
  printf 'missing _COMPLETE marker under %s\n' "$hdfs_dir" >&2
  exit 1
fi

ws_get_file "$(ws_join_uri "$hdfs_dir" "MANIFEST.jsonl")" "$manifest_local"
manifest_md5="$(ws_md5_file "$manifest_local")"
file_count="$(wc -l < "$manifest_local" | tr -d '[:space:]')"

append_metric() {
  local metric_line="$1"
  {
    flock 9
    printf '%s\n' "$metric_line" >> "$metrics_jsonl"
  } 9>> "$lock_file"
}

record_metric() {
  local file="$1" bytes="$2" get_ms="$3" skipped="$4"
  jq -nc \
    --arg file "$file" \
    --argjson bytes "$bytes" \
    --argjson get_ms "$get_ms" \
    --argjson skipped "$skipped" \
    '{file:$file, bytes:$bytes, get_ms:$get_ms, skipped:$skipped}'
}

stage_one() {
  local record="$1"
  local file size expected_md5 local_path actual_md5 get_start get_end get_ms file_uri
  file="$(jq -r '.file' <<<"$record")"
  size="$(jq -r '.size' <<<"$record")"
  expected_md5="$(jq -er '.md5' <<<"$record")"
  local_path="${local_dir%/}/${file}"

  if [[ -f "$local_path" ]]; then
    actual_md5="$(ws_md5_file "$local_path")"
    if [[ "$actual_md5" == "$expected_md5" ]]; then
      ws_log "skip staged ${file}"
      append_metric "$(record_metric "$file" "$size" 0 true)"
      return 0
    fi
    ws_log "refresh staged ${file}; MD5 mismatch"
  fi

  file_uri="$(ws_join_uri "$hdfs_dir" "$file")"
  get_start="$(date +%s%3N)"
  WS_EXPECTED_MD5="$expected_md5" WS_EXPECTED_SIZE="$size" ws_get_file "$file_uri" "$local_path"
  get_end="$(date +%s%3N)"
  actual_md5="$(ws_md5_file "$local_path")"
  if [[ "$actual_md5" != "$expected_md5" ]]; then
    ws_die "MD5 mismatch after staging ${file}: expected ${expected_md5}, got ${actual_md5}"
  fi
  get_ms=$((get_end - get_start))
  append_metric "$(record_metric "$file" "$size" "$get_ms" false)"
}

failed=0
active=0
while IFS= read -r record; do
  stage_one "$record" &
  active=$((active + 1))
  if (( active >= parallel )); then
    if ! wait -n; then
      failed=1
    fi
    active=$((active - 1))
  fi
done < "$manifest_local"

while (( active > 0 )); do
  if ! wait -n; then
    failed=1
  fi
  active=$((active - 1))
done

if (( failed != 0 )); then
  printf 'one or more stage jobs failed; leaving temp dir at %s\n' "$tmp_root" >&2
  exit 1
fi

staged_json="${local_dir%/}/.staged.json"
jq -n \
  --arg source_uri "$hdfs_dir" \
  --arg manifest_md5 "$manifest_md5" \
  --arg staged_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson file_count "$file_count" \
  '{source_uri:$source_uri, manifest_md5:$manifest_md5, file_count:$file_count, staged_at:$staged_at}' \
  > "$staged_json"

get_stats="$(jq -s '
  map(select(.skipped == false))
  | {
      bytes: (map(.bytes) | add // 0),
      ms: (map(.get_ms) | add // 0)
    }
' "$metrics_jsonl")"
skipped_count="$(jq -s 'map(select(.skipped == true)) | length' "$metrics_jsonl")"
get_mbps="$(jq -r 'if .ms > 0 then ((.bytes / 1048576) / (.ms / 1000)) else 0 end | tostring' <<<"$get_stats")"

printf 'STAGED_DIR=%s\n' "$local_dir"
printf 'SOURCE_URI=%s\n' "$hdfs_dir"
printf 'FILES=%s\n' "$file_count"
printf 'SKIPPED=%s\n' "$skipped_count"
printf 'HDFS_GET_MBPS=%s\n' "$get_mbps"
printf 'STAGED_JSON=%s\n' "$staged_json"
printf 'METRICS=%s\n' "$metrics_jsonl"
