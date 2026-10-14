#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=experiments/glm53_flash_sft_ladder/lib/waystone.sh
source "${script_dir}/lib/waystone.sh"

usage() {
  cat >&2 <<'USAGE'
Usage: publish.sh [--parallel N] <local-dir> <relpath-under-torchtitan>

Publishes a local artifact directory to the TorchTitan Waystone prefix with
per-file MD5 verification, then uploads MANIFEST.jsonl and _COMPLETE.
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

local_dir="${1%/}"
relpath="$2"
if [[ ! -d "$local_dir" ]]; then
  printf 'local directory does not exist: %s\n' "$local_dir" >&2
  exit 1
fi

hdfs_dir="$(ws_child_uri "$relpath")"
cache_root="${GLM_LADDER_CACHE:-/data05/philip.yang/glm53-ladder-cache}"
tmp_root="${cache_root}/publish-tmp/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$tmp_root"

files_nul="${tmp_root}/files.nul"
manifest_unsorted="${tmp_root}/MANIFEST.unsorted.jsonl"
manifest_sorted="${tmp_root}/MANIFEST.jsonl"
metrics_jsonl="${tmp_root}/metrics.jsonl"
lock_file="${tmp_root}/publish.lock"
: > "$manifest_unsorted"
: > "$metrics_jsonl"
: > "$lock_file"

find "$local_dir" -type f -print0 | sort -z > "$files_nul"

append_outputs() {
  local manifest_line="$1"
  local metric_line="$2"
  {
    flock 9
    printf '%s\n' "$manifest_line" >> "$manifest_unsorted"
    printf '%s\n' "$metric_line" >> "$metrics_jsonl"
  } 9>> "$lock_file"
}

record_metric() {
  local file="$1" bytes="$2" put_ms="$3"
  jq -nc \
    --arg file "$file" \
    --argjson bytes "$bytes" \
    --argjson put_ms "$put_ms" \
    '{file:$file, bytes:$bytes, put_ms:$put_ms}'
}

publish_one() {
  local local_path="$1"
  local rel file_uri size_bytes md5 put_start put_end put_ms
  rel="${local_path#"${local_dir}"/}"
  case "$rel" in
    MANIFEST.jsonl|_COMPLETE)
      return 0
      ;;
  esac
  size_bytes="$(ws_file_size_bytes "$local_path")"
  md5="$(ws_md5_file "$local_path")"
  file_uri="$(ws_join_uri "$hdfs_dir" "$rel")"
  put_start="$(date +%s%3N)"
  ws_put_file "$local_path" "$file_uri"
  put_end="$(date +%s%3N)"
  put_ms=$((put_end - put_start))
  append_outputs \
    "$(jq -nc --arg file "$rel" --argjson size "$size_bytes" --arg md5 "$md5" '{file:$file, size:$size, md5:$md5}')" \
    "$(record_metric "$rel" "$size_bytes" "$put_ms")"
}

failed=0
active=0
while IFS= read -r -d '' local_path; do
  publish_one "$local_path" &
  active=$((active + 1))
  if (( active >= parallel )); then
    if ! wait -n; then
      failed=1
    fi
    active=$((active - 1))
  fi
done < "$files_nul"

while (( active > 0 )); do
  if ! wait -n; then
    failed=1
  fi
  active=$((active - 1))
done

if (( failed != 0 )); then
  printf 'one or more publish jobs failed; leaving temp dir at %s\n' "$tmp_root" >&2
  exit 1
fi

jq -s -c 'sort_by(.file)[]' "$manifest_unsorted" > "$manifest_sorted"
complete_file="${tmp_root}/_COMPLETE"
{
  printf 'source_dir=%s\n' "$local_dir"
  printf 'relpath=%s\n' "$relpath"
  printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$complete_file"

ws_put_file "$manifest_sorted" "$(ws_join_uri "$hdfs_dir" "MANIFEST.jsonl")"
ws_put_file "$complete_file" "$(ws_join_uri "$hdfs_dir" "_COMPLETE")"

put_stats="$(jq -s '
  {
    bytes: (map(.bytes) | add // 0),
    ms: (map(.put_ms) | add // 0)
  }
' "$metrics_jsonl")"
file_count="$(wc -l < "$manifest_sorted" | tr -d '[:space:]')"
put_mbps="$(jq -r 'if .ms > 0 then ((.bytes / 1048576) / (.ms / 1000)) else 0 end | tostring' <<<"$put_stats")"

printf 'HDFS_URI=%s\n' "$hdfs_dir"
printf 'FILES=%s\n' "$file_count"
printf 'HDFS_PUT_MBPS=%s\n' "$put_mbps"
printf 'LOCAL_MANIFEST=%s\n' "$manifest_sorted"
printf 'METRICS=%s\n' "$metrics_jsonl"
