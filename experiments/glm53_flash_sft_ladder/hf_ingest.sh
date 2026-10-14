#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=experiments/glm53_flash_sft_ladder/lib/waystone.sh
source "${script_dir}/lib/waystone.sh"

usage() {
  cat >&2 <<'USAGE'
Usage: hf_ingest.sh --repo <org/name> --revision <rev|main> [--include <glob>]... [--parallel N]
                    [--keep-local DIR]

Downloads one Hugging Face model revision file-by-file, uploads each file under
the TorchTitan Waystone model-weights prefix with MD5 verification, and publishes
MANIFEST.jsonl plus _COMPLETE. Re-running a completed ingest skips files already
present on HDFS with the recorded size.

--keep-local DIR keeps every downloaded file in DIR (a staged cache) instead of
deleting it after upload, and writes DIR/MANIFEST.jsonl and DIR/.staged.json.
Use it when the model will be trained on this host soon: one HF download feeds
both HDFS and the local cache, avoiding an HDFS round trip.
USAGE
}

repo=""
revision=""
parallel=2
keep_local=""
includes=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      repo="$2"
      shift 2
      ;;
    --revision)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      revision="$2"
      shift 2
      ;;
    --include)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      includes+=("$2")
      shift 2
      ;;
    --parallel)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      parallel="$2"
      shift 2
      ;;
    --keep-local)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      keep_local="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$repo" || -z "$revision" ]]; then
  usage
  exit 2
fi
if [[ "$repo" != */* || "$repo" == */*/* ]]; then
  printf 'repo must have shape <org/name>: %s\n' "$repo" >&2
  exit 2
fi
if ! [[ "$parallel" =~ ^[0-9]+$ ]] || (( parallel < 1 )); then
  printf 'parallel must be a positive integer: %s\n' "$parallel" >&2
  exit 2
fi

ws_require jq
ws_require curl
ws_require flock
ws_require awk

cache_root="${GLM_LADDER_CACHE:-/data05/philip.yang/glm53-ladder-cache}"
tmp_root="${cache_root}/ingest-tmp/${repo//\//__}-${revision//\//__}-$$"
download_root="${tmp_root}/downloads"
if [[ -n "$keep_local" ]]; then
  mkdir -p "$keep_local"
  download_root="$(cd -- "$keep_local" && pwd)"
fi
mkdir -p "$download_root"

api_json="${tmp_root}/revision.json"
tree_json="${tmp_root}/tree.json"
files_jsonl="${tmp_root}/files.jsonl"
remote_manifest="${tmp_root}/remote-MANIFEST.jsonl"
manifest_unsorted="${tmp_root}/MANIFEST.unsorted.jsonl"
manifest_sorted="${tmp_root}/MANIFEST.jsonl"
metrics_jsonl="${tmp_root}/metrics.jsonl"
lock_file="${tmp_root}/manifest.lock"
mkdir -p "$tmp_root"
: > "$manifest_unsorted"
: > "$metrics_jsonl"
: > "$lock_file"

api_url="https://huggingface.co/api/models/${repo}/revision/${revision}"
curl -L --fail --retry 5 -sS "$api_url" -o "$api_json"
sha="$(jq -er '.sha' "$api_json")"
org="${repo%%/*}"
name="${repo#*/}"
hdfs_dir="$(ws_child_uri "model-weights/${org}/${name}/${sha}")"

tree_url="https://huggingface.co/api/models/${repo}/tree/${sha}?recursive=1"
curl -L --fail --retry 5 -sS "$tree_url" -o "$tree_json"

jq -r '.siblings[]?.rfilename // empty' "$api_json" > "${tmp_root}/siblings.txt"
jq -cr '
  .[]
  | select(.type == "file")
  | {
      file: .path,
      size: (.size // .lfs.size // null),
      sha256: (.lfs.oid // null)
    }
' "$tree_json" > "${tmp_root}/tree-files.jsonl"

if [[ -s "${tmp_root}/siblings.txt" ]]; then
  while IFS= read -r record; do
    file="$(jq -r '.file' <<<"$record")"
    if grep -Fxq -- "$file" "${tmp_root}/siblings.txt"; then
      printf '%s\n' "$record"
    fi
  done < "${tmp_root}/tree-files.jsonl" > "$files_jsonl"
else
  cp "${tmp_root}/tree-files.jsonl" "$files_jsonl"
fi

if ((${#includes[@]} > 0)); then
  filtered_jsonl="${tmp_root}/files.filtered.jsonl"
  : > "$filtered_jsonl"
  while IFS= read -r record; do
    file="$(jq -r '.file' <<<"$record")"
    matched=0
    for pattern in "${includes[@]}"; do
      # Intentionally use the right-hand side as a glob pattern.
      # shellcheck disable=SC2053
      if [[ "$file" == $pattern ]]; then
        matched=1
        break
      fi
    done
    if (( matched == 1 )); then
      printf '%s\n' "$record" >> "$filtered_jsonl"
    fi
  done < "$files_jsonl"
  mv "$filtered_jsonl" "$files_jsonl"
fi

file_count="$(wc -l < "$files_jsonl" | tr -d '[:space:]')"
if [[ "$file_count" == "0" ]]; then
  printf 'no files matched repo=%s revision=%s\n' "$repo" "$revision" >&2
  exit 1
fi

remote_complete=false
if ws_exists "$(ws_join_uri "$hdfs_dir" "_COMPLETE")"; then
  remote_complete=true
fi
if ws_exists "$(ws_join_uri "$hdfs_dir" "MANIFEST.jsonl")"; then
  "$WAYSTONE_BIN" --error-format json --overwrite --mkdir-parents \
    get "$(ws_join_uri "$hdfs_dir" "MANIFEST.jsonl")" "$remote_manifest" >/dev/null
else
  : > "$remote_manifest"
fi

urlencode_path() {
  jq -rn --arg value "$1" '$value | @uri' | sed 's/%2F/\//g'
}

remote_manifest_entry() {
  local file="$1"
  if [[ ! -s "$remote_manifest" ]]; then
    return 1
  fi
  jq -sce --arg file "$file" 'map(select(.file == $file)) | last // empty' "$remote_manifest"
}

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
  local file="$1" bytes="$2" download_ms="$3" put_ms="$4" skipped="$5"
  jq -nc \
    --arg file "$file" \
    --argjson bytes "$bytes" \
    --argjson download_ms "$download_ms" \
    --argjson put_ms "$put_ms" \
    --argjson skipped "$skipped" \
    '{file:$file, bytes:$bytes, download_ms:$download_ms, put_ms:$put_ms, skipped:$skipped}'
}

manifest_line() {
  local file="$1" size="$2" md5="$3" sha256="$4"
  if [[ -n "$sha256" && "$sha256" != "null" ]]; then
    jq -nc \
      --arg file "$file" \
      --argjson size "$size" \
      --arg md5 "$md5" \
      --arg sha256 "$sha256" \
      '{file:$file, size:$size, md5:$md5, "sha256-if-LFS-from-API":$sha256}'
  else
    jq -nc \
      --arg file "$file" \
      --argjson size "$size" \
      --arg md5 "$md5" \
      '{file:$file, size:$size, md5:$md5}'
  fi
}

ingest_one() {
  local record="$1"
  local file api_size sha256 file_uri existing_entry existing_size
  local remote_size local_path encoded_file url download_start download_end put_start put_end
  local size_bytes md5 download_ms put_ms

  file="$(jq -r '.file' <<<"$record")"
  api_size="$(jq -r '.size // empty' <<<"$record")"
  sha256="$(jq -r '.sha256 // empty' <<<"$record")"
  file_uri="$(ws_join_uri "$hdfs_dir" "$file")"

  existing_entry=""
  if existing_entry="$(remote_manifest_entry "$file")"; then
    existing_size="$(jq -r '.size' <<<"$existing_entry")"
    if remote_size="$(ws_hdfs_file_size "$file_uri" 2>/dev/null)" && [[ "$remote_size" == "$existing_size" ]]; then
      ws_log "skip existing ${file} (${existing_size} bytes)"
      append_outputs "$existing_entry" "$(record_metric "$file" "$existing_size" 0 0 true)"
      return 0
    fi
    ws_log "existing manifest entry did not match HDFS size for ${file}; refreshing upload"
  elif [[ -n "$api_size" ]] && remote_size="$(ws_hdfs_file_size "$file_uri" 2>/dev/null)" && [[ "$remote_size" == "$api_size" ]]; then
    ws_log "skip existing ${file} by HDFS size (${api_size} bytes); streaming MD5 for manifest"
    md5="$(ws_hdfs_md5 "$file_uri" "$(ws_timeout_for_size "$api_size")")"
    append_outputs "$(manifest_line "$file" "$api_size" "$md5" "$sha256")" \
      "$(record_metric "$file" "$api_size" 0 0 true)"
    return 0
  fi

  local_path="${download_root}/${file}"
  mkdir -p "$(dirname "$local_path")"
  encoded_file="$(urlencode_path "$file")"
  url="https://huggingface.co/${repo}/resolve/${sha}/${encoded_file}"
  download_start="$(date +%s%3N)"
  if [[ -n "$keep_local" && -n "$api_size" && -f "$local_path" ]] \
    && [[ "$(ws_file_size_bytes "$local_path")" == "$api_size" ]]; then
    ws_log "reuse kept local ${file}"
  else
    ws_log "download ${file}"
    curl -L --fail --retry 5 -C - -o "$local_path" "$url"
  fi
  download_end="$(date +%s%3N)"
  size_bytes="$(ws_file_size_bytes "$local_path")"
  if [[ -n "$api_size" && "$size_bytes" != "$api_size" ]]; then
    ws_die "size mismatch after download for ${file}: expected ${api_size}, got ${size_bytes}"
  fi
  md5="$(ws_md5_file "$local_path")"
  ws_log "upload ${file} (${size_bytes} bytes)"
  put_start="$(date +%s%3N)"
  ws_put_file "$local_path" "$file_uri"
  put_end="$(date +%s%3N)"
  if [[ -z "$keep_local" ]]; then
    rm -f "$local_path"
  fi
  download_ms=$((download_end - download_start))
  put_ms=$((put_end - put_start))
  append_outputs "$(manifest_line "$file" "$size_bytes" "$md5" "$sha256")" \
    "$(record_metric "$file" "$size_bytes" "$download_ms" "$put_ms" false)"
}

export -f ws_log ws_die ws_require ws_require_waystone ws_require_tools
export -f ws_file_size_bytes ws_md5_file ws_timeout_for_size ws_uri_dirname ws_uri_basename
export -f ws_join_uri ws_run_timed ws_exists ws_ls ws_hdfs_file_size ws_hdfs_md5 ws_put_file
export WAYSTONE_BIN WAYSTONE_PROJECT WAYSTONE_TRANSFER_RETRIES WAYSTONE_RETRY_BACKOFF_SECS
export WAYSTONE_BASE_TIMEOUT_SECS WAYSTONE_TIMEOUT_MIB_PER_SEC WAYSTONE_METADATA_TIMEOUT_SECS

failed=0
active=0
while IFS= read -r record; do
  ingest_one "$record" &
  active=$((active + 1))
  if (( active >= parallel )); then
    if ! wait -n; then
      failed=1
    fi
    active=$((active - 1))
  fi
done < "$files_jsonl"

while (( active > 0 )); do
  if ! wait -n; then
    failed=1
  fi
  active=$((active - 1))
done

if (( failed != 0 )); then
  printf 'one or more ingest jobs failed; leaving temp dir at %s\n' "$tmp_root" >&2
  exit 1
fi

jq -s -c 'sort_by(.file)[]' "$manifest_unsorted" > "$manifest_sorted"
complete_file="${tmp_root}/_COMPLETE"
{
  printf 'repo=%s\n' "$repo"
  printf 'revision=%s\n' "$revision"
  printf 'sha=%s\n' "$sha"
  printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$complete_file"

download_stats="$(jq -s '
  map(select(.skipped == false))
  | {
      bytes: (map(.bytes) | add // 0),
      ms: (map(.download_ms) | add // 0)
    }
' "$metrics_jsonl")"
put_stats="$(jq -s '
  map(select(.skipped == false))
  | {
      bytes: (map(.bytes) | add // 0),
      ms: (map(.put_ms) | add // 0)
    }
' "$metrics_jsonl")"
skipped_count="$(jq -s 'map(select(.skipped == true)) | length' "$metrics_jsonl")"
download_mbps="$(jq -r 'if .ms > 0 then ((.bytes / 1048576) / (.ms / 1000)) else 0 end | tostring' <<<"$download_stats")"
put_mbps="$(jq -r 'if .ms > 0 then ((.bytes / 1048576) / (.ms / 1000)) else 0 end | tostring' <<<"$put_stats")"

if [[ "$skipped_count" == "$file_count" && "$remote_complete" == true ]]; then
  ws_log "remote tree already complete; skipped MANIFEST.jsonl and _COMPLETE upload"
else
  ws_put_file "$manifest_sorted" "$(ws_join_uri "$hdfs_dir" "MANIFEST.jsonl")"
  ws_put_file "$complete_file" "$(ws_join_uri "$hdfs_dir" "_COMPLETE")"
fi

if [[ -n "$keep_local" ]]; then
  # Same format as stage.sh so later stage.sh calls treat the dir as staged.
  cp "$manifest_sorted" "${download_root}/MANIFEST.jsonl"
  jq -n \
    --arg source_uri "$hdfs_dir" \
    --arg manifest_md5 "$(ws_md5_file "$manifest_sorted")" \
    --arg staged_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson file_count "$file_count" \
    '{source_uri:$source_uri, manifest_md5:$manifest_md5, file_count:$file_count, staged_at:$staged_at, kept_from_ingest:true}' \
    > "${download_root}/.staged.json"
  printf 'KEPT_LOCAL=%s\n' "$download_root"
fi

printf 'HDFS_URI=%s\n' "$hdfs_dir"
printf 'FILES=%s\n' "$file_count"
printf 'SKIPPED=%s\n' "$skipped_count"
printf 'HF_DOWNLOAD_MBPS=%s\n' "$download_mbps"
printf 'HDFS_PUT_MBPS=%s\n' "$put_mbps"
printf 'LOCAL_MANIFEST=%s\n' "$manifest_sorted"
printf 'METRICS=%s\n' "$metrics_jsonl"
