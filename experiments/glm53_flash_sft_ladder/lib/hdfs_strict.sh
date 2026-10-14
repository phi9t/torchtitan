#!/usr/bin/env bash
# Verify that local input caches are faithful copies of completed HDFS trees.
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: hdfs_strict.sh [--quick] <local-dir>...

Requires each local directory to contain .staged.json and MANIFEST.jsonl.
Every manifest entry must exist locally with the recorded size and MD5.
In --quick mode, MD5 checks are cached in .verified.json by file mtime+size.
USAGE
}

quick=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      quick=1
      shift
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

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

fail() {
  printf '[hdfs-strict] ERROR: %s\n' "$*" >&2
  exit 1
}

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "required command not found: $1"
  fi
}

file_size_bytes() {
  stat -c '%s' "$1"
}

file_mtime_seconds() {
  stat -c '%Y' "$1"
}

file_md5() {
  md5sum "$1" | awk '{print $1}'
}

valid_manifest_path() {
  local relpath="$1"
  case "$relpath" in
    ""|.|..|/*|../*|*/../*|*/..)
      return 1
      ;;
  esac
  return 0
}

cache_entry_valid() {
  local cache_path="$1" relpath="$2" size="$3" mtime="$4" expected_md5="$5"
  [[ -f "$cache_path" ]] || return 1
  jq -e \
    --arg file "$relpath" \
    --arg md5 "$expected_md5" \
    --argjson size "$size" \
    --argjson mtime "$mtime" \
    '.files[$file].size == $size
      and .files[$file].mtime == $mtime
      and .files[$file].md5 == $md5' \
    "$cache_path" >/dev/null 2>&1
}

record_verified_entry() {
  local output_jsonl="$1" relpath="$2" size="$3" mtime="$4" md5="$5"
  jq -nc \
    --arg file "$relpath" \
    --arg md5 "$md5" \
    --argjson size "$size" \
    --argjson mtime "$mtime" \
    '{file:$file,size:$size,mtime:$mtime,md5:$md5}' >> "$output_jsonl"
}

write_quick_cache() {
  local output_jsonl="$1" cache_path="$2" source_uri="$3" manifest_md5="$4"
  local cache_tmp
  cache_tmp="${cache_path}.tmp.$$"
  jq -s \
    --arg source_uri "$source_uri" \
    --arg manifest_md5 "$manifest_md5" \
    --arg verified_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{
      version: 1,
      source_uri: $source_uri,
      manifest_md5: $manifest_md5,
      verified_at: $verified_at,
      files: (map({key: .file, value: {size: .size, mtime: .mtime, md5: .md5}}) | from_entries)
    }' "$output_jsonl" > "$cache_tmp"
  mv "$cache_tmp" "$cache_path"
}

verify_entry() {
  local dir="$1" cache_path="$2" quick_entries="$3" record="$4"
  local relpath expected_size expected_md5 local_path actual_size mtime actual_md5

  relpath="$(jq -er '.file' <<<"$record")"
  expected_size="$(jq -er '.size' <<<"$record")"
  expected_md5="$(jq -er '.md5' <<<"$record")"

  valid_manifest_path "$relpath" || fail "$dir manifest contains unsafe path: $relpath"
  [[ "$expected_size" =~ ^[0-9]+$ ]] || fail "$dir manifest has nonnumeric size for $relpath"
  [[ "$expected_md5" =~ ^[0-9a-fA-F]{32}$ ]] || fail "$dir manifest has invalid md5 for $relpath"

  local_path="${dir%/}/${relpath}"
  [[ -f "$local_path" ]] || fail "$dir missing manifest file: $relpath"

  actual_size="$(file_size_bytes "$local_path")"
  if [[ "$actual_size" != "$expected_size" ]]; then
    fail "$dir/$relpath size mismatch: expected $expected_size, got $actual_size"
  fi

  mtime="$(file_mtime_seconds "$local_path")"
  if (( quick == 1 )) && cache_entry_valid "$cache_path" "$relpath" "$actual_size" "$mtime" "$expected_md5"; then
    record_verified_entry "$quick_entries" "$relpath" "$actual_size" "$mtime" "$expected_md5"
    return 0
  fi

  actual_md5="$(file_md5 "$local_path")"
  if [[ "${actual_md5,,}" != "${expected_md5,,}" ]]; then
    fail "$dir/$relpath md5 mismatch: expected ${expected_md5,,}, got ${actual_md5,,}"
  fi

  if (( quick == 1 )); then
    record_verified_entry "$quick_entries" "$relpath" "$actual_size" "$mtime" "${expected_md5,,}"
  fi
}

verify_dir() {
  local dir="$1"
  local staged_json manifest_jsonl cache_json quick_entries source_uri staged_manifest_md5 manifest_md5
  local file_count staged_file_count record mode

  [[ -d "$dir" ]] || fail "local directory does not exist: $dir"
  staged_json="${dir%/}/.staged.json"
  manifest_jsonl="${dir%/}/MANIFEST.jsonl"
  cache_json="${dir%/}/.verified.json"
  [[ -f "$staged_json" ]] || fail "$dir missing .staged.json"
  [[ -f "$manifest_jsonl" ]] || fail "$dir missing MANIFEST.jsonl"

  source_uri="$(jq -er '.source_uri' "$staged_json")"
  staged_manifest_md5="$(jq -er '.manifest_md5' "$staged_json")"
  case "$source_uri" in
    hdfs://*) ;;
    *) fail "$dir .staged.json source_uri is not hdfs://: $source_uri" ;;
  esac

  manifest_md5="$(file_md5 "$manifest_jsonl")"
  if [[ "${manifest_md5,,}" != "${staged_manifest_md5,,}" ]]; then
    fail "$dir MANIFEST.jsonl md5 mismatch: staged=${staged_manifest_md5,,}, local=${manifest_md5,,}"
  fi

  file_count="$(wc -l < "$manifest_jsonl" | tr -d '[:space:]')"
  staged_file_count="$(jq -r '.file_count // empty' "$staged_json")"
  if [[ -n "$staged_file_count" && "$staged_file_count" != "$file_count" ]]; then
    fail "$dir manifest file count mismatch: staged=$staged_file_count, local=$file_count"
  fi

  quick_entries=""
  if (( quick == 1 )); then
    quick_entries="$(mktemp)"
  fi

  while IFS= read -r record; do
    [[ -n "$record" ]] || continue
    verify_entry "$dir" "$cache_json" "$quick_entries" "$record"
  done < "$manifest_jsonl"

  if (( quick == 1 )); then
    write_quick_cache "$quick_entries" "$cache_json" "$source_uri" "${manifest_md5,,}"
    rm -f "$quick_entries"
  fi

  mode="full"
  if (( quick == 1 )); then
    mode="quick"
  fi
  printf 'HDFS_STRICT_DIR=%s SOURCE_URI=%s MANIFEST_MD5=%s FILES=%s MODE=%s\n' \
    "$dir" "$source_uri" "${manifest_md5,,}" "$file_count" "$mode"
}

require_tool jq
require_tool md5sum
require_tool awk
require_tool stat

for dir in "$@"; do
  verify_dir "$dir"
done
