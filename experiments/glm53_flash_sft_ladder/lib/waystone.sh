#!/usr/bin/env bash
# Shared Waystone helpers for the GLM-5.3-Flash SFT ladder runners.
set -euo pipefail

: "${WAYSTONE_BIN:=/data02/home/philip.yang/workspace/waystone/scripts/waystone}"
: "${WAYSTONE_PROJECT:=torchtitan}"
: "${WAYSTONE_TRANSFER_RETRIES:=3}"
: "${WAYSTONE_RETRY_BACKOFF_SECS:=5}"
: "${WAYSTONE_BASE_TIMEOUT_SECS:=300}"
: "${WAYSTONE_TIMEOUT_MIB_PER_SEC:=20}"
: "${WAYSTONE_METADATA_TIMEOUT_SECS:=120}"

ws_log() {
  printf '[waystone] %s\n' "$*" >&2
}

ws_die() {
  ws_log "ERROR: $*"
  return 1
}

ws_require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    ws_die "required command not found: $1"
  fi
}

ws_require_waystone() {
  if [[ ! -x "$WAYSTONE_BIN" ]]; then
    ws_die "WAYSTONE_BIN is not executable: $WAYSTONE_BIN"
  fi
}

ws_require_tools() {
  ws_require_waystone
  ws_require md5sum
  ws_require timeout
}

ws_normalize_child() {
  local relpath="$1"
  relpath="${relpath#./}"
  relpath="${relpath#/}"
  case "$relpath" in
    "")
      ws_die "empty Waystone child path"
      ;;
    hdfs://*)
      ws_die "expected a child path, got an HDFS URI: $relpath"
      ;;
    "$WAYSTONE_PROJECT"|"$WAYSTONE_PROJECT"/*)
      printf '%s\n' "$relpath"
      ;;
    *)
      printf '%s/%s\n' "$WAYSTONE_PROJECT" "$relpath"
      ;;
  esac
}

ws_child_uri() {
  local child
  child="$(ws_normalize_child "$1")" || return 1
  "$WAYSTONE_BIN" storage-prefix --child "$child"
}

ws_layout() {
  "$WAYSTONE_BIN" layout-profile --project "$WAYSTONE_PROJECT" --json
}

ws_file_size_bytes() {
  wc -c < "$1" | tr -d '[:space:]'
}

ws_md5_file() {
  md5sum "$1" | awk '{print $1}'
}

ws_timeout_for_size() {
  local size_bytes="$1"
  local bytes_per_sec timeout_secs
  bytes_per_sec=$((WAYSTONE_TIMEOUT_MIB_PER_SEC * 1024 * 1024))
  if (( bytes_per_sec <= 0 )); then
    bytes_per_sec=$((20 * 1024 * 1024))
  fi
  timeout_secs=$((WAYSTONE_BASE_TIMEOUT_SECS + ((size_bytes + bytes_per_sec - 1) / bytes_per_sec)))
  printf '%s\n' "$timeout_secs"
}

ws_timeout_for_file() {
  ws_timeout_for_size "$(ws_file_size_bytes "$1")"
}

ws_uri_dirname() {
  local uri="$1"
  printf '%s\n' "${uri%/*}"
}

ws_uri_basename() {
  local uri="$1"
  printf '%s\n' "${uri##*/}"
}

ws_join_uri() {
  local base_uri="$1"
  local relpath="$2"
  base_uri="${base_uri%/}"
  relpath="${relpath#./}"
  relpath="${relpath#/}"
  if [[ -z "$relpath" ]]; then
    printf '%s\n' "$base_uri"
  else
    printf '%s/%s\n' "$base_uri" "$relpath"
  fi
}

ws_run_timed() {
  local timeout_secs="$1"
  shift
  set +e
  timeout --kill-after=15s "${timeout_secs}s" "$@"
  local status=$?
  set -e
  if [[ "$status" -eq 124 || "$status" -eq 137 || "$status" -eq 11 ]]; then
    ws_log "ERROR: command timed out after ${timeout_secs}s: $*"
  fi
  return "$status"
}

ws_exists() {
  ws_require_waystone
  set +e
  "$WAYSTONE_BIN" --error-format json \
    --command-timeout-secs "$WAYSTONE_METADATA_TIMEOUT_SECS" \
    raw test -e "$1" >/dev/null 2>&1
  local status=$?
  set -e
  return "$status"
}

ws_ls() {
  ws_require_waystone
  "$WAYSTONE_BIN" --error-format json --output json \
    --command-timeout-secs "$WAYSTONE_METADATA_TIMEOUT_SECS" \
    ls "$1"
}

ws_hdfs_file_size() {
  local uri="$1"
  local parent name listing
  parent="$(ws_uri_dirname "$uri")"
  name="$(ws_uri_basename "$uri")"
  listing="$(ws_ls "$parent")" || return 1
  jq -er --arg name "$name" \
    '.entries[] | select(.name == $name and .kind == "FILE") | .size' \
    <<<"$listing"
}

ws_hdfs_md5() {
  local uri="$1"
  local timeout_secs="${2:-$WAYSTONE_BASE_TIMEOUT_SECS}"
  local output
  set +e
  output="$(timeout --kill-after=15s "${timeout_secs}s" "$WAYSTONE_BIN" \
    --error-format json \
    --command-timeout-secs "$timeout_secs" \
    cat "$uri" | md5sum)"
  local status=$?
  set -e
  if [[ "$status" -eq 124 || "$status" -eq 137 || "$status" -eq 11 ]]; then
    ws_log "ERROR: HDFS MD5 stream timed out after ${timeout_secs}s: $uri"
    return "$status"
  fi
  if [[ "$status" -ne 0 ]]; then
    ws_log "ERROR: failed to stream HDFS file for MD5: $uri"
    return "$status"
  fi
  printf '%s\n' "${output%% *}"
}

ws_put_file() {
  local local_path="$1"
  local hdfs_uri="$2"
  ws_require_tools
  if [[ ! -f "$local_path" ]]; then
    ws_die "local file does not exist: $local_path"
  fi

  local size_bytes timeout_secs attempt status backoff_ms
  size_bytes="$(ws_file_size_bytes "$local_path")"
  timeout_secs="$(ws_timeout_for_size "$size_bytes")"
  backoff_ms=$((WAYSTONE_RETRY_BACKOFF_SECS * 1000))
  attempt=1
  while (( attempt <= WAYSTONE_TRANSFER_RETRIES )); do
    ws_log "put attempt ${attempt}/${WAYSTONE_TRANSFER_RETRIES}: ${local_path} -> ${hdfs_uri}"
    set +e
    ws_run_timed "$timeout_secs" "$WAYSTONE_BIN" \
      --error-format json \
      --verify-md5 \
      --overwrite \
      --mkdir-parents \
      --transfer-retries 0 \
      --retry-backoff-ms "$backoff_ms" \
      --command-timeout-secs "$timeout_secs" \
      put "$local_path" "$hdfs_uri"
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      return 0
    fi
    if [[ "$status" -eq 124 || "$status" -eq 137 || "$status" -eq 11 ]]; then
      ws_log "ERROR: timed out uploading ${local_path} to ${hdfs_uri}"
    else
      ws_log "ERROR: upload failed with status ${status}: ${local_path} -> ${hdfs_uri}"
    fi
    if (( attempt < WAYSTONE_TRANSFER_RETRIES )); then
      sleep "$WAYSTONE_RETRY_BACKOFF_SECS"
    fi
    attempt=$((attempt + 1))
  done
  return "$status"
}

ws_get_file() {
  local hdfs_uri="$1"
  local local_path="$2"
  ws_require_tools
  mkdir -p "$(dirname "$local_path")"

  local expected_md5="${WS_EXPECTED_MD5:-}"
  local expected_size="${WS_EXPECTED_SIZE:-}"
  local timeout_secs status sidecar_tmp
  if [[ "$expected_size" =~ ^[0-9]+$ ]]; then
    timeout_secs="$(ws_timeout_for_size "$expected_size")"
  else
    timeout_secs="$WAYSTONE_BASE_TIMEOUT_SECS"
  fi
  set +e
  ws_run_timed "$timeout_secs" "$WAYSTONE_BIN" \
    --error-format json \
    --verify-md5 \
    --overwrite \
    --mkdir-parents \
    --transfer-retries "$WAYSTONE_TRANSFER_RETRIES" \
    --retry-backoff-ms "$((WAYSTONE_RETRY_BACKOFF_SECS * 1000))" \
    --command-timeout-secs "$timeout_secs" \
    get "$hdfs_uri" "$local_path"
  status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    if [[ "$status" -eq 124 || "$status" -eq 137 || "$status" -eq 11 ]]; then
      ws_log "ERROR: timed out downloading ${hdfs_uri} to ${local_path}"
    else
      ws_log "ERROR: download failed with status ${status}: ${hdfs_uri} -> ${local_path}"
    fi
    return "$status"
  fi

  if [[ -z "$expected_md5" && -f "${local_path}.md5" ]]; then
    expected_md5="$(awk '{print $1}' "${local_path}.md5")"
  fi
  if [[ -z "$expected_md5" ]] && ws_exists "${hdfs_uri}.md5"; then
    sidecar_tmp="$(mktemp)"
    set +e
    "$WAYSTONE_BIN" --error-format json --overwrite \
      --command-timeout-secs "$WAYSTONE_METADATA_TIMEOUT_SECS" \
      get "${hdfs_uri}.md5" "$sidecar_tmp" >/dev/null
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      expected_md5="$(awk '{print $1}' "$sidecar_tmp")"
    fi
    rm -f "$sidecar_tmp"
  fi

  if [[ -n "$expected_md5" ]]; then
    local actual_md5
    actual_md5="$(ws_md5_file "$local_path")"
    if [[ "$actual_md5" != "$expected_md5" ]]; then
      ws_die "MD5 mismatch for ${local_path}: expected ${expected_md5}, got ${actual_md5}"
    fi
  else
    ws_log "warning: no manifest or sidecar MD5 supplied for ${hdfs_uri}; used Waystone transfer verification only"
  fi
}
