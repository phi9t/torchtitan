#!/usr/bin/env bash
# Fail-closed path and logging helpers for the Scout A host orchestrator.

scout_contract_error() {
  printf 'error: %s\n' "$*" >&2
  return 1
}

scout_require_directory() {
  local requested="$1"
  if [[ -L "${requested}" ]]; then
    scout_contract_error "managed directory must not be a symlink: ${requested}"
    return 1
  fi
  if [[ ! -d "${requested}" ]]; then
    scout_contract_error "managed path is not a directory: ${requested}"
    return 1
  fi
}

scout_ensure_directory_tree() {
  local base="$1"
  local relative="$2"
  scout_require_directory "${base}" || return 1
  local canonical_base current component candidate canonical
  canonical_base="$(cd -P -- "${base}" && pwd)" || return 1
  current="${canonical_base}"
  local -a components
  IFS='/' read -r -a components <<<"${relative}"
  for component in "${components[@]}"; do
    [[ -n "${component}" && "${component}" != "." && "${component}" != ".." ]] \
      || scout_contract_error "invalid managed directory component: ${relative}" \
      || return 1
    candidate="${current}/${component}"
    [[ ! -L "${candidate}" ]] \
      || scout_contract_error "managed directory must not be a symlink: ${candidate}" \
      || return 1
    if [[ -e "${candidate}" ]]; then
      [[ -d "${candidate}" ]] \
        || scout_contract_error "managed path is not a directory: ${candidate}" \
        || return 1
    else
      mkdir -- "${candidate}" || return 1
    fi
    [[ ! -L "${candidate}" ]] \
      || scout_contract_error "managed directory became a symlink: ${candidate}" \
      || return 1
    canonical="$(cd -P -- "${candidate}" && pwd)" || return 1
    case "${canonical}/" in
      "${canonical_base}/"*) ;;
      *) scout_contract_error "managed directory escapes approved root: ${candidate}" \
        || return 1 ;;
    esac
    current="${canonical}"
  done
  printf '%s\n' "${current}"
}

scout_require_regular_file() {
  local approved_root="$1"
  local requested="$2"
  scout_require_directory "${approved_root}" || return 1
  [[ ! -L "${requested}" ]] \
    || scout_contract_error "managed file must not be a symlink: ${requested}" \
    || return 1
  [[ -f "${requested}" ]] \
    || scout_contract_error "managed path is not a regular file: ${requested}" \
    || return 1
  local canonical_root canonical_parent
  canonical_root="$(cd -P -- "${approved_root}" && pwd)" || return 1
  canonical_parent="$(cd -P -- "$(dirname -- "${requested}")" && pwd)" \
    || return 1
  case "${canonical_parent}/" in
    "${canonical_root}/"*) ;;
    *) scout_contract_error "managed file escapes approved root: ${requested}" \
      || return 1 ;;
  esac
}

scout_create_fresh_log() {
  local approved_root="$1"
  local requested="$2"
  scout_require_directory "${approved_root}" || return 1
  local canonical_root parent relative_parent
  canonical_root="$(cd -P -- "${approved_root}" && pwd)" || return 1
  case "${requested}" in
    "${canonical_root}/"*) ;;
    *) scout_contract_error "log path escapes approved root: ${requested}" \
      || return 1 ;;
  esac
  parent="$(dirname -- "${requested}")"
  relative_parent="${parent#"${canonical_root}/"}"
  if [[ "${parent}" != "${canonical_root}" ]]; then
    scout_ensure_directory_tree "${canonical_root}" "${relative_parent}" >/dev/null \
      || return 1
  fi
  [[ ! -L "${requested}" ]] \
    || scout_contract_error "fresh log must not be a symlink: ${requested}" \
    || return 1
  [[ ! -e "${requested}" ]] \
    || scout_contract_error "fresh log already exists: ${requested}" \
    || return 1
  (set -o noclobber; : >"${requested}") || return 1
  scout_require_regular_file "${canonical_root}" "${requested}" || return 1
  chmod 0600 -- "${requested}"
}

scout_capture_fresh_file() {
  local approved_root="$1"
  local requested="$2"
  shift 2
  [[ "${1:-}" == "--" ]] \
    || scout_contract_error "scout_capture_fresh_file requires -- before the command" \
    || return 1
  shift
  scout_require_directory "${approved_root}" || return 1
  local canonical_root parent relative_parent
  canonical_root="$(cd -P -- "${approved_root}" && pwd)" || return 1
  case "${requested}" in
    "${canonical_root}/"*) ;;
    *) scout_contract_error "captured file escapes approved root: ${requested}" \
      || return 1 ;;
  esac
  parent="$(dirname -- "${requested}")"
  relative_parent="${parent#"${canonical_root}/"}"
  if [[ "${parent}" != "${canonical_root}" ]]; then
    scout_ensure_directory_tree "${canonical_root}" "${relative_parent}" >/dev/null \
      || return 1
  fi
  [[ ! -L "${requested}" ]] \
    || scout_contract_error "fresh captured file must not be a symlink: ${requested}" \
    || return 1
  [[ ! -e "${requested}" ]] \
    || scout_contract_error "fresh captured file already exists: ${requested}" \
    || return 1
  (set -o noclobber; "$@" >"${requested}") || return 1
  scout_require_regular_file "${canonical_root}" "${requested}" || return 1
  chmod 0444 -- "${requested}"
  [[ "$(stat -c '%a' -- "${requested}")" == "444" ]] \
    || scout_contract_error "captured file is not read-only: ${requested}"
}

scout_append_stage() {
  local approved_root="$1"
  local journal="$2"
  local stage_name="$3"
  local evidence_path="$4"
  local command_record="$5"
  scout_require_regular_file "${approved_root}" "${journal}" || return 1
  [[ "$(stat -c '%a' -- "${journal}")" == "600" ]] \
    || scout_contract_error "open stage journal must have mode 0600: ${journal}" \
    || return 1
  [[ "${stage_name}" != *$'\t'* && "${stage_name}" != *$'\n'* ]] \
    || scout_contract_error "stage name contains a delimiter" \
    || return 1
  [[ "${evidence_path}" != *$'\t'* && "${evidence_path}" != *$'\n'* ]] \
    || scout_contract_error "stage evidence path contains a delimiter" \
    || return 1
  [[ "${command_record}" != *$'\t'* && "${command_record}" != *$'\n'* ]] \
    || scout_contract_error "stage command contains a delimiter" \
    || return 1
  printf '%s\t0\t%s\t%s\n' \
    "${stage_name}" "${evidence_path}" "${command_record}" >>"${journal}"
}

scout_close_log() {
  local approved_root="$1"
  local requested="$2"
  scout_require_regular_file "${approved_root}" "${requested}" || return 1
  chmod 0444 -- "${requested}"
  [[ "$(stat -c '%a' -- "${requested}")" == "444" ]] \
    || scout_contract_error "closed log is not read-only: ${requested}"
}

scout_run_logged() {
  local approved_root="$1"
  local log="$2"
  local tee_bin="$3"
  shift 3
  [[ "${1:-}" == "--" ]] \
    || scout_contract_error "scout_run_logged requires -- before the command" \
    || return 1
  shift
  scout_require_regular_file "${approved_root}" "${log}" || return 1

  set +e
  "$@" 2>&1 | "${tee_bin}" -a "${log}"
  local -a pipeline_status=("${PIPESTATUS[@]}")
  set -e
  local command_status="${pipeline_status[0]}"
  local sink_status="${pipeline_status[1]}"
  scout_close_log "${approved_root}" "${log}" || return 1
  if [[ "${command_status}" -ne 0 ]]; then
    printf 'error: stage command failed with status %s (log sink status %s): %s\n' \
      "${command_status}" "${sink_status}" "${log}" >&2
    return "${command_status}"
  fi
  if [[ "${sink_status}" -ne 0 ]]; then
    printf 'error: log sink failed with status %s after successful stage: %s\n' \
      "${sink_status}" "${log}" >&2
    return "${sink_status}"
  fi
}
