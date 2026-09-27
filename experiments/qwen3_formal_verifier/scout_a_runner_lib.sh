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

# Emit the bash program every pytest stage runs inside the rootfs.
#
# Why one shared program instead of an inline guard per stage: the guard was
# first pasted into the owning stage only, so the focused and CUDA pytest
# stages ran a plain "pytest -q" and a skip there passed unnoticed. One copy,
# used by every pytest stage, is the only shape that cannot drift.
#
# Why "-rs", "--color=no" and the ANSI strip are all required: pytest lists
# skip reasons only under "-rs", and the first version of this guard ran
# grep -E "^SKIPPED" over COLOURED output, where it matched ZERO lines,
# because pytest emits "\x1b[33mSKIPPED". The gate then passed partly
# BECAUSE the guard did nothing, after having been "verified in both
# directions" against hand-written plain text. Both "--color=no" and the
# strip stay, because the flag check only proves the flag is present: a later
# "--color=yes" satisfies it and turns colour back on, so the strip is the
# defence that actually holds in that case.
#
# What the guard does NOT catch, stated because its name overpromises: it
# matches SKIPPED lines only, so an xfail and a deleted test are invisible to
# it -- both verified, not assumed. A collection-time importorskip IS caught,
# because pytest reports it as a SKIPPED line. The guard proves that no
# undeclared skip happened, not that any particular check ran. Callers needing
# the stronger property must bind the stage to an artifact the tests must
# produce.
#
# Program arguments:
#   $1   working directory for the pytest command
#   $2   writable scratch directory for the captured output
#   $3   number of declared-skip substrings that follow
#   ...  that many declared-skip lines, each matched whole with grep -Fx
#   ...  the pytest command, which must carry both -rs and --color=no
scout_pytest_guard_program() {
  cat <<'PROGRAM'
set -euo pipefail
workdir="$1"
scratch_dir="$2"
declared_count="$3"
shift 3
[[ "${declared_count}" =~ ^[0-9]+$ ]] || {
  printf 'error: declared-skip count must be a number: %s\n' \
    "${declared_count}" >&2
  exit 1
}
declared_skips=()
while ((declared_count > 0)); do
  (($# > 0)) || {
    printf 'error: fewer declared-skip substrings than declared\n' >&2
    exit 1
  }
  declared_skips+=("$1")
  shift
  declared_count=$((declared_count - 1))
done
(($# > 0)) || {
  printf 'error: pytest stage was given no command\n' >&2
  exit 1
}
saw_short_summary=0
saw_color_off=0
for argument in "$@"; do
  [[ "${argument}" != "-rs" ]] || saw_short_summary=1
  [[ "${argument}" != "--color=no" ]] || saw_color_off=1
done
((saw_short_summary == 1)) || {
  printf 'error: pytest stage command must pass -rs so skips are listed\n' >&2
  exit 1
}
((saw_color_off == 1)) || {
  printf 'error: pytest stage command must pass --color=no\n' >&2
  exit 1
}
cd "${workdir}"
captured="$(mktemp "${scratch_dir}/pytest-stage.XXXXXXXX")"
trap 'rm -f -- "${captured}"' EXIT
"$@" 2>&1 | tee "${captured}"
[[ -s "${captured}" ]] || {
  printf 'error: pytest stage captured no output\n' >&2
  exit 1
}
# --color=no should already have removed the escapes. Strip them anyway, so a
# colour default that survives the flag cannot silently disable the match.
skip_lines="$(sed -e 's/\x1b\[[0-9;]*m//g' "${captured}" \
  | grep -E '^SKIPPED' || true)"
# A declared skip must match the line's SHAPE and its whole
# file:line:reason tail, not merely appear somewhere in it. An unanchored
# match allowed any skip whose reason happened to contain the declared text,
# in any file. Only pytest's "[N]" occurrence count is left free, since it
# varies with how many tests share one reason.
undeclared_skips=""
while IFS= read -r skip_line; do
  [[ -n "${skip_line}" ]] || continue
  skip_declared=0
  for declared_skip in ${declared_skips[@]+"${declared_skips[@]}"}; do
    if [[ "${skip_line}" == "SKIPPED ["*"] "*"${declared_skip}" ]]; then
      skip_declared=1
      break
    fi
  done
  ((skip_declared == 1)) || undeclared_skips+="${skip_line}"$'\n'
done <<<"${skip_lines}"
undeclared_skips="$(printf '%s' "${undeclared_skips}" \
  | grep -E '^SKIPPED' || true)"
if [[ -n "${undeclared_skips}" ]]; then
  printf 'error: undeclared skip in a pytest stage:\n%s\n' \
    "${undeclared_skips}" >&2
  exit 1
fi
PROGRAM
}

# Re-verify that the source tree still matches the identity captured at the
# start of the gate, then print the stage's success marker.
#
# Every command is checked explicitly instead of relying on errexit:
# scout_run_logged does "set +e" before invoking a stage function and errexit
# is not function-local in bash, so a stage body runs with errexit off. An
# unguarded command here -- "git diff --check" was exactly that -- lets
# "result=success" print after a failure, which is worse than no check.
#
# Arguments: <git-binary> <repo-root> <expected-head> <baseline-status-file>
# <scout-label>, where the label is the bare scout letter, e.g. A or B.
# Report which git-status entries appeared or disappeared between two
# porcelain=v1 -z snapshots.
#
# Why this exists: the source recheck used to fail with only "dirty source
# identity changed", naming no path. The cause is almost always an edit made
# while the gate ran -- including an edit to a file that has nothing to do with
# the run, because the snapshot covers the whole tree. Without the paths the
# failure reads as a checker bug rather than as "you touched the tree", which
# cost a 20-minute GPU run to rediscover.
#
# This is diagnostic only. It must never change the caller's outcome, so every
# failure inside it returns 0 and the caller still fails on its own finding.
scout_report_status_drift() {
  local baseline="$1"
  local current="$2"
  local baseline_sorted current_sorted
  baseline_sorted="$(mktemp "${TMPDIR:-/tmp}/qfv-drift-base.XXXXXX")" || return 0
  current_sorted="$(mktemp "${TMPDIR:-/tmp}/qfv-drift-curr.XXXXXX")" || {
    rm -f -- "${baseline_sorted}"
    return 0
  }
  # porcelain -z separates entries with NUL; one entry per line is what comm
  # needs. Rename entries occupy two NUL fields, which surface as two lines
  # here -- acceptable for a diagnostic, and both lines name a real path.
  tr '\0' '\n' <"${baseline}" | LC_ALL=C sort >"${baseline_sorted}"
  tr '\0' '\n' <"${current}" | LC_ALL=C sort >"${current_sorted}"
  local entry
  while IFS= read -r entry; do
    [[ -n "${entry}" ]] || continue
    printf '  appeared during the run: %s\n' "${entry}" >&2
  done < <(LC_ALL=C comm -13 "${baseline_sorted}" "${current_sorted}")
  while IFS= read -r entry; do
    [[ -n "${entry}" ]] || continue
    printf '  disappeared during the run: %s\n' "${entry}" >&2
  done < <(LC_ALL=C comm -23 "${baseline_sorted}" "${current_sorted}")
  printf '  cause: some file changed while the gate ran. The snapshot covers\n' >&2
  printf '         the whole tree, so editing any tracked or untracked file --\n' >&2
  printf '         including notes and tickets -- moves it and refuses the run.\n' >&2
  printf '         Re-run without touching the tree.\n' >&2
  rm -f -- "${baseline_sorted}" "${current_sorted}"
}

scout_verify_source_identity() {
  local git_bin="$1"
  local repo_root="$2"
  local expected_head="$3"
  local baseline_status="$4"
  local scout_label="$5"
  [[ "${scout_label}" =~ ^[A-Z][A-Z0-9_]*$ ]] \
    || scout_contract_error "invalid scout label: ${scout_label}" \
    || return 1
  local current_head current_status
  current_head="$("${git_bin}" -C "${repo_root}" rev-parse --verify HEAD)" \
    || return 1
  if [[ "${current_head}" != "${expected_head}" ]]; then
    printf 'error: source HEAD changed during gate: %s -> %s\n' \
      "${expected_head}" "${current_head}" >&2
    return 1
  fi
  current_status="$(mktemp "${TMPDIR:-/tmp}/qfv-scout-status.XXXXXX")" \
    || return 1
  if ! "${git_bin}" -C "${repo_root}" status \
    --porcelain=v1 -z --untracked-files=all >"${current_status}"; then
    rm -f -- "${current_status}"
    return 1
  fi
  if ! cmp --silent "${baseline_status}" "${current_status}"; then
    printf 'error: dirty source identity changed during Scout %s gate\n' \
      "${scout_label}" >&2
    # Diff before deleting the snapshot -- the paths are the whole diagnosis.
    scout_report_status_drift "${baseline_status}" "${current_status}"
    rm -f -- "${current_status}"
    return 1
  fi
  rm -f -- "${current_status}"
  if ! "${git_bin}" -C "${repo_root}" diff --check; then
    printf 'error: dirty source diff fails git diff --check\n' >&2
    return 1
  fi
  printf 'SCOUT_%s_SOURCE_RECHECK result=success head=%s\n' \
    "${scout_label}" "${expected_head}"
}

# Emit the bash program the lint stage runs inside the rootfs.
#
# Why one shared program instead of a copy in each runner: the two copies were
# identical apart from the module name, and the clean-tree defect this replaces
# lived in both of them. One copy is also the only shape a test can execute --
# and executing it is the point, because a test that greps a runner for a
# string is how the defect survived.
#
# The repository root is a parameter rather than a hardcoded
# /workspace/torchtitan so the stage can be run against a throwaway tree.
#
# Program arguments:
#   $1  scout module name under torchtitan.experiments.qwen3_formal_verifier
#   $2  repository root: working directory, GIT_WORK_TREE, and --repo-root
#   $3  writable scratch directory for the bare repository and the path list
#   $4  lint identity, which names the throwaway bare repository
#   $5  source manifest path
#   $6  host-captured Git status file
#   $7  host-captured HEAD-commit path list
#   $8  expected Git HEAD
#   $9  attempt directory
scout_lint_stage_program() {
  cat <<'PROGRAM'
set -euo pipefail
scout_module="$1"
repo_root="$2"
scratch_dir="$3"
lint_identity="$4"
source_manifest="$5"
status_file="$6"
head_paths_file="$7"
head_id="$8"
attempt_dir="$9"
cd "${repo_root}"
git_dir="${scratch_dir}/${lint_identity}.git"
[[ ! -e "${git_dir}" ]]
git init --bare "${git_dir}" >/dev/null
export GIT_DIR="${git_dir}"
export GIT_WORK_TREE="${repo_root}"
declare -a source_files=()
# Write to a file first: consuming the producer through process substitution
# discards its exit status, so a producer that emitted some paths and then
# crashed would lint a partial set and still seal as a successful stage.
# mktemp, not a derived name: the scratch directory persists across runs, so a
# name derived from the stage identity collides when the same output root is
# reused, and the fail-closed check then aborts the stage before it writes
# anything -- an empty log rather than a diagnosis.
lint_paths_file="$(mktemp "${scratch_dir}/lint-paths.XXXXXXXX")"
# Removed when this stage exits, including on failure. The scratch directory
# persists between runs, so without this every lint stage of every run leaves
# one more file behind -- the accumulation that produced the original collision
# in the first place.
trap 'rm -f -- "${lint_paths_file}"' EXIT
# --status-file/--head-paths-file/--repo-root/--head/--attempt-dir make
# lint-time manifest validation identical to seal-time validation: without them
# the path list is shape-validated only, and the process-entry roster
# comparison -- the one check that can notice an ABSENT entry -- runs at
# finalize. --head-paths-file additionally decides what lint covers when the
# working tree is clean, where the dirty status yields nothing at all.
python -m "torchtitan.experiments.qwen3_formal_verifier.${scout_module}" \
  lint-paths --manifest "${source_manifest}" \
  --repo-root "${repo_root}" --status-file "${status_file}" \
  --head-paths-file "${head_paths_file}" \
  --head "${head_id}" --attempt-dir "${attempt_dir}" >"${lint_paths_file}"
while IFS= read -r -d "" source_file; do
  source_files+=("${source_file}")
done <"${lint_paths_file}"
# Retained as defence in depth even though lint-paths now fails with a named
# error when it has nothing to yield: this is the assertion that turned a clean
# tree into a ten-line log with no diagnosis, so it says what it means now.
((${#source_files[@]} > 0)) || {
  printf 'error: lint stage derived no source paths from %s\n' \
    "${source_manifest}" >&2
  exit 1
}
git add -- "${source_files[@]}"
for hook in \
  trailing-whitespace check-ast check-merge-conflict \
  no-commit-to-branch check-added-large-files end-of-file-fixer \
  insert-license flake8 ufmt pydoclint codespell \
  lychee-link-checker check-no-pii check-line-width
do
  pre-commit run "${hook}" --files "${source_files[@]}"
done
declare -a python_files=()
declare -a shell_files=()
for source_file in "${source_files[@]}"; do
  case "${source_file}" in
    *.py) python_files+=("${source_file}") ;;
    *.sh) shell_files+=("${source_file}") ;;
  esac
done
if ((${#python_files[@]} > 0)); then
  pyrefly check --remove-unused-ignores --summarize-errors "${python_files[@]}"
fi
if ((${#shell_files[@]} > 0)); then
  for shell_file in "${shell_files[@]}"; do
    bash -n "${shell_file}"
  done
fi
printf 'SCOUT_LINT_STAGE result=success num_paths=%s\n' "${#source_files[@]}"
PROGRAM
}
