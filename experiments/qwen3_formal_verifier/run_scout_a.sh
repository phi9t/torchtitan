#!/usr/bin/env bash
# Run the supported single-rank Qwen3 formal-verifier Scout A gate.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER_LIBRARY="${REPO_ROOT}/experiments/qwen3_formal_verifier/scout_a_runner_lib.sh"
# shellcheck source=experiments/qwen3_formal_verifier/scout_a_runner_lib.sh
source "${RUNNER_LIBRARY}"
ROOTFS_ENTRYPOINT="${TORCHTITAN_SCOUT_ROOTFS_ENTRYPOINT:-${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh}"
ROOTFS="${TORCHTITAN_SCOUT_ROOTFS:-${REPO_ROOT}/scripts/rootfs/rootfs}"
FORMAL_CACHE="${TORCHTITAN_FORMAL_CACHE_HOST:-${HOME:?HOME is required}/.cache/torchtitan/formal}"
TEE_BIN="${TORCHTITAN_SCOUT_TEE:-tee}"
GIT_BIN="${TORCHTITAN_SCOUT_GIT:-git}"
OUTPUT_ROOT="outputs/qwen3_formal_verifier/evidence"
RUN_ID="qfv-scout-a-seed42-fix1"
ATTEMPT_ID="single-rank-cuda-fix1"
SYNC_MODE="--check"

usage() {
  cat <<'EOF'
Usage: experiments/qwen3_formal_verifier/run_scout_a.sh [options]

Run the complete Scout A acceptance gate:
  1. immutable HEAD and dirty-source identity capture;
  2. focused pytest contracts;
  3. one real CUDA Qwen3 optimizer step through the core Trainer on GPU 0;
  4. the owning pytest suite;
  5. checked-in artifact sync against the normalized observed trace;
  6. a networked formal check and a fresh no-fetch formal check;
  7. changed-source lint plus a final source-identity check;
  8. complete evidence sealing and verification.

Options:
  --output-root DIR       Repo-local evidence root.
  --run-id ID             Stable run identity.
  --attempt-id ID         Stable attempt identity.
  --update-artifacts      Write the compact checked-in fixture and fact modules
                          before checking them. Use only when intentionally
                          accepting a newly observed canonical trace.
  -h, --help              Show this help.

Environment:
  TORCHTITAN_SCOUT_ROOTFS       Explicit bwrap rootfs directory.
  TORCHTITAN_FORMAL_CACHE_HOST Persistent formal cache outside the checkout.

The supported command runs from the host. Python, PyTorch, CUDA, pytest,
artifact generation, and formal checker processes all execute inside Insula.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="${2:?--output-root requires a value}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:?--run-id requires a value}"
      shift 2
      ;;
    --attempt-id)
      ATTEMPT_ID="${2:?--attempt-id requires a value}"
      shift 2
      ;;
    --update-artifacts)
      SYNC_MODE="--write"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done

[[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]] \
  || die "run Scout A from outside Insula so every stage gets a fresh sandbox"
[[ -x "${ROOTFS_ENTRYPOINT}" ]] \
  || die "rootfs entrypoint is not executable: ${ROOTFS_ENTRYPOINT}"
[[ -d "${ROOTFS}" ]] || die "rootfs directory does not exist: ${ROOTFS}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die "invalid run id: ${RUN_ID}"
[[ "${ATTEMPT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die "invalid attempt id: ${ATTEMPT_ID}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "0" ]]; then
  die "Scout A requires CUDA_VISIBLE_DEVICES=0"
fi
export CUDA_VISIBLE_DEVICES=0

case "${OUTPUT_ROOT}" in
  /*)
    case "${OUTPUT_ROOT}" in
      "${REPO_ROOT}/"*) output_relative="${OUTPUT_ROOT#"${REPO_ROOT}/"}" ;;
      *) die "evidence output root must stay inside the checkout: ${OUTPUT_ROOT}" ;;
    esac
    ;;
  *) output_relative="${OUTPUT_ROOT}" ;;
esac
repo_canonical="$(cd -P -- "${REPO_ROOT}" && pwd)"
output_host="$(scout_ensure_directory_tree "${repo_canonical}" "${output_relative}")" \
  || exit 1
output_relative="${output_host#"${repo_canonical}/"}"
output_inner="/workspace/torchtitan/${output_relative}"
run_host="$(scout_ensure_directory_tree "${output_host}" "${RUN_ID}")" \
  || exit 1
attempt_candidate="${run_host}/${ATTEMPT_ID}"
[[ ! -L "${attempt_candidate}" ]] \
  || die "fresh attempt path must not be a symlink: ${attempt_candidate}"
[[ ! -e "${attempt_candidate}" ]] \
  || die "fresh attempt path already exists: ${attempt_candidate}"
mkdir -- "${attempt_candidate}"
attempt_host="$(scout_ensure_directory_tree "${run_host}" "${ATTEMPT_ID}")" \
  || exit 1
attempt_inner="${output_inner}/${RUN_ID}/${ATTEMPT_ID}"
checker_host="$(scout_ensure_directory_tree "${attempt_host}" checker)" \
  || exit 1
checker_inner="${attempt_inner}/checker"
scout_ensure_directory_tree "${attempt_host}" manifests >/dev/null || exit 1
mkdir -p -- "${FORMAL_CACHE}"

HEAD_ID="$("${GIT_BIN}" -C "${REPO_ROOT}" rev-parse --verify HEAD)"
[[ "${HEAD_ID}" =~ ^[0-9a-f]{40,64}$ ]] \
  || die "invalid Git HEAD identity: ${HEAD_ID}"
source_status_host="${attempt_host}/manifests/source-status.porcelain-v1-z"
source_status_inner="${attempt_inner}/manifests/source-status.porcelain-v1-z"
source_manifest_host="${attempt_host}/manifests/source.json"
source_manifest_inner="${attempt_inner}/manifests/source.json"
stage_journal_host="${checker_host}/stages.tsv"
scout_capture_fresh_file \
  "${attempt_host}" "${source_status_host}" -- \
  "${GIT_BIN}" -C "${REPO_ROOT}" status --porcelain=v1 -z --untracked-files=all
scout_create_fresh_log "${attempt_host}" "${stage_journal_host}"

run_logged() {
  local stage_name="$1"
  local relative_log="$2"
  shift 2
  local command_record
  printf -v command_record '%q ' "$@"
  command_record="${command_record% }"
  local log="${attempt_host}/${relative_log}"
  scout_create_fresh_log "${attempt_host}" "${log}" || return 1
  scout_run_logged "${attempt_host}" "${log}" "${TEE_BIN}" -- "$@" \
    || return $?
  # A stage that exited zero but captured nothing is not evidence of anything.
  # Sealing only checks that the log is a read-only regular file, so without
  # this an empty sink would satisfy the bundle while proving no check ran.
  if [[ ! -s "${log}" ]]; then
    printf 'stage %s produced an empty log: %s\n' "${stage_name}" "${log}" >&2
    return 1
  fi
  scout_append_stage \
    "${attempt_host}" "${stage_journal_host}" \
    "${stage_name}" "${relative_log}" "${command_record}"
}

run_logged source_manifest checker/source-manifest.log \
  env TORCHTITAN_ROOTFS_NETWORK=offline \
  "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- bash -lc '
    set -euo pipefail
    cd /workspace/torchtitan
    python -m torchtitan.experiments.qwen3_formal_verifier.scout_a source \
      --status-file "$1" --output "$2" --repo-root /workspace/torchtitan \
      --head "$3" --attempt-dir "$4"
  ' scout-a \
    "${source_status_inner}" "${source_manifest_inner}" \
    "${HEAD_ID}" "${attempt_inner}"
scout_require_regular_file "${attempt_host}" "${source_manifest_host}"

run_logged focused_pytest checker/focused-pytest.log \
  env TORCHTITAN_ROOTFS_NETWORK=offline \
  "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- bash -lc \
  'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_qwen3_formal_scout_a.py'

run_logged cuda_pytest checker/cuda-pytest.log \
  env TORCHTITAN_ROOTFS_NETWORK=networked CUDA_VISIBLE_DEVICES=0 \
  "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- bash -lc '
    set -euo pipefail
    cd /workspace/torchtitan
    export QFV_SCOUT_A_OUTPUT_ROOT="$1"
    export QFV_SCOUT_A_RUN_ID="$2"
    export QFV_SCOUT_A_ATTEMPT_ID="$3"
    export QFV_SCOUT_SOURCE_MANIFEST="$4"
    export QFV_SCOUT_EXPECTED_HEAD="$5"
    export QFV_SCOUT_SOURCE_ROOT=/workspace/torchtitan
    exec torchrun --nproc-per-node=1 --rdzv-id=qfv-scout-a \
      --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 \
      -m pytest -q tests/integration_tests/test_qwen3_formal_scout_a.py -s
  ' scout-a \
    "${output_inner}" "${RUN_ID}" "${ATTEMPT_ID}" \
    "${source_manifest_inner}" "${HEAD_ID}"

run_logged owning_pytest checker/owning-pytest.log \
  env TORCHTITAN_ROOTFS_NETWORK=offline \
  "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- bash -lc '
    set -euo pipefail
    cd /workspace/torchtitan
    # -rs puts skip reasons in the sealed log; the guard below then
    # refuses any skip that is not declared, so a check silently
    # vanishing can never be sealed as a passing stage.
    pytest -q -rs --color=no \
      tests/unit_tests/test_qwen3_formal_scout_a.py \
      tests/unit_tests/test_qwen3_formal_verifier.py \
      tests/unit_tests/test_generic_accelerator_observability.py \
      tests/unit_tests/test_formal_toolchain.py \
      tests/unit_tests/test_rootfs_bwrap_plan.py \
      tests/unit_tests/test_check_no_pii.py \
      | tee /project/tmp/owning-pytest.out
    # Every skip must be declared. An undeclared skip means a check silently
    # stopped running, and the spec forbids a skipped check from satisfying an
    # acceptance criterion, so it fails the stage instead of sealing.
    [[ -s /project/tmp/owning-pytest.out ]]
    skip_lines="$(sed -e "s/\x1b\[[0-9;]*m//g" /project/tmp/owning-pytest.out \
      | grep -E "^SKIPPED")" || skip_lines=""
    undeclared_skips="$(printf "%s\n" "${skip_lines}" \
      | grep -E "^SKIPPED" \
      | grep -Fv "test_rootfs_bwrap_plan.py:500: host perf is not dynamically linked" \
      || true)"
    if [[ -n "${undeclared_skips}" ]]; then
      echo "undeclared skip in the owning suite:" >&2
      echo "${undeclared_skips}" >&2
      exit 1
    fi
  '

run_logged artifact_sync checker/artifact-sync.log \
  env TORCHTITAN_ROOTFS_NETWORK=offline \
  "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- bash -lc '
    set -euo pipefail
    cd /workspace/torchtitan
    python -m torchtitan.experiments.qwen3_formal_verifier.scout_a sync \
      --normalized "$1/normalized/scout_a.json" "$2"
    python -m torchtitan.experiments.qwen3_formal_verifier.scout_a sync \
      --normalized "$1/normalized/scout_a.json" --check
  ' scout-a "${attempt_inner}" "${SYNC_MODE}"

run_logged formal_networked checker/formal-networked.log \
  env TORCHTITAN_FORMAL_CACHE_HOST="${FORMAL_CACHE}" \
    TORCHTITAN_FORMAL_ROOTFS="${ROOTFS}" \
  "${REPO_ROOT}/scripts/run_formal_checks.sh" --networked --suite scout-a

run_logged formal_no_fetch checker/formal-no-fetch.log \
  env TORCHTITAN_FORMAL_CACHE_HOST="${FORMAL_CACHE}" \
    TORCHTITAN_FORMAL_ROOTFS="${ROOTFS}" \
  "${REPO_ROOT}/scripts/run_formal_checks.sh" --no-fetch --suite scout-a

run_lint_and_verify_source() {
  if env TORCHTITAN_ROOTFS_NETWORK=networked \
    "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- bash -lc '
      set -euo pipefail
      cd /workspace/torchtitan
      source_manifest="$1"
      lint_identity="$2"
      git_dir="/project/tmp/${lint_identity}.git"
      [[ ! -e "${git_dir}" ]]
      git init --bare "${git_dir}" >/dev/null
      export GIT_DIR="${git_dir}"
      export GIT_WORK_TREE=/workspace/torchtitan
      declare -a source_files=()
      # Write to a file first: consuming the producer through process
      # substitution discards its exit status, so a producer that emitted some
      # paths and then crashed would lint a partial set and still seal as a
      # successful stage.
      lint_paths_file="/project/tmp/${lint_identity}.paths"
      [[ ! -e "${lint_paths_file}" ]]
      python -m torchtitan.experiments.qwen3_formal_verifier.scout_a \
        lint-paths --manifest "${source_manifest}" >"${lint_paths_file}"
      while IFS= read -r -d "" source_file; do
        source_files+=("${source_file}")
      done <"${lint_paths_file}"
      ((${#source_files[@]} > 0))
      git add -- "${source_files[@]}"
      for hook in \
        trailing-whitespace check-ast check-merge-conflict \
        no-commit-to-branch check-added-large-files end-of-file-fixer \
        insert-license flake8 ufmt pydoclint codespell \
        lychee-link-checker check-no-pii
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
        pyrefly check --remove-unused-ignores --summarize-errors \
          "${python_files[@]}"
      fi
      if ((${#shell_files[@]} > 0)); then
        for shell_file in "${shell_files[@]}"; do
          bash -n "${shell_file}"
        done
      fi
    ' scout-a "${source_manifest_inner}" \
      "qfv-scout-lint-${output_relative//\//_}-${RUN_ID}-${ATTEMPT_ID}"
  then
    :
  else
    local lint_status=$?
    return "${lint_status}"
  fi

  local current_head current_status
  current_head="$("${GIT_BIN}" -C "${REPO_ROOT}" rev-parse --verify HEAD)" \
    || return 1
  if [[ "${current_head}" != "${HEAD_ID}" ]]; then
    printf 'error: source HEAD changed during gate: %s -> %s\n' \
      "${HEAD_ID}" "${current_head}" >&2
    return 1
  fi
  current_status="$(mktemp "${TMPDIR:-/tmp}/qfv-scout-status.XXXXXX")" \
    || return 1
  if ! "${GIT_BIN}" -C "${REPO_ROOT}" status \
    --porcelain=v1 -z --untracked-files=all >"${current_status}"; then
    rm -f -- "${current_status}"
    return 1
  fi
  if ! cmp --silent "${source_status_host}" "${current_status}"; then
    rm -f -- "${current_status}"
    printf 'error: dirty source identity changed during Scout A gate\n' >&2
    return 1
  fi
  rm -f -- "${current_status}"
  "${GIT_BIN}" -C "${REPO_ROOT}" diff --check
  printf 'SCOUT_A_SOURCE_RECHECK result=success head=%s\n' "${HEAD_ID}"
}

run_logged lint checker/lint.log run_lint_and_verify_source

scout_close_log "${attempt_host}" "${stage_journal_host}"

env TORCHTITAN_ROOTFS_NETWORK=offline \
  "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- bash -lc '
    set -euo pipefail
    cd /workspace/torchtitan
    python -m torchtitan.experiments.qwen3_formal_verifier.scout_a finalize \
      --attempt-dir "$1" --expected-head "$2" \
      --source-root /workspace/torchtitan
    python -m torchtitan.experiments.qwen3_formal_verifier.scout_a verify \
      --attempt-dir "$1" --expected-head "$2" \
      --source-root /workspace/torchtitan
  ' scout-a "${attempt_inner}" "${HEAD_ID}"

printf 'SCOUT_A_GATE result=success attempt_dir=%s\n' "${attempt_host}"
