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
  1. immutable HEAD, dirty-source, and HEAD-commit path capture;
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
head_paths_host="${attempt_host}/manifests/head-commit-paths.name-only-z"
head_paths_inner="${attempt_inner}/manifests/head-commit-paths.name-only-z"
source_manifest_host="${attempt_host}/manifests/source.json"
source_manifest_inner="${attempt_inner}/manifests/source.json"
stage_journal_host="${checker_host}/stages.tsv"
scout_capture_fresh_file \
  "${attempt_host}" "${source_status_host}" -- \
  "${GIT_BIN}" -C "${REPO_ROOT}" status --porcelain=v1 -z --untracked-files=all
# Captured on the host, and only here: the lint stage runs against a throwaway
# bare repository whose GIT_DIR has no history, so it cannot ask about HEAD.
# This list is what lint covers when the working tree is clean -- the committed
# state -- where the status capture above is empty and yields nothing to lint.
# -m so a merge HEAD reports paths and --root so a root commit does: measured,
# not assumed -- without -m "git diff-tree -r" prints nothing at all for a merge
# commit, and without --root it prints nothing for a repository's first commit.
scout_capture_fresh_file \
  "${attempt_host}" "${head_paths_host}" -- \
  "${GIT_BIN}" -C "${REPO_ROOT}" diff-tree --root -m --no-commit-id \
    --name-only -r -z HEAD
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
  # Sealing refuses an empty stage log too; this copy is deliberate defence in
  # depth, and it fails here with the stage name, before the journal row is
  # appended, rather than at finalize with only a path.
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
      --status-file "$1" --head-paths-file "$2" --output "$3" \
      --repo-root /workspace/torchtitan \
      --head "$4" --attempt-dir "$5"
  ' scout-a \
    "${source_status_inner}" "${head_paths_inner}" "${source_manifest_inner}" \
    "${HEAD_ID}" "${attempt_inner}"
scout_require_regular_file "${attempt_host}" "${source_manifest_host}"

# The one skip the owning suite is allowed to report. Everything else that
# skips fails its stage. See scout_pytest_guard_program in the runner library
# for why the guard exists and for what it cannot see.
DECLARED_OWNING_SKIP="test_rootfs_bwrap_plan.py:500: host perf is not dynamically linked"

# Guarded independently, with zero declared skips. Before, this stage ran a
# plain "pytest -q" and was covered only because owning_pytest happens to list
# the same file; a rename on either side would have dropped the coverage
# silently.
run_logged focused_pytest checker/focused-pytest.log \
  env TORCHTITAN_ROOTFS_NETWORK=offline \
  "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- \
  bash -lc "$(scout_pytest_guard_program)" scout-a \
    /workspace/torchtitan /project/tmp 0 \
    pytest -q -rs --color=no tests/unit_tests/test_qwen3_formal_scout_a.py

# Zero declared skips, deliberately: this stage is gated by a module-level
# pytest.mark.skipif and exits 0 when it skips. That was previously caught only
# by accident, because a skipped run leaves normalized/scout_a.json absent and
# artifact_sync then fails on a missing file. The guard makes the coupling
# explicit and fails this stage rather than a later one.
run_logged cuda_pytest checker/cuda-pytest.log \
  env TORCHTITAN_ROOTFS_NETWORK=networked CUDA_VISIBLE_DEVICES=0 \
  "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- \
  bash -lc "$(scout_pytest_guard_program)" scout-a \
    /workspace/torchtitan /project/tmp 0 \
    env "QFV_SCOUT_A_OUTPUT_ROOT=${output_inner}" \
      "QFV_SCOUT_A_RUN_ID=${RUN_ID}" \
      "QFV_SCOUT_A_ATTEMPT_ID=${ATTEMPT_ID}" \
      "QFV_SCOUT_SOURCE_MANIFEST=${source_manifest_inner}" \
      "QFV_SCOUT_EXPECTED_HEAD=${HEAD_ID}" \
      QFV_SCOUT_SOURCE_ROOT=/workspace/torchtitan \
    torchrun --nproc-per-node=1 --rdzv-id=qfv-scout-a \
      --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 \
      -m pytest -q -rs --color=no \
      tests/integration_tests/test_qwen3_formal_scout_a.py -s

# The owning suite includes test_formal_toolchain.py, whose TLC-running
# contracts skip without the vendored toolchain. Mount the formal cache so the
# gate executes them instead of recording nine undeclared skips.
run_logged owning_pytest checker/owning-pytest.log \
  env TORCHTITAN_ROOTFS_NETWORK=offline \
  TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST="${FORMAL_CACHE}" \
  "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- \
  bash -lc "$(scout_pytest_guard_program)" scout-a \
    /workspace/torchtitan /project/tmp 1 "${DECLARED_OWNING_SKIP}" \
    pytest -q -rs --color=no \
      tests/unit_tests/test_qwen3_formal_scout_a.py \
      tests/unit_tests/test_qwen3_formal_verifier.py \
      tests/unit_tests/test_generic_accelerator_observability.py \
      tests/unit_tests/test_formal_toolchain.py \
      tests/unit_tests/test_rootfs_bwrap_plan.py \
      tests/unit_tests/test_check_no_pii.py

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
  # The stage body lives in scout_lint_stage_program so both runners share one
  # copy and a test can execute it. Every argument the program needs is passed
  # explicitly; nothing about the sandbox is hardcoded inside it.
  if env TORCHTITAN_ROOTFS_NETWORK=networked \
    "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- \
    bash -lc "$(scout_lint_stage_program)" scout-a \
      scout_a /workspace/torchtitan /project/tmp \
      "qfv-scout-lint-${output_relative//\//_}-${RUN_ID}-${ATTEMPT_ID}" \
      "${source_manifest_inner}" "${source_status_inner}" \
      "${head_paths_inner}" "${HEAD_ID}" "${attempt_inner}"
  then
    :
  else
    local lint_status=$?
    return "${lint_status}"
  fi

  scout_verify_source_identity \
    "${GIT_BIN}" "${REPO_ROOT}" "${HEAD_ID}" "${source_status_host}" A
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
