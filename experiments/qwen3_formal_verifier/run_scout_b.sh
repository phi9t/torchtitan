#!/usr/bin/env bash
# Run the supported four-rank DPxTP Qwen3 formal-verifier Scout B gate.

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
RUN_ID="qfv-scout-b-seed42"
# Must match the attempt the checked-in ScoutB*Facts fixtures were
# generated under: the attempt ID is hashed into every event ID and so
# into every exported digest, and artifact synchronisation compares
# those digests. A different default makes the documented command fail.
ATTEMPT_ID="four-rank-dp2-tp2-v1"
MASTER_PORT="${TORCHTITAN_SCOUT_B_MASTER_PORT:-29643}"
SYNC_MODE="--check"

usage() {
  cat <<'EOF'
Usage: experiments/qwen3_formal_verifier/run_scout_b.sh [options]

Run the complete Scout B acceptance gate:
  1. immutable HEAD and dirty-source identity capture;
  2. focused pytest contracts;
  3. one real four-rank DP-shard-2/TP-2 Qwen3 Trainer step on GPUs 0-3;
  4. the owning pytest suite;
  5. checked-in artifact sync against the normalized observed trace;
  6. the complete source-bound Scout A regression gate;
  7. a networked formal check and a fresh no-fetch formal check;
  8. changed-source lint plus a final source-identity check;
  9. complete evidence sealing and verification.

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
  || die "run Scout B from outside Insula so every stage gets a fresh sandbox"
[[ -x "${ROOTFS_ENTRYPOINT}" ]] \
  || die "rootfs entrypoint is not executable: ${ROOTFS_ENTRYPOINT}"
[[ -d "${ROOTFS}" ]] || die "rootfs directory does not exist: ${ROOTFS}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die "invalid run id: ${RUN_ID}"
[[ "${ATTEMPT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die "invalid attempt id: ${ATTEMPT_ID}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "0,1,2,3" ]]; then
  die "Scout B requires CUDA_VISIBLE_DEVICES=0,1,2,3"
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || die "invalid master port: ${MASTER_PORT}"

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
    python -m torchtitan.experiments.qwen3_formal_verifier.scout_b source \
      --status-file "$1" --output "$2" --repo-root /workspace/torchtitan \
      --head "$3" --attempt-dir "$4"
  ' scout-b \
    "${source_status_inner}" "${source_manifest_inner}" \
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
  bash -lc "$(scout_pytest_guard_program)" scout-b \
    /workspace/torchtitan /project/tmp 0 \
    pytest -q -rs --color=no tests/unit_tests/test_qwen3_formal_scout_b.py

# Zero declared skips, deliberately: this stage is gated by a module-level
# pytest.mark.skipif and exits 0 when it skips. That was previously caught only
# by accident, because a skipped run leaves normalized/scout_b.json absent and
# artifact_sync then fails on a missing file. The guard makes the coupling
# explicit and fails this stage rather than a later one.
run_logged cuda_pytest checker/cuda-pytest.log \
  env TORCHTITAN_ROOTFS_NETWORK=offline CUDA_VISIBLE_DEVICES=0,1,2,3 \
    TORCH_NCCL_TRACE_BUFFER_SIZE=100000 TORCH_NCCL_ENABLE_TIMING=1 \
  "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- \
  bash -lc "$(scout_pytest_guard_program)" scout-b \
    /workspace/torchtitan /project/tmp 0 \
    env "QFV_SCOUT_B_OUTPUT_ROOT=${output_inner}" \
      "QFV_SCOUT_B_RUN_ID=${RUN_ID}" \
      "QFV_SCOUT_B_ATTEMPT_ID=${ATTEMPT_ID}" \
      "QFV_SCOUT_SOURCE_MANIFEST=${source_manifest_inner}" \
      "QFV_SCOUT_EXPECTED_HEAD=${HEAD_ID}" \
      QFV_SCOUT_SOURCE_ROOT=/workspace/torchtitan \
    torchrun --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
      --nnodes=1 --nproc-per-node=4 \
      -m pytest -q -rs --color=no \
      tests/integration_tests/test_qwen3_formal_scout_b.py -s

run_logged owning_pytest checker/owning-pytest.log \
  env TORCHTITAN_ROOTFS_NETWORK=offline \
  "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- \
  bash -lc "$(scout_pytest_guard_program)" scout-b \
    /workspace/torchtitan /project/tmp 1 "${DECLARED_OWNING_SKIP}" \
    pytest -q -rs --color=no \
      tests/unit_tests/test_qwen3_formal_scout_a.py \
      tests/unit_tests/test_qwen3_formal_scout_b.py \
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
    python -m torchtitan.experiments.qwen3_formal_verifier.scout_b sync \
      --normalized "$1/normalized/scout_b.json" "$2"
    python -m torchtitan.experiments.qwen3_formal_verifier.scout_b sync \
      --normalized "$1/normalized/scout_b.json" --check
  ' scout-b "${attempt_inner}" "${SYNC_MODE}"

run_logged scout_a_regression checker/scout-a-regression.log \
  env CUDA_VISIBLE_DEVICES=0 \
    TORCHTITAN_SCOUT_ROOTFS="${ROOTFS}" \
    TORCHTITAN_FORMAL_CACHE_HOST="${FORMAL_CACHE}" \
  "${REPO_ROOT}/experiments/qwen3_formal_verifier/run_scout_a.sh" \
    --output-root "${output_relative}/scout-a-regression" \
    --run-id "qfv-scout-a-seed42-fix1" \
    --attempt-id "single-rank-cuda-fix1"

run_logged formal_networked checker/formal-networked.log \
  env TORCHTITAN_FORMAL_CACHE_HOST="${FORMAL_CACHE}" \
    TORCHTITAN_FORMAL_ROOTFS="${ROOTFS}" \
  "${REPO_ROOT}/scripts/run_formal_checks.sh" --networked --suite scout-b

run_logged formal_no_fetch checker/formal-no-fetch.log \
  env TORCHTITAN_FORMAL_CACHE_HOST="${FORMAL_CACHE}" \
    TORCHTITAN_FORMAL_ROOTFS="${ROOTFS}" \
  "${REPO_ROOT}/scripts/run_formal_checks.sh" --no-fetch --suite scout-b

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
      # mktemp, not a derived name: /project/tmp persists across runs, so a
      # name derived from the stage identity collides when the same output
      # root is reused, and the fail-closed check then aborts the stage before
      # it writes anything -- an empty log rather than a diagnosis.
      lint_paths_file="$(mktemp /project/tmp/lint-paths.XXXXXXXX)"
      # --status-file/--repo-root/--head/--attempt-dir make lint-time manifest
      # validation identical to seal-time validation: without them the path
      # list is shape-validated only, and the process-entry roster comparison
      # -- the one check that can notice an ABSENT entry -- runs at finalize.
      python -m torchtitan.experiments.qwen3_formal_verifier.scout_b \
        lint-paths --manifest "${source_manifest}" \
        --repo-root /workspace/torchtitan --status-file "$3" \
        --head "$4" --attempt-dir "$5" >"${lint_paths_file}"
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
    ' scout-b "${source_manifest_inner}" \
      "qfv-scout-lint-${output_relative//\//_}-${RUN_ID}-${ATTEMPT_ID}" \
      "${source_status_inner}" "${HEAD_ID}" "${attempt_inner}"
  then
    :
  else
    local lint_status=$?
    return "${lint_status}"
  fi

  scout_verify_source_identity \
    "${GIT_BIN}" "${REPO_ROOT}" "${HEAD_ID}" "${source_status_host}" B
}

run_logged lint checker/lint.log run_lint_and_verify_source

scout_close_log "${attempt_host}" "${stage_journal_host}"

env TORCHTITAN_ROOTFS_NETWORK=offline \
  "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- bash -lc '
    set -euo pipefail
    cd /workspace/torchtitan
    python -m torchtitan.experiments.qwen3_formal_verifier.scout_b finalize \
      --attempt-dir "$1" --expected-head "$2" \
      --source-root /workspace/torchtitan
    python -m torchtitan.experiments.qwen3_formal_verifier.scout_b verify \
      --attempt-dir "$1" --expected-head "$2" \
      --source-root /workspace/torchtitan
  ' scout-b "${attempt_inner}" "${HEAD_ID}"

printf 'SCOUT_B_GATE result=success attempt_dir=%s\n' "${attempt_host}"
