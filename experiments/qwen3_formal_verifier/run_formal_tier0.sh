#!/usr/bin/env bash
# Tier 0 iteration loop for the Qwen3 formal verifier.
#
# Tier 0 is exactly the checks that read only hand-written sources: the
# abstract TLA+ models and their negatives, the Lean smoke, changed-source
# lint, and the focused formal-toolchain contracts. Nothing here reads a
# generated *Facts module, so nothing here needs a GPU run or a sealed
# attempt bundle.
#
# This is NOT a gate. It deliberately has no --output-root, --run-id or
# --attempt-id, writes nothing under outputs/, and seals no evidence, so a
# tier-0 pass cannot be presented as a Scout A or Scout B gate pass. The
# trace-dependent checks -- the refinement bridge, the fact modules, artifact
# sync, and the real CUDA step -- live only in run_scout_a.sh and
# run_scout_b.sh.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOTFS_ENTRYPOINT="${TORCHTITAN_SCOUT_ROOTFS_ENTRYPOINT:-${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh}"
ROOTFS="${TORCHTITAN_SCOUT_ROOTFS:-${REPO_ROOT}/scripts/rootfs/rootfs}"
FORMAL_CACHE="${TORCHTITAN_FORMAL_CACHE_HOST:-${HOME:?HOME is required}/.cache/torchtitan/formal}"
FORMAL_CHECKS="${TORCHTITAN_TIER0_FORMAL_CHECKS:-${REPO_ROOT}/scripts/run_formal_checks.sh}"
GIT_BIN="${TORCHTITAN_SCOUT_GIT:-git}"
FETCH_MODE="--networked"
RUN_LINT=1
RUN_PYTEST=1

usage() {
  cat <<'EOF'
Usage: experiments/qwen3_formal_verifier/run_formal_tier0.sh [options]

Run the trace-free tier 0 checks on CPU. It is minutes, not seconds: the
Scout B model target alone runs eleven checks over eight communicators.
Still no GPU, no observed trace and no sealed bundle.
  1. the tier0 Bazel formal suite (TLC and Lean smoke, the abstract Scout A
     model plus its relaxed-guard negative, and the abstract Scout B DPxTP
     model with all ten of its companion checks: non-vacuity, the
     order-divergence deadlock, that deadlock's witness shape, the
     operation-mismatch hang, the stream-edge shape plus its
     per-communicator-stream mutant, the unguarded rendezvous, and the four
     liveness checks -- the property under weak fairness, its unfair-spec
     control, its order-divergence lasso, and the refutation of its
     unconditional reading);
  2. changed-source lint inside Insula -- a deliberate subset of the
     pre-commit hooks, skipping the slow or gate-only ones
     (no-commit-to-branch, check-added-large-files, lychee-link-checker),
     plus pyrefly run directly on changed Python;
  3. the focused formal-toolchain pytest contracts.

Options:
  --networked   Allow Bazel to materialize pinned dependencies (default).
  --no-fetch    Reuse the materialized formal cache with networking disabled.
  --skip-lint   Skip changed-source lint.
  --skip-pytest Skip the focused pytest contracts.
  -h, --help    Show this help.

Environment:
  TORCHTITAN_SCOUT_ROOTFS       Explicit bwrap rootfs directory.
  TORCHTITAN_FORMAL_CACHE_HOST  Persistent formal cache outside the checkout.

Tier 0 is an iteration aid, not a gate. It writes no evidence bundle; run
experiments/qwen3_formal_verifier/run_scout_a.sh or run_scout_b.sh for that.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --networked)
      FETCH_MODE="--networked"
      shift
      ;;
    --no-fetch)
      FETCH_MODE="--no-fetch"
      shift
      ;;
    --skip-lint)
      RUN_LINT=0
      shift
      ;;
    --skip-pytest)
      RUN_PYTEST=0
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    # Named explicitly rather than falling into the catch-all: these are the
    # gate's evidence-bundle arguments, and a reader who reaches for them here
    # is expecting sealed evidence that tier 0 does not produce.
    --output-root | --run-id | --attempt-id | --update-artifacts)
      usage >&2
      die "$1 is a gate argument; tier 0 seals no evidence bundle"
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done

[[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]] \
  || die "run tier 0 from outside Insula so every stage gets a fresh sandbox"
[[ -x "${ROOTFS_ENTRYPOINT}" ]] \
  || die "rootfs entrypoint is not executable: ${ROOTFS_ENTRYPOINT}"
[[ -d "${ROOTFS}" ]] || die "rootfs directory does not exist: ${ROOTFS}"
[[ -x "${FORMAL_CHECKS}" ]] \
  || die "formal check wrapper is not executable: ${FORMAL_CHECKS}"
mkdir -p -- "${FORMAL_CACHE}"

printf 'QFV_TIER0 start mode=%s lint=%s pytest=%s\n' \
  "${FETCH_MODE#--}" "${RUN_LINT}" "${RUN_PYTEST}"
printf 'QFV_TIER0 scope=trace-free note=not-a-gate bundle=none\n'

env TORCHTITAN_FORMAL_CACHE_HOST="${FORMAL_CACHE}" \
  TORCHTITAN_FORMAL_ROOTFS="${ROOTFS}" \
  "${FORMAL_CHECKS}" "${FETCH_MODE}" --suite tier0

if ((RUN_LINT)); then
  # Changed relative to HEAD plus untracked files: the same set the gate's
  # lint stage derives from the sealed source manifest, computed here from git
  # directly because tier 0 builds no manifest.
  declare -a changed_files=()
  while IFS= read -r -d '' candidate; do
    [[ -f "${REPO_ROOT}/${candidate}" ]] || continue
    changed_files+=("${candidate}")
  done < <(
    {
      "${GIT_BIN}" -C "${REPO_ROOT}" diff --name-only -z HEAD
      "${GIT_BIN}" -C "${REPO_ROOT}" ls-files \
        --others --exclude-standard -z
    } | sort -z -u
  )
  if ((${#changed_files[@]} == 0)); then
    printf 'QFV_TIER0 lint result=skipped reason=no_changed_files\n'
  else
    env TORCHTITAN_ROOTFS_NETWORK=networked \
      "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- bash -lc '
        set -euo pipefail
        cd /workspace/torchtitan
        declare -a source_files=("$@")
        # pre-commit requires a work tree with a readable git dir, and the
        # git dir of this checkout sits outside the Insula mounts. Same
        # throwaway bare repository the gate lint stage uses, named by mktemp
        # so repeated runs over the persistent /project/tmp do not collide.
        git_dir="$(mktemp -d /project/tmp/qfv-tier0-lint.XXXXXXXX)/git"
        git init --bare "${git_dir}" >/dev/null
        export GIT_DIR="${git_dir}"
        export GIT_WORK_TREE=/workspace/torchtitan
        git add -- "${source_files[@]}"
        # A deliberate subset: the link checker and large-file hooks are slow
        # or network-bound, and no-commit-to-branch is a commit-time concern.
        # The gate still runs the complete set, so tier 0 stays well under the
        # model-checking time without lowering the bar at the commit boundary.
        # The split is
        # pinned by test_tier0_lint_classifies_every_pre_commit_hook, so adding
        # a hook to .pre-commit-config.yaml forces a choice here rather than
        # being skipped silently forever.
        for hook in \
          trailing-whitespace check-ast check-merge-conflict \
          end-of-file-fixer insert-license flake8 ufmt pydoclint \
          codespell check-no-pii
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
      ' tier0-lint "${changed_files[@]}"
    printf 'QFV_TIER0 lint result=success files=%s\n' "${#changed_files[@]}"
  fi
fi

if ((RUN_PYTEST)); then
  # The formal cache is mounted for this stage too, because some of these
  # contracts run the real TLC toolchain over a deliberately degenerate input
  # instead of grepping the runner's text. Without the mount they would skip,
  # which is a guard that checks nothing;
  # test_tier0_runner_mounts_the_formal_cache_for_the_pytest_stage pins it.
  env TORCHTITAN_ROOTFS_NETWORK=offline \
    TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST="${FORMAL_CACHE}" \
    "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- bash -lc \
    'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_formal_toolchain.py'
  printf 'QFV_TIER0 pytest result=success\n'
fi

printf 'QFV_TIER0 result=success mode=%s\n' "${FETCH_MODE#--}"
printf 'QFV_TIER0 reminder=tier 0 proves the models only; it is NOT a gate pass and sealed no bundle\n'
