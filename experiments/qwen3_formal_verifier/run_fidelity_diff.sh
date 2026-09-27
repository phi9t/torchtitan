#!/usr/bin/env bash
# Differential-test ScoutBModel.tla against ScoutBProtocol.lean.
#
# The two files are hand-written models of one protocol, joined only by the
# prose correspondence table in ScoutBProtocol.lean's header. This runner
# generates instances from a fixed seed, evaluates the nine shared predicates
# with BOTH checkers at the same pinned state, and compares them. See
# .scratch/qwen3-formal-verifier/issues/27-tla-lean-fidelity-differential.md.
#
# WHAT A PASSING RUN CLAIMS. That the two models agreed on every shared
# predicate at every probed argument, over the recorded number of instances
# drawn from the recorded seed. Nothing more: the result token carries
# scope=sampled-instances exhaustive=no because raising the instance count
# does not make the comparison exhaustive.
#
# --self-test is the other half of the evidence. It injects each named
# perturbation in turn and requires the comparator to report a disagreement.
# A differential harness that cannot fail says nothing when it passes, so the
# demonstration is part of the runner rather than a one-off by hand.
#
# The host shell only orchestrates. Python, Lean and TLC all run inside the
# bwrap rootfs.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOTFS_ENTRYPOINT="${TORCHTITAN_SCOUT_ROOTFS_ENTRYPOINT:-${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh}"
ROOTFS="${TORCHTITAN_SCOUT_ROOTFS:-${REPO_ROOT}/scripts/rootfs/rootfs}"
FORMAL_CACHE="${TORCHTITAN_FORMAL_CACHE_HOST:-${HOME:?HOME is required}/.cache/torchtitan/formal}"
MODULE="torchtitan.experiments.qwen3_formal_verifier.fidelity_diff"

SEED=42
INSTANCES=192
OUTPUT_ROOT="outputs/qwen3_formal_verifier/fidelity_diff"
SELF_TEST=0
SELF_TEST_INSTANCES=24
EXTRA_ARGS=()

# Every perturbation the harness knows. --self-test requires each one to be
# reported; a silent pass here would mean the comparator had stopped working.
PERTURBATIONS=(
  lean_same_site_negate
  lean_stuck_drop_start_conjunct
  lean_ops_agree_negate
  lean_completed_out_of_domain_negate
  tla_all_done_drop_running
  tla_completed_off_by_one
)

usage() {
  cat <<'EOF'
Usage: experiments/qwen3_formal_verifier/run_fidelity_diff.sh [options]

Options:
  --seed N            Generator seed (default 42).
  --instances N       Number of instances to compare (default 192).
  --output-root DIR   Repo-local work root for generated modules.
  --self-test         Also inject each perturbation and require a report.
  --self-test-instances N
                      Instances per perturbation run (default 24; the
                      stuck-conjunct perturbation needs roughly this many
                      before a startable communicator appears in the draw).
  -h, --help          Show this help.

Environment:
  TORCHTITAN_SCOUT_ROOTFS       Explicit bwrap rootfs directory.
  TORCHTITAN_FORMAL_CACHE_HOST  Persistent formal toolchain cache.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="${2:?--seed requires a value}"
      shift 2
      ;;
    --instances)
      INSTANCES="${2:?--instances requires a value}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?--output-root requires a value}"
      shift 2
      ;;
    --self-test)
      SELF_TEST=1
      shift
      ;;
    --self-test-instances)
      SELF_TEST_INSTANCES="${2:?--self-test-instances requires a value}"
      shift 2
      ;;
    --allow-vacuous)
      EXTRA_ARGS+=(--allow-vacuous)
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

[[ -x "${ROOTFS_ENTRYPOINT}" ]] \
  || die "rootfs entrypoint is not executable: ${ROOTFS_ENTRYPOINT}"
[[ -d "${ROOTFS}" ]] || die "rootfs directory not found: ${ROOTFS}"
mkdir -p -- "${FORMAL_CACHE}"
mkdir -p -- "${REPO_ROOT}/${OUTPUT_ROOT}"

# Every Python, Lean and TLC process runs inside the rootfs; nothing but this
# argument assembly happens on the host.
rootfs_python() {
  TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST="${FORMAL_CACHE}" \
    "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- \
    bash -lc "cd /workspace/torchtitan && python -m ${MODULE} $*"
}

printf 'QFV_FIDELITY_DIFF_STAGE stage=compare seed=%s instances=%s\n' \
  "${SEED}" "${INSTANCES}"
compare_status=0
rootfs_python \
  --seed "${SEED}" \
  --instances "${INSTANCES}" \
  --work-dir "${OUTPUT_ROOT}/compare" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" || compare_status=$?
[[ ${compare_status} -eq 0 ]] \
  || die "differential comparison failed (exit ${compare_status})"

if [[ ${SELF_TEST} -eq 1 ]]; then
  for perturbation in "${PERTURBATIONS[@]}"; do
    printf 'QFV_FIDELITY_DIFF_STAGE stage=self_test perturbation=%s\n' \
      "${perturbation}"
    status=0
    rootfs_python \
      --seed "${SEED}" \
      --instances "${SELF_TEST_INSTANCES}" \
      --work-dir "${OUTPUT_ROOT}/self_test_${perturbation}" \
      --perturb "${perturbation}" || status=$?
    # Exit 1 is the comparator reporting a disagreement, which is the expected
    # outcome here. Exit 0 means it missed the injected fault; exit 2 means the
    # run errored before comparing anything, which is not a demonstration.
    if [[ ${status} -eq 0 ]]; then
      die "perturbation ${perturbation} was NOT detected; the comparator is not working"
    fi
    if [[ ${status} -ne 1 ]]; then
      die "perturbation ${perturbation} errored (exit ${status}) instead of being reported"
    fi
    printf 'QFV_FIDELITY_DIFF_SELF_TEST perturbation=%s result=detected\n' \
      "${perturbation}"
  done
fi

printf 'QFV_FIDELITY_DIFF_RUNNER result=success seed=%s instances=%s self_test=%s\n' \
  "${SEED}" "${INSTANCES}" "${SELF_TEST}"
