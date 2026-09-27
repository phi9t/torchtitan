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
RUN_MUTATIONS=1
RUN_FIDELITY=1

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
  3. the focused formal-toolchain pytest contracts;
  4. the mutation manifest: every (target, textual mutation, expected failing
     test) triple in tests/mutations/manifest.toml must FAIL its named test on
     a mutated copy of the tree and PASS it unmutated. Fails closed when
     killed < total, and runs the runner's own self-tests first, because a
     mutation runner with an inverted condition would report killed=N/N
     forever and launder confidence rather than add any.

Options:
  --networked      Allow Bazel to materialize pinned dependencies (default).
  --no-fetch       Reuse the materialized formal cache with networking off.
  --skip-lint      Skip changed-source lint.
  --skip-pytest    Skip the focused pytest contracts.
  --skip-mutations Skip the mutation manifest.
  --skip-fidelity  Skip the TLA+/Lean fidelity differential.
  -h, --help       Show this help.

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
    --skip-mutations)
      RUN_MUTATIONS=0
      shift
      ;;
    --skip-fidelity)
      RUN_FIDELITY=0
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

printf 'QFV_TIER0 start mode=%s lint=%s pytest=%s mutations=%s fidelity=%s\n' \
  "${FETCH_MODE#--}" "${RUN_LINT}" "${RUN_PYTEST}" "${RUN_MUTATIONS}" \
  "${RUN_FIDELITY}"
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
          codespell check-no-pii check-line-width
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

if ((RUN_MUTATIONS)); then
  # The mutation manifest. Five guards in this project were written in a shape
  # that could not fail, and every one was found by mutating the subject,
  # re-running, and noticing the test still passed. This stage makes that
  # standing: each manifest entry must FAIL its named test on a mutated copy
  # of the tree and PASS it unmutated.
  #
  # The stage prints its own token instead of letting this script print one
  # after it, because the counts ARE the decision. A host-side printf would
  # have to carry them from somewhere other than the measurement, and an
  # aggregate result=success with no counts is exactly the shape this whole
  # phase exists to prevent. The scratch tree lives under /project/tmp, like
  # the lint stage's throwaway git dir, so tier 0 still writes nothing under
  # outputs/.
  #
  # Same offline network and formal-cache mount as the pytest stage: the named
  # tests include ones that run the real TLC and Lean toolchains, and without
  # the mount they would skip -- which the runner reports as a failed control
  # rather than passing over.
  env TORCHTITAN_ROOTFS_NETWORK=offline \
    TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST="${FORMAL_CACHE}" \
    "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- bash -lc '
      set -euo pipefail
      workdir="$1"
      scratch_parent="$2"
      cd "${workdir}"
      # The runner is itself a guard, so its polarity is checked before it is
      # believed. Its self-tests include a synthetic entry whose mutation
      # deliberately does NOT kill its named test, and require the runner to
      # report that as a failure; without them an inverted condition or a
      # swallowed exit code would report killed=N/N forever.
      python -m pytest -q -rs --color=no -p no:cacheprovider \
        tests/unit_tests/test_mutation_runner.py
      # Quote-free on purpose: this program is a single-quoted string here, and
      # test_tier0_mutations_stage_fails_closed_on_a_surviving_mutation
      # extracts and RUNS it, which an embedded single quote would break.
      scratch_dir=""
      remove_scratch_dir() {
        [[ -z "${scratch_dir}" ]] || rm -rf -- "${scratch_dir}"
      }
      trap remove_scratch_dir EXIT
      scratch_dir="$(mktemp -d "${scratch_parent}/qfv-tier0-mutations.XXXXXXXX")"
      summary_file="${scratch_dir}/summary.txt"
      # errexit plus pipefail: the runner already exits non-zero unless every
      # entry killed. The parse below is a second, independent check, so a
      # runner that lost its exit code still cannot get a token printed.
      python -m tests.mutations.runner --scratch-root "${scratch_dir}/tree" \
        | tee "${summary_file}"
      summary_count="$(grep -c -E "^QFV_MUT_SUMMARY " "${summary_file}" || true)"
      [[ "${summary_count}" == "1" ]] || {
        printf "error: expected one QFV_MUT_SUMMARY line, found %s\n" \
          "${summary_count}" >&2
        exit 1
      }
      summary="$(grep -E "^QFV_MUT_SUMMARY " "${summary_file}")"
      summary_field() { sed -n "s/.* $1=\([^ ]*\).*/\1/p" <<<"${summary}"; }
      killed_ratio="$(summary_field killed)"
      selected="$(summary_field entries_selected)"
      total="$(summary_field entries_total)"
      [[ "${killed_ratio}" == */* && -n "${selected}" && -n "${total}" ]] || {
        printf "error: QFV_MUT_SUMMARY is missing counts: %s\n" "${summary}" >&2
        exit 1
      }
      # A manifest with no entries would otherwise satisfy killed == total,
      # which is a check whose antecedent is never satisfied.
      [[ "${total}" =~ ^[1-9][0-9]*$ ]] || {
        printf "error: the manifest declares no entries: %s\n" "${summary}" >&2
        exit 1
      }
      # Scope before verdict, and in this order on purpose. The runner prints
      # killed out of entries_total, so a subset run already fails the ratio
      # below -- checking the ratio first would leave this arm with an
      # antecedent nothing can satisfy, which is the shape this stage exists
      # to catch. Checked first, it is the arm that reports a subset run.
      [[ "${selected}" == "${total}" ]] || {
        printf "error: measured %s of %s manifest entries\n" \
          "${selected}" "${total}" >&2
        exit 1
      }
      [[ "${killed_ratio}" == "${total}/${total}" ]] || {
        printf "error: a mutation survived its named test: %s\n" "${summary}" >&2
        exit 1
      }
      printf "QFV_TIER0 mutations result=success killed=%s entries=%s self_test=pass manifest=tests/mutations/manifest.toml\n" \
        "${killed_ratio}" "${total}"
    ' tier0-mutations /workspace/torchtitan /project/tmp
fi

if ((RUN_FIDELITY)); then
  # The TLA+/Lean fidelity differential. ScoutBModel.tla and ScoutBProtocol.lean
  # are two hand-written models of one protocol, and if they drift the unbounded
  # Lean proof describes a different system from the one TLC checks. This stage
  # compares their nine shared predicates on generated instances.
  #
  # Two runs, because the clean run alone cannot show the comparator still works.
  # The polarity run injects one named perturbation and REQUIRES a non-zero exit
  # with at least one disagreement -- the same reason the mutation stage checks
  # its runner's polarity before believing its counts. One perturbation rather
  # than all six: the full sweep lives in the pytest suite, and the point here is
  # that the stage can fail at all, which one is enough to establish.
  #
  # A TLA-side perturbation is chosen on purpose, so the polarity check exercises
  # the TLA+ emitter, TLC execution and output parsing rather than only Lean's.
  #
  # Offline network and the formal cache, like the pytest and mutation stages:
  # this runs the real Lean and TLC toolchains, and the scratch tree lives under
  # /project/tmp so tier 0 still writes nothing under outputs/.
  env TORCHTITAN_ROOTFS_NETWORK=offline \
    TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST="${FORMAL_CACHE}" \
    "${ROOTFS_ENTRYPOINT}" --rootfs "${ROOTFS}" -- bash -lc '
      set -euo pipefail
      cd /workspace/torchtitan
      module="torchtitan.experiments.qwen3_formal_verifier.fidelity_diff"
      clean_dir="$(mktemp -d /project/tmp/qfv-fidelity-clean.XXXXXXXX)"
      polarity_dir="$(mktemp -d /project/tmp/qfv-fidelity-polarity.XXXXXXXX)"
      trap '"'"'rm -rf -- "${clean_dir}" "${polarity_dir}"'"'"' EXIT

      clean_log="${clean_dir}/clean.log"
      python -m "${module}" --work-dir "${clean_dir}" >"${clean_log}" 2>&1 || {
        cat "${clean_log}" >&2
        echo "error: fidelity differential failed on the clean run" >&2
        exit 1
      }
      clean_token="$(grep -c "^QFV_FIDELITY_DIFF result=success" "${clean_log}" || true)"
      [[ "${clean_token}" == "1" ]] || {
        cat "${clean_log}" >&2
        echo "error: expected exactly one clean success token, got ${clean_token}" >&2
        exit 1
      }
      # Read the counts out of the measurement, never printed as literals.
      read -r instances compared disagreements < <(
        sed -n "s/^QFV_FIDELITY_DIFF result=success .*instances=\([0-9]\+\) predicates_compared=\([0-9]\+\) .*disagreements=\([0-9]\+\) .*/\1 \2 \3/p" \
          "${clean_log}" | head -1)
      [[ -n "${instances}" && -n "${compared}" && -n "${disagreements}" ]] || {
        cat "${clean_log}" >&2
        echo "error: clean token carried no counts" >&2
        exit 1
      }
      [[ "${disagreements}" == "0" ]] || {
        echo "error: clean run reported ${disagreements} disagreements" >&2
        exit 1
      }
      (( compared > 0 )) || {
        echo "error: clean run compared nothing" >&2
        exit 1
      }

      # Polarity: the comparator must still catch an injected divergence.
      polarity_log="${polarity_dir}/polarity.log"
      polarity_status=0
      python -m "${module}" --work-dir "${polarity_dir}" --instances 24 \
        --perturb tla_completed_off_by_one >"${polarity_log}" 2>&1 \
        || polarity_status=$?
      (( polarity_status != 0 )) || {
        cat "${polarity_log}" >&2
        echo "error: perturbed run exited 0; the comparator cannot fail" >&2
        exit 1
      }
      polarity_disagreements="$(
        sed -n "s/^QFV_FIDELITY_DIFF result=failure .*disagreements=\([0-9]\+\) .*/\1/p" \
          "${polarity_log}" | head -1)"
      [[ -n "${polarity_disagreements}" ]] && (( polarity_disagreements > 0 )) || {
        cat "${polarity_log}" >&2
        echo "error: perturbed run reported no disagreements" >&2
        exit 1
      }

      printf "QFV_TIER0 fidelity result=success instances=%s predicates_compared=%s disagreements=%s polarity=caught:%s\n" \
        "${instances}" "${compared}" "${disagreements}" "${polarity_disagreements}"
    '
fi

printf 'QFV_TIER0 result=success mode=%s\n' "${FETCH_MODE#--}"
printf 'QFV_TIER0 reminder=tier 0 proves the models only; it is NOT a gate pass and sealed no bundle\n'
