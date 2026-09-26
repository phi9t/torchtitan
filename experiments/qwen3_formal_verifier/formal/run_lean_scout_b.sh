#!/usr/bin/env bash
# Kernel-check Scout B DPxTP facts and reject one producer mutation.
#
# EVERY RESULT HERE IS AN EVALUATION OF THE OBSERVED TRACE, not a theorem about
# the protocol. The modules define Bool-valued predicates over the literal
# ScoutBFacts data and close them with `rfl` or `decide`, so what the kernel
# checks is that THIS 108-collective-per-rank run satisfies the predicate. That
# is genuine and the negatives fail as they should, but it is decidable
# arithmetic on constants and says nothing about any other trace. The tokens
# below therefore carry kind=evaluation scope=observed-trace, so the
# distinction is visible in the sealed log and not only in a source comment.
# The general, quantified theorems live in run_lean_scout_b_protocol.sh and
# carry kind=theorem scope=all-topologies-all-schedules bound=none.

set -euo pipefail

[[ $# -eq 1 ]] || {
  echo "usage: run_lean_scout_b.sh <lean>" >&2
  exit 2
}

resolve_runfile() {
  local requested="$1"
  local candidate
  for candidate in \
    "${requested}" \
    "${TEST_SRCDIR:-}/${requested}" \
    "${TEST_SRCDIR:-}/${requested#external/}"; do
    if [[ -f "${candidate}" ]]; then
      readlink -f -- "${candidate}"
      return 0
    fi
  done
  echo "runfile not found: ${requested}" >&2
  return 1
}

workspace="${TEST_WORKSPACE:-_main}"
fixture_dir="${TEST_SRCDIR}/${workspace}/experiments/qwen3_formal_verifier/formal"
[[ -d "${fixture_dir}" ]] || {
  echo "Scout B formal fixtures not found: ${fixture_dir}" >&2
  exit 2
}
# shellcheck source=experiments/qwen3_formal_verifier/formal/checker_contract.sh
source "${fixture_dir}/checker_contract.sh"

lean_bin="$(resolve_runfile "$1")"
version_output="$("${lean_bin}" --version 2>&1)" || {
  echo "Lean version check failed" >&2
  exit 1
}
grep -Fq 'Lean (version 4.34.0,' <<<"${version_output}" || {
  echo "unexpected Lean toolchain: ${version_output}" >&2
  exit 1
}
printf 'SCOUT_B_LEAN_TOOLCHAIN checker=lean version=4.34.0\n'

work_dir="${TEST_TMPDIR:?TEST_TMPDIR is required}/lean-scout-b"
mkdir -p "${work_dir}/valid" "${work_dir}/invalid"
valid_modules=(
  ScoutDistributed
  ScoutBFacts
  ScoutBBaseChecks
  ScoutBCollectiveEventRank0
  ScoutBCollectiveEventRank1
  ScoutBCollectiveEventRank2
  ScoutBCollectiveEventRank3
  ScoutBEventMutationChecks
  ScoutBEventRank0
  ScoutBEventRank1
  ScoutBEventRank2
  ScoutBEventRank3
  ScoutBPlacementChecks
  ScoutBProducerCheck
  ScoutBSyncChunk0
  ScoutBSyncChunk1
  ScoutBSyncChunk2
  ScoutBSyncChunk3
  ScoutBValid
)
for name in "${valid_modules[@]}"; do
  cp "${fixture_dir}/${name}.lean" "${work_dir}/valid/${name}.lean"
done
for name in ScoutDistributed ScoutBBadFacts ScoutBInvalid; do
  cp "${fixture_dir}/${name}.lean" "${work_dir}/invalid/${name}.lean"
done

set +e
(
  cd "${work_dir}/valid"
  LEAN_PATH=. timeout 120 "${lean_bin}" -o ScoutDistributed.olean ScoutDistributed.lean &&
    LEAN_PATH=. timeout 120 "${lean_bin}" -o ScoutBFacts.olean ScoutBFacts.lean || exit

  check_modules=(
    ScoutBBaseChecks
    ScoutBCollectiveEventRank0
    ScoutBCollectiveEventRank1
    ScoutBCollectiveEventRank2
    ScoutBCollectiveEventRank3
    ScoutBEventMutationChecks
    ScoutBEventRank0
    ScoutBEventRank1
    ScoutBEventRank2
    ScoutBEventRank3
    ScoutBPlacementChecks
    ScoutBProducerCheck
    ScoutBSyncChunk0
    ScoutBSyncChunk1
    ScoutBSyncChunk2
    ScoutBSyncChunk3
  )
  pids=()
  for name in "${check_modules[@]}"; do
    (
      LEAN_PATH=. timeout 120 "${lean_bin}" \
        -o "${name}.olean" "${name}.lean"
    ) >"${name}.log" 2>&1 &
    pids+=("$!")
  done
  check_status=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[${index}]}"; then
      check_status=1
    fi
    cat "${check_modules[${index}]}.log"
  done
  [[ "${check_status}" -eq 0 ]] || exit "${check_status}"
  LEAN_PATH=. timeout 120 "${lean_bin}" -o ScoutBValid.olean ScoutBValid.lean
) >"${work_dir}/valid.log" 2>&1
valid_status=$?
set -e
valid_output="$(<"${work_dir}/valid.log")"
formal_classify_lean_valid \
  "${valid_status}" "${valid_output}" \
  "Qwen3Formal.ScoutB.validDPxTP" || {
  cat "${work_dir}/valid.log" >&2
  echo "Lean Scout B theorem was not a clean axiom-free kernel check" >&2
  exit 1
}
printf 'SCOUT_B_LEAN_VALID theorem=Qwen3Formal.ScoutB.validDPxTP kind=evaluation scope=observed-trace axioms=[] exit=%s\n' \
  "${valid_status}"
for theorem in rejectsEventIdPairDrift rejectsWrongEventKind rejectsWrongEventOrder; do
  grep -Fxq \
    "'Qwen3Formal.ScoutBChecks.${theorem}' does not depend on any axioms" \
    <<<"${valid_output}" || {
      cat "${work_dir}/valid.log" >&2
      echo "Lean Scout B event mutation check was not axiom-free: ${theorem}" >&2
      exit 1
    }
  printf 'SCOUT_B_LEAN_MUTATION theorem=Qwen3Formal.ScoutBChecks.%s kind=evaluation scope=observed-trace result=rejected axioms=[]\n' \
    "${theorem}"
done

# Structural DTensor placement results. Reported separately from validDPxTP
# because validDPxTP keeps its original composition: the two placement booleans
# it reads still mean only "some parameter is sharded on each axis", and
# placementBooleansAgreeObserved is what ties them to the per-parameter
# facts.
for theorem in \
  placementSchemaAgreesObserved \
  meshAxisDegreeObserved \
  parameterPlacementWellFormedObserved \
  placementBooleansAgreeObserved \
  optimizerBoundaryObservedValid \
  noPartialObserved \
  shardedDimDividesAxisDegreeObserved \
  localShapeReflectsShardingObserved \
  stridedShardCompositionObserved
do
  grep -Fxq \
    "'Qwen3Formal.ScoutBChecks.${theorem}' does not depend on any axioms" \
    <<<"${valid_output}" || {
      cat "${work_dir}/valid.log" >&2
      echo "Lean Scout B placement check was not axiom-free: ${theorem}" >&2
      exit 1
    }
  printf 'SCOUT_B_LEAN_PLACEMENT theorem=Qwen3Formal.ScoutBChecks.%s kind=evaluation scope=observed-trace axioms=[]\n' \
    "${theorem}"
done

# The observed run has zero Partial placements, so the positive result above
# proves nothing about whether one would be caught. These are the derived
# injection: the mutation is not a no-op, the Partial check rejects it, and
# nothing else does.
# The role is spelled out per theorem: only one of the three is the rejection,
# and labelling the guard and the isolation "rejected" too would misreport what
# the log shows.
for entry in \
  injectedPartialIsIsolated:mutation-is-not-a-no-op \
  rejectsInjectedPartial:partial-check-rejects-it \
  injectedPartialPassesTheOtherPlacementChecks:survivors-hold \
  injectedPartialBreaksLocalShapeAgreement:local-shape-cannot-survive
do
  theorem="${entry%%:*}"
  role="${entry##*:}"
  grep -Fxq \
    "'Qwen3Formal.ScoutBChecks.${theorem}' does not depend on any axioms" \
    <<<"${valid_output}" || {
      cat "${work_dir}/valid.log" >&2
      echo "Lean Scout B Partial injection was not axiom-free: ${theorem}" >&2
      exit 1
    }
  printf 'SCOUT_B_LEAN_PLACEMENT_MUTATION theorem=Qwen3Formal.ScoutBChecks.%s kind=evaluation scope=observed-trace role=%s axioms=[]\n' \
    "${theorem}" "${role}"
done

set +e
(
  cd "${work_dir}/invalid"
  LEAN_PATH=. timeout 120 "${lean_bin}" -o ScoutDistributed.olean ScoutDistributed.lean &&
    LEAN_PATH=. timeout 120 "${lean_bin}" -o ScoutBBadFacts.olean ScoutBBadFacts.lean &&
    LEAN_PATH=. timeout 120 "${lean_bin}" ScoutBInvalid.lean
) >"${work_dir}/invalid.log" 2>&1
invalid_status=$?
set -e
invalid_output="$(<"${work_dir}/invalid.log")"
formal_classify_lean_negative \
  "${invalid_status}" "${invalid_output}" \
  "Qwen3Formal.ScoutB.ControlledInvalidProducerProposition" || {
  cat "${work_dir}/invalid.log" >&2
  echo "Lean Scout B negative did not reject the named proposition" >&2
  exit 1
}
printf 'SCOUT_B_LEAN_NEGATIVE proposition=Qwen3Formal.ScoutB.ControlledInvalidProducerProposition kind=evaluation scope=observed-trace result=rejected exit=%s\n' \
  "${invalid_status}"
