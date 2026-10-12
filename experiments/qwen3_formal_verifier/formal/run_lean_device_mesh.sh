#!/usr/bin/env bash
# Kernel-check Device-mesh DPxTP facts and reject one producer mutation.
#
# EVERY RESULT HERE IS AN EVALUATION OF THE OBSERVED TRACE, not a theorem about
# the protocol. The modules define Bool-valued predicates over the literal
# DeviceMeshFacts data and close them with `rfl` or `decide`, so what the kernel
# checks is that THIS 108-collective-per-rank run satisfies the predicate. That
# is genuine and the negatives fail as they should, but it is decidable
# arithmetic on constants and says nothing about any other trace. The tokens
# below therefore carry kind=evaluation scope=observed-trace, so the
# distinction is visible in the sealed log and not only in a source comment.
# The general, quantified theorems live in run_lean_device_mesh_protocol.sh and
# carry kind=theorem scope=all-topologies-all-schedules bound=none.

set -euo pipefail

[[ $# -eq 1 ]] || {
  echo "usage: run_lean_device_mesh.sh <lean>" >&2
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
  echo "Device-mesh formal fixtures not found: ${fixture_dir}" >&2
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
printf 'DEVICE_MESH_LEAN_TOOLCHAIN checker=lean version=4.34.0\n'

work_dir="${TEST_TMPDIR:?TEST_TMPDIR is required}/lean-device-mesh"
mkdir -p "${work_dir}/valid" "${work_dir}/invalid"
valid_modules=(
  MeshTopology
  DeviceMeshFacts
  DeviceMeshBaseChecks
  DeviceMeshCollectiveEventRank0
  DeviceMeshCollectiveEventRank1
  DeviceMeshCollectiveEventRank2
  DeviceMeshCollectiveEventRank3
  DeviceMeshEventMutationChecks
  DeviceMeshEventRank0
  DeviceMeshEventRank1
  DeviceMeshEventRank2
  DeviceMeshEventRank3
  DeviceMeshPayloadChecks
  DeviceMeshPayloadMutationChecks
  DeviceMeshPlacementChecks
  DeviceMeshProducerCheck
  DeviceMeshSyncChunk0
  DeviceMeshSyncChunk1
  DeviceMeshSyncChunk2
  DeviceMeshSyncChunk3
  DeviceMeshValid
)
for name in "${valid_modules[@]}"; do
  cp "${fixture_dir}/${name}.lean" "${work_dir}/valid/${name}.lean"
done
for name in MeshTopology DeviceMeshBadFacts DeviceMeshInvalid; do
  cp "${fixture_dir}/${name}.lean" "${work_dir}/invalid/${name}.lean"
done

set +e
(
  cd "${work_dir}/valid"
  LEAN_PATH=. timeout 120 "${lean_bin}" -o MeshTopology.olean MeshTopology.lean &&
    LEAN_PATH=. timeout 120 "${lean_bin}" -o DeviceMeshFacts.olean DeviceMeshFacts.lean || exit

  check_modules=(
    DeviceMeshBaseChecks
    DeviceMeshCollectiveEventRank0
    DeviceMeshCollectiveEventRank1
    DeviceMeshCollectiveEventRank2
    DeviceMeshCollectiveEventRank3
    DeviceMeshEventMutationChecks
    DeviceMeshEventRank0
    DeviceMeshEventRank1
    DeviceMeshEventRank2
    DeviceMeshEventRank3
    DeviceMeshPayloadChecks
    DeviceMeshPayloadMutationChecks
    DeviceMeshPlacementChecks
    DeviceMeshProducerCheck
    DeviceMeshSyncChunk0
    DeviceMeshSyncChunk1
    DeviceMeshSyncChunk2
    DeviceMeshSyncChunk3
  )
  # Per-module budget. 120s suits the small modules. The payload modules
  # evaluate their predicates over all 432 works with no caching between
  # theorems, MEASURED at about 26s per agreement evaluation and 13s per
  # well-formedness evaluation at that scale, so they get 300s. Splitting the
  # payload positives from the payload mutations is the other half of that
  # decision; the all-pairs form of the agreement predicate had not finished
  # after four minutes and is why the Lean facts carry a per-collective row.
  module_timeout() {
    case "$1" in
      DeviceMeshPayloadChecks | DeviceMeshPayloadMutationChecks) printf '300\n' ;;
      *) printf '120\n' ;;
    esac
  }
  pids=()
  for name in "${check_modules[@]}"; do
    (
      LEAN_PATH=. timeout "$(module_timeout "${name}")" "${lean_bin}" \
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
  LEAN_PATH=. timeout 120 "${lean_bin}" -o DeviceMeshValid.olean DeviceMeshValid.lean
) >"${work_dir}/valid.log" 2>&1
valid_status=$?
set -e
valid_output="$(<"${work_dir}/valid.log")"
formal_classify_lean_valid \
  "${valid_status}" "${valid_output}" \
  "Qwen3Formal.DeviceMesh.validDPxTP" || {
  cat "${work_dir}/valid.log" >&2
  echo "Lean Device-mesh theorem was not a clean axiom-free kernel check" >&2
  exit 1
}
printf 'DEVICE_MESH_LEAN_VALID theorem=Qwen3Formal.DeviceMesh.validDPxTP kind=evaluation scope=observed-trace axioms=[] exit=%s\n' \
  "${valid_status}"
for theorem in rejectsEventIdPairDrift rejectsWrongEventKind rejectsWrongEventOrder; do
  grep -Fxq \
    "'Qwen3Formal.DeviceMeshChecks.${theorem}' does not depend on any axioms" \
    <<<"${valid_output}" || {
      cat "${work_dir}/valid.log" >&2
      echo "Lean Device-mesh event mutation check was not axiom-free: ${theorem}" >&2
      exit 1
    }
  printf 'DEVICE_MESH_LEAN_MUTATION theorem=Qwen3Formal.DeviceMeshChecks.%s kind=evaluation scope=observed-trace result=rejected axioms=[]\n' \
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
    "'Qwen3Formal.DeviceMeshChecks.${theorem}' does not depend on any axioms" \
    <<<"${valid_output}" || {
      cat "${work_dir}/valid.log" >&2
      echo "Lean Device-mesh placement check was not axiom-free: ${theorem}" >&2
      exit 1
    }
  printf 'DEVICE_MESH_LEAN_PLACEMENT theorem=Qwen3Formal.DeviceMeshChecks.%s kind=evaluation scope=observed-trace axioms=[]\n' \
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
    "'Qwen3Formal.DeviceMeshChecks.${theorem}' does not depend on any axioms" \
    <<<"${valid_output}" || {
      cat "${work_dir}/valid.log" >&2
      echo "Lean Device-mesh Partial injection was not axiom-free: ${theorem}" >&2
      exit 1
    }
  printf 'DEVICE_MESH_LEAN_PLACEMENT_MUTATION theorem=Qwen3Formal.DeviceMeshChecks.%s kind=evaluation scope=observed-trace role=%s axioms=[]\n' \
    "${theorem}" "${role}"
done

# Collective payload identity. OBSERVED off each rank's own NCCL Flight
# Recorder entry -- unlike the producer and stream fields, which are attributed
# by a positional zip -- and per tensor: one shape and one dtype name per tensor.
for theorem in \
  payloadWellFormedObserved \
  payloadCoversCollectivesObserved \
  payloadCommKeyObserved \
  payloadAgreementObserved \
  payloadSizeRelationObserved
do
  grep -Fxq \
    "'Qwen3Formal.DeviceMeshChecks.${theorem}' does not depend on any axioms" \
    <<<"${valid_output}" || {
      cat "${work_dir}/valid.log" >&2
      echo "Lean Device-mesh payload check was not axiom-free: ${theorem}" >&2
      exit 1
    }
  printf 'DEVICE_MESH_LEAN_PAYLOAD theorem=Qwen3Formal.DeviceMeshChecks.%s kind=evaluation scope=observed-trace axioms=[]\n' \
    "${theorem}"
done

# The observed run has no payload mismatch and no mislabelled operation, so the
# positives above prove nothing about whether either would be caught. These are
# the two derived injections, with the role of each theorem spelled out: only one
# of each group is the rejection, and labelling the guard and the survivors
# "rejected" too would misreport what the log shows.
for entry in \
  injectedSizeMismatchIsIsolated:mutation-is-not-a-no-op \
  rejectsInjectedSizeMismatch:agreement-rejects-it \
  injectedSizeMismatchPassesTheOtherPayloadChecks:survivors-hold \
  mislabelledEveryMemberOfOneCollective:mutation-covers-every-member \
  rejectsMislabelledOperation:size-relation-rejects-it \
  mislabelledOperationPassesTheOtherPayloadChecks:survivors-hold
do
  theorem="${entry%%:*}"
  role="${entry##*:}"
  grep -Fxq \
    "'Qwen3Formal.DeviceMeshChecks.${theorem}' does not depend on any axioms" \
    <<<"${valid_output}" || {
      cat "${work_dir}/valid.log" >&2
      echo "Lean Device-mesh payload injection was not axiom-free: ${theorem}" >&2
      exit 1
    }
  printf 'DEVICE_MESH_LEAN_PAYLOAD_MUTATION theorem=Qwen3Formal.DeviceMeshChecks.%s kind=evaluation scope=observed-trace role=%s axioms=[]\n' \
    "${theorem}" "${role}"
done

set +e
(
  cd "${work_dir}/invalid"
  LEAN_PATH=. timeout 120 "${lean_bin}" -o MeshTopology.olean MeshTopology.lean &&
    LEAN_PATH=. timeout 120 "${lean_bin}" -o DeviceMeshBadFacts.olean DeviceMeshBadFacts.lean &&
    LEAN_PATH=. timeout 120 "${lean_bin}" DeviceMeshInvalid.lean
) >"${work_dir}/invalid.log" 2>&1
invalid_status=$?
set -e
invalid_output="$(<"${work_dir}/invalid.log")"
formal_classify_lean_negative \
  "${invalid_status}" "${invalid_output}" \
  "Qwen3Formal.DeviceMesh.ControlledInvalidProducerProposition" || {
  cat "${work_dir}/invalid.log" >&2
  echo "Lean Device-mesh negative did not reject the named proposition" >&2
  exit 1
}
printf 'DEVICE_MESH_LEAN_NEGATIVE proposition=Qwen3Formal.DeviceMesh.ControlledInvalidProducerProposition kind=evaluation scope=observed-trace result=rejected exit=%s\n' \
  "${invalid_status}"
