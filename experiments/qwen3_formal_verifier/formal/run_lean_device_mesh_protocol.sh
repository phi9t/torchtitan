#!/usr/bin/env bash
# Kernel-check the general DeviceMeshModel protocol theorems and reject one
# controlled negative.
#
# This runner is the counterpart of run_lean_device_mesh.sh, and the DIFFERENCE is
# the point: run_lean_device_mesh.sh evaluates Bool-valued predicates against the
# observed 108-collective trace and emits tokens labelled
# kind=evaluation scope=observed-trace. A reader of the sealed log can tell
# the two apart without opening a source file.
#
# THREE VOCABULARIES HERE, NOT ONE. Labelling every result in this runner as a
# general theorem would repeat, one layer up, exactly the category error the
# relabelling exists to prevent.
#
#   kind=theorem scope=all-topologies-all-schedules bound=none
#       Universally quantified over Topology, State and Step. `bound=none`
#       means NO PARTICULAR BOUND IS ASSUMED -- not that maxIssues is absent
#       from the model. `Reachable` does depend on maxIssues through
#       issueAllowedB; the legitimacy of bound=none is that the Topology, and
#       hence its maxIssues, is universally quantified, so the result holds at
#       every bound including 108.
#   kind=witness scope=fixed-instance bound=<n>
#       `by decide` over ONE fixed finite topology whose maxIssues is <n>.
#       These are real kernel-checked results about that instance and nothing
#       more; two of them are existential rather than universal. They are what
#       shows the general theorems' hypotheses and conjunctions are
#       load-bearing, so they belong in the log -- under their own label.
#   kind=theorem-negative scope=fixed-instance bound=<n>
#       A proved NEGATION of a general statement, refuted by a fixed instance.
#
# Two conditional results additionally carry conditional=<hypothesis>, because
# "DeadlockFreedom holds" and "DeadlockFreedom holds where the wait-for graph
# is acyclic" are different claims and the second must not read as the first.
#
# One token per theorem, on purpose. The three inductive obligations --
# initiation, consecution, sufficiency -- are separate results because a
# missing one silently weakens the claim, and a single aggregate token would
# not show which one was missing.

set -euo pipefail

[[ $# -eq 1 ]] || {
  echo "usage: run_lean_device_mesh_protocol.sh <lean>" >&2
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
printf 'DEVICE_MESH_PROTOCOL_LEAN_TOOLCHAIN checker=lean version=4.34.0\n'

work_dir="${TEST_TMPDIR:?TEST_TMPDIR is required}/lean-device-mesh-protocol"
mkdir -p "${work_dir}"
for name in DeviceMeshProtocol DeviceMeshInductiveInvariant DeviceMeshWaitGraph \
  DeviceMeshProtocolInvalid; do
  cp "${fixture_dir}/${name}.lean" "${work_dir}/${name}.lean"
done

set +e
(
  cd "${work_dir}"
  LEAN_PATH=. timeout 300 "${lean_bin}" \
    -o DeviceMeshProtocol.olean DeviceMeshProtocol.lean &&
    LEAN_PATH=. timeout 300 "${lean_bin}" \
      -o DeviceMeshInductiveInvariant.olean DeviceMeshInductiveInvariant.lean &&
    LEAN_PATH=. timeout 300 "${lean_bin}" \
      -o DeviceMeshWaitGraph.olean DeviceMeshWaitGraph.lean
) >"${work_dir}/valid.log" 2>&1
valid_status=$?
set -e
valid_output="$(<"${work_dir}/valid.log")"

# theorem|label|role triples, in the order a reader should read them. The label
# is the full kind/scope/bound (plus conditional= where it applies), so a new
# entry cannot inherit a stronger label by being appended to the wrong list.
THEOREM_LABEL="kind=theorem scope=all-topologies-all-schedules bound=none"
WITNESS2_LABEL="kind=witness scope=fixed-instance bound=2"
WITNESS1_LABEL="kind=witness scope=fixed-instance bound=1"
NEGATIVE2_LABEL="kind=theorem-negative scope=fixed-instance bound=2"
declare -a theorems=(
  "idxOfOn_spec|${THEOREM_LABEL}|fidelity-index-reads-the-right-issue"
  "idxOfOn_isSome_of_le|${THEOREM_LABEL}|fidelity-index-exists-under-its-guard"
  "opAtOn_append_of_le|${THEOREM_LABEL}|fidelity-append-leaves-earlier-reads-alone"
  "initiation|${THEOREM_LABEL}|obligation-1-initiation"
  "consecution|${THEOREM_LABEL}|obligation-2-consecution"
  "sufficiency|${THEOREM_LABEL}|obligation-3-sufficiency"
  "invOfReachable|${THEOREM_LABEL}|composition-invariant-holds-at-every-reachable-state"
  "safetyOfReachable|${THEOREM_LABEL}|composition-safety-holds-at-every-reachable-state"
  "commFifoAloneIsNotInductive|${WITNESS1_LABEL}|strengthening-conjunction-is-required"
  "blockerOfStuckPending|${THEOREM_LABEL}|waitgraph-stuck-pending-communicator-has-a-blocker"
  "noChainInAcyclicRelation|${THEOREM_LABEL}|waitgraph-acyclic-relation-admits-no-chain"
  "noCircularWaitOfAcyclicWaitGraph|${THEOREM_LABEL} conditional=acyclic-wait-for-graph|waitgraph-no-circular-wait"
  "deadlockFreeOfAcyclicWaitGraph|${THEOREM_LABEL} conditional=acyclic-wait-for-graph|waitgraph-deadlock-freedom-is-conditional-not-unconditional"
  "orderAgreementAndAcyclicWaitGraphExcludeBothHazards|${THEOREM_LABEL} conditional=acyclic-wait-for-graph|general-protocol-theorem-returns-full-safety-plus-no-circular-wait"
  "cyclicWitnessIsStuck|${WITNESS2_LABEL}|witness-is-stuck"
  "waitGraphWitnessIsNotVacuous|${WITNESS2_LABEL}|witness-non-vacuity"
  "cyclicWitnessIsWaitClosed|${WITNESS2_LABEL}|witness-is-wait-closed"
  "cyclicWitnessHasTwoCycle|${WITNESS2_LABEL}|witness-has-a-two-communicator-cycle"
  "cyclicWitnessAdmitsNoRankFunction|${WITNESS2_LABEL}|witness-admits-no-rank-function"
  "acyclicityIsLoadBearing|${NEGATIVE2_LABEL}|hypothesis-negative-dropping-acyclicity-fails"
)


for entry in "${theorems[@]}"; do
  IFS='|' read -r theorem label role <<<"${entry}"
  formal_classify_lean_valid \
    "${valid_status}" "${valid_output}" \
    "Qwen3Formal.DeviceMeshProtocol.${theorem}" || {
      cat "${work_dir}/valid.log" >&2
      echo "Lean protocol theorem was not a clean axiom-free kernel check:" \
        "${theorem}" >&2
      exit 1
    }
  printf 'DEVICE_MESH_PROTOCOL_LEAN_THEOREM theorem=Qwen3Formal.DeviceMeshProtocol.%s %s role=%s axioms=[] exit=%s\n' \
    "${theorem}" "${label}" "${role}" "${valid_status}"
done

set +e
(
  cd "${work_dir}"
  LEAN_PATH=. timeout 300 "${lean_bin}" DeviceMeshProtocolInvalid.lean
) >"${work_dir}/invalid.log" 2>&1
invalid_status=$?
set -e
invalid_output="$(<"${work_dir}/invalid.log")"
formal_classify_lean_negative \
  "${invalid_status}" "${invalid_output}" \
  "Qwen3Formal.DeviceMeshProtocol.ControlledInvalidWaitGraphProposition" || {
  cat "${work_dir}/invalid.log" >&2
  echo "Lean protocol negative did not reject the named proposition" >&2
  exit 1
}
printf 'DEVICE_MESH_PROTOCOL_LEAN_NEGATIVE proposition=Qwen3Formal.DeviceMeshProtocol.ControlledInvalidWaitGraphProposition kind=theorem-negative scope=fixed-instance bound=2 result=rejected exit=%s\n' \
  "${invalid_status}"
