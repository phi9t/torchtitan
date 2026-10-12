#!/usr/bin/env bash
# Check the bounded observed Device-mesh DPxTP facts and producer negative.

set -euo pipefail

[[ $# -eq 2 ]] || {
  echo "usage: run_tlc_device_mesh.sh <java> <tla2tools.jar>" >&2
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

java_bin="$(resolve_runfile "$1")"
tla_jar="$(resolve_runfile "$2")"
java_version_output="$("${java_bin}" -version 2>&1)" || {
  echo "Java runtime version check failed" >&2
  exit 1
}
grep -Eq ' version "17([."]|$)' <<<"${java_version_output}" || {
  echo "unexpected Java runtime: ${java_version_output}" >&2
  exit 1
}
printf 'DEVICE_MESH_TLA_TOOLCHAIN checker=tlc release=1.7.4 java_major=17\n'

work_dir="${TEST_TMPDIR:?TEST_TMPDIR is required}/tlc-device-mesh"
mkdir -p "${work_dir}/valid" "${work_dir}/invalid"
for name in MeshTopology DeviceMeshFacts DeviceMeshValid; do
  cp "${fixture_dir}/${name}.tla" "${work_dir}/valid/${name}.tla"
done
cp "${fixture_dir}/DeviceMeshValid.cfg" "${work_dir}/valid/DeviceMeshValid.cfg"
for name in MeshTopology DeviceMeshBadFacts DeviceMeshInvalid; do
  cp "${fixture_dir}/${name}.tla" "${work_dir}/invalid/${name}.tla"
done
cp "${fixture_dir}/DeviceMeshInvalid.cfg" "${work_dir}/invalid/DeviceMeshInvalid.cfg"

set +e
(
  cd "${work_dir}/valid"
  timeout 120 "${java_bin}" -XX:+UseParallelGC -cp "${tla_jar}" tlc2.TLC \
    -workers 1 -metadir "${work_dir}/valid/states" \
    DeviceMeshValid.tla -config DeviceMeshValid.cfg
) >"${work_dir}/valid.log" 2>&1
valid_status=$?
set -e
valid_output="$(<"${work_dir}/valid.log")"
formal_classify_tlc_valid "${valid_status}" "${valid_output}" || {
  cat "${work_dir}/valid.log" >&2
  echo "TLC Device-mesh valid facts were not a clean checker success" >&2
  exit 1
}
# Name the invariants, do not just count them. This run checks 19 of them and
# the token used to name none, so the only invariant a log reader ever met by
# name was whichever one a failure reported. Same derivation the DPxTP model
# runner uses, out of the cfg the checker was actually given.
valid_invariants="$(formal_cfg_invariants "${fixture_dir}/DeviceMeshValid.cfg")"
[[ -n "${valid_invariants}" ]] || {
  echo "could not read the invariant list out of DeviceMeshValid.cfg" >&2
  exit 1
}
printf 'DEVICE_MESH_TLA_VALID invariant_set=dpxTP result=success invariants=%s exit=%s\n' \
  "${valid_invariants}" "${valid_status}"

set +e
(
  cd "${work_dir}/invalid"
  timeout 120 "${java_bin}" -XX:+UseParallelGC -cp "${tla_jar}" tlc2.TLC \
    -workers 1 -metadir "${work_dir}/invalid/states" \
    DeviceMeshInvalid.tla -config DeviceMeshInvalid.cfg
) >"${work_dir}/invalid.log" 2>&1
invalid_status=$?
set -e
invalid_output="$(<"${work_dir}/invalid.log")"
formal_classify_tlc_transition_negative \
  "${invalid_status}" "${invalid_output}" \
  "DeviceMeshCollectiveProducerCorrelation" || {
  cat "${work_dir}/invalid.log" >&2
  echo "TLC Device-mesh negative did not violate the named producer invariant" >&2
  exit 1
}
printf 'DEVICE_MESH_TLA_NEGATIVE invariant=DeviceMeshCollectiveProducerCorrelation result=named_violation exit=%s\n' \
  "${invalid_status}"

# Per-communicator issue-order agreement: the NCCL requirement whose violation
# hangs a job. The facts differ from the observed run in exactly one operation
# on one member rank, so the violation is attributable to that mutation.
mkdir -p "${work_dir}/issue-order"
for name in MeshTopology DeviceMeshFacts DeviceMeshIssueOrderInvalid; do
  cp "${fixture_dir}/${name}.tla" "${work_dir}/issue-order/${name}.tla"
done
cp "${fixture_dir}/DeviceMeshIssueOrderInvalid.cfg" \
  "${work_dir}/issue-order/DeviceMeshIssueOrderInvalid.cfg"

issue_order_status=0
(
  cd "${work_dir}/issue-order"
  timeout 120 "${java_bin}" -XX:+UseParallelGC -cp "${tla_jar}" tlc2.TLC \
    -workers 1 -metadir "${work_dir}/issue-order/states" \
    DeviceMeshIssueOrderInvalid.tla -config DeviceMeshIssueOrderInvalid.cfg
) >"${work_dir}/issue-order.log" 2>&1 || issue_order_status=$?
issue_order_output="$(<"${work_dir}/issue-order.log")"
formal_classify_tlc_transition_negative \
  "${issue_order_status}" "${issue_order_output}" \
  "DeviceMeshPerCommunicatorIssueOrder" || {
  cat "${work_dir}/issue-order.log" >&2
  echo "TLC Device-mesh issue-order negative did not violate exactly that invariant" >&2
  exit 1
}
issue_order_invariants="$(
  formal_cfg_invariants "${fixture_dir}/DeviceMeshIssueOrderInvalid.cfg"
)"
[[ -n "${issue_order_invariants}" ]] || {
  echo "could not read the invariant list out of DeviceMeshIssueOrderInvalid.cfg" >&2
  exit 1
}
printf 'DEVICE_MESH_TLA_ISSUE_ORDER_NEGATIVE invariant=DeviceMeshPerCommunicatorIssueOrder result=named_violation invariants=%s exit=%s\n' \
  "${issue_order_invariants}" "${issue_order_status}"

# No Partial placement on a parameter the optimizer steps. The observed run has
# zero Partial placements, so the positive check alone proves nothing about
# whether one would be caught; this injects one as a derived override of the
# real facts and requires exactly that invariant to fail.
mkdir -p "${work_dir}/placement-partial"
for name in MeshTopology DeviceMeshFacts DeviceMeshPlacementPartialInvalid; do
  cp "${fixture_dir}/${name}.tla" "${work_dir}/placement-partial/${name}.tla"
done
cp "${fixture_dir}/DeviceMeshPlacementPartialInvalid.cfg" \
  "${work_dir}/placement-partial/DeviceMeshPlacementPartialInvalid.cfg"

placement_status=0
(
  cd "${work_dir}/placement-partial"
  timeout 120 "${java_bin}" -XX:+UseParallelGC -cp "${tla_jar}" tlc2.TLC \
    -workers 1 -metadir "${work_dir}/placement-partial/states" \
    DeviceMeshPlacementPartialInvalid.tla \
    -config DeviceMeshPlacementPartialInvalid.cfg
) >"${work_dir}/placement-partial.log" 2>&1 || placement_status=$?
placement_output="$(<"${work_dir}/placement-partial.log")"
formal_classify_tlc_transition_negative \
  "${placement_status}" "${placement_output}" \
  "DeviceMeshNoPartialAtOptimizer" || {
  cat "${work_dir}/placement-partial.log" >&2
  echo "TLC Device-mesh Partial-placement negative did not violate exactly that" \
    "invariant" >&2
  exit 1
}
# The survivor set is what the attribution rests on: these are the invariants
# asserted over the MUTATED facts and satisfied by them, so the reported
# violation is the Partial check and not a side effect of the override. Derived
# from the cfg rather than restated here, so the token cannot claim a survivor
# the checker was not given. LocalShapeReflectsSharding is deliberately not
# among them; see the module header for why.
placement_invariants="$(
  formal_cfg_invariants "${fixture_dir}/DeviceMeshPlacementPartialInvalid.cfg"
)"
[[ -n "${placement_invariants}" ]] || {
  echo "could not read the invariant list out of" \
    "DeviceMeshPlacementPartialInvalid.cfg" >&2
  exit 1
}
printf 'DEVICE_MESH_TLA_PARTIAL_PLACEMENT_NEGATIVE invariant=DeviceMeshNoPartialAtOptimizer result=named_violation invariants=%s exit=%s\n' \
  "${placement_invariants}" "${placement_status}"

# Collective payload identity. Two derived negatives, because the observed run
# satisfies both payload properties and a positive result alone says nothing
# about what either would catch.
#
# 1. One member of a collective called it with twice the volume of its peers.
#    Ordering agreement is in that control's survivor set on purpose: a volume
#    mismatch leaves every communicator's operation order exactly as observed,
#    which is why the ordering property cannot stand in for this one.
# 2. One collective's operation relabelled on EVERY member, the way an exporter
#    bug would mislabel it. Ordering agreement survives that too -- all four
#    ranks agree on the wrong label -- and only the volume relation notices.
for entry in \
  DeviceMeshPayloadSizeInvalid:DeviceMeshCollectivePayloadAgreement:PAYLOAD_SIZE \
  DeviceMeshPayloadOperationInvalid:DeviceMeshCollectivePayloadSizeRelation:PAYLOAD_OPERATION
do
  payload_module="${entry%%:*}"
  payload_rest="${entry#*:}"
  payload_invariant="${payload_rest%%:*}"
  payload_token="${payload_rest##*:}"
  payload_dir="${work_dir}/${payload_module}"
  mkdir -p "${payload_dir}"
  for name in MeshTopology DeviceMeshFacts "${payload_module}"; do
    cp "${fixture_dir}/${name}.tla" "${payload_dir}/${name}.tla"
  done
  cp "${fixture_dir}/${payload_module}.cfg" \
    "${payload_dir}/${payload_module}.cfg"

  payload_status=0
  (
    cd "${payload_dir}"
    timeout 120 "${java_bin}" -XX:+UseParallelGC -cp "${tla_jar}" tlc2.TLC \
      -workers 1 -metadir "${payload_dir}/states" \
      "${payload_module}.tla" -config "${payload_module}.cfg"
  ) >"${payload_dir}.log" 2>&1 || payload_status=$?
  payload_output="$(<"${payload_dir}.log")"
  formal_classify_tlc_transition_negative \
    "${payload_status}" "${payload_output}" "${payload_invariant}" || {
    cat "${payload_dir}.log" >&2
    echo "TLC Device-mesh ${payload_module} did not violate exactly" \
      "${payload_invariant}" >&2
    exit 1
  }
  # The survivor set is the attribution, derived from the cfg the checker was
  # given rather than restated here.
  payload_invariants="$(
    formal_cfg_invariants "${fixture_dir}/${payload_module}.cfg"
  )"
  [[ -n "${payload_invariants}" ]] || {
    echo "could not read the invariant list out of" \
      "${payload_module}.cfg" >&2
    exit 1
  }
  printf 'DEVICE_MESH_TLA_%s_NEGATIVE invariant=%s result=named_violation invariants=%s exit=%s\n' \
    "${payload_token}" "${payload_invariant}" "${payload_invariants}" \
    "${payload_status}"
done

# Structural placement facts: how much of DeviceMeshFacts.tla they are. Parse cost
# is charged before a single state is generated, and the refinement runner's
# DEVICE_MESH_REFINE_PARSE token reports the whole-file parse time, so the share
# these facts take is reported here rather than being invisible.
placement_bytes="$(
  awk '/BEGIN structural placement facts/,/END structural placement facts/' \
    "${fixture_dir}/DeviceMeshFacts.tla" | wc -c
)"
facts_bytes="$(wc -c <"${fixture_dir}/DeviceMeshFacts.tla")"
placement_count="$(
  grep -c '^PlacementShardDim == ' "${fixture_dir}/DeviceMeshFacts.tla" || true
)"
[[ "${placement_bytes}" -gt 0 && "${placement_count}" -eq 1 ]] || {
  echo "DeviceMeshFacts.tla carries no delimited structural placement facts" >&2
  exit 1
}
printf 'DEVICE_MESH_TLA_PLACEMENT_FACTS facts_bytes=%s placement_bytes=%s\n' \
  "${facts_bytes}" "${placement_bytes}"

# Collective payload facts: the byte cost of this export on its own, for the
# same reason the placement block is delimited -- parse cost is charged before a
# single state is generated, and parse_ms measures machine load first.
payload_bytes="$(
  awk '/BEGIN collective payload facts/,/END collective payload facts/' \
    "${fixture_dir}/DeviceMeshFacts.tla" | wc -c
)"
payload_count="$(
  grep -c '^CollectivePayloadInputElements == ' \
    "${fixture_dir}/DeviceMeshFacts.tla" || true
)"
[[ "${payload_bytes}" -gt 0 && "${payload_count}" -eq 1 ]] || {
  echo "DeviceMeshFacts.tla carries no delimited collective payload facts" >&2
  exit 1
}
printf 'DEVICE_MESH_TLA_PAYLOAD_FACTS facts_bytes=%s payload_bytes=%s\n' \
  "${facts_bytes}" "${payload_bytes}"
