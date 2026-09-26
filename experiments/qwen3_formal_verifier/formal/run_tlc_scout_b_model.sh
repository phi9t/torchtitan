#!/usr/bin/env bash
# Model-check the abstract DPxTP collective protocol.
#
# Four checks:
#
#   safety      every safety invariant holds across EVERY admitted
#               interleaving of four ranks issuing concurrently;
#   non-vacuity the model actually reaches completion, so those invariants
#               are not true merely because a guard is unsatisfiable;
#   divergent   a program whose ranks issue different collective sequences
#               DEADLOCKS -- discovered by exploration, not asserted;
#   unguarded   without NCCL's own matching requirement, that divergence
#               instead runs a mismatched rendezvous.
#
# The last two are reported as separate results because they say different
# things: order agreement is necessary but not sufficient, and the guard is
# what turns silent corruption into a hang.

set -euo pipefail

[[ $# -eq 2 ]] || {
  echo "usage: run_tlc_scout_b_model.sh <java> <tla2tools.jar>" >&2
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
printf 'SCOUT_B_MODEL_TOOLCHAIN checker=tlc release=1.7.4 java_major=17\n'

work_dir="${TEST_TMPDIR:?TEST_TMPDIR is required}/tlc-scout-b-model"

stage() {
  local name="$1"
  shift
  mkdir -p "${work_dir}/${name}"
  local file
  for file in "$@"; do
    cp "${fixture_dir}/${file}" "${work_dir}/${name}/${file}"
  done
}

run_tlc() {
  local name="$1"
  local module="$2"
  local config="$3"
  local status=0
  (
    cd "${work_dir}/${name}"
    timeout 120 "${java_bin}" -XX:+UseParallelGC -cp "${tla_jar}" tlc2.TLC \
      -workers 1 -metadir "${work_dir}/${name}/states" \
      "${module}.tla" -config "${config}.cfg"
  ) >"${work_dir}/${name}.log" 2>&1 || status=$?
  return "${status}"
}

distinct_states() {
  sed -n 's/.*[0-9]\+ states generated, \([0-9]\+\) distinct states found.*/\1/p' \
    "${work_dir}/$1.log" | tail -1
}

# Maximum outdegree of the reachable state graph. This, not a state count, is
# what separates a model from a replay: a cursor walking a recorded sequence
# has outdegree 1 everywhere no matter how long the sequence is, so any
# threshold on state count would happily accept the very design this module
# exists to replace.
max_outdegree() {
  sed -n 's/.*the maximum \([0-9]\+\) and the 95th percentile.*/\1/p' \
    "${work_dir}/$1.log" | tail -1
}

# 1. Safety across every admitted interleaving.
stage safety ScoutBModel.tla ScoutBModel.cfg
safety_status=0
run_tlc safety ScoutBModel ScoutBModel || safety_status=$?
safety_output="$(<"${work_dir}/safety.log")"
formal_classify_tlc_valid "${safety_status}" "${safety_output}" || {
  cat "${work_dir}/safety.log" >&2
  echo "DPxTP model safety check was not a clean success" >&2
  exit 1
}
safety_states="$(distinct_states safety)"
safety_outdegree="$(max_outdegree safety)"
[[ "${safety_outdegree:-0}" -ge 2 ]] || {
  echo "DPxTP model never branches (max outdegree ${safety_outdegree:-0});" \
    "it admits a single behaviour and is a replay, not a model" >&2
  exit 1
}
printf 'SCOUT_B_MODEL_SAFETY result=success distinct_states=%s max_outdegree=%s exit=%s\n' \
  "${safety_states}" "${safety_outdegree}" "${safety_status}"

# 2. Non-vacuity. Start's guard uses CHOOSE and a stream-head condition that
# are easy to make unsatisfiable, which would make every safety invariant above
# hold trivially and this whole suite green for the wrong reason.
stage reach ScoutBModel.tla ScoutBModelReach.cfg
reach_status=0
run_tlc reach ScoutBModel ScoutBModelReach || reach_status=$?
reach_output="$(<"${work_dir}/reach.log")"
formal_classify_tlc_transition_negative \
  "${reach_status}" "${reach_output}" "ModelNeverCompletes" || {
  cat "${work_dir}/reach.log" >&2
  echo "DPxTP model never reaches completion; its invariants are vacuous" >&2
  exit 1
}
printf 'SCOUT_B_MODEL_NONVACUOUS result=completion_reachable exit=%s\n' \
  "${reach_status}"

# 3. A divergent program deadlocks.
stage divergent ScoutBModel.tla ScoutBModelDivergent.cfg
divergent_status=0
run_tlc divergent ScoutBModel ScoutBModelDivergent || divergent_status=$?
divergent_output="$(<"${work_dir}/divergent.log")"
formal_classify_tlc_transition_negative \
  "${divergent_status}" "${divergent_output}" "DeadlockFreedom" || {
  cat "${work_dir}/divergent.log" >&2
  echo "divergent schedule did not produce the expected deadlock" >&2
  exit 1
}
printf 'SCOUT_B_MODEL_DIVERGENT invariant=DeadlockFreedom result=named_violation exit=%s\n' \
  "${divergent_status}"

# 4. Without NCCL's matching requirement, the same divergence corrupts instead.
stage unguarded ScoutBModel.tla ScoutBModelUnguarded.cfg
unguarded_status=0
run_tlc unguarded ScoutBModel ScoutBModelUnguarded || unguarded_status=$?
unguarded_output="$(<"${work_dir}/unguarded.log")"
formal_classify_tlc_transition_negative \
  "${unguarded_status}" "${unguarded_output}" "RendezvousOpAgreement" || {
  cat "${work_dir}/unguarded.log" >&2
  echo "unguarded rendezvous did not produce the expected mismatch" >&2
  exit 1
}
printf 'SCOUT_B_MODEL_UNGUARDED invariant=RendezvousOpAgreement result=named_violation exit=%s\n' \
  "${unguarded_status}"
