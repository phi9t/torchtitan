#!/usr/bin/env bash
# Model-check the abstract DPxTP collective protocol.
#
# Seven checks. Each one names the exact statement it establishes; the module
# header of ScoutBModel.tla carries the same list as the claim inventory, and
# the two must agree.
#
#   safety       every safety invariant holds across every admitted
#                interleaving of four ranks and eight communicators up to the
#                bound MaxIssues per rank, with issue skew between ranks
#                unbounded within that bound. The bound is reported in the
#                result token: it is NOT a claim about the observed run.
#   non-vacuity  the model actually reaches completion, so those invariants
#                are not true merely because a guard is unsatisfiable;
#   divergent    relaxing ONLY communicator-site agreement -- operations still
#                agree at every position and NCCL's own guard stays on -- makes
#                DeadlockFreedom false;
#   witness      and the state it finds has the right shape: a cycle over at
#                least two distinct communicators, operations agreeing, issue
#                counts equal on every communicator of the chain, every rank at
#                its full budget. Without this, "DeadlockFreedom was violated"
#                says nothing about which stuck state was found;
#   op mismatch  relaxing operation agreement instead reaches a hang whose
#                cause is an intra-communicator operation mismatch, reported
#                under its own token and its own invariant so the two hangs
#                are never confused;
#   stream edge  the stream map is not refined by communicator identity on this
#                instance, so AtStreamHead is not implied by per-communicator
#                FIFO -- and substituting StreamOfIssue(e) == e.comm, the design
#                that made it redundant, makes that check stop failing. This
#                replaces an exhaustive run with the stream edge deleted, which
#                could not fail: see the lemma above StreamEdgeIsInert;
#   unguarded    without NCCL's matching requirement, that mismatch instead
#                runs a corrupt rendezvous.
#
# Every stage runs at one worker under a 120s timeout. The slowest is about 35s,
# so the headroom is roughly 3x and the numbers are reproducible run to run.

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

# One worker everywhere, deliberately: the reported state counts are then
# reproducible, and a negative's search order does not depend on how many cores
# the machine happens to have.
run_tlc() {
  local name="$1"
  local module="$2"
  local config="$3"
  local status=0
  (
    cd "${work_dir}/${name}"
    timeout 120 "${java_bin}" -XX:+UseParallelGC -cp "${tla_jar}" \
      tlc2.TLC -workers 1 -metadir "${work_dir}/${name}/states" \
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

# Scalar constant as the cfg actually binds it, so the reported bound cannot
# drift from the bound that was checked.
cfg_scalar() {
  sed -n "s/^[[:space:]]*$2[[:space:]]*=[[:space:]]*\([A-Za-z0-9_]\+\).*/\1/p" \
    "${fixture_dir}/$1" | tail -1
}

# Number of elements in a set-literal definition of the module, used to report
# the size of the instance the cfg binds rather than a number typed by hand.
set_size() {
  awk -v def="$1" '
    index($0, def " ==") == 1 { collecting = 1 }
    collecting { text = text $0 }
    collecting && index($0, "}") > 0 { exit }
    END {
      if (text == "") exit 1
      sub(/.*\{/, "", text)
      sub(/\}.*/, "", text)
      print split(text, parts, ",")
    }' "${fixture_dir}/ScoutBModel.tla"
}

# 1. Safety across every admitted interleaving, up to the declared bound.
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
max_issues="$(cfg_scalar ScoutBModel.cfg MaxIssues)"
num_ranks="$(set_size Ranks2x2)"
num_comms="$(set_size CommIds2x2)"
[[ -n "${max_issues}" && -n "${num_ranks}" && -n "${num_comms}" ]] || {
  echo "could not read the checked bound out of the cfg and the module" >&2
  exit 1
}
# The bound belongs in the token. A reader who meets only the state count has
# no way to tell that this is a claim about schedules of at most max_issues
# collectives per rank, not about the observed 108-per-rank run.
printf 'SCOUT_B_MODEL_SAFETY result=success distinct_states=%s max_outdegree=%s bound_max_issues_per_rank=%s bound_ranks=%s bound_communicators=%s bound_issue_skew=unbounded exit=%s\n' \
  "${safety_states}" "${safety_outdegree}" "${max_issues}" "${num_ranks}" \
  "${num_comms}" "${safety_status}"

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

# 3. Order divergence alone deadlocks. Operations still agree at every
# position, NCCL's guard is still on, and DeadlockFreedom is the circular-wait
# property itself, so this witness is a wait cycle rather than an operation
# mismatch. Stage 4 pins the rest of the shape.
stage divergent ScoutBModel.tla ScoutBModelDivergent.cfg
divergent_status=0
run_tlc divergent ScoutBModel ScoutBModelDivergent || divergent_status=$?
divergent_output="$(<"${work_dir}/divergent.log")"
formal_classify_tlc_transition_negative \
  "${divergent_status}" "${divergent_output}" "DeadlockFreedom" || {
  cat "${work_dir}/divergent.log" >&2
  echo "order divergence did not produce the expected circular wait" >&2
  exit 1
}
printf 'SCOUT_B_MODEL_DIVERGENT invariant=DeadlockFreedom result=named_violation exit=%s\n' \
  "${divergent_status}"

# 4. The witness shape, in the same configuration. Only the invariant differs
# from stage 3, which the runner checks, so this cannot become a statement about
# a different model.
stage witness ScoutBModel.tla ScoutBModelDivergent.cfg ScoutBModelWitness.cfg
cfg_body() {
  grep -v '^[[:space:]]*\\\*' "$1" | grep -v '^[[:space:]]*$'
}
witness_pair_diff="$(
  diff <(cfg_body "${work_dir}/witness/ScoutBModelDivergent.cfg") \
       <(cfg_body "${work_dir}/witness/ScoutBModelWitness.cfg") \
    || true
)"
expected_witness_diff="$(printf '%s\n' \
  '< INVARIANT DeadlockFreedom' \
  '> INVARIANT NoCrossCommunicatorCycleWitness')"
[[ "$(grep -E '^[<>]' <<<"${witness_pair_diff}")" == "${expected_witness_diff}" ]] || {
  printf 'witness cfg differs from the divergent cfg in more than the invariant:\n%s\n' \
    "${witness_pair_diff}" >&2
  exit 1
}
witness_status=0
run_tlc witness ScoutBModel ScoutBModelWitness || witness_status=$?
witness_output="$(<"${work_dir}/witness.log")"
formal_classify_tlc_transition_negative \
  "${witness_status}" "${witness_output}" "NoCrossCommunicatorCycleWitness" || {
  cat "${work_dir}/witness.log" >&2
  echo "the deadlock reached under order divergence is not a balanced" \
    "cross-communicator cycle at full budget" >&2
  exit 1
}
printf 'SCOUT_B_MODEL_WITNESS invariant=NoCrossCommunicatorCycleWitness result=named_violation exit=%s\n' \
  "${witness_status}"

# 5. Operation divergence is a different hazard and gets a different token.
stage opmismatch ScoutBModel.tla ScoutBModelOpMismatch.cfg
opmismatch_status=0
run_tlc opmismatch ScoutBModel ScoutBModelOpMismatch || opmismatch_status=$?
opmismatch_output="$(<"${work_dir}/opmismatch.log")"
formal_classify_tlc_transition_negative \
  "${opmismatch_status}" "${opmismatch_output}" "NoOpMismatchHang" || {
  cat "${work_dir}/opmismatch.log" >&2
  echo "operation divergence did not produce the expected mismatch hang" >&2
  exit 1
}
printf 'SCOUT_B_MODEL_OPMISMATCH invariant=NoOpMismatchHang result=named_violation exit=%s\n' \
  "${opmismatch_status}"

# 6. The stream edge is load-bearing, checked where the content actually is.
#
# An exhaustive run with RequireStreamOrder = FALSE cannot fail: with the edge
# deleted, StartAllowed reduces to FullyPending /\ OpsAgreeAtFront, which Stuck
# already denies for every communicator, so StuckByCircularWait is unsatisfiable
# by construction at any bound on any instance. What is contingent is the
# lemma's hypothesis: that the shipped stream map is NOT refined by
# communicator identity here. StreamEdgeIsInert mentions no variables, so TLC
# refutes it as a false constant expression rather than as an initial-state
# violation, which is why this stage uses its own classifier.
stage streamedge ScoutBModel.tla ScoutBModelStreamShape.cfg
streamedge_status=0
run_tlc streamedge ScoutBModel ScoutBModelStreamShape || streamedge_status=$?
streamedge_output="$(<"${work_dir}/streamedge.log")"
formal_classify_tlc_constant_false \
  "${streamedge_status}" "${streamedge_output}" "StreamEdgeIsInert" || {
  cat "${work_dir}/streamedge.log" >&2
  echo "no rank holds two distinct communicators on one stream, so the" \
    "stream-head conjunct is redundant with per-communicator FIFO" >&2
  exit 1
}
printf 'SCOUT_B_MODEL_STREAM_EDGE invariant=StreamEdgeIsInert result=named_violation exit=%s\n' \
  "${streamedge_status}"

# And the same check must PASS once the stream map is made per-communicator,
# which is what an earlier version of this module effectively had. Without this
# half, stage 6 could not tell a load-bearing stream map from a guard that
# happens to be satisfiable.
stage streamedge_inert ScoutBModel.tla ScoutBModelStreamShape.cfg
awk '
  /^StreamOfIssue\(e\) ==$/ { print "StreamOfIssue(e) == e.comm"; skip = 1; next }
  skip && /^$/ { skip = 0 }
  skip { next }
  { print }
' "${work_dir}/streamedge/ScoutBModel.tla" \
  >"${work_dir}/streamedge_inert/ScoutBModel.tla.mutated"
mv "${work_dir}/streamedge_inert/ScoutBModel.tla.mutated" \
   "${work_dir}/streamedge_inert/ScoutBModel.tla"
# Fail closed if the substitution did not apply: a renamed operator would
# otherwise leave this half checking the unmutated module and passing for the
# wrong reason.
grep -Fxq 'StreamOfIssue(e) == e.comm' \
  "${work_dir}/streamedge_inert/ScoutBModel.tla" || {
  echo "the StreamOfIssue mutation did not apply; the stream-edge regression" \
    "guard would be vacuous" >&2
  exit 1
}
! grep -q 'e.op = "all_gather"' \
  "${work_dir}/streamedge_inert/ScoutBModel.tla" || {
  echo "the StreamOfIssue mutation left the original CASE arms in place" >&2
  exit 1
}
streamedge_inert_status=0
run_tlc streamedge_inert ScoutBModel ScoutBModelStreamShape \
  || streamedge_inert_status=$?
streamedge_inert_output="$(<"${work_dir}/streamedge_inert.log")"
formal_classify_tlc_valid \
  "${streamedge_inert_status}" "${streamedge_inert_output}" || {
  cat "${work_dir}/streamedge_inert.log" >&2
  echo "a per-communicator stream map still failed StreamEdgeIsInert, so that" \
    "check does not detect the inert design it exists to detect" >&2
  exit 1
}
printf 'SCOUT_B_MODEL_STREAM_EDGE_MUTANT substitution=StreamOfIssue_is_comm invariant=StreamEdgeIsInert result=holds exit=%s\n' \
  "${streamedge_inert_status}"

# 7. Without NCCL's matching requirement, the same divergence corrupts instead.
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
