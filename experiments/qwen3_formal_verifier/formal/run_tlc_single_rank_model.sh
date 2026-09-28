#!/usr/bin/env bash
# Model-check the abstract single-rank step and bridge it to the observed run.
#
# Five checks, in two groups:
#
#   safety     the protocol's invariants hold across EVERY admitted
#              interleaving, not just the one that was observed;
#   negative   relaxing the one load-bearing guard makes TLC DISCOVER a
#              counterexample by exploration;
#   refinement the observed trace is an admitted behaviour of the model, and
#              the observed trace is not empty (an empty one is admitted by
#              every model and would read as a pass);
#   negative   a controlled corruption of that trace is refused, and refused
#              no earlier than the mutation;
#   isolation  relaxing RequireReadyGradients alone makes that same
#              corruption admitted, which attributes the refusal to that one
#              guard rather than inferring it from where the refusal landed.
#
# The two groups are selected by the third argument, because they have
# different inputs and so belong to different Bazel targets. The `abstract`
# half reads only the hand-written model and its two configurations, so it is
# trace-free and can run on CPU in seconds with no sealed bundle in reach.
# The `refine` half additionally reads the generated SingleRankFacts module, so it
# depends on an exported run. Both halves stay in the sealed Single-rank and
# Device-mesh suites; the split exists so model iteration does not require the
# generated facts, not to weaken the gate.
#
# Refinement polarity: TLC proves reachability by refutation, so the witness
# that the observed trace is admitted is a violation of
# ObservedTraceIsNotAdmitted. This script translates that into a positive
# result token so the evidence does not read backwards.

set -euo pipefail

[[ $# -eq 3 ]] || {
  echo "usage: run_tlc_single_rank_model.sh <java> <tla2tools.jar>" \
    "<abstract|refine>" >&2
  exit 2
}
half="$3"
case "${half}" in
  abstract | refine) ;;
  *)
    echo "unknown model-check half: ${half} (expected abstract or refine)" >&2
    exit 2
    ;;
esac

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
  echo "Single-rank formal fixtures not found: ${fixture_dir}" >&2
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
printf 'SINGLE_RANK_MODEL_TOOLCHAIN checker=tlc release=1.7.4 java_major=17\n'

work_dir="${TEST_TMPDIR:?TEST_TMPDIR is required}/tlc-single-rank-model-${half}"

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

run_abstract_half() {
  # 1. Safety of the abstract model over all interleavings.
  stage safety TraceLifecycle.tla SingleRankModel.tla SingleRankModel.cfg
  local safety_status=0
  run_tlc safety SingleRankModel SingleRankModel || safety_status=$?
  local safety_output
  safety_output="$(<"${work_dir}/safety.log")"
  formal_classify_tlc_valid "${safety_status}" "${safety_output}" || {
    cat "${work_dir}/safety.log" >&2
    echo "abstract model safety check was not a clean success" >&2
    exit 1
  }
  local safety_states safety_outdegree
  safety_states="$(distinct_states safety)"
  safety_outdegree="$(max_outdegree safety)"
  # Refuse a specification that never branches. Outdegree 1 everywhere means
  # the "model" admits exactly one behaviour, which is a replay of a recorded
  # sequence however many states it visits.
  [[ "${safety_outdegree:-0}" -ge 2 ]] || {
    echo "abstract model never branches (max outdegree" \
      "${safety_outdegree:-0}); it admits a single behaviour and is a" \
      "replay, not a model" >&2
    exit 1
  }
  printf 'SINGLE_RANK_MODEL_SAFETY result=success distinct_states=%s max_outdegree=%s exit=%s\n' \
    "${safety_states}" "${safety_outdegree}" "${safety_status}"

  # 2. Relaxing the one guard must be discovered by exploration.
  stage negative TraceLifecycle.tla SingleRankModel.tla SingleRankModelUnsafe.cfg
  local negative_status=0
  run_tlc negative SingleRankModel SingleRankModelUnsafe || negative_status=$?
  local negative_output
  negative_output="$(<"${work_dir}/negative.log")"
  formal_classify_tlc_transition_negative \
    "${negative_status}" "${negative_output}" \
    "MutationRequiresReadyGradients" || {
    cat "${work_dir}/negative.log" >&2
    echo "relaxed guard did not produce the expected named counterexample" >&2
    exit 1
  }
  printf 'SINGLE_RANK_MODEL_NEGATIVE invariant=MutationRequiresReadyGradients result=named_violation exit=%s\n' \
    "${negative_status}"
}

run_refine_half() {
  # 3. The observed trace must be an admitted behaviour.
  stage refine \
    TraceLifecycle.tla SingleRankModel.tla SingleRankFacts.tla \
    SingleRankRefine.tla SingleRankRefine.cfg
  local refine_status=0
  run_tlc refine SingleRankRefine SingleRankRefine || refine_status=$?
  local refine_output
  refine_output="$(<"${work_dir}/refine.log")"
  # Checked before the refinement classification, because an EMPTY observed
  # sequence is admitted by every model -- Init already satisfies
  # emitted = Observed -- and yields the same exit 0 with no violation that a
  # REFUSED trace yields. Without this branch the runner printed "not
  # admitted" for a trace that was vacuously admitted, i.e. the opposite of
  # what happened. ObservedIsNonEmpty mentions no variables, so TLC refutes a
  # false one as a constant expression with its own status and message.
  if formal_classify_tlc_constant_false \
    "${refine_status}" "${refine_output}" "ObservedIsNonEmpty"; then
    cat "${work_dir}/refine.log" >&2
    echo "the observed event-kind sequence is empty, so the refinement is" \
      "vacuous: the empty trace is admitted by every model and proves" \
      "nothing about this one" >&2
    exit 1
  fi
  formal_classify_tlc_transition_negative \
    "${refine_status}" "${refine_output}" \
    "ObservedTraceIsNotAdmitted" || {
    cat "${work_dir}/refine.log" >&2
    echo "observed trace was not admitted by the abstract model" >&2
    exit 1
  }
  printf 'SINGLE_RANK_REFINEMENT result=admitted witness=ObservedTraceIsNotAdmitted exit=%s\n' \
    "${refine_status}"

  # 4. A corrupted trace must be refused, AND refused for the right reason.
  #
  # Rejection alone is weak evidence: a corruption that breaks the lifecycle
  # earlier would also be refused, by a different guard than the one under
  # test. So this runs three checks over the same corrupted trace -- it is
  # not admitted, it WAS admitted right up to the mutation, and relaxing
  # RequireReadyGradients alone makes it admitted. The third is the decisive
  # one: it attributes the refusal to that guard directly instead of
  # triangulating where the refusal happened.
  stage refine_bad \
    TraceLifecycle.tla SingleRankModel.tla SingleRankFacts.tla \
    SingleRankRefineBad.tla SingleRankRefineBad.cfg SingleRankRefineBadReach.cfg \
    SingleRankRefineBadRelaxed.cfg
  local refine_bad_status=0
  run_tlc refine_bad SingleRankRefineBad SingleRankRefineBad || refine_bad_status=$?
  local refine_bad_output
  refine_bad_output="$(<"${work_dir}/refine_bad.log")"
  formal_classify_tlc_valid "${refine_bad_status}" "${refine_bad_output}" || {
    cat "${work_dir}/refine_bad.log" >&2
    echo "corrupted trace was not refused by the abstract model" >&2
    exit 1
  }
  # There is deliberately no second grep for
  # "Invariant ObservedTraceIsNotAdmitted is violated" here. It would be dead
  # code dressed as a second line of defence: TLC exits 12 when it reports a
  # violation, so formal_classify_tlc_valid's exit-status pin has already
  # failed above by the time such a line could exist.

  stage refine_reach \
    TraceLifecycle.tla SingleRankModel.tla SingleRankFacts.tla \
    SingleRankRefineBad.tla SingleRankRefineBadReach.cfg
  local refine_reach_status=0
  run_tlc refine_reach SingleRankRefineBad SingleRankRefineBadReach \
    || refine_reach_status=$?
  local refine_reach_output
  refine_reach_output="$(<"${work_dir}/refine_reach.log")"
  formal_classify_tlc_transition_negative \
    "${refine_reach_status}" "${refine_reach_output}" \
    "RejectionHappensBeforeTheMutation" || {
    cat "${work_dir}/refine_reach.log" >&2
    echo "corrupted trace was refused before reaching the mutation guard," \
      "so the control exercises the wrong guard" >&2
    exit 1
  }

  printf 'SINGLE_RANK_REFINEMENT_NEGATIVE result=rejected_at_mutation_guard reject_exit=%s reach_exit=%s\n' \
    "${refine_bad_status}" "${refine_reach_status}"

  # 5. Guard isolation, proved directly. The same corrupted trace, the same
  # module, the same corrupted Observed -- only RequireReadyGradients differs.
  # If relaxing that one constant makes the whole corrupted trace admitted,
  # then that constant is what refused it in check 4, which no combination of
  # "where did it stop" observations can establish on its own.
  stage refine_relaxed \
    TraceLifecycle.tla SingleRankModel.tla SingleRankFacts.tla \
    SingleRankRefineBad.tla SingleRankRefineBadRelaxed.cfg
  local refine_relaxed_status=0
  run_tlc refine_relaxed SingleRankRefineBad SingleRankRefineBadRelaxed \
    || refine_relaxed_status=$?
  local refine_relaxed_output
  refine_relaxed_output="$(<"${work_dir}/refine_relaxed.log")"
  formal_classify_tlc_transition_negative \
    "${refine_relaxed_status}" "${refine_relaxed_output}" \
    "ObservedTraceIsNotAdmitted" || {
    cat "${work_dir}/refine_relaxed.log" >&2
    echo "relaxing RequireReadyGradients did not make the corrupted trace" \
      "admitted, so the refusal in check 4 is not attributable to that" \
      "guard" >&2
    exit 1
  }
  printf 'SINGLE_RANK_REFINEMENT_GUARD_ISOLATION relaxed=RequireReadyGradients result=admitted witness=ObservedTraceIsNotAdmitted exit=%s\n' \
    "${refine_relaxed_status}"
}

case "${half}" in
  abstract) run_abstract_half ;;
  refine) run_refine_half ;;
esac
