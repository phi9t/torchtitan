#!/usr/bin/env bash
# Shared result classification for the formal smoke tests.

formal_has_infrastructure_error() {
  local output="$1"
  # A checker that crashed after printing the expected diagnostic must not be
  # classified as a clean result, so JVM stack traces and abort markers count
  # as infrastructure failures wherever they appear in the output. Frames may
  # be module-qualified (`at java.base/java.util.HashMap.get(...)`), so the
  # frame pattern must allow '/' -- omitting it silently matched nothing.
  #
  # `Attempted to compute the value of an expression of form` is TLC's
  # evaluation-error class: an expression could not be evaluated at all, so
  # the search was abandoned rather than completed. A partial CHOOSE applied
  # outside its domain lands here. That output is doubly deceptive -- TLC
  # still prints its state summary and its `Finished in` line, so
  # formal_completed_normally and formal_terminated_normally both return
  # TRUE -- which left the exit-status pin as the only thing rejecting it.
  # Verified against the real checker: an unguarded IndexOf over an absent
  # value exits 75 and prints `1 states generated, 1 distinct states found,
  # 1 states left on queue.` followed by `Finished in 00s`.
  grep -Eq \
    'Parse Error|Semantic errors:|unexpected token|invalid syntax|unknown declaration|no such file|not found|timed out|Could not find or load main class|Exception in thread|^[[:space:]]*at [a-zA-Z0-9_.$/]+\(|java\.(lang|io|util)\.[A-Za-z]+(Exception|Error)|AbortException|\*\*\* Abort messages|TLC threw an unexpected exception|OutOfMemoryError|Attempted to compute the value of an expression of form' \
    <<<"${output}"
}

formal_completed_normally() {
  # TLC prints this summary whenever it stops, including when it abandons the
  # search on an evaluation error -- the summary then reports states still
  # left on queue. It is therefore necessary but NOT sufficient: requiring it
  # stops a truncated run being read as a completed one, while the crash
  # classes in formal_has_infrastructure_error and the exit-status pin are
  # what reject an abandoned one. It used to claim TLC prints this only on a
  # finished search, which is false.
  grep -Eq '[0-9]+ states generated, [0-9]+ distinct states found' <<<"$1"
}

formal_terminated_normally() {
  # An initial-state violation aborts during "Computing initial states...", so
  # TLC never prints a search summary -- verified against the real checker, not
  # assumed. The weaker but still meaningful signal is that TLC reached its own
  # orderly finish line rather than dying mid-run.
  grep -Eq '^Finished in ' <<<"$1"
}

formal_classify_tlc_valid() {
  local status="$1"
  local output="$2"
  [[ "${status}" -eq 0 ]] || return 1
  ! formal_has_infrastructure_error "${output}" || return 1
  formal_completed_normally "${output}" || return 1
  grep -Fq 'Model checking completed. No error has been found.' <<<"${output}"
}

formal_classify_tlc_negative() {
  local status="$1"
  local output="$2"
  local invariant="$3"
  [[ "${status}" -eq 12 ]] || return 1
  ! formal_has_infrastructure_error "${output}" || return 1
  formal_terminated_normally "${output}" || return 1
  [[ "$(grep -c '^Error:' <<<"${output}")" -eq 1 ]] || return 1
  grep -Fxq "Error: Invariant ${invariant} is violated by the initial state:" \
    <<<"${output}"
}

# An invariant that mentions no VARIABLES is a constant expression, and TLC
# reports a false one as "The invariant of X is equal to FALSE" with exit 151 --
# not as an initial-state violation with exit 12, and not through the state
# predicate paths above. Verified against the real checker, not assumed.
# StreamEdgeIsInert is deliberately such an invariant: it reads only the
# constants, so refuting it needs its own classifier rather than being squeezed
# into one that would then accept a weaker outcome for every other negative.
formal_classify_tlc_constant_false() {
  local status="$1"
  local output="$2"
  local invariant="$3"
  [[ "${status}" -eq 151 ]] || return 1
  ! formal_has_infrastructure_error "${output}" || return 1
  formal_terminated_normally "${output}" || return 1
  [[ "$(grep -c '^Error:' <<<"${output}")" -eq 1 ]] || return 1
  grep -Fxq "Error: The invariant of ${invariant} is equal to FALSE" \
    <<<"${output}"
}

formal_classify_tlc_transition_negative() {
  local status="$1"
  local output="$2"
  local invariant="$3"
  [[ "${status}" -eq 12 ]] || return 1
  ! formal_has_infrastructure_error "${output}" || return 1
  local invariant_line="Error: Invariant ${invariant} is violated."
  formal_completed_normally "${output}" || return 1
  [[ "$(grep -Fxc "${invariant_line}" <<<"${output}")" -eq 1 ]] \
    || return 1
  ! grep '^Error:' <<<"${output}" \
    | grep -Fvx "${invariant_line}" \
    | grep -Fvx 'Error: The behavior up to this point is:' \
    | grep -q .
}

# A refuted TEMPORAL property is a different outcome again, and neither of the
# state-predicate classifiers above matches it. Shapes below are copied from
# real runs of ScoutBModelLiveDivergent.cfg and
# ScoutBModelLiveUnconditional.cfg, not authored:
#
#   exit 13, not 12 or 151;
#   TWO '^Error:' lines -- 'Temporal properties were violated.' and 'The
#   following behavior constitutes a counter-example:' -- so
#   formal_classify_tlc_transition_negative, which requires exactly one,
#   rejects this output outright;
#   no 'Invariant X is violated' line at all, since no invariant was refuted;
#   the counter-example is a LASSO: a finite prefix closed either by a
#   'State N: Stuttering' line, when the behaviour reaches a state with no
#   successors, or by a 'Back to state N' line, when it closes into a cycle.
#   Both are accepted; a trace with neither is a truncated report, not a
#   lasso, and is rejected.
#
# TLC does not name the property it refuted, so this cannot check the name.
# The binding between a stage and its property lives in the cfg instead, and
# formal_cfg_declares_one_property is what asserts it.
formal_classify_tlc_liveness_negative() {
  local status="$1"
  local output="$2"
  [[ "${status}" -eq 13 ]] || return 1
  ! formal_has_infrastructure_error "${output}" || return 1
  formal_terminated_normally "${output}" || return 1
  formal_completed_normally "${output}" || return 1
  grep -Fxq 'Error: Temporal properties were violated.' <<<"${output}" \
    || return 1
  grep -Fxq 'Error: The following behavior constitutes a counter-example:' \
    <<<"${output}" || return 1
  # Nothing else on an Error line: another diagnostic means another failure.
  ! grep '^Error:' <<<"${output}" \
    | grep -Fvx 'Error: Temporal properties were violated.' \
    | grep -Fvx 'Error: The following behavior constitutes a counter-example:' \
    | grep -q . || return 1
  grep -Eq '^(State [0-9]+: Stuttering|Back to state [0-9]+)' <<<"${output}"
}

# The temporal properties a cfg declares, one per line, in order. Parsed out of
# the PROPERTY/PROPERTIES block rather than substring-matched, so a commented
# mention proves nothing and a second property cannot hide on a block line.
formal_cfg_temporal_properties() {
  awk '
    function is_keyword(word) {
      return word == "CONSTANT" || word == "CONSTANTS" \
          || word == "SPECIFICATION" || word == "INVARIANT" \
          || word == "INVARIANTS" || word == "CONSTRAINT" \
          || word == "CONSTRAINTS" || word == "ACTION_CONSTRAINT" \
          || word == "CHECK_DEADLOCK" \
          || word == "SYMMETRY" || word == "VIEW" || word == "INIT" \
          || word == "NEXT" || word == "ALIAS" || word == "POSTCONDITION"
    }
    { sub(/^[ \t]+/, ""); sub(/[ \t]+$/, "") }
    $0 == "" { next }
    /^\\\*/ { next }
    {
      if ($1 == "PROPERTY" || $1 == "PROPERTIES") {
        collecting = 1
        for (i = 2; i <= NF; i++) print $i
        next
      }
      if (is_keyword($1)) { collecting = 0; next }
      if (collecting) print $1
    }
  ' "$1"
}

# TLC's liveness diagnostic names no property, so a liveness stage wired to the
# wrong cfg would still classify. This is what pins the stage to its property.
formal_cfg_declares_one_property() {
  local cfg_file="$1"
  local expected="$2"
  [[ -f "${cfg_file}" ]] || return 1
  [[ "$(formal_cfg_temporal_properties "${cfg_file}")" == "${expected}" ]]
}
formal_classify_lean_valid() {
  local status="$1"
  local output="$2"
  local theorem="$3"
  [[ "${status}" -eq 0 ]] || return 1
  ! formal_has_infrastructure_error "${output}" || return 1
  grep -Fxq "'${theorem}' does not depend on any axioms" <<<"${output}" \
    || return 1
  ! grep -Eq 'sorryAx|depends on (any )?axioms' <<<"${output}"
}

formal_classify_lean_negative() {
  local status="$1"
  local output="$2"
  local proposition="$3"
  local expected_diagnostic
  [[ "${status}" -eq 1 ]] || return 1
  ! formal_has_infrastructure_error "${output}" || return 1
  [[ "$(grep -c 'error:' <<<"${output}")" -eq 1 ]] || return 1
  printf -v expected_diagnostic \
    'Tactic `decide` proved that the proposition\n  %s\nis false' \
    "${proposition}"
  [[ "${output}" == *"${expected_diagnostic}"* ]]
}
