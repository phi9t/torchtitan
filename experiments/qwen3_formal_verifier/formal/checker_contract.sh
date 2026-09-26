#!/usr/bin/env bash
# Shared result classification for the formal smoke tests.

formal_has_infrastructure_error() {
  local output="$1"
  # A checker that crashed after printing the expected diagnostic must not be
  # classified as a clean result, so JVM stack traces and abort markers count
  # as infrastructure failures wherever they appear in the output. Frames may
  # be module-qualified (`at java.base/java.util.HashMap.get(...)`), so the
  # frame pattern must allow '/' -- omitting it silently matched nothing.
  grep -Eq \
    'Parse Error|Semantic errors:|unexpected token|invalid syntax|unknown declaration|no such file|not found|timed out|Could not find or load main class|Exception in thread|^[[:space:]]*at [a-zA-Z0-9_.$/]+\(|java\.(lang|io|util)\.[A-Za-z]+(Exception|Error)|AbortException|\*\*\* Abort messages|TLC threw an unexpected exception|OutOfMemoryError' \
    <<<"${output}"
}

formal_completed_normally() {
  # TLC prints this summary only when it finished a state-space search.
  # Requiring it stops a truncated or crashed run being read as a completed one.
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
