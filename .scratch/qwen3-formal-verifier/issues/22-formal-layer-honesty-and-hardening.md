# 22 — Formal-layer honesty and hardening

**What to build:** Close the non-blocking findings from the ticket 10 review.
Individually small; together they are the difference between a layer that reads
as authoritative and one that is.

**Blocked by:** none. Cheap; do it before 12 so the DPxTP port does not inherit
the `IndexOf` hazard.

**Status:** ready-for-agent

Locations are described by content, not line number: ticket 19's target split
shifted the line numbers in `run_tlc_scout_a_model.sh`.

## 1. `IndexOf` is unguarded, and the classifiers do not catch its crash

`ScoutLifecycle.tla` defines `IndexOf == CHOOSE index \in 1..Len(s) :
s[index] = value` with no guard on either edge case.

- **Absent value:** real TLC output is `Error: Attempted to compute the value of
  an expression of form CHOOSE x \in S: P, but no element of S satisfied P.`,
  exit 75. That log *also* contains `1 states generated, 1 distinct states
  found, 1 states left on queue.` and `Finished in 00s`. Running the shipped
  classifiers on it: `formal_has_infrastructure_error -> FALSE`,
  `formal_completed_normally -> TRUE`, `formal_terminated_normally -> TRUE`.
  Only the exit-status pin rejects it, so the defence is single-layer where the
  commit message advertises layered classification.
- **Duplicates:** `IndexOf(<<"a","b","a","c">>, "a") = 1` because TLC returns
  the first match. That is TLC implementation behaviour, not TLA+ semantics --
  `CHOOSE` is deterministic but unspecified. `BeforeIfPresent` agreeing with
  the independent Lean `occursBeforeIfPresent` silently depends on it, which
  undercuts the point of having two implementations.
- Neither case is reachable today only because `scout_a.py:409` pins the kind
  list. The safety comes from Python, not from the module. **Ticket 12's DPxTP
  port makes duplicates real**, since collective kinds repeat per rank.

- [ ] Guard it: `CHOOSE i \in 1..Len(s) : s[i] = value /\ \A j \in 1..i-1 :
  s[j] # value`, plus a `HasKind` precondition at each call site.
- [ ] Add TLC's evaluation-error class (`Attempted to compute the value of an
  expression of form`) to `formal_has_infrastructure_error`.
- [ ] Correct the false comment on `formal_completed_normally`: it claims TLC
  prints the state summary "only when it finished a state-space search". TLC
  also prints it on an evaluation crash with states still queued.

## 2. Overclaims to retract

- [ ] `ScoutAModel.tla` and `formal/README.md` say two nondeterminism sources
  are modelled "because both are real executions". The tracer contradicts
  this: `scout_a.py:1350-1352` raises on
  backward-completed-before-gradient-ready, and `gradient.missing` is produced
  only by the synthetic corruptor. Both branches are executions the
  instrumentation treats as errors. State them as modelled possibilities, not
  observed ones. (The branching floor does not depend on this: dropping the
  missing branch alone still leaves max outdegree 2.)
- [ ] The ticket and `README.md:164` claim TLC "prints the alignment of model
  actions to observed events". It does not -- every step of the witness is
  labelled `<ConstrainedNext line 34, col 3 to line 35, col 33 of module
  ScoutARefine>`. The alignment is inferable from the state variables, not
  printed.
- [ ] `README.md:162-166` calls the bridge "the bridge that makes runtime
  evidence mean something", but `scout_a.py:409` requires the kinds to equal
  one 10-element literal, so `ScoutARefine` can only ever prove that one
  constant is in the model's language; a deviating run is rejected by the
  Python validator before facts are exported. Say what it proves.
- [ ] `formal/README.md` was never updated for `ScoutBModel`,
  `ScoutBIssueOrderInvalid`, or the new invariants; its "bounded abstract
  model" paragraph still describes Scout A only.

## 3. Tests that assert spelling rather than behaviour

`test_model_runner_refuses_a_non_branching_specification` asserts only that
`"max_outdegree"`, `"-ge 2"` and `"replay, not a model"` appear in the runner
text; `test_refinement_negative_is_rejected_for_the_right_reason` likewise greps
file contents. Both behaviours do work -- the reviewer executed them -- but the
tests would keep passing if `-ge 2` were applied to a variable that was never
populated. This is the shape that has burned this project twice.

- [ ] Convert both to tests that run the checker against a deliberately
  degenerate input and assert the runner's exit status and message.

## 4. Runner defects

- [ ] In `run_tlc_scout_a_model.sh`, the guard `grep -Fq 'Invariant
  ObservedTraceIsNotAdmitted is violated'` in the negative's rejection branch
  is unreachable: if that string were present TLC would have exited 12 and
  `formal_classify_tlc_valid` would already have failed. It looks like a
  second line of defence and is not one.
- [ ] In the same runner, the branch printing `observed trace was not admitted
  by the abstract model` is reached when `Observed` is empty -- in which case
  the refinement is *vacuously admitted*, so the message states the opposite
  of what happened. It fails closed, which is the right direction.
- [ ] The `mktemp` lint-path file is never removed, so `/project/tmp`
  accumulates one per stage per run -- the same persistence that caused the
  original collision.

## 5. Strengthening worth taking

- [ ] Add the one decisive check the Scout A runner lacks: rerun
  `ScoutARefineBad` with `RequireReadyGradients = FALSE` and require
  `ObservedTraceIsNotAdmitted` to be violated. One extra TLC invocation, and
  it proves guard isolation directly instead of assembling it from three
  weaker checks.
