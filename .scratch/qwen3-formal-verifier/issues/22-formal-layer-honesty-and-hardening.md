# 22 — Formal-layer honesty and hardening

**What to build:** Close the non-blocking findings from the ticket 10 review.
Individually small; together they are the difference between a layer that reads
as authoritative and one that is.

**Blocked by:** none. Cheap; do it before 12 so the DPxTP port does not inherit
the `IndexOf` hazard.

**Status:** done

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

- [x] Guard it: `CHOOSE i \in 1..Len(s) : s[i] = value /\ \A j \in 1..i-1 :
  s[j] # value`, plus a `HasKind` precondition at each call site.
- [x] Add TLC's evaluation-error class (`Attempted to compute the value of an
  expression of form`) to `formal_has_infrastructure_error`.
- [x] Correct the false comment on `formal_completed_normally`: it claims TLC
  prints the state summary "only when it finished a state-space search". TLC
  also prints it on an evaluation crash with states still queued.

## 2. Overclaims to retract

- [x] `ScoutAModel.tla` and `formal/README.md` say two nondeterminism sources
  are modelled "because both are real executions". The tracer contradicts
  this: `scout_a.py:1350-1352` raises on
  backward-completed-before-gradient-ready, and `gradient.missing` is produced
  only by the synthetic corruptor. Both branches are executions the
  instrumentation treats as errors. State them as modelled possibilities, not
  observed ones. (The branching floor does not depend on this: dropping the
  missing branch alone still leaves max outdegree 2.)
- [x] The ticket and `README.md:164` claim TLC "prints the alignment of model
  actions to observed events". It does not -- every step of the witness is
  labelled `<ConstrainedNext line 34, col 3 to line 35, col 33 of module
  ScoutARefine>`. The alignment is inferable from the state variables, not
  printed.
- [x] `README.md:162-166` calls the bridge "the bridge that makes runtime
  evidence mean something", but `scout_a.py:409` requires the kinds to equal
  one 10-element literal, so `ScoutARefine` can only ever prove that one
  constant is in the model's language; a deviating run is rejected by the
  Python validator before facts are exported. Say what it proves.
- [x] `formal/README.md` was never updated for `ScoutBModel`,
  `ScoutBIssueOrderInvalid`, or the new invariants; its "bounded abstract
  model" paragraph still describes Scout A only.

## 3. Tests that assert spelling rather than behaviour

`test_model_runner_refuses_a_non_branching_specification` asserts only that
`"max_outdegree"`, `"-ge 2"` and `"replay, not a model"` appear in the runner
text; `test_refinement_negative_is_rejected_for_the_right_reason` likewise greps
file contents. Both behaviours do work -- the reviewer executed them -- but the
tests would keep passing if `-ge 2` were applied to a variable that was never
populated. This is the shape that has burned this project twice.

- [x] Convert both to tests that run the checker against a deliberately
  degenerate input and assert the runner's exit status and message.

## 4. Runner defects

- [x] In `run_tlc_scout_a_model.sh`, the guard `grep -Fq 'Invariant
  ObservedTraceIsNotAdmitted is violated'` in the negative's rejection branch
  is unreachable: if that string were present TLC would have exited 12 and
  `formal_classify_tlc_valid` would already have failed. It looks like a
  second line of defence and is not one.
- [x] In the same runner, the branch printing `observed trace was not admitted
  by the abstract model` is reached when `Observed` is empty -- in which case
  the refinement is *vacuously admitted*, so the message states the opposite
  of what happened. It fails closed, which is the right direction.
- [x] The `mktemp` lint-path file is never removed, so `/project/tmp`
  accumulates one per stage per run -- the same persistence that caused the
  original collision.

## 5. Strengthening worth taking

- [x] Add the one decisive check the Scout A runner lacks: rerun
  `ScoutARefineBad` with `RequireReadyGradients = FALSE` and require
  `ObservedTraceIsNotAdmitted` to be violated. One extra TLC invocation, and
  it proves guard isolation directly instead of assembling it from three
  weaker checks.

### Gate evidence

Full nine-stage Scout B gate, all stages exit 0, sealed and verified.

- Scout B evidence ID:
  `sha256:169a0b7b0d2b32caf5cb17bc0f833587349ff0e52ebd251b18ccfc1cca1e2abe`
- Scout B source ID:
  `sha256:99243482622662a7050352d96ef2988d0e37c9bd388326b7a5924b3aaf1958da`
- Nested Scout A evidence ID:
  `sha256:4dbe499d995033396596df938421d77447770d36bd4a3be0a73bc5a4ed416620`
- New token in the sealed log:
  `SCOUT_A_REFINEMENT_GUARD_ISOLATION relaxed=RequireReadyGradients
  result=admitted witness=ObservedTraceIsNotAdmitted exit=12`
- `owning_pytest`: 250 passed, 1 skipped -- the declared `host perf` skip only.

### A gate break caught before the run, not by it

The new TLC-running contracts skip when the vendored toolchain is absent, and
`owning_pytest` did not mount the formal cache. Under the gate's exact
conditions that produced **nine** skipped tests, which the undeclared-skip
guard -- made fatal on every pytest stage by ticket 18 -- would have failed the
gate on.

Fixed by mounting the cache for `owning_pytest` in both runners rather than
declaring the skips, so the gate executes these contracts instead of recording
their absence. Measured before and after under the gate's conditions:
`72 passed, 9 skipped` became `81 passed, 0 skipped`.

Worth recording as a pattern rather than an incident: ticket 18 made undeclared
skips fatal, and the very next phase tried to introduce nine of them. The guard
worked. What did not work was the report describing it as two tests.
