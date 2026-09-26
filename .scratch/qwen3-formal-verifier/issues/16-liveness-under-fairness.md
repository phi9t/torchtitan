# 16 — Liveness under fairness

**What to build:** A temporal liveness property for `ScoutBModel`: every
issued collective eventually completes, under weak fairness.

**Blocked by:** 11 — DPxTP collective protocol model.

**Status:** ready-for-agent

## Why safety is not enough here

`DeadlockFreedom` is a safety invariant: no reachable state is stuck. That
rules out a terminal deadlock, but not a livelock in which the system keeps
taking steps while some collective never starts. For a step model whose whole
purpose is to reason about jobs that hang, "the job always has something to
do" is not the property anyone cares about; "the step finishes" is.

## Shape

`Spec == Init /\ [][Next]_vars /\ WF_vars(...)` with fairness on `Start` and
`Complete`, then

```tla
EveryIssuedCollectiveCompletes == \A c \in CommIds : ...~> doneOn[c] = ...
```

`Issue` must **not** be fair -- it is unconstrained by design, and that is the
source of rank skew. Fair `Issue` would let a rank run ahead forever and the
property would be about an artifact of the fairness assumption.

## Acceptance criteria

- [ ] The liveness property is checked as its own run, reported with its own
      result token. Liveness checking is much slower than safety, so it likely
      needs a smaller bound than the safety runs -- report the bound actually
      achieved rather than implying the safety bound carried over.
- [ ] Fairness is on `Start`/`Complete` only, with the reason for excluding
  `Issue` in the module header.
- [ ] A negative: with `RequireUniformCommSchedule = FALSE`, TLC produces a
      counterexample trace to liveness. Liveness counterexamples are lassos,
      so `formal_classify_tlc_transition_negative` -- which expects exactly
      one `^Error:` line and a state-space summary -- almost certainly does
      not classify this correctly. Verify the real TLC output shape and add a
      classifier for it; **do not** hand-author the fixture.
- [ ] That last point is the standing lesson from this project: a check
      verified against hand-authored fixtures confirms the author's
      assumptions instead of testing the system. Fixtures come from the tool
      whose output they imitate.
