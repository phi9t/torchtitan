# 13 — Close the bounded-checking gap

**What to build:** An inductive invariant for `ScoutBModel`, so its safety
claim holds for all schedule lengths rather than up to `MaxIssues`.

**Blocked by:** 11 — DPxTP collective protocol model.

**Status:** ready-for-agent

## The gap, stated plainly

TLC explored 3,671 distinct states at `MaxIssues = 2`. That establishes safety
for schedules of at most two issues per rank. The observed run has 108 per
rank.

The composed claim currently shipped is: the guards are safe for all schedules
up to the bound, **and separately** the observed 108-length schedule satisfies
the per-communicator agreement property. Both halves are true. Their
conjunction is weaker than "the protocol is safe", and a casual reader will
not notice the difference. That is the honesty problem this ticket exists to
fix.

## Approach

Strengthen the safety invariants into an inductive invariant `Inv` such that
`Init => Inv`, `Inv /\ Next => Inv'`, and `Inv => Safety`. Then the claim is
unbounded and no longer depends on `MaxIssues`.

Apalache can check inductiveness directly (`--init`, `--inv`, `--length=1`)
without enumerating reachable states, which is the cheaper route. TLAPS is the
alternative if a machine-checked proof is wanted.

## Acceptance criteria

- [ ] `Inv` is stated explicitly and separately from the reachability-checked
  invariants, so a reader can see which claim rests on which method.
- [ ] Initiation, consecution, and sufficiency are each checked, and each is
  reported as its own result token. A missing one silently weakens the claim.
- [ ] The bound `MaxIssues` no longer appears in the safety claim's statement.
- [ ] The inductive check does not need a fresh GPU run: it is a property of
      the model, not of the trace, so it belongs in the CPU-only formal suite.
- [ ] If an inductive invariant cannot be found, that is recorded as a finding
  with the strengthening attempts made, not left as a silent gap.
- [ ] Apalache, if used, is pinned like every other toolchain: a stable
      release, integrity-checked, with its digest corroborated against what
      upstream publishes. `MODULE.bazel` records why TLA+ 1.7.4 was chosen
      over the rolling 1.8.0 prerelease; the same discipline applies.

## Note

This is the single largest gap between what the artifacts state and what a
reader will assume they state. It should be prioritised above adding further
modelling breadth.
