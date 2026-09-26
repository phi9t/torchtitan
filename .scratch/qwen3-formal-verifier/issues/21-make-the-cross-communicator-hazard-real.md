# 21 — Make the cross-communicator hazard real in ScoutBModel

**What to build:** A checked configuration in which order divergence across
overlapping communicators can actually deadlock the model, so
`DeadlockFreedom` states the property the model was built to state.

**Blocked by:** none. **Blocks:** 12 — DPxTP refinement bridge.

**Status:** ready-for-agent

## What is wrong

Three claims attached to `ScoutBModel` are false. All three were falsified by
running TLC, and the first was reconfirmed independently:

1. **Streams are inert.** Neutralising the stream-head conjunct in
   `MemberReady` gives a bit-identical state graph:

   | run | shipped | streams removed |
   |---|---|---|
   | safety | `12142 gen, 3671 distinct`, outdeg 12 | **identical** |
   | divergent | `124620 / 35915 / 18308`, deadlock | **identical** |

   The cause is structural, not accidental. `StreamOfIssue` sends fsdp
   reduce-scatter to `rs`, other fsdp to `ag`, and everything else to
   `compute`. In `CommIdsSmall` -- the only instance any cfg binds -- each rank
   has exactly one tp communicator and one fsdp communicator, so two issues by
   one rank share a stream **only when they share a communicator**, where
   per-communicator FIFO already orders them. `AtStreamHead` is exactly
   redundant by construction.

2. **The wait cycle is in no checked configuration.** Every cfg binds
   `CommIds <- CommIdsSmall` (four communicators). `CommIdsWide`,
   `CommMembersWide` and `CommClassWide` -- the instance that actually has
   three communicators over `{0,2}` -- are dead definitions, referenced only by
   the module itself and by a test that counts `"{0, 2}"` occurrences in text no
   checker loads. Run against the wide instance, safety still passes with no
   deadlock (`21833 / 6549`), because there are no cross-communicator wait edges
   at all. `DeadlockFreedom`'s comment names a property the model cannot
   express.

3. **The divergent negative is refuted by the wrong mechanism.** Its witness is
   rank 1 issuing `fsdp13/all_gather` where rank 3 issues
   `fsdp13/reduce_scatter` -- an intra-communicator *operation* mismatch, the
   same mechanism as the unguarded negative, and the very thing ticket 11 calls
   unrealizable ("a single program cannot diverge that way"). Order divergence,
   which is what hangs real jobs, cannot deadlock this model: with the
   `CommClass` conjunct dropped but op agreement kept, TLC reports no violation
   and an identical state count.

So the safety result is real and the machinery reproduces, but the hazard the
model exists to reason about is absent from it.

## The repair

On ranks 0 and 2 the real run has `mesh_batch` and `mesh_loss` as *separate*
communicators over the same pair, both carrying `all_reduce`, and therefore both
on the compute stream. That is the wait edge: rank 0 issuing batch-then-loss
while rank 2 issues loss-then-batch leaves each at a stream head waiting for a
collective the other has not reached. Four-cycle, no op mismatch anywhere.

Making that configuration the checked one is the fix. Streams become
load-bearing because two distinct communicators finally share a stream.

## Deeper root cause, found while scoping this ticket

The review said the hazard is absent. It is worse: `DeadlockFreedom` cannot
distinguish a hazard from an artifact of the bound. Establish these four facts
before designing the repair -- each was produced by running TLC, and the last
one invalidates the obvious probe.

**1. `AllDone` makes budget exhaustion look like deadlock.**

```
AllDone ==
  /\ running = {}
  /\ \A r \in Ranks : Len(issued[r]) = MaxIssues
  /\ \A c \in CommIds : \A r \in CommMembers[c] : CommCount(r, c) = doneOn[c]
```

The last conjunct requires every member of a communicator to have issued the
same *count*. So if rank 0 issues `fsdp02` twice and rank 2 issues it once, no
value of `doneOn` satisfies it and `AllDone` is false forever. Once both ranks
also reach `MaxIssues`, nothing can issue and nothing can start: `Stuck` holds,
`AllDone` does not, and `DeadlockFreedom` is violated. That is a finite-budget
artifact, not a circular wait. `Stuck => AllDone` is simply too strong here.

**2. `UniformProgramScheduleOK` compares communicator class, not identity.**

It requires `CommClass[...]` and `.op` to agree at each position, so two ranks
issuing `[tp, fsdp]` satisfy it while choosing *different* tp and fsdp
communicators. It is weaker than "the same program", and it is exactly the
cross-communicator reordering freedom that ought to be constrained.

**3. The Small instance passes safety for an accidental reason.** Each rank
belongs to exactly one communicator per class -- rank 0 to `tp01`, `fsdp02` --
so the class-only guard forces the choice. Safety holds because of the instance
shape, not because the guard is strong. `CommIdsWide` preserves this property
(`tp01`, `fsdp02`, `loss02` are one per class), which is why widening alone
changes nothing.

**4. Do not use a uniform-class override as the discriminating probe.**
Setting every communicator to one class makes operation mismatch impossible,
which looks like the clean way to isolate order divergence. It is not:
collapsing the classes also makes `UniformProgramScheduleOK` vacuous and lets
each rank pick freely among all its communicators, so the resulting violation
is finding 1's count mismatch, not a stream-ordered circular wait. Measured:
Wide + uniform class + schedule *required* still violates `DeadlockFreedom`
(`110899 generated, 30935 distinct`), and it violates it with streams
neutralised too, which is the tell.

## What the repair therefore has to do

- [ ] Split the claim. `DeadlockFreedom` must distinguish a genuine circular
  wait -- a communicator every member has pending that still cannot start --
  from budget exhaustion and from a never-issued communicator. State the
  circular-wait property directly rather than as `Stuck => AllDone`.
- [ ] Decide what the schedule guard is for and make it say that. If it is meant
  to model one SPMD program, it must agree on communicator identity wherever
  member sets coincide, not merely on class.
- [ ] Build an instance where a rank holds **two communicators of the same
  class** so the guard is non-trivial and same-class reordering is
  expressible. No current instance has this, which is why no current
  configuration can exhibit the hazard.
- [ ] Only then is the discriminating test meaningful: op agreement required,
  membership complete, order free -- and a deadlock found. Re-derive it; do not
  reuse the uniform-class probe.
- [ ] Recheck whether streams are load-bearing *after* the above. They were
  inert in every configuration measured so far, including Wide, so the claim
  must be re-earned or dropped rather than assumed to return.

## Acceptance criteria

- [ ] The safety configuration binds an instance with at least two distinct
  same-member communicators on one stream. `CommIdsWide` exists for this; either
  use it or delete it.
- [ ] **The discriminating test:** with op agreement required but the
  uniform-schedule guard relaxed so only *order* may diverge, TLC finds a
  `DeadlockFreedom` violation. Today that probe reports no violation and an
  identical state count, which is the precise statement of the defect. This
  criterion is what separates a repaired model from the current one; it is not
  optional.
- [ ] Neutralising the stream-head conjunct now **changes** the result. Assert
  it as a test that runs TLC, not one that greps for the identifier
  `AtStreamHead`. The existing test greps, and cannot tell.
- [ ] The divergent negative's witness is an order divergence, not an
  operation mismatch. Print or assert the witness shape so the two negatives are
  distinguishable.
- [ ] Any conjunct shown to be inert is removed or justified. The `CommClass`
  conjunct is provably inert -- `CommOps` already ties operation to class -- so
  the test asserting `"CommClass" in schedule` guards nothing.
- [ ] `MaxSkew` binds. At `MaxIssues = 2` the skew guard reduces to a tautology,
  so a reader of the cfg believes skew is bounded to 2 when it is unbounded.
  Either raise `MaxIssues` until skew is reachable or drop the constant.
- [ ] Correct the claim inventory: the module header, ticket 11's criteria, the
  commit-message assertions, and the runner comment claiming safety "across
  EVERY admitted interleaving of four ranks" must match what is checked.
- [ ] Say how many communicator instances there are consistently. The commit
  message says eight, the ticket says four, the cfgs bind four.
- [ ] Report the bound where the result is read: the evidence token
  (`SCOUT_B_MODEL_SAFETY ... distinct_states=3671`) records no bound, and
  the module header says "verified against the observed 2x2 execution". The
  `MaxIssues = 2` versus observed-108 composition currently appears only in the
  ticket and the commit message.
- [ ] Fix the SYMMETRY rationale. The stated reason -- that `Permutations`
  cannot express a consistent joint permutation -- is wrong; `SYMMETRY` accepts
  any defined permutation set. The real obstacle is that symmetry requires model
  values, and `Ranks` are naturals while `CommIds` are strings. The "do not add
  it" instruction is right for a reason the comment does not give.
