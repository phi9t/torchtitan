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
