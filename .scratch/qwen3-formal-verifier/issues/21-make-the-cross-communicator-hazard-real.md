# 21 — Make the cross-communicator hazard real in ScoutBModel

**What to build:** A checked configuration in which order divergence across
overlapping communicators can actually deadlock the model, so
`DeadlockFreedom` states the property the model was built to state.

**Blocked by:** none. **Blocks:** 12 — DPxTP refinement bridge.

**Status:** resolved -- reviewed twice, the second a dedicated soundness review
(SOUND WITH CAVEATS), gated in `4f252960d`

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

- [x] Split the claim. `DeadlockFreedom` must distinguish a genuine circular
  wait -- a communicator every member has pending that still cannot start --
  from budget exhaustion and from a never-issued communicator. State the
  circular-wait property directly rather than as `Stuck => AllDone`.
- [x] Decide what the schedule guard is for and make it say that. If it is meant
  to model one SPMD program, it must agree on communicator identity wherever
  member sets coincide, not merely on class.
- [x] Build an instance where a rank holds **two communicators of the same
  class** so the guard is non-trivial and same-class reordering is
  expressible. No current instance has this, which is why no current
  configuration can exhibit the hazard.
- [x] Only then is the discriminating test meaningful: op agreement required,
  membership complete, order free -- and a deadlock found. Re-derive it; do not
  reuse the uniform-class probe.
- [x] Recheck whether streams are load-bearing *after* the above. They were
  inert in every configuration measured so far, including Wide, so the claim
  must be re-earned or dropped rather than assumed to return.

## Acceptance criteria

- [x] The safety configuration binds an instance with at least two distinct
  same-member communicators on one stream. `CommIdsWide` exists for this; either
  use it or delete it.
- [x] **The discriminating test:** with op agreement required but the
  uniform-schedule guard relaxed so only *order* may diverge, TLC finds a
  `DeadlockFreedom` violation. Today that probe reports no violation and an
  identical state count, which is the precise statement of the defect. This
  criterion is what separates a repaired model from the current one; it is not
  optional.
- [x] Neutralising the stream-head conjunct now **changes** the result. Assert
  it as a test that runs TLC, not one that greps for the identifier
  `AtStreamHead`. The existing test greps, and cannot tell.
- [x] The divergent negative's witness is an order divergence, not an
  operation mismatch. Print or assert the witness shape so the two negatives are
  distinguishable.
- [x] Any conjunct shown to be inert is removed or justified. The `CommClass`
  conjunct is provably inert -- `CommOps` already ties operation to class -- so
  the test asserting `"CommClass" in schedule` guards nothing.
- [x] `MaxSkew` binds. At `MaxIssues = 2` the skew guard reduces to a tautology,
  so a reader of the cfg believes skew is bounded to 2 when it is unbounded.
  Either raise `MaxIssues` until skew is reachable or drop the constant.
- [x] Correct the claim inventory: the module header, ticket 11's criteria, the
  commit-message assertions, and the runner comment claiming safety "across
  EVERY admitted interleaving of four ranks" must match what is checked.
- [x] Say how many communicator instances there are consistently. The commit
  message says eight, the ticket says four, the cfgs bind four.
- [x] Report the bound where the result is read: the evidence token
  (`SCOUT_B_MODEL_SAFETY ... distinct_states=3671`) records no bound, and
  the module header says "verified against the observed 2x2 execution". The
  `MaxIssues = 2` versus observed-108 composition currently appears only in the
  ticket and the commit message.
- [x] Fix the SYMMETRY rationale. The stated reason -- that `Permutations`
  cannot express a consistent joint permutation -- is wrong; `SYMMETRY` accepts
  any defined permutation set. The real obstacle is that symmetry requires model
  values, and `Ranks` are naturals while `CommIds` are strings. The "do not add
  it" instruction is right for a reason the comment does not give.

## Repair, as landed

Every number below is TLC's own, from `--suite tier0` on the repaired module.

### What changed in the model

- **`DeadlockFreedom` is now the circular-wait property itself**, not
  `Stuck => AllDone`. Stuck states are classified into three disjoint,
  exhaustive classes: `StuckByCircularWait` (some communicator has every member
  waiting at the same per-communicator position, the members agree on the
  operation, issuance is balanced everywhere, and it still cannot start),
  `StuckByOpMismatch`, and `StuckByBudget` (unequal per-communicator issue
  counts, or nobody blocked on a peer at all). The old statement survives under
  its accurate name `StuckImpliesAllDone` and is still checked in the safety
  configuration, where it holds; nothing was dropped to make something pass.
- **`IssuanceBalanced` is the load-bearing conjunct of the hazard class.**
  Without it, a first TLC witness was accepted whose blocking chain ran through
  a communicator one member had simply not reached yet -- resolvable by a larger
  budget, and therefore the very artifact this ticket is about. With it, every
  rendezvous in the state is fully issued, so the blocking chain can only close
  into a cycle.
- **The schedule guard was split in two and now says "one SPMD program".**
  `RequireUniformProgramOps` fixes the operation at every position.
  `RequireUniformProgramComms` fixes communicator IDENTITY wherever two ranks'
  communicators share a member -- two distinct communicators that share a rank
  are two different program sites. Disjoint communicators stay unconstrained, so
  rank 0 still uses `fsdp02` where rank 1 uses `fsdp13`. Both halves were
  checked against the observed run: over all 108 positions of all four ranks,
  zero operation-agreement violations and zero site violations.
- **The `CommClass` constant is gone.** `CommOps` is now supplied per
  communicator, which the observed run requires (`mesh_fsdp` carries
  `all_reduce` as well as the two directional collectives), and `StreamOfIssue`
  keys on the operation, which is how PyTorch actually behaves. Measured: adding
  a four-way role-agreement conjunct back changes no verdict and cuts the
  reachable graph from 38,321 to 13,583 distinct states, so it removed
  behaviours rather than hazards.
- **`MaxSkew` is gone.** At `MaxIssues = 2` its guard was a tautology. Dropped
  rather than tightened, because a real skew bound would have cut interleavings
  out of the safety claim. Issue skew is unbounded within `MaxIssues`, and the
  safety token now says so.
- **One instance, eight communicators**, one per observed `canonical_id`:
  `tp01 tp23 fsdp02 fsdp13 batch02 batch13 loss02 loss13`. `CommIdsSmall` and
  `CommIdsWide` are deleted. Every rank holds four distinct communicators on the
  compute stream, matching the observed run exactly, so the stream-head conjunct
  is no longer redundant with per-communicator FIFO.

### Results

```text
SCOUT_B_MODEL_SAFETY     result=success distinct_states=38321 max_outdegree=24
                         bound_max_issues_per_rank=2 bound_ranks=4
                         bound_communicators=8 bound_issue_skew=unbounded
SCOUT_B_MODEL_NONVACUOUS result=completion_reachable
SCOUT_B_MODEL_DIVERGENT  invariant=DeadlockFreedom result=named_violation
SCOUT_B_MODEL_WITNESS    invariant=NoCrossCommunicatorCycleWitness
                         result=named_violation
SCOUT_B_MODEL_OPMISMATCH invariant=NoOpMismatchHang result=named_violation
SCOUT_B_MODEL_STREAM_EDGE invariant=StreamEdgeIsInert result=named_violation
SCOUT_B_MODEL_STREAM_EDGE_MUTANT substitution=StreamOfIssue_is_comm
                         invariant=StreamEdgeIsInert result=holds
SCOUT_B_MODEL_UNGUARDED  invariant=RendezvousOpAgreement result=named_violation
```

### The stream claim, restated after review

The first repair rechecked the divergent configuration with
`RequireStreamOrder = FALSE`, found `DeadlockFreedom` satisfied over 1,146,243
distinct states, and read that as earning the claim. **It does not.** With the
edge deleted, `AtStreamHead` is `TRUE`, so `MemberReady(c, r)` reduces to
`CommCount(r, c) >= Front(c)` and `StartAllowed(c)` to
`FullyPending(c) /\ OpsAgreeAtFront(c)` -- which `Stuck` already denies for every
`c`. `StuckByCircularWait` is therefore unsatisfiable by construction, at any
bound on any instance, and that exhaustive search proved a two-line lemma.
Corroborated twice by the reviewer: deleting the balance conjunct changed nothing
there, and `StuckImpliesAllDone` WAS violated under that cfg, so removing the
edge removed the classification rather than the stuckness.

`ScoutBModelStreams.cfg` and its stage are deleted. The lemma is stated in the
module header instead, and what is checked is its hypothesis -- that the shipped
stream map is not refined by communicator identity on this instance.
`StreamEdgeIsInert` must be false here, and must hold once `StreamOfIssue(e)` is
replaced by `e.comm`, which is what the original module effectively had. Both
halves are TLC runs and both are instant, because the cfg is pinned to the
initial state.

`StreamEdgeIsInert` mentions no variables, so TLC refutes it as a false constant
expression (exit 151, `The invariant of X is equal to FALSE`) rather than as an
initial-state violation. That outcome gets its own classifier,
`formal_classify_tlc_constant_false`, with its own focused test; the
state-predicate classifiers reject it, which the test also checks.

### The witness

`DeadlockFreedom was violated` says nothing about WHICH stuck state was found, so
the shape is now a checked property rather than a sentence in a report.
`NoCrossCommunicatorCycleWitness` is violated only by a state where the cycle
spans at least two distinct communicators, the operations agree, issue counts are
equal on every communicator of the chain, and every rank has spent its whole
budget. Its cfg differs from the divergent cfg in the invariant and nothing else,
which the runner verifies.

This is the guard that would have caught the balance defect below.

### F1: the balance conjunct was quantified over the wrong set

`IssuanceBalanced` ranged over all of `CommIds`. A wait cycle is local to its
blocking chain, so any unrelated imbalance anywhere in the state reclassified a
real deadlock as `StuckByBudget`. The header sentence claiming the conjunct is
"what makes it independent of MaxIssues" was false in the other direction: it
bought freedom from false positives with false negatives.

The reviewer's witness, at the shipped `MaxIssues = 2` under the shipped
`ScoutBModelDivergent.cfg`:

```text
rank 0: tp01    all_reduce, tp01    all_reduce
rank 1: fsdp13  all_reduce, batch13 all_reduce
rank 2: tp23    all_reduce, tp23    all_reduce
rank 3: batch13 all_reduce, fsdp13  all_reduce
```

Ranks 1 and 3 are a two-cycle: `fsdp13` needs rank 3, whose `fsdp13` sits behind
`batch13`; `batch13` needs rank 1, whose `batch13` sits behind `fsdp13`. Both
fully pending, operations agreeing, same stream, no missing issue, hung at any
budget. The global conjunct discarded it because rank 0 issued `tp01` twice and
rank 1 not at all.

The fix imposes the condition on the blocking closure instead. `BlockedBy(c)`
collects the communicators of the uncompleted earlier same-stream issues of c's
members; `BlockingClosure(c)` is its least fixpoint; `CircularWaitAt(c)` requires
every communicator in that closure to be fully pending with agreeing operations.
Each closure element then has a blocker inside the closure, the closure is
finite, so the chain must close into a cycle -- and extra budget cannot help,
because a new issue is appended after the blocker.

Note this is deliberately WEAKER than count equality on the closure: two members
of a chain communicator may have issued it a different number of times and the
cycle is still a cycle. Count equality is asserted of the witness instead, where
it is a statement about the counterexample rather than a condition on the hazard.

Evaluated on the reviewer's witness pinned as an initial state:

```text
<<"Stuck", TRUE>>
<<"StuckByCircularWait", TRUE>>
<<"StuckByOpMismatch", FALSE>>
<<"StuckByBudget", FALSE>>
<<"AllDone", FALSE>>
<<"globalBalance", FALSE>>
<<"closure(fsdp13)", {"fsdp13", "batch13"}>>
<<"closure(batch13)", {"fsdp13", "batch13"}>>
<<"chainBalanced(fsdp13)", TRUE>>
<<"CircularWaitAt(fsdp13)", TRUE>>
<<"witnessShapeHolds", TRUE>>
```

`globalBalance` is FALSE, which is exactly why the old definition filed it as
`StuckByBudget`; the closure is the two-cycle and nothing else.

### Guards that were passing vacuously

The review found five mutations that the first repair's tests accepted. All five
now fail, verified by applying each one and rerunning:

| mutation | before | after |
|---|---|---|
| `StreamOfIssue(e) == e.comm` | 9 passed | 1 failed |
| `AtStreamHead` deleted from `MemberReady` | 9 passed | 1 failed |
| `all_reduce` routed to a per-communicator stream | 9 passed | 1 failed |
| state-wide `IssuanceBalanced` restored | n/a | 1 failed |
| `StuckImpliesAllDone` dropped from `INVARIANTS` | 10 passed | 1 failed |

The stream test claimed to reconstruct `StreamOfIssue` and did not: it collected
every communicator a rank belonged to and never applied the stream map. It now
parses `CommMembers2x2`, `CommOps2x2` (resolving the `Ops` alias and the `OTHER`
arm) and `StreamOfIssue` from the module, derives the per-rank per-stream
communicator sets, and requires one rank to hold two distinct communicators on
one stream. The `AtStreamHead in MemberReady` assertion, dropped in the first
repair with nothing replacing it, is back. `_cfg_invariants` parses the
`INVARIANT`/`INVARIANTS` block, so a `\*` comment naming an invariant no longer
satisfies a guard, and every negative cfg is required to name exactly one.

### Accuracy corrections

- The "190262 distinct to the witness" figure is gone. It was search progress at
  the moment of the first violation, not a property of anything: it moves with
  the worker count (the reviewer measured 137920 at 4 and 106229 at 8).
- The header sentence that ran the observed-run measurement together with the
  four-communicators-on-compute statement is split. Zero operation-agreement and
  zero `SameSite` violations over 108 positions is a measurement against
  recorded ids, members and operations. "Four distinct communicators per rank on
  the compute stream" is the design constant applied to the trace's operation
  labels, and is now labelled as such.
- Every stage runs at one worker under a 120s timeout. The slowest is about 35s,
  so the headroom is roughly 3.4x and the state counts are reproducible.
  `timeout = "long"` on the Bazel target is gone with the 1.1M-state stage.
- On the role-agreement conjunct the ticket's "provably inert" claim was right
  about the OLD module, where `CommOps` derived operations from the class, and
  wrong about the repaired one, where `CommOps` is per communicator and gives
  `mesh_fsdp` `all_reduce` too. The header now says that, and says the conjunct
  is dropped because what it removes are not hazards (38321 -> 13583 distinct
  states), not because it removes nothing.

### Cost

Tier 0 is minutes, not seconds: the Scout B model target is seven TLC searches
over eight communicators. The stale "in seconds" text in `run_formal_tier0.sh`,
`formal/BUILD.bazel`, `formal/README.md` and
`experiments/qwen3_formal_verifier/README.md` is corrected, and the seven checks
are listed where the tier-0 targets are.

### Gate evidence (shared with ticket 18)

Full nine-stage Scout B gate, all stages exit 0, sealed and verified.

- Scout B evidence ID:
  `sha256:b24ddf9593af32c96dcd8c8588f9845aaad37a9473e2e140fc74b63be98c986f`
- Scout B source ID:
  `sha256:e179d91bb4bdb62ae01cb56221289f19761a0ec8406312b054c99510afff7c12`
- Nested Scout A evidence ID:
  `sha256:439517d3a89579fe7dcd34b4c5e5d26d625bfd7812541403c3ea0c9aa2f244fc`
- Source manifest: 26 verified paths, 7 process paths.

Result tokens present in the sealed formal log:

```
SCOUT_B_MODEL_SAFETY     distinct_states=38321 max_outdegree=24
                         bound_max_issues_per_rank=2 bound_ranks=4
                         bound_communicators=8 bound_issue_skew=unbounded
SCOUT_B_MODEL_NONVACUOUS  completion_reachable
SCOUT_B_MODEL_DIVERGENT   DeadlockFreedom                  named_violation
SCOUT_B_MODEL_WITNESS     NoCrossCommunicatorCycleWitness  named_violation
SCOUT_B_MODEL_OPMISMATCH  NoOpMismatchHang                 named_violation
SCOUT_B_MODEL_UNGUARDED   RendezvousOpAgreement            named_violation
SCOUT_B_MODEL_STREAM_EDGE        StreamEdgeIsInert  named_violation exit=151
SCOUT_B_MODEL_STREAM_EDGE_MUTANT StreamOfIssue_is_comm    holds     exit=0
```

This run needed two passes: the first, with `--update-artifacts`, regenerated
the facts fixtures and then failed its own source-identity recheck, for the
structural reason recorded under ticket 18. The IDs above are from the clean
second pass.

The IDs above are from the final run, after the soundness review's stream-order
caveat was applied to `BlockingPairs`. An earlier run of the same tree is
superseded and is not cited anywhere.
