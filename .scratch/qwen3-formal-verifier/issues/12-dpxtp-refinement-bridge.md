# 12 — DPxTP refinement bridge

**What to build:** Prove the observed four-rank run is an admitted behaviour
of `ScoutBModel` by replaying its issue order through the model's own guards.

**Blocked by:** 11 — DPxTP collective protocol model.

**Status:** review-pending

Delivers the distributed half of 07. Ticket 10 delivered the single-rank half.

## Why this matters

`ScoutBModel` and the observed run are currently unrelated artifacts. The
model says what the protocol permits; the facts say what happened. Without a
bridge, a reader may reasonably assume the model was checked against reality,
and it was not.

## The one trap

Replay the **issue order only**. The observer appends each work's
enqueued/started/completed triple contiguously, post hoc, from the Flight
Recorder, and the projection drops `observed_time_ns`. So the per-rank order
of `started`/`completed` asserts a full serialization of every collective
before the next, which cannot have happened -- FSDP all-gathers overlap by
construction.

A bridge that replayed per-rank event order would therefore certify a fiction,
and worse it would **pass**, because a fully serialized schedule satisfies
every guard trivially. Only the `enqueued` sub-order is faithful; it inherits
`flight_record_id`, which is contiguous 8..115 and identical across ranks.

Let the model schedule `Start` and `Complete` by its own guards.

## Acceptance criteria

- [ ] `ScoutBRefine` replays each rank's observed issue sequence, ordered by
      the `collective.enqueued` event, and takes no ordering information from
      the `started`/`completed` events.
- [ ] `Start` and `Complete` are taken by the model's guards, not driven by
  evidence.
- [ ] The observed run is admitted, reported as a positive result token; the
  runner translates the reachability-by-refutation polarity as it already does
  for Scout A.
- [ ] Cost is linear in the trace, not combinatorial. A prefix-constrained
  search over all 4x335 events explores roughly `336^4` and is not viable; the
  greedy replay is about 760 steps.
- [ ] The module header records the confluence argument that makes greedy
      replay without loss of generality: every guard is monotone in `issued`
      and `doneOn`, and no action disables another rank's `Issue`, so if any
      evidence-consistent linearization enables `Start(c)` the greedy one does
      too.
- [ ] A negative control corrupts one member's issue order on one communicator
      so the ops disagree at a position, and is refused **at the rendezvous
      guard**. Transposing across different communicators would be refused by
      the stream-head or deadlock guard instead -- the wrong guard.
- [ ] The negative carries the four-invariant discipline of `ScoutARefineBad`:
  corruption isolated, not admitted, refusal not earlier than the rendezvous,
  and the mismatched collective never runs.
- [ ] Focused tests, full gate, and fresh independent review.

## Follow-on

The prose confluence argument should later be replaced by a machine check: a
bounded-skew replay (`MaxReplaySkew = 2`) that explores all linearizations
within a small rank skew. That covers a defined fraction of the interleaving
space rather than arguing one case, and costs roughly `108 * (2K+1)^3`.

## Scoping note (2026-09-26): no exporter change is needed

Measured against the checked-in `ScoutBFacts.tla`:

- `CollectiveWorkIds`, `CollectiveRank`, `CollectiveComm`, `CollectiveOperation`
  and `CollectiveIssueOrder` are all already exported. Per-rank issue sequences
  can therefore be derived inside the new module from existing facts, so this
  ticket is a TLA+ module plus a runner -- no change to `scout_b.py`, and no
  digest movement.
- 432 works, exactly 108 per rank.
- `CollectiveIssueOrder` values are positions in each rank's own event stream:
  distinct within a rank, spanning 11..332 with gaps, and repeating across
  ranks. They are not global indices, so the derivation must group by rank
  before ordering.
- All four ranks have positionally identical operation sequences, so operation
  agreement already holds on the observed run. The bridge's rendezvous guard
  will be satisfied by the real trace; what the bridge tests is ordering and
  stream feasibility, not op agreement.

Derive the sequences with the `CHOOSE`-based nth-by-order pattern already in
`ScoutDistributed.tla` (`PerCommunicatorIssueOrderAgreement`); the community
`SequencesExt` module is not available in the hermetic toolchain. Define them at
constant level so TLC evaluates them while computing initial states rather than
per state -- the derivation is O(n^2) in a rank's 108 works.

Depends on ticket 21: the bridge must not be built onto a model whose deadlock
property cannot distinguish a circular wait from a budget artifact.

## Inherited hazard to fix while you are in this module

`ScoutBIssueOrderInvalid.tla` derives its mutation with

```
MutatedWork == CHOOSE work \in SequenceElements(CollectiveWorkIds) :
  CollectiveRank[work] = 0 /\ ...
```

which is the same unguarded `CHOOSE` that ticket 22 just fixed in
`ScoutLifecycle.tla`: it aborts with exit 75 and
`Error: Attempted to compute the value of an expression of form CHOOSE ...`
if rank 0 has no collective. That log also prints a state summary and
`Finished in`, so before ticket 22 only the exit-status pin rejected it; the
evaluation-error class now catches it, but the module should still name its
missing target rather than crash. Use the sentinel-plus-named-invariant pattern
`ScoutARefineBad` now uses (`ControlEventsArePresent`).

Two related notes from ticket 22's implementation:

- The duplicate-`IndexOf` guard is **not falsifiable by execution**: TLC's
  `CHOOSE` already returns its first witness, so no run distinguishes guarded
  from unguarded. It is still correct -- it makes the TLA+/Lean agreement a
  specification fact rather than a TLC fact -- but do not expect a
  failing-before, passing-after demonstration for it.
- `IndexOf` is now partial by contract, with a `HasKind` precondition at each
  call site. The DPxTP port repeats collective kinds per rank, which is exactly
  the case that made the guard necessary, so honour the precondition rather than
  assuming a unique match.

## Resolution (2026-09-26): implemented as ScoutBRefine

`experiments/qwen3_formal_verifier/formal/ScoutBRefine.tla` plus seven
configurations and `run_tlc_scout_b_refine.sh`, wired into
`scout_b_formal_tests` as `tlc_scout_b_refine_test`. No exporter change, no
digest movement: the scoping note above held.

Shape follows `ScoutARefine`. No cursor variable; `Len(issued[r])` is the
per-rank cursor; reachability is proved by refuting
`ReplayedRunIsNotAdmitted` and the runner flips the polarity into a positive
token. Measured results, one worker:

- `SCOUT_B_REFINE_PARSE facts_bytes=1642832 parse_ms=881` -- parse plus
  semantic processing of the 1.6 MB facts module, charged before any state.
- `SCOUT_B_REFINE_MAPPING result=success` -- the derived communicator mapping
  is a bijection agreeing with `CommMembers` and `CommOps`; the mapping is
  printed by the checker.
- `SCOUT_B_REFINE result=admitted issues_per_rank=108 search_skew_bound=1
  distinct_states=5371 witness_depth=865` in about 25 s. The same witness is
  found at looser spreads and costs more: 9,199 states at 2, 11,629 at 3,
  13,261 at 4.
- `SCOUT_B_REFINE_CONFLUENCE result=no_dead_end bound_issues_per_rank=54
  bound_issue_skew=2 bound_outstanding_per_comm=1 distinct_states=53405` in
  3 min 21 s, plus `SCOUT_B_REFINE_CONFLUENCE_OVERLAP` refuting
  `NoTwoCollectivesRunConcurrently` so the fragment is not a serialized space.
- `SCOUT_B_REFINE_ORDER_UNIFORM permutation=reverse_every_rank result=admitted
  distinct_states=5371 witness_depth=865` and
  `SCOUT_B_REFINE_ORDER_SINGLE_RANK permutation=reverse_one_rank
  result=refused distinct_states=18` -- the order-sensitivity pair; see below.
- `SCOUT_B_REFINE_NEGATIVE result=rejected_at_rendezvous_guard
  distinct_states=2573 reach_distinct_states=2531` with
  `RejectionHappensBeforeTheRendezvous` refuted at depth 408, and
  `SCOUT_B_REFINE_GUARD_ISOLATION relaxed=RequireMatchedIssueOrder
  result=admitted`. The state counts are now emitted by the runner rather than
  quoted from a scratch log.
- `SCOUT_B_REFINE_NEGATIVE_GUARD_SET guards=positive result=refused_at_spmd_guard
  witness=CorruptedColumnIsNeverFormed distinct_states=2555` -- the same
  corruption under the positive's guard set.

### Three things in this ticket were wrong

1. **`MaxSkew` is gone**, so bounded-skew replay is not free. Commit
   `4f252960d` dropped it as a tautology at `MaxIssues = 2`. The cost control is
   in the refinement module instead and `Next` is untouched.
2. **A skew bound alone does not bound the state space.** All four ranks can
   race to their issue bound with nothing completed, and the completions then
   interleave combinatorially: `MaxIssues = 12, MaxReplaySkew = 2` with work in
   flight unbounded passed 240,000 distinct states at depth 43 and was
   abandoned. `MaxOutstanding`, a per-rank per-communicator overlap window,
   is what makes an exhaustive bounded run finish.
3. **The follow-on cost estimate `108 * (2K+1)^3` is optimistic** even with
   both bounds. At `MaxReplaySkew = 2, MaxOutstanding = 1` the full 108 issues
   did not finish in ten minutes (abandoned at depth 500 of roughly 1000, 62,000
   distinct states and slowing, because the model's guards are quadratic in the
   issue count). The shipped fragment is the first 54 issues of each rank:
   53,405 states in 3 min 21 s. 54 is chosen, not convenient -- rank 0's first
   `reduce_scatter` is at position 52 and the pair the negative control
   transposes is 52/53, so a shorter prefix contains no `reduce_scatter` at all
   and the `rs` stream never appears. `all_gather` is in play from position 2,
   so the `ag` stream is not what the bound buys; an earlier draft of this note
   said otherwise and was wrong.

### What the positive does and does not establish (review finding, repaired)

The first draft of this module, its README section and this note all said the
positive was evidence about the observed ISSUE ORDER. It is not, and review
measured it: feeding the unmodified positive configuration each rank *reversed*,
each rank *rotated by one*, or a `37n mod 109` permutation admits all three, and
the reversed order reproduces the positive's numbers exactly -- 9,172 generated,
5,371 distinct, depth 865. Relaxing `RequireMatchedIssueOrder`,
`RequireStreamOrder` or either SPMD guard on the observed order also leaves
those numbers unchanged. Only a rank-0-only scramble is refused, at depth 6.

That is not a defect in the bridge; it is what the protocol says.
`OpsAgreeAtFront` quantifies over the members at a communicator's front, and
`UniformProgramOpsOK`/`UniformProgramCommsOK` quantify over the ranks at a
position. All three are relational across ranks at a position and none mentions
which collective a position ought to carry, because NCCL requires members of a
communicator to AGREE on their collective sequence, not to follow any particular
one. A permutation applied uniformly to all four ranks is an equally valid SPMD
program and must be admitted.

So the claim is now stated as: every one of the 108 cross-rank columns is
model-typable and cross-rank consistent, and the collectives those columns
induce drain to completion under the model's own guards. Absolute order is
explicitly NOT constrained, and the reason is given. The limit is on the record
as a checked pair rather than a caveat -- `ScoutBRefineUniformPermutation.cfg`
(uniform permutation, expected ADMITTED) and
`ScoutBRefineSingleRankPermutation.cfg` (one rank only, expected REFUSED) --
and the refused half is the positive evidence that the bridge detects the
agreement violation a real job hangs on.

### Which guard refuses the corruption, and under which configuration

All three isolation configurations relax `RequireUniformProgramOps`, so none of
them runs under the positive's guard set. Under the positive's guard set the
same corruption is refused too, and differently: `RequireUniformProgramOps`
refuses every PEER's issue at the corrupted position, so the corrupted column
never forms and the job cannot advance past position 51.
`ScoutBRefineBadUniform.cfg` checks that -- `CorruptedColumnIsNeverFormed`
holds, 2,555 distinct states, depth 414.

The corrupted record itself IS issued even with that guard on, which was not the
first guess: `UniformProgramOpsOK` compares positions only up to the shorter of
two ranks' sequences, so at the frontier, where the skew bound leaves the peers
one issue behind, the corrupted position is not yet compared with anything. The
measurement is what corrected the explanation.

Relaxing the SPMD guard is necessary for the isolation stage -- with it on,
flipping `RequireMatchedIssueOrder` does not admit the corruption and the
attribution to that one constant collapses. It is NOT necessary for the reach
stage: with the guard TRUE that witness appears at the same depth. It is set
there only to keep the three configurations on one constant set.

### Two things the ticket did not anticipate

`MaxIssues = 108` overflows TLC's default 1 MB thread stack. `IssueAllowed`'s
SPMD-program conjuncts contain `\A k \in 1..MinOf(len1, len2)` evaluated in
action position, where TLC recurses once per bound element, so the recursion is
4 x 4 x issue-count deep. Attribution was checked by rerunning the same
configuration with both uniformity guards relaxed, which completes the whole
865-step replay at the default stack. The runner uses `-Xss32m`; `-Xss8m` also
suffices.

And **admission cannot be a length test**. `Issue`'s guard never reads `doneOn`,
so "every rank reached its observed issue count" is reachable by issuing
everything and running nothing, for the corrupted order as much as for the real
one -- the bridge and its negative control would both have passed vacuously.
Admission is `AllDone`. `test_dpxtp_bridge_length_only_admission_would_accept_the_corruption`
runs that vacuous variant against the corrupted order and shows it is reachable.

### Inherited hazards, both fixed

`ScoutBIssueOrderInvalid.tla` now derives `MutatedWork` through
`Rank0HasACollective` with an empty-string sentinel and names that invariant
first in its cfg, so a trace where rank 0 issued no collective is reported
rather than crashing with exit 75. The `NthByIssueOrder` call site in
`ScoutBRefine` establishes its witness before the `CHOOSE`, with a sentinel
record and `ObservedIssueOrderIsDistinctWithinRank` as the named invariant, for
the same reason `IndexOf` is now guarded at every call site: these are constant
definitions, folded before any invariant could report the absence.

### Gate evidence

Full nine-stage Scout B gate, all stages exit 0, sealed and verified.

- Scout B evidence ID:
  `sha256:cfb17f931dbaa7d2c5c17b3b7671a0c519eb03842f2959aa8cfb2d3d15b04f10`
- Scout B source ID:
  `sha256:203bdd35c290e2f3d4acb01926840bd8d94a2d9be0f6c854d2426d19b422dafe`
- Nested Scout A evidence ID:
  `sha256:47b8943ebb394b0f2defa5574ce1925a9506be9625ddc9bba2ee473895dc6235`
- `--suite scout-b`: 10 of 10 targets pass.

The eleven tokens in the sealed log, with the pair that documents the limit
sitting adjacent so a reader cannot miss it:

```
SCOUT_B_REFINE           result=admitted distinct_states=5371 depth=865
SCOUT_B_REFINE_ORDER_UNIFORM     permutation=reverse_every_rank
                         result=admitted distinct_states=5371 depth=865
SCOUT_B_REFINE_ORDER_SINGLE_RANK permutation=reverse_one_rank
                         result=refused distinct_states=18
SCOUT_B_REFINE_CONFLUENCE  result=no_dead_end bound_issues_per_rank=54
                         bound_issue_skew=2 bound_outstanding_per_comm=1
                         distinct_states=53405 max_outdegree=7
SCOUT_B_REFINE_CONFLUENCE_OVERLAP  NoTwoCollectivesRunConcurrently violated
SCOUT_B_REFINE_NEGATIVE    result=rejected_at_rendezvous_guard
                         distinct_states=2573 reach_distinct_states=2531
SCOUT_B_REFINE_GUARD_ISOLATION relaxed=RequireMatchedIssueOrder result=admitted
SCOUT_B_REFINE_NEGATIVE_GUARD_SET guards=positive
                         result=refused_at_spmd_guard
                         witness=CorruptedColumnIsNeverFormed
SCOUT_B_REFINE_PARSE         facts_bytes=1642832 parse_ms=814
SCOUT_B_REFINE_MAPPING       result=success
SCOUT_B_REFINE_TOOLCHAIN     checker=tlc release=1.7.4 java_major=17
```
