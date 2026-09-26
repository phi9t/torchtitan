# 12 — DPxTP refinement bridge

**What to build:** Prove the observed four-rank run is an admitted behaviour
of `ScoutBModel` by replaying its issue order through the model's own guards.

**Blocked by:** 11 — DPxTP collective protocol model.

**Status:** ready-for-agent

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
