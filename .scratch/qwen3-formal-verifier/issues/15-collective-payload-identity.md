# 15 — Collective payload identity

**What to build:** Export collective input/output sizes and dtypes, and check
the shape and volume agreement that per-communicator order agreement does not
cover.

**Blocked by:** 11 — DPxTP collective protocol model.

**Status:** ready-for-agent

## Why this is cheap

The data is already captured and then dropped. NCCL Flight Recorder entries
carry `input_sizes`, `output_sizes` and dtype per work;
`_collective_observations` never reads those fields, so they do not reach the
raw trace and therefore do not reach the formal facts. **No new
instrumentation and no new GPU run is needed** -- this is an exporter change
plus a predicate, the same shape as ticket 11's P1.

Confirm that before building: if the fields turn out to be absent rather than
merely unread, this becomes an instrumentation ticket and its cost changes,
which also makes it ticket 08's problem.

## What it buys

Matching op order on a communicator is necessary but does not make a
collective well-formed. Members must also agree on sizes and dtype; a mismatch
is a different failure than a hang -- it corrupts or errors. So this extends
the safety property along a dimension the current model cannot express at all,
since `Issue` carries only `comm` and `op`.

It also makes the model's abstraction auditable: with sizes present, an
all-gather's output should be the member count times its input, and a
reduce-scatter's the inverse. That relation is checkable against the observed
run and would catch an exporter that mislabelled an operation -- exactly the
class of error that produced the eight-vs-four communicator confusion.

## Acceptance criteria

- [ ] Verify first, in the raw Flight Recorder output, that sizes and dtype
      are present per work. Record what was found either way.
- [ ] `CollectivePayload` exported keyed the same way as `CollectiveComm`, on
      `process_group.canonical_id` -- never `runtime_pg_id`, which is per-rank
      local numbering and denotes different communicators on different ranks.
- [ ] A predicate requiring members of one collective to agree on dtype and on
  the size relation implied by the operation.
- [ ] A negative mutating one member's size, derived from the facts rather
      than hand-written, refused at the payload predicate and not at an
      earlier guard.
- [ ] This changes the raw schema, so every raw digest moves. Sequence it as
      its own commit with an intentional `--update-artifacts`, and say so in
      the ticket -- an unexplained digest change is indistinguishable from a
      regression.

## First criterion discharged, with two corrections to this ticket

Measured before starting the work, because the ticket's cost estimate turned on
it.

**The fields exist.** `torch 2.13.0`'s
`torch/include/torch/csrc/distributed/c10d/FlightRecorderDetail.hpp` emits
`input_sizes`, `output_sizes`, `input_dtypes` and `output_dtypes`. So the
premise holds: the Flight Recorder carries payload identity and
`_collective_observations` does not read it -- it reads `is_p2p`,
`profiling_name`, `state`, `thread_id` and `thread_name`, and nothing else.

**Correction 1: the fields are plural and per-tensor.** This ticket says
"dtype". It is `input_dtypes` and `output_dtypes`, lists parallel to the size
lists.
A single-dtype model would be wrong for a collective over multiple tensors.

**Correction 2: "no new instrumentation and no re-run" is wrong.** The raw
per-rank evidence holds only `device`, `events`, `identity`, `lineage`, `mesh`,
`process_groups`, `profile`, `schema` and `tensor_placements` -- there is
**no flight-recorder entries list preserved at all**. `_flight_snapshot()` calls
`_dump_nccl_trace_json()` at runtime and only the projection survives into the
artifact. So reading sizes requires a collector change, which changes the raw
schema, which requires a fresh 4-GPU run and a new canonical trace.

That is why this ticket is sequenced last, and the reason is now measured rather
than assumed. It also means the work pairs naturally with ticket 23, which needs
a fresh run for the same reason -- one run can serve both if the collector
changes land together.

## Implementation record

### What changed

Collector, `torchtitan/experiments/qwen3_formal_verifier/scout_b.py`:

- `_flight_payload` reads `input_sizes`, `output_sizes`, `input_dtypes` and
  `output_dtypes` off each Flight Recorder entry and carries them into the raw
  per-rank evidence as `observation.payload`, unreduced and per tensor. No
  element count is derived there: the raw trace stays what the runtime reported.
- `payload` joins the stable fields of `_validate_raw_collectives`, so it must be
  identical across a work's three lifecycle events. Without that, a trace could
  carry one size at enqueue and another at completion and the facts -- which read
  the first event -- would not show it.
- An empty shape is accepted and means a 0-dim tensor of one element, which the
  reduced scalar loss produces. A zero or negative dimension is refused.
- `_collective_payload` projects the four lists plus derived total element counts
  and the operation's implied size relation. Element counts, not shapes, because
  an all-gather may concatenate along any dimension.

### Raw schema

`RAW_SCHEMA` bumped `qwen3.formal.raw.v0` -> `v1`, and
`PROVENANCE_CONTRACT_VERSION` `v1` -> `v2` with a new `observed_payload` block
stating that these four fields are OBSERVED, per tensor, and not the positional
inference that produces the producer and stream fields. The bump is for exactly
the reason the contract carries a version at all: the string is compared by
equality, so without it every previously sealed trace fails on a missing-field
message a third party cannot tell from tampering.

Keyed off `RAW_SCHEMA`: only `raw_trace()`, `_validate_raw_rank_trace` and the
unit-test bundle. `scout_a.py` has its own constant of the same value, left
alone -- a single-rank trace observes no collective. `SCOUT_SCHEMA` and
`RAW_EVENT_PROJECTION_SCHEMA` are NOT bumped: the projection rule (same
exclusions, same digest) is unchanged, and a stale normalized bundle now fails on
the contract version, which is the designed path.

### The facts, and what they are keyed on

Per collective work, keyed by work id, with the communicator key being
`process_group.canonical_id` -- never `runtime_pg_id`. TLA+ gains
`CollectivePayloadComm`, `...SizeRelation`, `...InputSizes`, `...OutputSizes`,
`...InputDtypes`, `...OutputDtypes`, `...InputElements`, `...OutputElements`, in
a delimited block so `SCOUT_B_TLA_PAYLOAD_FACTS` can report its byte share.

Lean's shape differs, and the reason is measured rather than assumed: TLC handles
an all-pairs agreement predicate over 432 works, the Lean kernel does not -- a
`rfl` over that form had not finished after four minutes against a 120s
per-module budget. So the Lean facts add `payloadCollectiveRows` (one row per
collective, reached by a `Nat` index) and `payloadCommTable` (one row per
distinct communicator key), and the predicates join through those.

### The checks

Four invariants, `ScoutBValid` 19 -> 23: well-formedness with the element counts
RECOMPUTED from the shapes, the communicator-key property, member agreement on
dtype and volume, and the operation's implied volume relation with its exported
label checked against the operation.

The keying check has two halves and only one is load-bearing. Equality with
`CollectiveComm` is a drift tripwire. The refutation of a `runtime_pg_id` keying
is that works sharing a payload key must share a member set -- which is the
eight-versus-four confusion, refuted rather than commented on.

### The negatives, and their survivor sets

`ScoutBPayloadSizeInvalid` doubles the first dimension of every shape of ONE
member and recomputes that member's element counts. Doubling both sides preserves
the operation's relation, so the only thing it breaks is agreement. TLC reports
`ScoutBCollectivePayloadAgreement` at exit 12. Survivors, all asserted over the
mutated facts: `ThereIsASizedWorkWithAPeer`, `MutationIsIsolated`,
`StructureSurvivesTheMutation`, `OtherCollectiveChecksSurviveTheMutation` --
which includes `PerCommunicatorIssueOrderAgreement`.

`ScoutBPayloadOperationInvalid` relabels one collective's operation on EVERY
member, with the relation label moved to match, which is what an exporter bug
does. TLC reports `ScoutBCollectivePayloadSizeRelation` at exit 12. Survivors:
`ThereIsAVolumeChangingCollective`, `MutationIsIsolated`,
`StructureSurvivesTheMutation`, `IssueOrderAgreementSurvivesTheMislabelling`.

That last survivor is the argument for the whole ticket: a symmetric mislabelling
is invisible to order agreement because all four ranks agree on the wrong label.
Both cfgs pin their invariant order -- sentinel first, property last -- with the
reason in the file, and both are covered by
`test_placement_negative_configurations_pin_their_invariant_order` and by a
missing-target control over a hand-written stand-in facts module.

### Two further corrections to this ticket

- An empty `input_sizes` shape is not malformed. It is a 0-dim tensor of one
  element, so the product of no dimensions is 1 and not 0. A collector that
  refused it would refuse the reduced scalar loss.
- "A predicate requiring members to agree on dtype and on the size relation" runs
  two properties together. They need separate invariants and separate negatives:
  a volume mismatch between members and an exporter mislabelling an operation are
  different faults, and a single mutation cannot be attributed to both.

### Cost, measured rather than assumed

The two checkers pay very different costs for the same property, which is why
their facts are shaped differently.

- TLC, all four payload invariants over a 432-work stand-in at the observed
  cardinality: 2s and 1s on two runs of an otherwise idle machine. The all-pairs
  agreement and keying quantifiers are not a problem for TLC.
- The Lean kernel, same cardinality: the all-pairs form of agreement had NOT
  finished after four minutes and was killed, against a 120s per-module budget.
  Restated over a per-collective row reached by a `Nat` index it is about 26s,
  with well-formedness about 13s and the keying and relation checks about 1s
  each, and `rfl` caches nothing between theorems -- hence the split into two
  modules with a 300s budget each.
- A cheaper-looking alternative, measured and rejected: proving the rows'
  collective ids pairwise distinct with `eraseDups` costs about 53s, worse than
  the per-row count it would have replaced.

Byte cost is reported in-gate by `SCOUT_B_TLA_PAYLOAD_FACTS` rather than
estimated here, for the reason ticket 17 records: `payload_bytes=` is exact and
`parse_ms=` measures machine load first.

### Verified without a fresh trace

- Exporter driven for real over the four-rank unit-test bundle: payload facts
  present, per-tensor, element counts equal to the products, keyed on the
  canonical id and never on `runtime_pg_id`, with the fixture's shared runtime id
  across two member sets making that claim non-vacuous.
- `run_tlc_scout_b.sh` run end to end against exporter-generated facts: all 23
  invariants clean, and both new negatives reporting exactly their invariant at
  exit 12 with their survivor sets in the token.
- Both Lean payload modules compiled against exporter-generated facts: 11
  results, every one axiom-free.
- Real TLC over the shipped `ScoutDistributed` predicates and hand-written payload
  facts: one good input accepted and ten bad ones refused -- unpaired dtype list,
  zero dimension, an element count the shapes do not support, member volume
  mismatch, member dtype mismatch, mixed input/output dtype, symmetric
  mislabelling, a relation label disagreeing with its operation, an unknown
  operation, and a `runtime_pg_id` keying.
- 187 cases in the two owning pytest files, up from 167.

### Verifiable only after the gate produces a new trace

- That the OBSERVED payload satisfies the two properties. Two specific claims to
  watch, because both are about the real run rather than about the code. The size
  relation: if FSDP2's all-gather or reduce-scatter volumes do not come out as
  member-count multiples, the predicate or the collector's operation-family
  mapping needs revisiting, not the trace. And one dtype per collective: NCCL
  requires it, and FSDP2 collectives move one flat buffer, so it should hold, but
  a collective recorded over tensors of mixed dtypes would refute it and the
  invariant would then have to weaken to per-buffer uniformity.
- The checked-in `ScoutBFacts.tla`, `ScoutBBadFacts.tla`, `ScoutBFacts.lean` and
  `ScoutBBadFacts.lean` are STALE until an `--update-artifacts` run regenerates
  them. Until then the `scout-b` suite fails at parse time on the missing
  `CollectivePayload*` operators and at `rfl` on the missing `collectivePayloads`,
  which is the correct report, not a regression.
- The real byte and parse cost of the payload block, and the real Lean kernel
  time for `ScoutBPayloadChecks` and `ScoutBPayloadMutationChecks` at 432 works.
  The measured 26s/13s per evaluation is why those two modules get 300s rather
  than the shared 120s, and why the positives and mutations are separate modules.
- `run_lean_scout_b.sh`'s payload token emission. The runner cannot be exercised
  end to end on the unit-test bundle, because that bundle's event schedule does
  not satisfy the maintained Lean event-kind grammar -- a pre-existing property of
  the fixture, unrelated to payloads.

### Suite results at this commit

- `--suite tier0` (`run_formal_tier0.sh --no-fetch`): 5/5 Bazel targets pass,
  lint success over 19 changed files with pyrefly clean, 122 focused pytest
  contracts pass.
- `--suite scout-a`: 6/6 pass.
- `--suite scout-b`: 9/11 pass. The two failures are
  `tlc_scout_b_test` -- "Unknown operator: `CollectivePayloadComm'" and its
  siblings -- and `lean_scout_b_test` -- "Unknown identifier
  `ScoutBFacts.collectivePayloads`". Both are the stale checked-in facts modules
  and nothing else; they clear when an `--update-artifacts` run regenerates them
  from a fresh four-rank trace.
- `tests/unit_tests/test_qwen3_formal_scout_b.py` and
  `tests/unit_tests/test_formal_toolchain.py`: 187 pass, up from 167.

### Fixture finding, out of scope here

The unit-test four-rank bundle is not a trace `ScoutBValid` accepts: it shards
both `tok_embeddings.weight` axes on tensor dim 0 with no strided flag and a
local shape of half each dim, which contradicts `LocalShapeReflectsSharding` and
`StridedShardIsAnOuterComposedShard` both. Only its projection was ever
unit-tested. Moving the tp shard to dim 1 makes it consistent and would let the
whole runner be exercised on it; worth a follow-up.

### Gate evidence

Two gate runs, as the digest-moving sequence requires. Pass 1 with
`--update-artifacts` produced the new v1 trace, wrote the fixtures, and then
failed at `lint` with `dirty source identity changed during Scout B gate` --
correct behaviour, because the fixtures genuinely changed mid-run. Pass 2 is the
citable one: all nine stages exit 0 with `artifact_sync` in check mode.

- Scout B evidence ID:
  `sha256:0abb94c875d45ee32f27359e267c6f4d736f77dcbcdd370b4de7506dac4108c0`
- Scout B source ID:
  `sha256:fadac9baa8247371d5af3ada01bc894e136374cbb786d2f71989dccafadeb7f5`
- Nested Scout A evidence ID:
  `sha256:d83635dd08514f25be20d5ac6af3b93b0de5023cc437c5bd1c4137603be8269b`
- Raw schema `qwen3.formal.raw.v1`; provenance contract
  `qwen3.formal.scout.provenance-contract.v2` carrying the `observed_payload`
  block, which distinguishes these four fields from the inferred
  producer/stream attribution.

The four payload invariants are named in the sealed log alongside the other 19:
`ScoutBCollectivePayloadWellFormed`, `ScoutBPayloadCommKey`,
`ScoutBCollectivePayloadAgreement`, `ScoutBCollectivePayloadSizeRelation`.

### What the real trace settled

Every risk flagged as unverifiable before the run resolved from it:

- **The size relation holds on the observed payload.** Had FSDP2 volumes not
  been member-count multiples, `formal_networked` would have failed; it passed.
- **One dtype per buffer per collective.** 216 dtype entries across 108
  collectives per rank -- `Long` 4, `BFloat16` 140, `Float` 72 -- so the
  uniformity check did not need weakening.
- **The 0-dim correction was load-bearing, not theoretical.** Four collectives
  carry `input_sizes: [[]]`, a 0-dim `Long` tensor. A collector treating an
  empty shape as malformed, which this ticket's original wording implied, would
  have rejected the real run. The product of no dimensions is 1.
