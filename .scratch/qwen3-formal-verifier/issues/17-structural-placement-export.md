# 17 — Structural DTensor placement export

**What to build:** Export real per-parameter DTensor placements and check the
sharding contract, replacing the two booleans per rank in use today.

**Blocked by:** none. Independent of the protocol model work.

**Status:** resolved -- `_placement_facts` and `ScoutBPlacementChecks.lean`
landed in `19a485a7d`. This line read `ready-for-agent` until a status audit
corrected it; the work had been committed for some time and the ledger was
advertising it as available.

## What is lost today

The run observes 74 parameters per rank with real placements -- 115
`Shard(dim)` and 33 `Replicate` across ranks, with strided flags, zero
`Partial`. The formal facts collapse all of it to `PlacementHasDpShard` and
`PlacementHasTpShard`: two booleans per rank. `PlacementValid` therefore
checks that a 2x2 DPxTP run shards along both axes somewhere, which is true of
almost any non-degenerate configuration.

## What becomes checkable with structure

- No `Partial` placement survives into the optimizer step. A `Partial`
  reaching the optimizer means an unreduced gradient and silently wrong math
  -- exactly the class of bug `.claude/rules/distributed.md` warns about, and
  currently invisible to the formal layer.
- Every parameter's placement is consistent with the mesh axis it claims, and
  sharded dimensions divide evenly by the axis degree.
- TP-sharded and DP-sharded dimensions are distinguished per parameter rather
  than aggregated, so a parameter sharded on the wrong axis is detectable.

## Acceptance criteria

- [ ] Placements exported per parameter with axis and dim, not aggregated to
      booleans. Sizes grow: measure the effect on `ScoutBFacts.tla` parse time
      and report it as its own token, since that file is already about 1.4 MB
      and parse cost is charged before a single state is generated.
- [ ] Keep the two booleans, or derive them from the structural facts, so the
  existing `PlacementValid` result does not silently change meaning.
- [ ] An invariant that no `Partial` placement is present at the optimizer
  boundary, and a negative that injects one, derived from the facts.
- [ ] Divisibility of every sharded dim by its axis degree, checked against
      the mesh facts already exported.
- [ ] Raw schema changes, so digests move: own commit, intentional
  `--update-artifacts`, stated in the ticket.

## Implementation record

Digests move, so this is its own commit and needs an intentional
`--update-artifacts` gate run followed by a plain re-run (ticket 18): the
first regenerates the fixtures and then fails its own source-identity
recheck, and its bundle must not be cited.

The four regenerated fact modules are `ScoutBFacts.tla`, `ScoutBBadFacts.tla`,
`ScoutBFacts.lean` and `ScoutBBadFacts.lean`.

Size and parse cost, measured with the pinned SANY over `ScoutBValid`:

- `ScoutBFacts.tla` 1,642,832 -> 1,689,740 bytes (+46,908, +2.9%), all of it in
  the delimited `structural placement facts` block.
- `ScoutBFacts.lean` 1,505,426 -> 1,528,490 bytes (+23,064, +1.5%).
- Controlled A/B on the same root module the refinement runner parses
  (`ScoutBRefine.tla`), interleaved on an idle machine, with "before" being the
  same facts module with the delimited placement block stripped: 921/835/896 ms
  before (median 896) versus 855/862/979 ms after (median 862). CORRECTED after
  review: three samples per arm was too few to resolve the effect, and a longer
  interleaved run puts it at a real and reproducible +35 to +50 ms, about 4-5%.
  The interleaved A/B is the trustworthy shape; the conclusion is "small and
  measurable", not "no measurable regression".
- The gate's own token read `parse_ms=1671` under Bazel, against 814-881 ms
  recorded historically. That is sandbox contention, not the facts: the refine
  target parsed while `tlc_scout_b_model_test` and another action were running.
  Run standalone through the same runner, the same tree reported 931 ms.

Two corrections to this ticket's text:

- The counts 115 `Shard` / 33 `Replicate` are PER RANK, not "across ranks".
  Across the four ranks the trace carries 460 and 132. All four ranks agree on
  the whole placement list, which is why the facts are exported once with the
  per-rank digests beside them.
- The trace has an identifiable optimizer boundary -- `optimizer.started` and
  `optimizer.mutated` per rank, from the AdamW step pre/post hooks -- but the
  placement snapshot is NOT taken there. It is taken once, after the model is
  built and before `Trainer.train`, over `model_parts[0].named_parameters()`,
  and no gradient placement is observed at all. So a `Partial` GRADIENT is not
  visible to this evidence. What is checkable, and what the invariant is named
  and documented for, is that no parameter the observed optimizer step updates
  carries an unreduced placement, with `OptimizerBoundaryObserved` conjoined so
  the claim refers to a step the trace actually contains. Closing that gap
  needs a second snapshot inside the optimizer pre-hook plus gradient
  placements, which is a new GPU run and therefore a separate ticket.

Strided flags are real and carry more than a flag: all 25 per-rank
`_StridedShard` placements are on the outer `dp_shard` axis and occur exactly
where `tp` shards the same tensor dim. That correspondence is now an invariant
(`StridedShardIsAnOuterComposedShard`).

One thing the Partial negative cannot claim: local-shape agreement is not in its
survivor set. The mutated parameter's local shape is half its global shape
because the DP-shard axis shards it, so an override that stops that axis from
sharding without rewriting the local shape contradicts
`LocalShapeReflectsSharding` as well. The cfg lists
`ScoutBNoPartialAtOptimizer`, so that is the invariant TLC reports, and
`StructureSurvivesTheMutation` plus
`OtherPlacementChecksSurviveTheMutation` -- well-formedness, divisibility,
strided composition and schema agreement over the mutated facts -- are the
attribution a reader can rely on.

## Review findings addressed

F1 (MEDIUM), the schema digest was trusted, never verified. Real finding.
`validate_normalized_bundle` now recomputes `_sha256_json(tensor_placements)`
per rank and refuses a mismatch before the digests are compared with each other,
and `num_tensors` is checked against the actual list length rather than merely
being a positive integer. Demonstrated on the real bundle: rank 1's 33
`replicate` placements turned `partial` with the digest left stale was ACCEPTED
before with 0 partials in the exported facts, and is now refused with
"rank 1 placement schema digest does not cover its tensor placements", at
`validate_normalized_bundle` and therefore also at `_formal_projection`; the
untouched bundle still exports 148 placements and 0 partials. Note what the
existing chain did not cover: `trace_id` covers `placement_summary`, so it moved
with the digest but never with the placement bytes, which is why nothing caught
this. `_placement_facts`'s docstring claimed the bundle "already refuses a run
whose ranks disagree" -- true of how the digest is produced, not of what was
enforced where it is consumed -- and now states the recomputation.

F2 (LOW/MEDIUM), the guard justification was factually wrong. Confirmed, and the
measurement went further than the review's. With the sentinel invariant listed
first, guarded and unguarded are identical: both report the sentinel FALSE at
exit 151, so the cfg ORDER was doing the work the comment credited to the
`CHOOSE` guard. But the guard IS load-bearing in the configuration where the
sentinel is not listed: guarded reports `MutationIsIsolated` by name at 151,
unguarded abandons the search at exit 75 on the `CHOOSE`. Both comments now
state that split with the measurement, and the test was replaced by
`test_the_derived_mutation_guard_is_load_bearing`, which runs the sentinel-less
cfg both ways and so fails if the guard is deleted.

Inherited justification: `ScoutBIssueOrderInvalid` carried the same wrong claim
AND a real defect behind it. With its sentinel unlisted, its guarded form also
crashed at exit 75 -- on "Attempted to apply function", because
`MutationIsIsolated` applied `MutatedOperation` to the `""` sentinel without
first checking domain membership. It has been given the
`MutatedWork \in DOMAIN CollectiveOperation` conjunct, first in the conjunction
since TLC short-circuits left to right, and now degrades to a named invariant
like the placement module. Its comment is corrected and the new test covers both
modules. `ScoutARefineBad`'s `ControlEventsArePresent` does NOT carry the bad
claim: its comment says the indices are read while `Observed` is being
constructed, so no invariant can be reported ahead of them, which is a different
and correct justification. Nothing to fix there.

F3 (LOW), the invariants were invisible in the sealed log. `SCOUT_B_TLA_VALID`
now carries `invariants=` derived with the shared `formal_cfg_invariants`, all
19 named, and both derived negatives carry their invariant list too -- for the
Partial negative that list IS the survivor set the attribution rests on. All
three fail closed on an empty list.
`test_scout_b_tokens_name_the_invariants_they_checked` enforces it.

F4 (LOW), the survivor-set attribution depended on unpinned cfg ordering. Both
negative configurations now carry the reason in the file and are pinned by
`test_placement_negative_configurations_pin_their_invariant_order`: the sentinel
first, the property under test last. Reordering either way now fails a test
rather than silently producing a green negative with unevaluated survivors.

Lower priority, all done. (a) `PlacementBooleansDerivable` oversold its name and
is now `PlacementBooleansAgreeWithFacts` / `placementBooleansAgreeWithFacts` /
`ScoutBPlacementBooleansAgree`, with the two limits written into the predicate
header: no rank argument, so for ranks 1-3 it is agreement with the one exported
list rather than a derivation, and one bit per axis, so a projection dropping
all but one shard placement per axis would still satisfy it. (b) The Lean
control now asserts the same survivor set as the TLA one -- well-formedness,
divisibility, strided composition, schema agreement -- as
`injectedPartialPassesTheOtherPlacementChecks`, and states the exclusion
positively as `injectedPartialBreaksLocalShapeAgreement`; the misleading
`role=no-other-check-catches-it` label is gone. (c) The README now says PER RANK
and gives the across-rank figures. (d) Counts, verified rather than recalled: 9
new TLA predicates plus 2 helpers, 8 new named invariants taking `ScoutBValid`
from 11 to 19, 13 `ValueError` paths in `_placement_facts` plus the new digest
check, and 32 collected test cases in the two owning files covering this work.
(e) The README now records that `RankCoordinates` is synthesized as `rank // 2`
and `rank % 2` in the projection, with the link to observation being
`_runtime_topology`'s refusal of any run whose real mesh coordinate differs --
sound, but not two independent observations agreeing. (f) `parse_ms` is KEPT
with a caveat rather than dropped: it is still the only in-gate record that the
parse completed and roughly what the fixed cost was, but the README now says
plainly that it measures machine load first (32 concurrent SANY parsers
reproduce 1671 ms on the unchanged fixture, 96 bracket 2008), that a cost claim
needs an interleaved A/B, and that `placement_bytes=` is the exact number
tracking the export. The earlier "no measurable regression" is corrected to a
real and reproducible +35 to +50 ms (4-5%).

### Gate evidence

Full nine-stage Scout B gate, all stages exit 0, sealed and verified, after
the four review findings were fixed. `artifact_sync` ran in check mode rather
than write mode, so the fixtures were genuinely verified.

- Scout B evidence ID:
  `sha256:e25b663d90b1c8025a031a58c65e9930922fcbaab8e133afd6dd691735a341e2`
- Scout B source ID:
  `sha256:bd66c132e00bd6ec2552cfff417f17642e20f15c7ed2c968ea90aca794f7d920`
- Nested Scout A evidence ID:
  `sha256:1b90e620bda4fc0ac8229f6099b231c0a792bf7b8a46d64ec75966c04e8bbfed`

All 19 invariants are now named in the sealed log rather than hidden behind
an aggregate result, so a reader of a bundle can tell what passed:

```
SCOUT_B_TLA_VALID invariant_set=dpxTP result=success invariants=
  ScoutBMeshWellFormed ScoutBInputIdentity ScoutBPlacementValid
  ScoutBPlacementSchemaAgrees ScoutBMeshAxisDegree
  ScoutBParameterPlacementWellFormed ScoutBPlacementBooleansAgree
  ScoutBNoPartialAtOptimizer ScoutBShardedDimDividesAxisDegree
  ScoutBLocalShapeReflectsSharding ScoutBStridedShardComposition
  ScoutBEventEvidence ScoutBCrossRankSynchronization
  ScoutBCollectiveLifecycle ScoutBCollectiveCoverage
  ScoutBCollectiveProducerCorrelation ScoutBCollectiveEventEvidence
  ScoutBPerCommunicatorIssueOrder ScoutBRuntimePgIdConsistent
```
