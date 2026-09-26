import ScoutBFacts

/-
Structural DTensor placement checks over the observed trace, plus the derived
negative the positive results alone cannot stand in for.

EVALUATION OF THE OBSERVED TRACE, not a theorem about the protocol: every
definition here is Bool-valued over the literal ScoutBFacts data and is closed
with `rfl`.

The observed run has zero Partial placements, so `noPartialObserved` on its own
proves nothing about whether one would be caught. `rejectsInjectedPartial`
injects one as a derived override of the real facts -- the same shape as
ScoutBEventMutationChecks -- rather than as a checked-in mutated copy of a
1.5 MB generated module.
-/

namespace Qwen3Formal.ScoutBChecks

open Qwen3Formal

set_option maxHeartbeats 0
set_option maxRecDepth 100000

theorem placementSchemaAgreesObserved :
    placementSchemaAgreesAcrossRanks
      ScoutBFacts.placementSchemaDigest
      ScoutBFacts.placementSchemaDigestByRank = true := by
  rfl

theorem meshAxisDegreeObserved :
    meshAxisDegreeAgreesWithCoordinates
      ScoutBFacts.rankCoordinates
      ScoutBFacts.meshAxisDegrees = true := by
  rfl

theorem parameterPlacementWellFormedObserved :
    parameterPlacementWellFormed
      ScoutBFacts.parameterPlacements
      ScoutBFacts.meshAxisDegrees = true := by
  rfl

theorem placementBooleansAgreeObserved :
    placementBooleansAgreeWithFacts
      ScoutBFacts.rankCoordinates
      ScoutBFacts.parameterPlacements = true := by
  rfl

theorem optimizerBoundaryObservedValid :
    optimizerBoundaryObserved ScoutBFacts.eventEvidenceByRank = true := by
  rfl

theorem noPartialObserved :
    noPartialParameterPlacement ScoutBFacts.parameterPlacements = true := by
  rfl

theorem shardedDimDividesAxisDegreeObserved :
    shardedDimDividesAxisDegree
      ScoutBFacts.parameterPlacements
      ScoutBFacts.meshAxisDegrees = true := by
  rfl

theorem localShapeReflectsShardingObserved :
    localShapeReflectsSharding
      ScoutBFacts.parameterPlacements
      ScoutBFacts.meshAxisDegrees = true := by
  rfl

theorem stridedShardCompositionObserved :
    stridedShardIsAnOuterComposedShard
      ScoutBFacts.parameterPlacements = true := by
  rfl

/--
The first DP-shard placement of the first parameter that has one, turned
Partial. DP-shard is the outer FSDP axis, whose reduce-scatter produces the
gradient the optimizer consumes, so a Partial there is the realistic failure
rather than an arbitrary one.
-/
def injectPartial : List ParameterPlacements :=
  let isTarget (item : ParameterPlacements) : Bool :=
    item.placements.any fun p => p.axis == "dp_shard" && p.kind == "shard"
  match ScoutBFacts.parameterPlacements.findIdx? isTarget with
  | none => ScoutBFacts.parameterPlacements
  | some index =>
      ScoutBFacts.parameterPlacements.modify index fun item =>
        { item with
          placements := item.placements.map fun p =>
            if p.axis == "dp_shard" && p.kind == "shard" then
              { p with kind := "partial", shardDim := none }
            else p }

/-- Guards the control against silently becoming a no-op. -/
theorem injectedPartialIsIsolated :
    (injectPartial != ScoutBFacts.parameterPlacements &&
      injectPartial.length
        == ScoutBFacts.parameterPlacements.length) = true := by
  rfl

/-- The injected Partial must be caught by the Partial check. -/
theorem rejectsInjectedPartial :
    noPartialParameterPlacement injectPartial = false := by
  rfl

/--
And by nothing else, so the violation is attributable. The same survivor set the
TLA control asserts: well-formedness, divisibility, strided composition and
schema agreement, all over the INJECTED facts.

`localShapeReflectsSharding` is deliberately absent, and its absence is the
honest part. The mutated parameter's local shape is half its global shape
BECAUSE the dp_shard axis shards it, so an injection that stops that axis from
sharding without rewriting the local shape contradicts that check too. Claiming
it here would be a claim this control cannot make.
-/
theorem injectedPartialPassesTheOtherPlacementChecks :
    (parameterPlacementWellFormed injectPartial ScoutBFacts.meshAxisDegrees &&
      shardedDimDividesAxisDegree injectPartial ScoutBFacts.meshAxisDegrees &&
      stridedShardIsAnOuterComposedShard injectPartial &&
      placementSchemaAgreesAcrossRanks
        ScoutBFacts.placementSchemaDigest
        ScoutBFacts.placementSchemaDigestByRank) = true := by
  rfl

/-- And the one check that cannot survive it does not, stated not hidden. -/
theorem injectedPartialBreaksLocalShapeAgreement :
    localShapeReflectsSharding injectPartial ScoutBFacts.meshAxisDegrees
      = false := by
  rfl

#print axioms placementSchemaAgreesObserved
#print axioms meshAxisDegreeObserved
#print axioms parameterPlacementWellFormedObserved
#print axioms placementBooleansAgreeObserved
#print axioms optimizerBoundaryObservedValid
#print axioms noPartialObserved
#print axioms shardedDimDividesAxisDegreeObserved
#print axioms localShapeReflectsShardingObserved
#print axioms stridedShardCompositionObserved
#print axioms injectedPartialIsIsolated
#print axioms rejectsInjectedPartial
#print axioms injectedPartialPassesTheOtherPlacementChecks
#print axioms injectedPartialBreaksLocalShapeAgreement

end Qwen3Formal.ScoutBChecks
