import DeviceMeshFacts

/-
Structural DTensor placement checks over the observed trace, plus the derived
negative the positive results alone cannot stand in for.

EVALUATION OF THE OBSERVED TRACE, not a theorem about the protocol: every
definition here is Bool-valued over the literal DeviceMeshFacts data and is closed
with `rfl`.

The observed run has zero Partial placements, so `noPartialObserved` on its own
proves nothing about whether one would be caught. `rejectsInjectedPartial`
injects one as a derived override of the real facts -- the same shape as
DeviceMeshEventMutationChecks -- rather than as a checked-in mutated copy of a
1.5 MB generated module.
-/

namespace Qwen3Formal.DeviceMeshChecks

open Qwen3Formal

set_option maxHeartbeats 0
set_option maxRecDepth 100000

theorem placementSchemaAgreesObserved :
    placementSchemaAgreesAcrossRanks
      DeviceMeshFacts.placementSchemaDigest
      DeviceMeshFacts.placementSchemaDigestByRank = true := by
  rfl

theorem meshAxisDegreeObserved :
    meshAxisDegreeAgreesWithCoordinates
      DeviceMeshFacts.rankCoordinates
      DeviceMeshFacts.meshAxisDegrees = true := by
  rfl

theorem parameterPlacementWellFormedObserved :
    parameterPlacementWellFormed
      DeviceMeshFacts.parameterPlacements
      DeviceMeshFacts.meshAxisDegrees = true := by
  rfl

theorem placementBooleansAgreeObserved :
    placementBooleansAgreeWithFacts
      DeviceMeshFacts.rankCoordinates
      DeviceMeshFacts.parameterPlacements = true := by
  rfl

theorem optimizerBoundaryObservedValid :
    optimizerBoundaryObserved DeviceMeshFacts.eventEvidenceByRank = true := by
  rfl

theorem noPartialObserved :
    noPartialParameterPlacement DeviceMeshFacts.parameterPlacements = true := by
  rfl

theorem shardedDimDividesAxisDegreeObserved :
    shardedDimDividesAxisDegree
      DeviceMeshFacts.parameterPlacements
      DeviceMeshFacts.meshAxisDegrees = true := by
  rfl

theorem localShapeReflectsShardingObserved :
    localShapeReflectsSharding
      DeviceMeshFacts.parameterPlacements
      DeviceMeshFacts.meshAxisDegrees = true := by
  rfl

theorem stridedShardCompositionObserved :
    stridedShardIsAnOuterComposedShard
      DeviceMeshFacts.parameterPlacements = true := by
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
  match DeviceMeshFacts.parameterPlacements.findIdx? isTarget with
  | none => DeviceMeshFacts.parameterPlacements
  | some index =>
      DeviceMeshFacts.parameterPlacements.modify index fun item =>
        { item with
          placements := item.placements.map fun p =>
            if p.axis == "dp_shard" && p.kind == "shard" then
              { p with kind := "partial", shardDim := none }
            else p }

/-- Guards the control against silently becoming a no-op. -/
theorem injectedPartialIsIsolated :
    (injectPartial != DeviceMeshFacts.parameterPlacements &&
      injectPartial.length
        == DeviceMeshFacts.parameterPlacements.length) = true := by
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
    (parameterPlacementWellFormed injectPartial DeviceMeshFacts.meshAxisDegrees &&
      shardedDimDividesAxisDegree injectPartial DeviceMeshFacts.meshAxisDegrees &&
      stridedShardIsAnOuterComposedShard injectPartial &&
      placementSchemaAgreesAcrossRanks
        DeviceMeshFacts.placementSchemaDigest
        DeviceMeshFacts.placementSchemaDigestByRank) = true := by
  rfl

/-- And the one check that cannot survive it does not, stated not hidden. -/
theorem injectedPartialBreaksLocalShapeAgreement :
    localShapeReflectsSharding injectPartial DeviceMeshFacts.meshAxisDegrees
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

end Qwen3Formal.DeviceMeshChecks
