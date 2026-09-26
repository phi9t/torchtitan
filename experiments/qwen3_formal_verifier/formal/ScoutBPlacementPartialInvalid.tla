--------------------- MODULE ScoutBPlacementPartialInvalid ---------------------
(***************************************************************************)
(* Named negative for the no-Partial-placement invariant.                  *)
(*                                                                         *)
(* A Partial placement on a parameter the optimizer steps is an unreduced   *)
(* value: the reduction never happened and the update is wrong without any  *)
(* error being raised. The observed run has zero Partial placements, so the *)
(* positive check on its own proves nothing about whether one would be      *)
(* caught. This module injects one and requires the invariant to fail.      *)
(*                                                                         *)
(* The injection is an override of the real facts, not a mutated copy of    *)
(* them: a copy would be a second 1.6 MB generated module whose divergence  *)
(* from the original is invisible in review, and because the target is      *)
(* DERIVED from the facts it cannot drift away from the observed run.       *)
(* Same shape as ScoutBIssueOrderInvalid.                                  *)
(*                                                                         *)
(* The target is the first DP-shard placement in PlacementIds order, which  *)
(* is the outer FSDP axis -- the axis whose reduce-scatter produces the     *)
(* gradient the optimizer consumes, so a Partial there is the realistic     *)
(* failure rather than an arbitrary one.                                    *)
(*                                                                         *)
(* The `cursor = 0` disjunct is the presentation device ScoutBInvalid and   *)
(* ScoutBIssueOrderInvalid already use: these are constant expressions, so  *)
(* without it TLC reports "the invariant is equal to FALSE" with a distinct *)
(* exit code rather than a named transition violation the shared checker    *)
(* contract can classify.                                                  *)
(***************************************************************************)
EXTENDS ScoutDistributed, ScoutBFacts

VARIABLE cursor

vars == <<cursor>>

Init == cursor = 0

Next ==
  \/ /\ cursor < 1
     /\ cursor' = cursor + 1
  \/ /\ cursor = 1
     /\ UNCHANGED cursor

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* The derived target.                                                     *)
(*                                                                         *)
(* CHOOSE is partial and MutatedPlacement is a constant definition TLC      *)
(* folds before it checks anything, so on a trace with no DP-shard          *)
(* placement an unprotected reading abandons the search with an evaluation  *)
(* error and exit 75 -- output that still prints a state summary, so only   *)
(* the exit-status pin would reject it. Two separate things prevent that,   *)
(* and it is worth being exact about which does what. MEASURED, both ways:  *)
(*                                                                         *)
(*   - ThereIsADpShardPlacement listed FIRST is what produces the named     *)
(*     result. These are constant expressions, so TLC reports the first     *)
(*     cfg-listed invariant that is FALSE and never forces the CHOOSE. In   *)
(*     that configuration the ELSE "" guard changes nothing: guarded and    *)
(*     unguarded both report ThereIsADpShardPlacement equal to FALSE with   *)
(*     exit 151. The cfg pins that order and says why.                      *)
(*   - The guard is what keeps the module total if another invariant is     *)
(*     evaluated first. With the sentinel removed from the cfg, the         *)
(*     unguarded form crashes at exit 75 on the CHOOSE while the guarded    *)
(*     form reports MutationIsIsolated by name. That is what               *)
(*     test_the_partial_injection_guard_is_load_bearing pins, by stripping  *)
(*     the guard and requiring the crash to come back.                      *)
(*                                                                         *)
(* MutationIsIsolated leads with MutatedPlacement \in DOMAIN PlacementKind  *)
(* for the same reason: TLC short-circuits left to right, so without it the *)
(* guarded form would crash on "Attempted to apply function" instead of     *)
(* naming an invariant. ScoutBIssueOrderInvalid was missing that conjunct   *)
(* and has been given it.                                                   *)
(*                                                                         *)
(* ScoutARefineBad's ControlEventsArePresent is a related shape but not the *)
(* same case: its indices are read while Observed is being constructed, so  *)
(* no invariant can be reported ahead of them and its guard is load-bearing *)
(* unconditionally.                                                        *)
(***************************************************************************)
PlacementPosition ==
  [ placement \in SequenceElements(PlacementIds) |->
      CHOOSE index \in DOMAIN PlacementIds :
        PlacementIds[index] = placement ]

DpShardPlacements ==
  { placement \in SequenceElements(PlacementIds) :
      /\ PlacementAxis[placement] = "dp_shard"
      /\ PlacementKind[placement] = "shard" }

ThereIsADpShardPlacement == DpShardPlacements # {}

MutatedPlacement ==
  IF ThereIsADpShardPlacement
  THEN CHOOSE placement \in DpShardPlacements :
         \A other \in DpShardPlacements :
           PlacementPosition[placement] <= PlacementPosition[other]
  ELSE ""

MutatedPlacementKind ==
  [ placement \in DOMAIN PlacementKind |->
      IF placement = MutatedPlacement
        THEN "partial"
        ELSE PlacementKind[placement] ]

\* PlacementShardDim's domain is exactly the shard placements, so the override
\* must drop the mutated placement from it as well. Without that the mutation
\* would also break ParameterPlacementWellFormed, and the violation would no
\* longer be attributable to the Partial check alone. The Lean injection does
\* the same thing by setting shardDim := none.
MutatedPlacementShardDim ==
  [ placement \in DOMAIN PlacementShardDim \ {MutatedPlacement} |->
      PlacementShardDim[placement] ]

\* Guards the control against silently becoming a no-op: the kind must actually
\* become "partial" from something else, and nothing else may change.
MutationIsIsolated ==
  /\ MutatedPlacement \in DOMAIN PlacementKind
  /\ PlacementKind[MutatedPlacement] # "partial"
  /\ MutatedPlacementKind[MutatedPlacement] = "partial"
  /\ \A placement \in DOMAIN PlacementKind :
       placement # MutatedPlacement
         => MutatedPlacementKind[placement] = PlacementKind[placement]
  /\ DOMAIN MutatedPlacementShardDim
       = DOMAIN PlacementShardDim \ {MutatedPlacement}
  /\ \A placement \in DOMAIN MutatedPlacementShardDim :
       MutatedPlacementShardDim[placement] = PlacementShardDim[placement]

\* The mutation must be caught by the Partial check and by nothing else, so the
\* violation is attributable. Asserted over the MUTATED facts: if the override
\* also broke well-formedness, a reader could not tell which contract the
\* injected Partial actually violated.
StructureSurvivesTheMutation ==
  \/ cursor = 0
  \/ ParameterPlacementWellFormed(
       ParameterNames,
       ParameterMeshAxes,
       ParameterGlobalShape,
       ParameterLocalShape,
       PlacementIds,
       PlacementParameter,
       PlacementAxis,
       MutatedPlacementKind,
       PlacementStrided,
       MutatedPlacementShardDim,
       MeshAxisDegree
     )

\* And the other placement contracts still hold, so the Partial check is the
\* only thing the injection trips.
\*
\* LocalShapeReflectsSharding is deliberately absent, and its absence is the
\* honest part: the observed local shape of the mutated parameter is half its
\* global shape BECAUSE the dp_shard axis shards it, so an override that stops
\* that axis from sharding without also rewriting the local shape contradicts
\* that check too. Listing it here would be a claim this control cannot make.
\* The cfg lists ScoutBNoPartialAtOptimizer, so that is the invariant TLC
\* reports, and the survivors below are the ones a reader can rely on.
OtherPlacementChecksSurviveTheMutation ==
  \/ cursor = 0
  \/ /\ ShardedDimDividesAxisDegree(
          PlacementIds,
          PlacementParameter,
          PlacementAxis,
          MutatedPlacementKind,
          MutatedPlacementShardDim,
          ParameterGlobalShape,
          MeshAxisDegree
        )
     /\ StridedShardIsAnOuterComposedShard(
          ParameterMeshAxes,
          PlacementIds,
          PlacementParameter,
          PlacementAxis,
          MutatedPlacementKind,
          PlacementStrided,
          MutatedPlacementShardDim
        )
     /\ PlacementSchemaAgreesAcrossRanks(
          RankSet, PlacementSchemaDigestByRank, PlacementSchemaDigest
        )

ScoutBNoPartialAtOptimizer ==
  \/ cursor = 0
  \/ /\ OptimizerBoundaryObserved(
          RankSet, EventIds, EventRank, EventKind, EventOrder)
     /\ NoPartialParameterPlacement(PlacementIds, MutatedPlacementKind)

=============================================================================
