--------------------- MODULE DeviceMeshPayloadSizeInvalid ---------------------
(***************************************************************************)
(* Named negative for collective payload agreement.                        *)
(*                                                                         *)
(* One member of a collective calling it with a different volume than its   *)
(* peers is not a hang: NCCL errors or silently corrupts, so nothing in the *)
(* ordering predicates can see it. The observed run has no such mismatch,   *)
(* so the positive result on its own proves nothing about whether one would *)
(* be caught. This module injects one and requires the agreement invariant  *)
(* to fail.                                                                *)
(*                                                                         *)
(* The injection is an override of the real facts, not a mutated copy of    *)
(* them -- a copy would be a second multi-megabyte generated module whose   *)
(* divergence from the original is invisible in review. Same shape as       *)
(* DeviceMeshPlacementPartialInvalid and DeviceMeshIssueOrderInvalid.               *)
(*                                                                         *)
(* WHAT THE MUTATION IS. The first work, in CollectiveWorkIds order, whose  *)
(* every input and output shape has at least one dimension and whose        *)
(* collective has at least one other member, with the FIRST dimension of    *)
(* every one of its shapes doubled. Doubling both sides keeps the operation's*)
(* implied volume relation intact -- an all-reduce's input still equals its  *)
(* output, an all-gather's output is still its member count times its input *)
(* -- so the only thing the override breaks is agreement between members.   *)
(* That is what makes the violation attributable to one invariant.          *)
(*                                                                         *)
(* The mutated element counts are recomputed from the mutated shapes with   *)
(* the same TotalElements the checker uses, so well-formedness survives by  *)
(* construction. MutationIsIsolated pins that they really did double, so    *)
(* the override cannot become a no-op that still classifies green.          *)
(*                                                                         *)
(* GUARDS. CHOOSE is partial and MutatedWork is a constant definition TLC   *)
(* folds before it checks anything, so on facts with no eligible work an    *)
(* unprotected reading abandons the search with an evaluation error and     *)
(* exit 75 -- output that still prints a state summary. Two separate things *)
(* prevent that, and which does what is measured, not assumed:              *)
(*                                                                         *)
(*   - ThereIsASizedWorkWithAPeer listed FIRST in the cfg is what produces  *)
(*     the named result: these are constant expressions, so TLC reports the *)
(*     first cfg-listed invariant that is FALSE and never forces the        *)
(*     CHOOSE.                                                             *)
(*   - The ELSE "" guard is what keeps the module total if another          *)
(*     invariant is evaluated first, and MutationIsIsolated leads with      *)
(*     domain membership because TLC short-circuits left to right --        *)
(*     without it the guarded form would crash on "Attempted to apply       *)
(*     function" instead of naming an invariant.                            *)
(*                                                                         *)
(* The `cursor = 0` disjunct is the presentation device the sibling         *)
(* negatives use: without it these are constant expressions and TLC reports *)
(* "the invariant is equal to FALSE" with its own exit code rather than a   *)
(* named transition violation the shared checker contract can classify.     *)
(***************************************************************************)
EXTENDS MeshTopology, DeviceMeshFacts

VARIABLE cursor

vars == <<cursor>>

Init == cursor = 0

Next ==
  \/ /\ cursor < 1
     /\ cursor' = cursor + 1
  \/ /\ cursor = 1
     /\ UNCHANGED cursor

Spec == Init /\ [][Next]_vars

WorkPosition ==
  [ work \in SequenceElements(CollectiveWorkIds) |->
      CHOOSE index \in DOMAIN CollectiveWorkIds :
        CollectiveWorkIds[index] = work ]

PeersOf(work) ==
  { peer \in SequenceElements(CollectiveWorkIds) :
      /\ peer # work
      /\ CollectiveId[peer] = CollectiveId[work] }

\* A work whose shapes can be scaled at all, and whose collective has another
\* member -- without one, "one member disagrees with the others" is not a
\* statement these facts can make.
ScalableWorks ==
  { work \in SequenceElements(CollectiveWorkIds) :
      /\ \A index \in DOMAIN CollectivePayloadInputSizes[work] :
           Len(CollectivePayloadInputSizes[work][index]) > 0
      /\ \A index \in DOMAIN CollectivePayloadOutputSizes[work] :
           Len(CollectivePayloadOutputSizes[work][index]) > 0
      /\ PeersOf(work) # {} }

ThereIsASizedWorkWithAPeer == ScalableWorks # {}

MutatedWork ==
  IF ThereIsASizedWorkWithAPeer
  THEN CHOOSE work \in ScalableWorks :
         \A other \in ScalableWorks :
           WorkPosition[work] <= WorkPosition[other]
  ELSE ""

DoubleFirstDim(shapes) ==
  [ index \in DOMAIN shapes |->
      [ dim \in DOMAIN shapes[index] |->
          IF dim = 1
            THEN 2 * shapes[index][dim]
            ELSE shapes[index][dim] ] ]

MutatedInputSizes ==
  [ work \in DOMAIN CollectivePayloadInputSizes |->
      IF work = MutatedWork
        THEN DoubleFirstDim(CollectivePayloadInputSizes[work])
        ELSE CollectivePayloadInputSizes[work] ]

MutatedOutputSizes ==
  [ work \in DOMAIN CollectivePayloadOutputSizes |->
      IF work = MutatedWork
        THEN DoubleFirstDim(CollectivePayloadOutputSizes[work])
        ELSE CollectivePayloadOutputSizes[work] ]

\* Derived from the mutated shapes with the checker's own TotalElements, not
\* patched independently: an override that left the counts and the shapes
\* disagreeing would break well-formedness and the violation would no longer be
\* attributable to the agreement check.
MutatedInputElements ==
  [ work \in DOMAIN CollectivePayloadInputElements |->
      TotalElements(MutatedInputSizes[work]) ]

MutatedOutputElements ==
  [ work \in DOMAIN CollectivePayloadOutputElements |->
      TotalElements(MutatedOutputSizes[work]) ]

\* Guards the control against silently becoming a no-op, and against mutating a
\* whole collective rather than one of its members.
MutationIsIsolated ==
  /\ MutatedWork \in DOMAIN CollectivePayloadInputSizes
  /\ PeersOf(MutatedWork) # {}
  /\ MutatedInputElements[MutatedWork]
       = 2 * CollectivePayloadInputElements[MutatedWork]
  /\ MutatedOutputElements[MutatedWork]
       = 2 * CollectivePayloadOutputElements[MutatedWork]
  /\ \A work \in DOMAIN CollectivePayloadInputSizes :
       work # MutatedWork =>
         /\ MutatedInputSizes[work] = CollectivePayloadInputSizes[work]
         /\ MutatedOutputSizes[work] = CollectivePayloadOutputSizes[work]
         /\ MutatedInputElements[work] = CollectivePayloadInputElements[work]
         /\ MutatedOutputElements[work] = CollectivePayloadOutputElements[work]

\* The payload facts are still well formed under the override, so the reported
\* violation is a disagreement between members and not a broken projection.
StructureSurvivesTheMutation ==
  \/ cursor = 0
  \/ CollectivePayloadWellFormed(
       CollectiveWorkIds,
       CollectivePayloadComm,
       MutatedInputSizes,
       MutatedOutputSizes,
       CollectivePayloadInputDtypes,
       CollectivePayloadOutputDtypes,
       MutatedInputElements,
       MutatedOutputElements
     )

\* And the other collective contracts still hold. The issue-order agreement is
\* the load-bearing member of this list: a volume mismatch leaves the order of
\* operations on every communicator exactly as observed, which is precisely why
\* the ordering property cannot stand in for the payload one.
OtherCollectiveChecksSurviveTheMutation ==
  \/ cursor = 0
  \/ /\ CollectivePayloadSizeRelationHolds(
          CollectiveWorkIds,
          CollectiveOperation,
          CollectiveMembers,
          CollectivePayloadSizeRelation,
          MutatedInputElements,
          MutatedOutputElements
        )
     /\ PayloadCommKeyIsCommunicatorIdentity(
          CollectiveWorkIds,
          CollectivePayloadComm,
          CollectiveComm,
          CollectiveMembers
        )
     /\ PerCommunicatorIssueOrderAgreement(
          CollectiveWorkIds,
          CollectiveRank,
          CollectiveComm,
          CollectiveOperation,
          CollectiveMembers,
          CollectiveIssueOrder
        )

DeviceMeshCollectivePayloadAgreement ==
  \/ cursor = 0
  \/ CollectivePayloadAgreement(
       CollectiveWorkIds,
       CollectiveId,
       CollectivePayloadInputDtypes,
       CollectivePayloadOutputDtypes,
       MutatedInputElements,
       MutatedOutputElements
     )

=============================================================================
