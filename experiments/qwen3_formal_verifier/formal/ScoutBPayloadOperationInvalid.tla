------------------- MODULE ScoutBPayloadOperationInvalid -------------------
(***************************************************************************)
(* Named negative for the payload volume relation.                         *)
(*                                                                         *)
(* This is the failure class the relation exists for: an EXPORTER that      *)
(* mislabelled an operation. That is the same class of error as the         *)
(* eight-versus-four communicator confusion, and it is invisible to every   *)
(* other predicate here, because an exporter bug mislabels the operation    *)
(* the SAME way on every member. Per-communicator order agreement then      *)
(* still holds -- all four ranks agree on the wrong label -- and only the   *)
(* volume relation notices that an operation which was labelled as leaving  *)
(* the volume unchanged in fact multiplied it by the member count.          *)
(*                                                                         *)
(* WHAT THE MUTATION IS. The first collective, in CollectiveWorkIds order,  *)
(* whose operation changes volume and whose observed input and output        *)
(* volumes really differ, relabelled as "all_reduce" on EVERY one of its     *)
(* member works, with the exported relation label moved to the one          *)
(* "all_reduce" implies -- exactly what the exporter would have written had  *)
(* it read the operation wrongly. Nothing else changes: not a shape, not a   *)
(* dtype, not an element count, not an issue order, not CollectiveId.        *)
(* Leaving CollectiveId alone keeps the member grouping identical to the     *)
(* observed run, so the mutation is the label and only the label.            *)
(*                                                                         *)
(* Because the relabelling is symmetric across members,                     *)
(* PerCommunicatorIssueOrderAgreement is in the SURVIVOR set rather than    *)
(* being broken by the override. That is the whole argument for adding the   *)
(* volume relation: the ordering property cannot stand in for it.           *)
(*                                                                         *)
(* The CHOOSE guard and the cfg ordering are the same contract as           *)
(* ScoutBPayloadSizeInvalid; see that module's header for which of the two   *)
(* does what.                                                              *)
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

WorkPosition ==
  [ work \in SequenceElements(CollectiveWorkIds) |->
      CHOOSE index \in DOMAIN CollectiveWorkIds :
        CollectiveWorkIds[index] = work ]

\* An operation whose implied relation is not "input_equals_output", AND whose
\* observed volumes really differ. The second half is what keeps the negative
\* non-vacuous: relabelling a volume-changing operation whose volumes happened
\* to be equal would refute nothing.
VolumeChangingWorks ==
  { work \in SequenceElements(CollectiveWorkIds) :
      /\ ImpliedSizeRelation(CollectiveOperation[work]) # "input_equals_output"
      /\ ImpliedSizeRelation(CollectiveOperation[work])
           \in PayloadSizeRelations
      /\ CollectivePayloadInputElements[work]
           # CollectivePayloadOutputElements[work] }

ThereIsAVolumeChangingCollective == VolumeChangingWorks # {}

MutatedWork ==
  IF ThereIsAVolumeChangingCollective
  THEN CHOOSE work \in VolumeChangingWorks :
         \A other \in VolumeChangingWorks :
           WorkPosition[work] <= WorkPosition[other]
  ELSE ""

\* Every member work of that one collective, which is what an exporter bug
\* would have mislabelled: the bug is in the projection, not in one rank.
MutatedCollectiveId ==
  IF MutatedWork \in DOMAIN CollectiveId THEN CollectiveId[MutatedWork] ELSE ""

MutatedWorks ==
  { work \in SequenceElements(CollectiveWorkIds) :
      /\ MutatedWork \in DOMAIN CollectiveId
      /\ CollectiveId[work] = MutatedCollectiveId }

MutatedOperation ==
  [ work \in DOMAIN CollectiveOperation |->
      IF work \in MutatedWorks
        THEN "all_reduce"
        ELSE CollectiveOperation[work] ]

MutatedSizeRelation ==
  [ work \in DOMAIN CollectivePayloadSizeRelation |->
      IF work \in MutatedWorks
        THEN ImpliedSizeRelation("all_reduce")
        ELSE CollectivePayloadSizeRelation[work] ]

\* Guards the control against silently becoming a no-op, and pins that the
\* relabelling covers every member so the ordering property is not what fails.
MutationIsIsolated ==
  /\ MutatedWork \in DOMAIN CollectiveOperation
  /\ CollectiveOperation[MutatedWork] # "all_reduce"
  /\ MutatedOperation[MutatedWork] = "all_reduce"
  /\ MutatedWorks = { work \in SequenceElements(CollectiveWorkIds) :
                        CollectiveId[work] = CollectiveId[MutatedWork] }
  /\ { CollectiveRank[work] : work \in MutatedWorks }
       = CollectiveMembers[MutatedWork]
  /\ \A work \in DOMAIN CollectiveOperation :
       work \notin MutatedWorks =>
         /\ MutatedOperation[work] = CollectiveOperation[work]
         /\ MutatedSizeRelation[work] = CollectivePayloadSizeRelation[work]

\* The payload facts themselves are untouched: no shape, dtype or element count
\* moved, so the reported violation is the label against the volume and nothing
\* else.
StructureSurvivesTheMutation ==
  \/ cursor = 0
  \/ /\ CollectivePayloadWellFormed(
          CollectiveWorkIds,
          CollectivePayloadComm,
          CollectivePayloadInputSizes,
          CollectivePayloadOutputSizes,
          CollectivePayloadInputDtypes,
          CollectivePayloadOutputDtypes,
          CollectivePayloadInputElements,
          CollectivePayloadOutputElements
        )
     /\ CollectivePayloadAgreement(
          CollectiveWorkIds,
          CollectiveId,
          CollectivePayloadInputDtypes,
          CollectivePayloadOutputDtypes,
          CollectivePayloadInputElements,
          CollectivePayloadOutputElements
        )
     /\ PayloadCommKeyIsCommunicatorIdentity(
          CollectiveWorkIds,
          CollectivePayloadComm,
          CollectiveComm,
          CollectiveMembers
        )

\* THE point of this control: the ordering property still holds over the
\* mislabelled facts. Asserted, not asserted about -- if a future change made a
\* symmetric relabelling visible to the ordering check, this invariant would
\* fail and the claim would stop being made.
IssueOrderAgreementSurvivesTheMislabelling ==
  \/ cursor = 0
  \/ PerCommunicatorIssueOrderAgreement(
       CollectiveWorkIds,
       CollectiveRank,
       CollectiveComm,
       MutatedOperation,
       CollectiveMembers,
       CollectiveIssueOrder
     )

ScoutBCollectivePayloadSizeRelation ==
  \/ cursor = 0
  \/ CollectivePayloadSizeRelationHolds(
       CollectiveWorkIds,
       MutatedOperation,
       CollectiveMembers,
       MutatedSizeRelation,
       CollectivePayloadInputElements,
       CollectivePayloadOutputElements
     )

=============================================================================
