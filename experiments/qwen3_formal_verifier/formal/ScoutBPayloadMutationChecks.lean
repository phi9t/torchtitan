import ScoutBFacts

/-
The two derived negatives for collective payload identity.

EVALUATION OF THE OBSERVED TRACE, not a theorem about the protocol: every
definition here is Bool-valued over the literal ScoutBFacts data, or over a
derived override of it, and is closed with `rfl`.

The observed run has no payload mismatch and no mislabelled operation, so the
positive results in ScoutBPayloadChecks prove nothing on their own about whether
either would be caught. These injections are derived overrides of the real facts
-- the same shape as ScoutBPlacementChecks -- rather than checked-in mutated
copies of a multi-megabyte generated module.

Separate module from the positives for a measured cost reason; see that module's
header.

WHAT THIS CHECKER CANNOT SAY. The TLA controls put
`PerCommunicatorIssueOrderAgreement` in both survivor sets, which is the whole
argument for having these properties: neither mutation is visible to the ordering
check. The Lean collective observations carry no communicator or operation, so
that survivor cannot be restated here. It is asserted in
ScoutBPayloadSizeInvalid.tla and ScoutBPayloadOperationInvalid.tla, and not
claimed twice.
-/

namespace Qwen3Formal.ScoutBChecks

open Qwen3Formal

set_option maxHeartbeats 0
set_option maxRecDepth 100000

/--
The first work whose shapes can be scaled at all and whose collective has more
than one member. Without a peer, "one member disagrees with the others" is not a
statement these facts can make.
-/
def firstScalableWorkIndex? : Option Nat :=
  ScoutBFacts.collectivePayloads.findIdx? fun item =>
    item.members.length > 1 &&
    item.inputSizes.all (fun shape => !shape.isEmpty) &&
    item.outputSizes.all (fun shape => !shape.isEmpty)

def doubleFirstDim (shapes : List (List Nat)) : List (List Nat) :=
  shapes.map fun shape =>
    match shape with
    | [] => []
    | first :: rest => (2 * first) :: rest

/--
One member of a collective calling it with twice the volume of its peers.
Doubling BOTH sides keeps the operation's implied relation intact, so the only
thing the override breaks is agreement between members. The element counts are
recomputed with the checker's own `totalElements`, so well-formedness survives by
construction rather than by a second hand-patched number.

The collective ROW is deliberately left alone: the row is what the other members
still report, so the mismatch is exactly "this member disagrees with its peers".
-/
def injectSizeMismatch : List CollectivePayload :=
  match firstScalableWorkIndex? with
  | none => ScoutBFacts.collectivePayloads
  | some index =>
      ScoutBFacts.collectivePayloads.modify index fun item =>
        let inputs := doubleFirstDim item.inputSizes
        let outputs := doubleFirstDim item.outputSizes
        { item with
          inputSizes := inputs,
          outputSizes := outputs,
          inputElements := totalElements inputs,
          outputElements := totalElements outputs }

/-- Guards the control against silently becoming a no-op. -/
theorem injectedSizeMismatchIsIsolated :
    (injectSizeMismatch != ScoutBFacts.collectivePayloads &&
      injectSizeMismatch.length
        == ScoutBFacts.collectivePayloads.length) = true := by
  rfl

/-- The injected mismatch must be caught by the agreement check. -/
theorem rejectsInjectedSizeMismatch :
    collectivePayloadAgreement
      injectSizeMismatch ScoutBFacts.payloadCollectiveRows = false := by
  rfl

/--
And by nothing else that reads the mutated works, so the violation is
attributable: the payload is still well formed and still satisfies the relation
its operation implies.

`payloadCommKeyIsCommunicatorIdentity` is NOT listed. It reads the two tables and
not the works at all, so it survives this override trivially, and claiming it as
a survivor would overstate what the control shows. The TLA counterpart asserts
the keying and the issue-order agreement over the mutated facts.
-/
theorem injectedSizeMismatchPassesTheOtherPayloadChecks :
    (collectivePayloadWellFormed injectSizeMismatch &&
      collectivePayloadSizeRelationHolds injectSizeMismatch) = true := by
  rfl

/--
The first collective whose operation changes volume and whose observed volumes
really differ. The second condition keeps the mislabelling negative non-vacuous:
relabelling a volume-changing operation whose volumes happened to be equal would
refute nothing.
-/
def firstVolumeChangingPayload? : Option CollectivePayload :=
  ScoutBFacts.collectivePayloads.find? fun item =>
    impliedSizeRelation item.operation != "input_equals_output" &&
    payloadSizeRelations.contains (impliedSizeRelation item.operation) &&
    item.inputElements != item.outputElements

def mislabelledCollectiveId : String :=
  match firstVolumeChangingPayload? with
  | none => ""
  | some target => target.collectiveId

/--
One collective's operation relabelled on EVERY member work, with the exported
relation label moved to the one the wrong operation implies -- exactly what the
exporter would have written had it read the operation wrongly. No shape, dtype or
element count moves, and `collectiveId` is left alone so the member grouping is
identical to the observed run.
-/
def mislabelOperation : List CollectivePayload :=
  ScoutBFacts.collectivePayloads.map fun item =>
    if mislabelledCollectiveId != "" &&
        item.collectiveId == mislabelledCollectiveId then
      { item with
        operation := "all_reduce",
        sizeRelation := impliedSizeRelation "all_reduce" }
    else item

/-- The row moves with the works, because an exporter mislabelling would write
both. Leaving the row behind would break agreement instead, and the violation
would no longer be attributable to the volume relation. -/
def mislabelledRows : List PayloadCollectiveRow :=
  ScoutBFacts.payloadCollectiveRows.map fun row =>
    if mislabelledCollectiveId != "" &&
        row.collectiveId == mislabelledCollectiveId then
      { row with
        operation := "all_reduce",
        sizeRelation := impliedSizeRelation "all_reduce" }
    else row

/--
Guards the control against becoming a no-op, and pins that the relabelling covers
every member of exactly one collective. A mislabelling on a single member would
be a different and weaker control, because an asymmetric one is visible to the
ordering check as well.
-/
theorem mislabelledEveryMemberOfOneCollective :
    (mislabelOperation != ScoutBFacts.collectivePayloads &&
      mislabelOperation.length == ScoutBFacts.collectivePayloads.length &&
      ((mislabelOperation.zip ScoutBFacts.collectivePayloads).filter
        (fun pair => pair.fst.operation != pair.snd.operation)).length
        == (match firstVolumeChangingPayload? with
            | none => 0
            | some target => target.members.length)) = true := by
  rfl

/-- The mislabelling must be caught by the volume relation. -/
theorem rejectsMislabelledOperation :
    collectivePayloadSizeRelationHolds mislabelOperation = false := by
  rfl

/--
And by nothing else: the payload bytes are untouched, so well-formedness, member
agreement against the equally relabelled rows, and the communicator key all still
hold over the mislabelled facts.
-/
theorem mislabelledOperationPassesTheOtherPayloadChecks :
    (collectivePayloadWellFormed mislabelOperation &&
      collectivePayloadAgreement mislabelOperation mislabelledRows &&
      payloadCommKeyIsCommunicatorIdentity
        mislabelledRows ScoutBFacts.payloadCommTable) = true := by
  rfl

#print axioms injectedSizeMismatchIsIsolated
#print axioms rejectsInjectedSizeMismatch
#print axioms injectedSizeMismatchPassesTheOtherPayloadChecks
#print axioms mislabelledEveryMemberOfOneCollective
#print axioms rejectsMislabelledOperation
#print axioms mislabelledOperationPassesTheOtherPayloadChecks

end Qwen3Formal.ScoutBChecks
