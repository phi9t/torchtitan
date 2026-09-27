import ScoutBFacts

/-
Collective payload identity over the observed trace.

The two derived negatives live in ScoutBPayloadMutationChecks, and the split is
a cost decision, MEASURED: a `rfl` over the agreement predicate at the observed
scale of 432 works takes about 26s of kernel reduction and the well-formedness
predicate about 13s, with no caching between theorems, so evaluating the
positives and both mutations in one module runs to roughly three minutes. Split,
the two halves run in parallel and each stays well inside its budget.

EVALUATION OF THE OBSERVED TRACE, not a theorem about the protocol: every
definition here is Bool-valued over the literal ScoutBFacts data and is closed
with `rfl`.

The observed run has no payload mismatch and no mislabelled operation, so
`payloadAgreementObserved` and `payloadSizeRelationObserved` on their own prove
nothing about whether either would be caught. The two injections below are
derived overrides of the real facts -- the same shape as ScoutBPlacementChecks --
rather than checked-in mutated copies of a multi-megabyte generated module.

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

theorem payloadWellFormedObserved :
    collectivePayloadWellFormed ScoutBFacts.collectivePayloads = true := by
  rfl

theorem payloadCoversCollectivesObserved :
    payloadCoversCollectives
      ScoutBFacts.collectivePayloads
      ScoutBFacts.collectiveLifecycle = true := by
  rfl

theorem payloadCommKeyObserved :
    payloadCommKeyIsCommunicatorIdentity
      ScoutBFacts.payloadCollectiveRows
      ScoutBFacts.payloadCommTable = true := by
  rfl

theorem payloadAgreementObserved :
    collectivePayloadAgreement
      ScoutBFacts.collectivePayloads
      ScoutBFacts.payloadCollectiveRows = true := by
  rfl

theorem payloadSizeRelationObserved :
    collectivePayloadSizeRelationHolds
      ScoutBFacts.collectivePayloads = true := by
  rfl

#print axioms payloadWellFormedObserved
#print axioms payloadCoversCollectivesObserved
#print axioms payloadCommKeyObserved
#print axioms payloadAgreementObserved
#print axioms payloadSizeRelationObserved

end Qwen3Formal.ScoutBChecks
