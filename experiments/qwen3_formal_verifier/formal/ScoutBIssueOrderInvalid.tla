-------------------------- MODULE ScoutBIssueOrderInvalid --------------------------
(***************************************************************************)
(* Named negative for per-communicator issue-order agreement.              *)
(*                                                                         *)
(* The mutation is expressed as an override of the real facts rather than  *)
(* as a mutated copy of them. A copy would be a second 1.6 MB generated     *)
(* module whose divergence from the original is invisible in review; this   *)
(* way the change is three lines, and because the target is DERIVED from    *)
(* the facts it cannot drift away from the observed run.                    *)
(*                                                                         *)
(* The target is rank 0's first collective in issue order. Flipping its     *)
(* operation makes rank 0's issue sequence on that communicator disagree    *)
(* with its peer's at the same position, which is the NCCL violation that   *)
(* hangs a job.                                                             *)
(*                                                                         *)
(* The `cursor = 0` disjunct is the presentation device already used by     *)
(* ScoutBInvalid: these are constant expressions, so without it TLC reports *)
(* "the invariant is equal to FALSE" with a distinct exit code rather than  *)
(* a named transition violation the shared checker contract can classify.   *)
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

\* Rank 0's first collective in issue order, derived rather than hand-written.
MutatedWork ==
  CHOOSE work \in SequenceElements(CollectiveWorkIds) :
    /\ CollectiveRank[work] = 0
    /\ \A other \in SequenceElements(CollectiveWorkIds) :
         CollectiveRank[other] = 0 =>
           CollectiveIssueOrder[work] <= CollectiveIssueOrder[other]

MutatedOperation ==
  [work \in DOMAIN CollectiveOperation |->
     IF work = MutatedWork
       THEN IF CollectiveOperation[work] = "all_gather"
              THEN "reduce_scatter"
              ELSE "all_gather"
       ELSE CollectiveOperation[work]]

\* Guards the control against silently becoming a no-op: the operation must
\* actually change, and nothing else may.
MutationIsIsolated ==
  /\ MutatedOperation[MutatedWork] # CollectiveOperation[MutatedWork]
  /\ \A work \in DOMAIN CollectiveOperation :
       work # MutatedWork => MutatedOperation[work] = CollectiveOperation[work]

ScoutBPerCommunicatorIssueOrder ==
  \/ cursor = 0
  \/ PerCommunicatorIssueOrderAgreement(
       CollectiveWorkIds,
       CollectiveRank,
       CollectiveComm,
       MutatedOperation,
       CollectiveMembers,
       CollectiveIssueOrder
     )

=============================================================================
