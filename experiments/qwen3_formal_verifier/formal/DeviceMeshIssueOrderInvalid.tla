-------------------------- MODULE DeviceMeshIssueOrderInvalid --------------------------
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
(* DeviceMeshInvalid: these are constant expressions, so without it TLC reports *)
(* "the invariant is equal to FALSE" with a distinct exit code rather than  *)
(* a named transition violation the shared checker contract can classify.   *)
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

(***************************************************************************)
(* Rank 0's first collective in issue order, derived rather than            *)
(* hand-written.                                                           *)
(*                                                                         *)
(* CHOOSE is partial, and MutatedWork is a constant definition TLC folds    *)
(* before it checks anything, so on a trace where rank 0 issued no          *)
(* collective an unprotected reading abandons the search with an evaluation *)
(* error and exit 75 -- output that still prints a state summary and a      *)
(* "Finished in" line, so only the exit-status pin rejects it. Two separate *)
(* things protect against that, and an earlier version of this comment      *)
(* credited the wrong one. MEASURED, both ways:                             *)
(*                                                                         *)
(*   - Rank0HasACollective listed FIRST is what produces the named result.  *)
(*     These are constant expressions, so TLC reports the first cfg-listed  *)
(*     invariant that is FALSE and never forces the CHOOSE. In that         *)
(*     configuration the ELSE "" guard changes nothing: guarded and         *)
(*     unguarded both report Rank0HasACollective equal to FALSE, exit 151.  *)
(*     The cfg order is load-bearing and is pinned by a test.               *)
(*   - The guard is what keeps the module total if another invariant is     *)
(*     evaluated first. With the sentinel removed from the cfg, the         *)
(*     unguarded form crashes at exit 75 on the CHOOSE while the guarded    *)
(*     form reports MutationIsIsolated by name.                             *)
(*                                                                         *)
(* The second bullet only became true when MutationIsIsolated grew its      *)
(* domain conjunct. Without it the guarded form also crashed at exit 75, on *)
(* "Attempted to apply function" rather than on the CHOOSE, because the ""  *)
(* sentinel is not in CollectiveOperation's domain.                         *)
(*                                                                         *)
(* SingleRankRefineBad's ControlEventsArePresent is a related shape but not the *)
(* same case: its indices are read while Observed is being constructed, so  *)
(* no invariant can be reported ahead of them and its guard is load-bearing *)
(* unconditionally.                                                        *)
(***************************************************************************)
Rank0Works ==
  { work \in SequenceElements(CollectiveWorkIds) :
      CollectiveRank[work] = 0 }

Rank0HasACollective == Rank0Works # {}

MutatedWork ==
  IF Rank0HasACollective
  THEN CHOOSE work \in Rank0Works :
         \A other \in Rank0Works :
           CollectiveIssueOrder[work] <= CollectiveIssueOrder[other]
  ELSE ""

MutatedOperation ==
  [work \in DOMAIN CollectiveOperation |->
     IF work = MutatedWork
       THEN IF CollectiveOperation[work] = "all_gather"
              THEN "reduce_scatter"
              ELSE "all_gather"
       ELSE CollectiveOperation[work]]

\* Guards the control against silently becoming a no-op: the operation must
\* actually change, and nothing else may.
\*
\* The domain conjunct comes first on purpose. TLC short-circuits a conjunction
\* left to right, so it is what lets this report a named violation instead of
\* crashing on "Attempted to apply function" when the "" sentinel is in play.
MutationIsIsolated ==
  /\ MutatedWork \in DOMAIN CollectiveOperation
  /\ MutatedOperation[MutatedWork] # CollectiveOperation[MutatedWork]
  /\ \A work \in DOMAIN CollectiveOperation :
       work # MutatedWork => MutatedOperation[work] = CollectiveOperation[work]

DeviceMeshPerCommunicatorIssueOrder ==
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
