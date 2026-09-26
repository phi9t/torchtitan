-------------------------------- MODULE ScoutBModel --------------------------------
(***************************************************************************)
(* An abstract model of collective execution on a 2x2 DP-shard x TP mesh.  *)
(*                                                                         *)
(* This models the PROTOCOL, not the observed run. Ranks issue collectives  *)
(* freely and concurrently, so TLC explores rank skew, and the invariants   *)
(* hold for every admitted schedule up to the bound declared below.         *)
(*                                                                         *)
(* ------------------------------------------------------------------------ *)
(* CLAIM INVENTORY. Each line names a configuration and the exact statement *)
(* it establishes. Nothing outside this list is claimed.                    *)
(*                                                                         *)
(*   ScoutBModel.cfg          TypeOK, RendezvousOpAgreement,               *)
(*                            RendezvousMembership, CommFifo,              *)
(*                            DeadlockFreedom and StuckImpliesAllDone hold *)
(*                            for every interleaving of 4 ranks and 8       *)
(*                            communicators in which each rank issues at    *)
(*                            most MaxIssues = 2 collectives, with issue    *)
(*                            skew between ranks unbounded within that.     *)
(*   ScoutBModelReach.cfg     that bound is non-vacuous: AllDone is         *)
(*                            reachable, so the invariants above are not    *)
(*                            true merely because a guard is unsatisfiable. *)
(*   ScoutBModelDivergent.cfg relaxing ONLY communicator-site agreement --  *)
(*                            operations still agree at every position and  *)
(*                            NCCL's own guard stays on -- makes            *)
(*                            DeadlockFreedom false.                        *)
(*   ScoutBModelWitness.cfg   and the state reached there has the right     *)
(*                            shape: NoCrossCommunicatorCycleWitness is     *)
(*                            false, so the cycle spans at least two        *)
(*                            distinct communicators, the operations agree, *)
(*                            issue counts are equal on every communicator  *)
(*                            of the chain, and every rank has spent its    *)
(*                            whole budget. Same constants as the divergent *)
(*                            cfg; only the invariant differs.              *)
(*   ScoutBModelOpMismatch.cfg relaxing operation agreement instead makes    *)
(*                            NoOpMismatchHang false: a DIFFERENT hazard,    *)
(*                            reported under a different result token.       *)
(*   ScoutBModelStreamShape.cfg StreamEdgeIsInert is false: some rank holds  *)
(*                            two DISTINCT communicators whose issues can   *)
(*                            land on one stream, so the stream-head        *)
(*                            conjunct is not implied by per-communicator   *)
(*                            FIFO. The runner reruns the same cfg against  *)
(*                            a module where StreamOfIssue(e) == e.comm,    *)
(*                            and there it must HOLD.                       *)
(*   ScoutBModelUnguarded.cfg dropping NCCL's matching requirement turns     *)
(*                            that hang into a mismatched rendezvous:        *)
(*                            RendezvousOpAgreement is false.                *)
(*                                                                         *)
(* The bound is MaxIssues = 2 per rank. That is NOT a claim about the        *)
(* observed 108-collective-per-rank run. The observed run is checked         *)
(* separately, by ScoutBIssueOrderInvalid against ScoutBFacts, for the       *)
(* per-communicator issue-order property only. Composing the two gives:      *)
(* the guards are safe for all schedules up to the bound, and the observed    *)
(* run satisfies per-communicator agreement. Closing the gap needs an         *)
(* inductive invariant (Apalache or TLAPS), which is not attempted here.     *)
(*                                                                         *)
(* ------------------------------------------------------------------------ *)
(* THE INSTANCE. Eight communicators, one per canonical_id observed in the   *)
(* 2x2 run:                                                                *)
(*                                                                         *)
(*   tp01 tp23           tp:{0,1}:mesh_tp, tp:{2,3}:mesh_tp                 *)
(*   fsdp02 fsdp13       dp_shard:{0,2}:mesh_fsdp, dp_shard:{1,3}:mesh_fsdp *)
(*   batch02 batch13     dp_shard:{0,2}:mesh_batch, ...                     *)
(*   loss02 loss13       dp_shard:{0,2}:mesh_loss_mesh, ...                 *)
(*                                                                         *)
(* mesh_fsdp, mesh_batch and mesh_loss_mesh are three SEPARATE communicators *)
(* spanning the same rank set, so rank 0 holds three communicators over      *)
(* {0,2}. That overlap is the whole point: it is the only way a rank can put *)
(* two DIFFERENT communicators on one stream, and therefore the only way a   *)
(* cross-communicator wait edge can exist. The observed runtime_pg_id is     *)
(* per-rank local and maps to two different communicators, so it is never a  *)
(* key here.                                                                *)
(*                                                                         *)
(* Admissible operations are supplied PER COMMUNICATOR rather than derived   *)
(* from a class, because the observed run does not factor that way:          *)
(* mesh_fsdp carries all_gather, reduce_scatter AND all_reduce (the gradient *)
(* norm), while mesh_tp, mesh_batch and mesh_loss_mesh carry all_reduce      *)
(* only.                                                                    *)
(*                                                                         *)
(* ------------------------------------------------------------------------ *)
(* STREAMS. StreamOfIssue is a DESIGN CONSTANT, not a reading of the trace.  *)
(* It encodes PyTorch's behaviour: FSDP2 runs all-gather and reduce-scatter  *)
(* on dedicated streams, while an eager all_reduce runs on the current       *)
(* (compute) stream. So every all_reduce of a rank shares one stream: on     *)
(* rank 0 that is tp01, batch02, loss02 and fsdp02's gradient-norm           *)
(* all_reduce, four DISTINCT communicators ordered against each other. That  *)
(* is what creates the wait edge, and it is why the instance must carry the  *)
(* overlapping communicators rather than one per role.                       *)
(*                                                                         *)
(* It is deliberately NOT taken from the observed facts. Stream identity     *)
(* there is inferred by zipping Flight Recorder entries positionally to      *)
(* Kineto kernels, and it assigns a distinct stream id to every              *)
(* (communicator, operation) pair on a rank -- six ids on rank 0 -- which is *)
(* an artifact of that zip. No claim in this module rests on observed stream *)
(* identity.                                                                *)
(*                                                                         *)
(* Streams being load-bearing is a CHECKED claim, but not in the obvious way. *)
(* Rerunning the divergent configuration with RequireStreamOrder = FALSE      *)
(* cannot fail: StuckByCircularWait is then unsatisfiable by construction,    *)
(* and the lemma is stated in full above StreamEdgeIsInert. What is           *)
(* contingent, and so what is checked, is the lemma's hypothesis -- that the  *)
(* shipped stream map is not refined by communicator identity here. An        *)
(* earlier version of this module put every communicator of a rank on its own *)
(* stream, which made AtStreamHead exactly redundant with per-communicator    *)
(* FIFO and left the state graph bit-identical when the conjunct was          *)
(* neutralised. ScoutBModelStreamShape.cfg fails on the shipped map and holds *)
(* on that earlier design, which is the regression guard.                     *)
(*                                                                         *)
(* ------------------------------------------------------------------------ *)
(* WHAT THE SCHEDULE GUARD MEANS. RequireUniformProgramOps and              *)
(* RequireUniformProgramComms together say "every rank runs one SPMD         *)
(* program". A program is a sequence of collective SITES. A site fixes the   *)
(* operation, and it fixes the communicator for every rank that takes part   *)
(* in it. Hence:                                                            *)
(*                                                                         *)
(*   - ops must agree at every position (RequireUniformProgramOps);          *)
(*   - communicators must be IDENTICAL at a position whenever their member   *)
(*     sets intersect (RequireUniformProgramComms), because two distinct     *)
(*     communicators that share a rank are two different sites and one       *)
(*     program cannot be at both at once;                                    *)
(*   - communicators with DISJOINT members are deliberately unconstrained:   *)
(*     that is one site with several groups, rank 0 in tp01 where rank 2 is  *)
(*     in tp23.                                                             *)
(*                                                                         *)
(* Both halves were checked against the observed run, not assumed. Over all  *)
(* 108 positions of all four ranks, ScoutBFacts shows zero                   *)
(* UniformProgramOpsOK violations and zero SameSite violations. That is a     *)
(* measurement of the trace against these two predicates -- both read only   *)
(* recorded communicator ids, member sets and operations -- and it is not a   *)
(* checked refinement: the refinement bridge is separate work.                *)
(*                                                                         *)
(* A separate statement, often quoted with it and easy to confuse with it:    *)
(* applying StreamOfIssue to the trace's operation labels puts four distinct  *)
(* communicators of each rank on the compute stream. That is the DESIGN       *)
(* CONSTANT applied to observed labels, not an observation. The trace's own   *)
(* stream field says something different, and why it cannot be used is under  *)
(* STREAMS above.                                                            *)
(*                                                                         *)
(* Communicator class is NOT constrained, and there is no CommClass constant. *)
(* A previous version required the mesh role ("tp", "fsdp", "loss") to agree  *)
(* per position. Two reasons it is gone.                                     *)
(*                                                                         *)
(* First, a correction to the record. In the OLD module the conjunct really   *)
(* was inert, because CommOps derived the admissible operations from the      *)
(* class, so requiring operations to agree already forced the class. Here     *)
(* CommOps is per communicator and gives mesh_fsdp all_reduce too, so         *)
(* operation agreement no longer implies role agreement and the conjunct does *)
(* restrict: measured, safety still passes and the reachable state graph      *)
(* falls from 38321 distinct states to 13583. It is dropped because what it   *)
(* removes are not hazards -- it forbids rank 0 issuing batch02 against rank  *)
(* 1 issuing loss13, which are disjoint and cannot wait on each other -- so   *)
(* keeping it would shrink coverage for no gain.                              *)
(*                                                                         *)
(* Second, the mesh role is FINER than the grouping that governs the hazard.  *)
(* On rank 0 the observed run issues mesh_batch all_reduce, mesh_fsdp         *)
(* all_gather and mesh_tp all_reduce within its first three collectives, so   *)
(* mesh_batch and mesh_tp share a stream while having different member sets.  *)
(* Grouping by role hides that; grouping by stream, which is what             *)
(* StreamOfIssue does, exposes it. A role-based guard also forced each rank's *)
(* choice outright whenever a rank held one communicator per role, which is   *)
(* how an earlier instance came to admit no cross-communicator wait edge at   *)
(* all.                                                                     *)
(*                                                                         *)
(* ------------------------------------------------------------------------ *)
(* WHY Stuck => AllDone IS NOT THE HAZARD PROPERTY. AllDone requires every   *)
(* member of a communicator to have issued the same COUNT on it. Under a     *)
(* finite MaxIssues, a rank can spend its budget on one communicator while a *)
(* peer spends it on another; no value of doneOn then satisfies AllDone, and *)
(* once both ranks are out of budget nothing can issue and nothing can       *)
(* start. Stuck holds, AllDone does not, and Stuck => AllDone is violated by *)
(* arithmetic rather than by a wait cycle.                                   *)
(*                                                                         *)
(* So stuck states are classified instead, and the three classes are         *)
(* pairwise disjoint and exhaustive over Stuck /\ ~AllDone:                  *)
(*                                                                         *)
(*   StuckByCircularWait  some communicator has EVERY member waiting in it   *)
(*                        at the same per-communicator position, the members *)
(*                        agree on the operation so NCCL would admit the     *)
(*                        rendezvous, issuance is balanced across every       *)
(*                        communicator so no member is merely behind, and it  *)
(*                        still cannot start. That is a hang, and the         *)
(*                        balance conjunct is what makes it independent of    *)
(*                        MaxIssues. DeadlockFreedom is its negation.        *)
(*   StuckByOpMismatch    every member is waiting in some communicator but   *)
(*                        they disagree on the operation. A different fault: *)
(*                        a divergent program, not a wait cycle.             *)
(*   StuckByBudget        either some communicator was issued by one member   *)
(*                        and not another, or no communicator has all its     *)
(*                        members waiting. The finite-budget artifact; not a  *)
(*                        hazard.                                            *)
(*                                                                         *)
(* StuckImpliesAllDone is kept and still checked in the safety configuration *)
(* because it is strictly stronger than the conjunction of the two hazard    *)
(* invariants: it additionally says the bound admits no budget artifact at   *)
(* all. It is the right thing to check where it holds and the wrong thing to *)
(* read as deadlock freedom where it fails.                                  *)
(*                                                                         *)
(* ------------------------------------------------------------------------ *)
(* No SYMMETRY. The topology does have a Klein-four automorphism, and TLC's  *)
(* SYMMETRY accepts any permutation set the module defines, so expressing a  *)
(* joint permutation of Ranks and CommIds is not the obstacle. The obstacle  *)
(* is that SYMMETRY requires the permuted elements to be MODEL VALUES, while *)
(* Ranks here are naturals and CommIds are strings. Declaring either set     *)
(* symmetric on its own would also be unsound, since CommMembers is not      *)
(* invariant under a permutation of one of them alone. Do not add it.        *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
  Ranks,
  CommIds,
  CommMembers,
  \* Admissible operations per communicator; see the instance notes above.
  CommOps,
  MaxIssues,
  \* The NCCL requirement: members of a communicator must agree on the
  \* operation at each per-communicator position. Relaxing this is one
  \* negative control.
  RequireMatchedIssueOrder,
  \* One SPMD program, operation half: same operation at each position.
  RequireUniformProgramOps,
  \* One SPMD program, site half: the same communicator at each position
  \* wherever two ranks' communicators share a member.
  RequireUniformProgramComms,
  \* CUDA stream in-order execution. Relaxing this is the cross-check that
  \* earns the claim that streams create the wait edge.
  RequireStreamOrder

(***************************************************************************)
(* The concrete instance, supplied to TLC by definition override because the *)
(* configuration file format cannot express function literals. There is      *)
(* exactly one instance. There used to be two: a narrow one that every cfg   *)
(* bound and could not express the hazard, and a wider one that described the *)
(* overlap and was bound by no cfg at all. Both are deleted.                  *)
(***************************************************************************)
Ranks2x2 == {0, 1, 2, 3}

CommIds2x2 ==
  {"tp01", "tp23",
   "fsdp02", "fsdp13",
   "batch02", "batch13",
   "loss02", "loss13"}

CommMembers2x2 ==
  [c \in CommIds2x2 |->
     CASE c = "tp01"    -> {0, 1}
       [] c = "tp23"    -> {2, 3}
       [] c = "fsdp02"  -> {0, 2}
       [] c = "fsdp13"  -> {1, 3}
       [] c = "batch02" -> {0, 2}
       [] c = "batch13" -> {1, 3}
       [] c = "loss02"  -> {0, 2}
       [] OTHER         -> {1, 3}]

Ops == {"all_reduce", "all_gather", "reduce_scatter"}

\* mesh_fsdp carries the two directional collectives plus the gradient-norm
\* all_reduce; every other communicator carries all_reduce only.
CommOps2x2 ==
  [c \in CommIds2x2 |->
     CASE c = "fsdp02" -> Ops
       [] c = "fsdp13" -> Ops
       [] OTHER        -> {"all_reduce"}]

\* The codomain of StreamOfIssue, for the reader. Nothing quantifies over it,
\* because a stream is only ever compared to another stream of the same rank.
Streams == {"compute", "ag", "rs"}

\* Keyed on the operation, not the communicator: FSDP2's dedicated streams
\* carry all-gather and reduce-scatter, and an eager all_reduce rides the
\* current compute stream whichever communicator it uses.
StreamOfIssue(e) ==
  CASE e.op = "all_gather"     -> "ag"
    [] e.op = "reduce_scatter" -> "rs"
    [] OTHER                   -> "compute"

VARIABLES
  issued,   \* [Ranks -> Seq([comm |-> CommIds, op |-> Ops])] issue order
  running,  \* SUBSET CommIds
  doneOn    \* [CommIds -> Nat]

vars == <<issued, running, doneOn>>

MinOf(a, b) == IF a <= b THEN a ELSE b

\* How many of r's first k issues were on the same communicator as issue k.
CommPos(r, k) == Cardinality({j \in 1..k : issued[r][j].comm = issued[r][k].comm})

CommCount(r, c) == Cardinality({j \in DOMAIN issued[r] : issued[r][j].comm = c})

Completed(r, k) == CommPos(r, k) <= doneOn[issued[r][k].comm]

\* Index in issued[r] of r's n-th issue on communicator c.
IdxOf(r, c, n) ==
  CHOOSE k \in DOMAIN issued[r] :
    /\ issued[r][k].comm = c
    /\ CommPos(r, k) = n

\* The stream executes in issue order, so every earlier issue of this rank on
\* the same stream must have completed. This is the wait edge, and it is the
\* only conjunct that can relate two DIFFERENT communicators on one rank.
AtStreamHead(r, k) ==
  \/ ~RequireStreamOrder
  \/ \A j \in 1..(k - 1) :
       StreamOfIssue(issued[r][j]) = StreamOfIssue(issued[r][k])
         => Completed(r, j)

Front(c) == doneOn[c] + 1

OpAt(r, c, n) == issued[r][IdxOf(r, c, n)].op

\* Every member has issued its Front(c)-th collective on c, so every member is
\* waiting inside c. Used to tell a wait cycle from an unissued communicator.
FullyPending(c) ==
  \A r \in CommMembers[c] : CommCount(r, c) >= Front(c)

\* Index in issued[r] of r's Front(c)-th issue on c: the issue r is waiting in.
HeadIdx(r, c) == IdxOf(r, c, Front(c))

(***************************************************************************)
(* The blocking relation, and why the hazard class is defined over its       *)
(* closure rather than over all of CommIds.                                 *)
(*                                                                         *)
(* A wait cycle is LOCAL to the communicators it runs through. An earlier    *)
(* version required issue counts to be equal on every communicator in the    *)
(* state, meaning to rule out a stuck state that a larger budget would       *)
(* resolve. It bought that at the price of false negatives: any unrelated    *)
(* imbalance somewhere else in the state reclassified a real deadlock as     *)
(* StuckByBudget. A witness exists at MaxIssues = 2 under                    *)
(* ScoutBModelDivergent.cfg -- ranks 1 and 3 in a two-cycle over fsdp13 and  *)
(* batch13, hung at any budget -- that the global conjunct discarded because *)
(* rank 0 had issued tp01 twice and rank 1 not at all.                      *)
(*                                                                         *)
(* What the budget-independence argument actually needs is that no step of   *)
(* the blocking chain is waiting for an issue that was never made. So the    *)
(* condition is imposed exactly on the chain: every communicator reachable   *)
(* from c through BlockedBy must itself be fully pending with agreeing        *)
(* operations. Then each element of the closure has a blocker inside the      *)
(* closure, the closure is finite, and a chain in which every element has a   *)
(* successor must close into a cycle. Extra budget cannot help, because a new *)
(* issue is appended after the blocker and never removes it.                 *)
(*                                                                         *)
(* Note this is WEAKER than count equality on the closure, deliberately: two *)
(* members of a chain communicator may have issued it a different number of  *)
(* times and the cycle is still a cycle. Count equality is asserted of the   *)
(* witness, where it is a statement about the counterexample rather than a   *)
(* condition on the hazard.                                                 *)
(*                                                                         *)
(* WHAT THIS DOES NOT CLAIM, so the next reader does not infer it:          *)
(*                                                                         *)
(* It is not a detector at the moment the hazard forms. StuckByCircularWait *)
(* is conjoined with Stuck, so a cycle that is already closed and already   *)
(* permanent stays unclassified while any rank can still issue. An          *)
(* independent review found exactly that at depth 5 of the divergent        *)
(* configuration: ranks 0 and 2 in a two-cycle on fsdp02 and batch02, with  *)
(* ranks 1 and 3 holding their whole budget. The invariant fires only in    *)
(* descendants where issuing has stopped, so "it eventually fires" rests on *)
(* MaxIssues being finite. Sound here; not a liveness or detection claim.   *)
(*                                                                         *)
(* A cycle passing through an operation-mismatched communicator is filed    *)
(* StuckByOpMismatch, which is the right attribution -- that link can never *)
(* run for an independent reason. But ScoutBModelDivergent.cfg names only   *)
(* DeadlockFreedom, so a future relaxation admitting a mixed cycle would go *)
(* unreported there. The safety configuration covers both, via              *)
(* StuckImpliesAllDone.                                                    *)
(*                                                                         *)
(* In the safety configuration StuckImpliesAllDone holds, so that space has *)
(* no Stuck /\ ~AllDone state at all, so DeadlockFreedom adds no independent*)
(* strength there. What corroborates the safe space independently is that an*)
(* opposite-direction hung-core fixpoint is empty over all 38321 states.    *)
(***************************************************************************)

\* The issues that stop c from starting: an earlier issue of some member, on
\* the same stream as that member's Front(c)-th issue on c, not yet completed.
\* Guarded by FullyPending because HeadIdx is undefined for a member that has
\* not reached c, and because a communicator nobody is waiting in is blocked on
\* a missing issue rather than on stream order.
\* The wait relation of the CONFIGURED spec, not of the stream map alone.
\* With RequireStreamOrder = FALSE nothing actually blocks, because AtStreamHead
\* is then vacuously TRUE; reporting same-stream predecessors anyway would make
\* the closure non-trivial over a relation that is not this spec's wait
\* relation. Harmless while CircularWaitAt is only ever evaluated under Stuck
\* -- StreamEdgeIsInert makes its hypothesis unsatisfiable there -- but a
\* latent false positive the moment it is evaluated outside Stuck, which the
\* budget caveat below invites.
BlockingPairs(c) ==
  IF ~RequireStreamOrder \/ ~FullyPending(c)
    THEN {}
    ELSE {p \in CommMembers[c] \X (1..MaxIssues) :
            /\ p[2] \in DOMAIN issued[p[1]]
            /\ p[2] < HeadIdx(p[1], c)
            /\ StreamOfIssue(issued[p[1]][p[2]])
                 = StreamOfIssue(issued[p[1]][HeadIdx(p[1], c)])
            /\ ~Completed(p[1], p[2])}

BlockedBy(c) == {issued[p[1]][p[2]].comm : p \in BlockingPairs(c)}

\* Least fixpoint of BlockedBy from c. Iterating Cardinality(CommIds) times is
\* enough: each round either adds a communicator or has already converged, and
\* there are only that many to add.
BlockingClosure(c) ==
  LET rounds == Cardinality(CommIds)
      step(S) == S \cup UNION {BlockedBy(d) : d \in S}
      f[i \in 0..rounds] == IF i = 0 THEN {c} ELSE step(f[i - 1])
  IN f[rounds]

MemberReady(c, r) ==
  /\ CommCount(r, c) >= Front(c)
  /\ AtStreamHead(r, IdxOf(r, c, Front(c)))

OpsAgreeAtFront(c) ==
  \A r1 \in CommMembers[c] :
    \A r2 \in CommMembers[c] :
      OpAt(r1, c, Front(c)) = OpAt(r2, c, Front(c))

Init ==
  /\ issued  = [r \in Ranks |-> << >>]
  /\ running = {}
  /\ doneOn  = [c \in CommIds |-> 0]

TypeOK ==
  /\ running \subseteq CommIds
  /\ doneOn \in [CommIds -> 0..MaxIssues]
  /\ \A r \in Ranks :
       \A k \in DOMAIN issued[r] :
         /\ issued[r][k].comm \in CommIds
         /\ issued[r][k].op \in CommOps[issued[r][k].comm]
         /\ r \in CommMembers[issued[r][k].comm]

(***************************************************************************)
(* One SPMD program. See the module header for the decision this encodes.    *)
(***************************************************************************)
UniformProgramOpsOK(candidate) ==
  \A r1 \in Ranks :
    \A r2 \in Ranks :
      \A k \in 1..MinOf(Len(candidate[r1]), Len(candidate[r2])) :
        candidate[r1][k].op = candidate[r2][k].op

\* Two distinct communicators that share a rank are two different program
\* sites; one program cannot be at both at position k. Disjoint communicators
\* are the same site seen by different groups and stay unconstrained.
SameSite(c1, c2) ==
  \/ c1 = c2
  \/ CommMembers[c1] \cap CommMembers[c2] = {}

UniformProgramCommsOK(candidate) ==
  \A r1 \in Ranks :
    \A r2 \in Ranks :
      \A k \in 1..MinOf(Len(candidate[r1]), Len(candidate[r2])) :
        SameSite(candidate[r1][k].comm, candidate[r2][k].comm)

IssueCandidate(r, c, op) ==
  [issued EXCEPT ![r] = Append(issued[r], [comm |-> c, op |-> op])]

IssueAllowed(r, c, op) ==
  /\ r \in CommMembers[c]
  /\ op \in CommOps[c]
  /\ Len(issued[r]) < MaxIssues
  /\ (RequireUniformProgramOps =>
        UniformProgramOpsOK(IssueCandidate(r, c, op)))
  /\ (RequireUniformProgramComms =>
        UniformProgramCommsOK(IssueCandidate(r, c, op)))

Issue(r, c, op) ==
  /\ IssueAllowed(r, c, op)
  /\ issued' = IssueCandidate(r, c, op)
  /\ UNCHANGED <<running, doneOn>>

StartAllowed(c) ==
  /\ c \notin running
  /\ \A r \in CommMembers[c] : MemberReady(c, r)
  /\ (RequireMatchedIssueOrder => OpsAgreeAtFront(c))

Start(c) ==
  /\ StartAllowed(c)
  /\ running' = running \cup {c}
  /\ UNCHANGED <<issued, doneOn>>

Complete(c) ==
  /\ c \in running
  /\ running' = running \ {c}
  /\ doneOn'  = [doneOn EXCEPT ![c] = @ + 1]
  /\ UNCHANGED issued

AllDone ==
  /\ running = {}
  /\ \A r \in Ranks : Len(issued[r]) = MaxIssues
  /\ \A c \in CommIds : \A r \in CommMembers[c] : CommCount(r, c) = doneOn[c]

Terminated == AllDone /\ UNCHANGED vars

Next ==
  \/ \E r \in Ranks : \E c \in CommIds : \E op \in Ops : Issue(r, c, op)
  \/ \E c \in CommIds : Start(c)
  \/ \E c \in CommIds : Complete(c)
  \/ Terminated

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* Safety.                                                                 *)
(***************************************************************************)

\* A running collective is the same operation at the same per-communicator
\* position for every member. This is the protocol twin of the observed
\* CollectiveCoverage predicate, and it is what the unguarded negative
\* violates.
RendezvousOpAgreement ==
  \A c \in running :
    \A r1 \in CommMembers[c] :
      \A r2 \in CommMembers[c] :
        OpAt(r1, c, doneOn[c] + 1) = OpAt(r2, c, doneOn[c] + 1)

\* A running collective was genuinely issued on every member.
RendezvousMembership ==
  \A c \in running :
    \A r \in CommMembers[c] : CommCount(r, c) >= doneOn[c] + 1

\* A communicator never runs ahead of any member's issue count.
CommFifo ==
  \A c \in CommIds :
    \A r \in CommMembers[c] : doneOn[c] <= CommCount(r, c)

\* Stuck is written out explicitly rather than with ENABLED so TLC evaluates it
\* directly and so it cannot drift from Next: both use the same predicates.
\* Note AllDone => Stuck, which is why the classes below exclude AllDone.
Stuck ==
  /\ running = {}
  /\ \A r \in Ranks :
       \A c \in CommIds : \A op \in Ops : ~IssueAllowed(r, c, op)
  /\ \A c \in CommIds : ~StartAllowed(c)

\* c heads a budget-independent wait cycle. Every member is waiting in c and
\* they agree on the operation, so NCCL would admit the rendezvous; the same
\* holds of every communicator reachable through the blocking relation, so no
\* step of the chain is waiting for an issue that was never made. Under Stuck
\* nothing in the system can move, so with members present and operations
\* agreeing the only unmet conjunct of StartAllowed is AtStreamHead: this is
\* exactly a stream-ordered circular wait. See the closure commentary above.
CircularWaitAt(c) ==
  /\ FullyPending(c)
  /\ OpsAgreeAtFront(c)
  /\ \A d \in BlockingClosure(c) :
       FullyPending(d) /\ OpsAgreeAtFront(d)

StuckByCircularWait ==
  /\ Stuck
  /\ \E c \in CommIds : CircularWaitAt(c)

\* Every member of some communicator is waiting in it and they disagree on the
\* operation. A divergent program, not a wait cycle.
StuckByOpMismatch ==
  /\ Stuck
  /\ ~StuckByCircularWait
  /\ \E c \in CommIds : FullyPending(c) /\ ~OpsAgreeAtFront(c)

\* The finite-MaxIssues artifact, defined as the remainder so the three classes
\* are exhaustive over Stuck /\ ~AllDone by construction. It is the budget case
\* and nothing else: a stuck state here has no fully pending communicator at
\* all, or every fully pending one with agreeing operations has a communicator
\* in its blocking chain that some member never reached -- which is a missing
\* issue, and a missing issue at MaxIssues is what the bound cut off.
StuckByBudget ==
  /\ Stuck
  /\ ~AllDone
  /\ ~StuckByCircularWait
  /\ ~StuckByOpMismatch

\* The hazard this module exists to state: no admissible rendezvous is
\* permanently unable to start.
DeadlockFreedom == ~StuckByCircularWait

\* The other hang mechanism, kept separate so the two negatives are
\* distinguishable by which invariant they name.
NoOpMismatchHang == ~StuckByOpMismatch

\* Strictly stronger than DeadlockFreedom /\ NoOpMismatchHang: it additionally
\* says the bound admits no StuckByBudget state at all. Checked in the safety
\* configuration, where it holds. It is NOT the hazard property: under a finite
\* MaxIssues it is violated by arithmetic wherever two members spend their
\* budget on different communicators.
StuckImpliesAllDone == Stuck => AllDone

(***************************************************************************)
(* The witness shape, checked by refutation in ScoutBModelWitness.cfg.       *)
(*                                                                         *)
(* Reporting "DeadlockFreedom was violated" says nothing about WHICH stuck   *)
(* state was found. This pins the shape: a cycle spanning at least two        *)
(* distinct communicators, with the operations agreeing, with issue counts    *)
(* equal on every communicator of the chain, and with every rank having spent *)
(* its whole budget so no rank merely stopped early. It is the guard that     *)
(* would have caught the global-balance defect, because a witness that needed *)
(* an imbalance elsewhere in the state to exist would not match it.           *)
(***************************************************************************)
ChainIssuanceBalanced(S) ==
  \A c \in S :
    \A r1 \in CommMembers[c] :
      \A r2 \in CommMembers[c] :
        CommCount(r1, c) = CommCount(r2, c)

NoCrossCommunicatorCycleWitness ==
  ~( /\ Stuck
     /\ \A r \in Ranks : Len(issued[r]) = MaxIssues
     /\ \E c \in CommIds :
          /\ CircularWaitAt(c)
          /\ Cardinality(BlockingClosure(c)) >= 2
          /\ ChainIssuanceBalanced(BlockingClosure(c)) )

(***************************************************************************)
(* The stream edge, checked by refutation in ScoutBModelStreamShape.cfg.     *)
(*                                                                         *)
(* LEMMA. If StreamOfIssue is refined by communicator identity -- that is,   *)
(* if two issues of one rank share a stream only when they share a           *)
(* communicator -- then AtStreamHead is implied by per-communicator FIFO and *)
(* StuckByCircularWait is unsatisfiable at any bound on any instance. Proof: *)
(* the only earlier same-stream issues of r before HeadIdx(r, c) are then on  *)
(* c itself, at positions 1..doneOn[c], all of which are Completed; so        *)
(* MemberReady(c, r) reduces to CommCount(r, c) >= Front(c) and              *)
(* StartAllowed(c) to FullyPending(c) /\ OpsAgreeAtFront(c). Stuck asserts    *)
(* ~StartAllowed(c) for every c, so no c can satisfy CircularWaitAt.         *)
(*                                                                         *)
(* So an exhaustive run with the stream edge deleted proves this lemma and    *)
(* nothing about the instance: it cannot fail. What has to be checked instead *)
(* is the lemma's hypothesis -- that the shipped stream map is NOT refined by *)
(* communicator identity on this instance. That is what fails below, and      *)
(* substituting StreamOfIssue(e) == e.comm, which is what an earlier version  *)
(* of this module effectively had, makes it stop failing.                    *)
(***************************************************************************)
StreamEdgeIsInert ==
  ~( \E r \in Ranks :
       \E c1 \in CommIds : \E c2 \in CommIds :
         \E op1 \in CommOps[c1] : \E op2 \in CommOps[c2] :
           /\ c1 # c2
           /\ r \in CommMembers[c1]
           /\ r \in CommMembers[c2]
           /\ StreamOfIssue([comm |-> c1, op |-> op1])
                = StreamOfIssue([comm |-> c2, op |-> op2]) )

\* Pins the stream-shape configuration to the initial state: StreamEdgeIsInert
\* reads only the constants, so exploring the transition relation would add
\* cost and no evidence.
AtInitialState == \A r \in Ranks : Len(issued[r]) = 0

\* Non-vacuity, checked by refutation in its own configuration. Start's guard
\* uses CHOOSE and a stream-head condition that are easy to make unsatisfiable,
\* which would make every safety invariant above vacuously true.
ModelNeverCompletes == ~AllDone

\* Belt to the guards' braces: keeps TLC inside the declared bound even if a
\* guard is later loosened. There is no skew conjunct here and no MaxSkew
\* constant: at MaxIssues = 2 a skew bound of 2 is a tautology, so the cfg
\* advertised a bound it did not impose. Issue skew is unbounded within
\* MaxIssues, which is what the claim inventory says.
ModelBounded == \A r \in Ranks : Len(issued[r]) <= MaxIssues

=============================================================================
