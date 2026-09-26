-------------------------------- MODULE ScoutBModel --------------------------------
(***************************************************************************)
(* An abstract model of collective execution on a 2x2 DP-shard x TP mesh.  *)
(*                                                                         *)
(* This models the PROTOCOL, not the observed run. Ranks issue collectives  *)
(* freely and concurrently, so TLC explores rank skew, and the invariants   *)
(* hold for every admitted schedule rather than for one recording.          *)
(*                                                                         *)
(* Faithfulness notes, each verified against the observed 2x2 execution:    *)
(*                                                                         *)
(* - Eight communicator instances, not four. mesh_batch, mesh_fsdp and      *)
(*   mesh_loss_mesh are SEPARATE communicators that happen to span the same *)
(*   rank set {0,2}, which is what creates a cross-communicator wait cycle. *)
(*   The observed runtime_pg_id is per-rank local and maps to two different *)
(*   communicators, so it is never a key here.                              *)
(*                                                                         *)
(* - Streams are load-bearing, not decoration. Without them this model has  *)
(*   NO deadlock: every rank would eventually issue everything and every    *)
(*   rendezvous would complete. The wait edge exists because a CUDA stream  *)
(*   executes in issue order, so a collective cannot start until earlier    *)
(*   work on its stream has completed. FSDP uses dedicated all-gather and   *)
(*   reduce-scatter streams; TP collectives ride the compute stream.        *)
(*                                                                         *)
(* - Per-collective lifecycle collapses into `running` plus `doneOn`        *)
(*   because NCCL is FIFO per communicator. That is a faithful reduction,   *)
(*   and it is the largest state-space saving available.                    *)
(*                                                                         *)
(* No SYMMETRY. The topology does have a Klein-four automorphism, but       *)
(* exploiting it requires permuting Ranks and CommIds *consistently* with   *)
(* CommMembers, which TLC's Permutations cannot express. Declaring either   *)
(* set symmetric on its own would be unsound. Do not add it.               *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
  Ranks,
  CommIds,
  CommMembers,
  CommClass,
  MaxIssues,
  MaxSkew,
  \* The NCCL requirement: members of a communicator must agree on the
  \* operation at each position. Relaxing this alone is one negative control.
  RequireMatchedIssueOrder,
  \* TorchTitan's obligation on top of it: every rank runs the same program, so
  \* it issues the same (class, operation) sequence. Relaxing this models a
  \* divergent program, which is what actually hangs jobs in practice.
  RequireUniformProgramSchedule

(***************************************************************************)
(* Concrete instances, supplied to TLC by definition override because the   *)
(* configuration file format cannot express function literals.              *)
(*                                                                         *)
(* Small: the four communicators needed for a cross-communicator cycle.     *)
(* Wide: adds loss and batch, the three-communicators-over-{0,2} overlap    *)
(* actually present in the observed run.                                    *)
(***************************************************************************)
RanksSmall == {0, 1, 2, 3}

CommIdsSmall == {"tp01", "tp23", "fsdp02", "fsdp13"}

CommMembersSmall ==
  [c \in CommIdsSmall |->
     CASE c = "tp01"   -> {0, 1}
       [] c = "tp23"   -> {2, 3}
       [] c = "fsdp02" -> {0, 2}
       [] OTHER        -> {1, 3}]

CommClassSmall ==
  [c \in CommIdsSmall |->
     CASE c = "tp01" -> "tp"
       [] c = "tp23" -> "tp"
       [] OTHER      -> "fsdp"]

CommIdsWide ==
  {"tp01", "tp23", "fsdp02", "fsdp13", "loss02", "loss13"}

CommMembersWide ==
  [c \in CommIdsWide |->
     CASE c = "tp01"   -> {0, 1}
       [] c = "tp23"   -> {2, 3}
       [] c = "fsdp02" -> {0, 2}
       [] c = "fsdp13" -> {1, 3}
       [] c = "loss02" -> {0, 2}
       [] OTHER        -> {1, 3}]

CommClassWide ==
  [c \in CommIdsWide |->
     CASE c = "tp01" -> "tp"
       [] c = "tp23" -> "tp"
       [] c = "fsdp02" -> "fsdp"
       [] c = "fsdp13" -> "fsdp"
       [] OTHER        -> "loss"]

Ops == {"all_reduce", "all_gather", "reduce_scatter"}
Streams == {"compute", "ag", "rs"}

\* Only fsdp carries the two directional operations; tp, loss and batch are
\* all_reduce in the observed run.
CommOps(c) ==
  IF CommClass[c] = "fsdp" THEN {"all_gather", "reduce_scatter"}
                           ELSE {"all_reduce"}

StreamOfIssue(e) ==
  IF CommClass[e.comm] = "fsdp"
    THEN IF e.op = "reduce_scatter" THEN "rs" ELSE "ag"
    ELSE "compute"

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
\* the same stream must have completed. This is the wait edge.
AtStreamHead(r, k) ==
  \A j \in 1..(k - 1) :
    StreamOfIssue(issued[r][j]) = StreamOfIssue(issued[r][k]) => Completed(r, j)

Front(c) == doneOn[c] + 1

OpAt(r, c, n) == issued[r][IdxOf(r, c, n)].op

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
         /\ issued[r][k].op \in Ops
         /\ r \in CommMembers[issued[r][k].comm]

(***************************************************************************)
(* Every rank runs the same program, so it issues the same sequence of      *)
(* (communicator class, operation) pairs. Which communicator INSTANCE it    *)
(* uses is fixed by its mesh coordinate -- rank 0 uses fsdp02 where rank 1  *)
(* uses fsdp13 -- so the instance is deliberately not constrained.          *)
(*                                                                         *)
(* The operation must be part of this. An earlier version constrained only  *)
(* the class, and TLC promptly found a counterexample where two members of  *)
(* one communicator issued all_gather and reduce_scatter at the same        *)
(* position: that communicator can then never start, and the run hangs. The *)
(* model was wrong, not the protocol -- a real program cannot diverge that  *)
(* way, and NCCL requires that it does not.                                 *)
(***************************************************************************)
UniformProgramScheduleOK(candidate) ==
  \A r1 \in Ranks :
    \A r2 \in Ranks :
      \A k \in 1..MinOf(Len(candidate[r1]), Len(candidate[r2])) :
        /\ CommClass[candidate[r1][k].comm] = CommClass[candidate[r2][k].comm]
        /\ candidate[r1][k].op = candidate[r2][k].op

IssueCandidate(r, c, op) ==
  [issued EXCEPT ![r] = Append(issued[r], [comm |-> c, op |-> op])]

IssueAllowed(r, c, op) ==
  /\ r \in CommMembers[c]
  /\ op \in CommOps(c)
  /\ Len(issued[r]) < MaxIssues
  /\ \A q \in Ranks : Len(issued[r]) + 1 <= Len(issued[q]) + MaxSkew
  /\ (RequireUniformProgramSchedule =>
        UniformProgramScheduleOK(IssueCandidate(r, c, op)))

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
\* CollectiveCoverage predicate, and it is what the first negative violates.
RendezvousOpAgreement ==
  \A c \in running :
    \A r1 \in CommMembers[c] :
      \A r2 \in CommMembers[c] :
        OpAt(r1, c, doneOn[c] + 1) = OpAt(r2, c, doneOn[c] + 1)

\* A running collective was genuinely issued and stream-ready on every member.
RendezvousMembership ==
  \A c \in running :
    \A r \in CommMembers[c] : CommCount(r, c) >= doneOn[c] + 1

\* A communicator never runs ahead of any member's issue count.
CommFifo ==
  \A c \in CommIds :
    \A r \in CommMembers[c] : doneOn[c] <= CommCount(r, c)

\* Stuck is written out explicitly rather than with ENABLED so TLC evaluates it
\* directly and so it cannot drift from Next: both use the same predicates.
Stuck ==
  /\ running = {}
  /\ \A r \in Ranks :
       \A c \in CommIds : \A op \in Ops : ~IssueAllowed(r, c, op)
  /\ \A c \in CommIds : ~StartAllowed(c)

\* Cross-communicator deadlock freedom: being unable to move must mean done.
DeadlockFreedom == Stuck => AllDone

\* Non-vacuity, checked by refutation in its own configuration. Start's guard
\* uses CHOOSE and a stream-head condition that are easy to make unsatisfiable,
\* which would make every safety invariant above vacuously true.
ModelNeverCompletes == ~AllDone

\* Belt to the guards' braces: keeps TLC inside the declared bound even if a
\* guard is later loosened.
ModelBounded ==
  /\ \A r \in Ranks : Len(issued[r]) <= MaxIssues
  /\ \A r1 \in Ranks : \A r2 \in Ranks :
       Len(issued[r1]) <= Len(issued[r2]) + MaxSkew

=============================================================================
