------------------------------- MODULE DeviceMeshRefine -------------------------------
(***************************************************************************)
(* Trace refinement for the DPxTP collective protocol: is the OBSERVED      *)
(* four-rank run an admitted behaviour of DeviceMeshModel?                     *)
(*                                                                         *)
(* DeviceMeshModel says what the protocol permits; DeviceMeshFacts says what a      *)
(* particular 2x2 DP-shard x TP run did. Without this module the two are    *)
(* unrelated artifacts and a reader may reasonably assume the model was     *)
(* checked against reality. This module joins them.                        *)
(*                                                                         *)
(* ------------------------------------------------------------------------ *)
(* WHAT IS REPLAYED, AND WHAT IS NOT. Only the per-rank ISSUE ORDER.        *)
(*                                                                         *)
(* The observer appends each collective's enqueued/started/completed triple *)
(* contiguously, post hoc, from the Flight Recorder, and the raw-event      *)
(* projection drops observed_time_ns. So the per-rank order of the started  *)
(* and completed events asserts a full serialization of every collective    *)
(* before the next, which cannot have happened: FSDP all-gathers overlap by *)
(* construction. A bridge that replayed per-rank EVENT order would certify  *)
(* a fiction and would PASS, because a fully serialized schedule satisfies  *)
(* every guard trivially.                                                  *)
(*                                                                         *)
(* Only the enqueued sub-order is faithful -- it inherits the Flight        *)
(* Recorder record sequence -- so only CollectiveIssueOrder is read here.   *)
(* Start and Complete are taken by the model's own guards. The schedule of  *)
(* Start and Complete along the witness path is therefore MODEL-CHOSEN and  *)
(* is not claimed to be the schedule that ran.                             *)
(*                                                                         *)
(* ------------------------------------------------------------------------ *)
(* CLAIM INVENTORY. Each line names a configuration and the exact statement *)
(* it establishes. Nothing outside this list is claimed.                    *)
(*                                                                         *)
(*   DeviceMeshRefineMapping.cfg  the observed-to-model communicator mapping is *)
(*                            derivable, bijective, and agrees with the     *)
(*                            model on member sets and admissible           *)
(*                            operations; the derived per-rank issue        *)
(*                            sequences have distinct issue orders, equal   *)
(*                            lengths, and cover the replay bound. Pinned   *)
(*                            to the initial state: every conjunct reads    *)
(*                            constants only.                              *)
(*   DeviceMeshRefine.cfg         the observed issue order of all four ranks,   *)
(*                            all 108 issues each, RUNS TO COMPLETION under *)
(*                            every guard of DeviceMeshModel switched on. The   *)
(*                            witness is a violation of                     *)
(*                            ReplayedRunIsNotAdmitted; see the polarity    *)
(*                            note below. The search is narrowed by the     *)
(*                            greedy policy, which is sound for a witness;  *)
(*                            see SOUNDNESS OF THE GREEDY REPLAY.           *)
(*   DeviceMeshRefineSkew.cfg     over the first MaxIssues observed issues of   *)
(*                            each rank, EVERY schedule whose per-rank      *)
(*                            issue counts stay within MaxReplaySkew of one *)
(*                            another and which keeps at most              *)
(*                            MaxOutstanding issues in flight per rank and *)
(*                            communicator reaches completion --           *)
(*                            exhaustively, with no dead end. This is the   *)
(*                            machine-checked replacement for the prose     *)
(*                            confluence argument, over a defined fragment  *)
(*                            rather than over the whole interleaving       *)
(*                            space.                                       *)
(*   DeviceMeshRefineOverlap.cfg  and that fragment is not a serialized space:  *)
(*                            NoTwoCollectivesRunConcurrently is false      *)
(*                            there, so it contains states with two        *)
(*                            collectives in flight at once. Same constants *)
(*                            as the skew cfg; only the invariant and the   *)
(*                            deadlock switch differ.                      *)
(*   DeviceMeshRefineUniformPermutation.cfg                                     *)
(*                            a permutation applied uniformly to all four   *)
(*                            ranks -- every rank's sequence reversed -- is *)
(*                            ADMITTED. Not a defect: the protocol          *)
(*                            constrains cross-rank agreement, not absolute *)
(*                            order. Checked so the limit of the positive   *)
(*                            is a recorded fact.                          *)
(*   DeviceMeshRefineSingleRankPermutation.cfg                                  *)
(*                            reversing ONE rank's sequence, which breaks   *)
(*                            that agreement, is REFUSED. This is the       *)
(*                            evidence that the bridge detects the property *)
(*                            NCCL imposes.                                *)
(*   DeviceMeshRefineBad.cfg      transposing two ADJACENT issues of ONE rank   *)
(*                            on ONE communicator, with differing           *)
(*                            operations, is refused: the run never         *)
(*                            completes and the mismatched collective never *)
(*                            starts.                                      *)
(*   DeviceMeshRefineBadReach.cfg and it is refused NO EARLIER than that        *)
(*                            rendezvous: every earlier collective on that  *)
(*                            communicator completes.                      *)
(*   DeviceMeshRefineBadRelaxed.cfg and relaxing RequireMatchedIssueOrder alone *)
(*                            makes the same corruption complete, which     *)
(*                            attributes the refusal to NCCL's matching     *)
(*                            requirement rather than to the stream-head or *)
(*                            budget guard.                                *)
(*   DeviceMeshRefineBadUniform.cfg under the POSITIVE's guard set -- that is,   *)
(*                            with RequireUniformProgramOps TRUE -- the     *)
(*                            same corruption is refused too, and refused   *)
(*                            differently: CorruptedColumnIsNeverFormed     *)
(*                            holds, so no peer ever reaches the corrupted  *)
(*                            position and the refuser is the SPMD-program  *)
(*                            guard in IssueAllowed rather than the         *)
(*                            rendezvous guard in StartAllowed.             *)
(*                                                                         *)
(* ------------------------------------------------------------------------ *)
(* POLARITY. TLC proves reachability by refutation, so the witness that the *)
(* observed run is admitted is a VIOLATION of ReplayedRunIsNotAdmitted. The *)
(* runner translates that into a positive result token so the evidence does *)
(* not read backwards. This is the same convention as SingleRankRefine.         *)
(*                                                                         *)
(* There is no cursor variable, exactly as in SingleRankRefine: Len(issued[r])  *)
(* is the implicit per-rank cursor.                                         *)
(*                                                                         *)
(* ------------------------------------------------------------------------ *)
(* WHY COMPLETION, NOT LENGTH, IS THE ADMISSION TEST. Issue's guard does    *)
(* not read doneOn at all, so a length-only test -- "some state has         *)
(* Len(issued[r]) = 108 for every r" -- is satisfied by issuing everything  *)
(* and running nothing, for a CORRUPTED issue order just as much as for the *)
(* real one. It would be vacuous and the negative control would pass.       *)
(*                                                                         *)
(* So admission is AllDone: every rank has issued the replayed prefix and   *)
(* every collective has completed. With MaxIssues bound to the observed     *)
(* per-rank issue count, AllDone says exactly that.                        *)
(*                                                                         *)
(* ------------------------------------------------------------------------ *)
(* WHAT THE POSITIVE ESTABLISHES, STATED EXACTLY.                           *)
(*                                                                         *)
(* At each of the 108 positions the four ranks' recorded issues form a      *)
(* COLUMN. The positive establishes two things about those columns, and     *)
(* nothing more:                                                           *)
(*                                                                         *)
(*   - every column is model-typable and cross-rank consistent: each        *)
(*     rank's issue names a communicator that rank is a member of, carrying *)
(*     an operation that communicator admits, and at every position the     *)
(*     four ranks agree on the operation and are at the same communicator   *)
(*     wherever their communicators share a rank;                           *)
(*   - the collectives those columns induce DRAIN TO COMPLETION under the   *)
(*     model's own Start and Complete guards -- rendezvous operation        *)
(*     agreement, per-communicator FIFO and CUDA stream order hold          *)
(*     throughout, and every collective runs.                              *)
(*                                                                         *)
(* It does NOT establish that the observed ABSOLUTE order is the only order *)
(* the model admits, and it cannot, because the protocol does not constrain *)
(* absolute order. OpsAgreeAtFront quantifies over the MEMBERS at a         *)
(* communicator's front; UniformProgramOpsOK and UniformProgramCommsOK      *)
(* quantify over the RANKS at a position. All three are relational across   *)
(* ranks at a position; none of them mentions which collective a position   *)
(* ought to carry. That is correct NCCL semantics -- members of a           *)
(* communicator must AGREE on their collective sequence, not follow any     *)
(* particular one -- so a permutation applied uniformly to all four ranks   *)
(* is an equally valid SPMD program and must be admitted.                   *)
(*                                                                         *)
(* Measured, so the limit is not left implicit: reversing every rank's      *)
(* sequence is admitted with the SAME 9172 states generated, 5371 distinct  *)
(* and depth 865 the positive reports, because the search only ever sees    *)
(* the column structure and the reversal permutes columns without breaking  *)
(* one. Relaxing RequireMatchedIssueOrder, RequireStreamOrder, or either    *)
(* SPMD guard on the observed order also leaves those numbers unchanged.    *)
(*                                                                         *)
(* So the limit is on the record as a CHECKED PAIR rather than as a caveat: *)
(* DeviceMeshRefineUniformPermutation.cfg reverses every rank's sequence and is *)
(* ADMITTED; DeviceMeshRefineSingleRankPermutation.cfg reverses ONE rank's      *)
(* sequence and is REFUSED, at the first position where that rank's         *)
(* communicator disagrees with a peer that shares it. The second is the     *)
(* positive evidence that the bridge detects agreement violations, which is *)
(* the property NCCL actually imposes and the property a real job hangs on. *)
(*                                                                         *)
(* ------------------------------------------------------------------------ *)
(* SOUNDNESS OF THE GREEDY REPLAY. DeviceMeshModel's guards are O(k^2) in the   *)
(* length k of a rank's issue sequence, and MaxSkew was dropped from the    *)
(* model in favour of unbounded skew, so a prefix-constrained search over   *)
(* all interleavings of 4 x 108 issues is not viable: the length tuple      *)
(* alone spans 109^4.                                                      *)
(*                                                                         *)
(* The cost control therefore lives HERE, in the refinement module, and     *)
(* Next is untouched. GreedyReplay narrows the SEARCH, not the protocol:    *)
(* every step of ConstrainedNext is a step of Next, so a path it finds is a *)
(* behaviour of DeviceMeshModel. A witness is a witness; narrowing the search   *)
(* can only lose witnesses, never invent one. That is the whole soundness   *)
(* argument for the positive result, and it needs no confluence lemma.      *)
(*                                                                         *)
(* The witness path the greedy policy finds is SERIALIZED: at most one      *)
(* collective is in flight at a time. A serialized schedule satisfies every *)
(* guard easily, so the positive result is not evidence about SCHEDULING.   *)
(* Overlapping schedules are covered separately by DeviceMeshRefineSkew.cfg,    *)
(* and DeviceMeshRefineOverlap.cfg shows that space really does contain them.   *)
(*                                                                         *)
(* Confluence is needed for the CONVERSE -- reading a greedy failure as     *)
(* non-admission -- and that direction is not claimed from the greedy       *)
(* configuration. DeviceMeshRefineSkew.cfg is where it is checked, over a       *)
(* declared fragment. The prose form of the argument is that every guard is *)
(* monotone in doneOn and no action disables another rank's Issue; the      *)
(* machine-checked form is stated with DeviceMeshRefineSkew.cfg below.          *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets, DeviceMeshModel, DeviceMeshFacts,
        MeshTopology

CONSTANTS
  \* TRUE narrows the search to the drain-then-advance schedule described
  \* above ConstrainedNext. It restricts the search only; see the header.
  GreedyReplay,
  \* Maximum difference in per-rank issue counts the search may explore.
  MaxReplaySkew,
  \* Maximum number of issues one rank may have outstanding on one
  \* communicator: the overlap window. MaxReplaySkew alone does not bound the
  \* state space, because all four ranks may race to their issue bound with
  \* nothing completed and the completions then interleave combinatorially.
  \* This is the conjunct that bounds work in flight, and it is what makes an
  \* exhaustive bounded run finish. At 1 it still admits two collectives
  \* running at once on different communicators, which
  \* DeviceMeshRefineOverlap.cfg checks by refutation.
  MaxOutstanding,
  \* TRUE replays the corrupted issue order instead of the observed one.
  \* This is the ONLY difference between the positive and negative controls
  \* other than which guards are relaxed and which invariants are named.
  TransposeMutation,
  \* "observed" replays the recorded order. "uniform" reverses every rank's
  \* sequence, which is a permutation of the COLUMNS and so preserves
  \* cross-rank agreement. "single_rank" reverses one rank's sequence only,
  \* which breaks it. The two order-sensitivity controls; see WHAT THE
  \* POSITIVE ESTABLISHES in the header.
  ReplayPermutation

(***************************************************************************)
(* THE COMMUNICATOR MAPPING, DERIVED.                                      *)
(*                                                                         *)
(* The facts key collectives by the global canonical_id -- "tp:0,1:mesh_tp",*)
(* "dp_shard:0,2:mesh_fsdp" and so on -- while the model keys them by its   *)
(* own eight names. A hand-written table between the two that silently      *)
(* disagreed with CommMembers would invalidate every claim in this module,  *)
(* and confusing an eight-valued key with a four-valued one has already     *)
(* happened once in this project (runtime_pg_id).                           *)
(*                                                                         *)
(* So the mapping is DERIVED from the recorded member sets and the recorded *)
(* operations, and no communicator name of either side appears below. It is *)
(* then asserted: total on the observed communicators, injective, onto      *)
(* CommIds, member sets equal, and every observed operation admissible at   *)
(* the image. CommMappingIsWellFormed is the assertion and                  *)
(* DeviceMeshRefineMapping.cfg is where it is checked.                          *)
(*                                                                         *)
(* Member sets are read from CollectiveMembers rather than parsed out of    *)
(* the canonical_id: TLA+ has no string indexing, so a canonical_id cannot  *)
(* be decomposed here. CollectiveMembers is the exporter's decomposition of *)
(* that same id, and DeviceMeshValid separately checks that the issuing rank is *)
(* a member and that the member set is one of the mesh's allowed groups.    *)
(* MemberSetsAgreePerCommunicator adds that all works on one canonical_id   *)
(* record the same member set, which is what makes ObservedMembers a        *)
(* function of the communicator at all.                                    *)
(*                                                                         *)
(* THE MAPPING IS NOT UNIQUE, and it does not need to be. On this instance  *)
(* members plus admissible operations pin four of the eight images and      *)
(* leave two independent two-element swaps, so there are four valid         *)
(* mappings. The member classes are {0,1} and {2,3} with one observed and   *)
(* one model communicator each, pinned; and {0,2} and {1,3} with three of   *)
(* each, where mesh_fsdp is pinned by carrying three operations and         *)
(* mesh_batch and mesh_loss_mesh are interchangeable because the model      *)
(* gives them equal member sets and equal CommOps.                          *)
(*                                                                         *)
(* DeviceMeshModel reads a communicator only through                            *)
(* CommMembers, CommOps and identity, so swapping such a pair is an         *)
(* automorphism of Init, Next and every invariant: the choice is without    *)
(* loss of generality. CommMappingIsWellFormed asserts precisely the        *)
(* properties that make it so, rather than asserting one particular table.  *)
(***************************************************************************)
WorkSet == SequenceElements(CollectiveWorkIds)

WorksOnCommunicator(observedComm) ==
  { work \in WorkSet : CollectiveComm[work] = observedComm }

ObservedComms == { CollectiveComm[work] : work \in WorkSet }

\* Sound only under MemberSetsAgreePerCommunicator, which is asserted.
ObservedMembers(observedComm) ==
  UNION { CollectiveMembers[work] :
            work \in WorksOnCommunicator(observedComm) }

ObservedOps(observedComm) ==
  { CollectiveOperation[work] :
      work \in WorksOnCommunicator(observedComm) }

MemberSetsAgreePerCommunicator ==
  \A observedComm \in ObservedComms :
    \A a \in WorksOnCommunicator(observedComm) :
      \A b \in WorksOnCommunicator(observedComm) :
        CollectiveMembers[a] = CollectiveMembers[b]

MemberClasses == { ObservedMembers(oc) : oc \in ObservedComms }

ObservedCommsWithMembers(memberSet) ==
  { oc \in ObservedComms : ObservedMembers(oc) = memberSet }

ModelCommsWithMembers(memberSet) ==
  { c \in CommIds : CommMembers[c] = memberSet }

\* An injection from the observed communicators over one member set onto the
\* model communicators over that same member set, respecting the operations
\* each observed communicator was seen to carry.
ClassMappingOK(candidate, memberSet) ==
  /\ \A a \in DOMAIN candidate :
       \A b \in DOMAIN candidate :
         a # b => candidate[a] # candidate[b]
  /\ \A a \in DOMAIN candidate :
       ObservedOps(a) \subseteq CommOps[candidate[a]]

ClassMappingExists(memberSet) ==
  \E candidate \in [ObservedCommsWithMembers(memberSet)
                      -> ModelCommsWithMembers(memberSet)] :
    ClassMappingOK(candidate, memberSet)

(***************************************************************************)
(* Total by construction. CHOOSE is partial, and ModelCommOf is a constant  *)
(* definition, so TLC evaluates it while folding constants -- BEFORE any    *)
(* invariant could report the absence. The sentinel branch maps each        *)
(* observed communicator to itself, which is a canonical_id and therefore   *)
(* never an element of CommIds, so CommMappingIsWellFormed fails with a     *)
(* name instead of the search dying with an evaluation error.               *)
(***************************************************************************)
ClassMapping(memberSet) ==
  IF ClassMappingExists(memberSet)
  THEN CHOOSE candidate \in [ObservedCommsWithMembers(memberSet)
                               -> ModelCommsWithMembers(memberSet)] :
         ClassMappingOK(candidate, memberSet)
  ELSE [oc \in ObservedCommsWithMembers(memberSet) |-> oc]

ModelCommOf ==
  [oc \in ObservedComms |-> ClassMapping(ObservedMembers(oc))[oc]]

(***************************************************************************)
(* The mapping assertion. PrintT is the first conjunct so the derived       *)
(* mapping appears in the checker log exactly when it is checked: a         *)
(* reviewer reads the mapping the checker used, not one quoted in prose.    *)
(***************************************************************************)
CommMappingIsWellFormed ==
  /\ PrintT(<<"DEVICE_MESH_REFINE_COMM_MAPPING", ModelCommOf>>)
  /\ MemberSetsAgreePerCommunicator
  /\ \A memberSet \in MemberClasses : ClassMappingExists(memberSet)
  /\ DOMAIN ModelCommOf = ObservedComms
  /\ \A oc \in ObservedComms : ModelCommOf[oc] \in CommIds
  /\ \A a \in ObservedComms :
       \A b \in ObservedComms : a # b => ModelCommOf[a] # ModelCommOf[b]
  /\ { ModelCommOf[oc] : oc \in ObservedComms } = CommIds
  /\ \A oc \in ObservedComms :
       CommMembers[ModelCommOf[oc]] = ObservedMembers(oc)
  /\ \A oc \in ObservedComms :
       ObservedOps(oc) \subseteq CommOps[ModelCommOf[oc]]

(***************************************************************************)
(* THE OBSERVED ISSUE SEQUENCES.                                           *)
(*                                                                         *)
(* CollectiveIssueOrder values are positions in each rank's OWN event       *)
(* stream: distinct within a rank, spanning 11..332 with gaps, and          *)
(* repeating across ranks. They are not global indices, so the derivation   *)
(* groups by rank before ordering.                                         *)
(*                                                                         *)
(* The n-th element is found with MeshTopology's NthByIssueOrder, which *)
(* counts predecessors instead of sorting -- the hermetic toolchain has no  *)
(* community SequencesExt. That operator is a bare CHOOSE and so is partial *)
(* in the same way IndexOf is, and this is a constant definition that TLC   *)
(* folds before checking anything, so the call site establishes the witness *)
(* first and falls back to a sentinel record whose communicator is the      *)
(* empty string. ObservedIssueOrderIsDistinctWithinRank is the named        *)
(* invariant that turns the sentinel into a reported failure; it is checked *)
(* on the initial state, before Next is ever evaluated on the sentinel.     *)
(***************************************************************************)
RankWorks(rank) == { work \in WorkSet : CollectiveRank[work] = rank }

NthIssueRecord(works, n) ==
  LET predecessorsOf(work) ==
        Cardinality({ other \in works :
                        CollectiveIssueOrder[other]
                          < CollectiveIssueOrder[work] })
  IN IF \E work \in works : predecessorsOf(work) = n - 1
     THEN LET work == NthByIssueOrder(works, CollectiveIssueOrder, n)
          IN [comm |-> ModelCommOf[CollectiveComm[work]],
              op   |-> CollectiveOperation[work]]
     ELSE [comm |-> "", op |-> ""]

\* A constant definition on purpose: TLC folds it once while computing the
\* initial states rather than re-deriving it per state. The derivation is
\* quadratic in a rank's issue count.
ObservedIssues ==
  [rank \in Ranks |->
     LET works == RankWorks(rank)
     IN [n \in 1..Cardinality(works) |-> NthIssueRecord(works, n)]]

ObservedIndices == UNION { DOMAIN ObservedIssues[r] : r \in Ranks }

\* The model's rank set must be the observed one. Otherwise a trace from a
\* wider mesh would be replayed as whichever of its ranks the model happens to
\* name, silently dropping the rest, and every claim below would be about a
\* sub-run nobody asked about.
ObservedRanksMatchTheModel ==
  { CollectiveRank[work] : work \in WorkSet } = Ranks

ObservedIssueOrderIsDistinctWithinRank ==
  \A rank \in Ranks :
    \A a \in RankWorks(rank) :
      \A b \in RankWorks(rank) :
        a # b => CollectiveIssueOrder[a] # CollectiveIssueOrder[b]

\* AllDone requires Len(issued[r]) = MaxIssues for every rank, so unequal
\* observed issue counts would make completion unreachable for an arithmetic
\* reason and the positive result would read as a refusal.
ReplayLengthsAgreeAcrossRanks ==
  \A r1 \in Ranks :
    \A r2 \in Ranks : Len(ObservedIssues[r1]) = Len(ObservedIssues[r2])

\* The replay bound must lie inside the observed trace, or the run would be
\* asked to issue collectives that were never recorded.
ReplayBoundIsWithinTheObservedTrace ==
  \A rank \in Ranks : MaxIssues <= Len(ObservedIssues[rank])

ModelInstanceIsNonEmpty == Ranks # {} /\ CommIds # {}

(***************************************************************************)
(* THE NEGATIVE CONTROL.                                                   *)
(*                                                                         *)
(* The corruption must be refused for the RIGHT REASON, which here means at *)
(* the rendezvous guard -- StartAllowed's OpsAgreeAtFront under             *)
(* RequireMatchedIssueOrder -- and not at the stream-head conjunct or       *)
(* because a communicator was never issued.                                *)
(*                                                                         *)
(* That pins the shape of the mutation. Transposing two issues that are     *)
(* ADJACENT in one rank's issue order and are on the SAME communicator      *)
(* leaves the rank's communicator sequence, and therefore every member      *)
(* set, untouched: no communicator becomes unissued and no stream-head      *)
(* edge is created that was not there before. The two operations differ, so *)
(* that communicator's front position now carries different operations on   *)
(* two members, which is exactly the rendezvous mismatch. Transposing       *)
(* across DIFFERENT communicators would instead be refused because a        *)
(* member is missing from a communicator's front, which is the wrong guard. *)
(*                                                                         *)
(* The site is derived from the facts, minimal index first, then minimal    *)
(* rank, so it cannot drift away from the observed run; a checked-in        *)
(* mutated copy of the 1.6 MB facts module is not an option and was         *)
(* correctly refused once already. TransposableSites may be empty on some   *)
(* other trace, and MutationSite is folded with the constants, so the       *)
(* guarded form with the <<0, 0>> sentinel is required for the same reason  *)
(* as above; index 0 is outside every sequence domain, so the sentinel      *)
(* makes MutatedIssues equal to ObservedIssues and                          *)
(* TransposableIssuePairExists is the named invariant that reports it.      *)
(*                                                                         *)
(* WHICH GUARD REFUSES IT, AND UNDER WHICH CONFIGURATION. Say this plainly, *)
(* because a single-rank corruption breaks cross-rank AGREEMENT, and        *)
(* agreement is guarded in two places.                                     *)
(*                                                                         *)
(* On this trace the four ranks' operation sequences are positionally       *)
(* identical, so transposing on one rank makes the operations disagree at   *)
(* that position with every other rank. Under the POSITIVE's guard set       *)
(* RequireUniformProgramOps therefore refuses every PEER's issue at that     *)
(* position, so the corrupted COLUMN never forms: that is                    *)
(* DeviceMeshRefineBadUniform.cfg, where CorruptedColumnIsNeverFormed holds.     *)
(* The corrupted record itself is still issued, because the guard compares   *)
(* positions only up to the shorter of two sequences and the frontier        *)
(* position is not yet compared with anything. The communicator therefore    *)
(* never becomes fully pending and the job stops. That refusal is correct,   *)
(* and it is the real-world shape of the fault; it is simply not the guard   *)
(* this control exists to exercise.                                         *)
(*                                                                         *)
(* So the three configurations that isolate the rendezvous guard --         *)
(* DeviceMeshRefineBad, ...BadReach and ...BadRelaxed -- all set               *)
(* RequireUniformProgramOps = FALSE. That is necessary for ...BadRelaxed:   *)
(* with the SPMD guard on, flipping RequireMatchedIssueOrder does not admit *)
(* the corruption, so the attribution to that one constant collapses. It is *)
(* NOT necessary for ...BadReach, measured: with RequireUniformProgramOps   *)
(* TRUE that configuration's witness still appears with the same            *)
(* doneOn[MutationComm]. It is set there to keep the three on one constant  *)
(* set, so the only differences among them are the ones the runner's cfg    *)
(* diff checks.                                                            *)
(***************************************************************************)
IsTransposableAt(rank, index) ==
  IF index + 1 \in DOMAIN ObservedIssues[rank]
  THEN /\ ObservedIssues[rank][index].comm
            = ObservedIssues[rank][index + 1].comm
       /\ ObservedIssues[rank][index].op
            # ObservedIssues[rank][index + 1].op
  ELSE FALSE

TransposableSites ==
  { site \in Ranks \X ObservedIndices :
      IsTransposableAt(site[1], site[2]) }

TransposableIssuePairExists == TransposableSites # {}

MutationSite ==
  IF TransposableSites = {}
  THEN <<0, 0>>
  ELSE CHOOSE site \in TransposableSites :
         \A other \in TransposableSites :
           \/ site[2] < other[2]
           \/ (site[2] = other[2] /\ site[1] <= other[1])

MutationRank == MutationSite[1]
MutationIndex == MutationSite[2]

MutatedIssues ==
  [rank \in Ranks |->
     IF rank = MutationRank /\ MutationIndex # 0
     THEN [ index \in DOMAIN ObservedIssues[rank] |->
              CASE index = MutationIndex
                     -> ObservedIssues[rank][MutationIndex + 1]
                [] index = MutationIndex + 1
                     -> ObservedIssues[rank][MutationIndex]
                [] OTHER
                     -> ObservedIssues[rank][index] ]
     ELSE ObservedIssues[rank]]

(***************************************************************************)
(* THE ORDER-SENSITIVITY CONTROLS.                                         *)
(*                                                                         *)
(* A permutation of the POSITIONS applied uniformly to all four ranks       *)
(* permutes whole columns, so every column is still a column: the           *)
(* operations still agree at every position, the communicators are still at *)
(* the same site wherever they share a rank, and the induced collectives    *)
(* are the same multiset. It must therefore be admitted, and                *)
(* UniformPermutationPreservesAgreement asserts exactly the two predicates  *)
(* the model uses to decide that. Reversal is used because it is the        *)
(* furthest such permutation from the identity and needs no parameter.      *)
(*                                                                         *)
(* Applying it to ONE rank only breaks the columns, and                     *)
(* SingleRankPermutationBreaksAgreement asserts that it does, through the   *)
(* model's own predicates rather than by inspection. On this trace it is    *)
(* the SITE half that fails first -- rank 0's last issue is on              *)
(* mesh_loss_mesh while its peer's first is on mesh_batch, and those two    *)
(* communicators share ranks 0 and 2 -- and the operations happen to agree, *)
(* which is why the asserted property is the disjunction.                   *)
(***************************************************************************)
ReverseSeq(sequence) ==
  [index \in DOMAIN sequence |-> sequence[Len(sequence) - index + 1]]

UniformlyPermutedIssues ==
  [rank \in Ranks |-> ReverseSeq(ObservedIssues[rank])]

\* Guarded because Ranks could be empty; 0 is then never equal to any rank,
\* so the permutation degenerates to the observed order and
\* ModelInstanceIsNonEmpty is the named invariant that reports it.
ScrambleRank ==
  IF Ranks = {}
  THEN 0
  ELSE CHOOSE rank \in Ranks : \A other \in Ranks : rank <= other

SingleRankPermutedIssues ==
  [rank \in Ranks |->
     IF rank = ScrambleRank
     THEN ReverseSeq(ObservedIssues[rank])
     ELSE ObservedIssues[rank]]

ReplayIssues ==
  CASE TransposeMutation                    -> MutatedIssues
    [] ReplayPermutation = "uniform"        -> UniformlyPermutedIssues
    [] ReplayPermutation = "single_rank"    -> SingleRankPermutedIssues
    [] OTHER                                -> ObservedIssues

\* Non-vacuity for both controls: a permutation equal to the observed order
\* would make either configuration a restatement of the positive.
UniformPermutationIsNotTheObservedOrder ==
  \E rank \in Ranks : UniformlyPermutedIssues[rank] # ObservedIssues[rank]

SingleRankPermutationIsNotTheObservedOrder ==
  \E rank \in Ranks :
    SingleRankPermutedIssues[rank] # ObservedIssues[rank]

UniformPermutationPreservesAgreement ==
  /\ UniformProgramOpsOK(UniformlyPermutedIssues)
  /\ UniformProgramCommsOK(UniformlyPermutedIssues)

SingleRankPermutationBreaksAgreement ==
  \/ ~UniformProgramOpsOK(SingleRankPermutedIssues)
  \/ ~UniformProgramCommsOK(SingleRankPermutedIssues)

\* Exactly two positions change, on exactly one rank, on one communicator,
\* with the operations differing. Guards this control against silently
\* becoming a no-op or becoming a different kind of corruption.
MutationIsIsolated ==
  IF MutationIndex = 0
  THEN FALSE
  ELSE /\ \A rank \in Ranks :
             /\ DOMAIN MutatedIssues[rank] = DOMAIN ObservedIssues[rank]
             /\ { index \in DOMAIN ObservedIssues[rank] :
                    MutatedIssues[rank][index]
                      # ObservedIssues[rank][index] }
                  = (IF rank = MutationRank
                     THEN {MutationIndex, MutationIndex + 1}
                     ELSE {})
       /\ ObservedIssues[MutationRank][MutationIndex].comm
            = ObservedIssues[MutationRank][MutationIndex + 1].comm
       /\ ObservedIssues[MutationRank][MutationIndex].op
            # ObservedIssues[MutationRank][MutationIndex + 1].op

\* The communicator whose rendezvous the mutation corrupts, and the
\* per-communicator position of the corrupted rendezvous on it.
MutationComm ==
  IF MutationIndex = 0
  THEN ""
  ELSE ObservedIssues[MutationRank][MutationIndex].comm

MutationFront ==
  IF MutationIndex = 0
  THEN 0
  ELSE Cardinality({ index \in 1..MutationIndex :
                       ObservedIssues[MutationRank][index].comm
                         = MutationComm })

(***************************************************************************)
(* THE REPLAY.                                                             *)
(*                                                                         *)
(* SingleRankRefine writes the constraint as Next /\ IsPrefixOfObserved,        *)
(* filtering successors after Next has produced them. Here that form would  *)
(* evaluate IssueAllowed -- whose SPMD-program conjuncts are linear in the  *)
(* issue count -- for all 4 x 8 x 3 rank/communicator/operation triples per *)
(* state, and discard almost all of them. The forward form below is         *)
(* EQUIVALENT, not weaker: Issue appends exactly one record to exactly one  *)
(* rank, so if issued is a prefix of ReplayIssues and issued' is too, the   *)
(* appended record can only be ReplayIssues[r][Len(issued[r]) + 1]. Init    *)
(* establishes the premise with the empty prefix.                          *)
(*                                                                         *)
(* IssuedIsAReplayPrefix is checked as an invariant in every configuration  *)
(* so that "issued stays a prefix of the observed issue order" is a checked *)
(* fact about the run rather than a property of how it was written.         *)
(*                                                                         *)
(* Start and Complete are the model's own actions, unrestricted except by   *)
(* the search policy. No observed evidence reaches them.                    *)
(***************************************************************************)
ReplayIssue(rank) ==
  /\ Len(issued[rank]) < Len(ReplayIssues[rank])
  /\ LET nextIssue == ReplayIssues[rank][Len(issued[rank]) + 1]
     IN Issue(rank, nextIssue.comm, nextIssue.op)

IssuedIsAReplayPrefix ==
  \A rank \in Ranks :
    /\ Len(issued[rank]) <= Len(ReplayIssues[rank])
    /\ \A index \in DOMAIN issued[rank] :
         issued[rank][index] = ReplayIssues[rank][index]

(***************************************************************************)
(* THE SEARCH POLICY.                                                      *)
(*                                                                         *)
(* Drain-then-advance. A rank may issue only when nothing is running and    *)
(* every collective it has already issued has completed; a collective may   *)
(* start only when nothing else is running. That makes the reachable graph  *)
(* essentially linear in the trace: each round is four issues and the two   *)
(* start/complete pairs they enable, whose interleavings recombine          *)
(* immediately.                                                            *)
(*                                                                         *)
(* IssuerIsDrained is deliberately cheaper than "no communicator can        *)
(* start": it reads doneOn and CommCount only, where StartAllowed would go  *)
(* through IdxOf and AtStreamHead for all eight communicators at every      *)
(* state. It keeps the ranks close together, since a rank cannot issue      *)
(* again until the collectives it already issued have run, which needs its  *)
(* peers -- but it does not make them lockstep on its own: ranks 0 and 3    *)
(* share no communicator on this mesh, so nothing forces them together and  *)
(* the unbounded search does visit a spread of two. MaxReplaySkew is what   *)
(* bounds the spread, the shipped configuration binds it to 1, and the      *)
(* runner reports the bound the configuration binds.                        *)
(*                                                                         *)
(* This is a restriction of the SEARCH. See SOUNDNESS OF THE GREEDY REPLAY  *)
(* in the header for why a witness found under it is a witness for Next.    *)
(***************************************************************************)
IssuerIsDrained(rank) ==
  \A c \in CommIds :
    rank \in CommMembers[c] => doneOn[c] = CommCount(rank, c)

GreedyIssuePolicy(rank) ==
  \/ ~GreedyReplay
  \/ (running = {} /\ IssuerIsDrained(rank))

GreedyStartPolicy == ~GreedyReplay \/ running = {}

ReplaySkewOK(candidate) ==
  \A r1 \in Ranks :
    \A r2 \in Ranks :
      Len(candidate[r1]) <= Len(candidate[r2]) + MaxReplaySkew

\* Issue counts of a candidate issue sequence on one communicator. Written
\* over a sequence rather than reusing DeviceMeshModel's CommCount, which reads
\* the unprimed variable and so cannot speak about issued'.
CommCountIn(sequence, c) ==
  Cardinality({ index \in DOMAIN sequence : sequence[index].comm = c })

\* Complete only ever raises doneOn, so this is a condition on Issue alone.
ReplayOutstandingOK(candidate) ==
  \A rank \in Ranks :
    \A c \in CommIds :
      rank \in CommMembers[c] =>
        CommCountIn(candidate[rank], c) <= doneOn[c] + MaxOutstanding

ConstrainedNext ==
  \/ \E rank \in Ranks :
       /\ GreedyIssuePolicy(rank)
       /\ ReplayIssue(rank)
       /\ ReplaySkewOK(issued')
       /\ ReplayOutstandingOK(issued')
  \/ \E c \in CommIds : GreedyStartPolicy /\ Start(c)
  \/ \E c \in CommIds : Complete(c)
  \/ Terminated

RSpec == Init /\ [][ConstrainedNext]_vars

(***************************************************************************)
(* ADMISSION.                                                              *)
(*                                                                         *)
(* Violating this invariant is the refinement witness. AllDone is every     *)
(* rank at MaxIssues issues -- which IssuedIsAReplayPrefix pins to the      *)
(* observed order, and ReplayBoundIsWithinTheObservedTrace pins to a real   *)
(* prefix of the recorded trace -- with nothing running and every           *)
(* communicator's completion count equal to its members' issue counts.      *)
(***************************************************************************)
ReplayedRunIsNotAdmitted == ~AllDone

\* The spread the configuration actually imposes, checked rather than
\* described, so a cfg that bound MaxReplaySkew more loosely than advertised
\* cannot report the tighter number.
ReplaySkewIsWithinBound == ReplaySkewOK(issued)

\* The overlap window the configuration actually imposes, checked for the same
\* reason as the spread.
ReplayOutstandingIsWithinBound == ReplayOutstandingOK(issued)

(***************************************************************************)
(* WHAT THE SKEW CONFIGURATION ESTABLISHES, and why deadlock detection is   *)
(* the right instrument for it.                                            *)
(*                                                                         *)
(* DeviceMeshRefineSkew.cfg turns TLC's own deadlock check ON and explores the  *)
(* whole space exhaustively. Terminated self-loops on AllDone alone, so a   *)
(* state with no successor is a state that is NOT complete and cannot make  *)
(* progress. None being found means every maximal execution in that space   *)
(* ends in the self-loop, because the graph is acyclic apart from it: the   *)
(* pair                                                                    *)
(*                                                                         *)
(*   << SUM of Len(issued[r]) + SUM of doneOn[c], Cardinality(running) >>   *)
(*                                                                         *)
(* increases lexicographically at every step -- Issue and Complete raise    *)
(* the first component, Start leaves it and raises the second.              *)
(*                                                                         *)
(* So "no deadlock over the whole space" is exactly: EVERY schedule of the  *)
(* observed issue order within the bound reaches completion. That is the    *)
(* confluence statement, machine-checked, over the first MaxIssues issues   *)
(* of each rank and a per-rank issue skew of at most MaxReplaySkew. It is   *)
(* not a statement about the full 108-issue trace or about unbounded skew,  *)
(* and the runner reports both numbers so it cannot be read as one.         *)
(*                                                                         *)
(* The skew bound removes transitions, so it could in principle create a    *)
(* dead end of its own. It does not: a rank of minimal issue count can      *)
(* always issue without widening the spread, and if the minimum equals      *)
(* MaxIssues then every rank is at MaxIssues and no Issue was wanted.       *)
(***************************************************************************)

\* Non-vacuity for the bounded-skew configuration, checked by refutation in
\* DeviceMeshRefineOverlap.cfg. The greedy search serializes collectives by
\* construction, and a serialized schedule satisfies every guard trivially, so
\* a confluence claim drawn from a space that contained only serialized
\* schedules would be worth little. Refuting this shows the bounded-skew space
\* contains states with two collectives in flight at once -- the overlap that
\* FSDP produces and that the observed started/completed sub-order cannot
\* express.
NoTwoCollectivesRunConcurrently == Cardinality(running) <= 1

\* The corrupted rendezvous never starts and never completes; earlier
\* collectives on the same communicator are free to run. This closes the gap
\* that non-admission alone leaves open, namely that the model might admit
\* the mismatch and refuse something later.
MismatchedCollectiveNeverRuns ==
  IF MutationIndex = 0
  THEN FALSE
  ELSE /\ doneOn[MutationComm] < MutationFront
       /\ (MutationComm \in running
             => doneOn[MutationComm] + 1 < MutationFront)

(***************************************************************************)
(* Under the POSITIVE's guard set the corrupted COLUMN never forms. Measured *)
(* rather than assumed, because the first guess was wrong: the corrupted    *)
(* record itself IS issued. UniformProgramOpsOK compares positions only up  *)
(* to the shorter of two ranks' sequences, so at the frontier -- where the   *)
(* skew bound leaves the peers one issue behind -- the position the          *)
(* corrupted record occupies is not yet compared with anything, and the      *)
(* Issue is allowed. What the guard then refuses is every PEER's issue at    *)
(* that same position, because by then the disagreement is inside the        *)
(* compared range. So no other rank ever reaches MutationIndex, the          *)
(* communicator never becomes fully pending, and nothing downstream of the   *)
(* column can run.                                                          *)
(*                                                                         *)
(* That is the real-world shape of the fault: one rank runs ahead, its peers *)
(* cannot follow, and the job stops. Holding here and being violated in the  *)
(* relaxed configurations is what separates the two refusers.                *)
(***************************************************************************)
CorruptedColumnIsNeverFormed ==
  IF MutationIndex = 0
  THEN FALSE
  ELSE \E rank \in Ranks : Len(issued[rank]) < MutationIndex

\* Violating this is the witness that the model accepted every issue up to
\* AND INCLUDING the corrupted one, and ran every collective before the
\* corrupted rendezvous. The first disjunct strengthens the witness to say
\* the corrupted record was accepted as an issue; it does NOT by itself say
\* which guard refuses the run, and measured it does not distinguish them --
\* the witness appears at the same depth with RequireUniformProgramOps either
\* way, because the corrupted record is issued in both. What distinguishes
\* them is CorruptedColumnIsNeverFormed above. TRUE under the sentinel so a
\* trace with no transposable pair yields no witness rather than a vacuous
\* one.
RejectionHappensBeforeTheRendezvous ==
  IF MutationIndex = 0
  THEN TRUE
  ELSE \/ Len(issued[MutationRank]) < MutationIndex
       \/ doneOn[MutationComm] < MutationFront - 1

=============================================================================
