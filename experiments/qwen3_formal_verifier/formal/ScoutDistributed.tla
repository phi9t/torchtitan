------------------------------ MODULE ScoutDistributed ------------------------------
EXTENDS Naturals, Sequences, FiniteSets

SequenceElements(sequence) ==
  {sequence[index] : index \in DOMAIN sequence}

ExpectedCoordinates == {
  <<0, 0>>,
  <<0, 1>>,
  <<1, 0>>,
  <<1, 1>>
}

AllowedGroups == {
  {0, 1},
  {2, 3},
  {0, 2},
  {1, 3},
  {0, 1, 2, 3}
}

MeshWellFormed(rankSet, rankCoordinates) ==
  /\ DOMAIN rankCoordinates = rankSet
  /\ {rankCoordinates[rank] : rank \in rankSet} = ExpectedCoordinates
  /\ \A rank \in rankSet :
       /\ rankCoordinates[rank][1] \in 0..1
       /\ rankCoordinates[rank][2] \in 0..1

TpInputAgreement(rankSet, rankCoordinates, inputDigestByRank) ==
  /\ DOMAIN inputDigestByRank = rankSet
  /\ \A left \in rankSet :
       \A right \in rankSet :
         rankCoordinates[left][1] = rankCoordinates[right][1]
           => inputDigestByRank[left] = inputDigestByRank[right]

PlacementValid(rankSet, hasDpShard, hasTpShard) ==
  /\ DOMAIN hasDpShard = rankSet
  /\ DOMAIN hasTpShard = rankSet
  /\ \A rank \in rankSet : hasDpShard[rank] /\ hasTpShard[rank]

EventEvidenceValid(
  projectionSchema,
  eventIds,
  eventRank,
  eventOrder,
  eventKind,
  eventRawId,
  eventProjectionSchema,
  eventProjectionDigest,
  eventPredecessors
) ==
  LET eventSet == SequenceElements(eventIds)
  IN /\ projectionSchema = "qwen3.formal.raw-event-projection.v0"
     /\ Len(eventIds) = Cardinality(eventSet)
     /\ DOMAIN eventRank = eventSet
     /\ DOMAIN eventOrder = eventSet
     /\ DOMAIN eventKind = eventSet
     /\ DOMAIN eventRawId = eventSet
     /\ DOMAIN eventProjectionSchema = eventSet
     /\ DOMAIN eventProjectionDigest = eventSet
     /\ DOMAIN eventPredecessors = eventSet
     /\ \A event \in eventSet :
          /\ eventRank[event] \in 0..3
          /\ eventOrder[event] > 0
          /\ eventKind[event] # ""
          /\ eventRawId[event] # ""
          /\ eventProjectionSchema[event] = projectionSchema
          /\ eventProjectionDigest[event] # ""
          /\ \A predecessor \in SequenceElements(eventPredecessors[event]) :
               /\ predecessor \in eventSet
               /\ eventRank[predecessor] = eventRank[event]
               /\ eventOrder[predecessor] < eventOrder[event]

CrossRankSynchronizationValid(
  eventIds,
  eventRank,
  edgeIds,
  endpoints,
  relation,
  ordered,
  evidenceDigest
) ==
  LET eventSet == SequenceElements(eventIds)
      edgeSet == SequenceElements(edgeIds)
  IN /\ Len(edgeIds) = Cardinality(edgeSet)
     /\ DOMAIN endpoints = edgeSet
     /\ DOMAIN relation = edgeSet
     /\ DOMAIN ordered = edgeSet
     /\ DOMAIN evidenceDigest = edgeSet
     /\ \A edge \in edgeSet :
          /\ Len(endpoints[edge]) = 2
          /\ endpoints[edge][1] \in eventSet
          /\ endpoints[edge][2] \in eventSet
          /\ endpoints[edge][1] # endpoints[edge][2]
          /\ eventRank[endpoints[edge][1]] # eventRank[endpoints[edge][2]]
          /\ relation[edge] = "synchronizes"
          /\ ~ordered[edge]
          /\ evidenceDigest[edge] # ""

CollectiveLifecycleValid(
  workIds,
  collectiveRank,
  collectiveMembers,
  collectiveLifecycle,
  collectiveHasExecutor,
  collectiveHasStream,
  collectiveObservationCount
) ==
  LET workSet == SequenceElements(workIds)
  IN /\ DOMAIN collectiveRank = workSet
     /\ DOMAIN collectiveMembers = workSet
     /\ DOMAIN collectiveLifecycle = workSet
     /\ DOMAIN collectiveHasExecutor = workSet
     /\ DOMAIN collectiveHasStream = workSet
     /\ DOMAIN collectiveObservationCount = workSet
     /\ \A work \in workSet :
          /\ collectiveMembers[work] \in AllowedGroups
          /\ collectiveRank[work] \in collectiveMembers[work]
          /\ collectiveLifecycle[work] = <<"enqueued", "started", "completed">>
          /\ collectiveHasExecutor[work]
          /\ collectiveHasStream[work]
          /\ collectiveObservationCount[work] > 0

CollectiveCoverage(
  workIds,
  collectiveRank,
  collectiveId,
  collectiveMembers,
  collectiveObservationCount
) ==
  LET workSet == SequenceElements(workIds)
  IN \A work \in workSet :
       {collectiveRank[peer] : peer \in
         {candidate \in workSet : collectiveId[candidate] = collectiveId[work]}}
         = collectiveMembers[work]
       /\ \A peer \in workSet :
            collectiveId[peer] = collectiveId[work]
              => collectiveObservationCount[peer] = collectiveObservationCount[work]

CollectiveProducerCorrelation(workIds, collectiveProducers) ==
  LET workSet == SequenceElements(workIds)
  IN /\ DOMAIN collectiveProducers = workSet
     /\ \A work \in workSet :
          /\ Len(collectiveProducers[work]) = 3
          /\ Cardinality(SequenceElements(collectiveProducers[work])) = 1

CollectiveEventEvidence(
  workIds,
  collectiveRank,
  collectiveEventIds,
  eventIds,
  eventRank,
  eventKind
) ==
  LET workSet == SequenceElements(workIds)
      eventSet == SequenceElements(eventIds)
  IN /\ Len(workIds) = Cardinality(workSet)
     /\ DOMAIN collectiveEventIds = workSet
     /\ \A work \in workSet :
          /\ Len(collectiveEventIds[work]) = 3
          /\ \A event \in SequenceElements(collectiveEventIds[work]) :
               /\ event \in eventSet
               /\ eventRank[event] = collectiveRank[work]
          /\ eventKind[collectiveEventIds[work][1]] = "collective.enqueued"
          /\ eventKind[collectiveEventIds[work][2]] = "collective.started"
          /\ eventKind[collectiveEventIds[work][3]] = "collective.completed"


(***************************************************************************)
(* Per-communicator issue-order agreement.                                 *)
(*                                                                         *)
(* This is the NCCL requirement: every member rank of a communicator must  *)
(* issue the same collectives in the same order. Violating it hangs the    *)
(* job, which makes it the one property here whose failure has a direct    *)
(* operational meaning.                                                    *)
(*                                                                         *)
(* Three things make this checkable where the older predicates were not.   *)
(* The communicator key is the global canonical_id, not the per-rank       *)
(* runtime_pg_id (which denotes two different communicators depending on   *)
(* the rank holding it). The operation is a first-class value rather than  *)
(* a substring of an identifier. And the order is the collective.enqueued  *)
(* event order, which inherits the Flight Recorder record sequence -- the  *)
(* started/completed sub-order is appended post hoc and is not evidence.   *)
(*                                                                         *)
(* Ordering ACROSS communicators is deliberately unconstrained: it is not  *)
(* required, and requiring it here would reject correct executions.        *)
(***************************************************************************)

\* The works one rank issued on one communicator.
WorksOnComm(workIds, collectiveRank, collectiveComm, rank, comm) ==
  { work \in SequenceElements(workIds) :
      /\ collectiveRank[work] = rank
      /\ collectiveComm[work] = comm }

\* The n-th of those in issue order, expressed by counting predecessors so the
\* definition needs no sorting operator and therefore no community modules.
\* Issue orders are distinct within a rank, so this is well defined.
NthByIssueOrder(works, issueOrder, n) ==
  CHOOSE work \in works :
    Cardinality({ other \in works : issueOrder[other] < issueOrder[work] }) = n - 1

PerCommunicatorIssueOrderAgreement(
    workIds, collectiveRank, collectiveComm, collectiveOperation,
    collectiveMembers, issueOrder) ==
  LET workSet == SequenceElements(workIds)
      comms   == { collectiveComm[work] : work \in workSet }
  IN \A comm \in comms :
       LET onComm  == { w \in workSet : collectiveComm[w] = comm }
           members == UNION { collectiveMembers[w] : w \in onComm }
       IN \A r1 \in members :
            \A r2 \in members :
              LET w1 == WorksOnComm(workIds, collectiveRank, collectiveComm, r1, comm)
                  w2 == WorksOnComm(workIds, collectiveRank, collectiveComm, r2, comm)
              IN /\ Cardinality(w1) = Cardinality(w2)
                 /\ \A n \in 1..Cardinality(w1) :
                      collectiveOperation[NthByIssueOrder(w1, issueOrder, n)]
                        = collectiveOperation[NthByIssueOrder(w2, issueOrder, n)]

\* runtime_pg_id is per-rank local. Exported only so this can assert that,
\* within one rank, it agrees with the global communicator key -- catching an
\* exporter that mixed the two up.
RuntimePgIdAgreesWithinRank(workIds, collectiveRank, collectiveComm, runtimePgId) ==
  LET workSet == SequenceElements(workIds)
  IN \A a \in workSet :
       \A b \in workSet :
         collectiveRank[a] = collectiveRank[b] =>
           (collectiveComm[a] = collectiveComm[b] <=> runtimePgId[a] = runtimePgId[b])

=============================================================================
