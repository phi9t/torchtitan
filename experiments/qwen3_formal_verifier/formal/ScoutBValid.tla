------------------------------ MODULE ScoutBValid ------------------------------
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

ScoutBMeshWellFormed ==
  MeshWellFormed(RankSet, RankCoordinates)

ScoutBInputIdentity ==
  TpInputAgreement(RankSet, RankCoordinates, InputDigestByRank)

ScoutBPlacementValid ==
  PlacementValid(RankSet, PlacementHasDpShard, PlacementHasTpShard)

ScoutBEventEvidence ==
  EventEvidenceValid(
    RawEventProjectionSchema,
    EventIds,
    EventRank,
    EventOrder,
    EventKind,
    EventRawId,
    EventProjectionSchema,
    EventProjectionDigest,
    EventPredecessors
  )

ScoutBCrossRankSynchronization ==
  CrossRankSynchronizationValid(
    EventIds,
    EventRank,
    CrossRankEdgeIds,
    CrossRankEndpoints,
    CrossRankRelation,
    CrossRankOrdered,
    CrossRankEvidenceDigest
  )

ScoutBCollectiveLifecycle ==
  CollectiveLifecycleValid(
    CollectiveWorkIds,
    CollectiveRank,
    CollectiveMembers,
    CollectiveLifecycle,
    CollectiveHasExecutor,
    CollectiveHasStream,
    CollectiveObservationCount
  )

ScoutBCollectiveCoverage ==
  CollectiveCoverage(
    CollectiveWorkIds,
    CollectiveRank,
    CollectiveId,
    CollectiveMembers,
    CollectiveObservationCount
  )

ScoutBCollectiveProducerCorrelation ==
  CollectiveProducerCorrelation(CollectiveWorkIds, CollectiveProducers)

\* THE property: every member of a communicator issued the same collectives in
\* the same order. Violating this hangs the job.
ScoutBPerCommunicatorIssueOrder ==
  PerCommunicatorIssueOrderAgreement(
    CollectiveWorkIds,
    CollectiveRank,
    CollectiveComm,
    CollectiveOperation,
    CollectiveMembers,
    CollectiveIssueOrder
  )

\* Guards against the exporter confusing the global communicator key with the
\* per-rank runtime pg id, which denotes different communicators by rank.
ScoutBRuntimePgIdConsistent ==
  RuntimePgIdAgreesWithinRank(
    CollectiveWorkIds, CollectiveRank, CollectiveComm, CollectiveRuntimePgId
  )

ScoutBCollectiveEventEvidence ==
  CollectiveEventEvidence(
    CollectiveWorkIds,
    CollectiveRank,
    CollectiveEventIds,
    EventIds,
    EventRank,
    EventKind
  )

=============================================================================
