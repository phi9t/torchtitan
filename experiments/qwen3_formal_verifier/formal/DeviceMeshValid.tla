------------------------------ MODULE DeviceMeshValid ------------------------------
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

DeviceMeshMeshWellFormed ==
  MeshWellFormed(RankSet, RankCoordinates)

DeviceMeshInputIdentity ==
  TpInputAgreement(RankSet, RankCoordinates, InputDigestByRank)

DeviceMeshPlacementValid ==
  PlacementValid(RankSet, PlacementHasDpShard, PlacementHasTpShard)

\* Unchanged in meaning: still only "some parameter is DP-sharded and some
\* parameter is TP-sharded". DeviceMeshPlacementBooleansAgree below ties the two
\* booleans it reads to the per-parameter facts, so the
\* weak check keeps its original content instead of quietly becoming a
\* different, stronger claim under the same name.

DeviceMeshPlacementSchemaAgrees ==
  PlacementSchemaAgreesAcrossRanks(
    RankSet, PlacementSchemaDigestByRank, PlacementSchemaDigest
  )

DeviceMeshMeshAxisDegree ==
  MeshAxisDegreeAgreesWithCoordinates(RankSet, RankCoordinates, MeshAxisDegree)

DeviceMeshParameterPlacementWellFormed ==
  ParameterPlacementWellFormed(
    ParameterNames,
    ParameterMeshAxes,
    ParameterGlobalShape,
    ParameterLocalShape,
    PlacementIds,
    PlacementParameter,
    PlacementAxis,
    PlacementKind,
    PlacementStrided,
    PlacementShardDim,
    MeshAxisDegree
  )

\* Named for what it proves: the booleans agree with the one exported placement
\* list, which is one bit per axis, not a per-rank derivation. See the
\* predicate's header.
DeviceMeshPlacementBooleansAgree ==
  PlacementBooleansAgreeWithFacts(
    RankSet,
    PlacementHasDpShard,
    PlacementHasTpShard,
    PlacementIds,
    PlacementAxis,
    PlacementKind
  )

\* THE property: a Partial placement on a parameter the optimizer steps is an
\* unreduced value and silently wrong math. OptimizerBoundaryObserved is
\* conjoined so the claim is about an optimizer step the trace actually
\* contains.
DeviceMeshNoPartialAtOptimizer ==
  /\ OptimizerBoundaryObserved(
       RankSet, EventIds, EventRank, EventKind, EventOrder)
  /\ NoPartialParameterPlacement(PlacementIds, PlacementKind)

DeviceMeshShardedDimDividesAxisDegree ==
  ShardedDimDividesAxisDegree(
    PlacementIds,
    PlacementParameter,
    PlacementAxis,
    PlacementKind,
    PlacementShardDim,
    ParameterGlobalShape,
    MeshAxisDegree
  )

DeviceMeshLocalShapeReflectsSharding ==
  LocalShapeReflectsSharding(
    ParameterNames,
    ParameterGlobalShape,
    ParameterLocalShape,
    PlacementIds,
    PlacementParameter,
    PlacementAxis,
    PlacementKind,
    PlacementShardDim,
    MeshAxisDegree
  )

DeviceMeshStridedShardComposition ==
  StridedShardIsAnOuterComposedShard(
    ParameterMeshAxes,
    PlacementIds,
    PlacementParameter,
    PlacementAxis,
    PlacementKind,
    PlacementStrided,
    PlacementShardDim
  )

DeviceMeshEventEvidence ==
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

DeviceMeshCrossRankSynchronization ==
  CrossRankSynchronizationValid(
    EventIds,
    EventRank,
    CrossRankEdgeIds,
    CrossRankEndpoints,
    CrossRankRelation,
    CrossRankOrdered,
    CrossRankEvidenceDigest
  )

DeviceMeshCollectiveLifecycle ==
  CollectiveLifecycleValid(
    CollectiveWorkIds,
    CollectiveRank,
    CollectiveMembers,
    CollectiveLifecycle,
    CollectiveHasExecutor,
    CollectiveHasStream,
    CollectiveObservationCount
  )

DeviceMeshCollectiveCoverage ==
  CollectiveCoverage(
    CollectiveWorkIds,
    CollectiveRank,
    CollectiveId,
    CollectiveMembers,
    CollectiveObservationCount
  )

DeviceMeshCollectiveProducerCorrelation ==
  CollectiveProducerCorrelation(CollectiveWorkIds, CollectiveProducers)

\* THE property: every member of a communicator issued the same collectives in
\* the same order. Violating this hangs the job.
DeviceMeshPerCommunicatorIssueOrder ==
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
DeviceMeshRuntimePgIdConsistent ==
  RuntimePgIdAgreesWithinRank(
    CollectiveWorkIds, CollectiveRank, CollectiveComm, CollectiveRuntimePgId
  )

\* Collective payload identity. OBSERVED off the Flight Recorder, per tensor,
\* and keyed on the canonical communicator id -- never runtime_pg_id.
DeviceMeshCollectivePayloadWellFormed ==
  CollectivePayloadWellFormed(
    CollectiveWorkIds,
    CollectivePayloadComm,
    CollectivePayloadInputSizes,
    CollectivePayloadOutputSizes,
    CollectivePayloadInputDtypes,
    CollectivePayloadOutputDtypes,
    CollectivePayloadInputElements,
    CollectivePayloadOutputElements
  )

\* Guards the payload against being keyed on the per-rank runtime pg id, which
\* would put communicators with different member sets under one key.
DeviceMeshPayloadCommKey ==
  PayloadCommKeyIsCommunicatorIdentity(
    CollectiveWorkIds,
    CollectivePayloadComm,
    CollectiveComm,
    CollectiveMembers
  )

\* THE property: the members of one collective agree on dtype and on volume. A
\* mismatch corrupts or errors rather than hanging, so ordering agreement says
\* nothing about it.
DeviceMeshCollectivePayloadAgreement ==
  CollectivePayloadAgreement(
    CollectiveWorkIds,
    CollectiveId,
    CollectivePayloadInputDtypes,
    CollectivePayloadOutputDtypes,
    CollectivePayloadInputElements,
    CollectivePayloadOutputElements
  )

\* THE property: the volume relation the operation implies. This is what would
\* catch an exporter that mislabelled an operation.
DeviceMeshCollectivePayloadSizeRelation ==
  CollectivePayloadSizeRelationHolds(
    CollectiveWorkIds,
    CollectiveOperation,
    CollectiveMembers,
    CollectivePayloadSizeRelation,
    CollectivePayloadInputElements,
    CollectivePayloadOutputElements
  )

DeviceMeshCollectiveEventEvidence ==
  CollectiveEventEvidence(
    CollectiveWorkIds,
    CollectiveRank,
    CollectiveEventIds,
    EventIds,
    EventRank,
    EventKind
  )

=============================================================================
