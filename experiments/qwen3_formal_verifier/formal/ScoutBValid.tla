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

\* Unchanged in meaning: still only "some parameter is DP-sharded and some
\* parameter is TP-sharded". ScoutBPlacementBooleansAgree below ties the two
\* booleans it reads to the per-parameter facts, so the
\* weak check keeps its original content instead of quietly becoming a
\* different, stronger claim under the same name.

ScoutBPlacementSchemaAgrees ==
  PlacementSchemaAgreesAcrossRanks(
    RankSet, PlacementSchemaDigestByRank, PlacementSchemaDigest
  )

ScoutBMeshAxisDegree ==
  MeshAxisDegreeAgreesWithCoordinates(RankSet, RankCoordinates, MeshAxisDegree)

ScoutBParameterPlacementWellFormed ==
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
ScoutBPlacementBooleansAgree ==
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
ScoutBNoPartialAtOptimizer ==
  /\ OptimizerBoundaryObserved(
       RankSet, EventIds, EventRank, EventKind, EventOrder)
  /\ NoPartialParameterPlacement(PlacementIds, PlacementKind)

ScoutBShardedDimDividesAxisDegree ==
  ShardedDimDividesAxisDegree(
    PlacementIds,
    PlacementParameter,
    PlacementAxis,
    PlacementKind,
    PlacementShardDim,
    ParameterGlobalShape,
    MeshAxisDegree
  )

ScoutBLocalShapeReflectsSharding ==
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

ScoutBStridedShardComposition ==
  StridedShardIsAnOuterComposedShard(
    ParameterMeshAxes,
    PlacementIds,
    PlacementParameter,
    PlacementAxis,
    PlacementKind,
    PlacementStrided,
    PlacementShardDim
  )

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

\* Collective payload identity. OBSERVED off the Flight Recorder, per tensor,
\* and keyed on the canonical communicator id -- never runtime_pg_id.
ScoutBCollectivePayloadWellFormed ==
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
ScoutBPayloadCommKey ==
  PayloadCommKeyIsCommunicatorIdentity(
    CollectiveWorkIds,
    CollectivePayloadComm,
    CollectiveComm,
    CollectiveMembers
  )

\* THE property: the members of one collective agree on dtype and on volume. A
\* mismatch corrupts or errors rather than hanging, so ordering agreement says
\* nothing about it.
ScoutBCollectivePayloadAgreement ==
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
ScoutBCollectivePayloadSizeRelation ==
  CollectivePayloadSizeRelationHolds(
    CollectiveWorkIds,
    CollectiveOperation,
    CollectiveMembers,
    CollectivePayloadSizeRelation,
    CollectivePayloadInputElements,
    CollectivePayloadOutputElements
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
