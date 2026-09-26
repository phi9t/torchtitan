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

(***************************************************************************)
(* Structural DTensor placement facts.                                     *)
(*                                                                         *)
(* PlacementValid above is deliberately unchanged. It says only that some   *)
(* parameter is DP-sharded and some parameter is TP-sharded somewhere,      *)
(* which almost any non-degenerate 2x2 run satisfies. The predicates below  *)
(* read the per-parameter facts and check contracts whose violation is      *)
(* silent, with PlacementBooleansAgreeWithFacts tying the two booleans to    *)
(* those facts rather than leaving them a second, independent claim.         *)
(*                                                                         *)
(* WHERE THE FACTS COME FROM: one snapshot per rank of                      *)
(* model_parts[0].named_parameters(), taken after the model is built and    *)
(* before Trainer.train. It is the parameter set the observed AdamW step    *)
(* updates, and OptimizerBoundaryObserved pins that that step is in the     *)
(* trace, but the snapshot is NOT re-taken inside the optimizer pre-hook    *)
(* and NO gradient placement is observed. A Partial gradient would          *)
(* therefore not be visible here; what is checkable is that no parameter    *)
(* the optimizer steps carries an unreduced placement.                      *)
(*                                                                         *)
(* INDEXING: shard dims are 0-based, as PyTorch reports them. TLA sequences *)
(* are 1-based, so a shape lookup is shape[dim + 1].                        *)
(***************************************************************************)

PlacementKinds == {"shard", "replicate", "partial"}

\* The placements one parameter claims.
PlacementsOfParameter(placementIds, placementParameter, name) ==
  { placement \in SequenceElements(placementIds) :
      placementParameter[placement] = name }

\* Domains, kinds and the one-placement-per-mesh-axis shape. The predicates
\* below apply PlacementShardDim, and this is what pins its domain to exactly
\* the shard placements, which is what keeps those applications total.
ParameterPlacementWellFormed(
    parameterNames, parameterMeshAxes, parameterGlobalShape,
    parameterLocalShape, placementIds, placementParameter,
    placementAxis, placementKind, placementStrided, placementShardDim,
    meshAxisDegree) ==
  LET nameSet  == SequenceElements(parameterNames)
      idSet    == SequenceElements(placementIds)
      shardIds == { placement \in idSet : placementKind[placement] = "shard" }
  IN /\ nameSet # {}
     /\ Len(parameterNames) = Cardinality(nameSet)
     /\ Len(placementIds) = Cardinality(idSet)
     /\ DOMAIN parameterMeshAxes = nameSet
     /\ DOMAIN parameterGlobalShape = nameSet
     /\ DOMAIN parameterLocalShape = nameSet
     /\ DOMAIN placementParameter = idSet
     /\ DOMAIN placementAxis = idSet
     /\ DOMAIN placementKind = idSet
     /\ DOMAIN placementStrided = idSet
     /\ DOMAIN placementShardDim = shardIds
     /\ \A placement \in idSet :
          /\ placementParameter[placement] \in nameSet
          /\ placementKind[placement] \in PlacementKinds
          /\ placementAxis[placement] \in DOMAIN meshAxisDegree
     /\ \A name \in nameSet :
          LET own ==
                PlacementsOfParameter(
                  placementIds, placementParameter, name)
              axes == parameterMeshAxes[name]
          IN /\ Len(axes) > 0
             /\ Len(parameterGlobalShape[name]) > 0
             /\ Len(parameterGlobalShape[name]) = Len(parameterLocalShape[name])
             \* One placement per mesh axis: equal cardinality forces the axes
             \* to be distinct as well as complete.
             /\ Cardinality(own) = Len(axes)
             /\ { placementAxis[placement] : placement \in own }
                  = SequenceElements(axes)
     /\ \A placement \in shardIds :
          LET shape ==
                parameterGlobalShape[placementParameter[placement]]
          IN placementShardDim[placement] \in 0..(Len(shape) - 1)

\* THE property worth having. A Partial placement is an unreduced value: a
\* parameter the optimizer steps that still carries one means the reduction
\* never happened, and the resulting math is wrong without any error.
NoPartialParameterPlacement(placementIds, placementKind) ==
  /\ DOMAIN placementKind = SequenceElements(placementIds)
  /\ \A placement \in SequenceElements(placementIds) :
       placementKind[placement] # "partial"

\* Anchors the placement claim to an optimizer step that is actually in the
\* trace, so "no Partial at the optimizer" is not a statement about a step
\* nobody observed.
OptimizerBoundaryObserved(
    rankSet, eventIds, eventRank, eventKind, eventOrder) ==
  LET eventSet == SequenceElements(eventIds)
      ofKind(rank, kind) ==
        { event \in eventSet :
            eventRank[event] = rank /\ eventKind[event] = kind }
  IN \A rank \in rankSet :
       /\ Cardinality(ofKind(rank, "optimizer.started")) = 1
       /\ Cardinality(ofKind(rank, "optimizer.mutated")) = 1
       /\ \A started \in ofKind(rank, "optimizer.started") :
            \A mutated \in ofKind(rank, "optimizer.mutated") :
              eventOrder[started] < eventOrder[mutated]

\* A sharded tensor dim must divide evenly by the degree of the mesh axis that
\* shards it. The degrees come from MeshAxisDegree, which
\* MeshAxisDegreeAgreesWithCoordinates ties back to the mesh facts.
ShardedDimDividesAxisDegree(
    placementIds, placementParameter, placementAxis, placementKind,
    placementShardDim, parameterGlobalShape, meshAxisDegree) ==
  \A placement \in { candidate \in SequenceElements(placementIds) :
                       placementKind[candidate] = "shard" } :
    LET shape  == parameterGlobalShape[placementParameter[placement]]
        degree == meshAxisDegree[placementAxis[placement]]
    IN /\ degree > 0
       /\ shape[placementShardDim[placement] + 1] % degree = 0

\* The observed local shape must agree with which dims are sharded. Stated
\* without a product over the sharding axes -- TLA+ has no product operator and
\* a RECURSIVE fold would not earn its cost here -- so this checks the two
\* directions that need no arithmetic over several axes at once.
LocalShapeReflectsSharding(
    parameterNames, parameterGlobalShape, parameterLocalShape,
    placementIds, placementParameter, placementAxis, placementKind,
    placementShardDim, meshAxisDegree) ==
  \A name \in SequenceElements(parameterNames) :
    \A index \in DOMAIN parameterGlobalShape[name] :
      LET global == parameterGlobalShape[name][index]
          local  == parameterLocalShape[name][index]
          sharders ==
            { placement \in PlacementsOfParameter(
                              placementIds, placementParameter, name) :
                /\ placementKind[placement] = "shard"
                /\ placementShardDim[placement] = index - 1 }
      IN /\ local > 0
         /\ local <= global
         /\ global % local = 0
         /\ (sharders = {} => local = global)
         /\ ( (\E placement \in sharders :
                 meshAxisDegree[placementAxis[placement]] > 1)
                => local < global )

\* _StridedShard exists for exactly one situation: an outer mesh axis shards a
\* tensor dim that an inner axis also shards, so the outer axis must stride over
\* the inner axis's shards instead of taking a contiguous slice. "Outer" is the
\* earlier position in the parameter's own mesh-axis order. A strided flag
\* without that inner partner, or a missing flag where the partner exists, is a
\* layout the optimizer and the checkpoint would disagree about.
StridedShardIsAnOuterComposedShard(
    parameterMeshAxes, placementIds, placementParameter, placementAxis,
    placementKind, placementStrided, placementShardDim) ==
  LET idSet    == SequenceElements(placementIds)
      shardIds == { placement \in idSet : placementKind[placement] = "shard" }
  IN /\ \A placement \in idSet \ shardIds : ~placementStrided[placement]
     /\ \A placement \in shardIds :
          LET name == placementParameter[placement]
              axes == parameterMeshAxes[name]
              axisPositions(axis) ==
                { index \in DOMAIN axes : axes[index] = axis }
              outer == axisPositions(placementAxis[placement])
          IN placementStrided[placement]
               <=> \E inner \in shardIds :
                     /\ placementParameter[inner] = name
                     /\ placementShardDim[inner]
                          = placementShardDim[placement]
                     /\ \E outerIndex \in outer :
                          \E innerIndex \in
                            axisPositions(placementAxis[inner]) :
                            innerIndex > outerIndex

\* The degrees the placement facts use must be the degrees the mesh facts
\* already exported imply: one distinct coordinate value per axis position.
MeshAxisDegreeAgreesWithCoordinates(rankSet, rankCoordinates, meshAxisDegree) ==
  /\ DOMAIN meshAxisDegree = {"dp_shard", "tp"}
  /\ meshAxisDegree["dp_shard"]
       = Cardinality({ rankCoordinates[rank][1] : rank \in rankSet })
  /\ meshAxisDegree["tp"]
       = Cardinality({ rankCoordinates[rank][2] : rank \in rankSet })
  /\ \A rank \in rankSet :
       /\ rankCoordinates[rank][1] \in 0..(meshAxisDegree["dp_shard"] - 1)
       /\ rankCoordinates[rank][2] \in 0..(meshAxisDegree["tp"] - 1)

\* The placement facts are exported once because every rank agreed on the whole
\* list. This makes the checker assert that agreement instead of trusting the
\* exporter's word for it.
PlacementSchemaAgreesAcrossRanks(rankSet, schemaDigestByRank, schemaDigest) ==
  /\ DOMAIN schemaDigestByRank = rankSet
  /\ schemaDigest # ""
  /\ \A rank \in rankSet : schemaDigestByRank[rank] = schemaDigest

\* Ties the two booleans PlacementValid reads to the structural facts, so
\* PlacementValid keeps its original meaning and cannot drift away from the
\* per-parameter evidence.
\*
\* WHAT THIS IS NOT: it is not a derivation of each rank's own boolean.
\* HasShardOnAxis takes no rank argument, because the placements are exported
\* once for all four ranks (PlacementSchemaAgreesAcrossRanks is what licenses
\* that), so for ranks 1-3 this says the rank's boolean agrees with the one
\* exported list rather than with a list of its own. And it is one bit per axis:
\* a projection that dropped all but one shard placement per axis would still
\* satisfy it. The per-parameter content is checked by
\* ParameterPlacementWellFormed, ShardedDimDividesAxisDegree and
\* LocalShapeReflectsSharding; this only stops the booleans and the structural
\* facts from telling different stories. Hence "AgreeWith", not "DerivableFrom".
HasShardOnAxis(placementIds, placementAxis, placementKind, axis) ==
  \E placement \in SequenceElements(placementIds) :
    /\ placementAxis[placement] = axis
    /\ placementKind[placement] = "shard"

PlacementBooleansAgreeWithFacts(
    rankSet, hasDpShard, hasTpShard, placementIds, placementAxis,
    placementKind) ==
  /\ DOMAIN hasDpShard = rankSet
  /\ DOMAIN hasTpShard = rankSet
  /\ \A rank \in rankSet :
       /\ hasDpShard[rank]
            = HasShardOnAxis(placementIds, placementAxis, placementKind,
                             "dp_shard")
       /\ hasTpShard[rank]
            = HasShardOnAxis(placementIds, placementAxis, placementKind, "tp")

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
