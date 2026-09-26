/-
EVALUATION OF THE OBSERVED TRACE, not a theorem about the protocol.

Every predicate here is `Bool`-valued, and ScoutBValid.lean applies it to
literal ScoutBFacts data and closes it with `rfl` or `decide`. What the kernel
checks is that THIS run satisfies the predicate: genuine, axiom-free, and
silent about any other trace. The result tokens say so, carrying
`kind=evaluation scope=observed-trace`.

For theorems quantified over topologies, states and schedule lengths -- which
mention no trace and no bound -- see ScoutBProtocol.lean,
ScoutBInductiveInvariant.lean and ScoutBWaitGraph.lean, whose tokens carry
`kind=theorem scope=all-topologies-all-schedules bound=none`.
-/

namespace Qwen3Formal

structure RankCoordinate where
  rank : Nat
  dp : Nat
  tp : Nat
  inputDigest : String
  hasDpShard : Bool
  hasTpShard : Bool
deriving DecidableEq, Repr

inductive CollectiveState where
  | enqueued
  | started
  | completed
deriving BEq, DecidableEq, Repr

structure EventReference where
  eventId : String
  identity : Nat
  index : Nat
  rank : Nat
  order : Nat
  kind : String
deriving DecidableEq, Repr

structure CollectiveReference where
  workId : String
  index : Nat
  rank : Nat
  rankOrder : Nat
  collectiveId : String
deriving DecidableEq, Repr

structure CollectiveObservation where
  workId : String
  index : Nat
  rank : Nat
  rankOrder : Nat
  collectiveId : String
  members : List Nat
  lifecycle : List CollectiveState
  producers : List String
  eventReferences : List EventReference
  peerReferences : List CollectiveReference
  hasExecutor : Bool
  hasStream : Bool
  observationCount : Nat
deriving DecidableEq, Repr

structure EventEvidence where
  eventId : String
  identity : Nat
  index : Nat
  rank : Nat
  order : Nat
  kind : String
  rawEventId : String
  projectionSchema : String
  projectionDigest : String
  predecessors : List EventReference
deriving DecidableEq, Repr

structure CrossRankSynchronization where
  edgeId : String
  endpoints : List EventReference
  relation : String
  ordered : Bool
  evidenceDigest : String
deriving DecidableEq, Repr

structure RankStringRows where
  rank0 : List String
  rank1 : List String
  rank2 : List String
  rank3 : List String
deriving DecidableEq, Repr

structure RankEventRows where
  rank0 : List EventEvidence
  rank1 : List EventEvidence
  rank2 : List EventEvidence
  rank3 : List EventEvidence
deriving DecidableEq, Repr

structure RankCollectiveRows where
  rank0 : List CollectiveObservation
  rank1 : List CollectiveObservation
  rank2 : List CollectiveObservation
  rank3 : List CollectiveObservation
deriving DecidableEq, Repr

structure SynchronizationChunks where
  chunk0 : List CrossRankSynchronization
  chunk1 : List CrossRankSynchronization
  chunk2 : List CrossRankSynchronization
  chunk3 : List CrossRankSynchronization
deriving DecidableEq, Repr

def expectedCoordinates : List (Nat × Nat × Nat) :=
  [(0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)]

def allowedGroups : List (List Nat) :=
  [[0, 1], [2, 3], [0, 2], [1, 3], [0, 1, 2, 3]]

def meshWellFormed (ranks : List RankCoordinate) : Bool :=
  ranks.map (fun item => (item.rank, item.dp, item.tp)) == expectedCoordinates

def tpInputAgreement (ranks : List RankCoordinate) : Bool :=
  ranks.all fun left =>
    ranks.all fun right =>
      if left.dp == right.dp then left.inputDigest == right.inputDigest else true

def placementValid (ranks : List RankCoordinate) : Bool :=
  ranks.all fun item => item.hasDpShard && item.hasTpShard

/-
Structural DTensor placement facts.

`placementValid` above is deliberately unchanged: it says only that some
parameter is DP-sharded and some parameter is TP-sharded somewhere, which
almost any non-degenerate 2x2 run satisfies. The definitions below read the
per-parameter facts, and `placementBooleansAgreeWithFacts` ties the two booleans
are the aggregate of those facts rather than a second, independent claim.

WHERE THE FACTS COME FROM: one snapshot per rank of
`model_parts[0].named_parameters()`, taken after the model is built and before
`Trainer.train`. It is the parameter set the observed AdamW step updates, and
`optimizerBoundaryObserved` pins that that step is in the trace, but the
snapshot is NOT re-taken inside the optimizer pre-hook and NO gradient
placement is observed. A Partial gradient would therefore not be visible here;
what is checkable is that no parameter the optimizer steps carries an
unreduced placement.

Grouped by parameter, while the TLA export keys flat placement ids. The two
checkers pay different costs: grouping keeps this kernel evaluation linear
instead of filtering every placement once per parameter and tensor dim.
-/

structure PlacementRecord where
  axis : String
  kind : String
  strided : Bool
  /-- 0-based tensor dim, present exactly when `kind = "shard"`. -/
  shardDim : Option Nat
deriving DecidableEq, Repr

structure ParameterPlacements where
  parameter : String
  meshAxes : List String
  globalShape : List Nat
  localShape : List Nat
  placements : List PlacementRecord
deriving DecidableEq, Repr

structure MeshAxisDegree where
  axis : String
  degree : Nat
deriving DecidableEq, Repr

def placementKinds : List String := ["shard", "replicate", "partial"]

def axisDegree? (degrees : List MeshAxisDegree) (axis : String) : Option Nat :=
  (degrees.find? fun item => item.axis == axis).map (fun item => item.degree)

/-- Domains, kinds and the one-placement-per-mesh-axis shape. -/
def parameterPlacementWellFormed
    (parameters : List ParameterPlacements)
    (degrees : List MeshAxisDegree) : Bool :=
  !parameters.isEmpty &&
  (parameters.map (fun item => item.parameter)).eraseDups.length
    == parameters.length &&
  parameters.all fun item =>
    !item.meshAxes.isEmpty &&
    !item.globalShape.isEmpty &&
    item.globalShape.length == item.localShape.length &&
    item.placements.length == item.meshAxes.length &&
    item.placements.map (fun p => p.axis) == item.meshAxes &&
    item.meshAxes.eraseDups.length == item.meshAxes.length &&
    item.placements.all fun p =>
      placementKinds.contains p.kind &&
      (axisDegree? degrees p.axis).isSome &&
      (match p.shardDim with
        | none => p.kind != "shard"
        | some dim => p.kind == "shard" && dim < item.globalShape.length)

/--
THE property worth having. A Partial placement is an unreduced value: a
parameter the optimizer steps that still carries one means the reduction never
happened, and the resulting math is wrong without any error.
-/
def noPartialParameterPlacement
    (parameters : List ParameterPlacements) : Bool :=
  parameters.all fun item => item.placements.all fun p => p.kind != "partial"

/--
Anchors the placement claim to an optimizer step that is actually in the trace,
so "no Partial at the optimizer" is not a statement about a step nobody
observed.
-/
def optimizerBoundaryRankValid (events : List EventEvidence) : Bool :=
  let started := events.filter fun event => event.kind == "optimizer.started"
  let mutated := events.filter fun event => event.kind == "optimizer.mutated"
  -- Spelled without a multi-discriminant match on purpose: the equation
  -- compiler's splitter for one drags `propext` into every theorem that
  -- evaluates it, and the checker contract refuses any axiom dependency.
  started.length == 1 && mutated.length == 1 &&
  started.all fun start => mutated.all fun mutate => start.order < mutate.order

def optimizerBoundaryObserved (eventsByRank : RankEventRows) : Bool :=
  optimizerBoundaryRankValid eventsByRank.rank0 &&
  optimizerBoundaryRankValid eventsByRank.rank1 &&
  optimizerBoundaryRankValid eventsByRank.rank2 &&
  optimizerBoundaryRankValid eventsByRank.rank3

/-- A sharded tensor dim must divide evenly by the degree of its mesh axis. -/
def shardedDimDividesAxisDegree
    (parameters : List ParameterPlacements)
    (degrees : List MeshAxisDegree) : Bool :=
  parameters.all fun item =>
    item.placements.all fun p =>
      match p.shardDim with
      | none => true
      | some dim =>
          -- `.drop`/`.head?` rather than `globalShape[dim]?`: the `getElem?`
          -- path pulls `propext` into the evaluation, which the checker
          -- contract refuses. The `getD` fallbacks are unreachable behind the
          -- `isSome` guards; 1 is chosen so an unguarded reading would still
          -- be a divisor rather than a division by zero.
          (axisDegree? degrees p.axis).isSome &&
          ((item.globalShape.drop dim).head?).isSome &&
          (axisDegree? degrees p.axis).getD 1 > 0 &&
          ((item.globalShape.drop dim).head?).getD 0
            % (axisDegree? degrees p.axis).getD 1 == 0

/--
The observed local shape must agree with which dims are sharded. Stated without
a product over the sharding axes, so it checks the directions that need no
arithmetic over several axes at once.
-/
def localShapeDimsValid
    (item : ParameterPlacements) (degrees : List MeshAxisDegree) (dim : Nat) :
    List Nat -> List Nat -> Bool
  | [] => fun localSizes => localSizes.isEmpty
  | globalSize :: globalSizes => fun localSizes =>
      match localSizes with
      | [] => false
      | localSize :: rest =>
          let sharders := item.placements.filter fun p =>
            p.kind == "shard" && p.shardDim == some dim
          localSize > 0 && localSize <= globalSize &&
          globalSize % localSize == 0 &&
          (if sharders.isEmpty then localSize == globalSize else true) &&
          (if sharders.any (fun p => (axisDegree? degrees p.axis).getD 1 > 1)
            then localSize < globalSize else true) &&
          localShapeDimsValid item degrees (dim + 1) globalSizes rest

def localShapeReflectsSharding
    (parameters : List ParameterPlacements)
    (degrees : List MeshAxisDegree) : Bool :=
  parameters.all fun item =>
    localShapeDimsValid item degrees 0 item.globalShape item.localShape

/--
`_StridedShard` exists for exactly one situation: an outer mesh axis shards a
tensor dim that an inner axis also shards, so the outer axis must stride over
the inner axis's shards instead of taking a contiguous slice. "Outer" is the
earlier position in the parameter's own mesh-axis order. A strided flag without
that inner partner, or a missing flag where the partner exists, is a layout the
optimizer and the checkpoint would disagree about.
-/
def stridedShardIsAnOuterComposedShard
    (parameters : List ParameterPlacements) : Bool :=
  parameters.all fun item =>
    let indexed := item.placements.zipIdx
    indexed.all fun (p, position) =>
      match p.shardDim with
      | none => !p.strided
      | some dim =>
          p.strided
            == indexed.any fun (inner, innerPosition) =>
                 innerPosition > position &&
                 inner.kind == "shard" &&
                 inner.shardDim == some dim

/--
The degrees the placement facts use must be the degrees the mesh facts already
exported imply: one distinct coordinate value per axis position.
-/
def meshAxisDegreeAgreesWithCoordinates
    (ranks : List RankCoordinate) (degrees : List MeshAxisDegree) : Bool :=
  degrees.map (fun item => item.axis) == ["dp_shard", "tp"] &&
  (axisDegree? degrees "dp_shard").isSome &&
  (axisDegree? degrees "tp").isSome &&
  (ranks.map (fun item => item.dp)).eraseDups.length
    == (axisDegree? degrees "dp_shard").getD 0 &&
  (ranks.map (fun item => item.tp)).eraseDups.length
    == (axisDegree? degrees "tp").getD 0 &&
  ranks.all fun item =>
    item.dp < (axisDegree? degrees "dp_shard").getD 0 &&
    item.tp < (axisDegree? degrees "tp").getD 0

/--
The placement facts are exported once because every rank agreed on the whole
list. This asserts that agreement instead of trusting the exporter for it.
-/
def placementSchemaAgreesAcrossRanks
    (schemaDigest : String) (schemaDigestByRank : List String) : Bool :=
  schemaDigest != "" &&
  schemaDigestByRank.length == 4 &&
  schemaDigestByRank.all fun digest => digest == schemaDigest

/--
Ties the two booleans `placementValid` reads to the structural facts, so
`placementValid` keeps its original meaning and cannot drift away from the
per-parameter evidence.

WHAT THIS IS NOT: not a derivation of each rank's own boolean. `hasShardOnAxis`
takes no rank, because the placements are exported once for all four ranks, so
for ranks 1-3 this says the rank's boolean agrees with the one exported list
rather than with a list of its own. And it is one bit per axis: a projection
that dropped all but one shard placement per axis would still satisfy it. The
per-parameter content is checked by `parameterPlacementWellFormed`,
`shardedDimDividesAxisDegree` and `localShapeReflectsSharding`. Hence
"AgreeWith", not "DerivableFrom".
-/
def hasShardOnAxis
    (parameters : List ParameterPlacements) (axis : String) : Bool :=
  parameters.any fun item =>
    item.placements.any fun p => p.axis == axis && p.kind == "shard"

def placementBooleansAgreeWithFacts
    (ranks : List RankCoordinate)
    (parameters : List ParameterPlacements) : Bool :=
  ranks.all fun item =>
    item.hasDpShard == hasShardOnAxis parameters "dp_shard" &&
    item.hasTpShard == hasShardOnAxis parameters "tp"

def boundedIdentityValid
    (perRank index rank offset : Nat) : Bool :=
  perRank > 0 &&
  rank < 4 &&
  offset < perRank &&
  index == rank * perRank + offset

def eventKindCode? : String -> Option Nat
  | "step.started" => some 0
  | "batch.observed" => some 1
  | "forward.started" => some 2
  | "forward.completed" => some 3
  | "backward.started" => some 4
  | "gradient.ready" => some 5
  | "backward.completed" => some 6
  | "optimizer.started" => some 7
  | "optimizer.mutated" => some 8
  | "step.completed" => some 9
  | "bundle.collected" => some 10
  | "collective.enqueued" => some 11
  | "collective.started" => some 12
  | "collective.completed" => some 13
  | _ => none

-- The core trainer events are observed directly by hooks during the step, so
-- their per-rank order is faithful evidence and is asserted exactly. The final
-- bundle marker is likewise appended last by construction.
def scoutBCoreEventKind? (eventsPerRank order : Nat) : Option String :=
  if order == 1 then some "step.started"
  else if order == 2 then some "batch.observed"
  else if order == 3 then some "forward.started"
  else if order == 4 then some "forward.completed"
  else if order == 5 then some "backward.started"
  else if order == 6 then some "gradient.ready"
  else if order == 7 then some "backward.completed"
  else if order == 8 then some "optimizer.started"
  else if order == 9 then some "optimizer.mutated"
  else if order == 10 then some "step.completed"
  else if order == eventsPerRank then some "bundle.collected"
  else none

-- Collective lifecycle events are NOT ordered evidence. The observer appends
-- each work's enqueued/started/completed triple contiguously, post hoc, from
-- the NCCL Flight Recorder, and the projection drops observed_time_ns. The
-- resulting per-rank order therefore asserts a full serialization of every
-- collective before the next, which certainly did not happen -- FSDP
-- all-gathers overlap by construction.
--
-- An earlier version of this file asserted that serialization as a theorem, via
-- `(order - 11) % 3` naming the exact lifecycle phase at each order. That made
-- the artifact a checker of a fiction. What remains assertable is only that
-- these positions carry SOME collective lifecycle kind; which one, and in what
-- relative order across works, is not observed here. Issue order is recoverable
-- separately, from the enqueued events' flight_record_id.
def isCollectiveLifecycleKind (kind : String) : Bool :=
  kind == "collective.enqueued" ||
  kind == "collective.started" ||
  kind == "collective.completed"

def scoutBEventKindAdmissible (eventsPerRank order : Nat) (kind : String) : Bool :=
  match scoutBCoreEventKind? eventsPerRank order with
  | some expected => expected == kind
  | none => 10 < order && order < eventsPerRank && isCollectiveLifecycleKind kind

def eventIdentityValid
    (eventsPerRank identity index rank order : Nat)
    (kind : String) : Bool :=
  match eventKindCode? kind with
  | some kindCode =>
      order > 0 &&
      boundedIdentityValid eventsPerRank index rank (order - 1) &&
      scoutBEventKindAdmissible eventsPerRank order kind &&
      identity == (rank * eventsPerRank + order - 1) * 16 + kindCode
  | none => false

def eventReferenceValid
    (eventIdPrefix : String)
    (eventsPerRank : Nat)
    (reference : EventReference) : Bool :=
  eventIdentityValid
    eventsPerRank reference.identity reference.index reference.rank
    reference.order reference.kind &&
  !eventIdPrefix.isEmpty &&
  !reference.eventId.isEmpty

def eventEvidenceRankValid
    (projectionSchema : String)
    (eventIdPrefix : String)
    (eventsPerRank : Nat)
    (expectedRank : Nat)
    (expectedEventIds : List String)
    (events : List EventEvidence) : Bool :=
  projectionSchema == "qwen3.formal.raw-event-projection.v0" &&
  !eventIdPrefix.isEmpty &&
  expectedRank < 4 &&
  eventsPerRank > 11 &&
  (eventsPerRank - 11) % 3 == 0 &&
  expectedEventIds.length == eventsPerRank &&
  events.length == eventsPerRank &&
  (events.zip expectedEventIds).zipIdx.all fun ((event, expectedEventId), rankOrder) =>
    event.rank == expectedRank &&
    event.order == rankOrder + 1 &&
    event.index == expectedRank * eventsPerRank + rankOrder &&
    eventIdentityValid
      eventsPerRank event.identity event.index event.rank event.order event.kind &&
    event.eventId == expectedEventId &&
    !event.kind.isEmpty &&
    !event.rawEventId.isEmpty &&
    event.projectionSchema == projectionSchema &&
    !event.projectionDigest.isEmpty &&
    event.predecessors.all fun predecessor =>
      eventReferenceValid
        eventIdPrefix eventsPerRank predecessor &&
      predecessor.rank == event.rank &&
      predecessor.order < event.order

def eventEvidenceValid
    (projectionSchema : String)
    (eventIdPrefix : String)
    (eventsPerRank : Nat)
    (eventIdsByRank : RankStringRows)
    (eventsByRank : RankEventRows) : Bool :=
  eventEvidenceRankValid
    projectionSchema eventIdPrefix eventsPerRank 0
    eventIdsByRank.rank0 eventsByRank.rank0 &&
  eventEvidenceRankValid
    projectionSchema eventIdPrefix eventsPerRank 1
    eventIdsByRank.rank1 eventsByRank.rank1 &&
  eventEvidenceRankValid
    projectionSchema eventIdPrefix eventsPerRank 2
    eventIdsByRank.rank2 eventsByRank.rank2 &&
  eventEvidenceRankValid
    projectionSchema eventIdPrefix eventsPerRank 3
    eventIdsByRank.rank3 eventsByRank.rank3

def crossRankSynchronizationChunkValid
    (eventIdPrefix : String)
    (eventsPerRank : Nat)
    (edges : List CrossRankSynchronization) : Bool :=
  edges.all fun edge =>
    edge.endpoints.length == 2 &&
    edge.endpoints.zipIdx.all fun (left, leftIndex) =>
      eventReferenceValid eventIdPrefix eventsPerRank left &&
      edge.endpoints.zipIdx.all fun (right, rightIndex) =>
        if leftIndex == rightIndex then left.identity == right.identity
        else
          !(left.identity == right.identity) &&
          !(left.rank == right.rank) &&
          edge.relation == "synchronizes" &&
          !edge.ordered &&
          !edge.evidenceDigest.isEmpty

def crossRankSynchronizationValid
    (eventIdPrefix : String)
    (eventsPerRank : Nat)
    (edgeChunks : SynchronizationChunks) : Bool :=
  crossRankSynchronizationChunkValid
    eventIdPrefix eventsPerRank edgeChunks.chunk0 &&
  crossRankSynchronizationChunkValid
    eventIdPrefix eventsPerRank edgeChunks.chunk1 &&
  crossRankSynchronizationChunkValid
    eventIdPrefix eventsPerRank edgeChunks.chunk2 &&
  crossRankSynchronizationChunkValid
    eventIdPrefix eventsPerRank edgeChunks.chunk3

def expectedLifecycle : List CollectiveState :=
  [.enqueued, .started, .completed]

def collectiveLifecycleValid
    (collectives : List CollectiveObservation) : Bool :=
  collectives.all fun item =>
    allowedGroups.contains item.members &&
    item.members.contains item.rank &&
    item.lifecycle == expectedLifecycle &&
    item.hasExecutor &&
    item.hasStream &&
    item.observationCount > 0

def collectiveReferenceValid
    (collectivesPerRank : Nat)
    (reference : CollectiveReference) : Bool :=
  boundedIdentityValid
    collectivesPerRank reference.index reference.rank reference.rankOrder &&
  !reference.workId.isEmpty &&
  !reference.collectiveId.isEmpty

def collectivePeerValid
    (collectivesPerRank : Nat)
    (item : CollectiveObservation)
    (member : Nat)
    (reference : CollectiveReference) : Bool :=
  collectiveReferenceValid collectivesPerRank reference &&
  reference.rank == member &&
  reference.collectiveId == item.collectiveId

def collectiveCoverageRankValid
    (expectedRank : Nat)
    (expectedWorkIds expectedCollectiveIds : List String)
    (collectivesPerRank : Nat)
    (collectives : List CollectiveObservation) : Bool :=
  expectedRank < 4 &&
  expectedWorkIds.length == collectivesPerRank &&
  expectedCollectiveIds.length == collectivesPerRank &&
  collectives.length == collectivesPerRank &&
  ((collectives.zip expectedWorkIds).zip expectedCollectiveIds).zipIdx.all
    fun (((item, expectedWorkId), expectedCollectiveId), rankOrder) =>
    item.rank == expectedRank &&
    item.rankOrder == rankOrder &&
    item.index == expectedRank * collectivesPerRank + rankOrder &&
    boundedIdentityValid
      collectivesPerRank item.index item.rank item.rankOrder &&
    item.workId == expectedWorkId &&
    item.collectiveId == expectedCollectiveId &&
    item.peerReferences.length == item.members.length &&
    (item.members.zip item.peerReferences).all fun (member, reference) =>
      collectivePeerValid collectivesPerRank item member reference

def collectiveCoverage
    (workIdsByRank collectiveIdsByRank : RankStringRows)
    (collectivesPerRank : Nat)
    (collectivesByRank : RankCollectiveRows) : Bool :=
  collectiveCoverageRankValid
    0 workIdsByRank.rank0 collectiveIdsByRank.rank0
    collectivesPerRank collectivesByRank.rank0 &&
  collectiveCoverageRankValid
    1 workIdsByRank.rank1 collectiveIdsByRank.rank1
    collectivesPerRank collectivesByRank.rank1 &&
  collectiveCoverageRankValid
    2 workIdsByRank.rank2 collectiveIdsByRank.rank2
    collectivesPerRank collectivesByRank.rank2 &&
  collectiveCoverageRankValid
    3 workIdsByRank.rank3 collectiveIdsByRank.rank3
    collectivesPerRank collectivesByRank.rank3

def oneProducer : List String -> Bool
  | [] => false
  | first :: rest =>
      rest.length == 2 && rest.all (fun current => current == first)

def collectiveProducerCorrelation
    (collectives : List CollectiveObservation) : Bool :=
  collectives.all fun item => oneProducer item.producers

def eventReferenceHasKindAndRank
    (eventIdPrefix : String)
    (eventsPerRank : Nat)
    (reference : EventReference)
    (rank : Nat)
    (kind : String) : Bool :=
  eventReferenceValid
    eventIdPrefix eventsPerRank reference &&
  reference.rank == rank &&
  reference.kind == kind

def expectedCollectiveEventKinds : List String :=
  ["collective.enqueued", "collective.started", "collective.completed"]

def collectiveEventEvidenceRankValid
    (eventIdPrefix : String)
    (eventsPerRank : Nat)
    (expectedRank : Nat)
    (collectives : List CollectiveObservation) : Bool :=
  collectives.all fun item =>
    item.rank == expectedRank &&
    item.eventReferences.length == expectedCollectiveEventKinds.length &&
    (item.eventReferences.zip expectedCollectiveEventKinds).all
      fun (reference, expectedKind) =>
        eventReferenceHasKindAndRank
          eventIdPrefix eventsPerRank reference item.rank expectedKind

def collectiveEventEvidence
    (eventIdPrefix : String)
    (eventsPerRank : Nat)
    (collectivesByRank : RankCollectiveRows) : Bool :=
  collectiveEventEvidenceRankValid
    eventIdPrefix eventsPerRank 0 collectivesByRank.rank0 &&
  collectiveEventEvidenceRankValid
    eventIdPrefix eventsPerRank 1 collectivesByRank.rank1 &&
  collectiveEventEvidenceRankValid
    eventIdPrefix eventsPerRank 2 collectivesByRank.rank2 &&
  collectiveEventEvidenceRankValid
    eventIdPrefix eventsPerRank 3 collectivesByRank.rank3

def validDPxTP
    (ranks : List RankCoordinate)
    (collectives : List CollectiveObservation)
    (collectivesByRank : RankCollectiveRows)
    (eventsByRank : RankEventRows)
    (edgeChunks : SynchronizationChunks)
    (projectionSchema : String)
    (eventIdPrefix : String)
    (eventsPerRank collectivesPerRank : Nat)
    (eventIdsByRank : RankStringRows)
    (collectiveWorkIdsByRank collectiveIdsByRank : RankStringRows) : Bool :=
  meshWellFormed ranks &&
  tpInputAgreement ranks &&
  placementValid ranks &&
  eventEvidenceValid
    projectionSchema eventIdPrefix eventsPerRank eventIdsByRank eventsByRank &&
  crossRankSynchronizationValid
    eventIdPrefix eventsPerRank edgeChunks &&
  collectiveLifecycleValid collectives &&
  collectiveCoverage
    collectiveWorkIdsByRank collectiveIdsByRank collectivesPerRank
    collectivesByRank &&
  collectiveProducerCorrelation collectives &&
  collectiveEventEvidence
    eventIdPrefix eventsPerRank collectivesByRank

end Qwen3Formal
