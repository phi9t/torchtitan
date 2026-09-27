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

/-
Collective payload identity.

Matching operation order on a communicator is necessary but does not make a
collective well formed: its members must also agree on dtype and on volume, and
a mismatch there errors or corrupts rather than deadlocking, so none of the
ordering predicates above can express it.

WHERE THE FACTS COME FROM: `input_sizes`, `output_sizes`, `input_dtypes` and
`output_dtypes` off each rank's own NCCL Flight Recorder entry. OBSERVED, unlike
the producer and stream fields, which are attributed by a positional zip. Also
PLURAL and per tensor -- one shape and one dtype name per tensor, the two lists
running in parallel -- so a single-dtype model would be wrong for a collective
over several tensors.

Kept in its own list rather than added to `CollectiveObservation`: these
predicates read nothing else, so a self-contained list keeps their evaluation
independent of the lifecycle checks. `payloadCoversCollectives` is the bridge
that stops the two lists describing different works.

An EMPTY shape is a 0-dim tensor: no dimensions, one element. So the product of
no sizes is deliberately 1 and not 0.
-/

structure CollectivePayload where
  workId : String
  rank : Nat
  collectiveId : String
  /-- Dense index of the collective this work belongs to, into
  `payloadCollectiveRows`. A join key the kernel can compare in one Nat step;
  see the cost note on `collectivePayloadAgreement`. -/
  collectiveIndex : Nat
  /-- The GLOBAL communicator key, `process_group.canonical_id`. Never the
  per-rank `runtime_pg_id`. -/
  comm : String
  operation : String
  sizeRelation : String
  members : List Nat
  /-- One shape per tensor; an empty shape is a 0-dim tensor. -/
  inputSizes : List (List Nat)
  outputSizes : List (List Nat)
  /-- Parallel to the size lists, one dtype name per tensor. -/
  inputDtypes : List String
  outputDtypes : List String
  /-- DERIVED by the exporter and recomputed here, never trusted. -/
  inputElements : Nat
  outputElements : Nat
deriving DecidableEq, Repr

/--
One row per collective: the values its members are required to agree on, taken
from ONE member. The exporter does not check that the others match -- if it did,
a real mismatch would be refused at export and the checkers would never see it.
Verifying the agreement is `collectivePayloadAgreement`'s job.
-/
structure PayloadCollectiveRow where
  collectiveId : String
  comm : String
  operation : String
  sizeRelation : String
  members : List Nat
  inputDtypes : List String
  outputDtypes : List String
  inputElements : Nat
  outputElements : Nat
deriving DecidableEq, Repr

/-- One row per DISTINCT communicator key, with the member set it denotes. -/
structure PayloadCommMembership where
  comm : String
  members : List Nat
deriving DecidableEq, Repr

def payloadSizeRelations : List String :=
  [ "output_is_member_count_times_input",
    "input_is_member_count_times_output",
    "input_equals_output" ]

/-- The relation each operation family implies. An operation this does not know
about maps to a label outside `payloadSizeRelations`, so it fails rather than
passing vacuously. -/
def impliedSizeRelation (operation : String) : String :=
  if operation == "all_gather" then "output_is_member_count_times_input"
  else if operation == "reduce_scatter" then "input_is_member_count_times_output"
  else if operation == "all_reduce" then "input_equals_output"
  else if operation == "broadcast" then "input_equals_output"
  else "unsupported_operation"

def totalElements (shapes : List (List Nat)) : Nat :=
  (shapes.map fun shape => shape.foldl (fun acc size => acc * size) 1).foldl
    (fun acc elements => acc + elements) 0

/-- Domains, the per-tensor pairing of shapes with dtype names, and the element
counts recomputed from the shapes. -/
def collectivePayloadWellFormed (payloads : List CollectivePayload) : Bool :=
  !payloads.isEmpty &&
  payloads.all fun item =>
    !item.comm.isEmpty &&
    !item.workId.isEmpty &&
    !item.collectiveId.isEmpty &&
    !item.members.isEmpty &&
    !item.inputSizes.isEmpty &&
    !item.outputSizes.isEmpty &&
    item.inputSizes.length == item.inputDtypes.length &&
    item.outputSizes.length == item.outputDtypes.length &&
    item.inputDtypes.all (fun name => !name.isEmpty) &&
    item.outputDtypes.all (fun name => !name.isEmpty) &&
    item.inputSizes.all (fun shape => shape.all fun size => size > 0) &&
    item.outputSizes.all (fun shape => shape.all fun size => size > 0) &&
    item.inputElements == totalElements item.inputSizes &&
    item.outputElements == totalElements item.outputSizes

/-- The bridge between the payload list and the collective observations: same
works, same order, same collective identity and same member set. Without it the
payload predicates could hold over a list describing different works. -/
def payloadCoversCollectives
    (payloads : List CollectivePayload)
    (collectives : List CollectiveObservation) : Bool :=
  payloads.length == collectives.length &&
  (payloads.zip collectives).all fun (payload, item) =>
    payload.workId == item.workId &&
    payload.rank == item.rank &&
    payload.collectiveId == item.collectiveId &&
    payload.members == item.members

def payloadRowAt
    (rows : List PayloadCollectiveRow) (index : Nat) :
    Option PayloadCollectiveRow :=
  (rows.drop index).head?

/--
THE first property: the members of one collective agree on what they exchange. A
dtype or volume mismatch between members is a corrupt or failed collective, not a
hang, so per-communicator order agreement cannot see it. One dtype per
collective is included because NCCL requires it of both buffers and every
member, and because a per-tensor list makes it expressible at all.

SHAPE, AND WHY IT IS NOT ALL-PAIRS. The obvious form -- for every pair of works
with the same collective id, compare their payloads -- is quadratic in the work
count with a string comparison at each step. MEASURED at the observed scale of
432 works: a `rfl` over that form had not finished after four minutes, against a
120s per-module budget in the runner. So each work is compared against ONE row
for its collective, reached by a Nat index, and a per-row count pins that the
row really covers every member.

WHAT THAT LEAVES TO ANOTHER PREDICATE. Nothing rules out the exporter splitting
one collective across two rows here. It does not need to: `collectiveCoverage`
already pins that the ranks observing one collective id are exactly its member
set, so with every member of a row agreeing on the collective id and the row's
count equal to the member count, a split would have to invent a work.
-/
def collectivePayloadAgreement
    (payloads : List CollectivePayload)
    (rows : List PayloadCollectiveRow) : Bool :=
  !rows.isEmpty &&
  payloads.all (fun item =>
    (item.inputDtypes ++ item.outputDtypes).eraseDups.length == 1 &&
    (match payloadRowAt rows item.collectiveIndex with
      | none => false
      | some row =>
          row.collectiveId == item.collectiveId &&
          row.comm == item.comm &&
          row.operation == item.operation &&
          row.sizeRelation == item.sizeRelation &&
          row.members == item.members &&
          row.inputDtypes == item.inputDtypes &&
          row.outputDtypes == item.outputDtypes &&
          row.inputElements == item.inputElements &&
          row.outputElements == item.outputElements)) &&
  rows.zipIdx.all fun (row, index) =>
    (payloads.filter fun item => item.collectiveIndex == index).length
      == row.members.length

/--
The payload is keyed on the canonical communicator id, and works sharing that key
share a member set. A per-rank local numbering fails that outright, because one
such id then denotes communicators with different members -- the
eight-versus-four confusion.

Stated over the small distinct-key table rather than over every pair of works,
for the cost reason above: the table's keys are pairwise distinct, so two works
with the same key necessarily land on the same row and therefore on the same
member set.

WHAT IS NOT CHECKED HERE, and where it is instead. The collector builds
`collectiveId` as `canonical_id ++ ":seq..."`, so the communicator key is
literally a prefix of it -- the one check no integer key could satisfy. Every
route to a prefix in Lean goes through `String.toList`, which is not axiom-free
under `rfl`, and the checker contract refuses any axiom dependency. It is
asserted at the exporter seam instead, by
test_collective_payload_is_keyed_on_the_canonical_communicator_id.
-/
def payloadCommKeyIsCommunicatorIdentity
    (rows : List PayloadCollectiveRow)
    (commTable : List PayloadCommMembership) : Bool :=
  !commTable.isEmpty &&
  commTable.all (fun entry => !entry.comm.isEmpty && !entry.members.isEmpty) &&
  (commTable.map fun entry => entry.comm).eraseDups.length
    == commTable.length &&
  rows.all fun row =>
    commTable.any fun entry =>
      entry.comm == row.comm && entry.members == row.members

/--
THE second property: the volume relation the operation implies. An all-gather's
output is its member count times its input, a reduce-scatter's the inverse, an
all-reduce's the same. This is what catches an exporter that mislabelled an
operation. The exported relation label is checked against the operation first, so
a label and an operation cannot disagree silently.
-/
def collectivePayloadSizeRelationHolds
    (payloads : List CollectivePayload) : Bool :=
  payloads.all fun item =>
    payloadSizeRelations.contains item.sizeRelation &&
    item.sizeRelation == impliedSizeRelation item.operation &&
    (if item.sizeRelation == "output_is_member_count_times_input" then
        item.outputElements == item.members.length * item.inputElements
      else if item.sizeRelation == "input_is_member_count_times_output" then
        item.inputElements == item.members.length * item.outputElements
      else item.inputElements == item.outputElements)

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
