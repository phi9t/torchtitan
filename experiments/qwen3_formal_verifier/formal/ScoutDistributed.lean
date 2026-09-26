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
