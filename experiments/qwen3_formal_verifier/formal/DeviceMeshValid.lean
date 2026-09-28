import DeviceMeshBaseChecks
import DeviceMeshCollectiveEventRank0
import DeviceMeshCollectiveEventRank1
import DeviceMeshCollectiveEventRank2
import DeviceMeshCollectiveEventRank3
import DeviceMeshEventRank0
import DeviceMeshEventRank1
import DeviceMeshEventRank2
import DeviceMeshEventRank3
import DeviceMeshProducerCheck
import DeviceMeshSyncChunk0
import DeviceMeshSyncChunk1
import DeviceMeshSyncChunk2
import DeviceMeshSyncChunk3

namespace Qwen3Formal.DeviceMesh

open Qwen3Formal.DeviceMeshChecks

set_option maxHeartbeats 0
set_option maxRecDepth 100000

theorem eventEvidenceObserved :
    Qwen3Formal.eventEvidenceValid
      DeviceMeshFacts.rawEventProjectionSchema
      DeviceMeshFacts.eventIdPrefix
      DeviceMeshFacts.eventsPerRank
      DeviceMeshFacts.eventIdsByRank
      DeviceMeshFacts.eventEvidenceByRank = true := by
  change
    (Qwen3Formal.eventEvidenceRankValid
      DeviceMeshFacts.rawEventProjectionSchema
      DeviceMeshFacts.eventIdPrefix DeviceMeshFacts.eventsPerRank 0
      DeviceMeshFacts.eventIdsByRankRank0
      DeviceMeshFacts.eventEvidenceRank0 &&
    Qwen3Formal.eventEvidenceRankValid
      DeviceMeshFacts.rawEventProjectionSchema
      DeviceMeshFacts.eventIdPrefix DeviceMeshFacts.eventsPerRank 1
      DeviceMeshFacts.eventIdsByRankRank1
      DeviceMeshFacts.eventEvidenceRank1 &&
    Qwen3Formal.eventEvidenceRankValid
      DeviceMeshFacts.rawEventProjectionSchema
      DeviceMeshFacts.eventIdPrefix DeviceMeshFacts.eventsPerRank 2
      DeviceMeshFacts.eventIdsByRankRank2
      DeviceMeshFacts.eventEvidenceRank2 &&
    Qwen3Formal.eventEvidenceRankValid
      DeviceMeshFacts.rawEventProjectionSchema
      DeviceMeshFacts.eventIdPrefix DeviceMeshFacts.eventsPerRank 3
      DeviceMeshFacts.eventIdsByRankRank3
      DeviceMeshFacts.eventEvidenceRank3) = true
  rw [eventRank0Valid, eventRank1Valid, eventRank2Valid, eventRank3Valid]
  rfl

theorem crossRankSynchronizationObserved :
    Qwen3Formal.crossRankSynchronizationValid
      DeviceMeshFacts.eventIdPrefix
      DeviceMeshFacts.eventsPerRank
      DeviceMeshFacts.crossRankSynchronizationChunks = true := by
  change
    (Qwen3Formal.crossRankSynchronizationChunkValid
      DeviceMeshFacts.eventIdPrefix DeviceMeshFacts.eventsPerRank
      DeviceMeshFacts.crossRankSynchronizationsChunk0 &&
    Qwen3Formal.crossRankSynchronizationChunkValid
      DeviceMeshFacts.eventIdPrefix DeviceMeshFacts.eventsPerRank
      DeviceMeshFacts.crossRankSynchronizationsChunk1 &&
    Qwen3Formal.crossRankSynchronizationChunkValid
      DeviceMeshFacts.eventIdPrefix DeviceMeshFacts.eventsPerRank
      DeviceMeshFacts.crossRankSynchronizationsChunk2 &&
    Qwen3Formal.crossRankSynchronizationChunkValid
      DeviceMeshFacts.eventIdPrefix DeviceMeshFacts.eventsPerRank
      DeviceMeshFacts.crossRankSynchronizationsChunk3) = true
  rw [syncChunk0Valid, syncChunk1Valid, syncChunk2Valid, syncChunk3Valid]
  rfl

theorem collectiveEventEvidenceObserved :
    Qwen3Formal.collectiveEventEvidence
      DeviceMeshFacts.eventIdPrefix
      DeviceMeshFacts.eventsPerRank
      DeviceMeshFacts.collectiveLifecycleByRank = true := by
  change
    (Qwen3Formal.collectiveEventEvidenceRankValid
      DeviceMeshFacts.eventIdPrefix DeviceMeshFacts.eventsPerRank 0
      DeviceMeshFacts.collectiveLifecycleRank0 &&
    Qwen3Formal.collectiveEventEvidenceRankValid
      DeviceMeshFacts.eventIdPrefix DeviceMeshFacts.eventsPerRank 1
      DeviceMeshFacts.collectiveLifecycleRank1 &&
    Qwen3Formal.collectiveEventEvidenceRankValid
      DeviceMeshFacts.eventIdPrefix DeviceMeshFacts.eventsPerRank 2
      DeviceMeshFacts.collectiveLifecycleRank2 &&
    Qwen3Formal.collectiveEventEvidenceRankValid
      DeviceMeshFacts.eventIdPrefix DeviceMeshFacts.eventsPerRank 3
      DeviceMeshFacts.collectiveLifecycleRank3) = true
  rw [
    collectiveEventRank0Valid,
    collectiveEventRank1Valid,
    collectiveEventRank2Valid,
    collectiveEventRank3Valid,
  ]
  rfl

theorem validDPxTP :
    Qwen3Formal.validDPxTP
      DeviceMeshFacts.rankCoordinates
      DeviceMeshFacts.collectiveLifecycle
      DeviceMeshFacts.collectiveLifecycleByRank
      DeviceMeshFacts.eventEvidenceByRank
      DeviceMeshFacts.crossRankSynchronizationChunks
      DeviceMeshFacts.rawEventProjectionSchema
      DeviceMeshFacts.eventIdPrefix
      DeviceMeshFacts.eventsPerRank
      DeviceMeshFacts.collectivesPerRank
      DeviceMeshFacts.eventIdsByRank
      DeviceMeshFacts.collectiveWorkIdsByRank
      DeviceMeshFacts.collectiveIdsByRank = true := by
  unfold Qwen3Formal.validDPxTP
  rw [
    meshWellFormedValid,
    tpInputAgreementValid,
    placementValidObserved,
    eventEvidenceObserved,
    crossRankSynchronizationObserved,
    collectiveLifecycleObserved,
    collectiveCoverageObserved,
    collectiveProducerObserved,
    collectiveEventEvidenceObserved,
  ]
  rfl

#print axioms eventEvidenceObserved
#print axioms crossRankSynchronizationObserved
#print axioms collectiveEventEvidenceObserved
#print axioms validDPxTP

end Qwen3Formal.DeviceMesh
