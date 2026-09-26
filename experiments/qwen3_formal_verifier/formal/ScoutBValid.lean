import ScoutBBaseChecks
import ScoutBCollectiveEventRank0
import ScoutBCollectiveEventRank1
import ScoutBCollectiveEventRank2
import ScoutBCollectiveEventRank3
import ScoutBEventRank0
import ScoutBEventRank1
import ScoutBEventRank2
import ScoutBEventRank3
import ScoutBProducerCheck
import ScoutBSyncChunk0
import ScoutBSyncChunk1
import ScoutBSyncChunk2
import ScoutBSyncChunk3

namespace Qwen3Formal.ScoutB

open Qwen3Formal.ScoutBChecks

set_option maxHeartbeats 0
set_option maxRecDepth 100000

theorem eventEvidenceObserved :
    Qwen3Formal.eventEvidenceValid
      ScoutBFacts.rawEventProjectionSchema
      ScoutBFacts.eventIdPrefix
      ScoutBFacts.eventsPerRank
      ScoutBFacts.eventIdsByRank
      ScoutBFacts.eventEvidenceByRank = true := by
  change
    (Qwen3Formal.eventEvidenceRankValid
      ScoutBFacts.rawEventProjectionSchema
      ScoutBFacts.eventIdPrefix ScoutBFacts.eventsPerRank 0
      ScoutBFacts.eventIdsByRankRank0
      ScoutBFacts.eventEvidenceRank0 &&
    Qwen3Formal.eventEvidenceRankValid
      ScoutBFacts.rawEventProjectionSchema
      ScoutBFacts.eventIdPrefix ScoutBFacts.eventsPerRank 1
      ScoutBFacts.eventIdsByRankRank1
      ScoutBFacts.eventEvidenceRank1 &&
    Qwen3Formal.eventEvidenceRankValid
      ScoutBFacts.rawEventProjectionSchema
      ScoutBFacts.eventIdPrefix ScoutBFacts.eventsPerRank 2
      ScoutBFacts.eventIdsByRankRank2
      ScoutBFacts.eventEvidenceRank2 &&
    Qwen3Formal.eventEvidenceRankValid
      ScoutBFacts.rawEventProjectionSchema
      ScoutBFacts.eventIdPrefix ScoutBFacts.eventsPerRank 3
      ScoutBFacts.eventIdsByRankRank3
      ScoutBFacts.eventEvidenceRank3) = true
  rw [eventRank0Valid, eventRank1Valid, eventRank2Valid, eventRank3Valid]
  rfl

theorem crossRankSynchronizationObserved :
    Qwen3Formal.crossRankSynchronizationValid
      ScoutBFacts.eventIdPrefix
      ScoutBFacts.eventsPerRank
      ScoutBFacts.crossRankSynchronizationChunks = true := by
  change
    (Qwen3Formal.crossRankSynchronizationChunkValid
      ScoutBFacts.eventIdPrefix ScoutBFacts.eventsPerRank
      ScoutBFacts.crossRankSynchronizationsChunk0 &&
    Qwen3Formal.crossRankSynchronizationChunkValid
      ScoutBFacts.eventIdPrefix ScoutBFacts.eventsPerRank
      ScoutBFacts.crossRankSynchronizationsChunk1 &&
    Qwen3Formal.crossRankSynchronizationChunkValid
      ScoutBFacts.eventIdPrefix ScoutBFacts.eventsPerRank
      ScoutBFacts.crossRankSynchronizationsChunk2 &&
    Qwen3Formal.crossRankSynchronizationChunkValid
      ScoutBFacts.eventIdPrefix ScoutBFacts.eventsPerRank
      ScoutBFacts.crossRankSynchronizationsChunk3) = true
  rw [syncChunk0Valid, syncChunk1Valid, syncChunk2Valid, syncChunk3Valid]
  rfl

theorem collectiveEventEvidenceObserved :
    Qwen3Formal.collectiveEventEvidence
      ScoutBFacts.eventIdPrefix
      ScoutBFacts.eventsPerRank
      ScoutBFacts.collectiveLifecycleByRank = true := by
  change
    (Qwen3Formal.collectiveEventEvidenceRankValid
      ScoutBFacts.eventIdPrefix ScoutBFacts.eventsPerRank 0
      ScoutBFacts.collectiveLifecycleRank0 &&
    Qwen3Formal.collectiveEventEvidenceRankValid
      ScoutBFacts.eventIdPrefix ScoutBFacts.eventsPerRank 1
      ScoutBFacts.collectiveLifecycleRank1 &&
    Qwen3Formal.collectiveEventEvidenceRankValid
      ScoutBFacts.eventIdPrefix ScoutBFacts.eventsPerRank 2
      ScoutBFacts.collectiveLifecycleRank2 &&
    Qwen3Formal.collectiveEventEvidenceRankValid
      ScoutBFacts.eventIdPrefix ScoutBFacts.eventsPerRank 3
      ScoutBFacts.collectiveLifecycleRank3) = true
  rw [
    collectiveEventRank0Valid,
    collectiveEventRank1Valid,
    collectiveEventRank2Valid,
    collectiveEventRank3Valid,
  ]
  rfl

theorem validDPxTP :
    Qwen3Formal.validDPxTP
      ScoutBFacts.rankCoordinates
      ScoutBFacts.collectiveLifecycle
      ScoutBFacts.collectiveLifecycleByRank
      ScoutBFacts.eventEvidenceByRank
      ScoutBFacts.crossRankSynchronizationChunks
      ScoutBFacts.rawEventProjectionSchema
      ScoutBFacts.eventIdPrefix
      ScoutBFacts.eventsPerRank
      ScoutBFacts.collectivesPerRank
      ScoutBFacts.eventIdsByRank
      ScoutBFacts.collectiveWorkIdsByRank
      ScoutBFacts.collectiveIdsByRank = true := by
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

end Qwen3Formal.ScoutB
