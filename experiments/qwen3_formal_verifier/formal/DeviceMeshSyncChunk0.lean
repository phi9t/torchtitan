import DeviceMeshFacts

namespace Qwen3Formal.DeviceMeshChecks

set_option maxHeartbeats 0
set_option maxRecDepth 100000

theorem syncChunk0Valid :
    crossRankSynchronizationChunkValid
      DeviceMeshFacts.eventIdPrefix
      DeviceMeshFacts.eventsPerRank
      DeviceMeshFacts.crossRankSynchronizationsChunk0 = true := by
  rfl

#print axioms syncChunk0Valid

end Qwen3Formal.DeviceMeshChecks
