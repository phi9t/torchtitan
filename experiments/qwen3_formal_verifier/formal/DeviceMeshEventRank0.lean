import DeviceMeshFacts

namespace Qwen3Formal.DeviceMeshChecks

set_option maxHeartbeats 0
set_option maxRecDepth 100000

theorem eventRank0Valid :
    eventEvidenceRankValid
      DeviceMeshFacts.rawEventProjectionSchema
      DeviceMeshFacts.eventIdPrefix
      DeviceMeshFacts.eventsPerRank
      0
      DeviceMeshFacts.eventIdsByRankRank0
      DeviceMeshFacts.eventEvidenceRank0 = true := by
  rfl

#print axioms eventRank0Valid

end Qwen3Formal.DeviceMeshChecks
