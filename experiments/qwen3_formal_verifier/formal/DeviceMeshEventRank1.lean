import DeviceMeshFacts

namespace Qwen3Formal.DeviceMeshChecks

set_option maxHeartbeats 0
set_option maxRecDepth 100000

theorem eventRank1Valid :
    eventEvidenceRankValid
      DeviceMeshFacts.rawEventProjectionSchema
      DeviceMeshFacts.eventIdPrefix
      DeviceMeshFacts.eventsPerRank
      1
      DeviceMeshFacts.eventIdsByRankRank1
      DeviceMeshFacts.eventEvidenceRank1 = true := by
  rfl

#print axioms eventRank1Valid

end Qwen3Formal.DeviceMeshChecks
