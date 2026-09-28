import DeviceMeshFacts

namespace Qwen3Formal.DeviceMeshChecks

set_option maxHeartbeats 0
set_option maxRecDepth 100000

theorem collectiveEventRank1Valid :
    collectiveEventEvidenceRankValid
      DeviceMeshFacts.eventIdPrefix
      DeviceMeshFacts.eventsPerRank
      1
      DeviceMeshFacts.collectiveLifecycleRank1 = true := by
  rfl

#print axioms collectiveEventRank1Valid

end Qwen3Formal.DeviceMeshChecks
