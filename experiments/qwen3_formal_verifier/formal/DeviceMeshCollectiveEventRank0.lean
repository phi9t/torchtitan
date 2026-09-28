import DeviceMeshFacts

namespace Qwen3Formal.DeviceMeshChecks

set_option maxHeartbeats 0
set_option maxRecDepth 100000

theorem collectiveEventRank0Valid :
    collectiveEventEvidenceRankValid
      DeviceMeshFacts.eventIdPrefix
      DeviceMeshFacts.eventsPerRank
      0
      DeviceMeshFacts.collectiveLifecycleRank0 = true := by
  rfl

#print axioms collectiveEventRank0Valid

end Qwen3Formal.DeviceMeshChecks
