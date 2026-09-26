import ScoutBFacts

namespace Qwen3Formal.ScoutBChecks

set_option maxHeartbeats 0
set_option maxRecDepth 100000

theorem eventRank1Valid :
    eventEvidenceRankValid
      ScoutBFacts.rawEventProjectionSchema
      ScoutBFacts.eventIdPrefix
      ScoutBFacts.eventsPerRank
      1
      ScoutBFacts.eventIdsByRankRank1
      ScoutBFacts.eventEvidenceRank1 = true := by
  rfl

#print axioms eventRank1Valid

end Qwen3Formal.ScoutBChecks
