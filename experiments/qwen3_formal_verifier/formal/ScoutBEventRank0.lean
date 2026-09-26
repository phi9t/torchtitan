import ScoutBFacts

namespace Qwen3Formal.ScoutBChecks

set_option maxHeartbeats 0
set_option maxRecDepth 100000

theorem eventRank0Valid :
    eventEvidenceRankValid
      ScoutBFacts.rawEventProjectionSchema
      ScoutBFacts.eventIdPrefix
      ScoutBFacts.eventsPerRank
      0
      ScoutBFacts.eventIdsByRankRank0
      ScoutBFacts.eventEvidenceRank0 = true := by
  rfl

#print axioms eventRank0Valid

end Qwen3Formal.ScoutBChecks
