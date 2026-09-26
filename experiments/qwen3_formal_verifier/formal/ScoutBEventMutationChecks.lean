import ScoutBFacts

namespace Qwen3Formal.ScoutBChecks

set_option maxHeartbeats 0
set_option maxRecDepth 100000

def mutateFirstEvent
    (mutation : EventEvidence -> EventEvidence) : List EventEvidence :=
  match ScoutBFacts.eventEvidenceRank0 with
  | [] => []
  | first :: rest => mutation first :: rest

def wrongEventId : List EventEvidence :=
  mutateFirstEvent fun event =>
    { event with eventId := "controlled-wrong-id" }

def wrongEventKind : List EventEvidence :=
  mutateFirstEvent fun event =>
    { event with kind := "batch.observed" }

def wrongEventOrder : List EventEvidence :=
  mutateFirstEvent fun event =>
    { event with order := event.order + 1 }

theorem rejectsEventIdPairDrift :
    eventEvidenceRankValid
      ScoutBFacts.rawEventProjectionSchema
      ScoutBFacts.eventIdPrefix
      ScoutBFacts.eventsPerRank
      0
      ScoutBFacts.eventIdsByRankRank0
      wrongEventId = false := by
  rfl

theorem rejectsWrongEventKind :
    eventEvidenceRankValid
      ScoutBFacts.rawEventProjectionSchema
      ScoutBFacts.eventIdPrefix
      ScoutBFacts.eventsPerRank
      0
      ScoutBFacts.eventIdsByRankRank0
      wrongEventKind = false := by
  rfl

theorem rejectsWrongEventOrder :
    eventEvidenceRankValid
      ScoutBFacts.rawEventProjectionSchema
      ScoutBFacts.eventIdPrefix
      ScoutBFacts.eventsPerRank
      0
      ScoutBFacts.eventIdsByRankRank0
      wrongEventOrder = false := by
  rfl

#print axioms rejectsEventIdPairDrift
#print axioms rejectsWrongEventKind
#print axioms rejectsWrongEventOrder

end Qwen3Formal.ScoutBChecks
