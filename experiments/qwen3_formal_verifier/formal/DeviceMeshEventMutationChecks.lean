import DeviceMeshFacts

namespace Qwen3Formal.DeviceMeshChecks

set_option maxHeartbeats 0
set_option maxRecDepth 100000

def mutateFirstEvent
    (mutation : EventEvidence -> EventEvidence) : List EventEvidence :=
  match DeviceMeshFacts.eventEvidenceRank0 with
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
      DeviceMeshFacts.rawEventProjectionSchema
      DeviceMeshFacts.eventIdPrefix
      DeviceMeshFacts.eventsPerRank
      0
      DeviceMeshFacts.eventIdsByRankRank0
      wrongEventId = false := by
  rfl

theorem rejectsWrongEventKind :
    eventEvidenceRankValid
      DeviceMeshFacts.rawEventProjectionSchema
      DeviceMeshFacts.eventIdPrefix
      DeviceMeshFacts.eventsPerRank
      0
      DeviceMeshFacts.eventIdsByRankRank0
      wrongEventKind = false := by
  rfl

theorem rejectsWrongEventOrder :
    eventEvidenceRankValid
      DeviceMeshFacts.rawEventProjectionSchema
      DeviceMeshFacts.eventIdPrefix
      DeviceMeshFacts.eventsPerRank
      0
      DeviceMeshFacts.eventIdsByRankRank0
      wrongEventOrder = false := by
  rfl

#print axioms rejectsEventIdPairDrift
#print axioms rejectsWrongEventKind
#print axioms rejectsWrongEventOrder

end Qwen3Formal.DeviceMeshChecks
