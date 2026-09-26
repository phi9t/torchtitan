import ScoutBBadFacts

namespace Qwen3Formal.ScoutB

set_option maxHeartbeats 0
set_option maxRecDepth 100000

def ControlledInvalidProducerProposition : Prop :=
  collectiveProducerCorrelation ScoutBBadFacts.collectiveLifecycle = true

instance : Decidable ControlledInvalidProducerProposition := by
  unfold ControlledInvalidProducerProposition
  infer_instance

end Qwen3Formal.ScoutB

example : Qwen3Formal.ScoutB.ControlledInvalidProducerProposition := by
  decide
