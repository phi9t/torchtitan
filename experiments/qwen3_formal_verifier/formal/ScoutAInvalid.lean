import ScoutABadFacts

namespace Qwen3Formal.ScoutA

def ControlledInvalidProposition : Prop :=
  gradientReadyBeforeOptimizer ScoutABadFacts.observedEvents = true

instance : Decidable ControlledInvalidProposition := by
  unfold ControlledInvalidProposition
  infer_instance

end Qwen3Formal.ScoutA

example : Qwen3Formal.ScoutA.ControlledInvalidProposition := by
  decide
