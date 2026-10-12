import SingleRankBadFacts

namespace Qwen3Formal.SingleRank

def ControlledInvalidProposition : Prop :=
  gradientReadyBeforeOptimizer SingleRankBadFacts.observedEvents = true

instance : Decidable ControlledInvalidProposition := by
  unfold ControlledInvalidProposition
  infer_instance

end Qwen3Formal.SingleRank

example : Qwen3Formal.SingleRank.ControlledInvalidProposition := by
  decide
