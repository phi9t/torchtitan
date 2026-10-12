import DeviceMeshBadFacts

namespace Qwen3Formal.DeviceMesh

set_option maxHeartbeats 0
set_option maxRecDepth 100000

def ControlledInvalidProducerProposition : Prop :=
  collectiveProducerCorrelation DeviceMeshBadFacts.collectiveLifecycle = true

instance : Decidable ControlledInvalidProducerProposition := by
  unfold ControlledInvalidProducerProposition
  infer_instance

end Qwen3Formal.DeviceMesh

example : Qwen3Formal.DeviceMesh.ControlledInvalidProducerProposition := by
  decide
