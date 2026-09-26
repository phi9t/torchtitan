/-
A controlled negative for the wait-graph witness. Lean must REJECT this module.

`acyclicityIsLoadBearing` in ScoutBWaitGraph.lean is a positive theorem that
refutes the weakened statement, which is the stronger form of negative: it is
kernel-checked rather than merely uncompilable. This module adds the
uncompilable form as well, so the witness's central property -- that the state
really is stuck -- is checked from both sides and so that this suite has the
same shape of negative as ScoutAInvalid and ScoutBInvalid.

The proposition below claims the cyclic witness is NOT stuck. It is false, and
`decide` says so, and the runner classifies the resulting diagnostic. A run in
which this module COMPILES is a failure of the suite, not a success.
-/
import ScoutBWaitGraph

namespace Qwen3Formal.ScoutBProtocol

/-- False by construction: the witness is stuck. -/
def ControlledInvalidWaitGraphProposition : Prop :=
  stuckB cyclicTopology cyclicState = false

instance : Decidable ControlledInvalidWaitGraphProposition := by
  unfold ControlledInvalidWaitGraphProposition
  infer_instance

end Qwen3Formal.ScoutBProtocol

example : Qwen3Formal.ScoutBProtocol.ControlledInvalidWaitGraphProposition := by
  decide
