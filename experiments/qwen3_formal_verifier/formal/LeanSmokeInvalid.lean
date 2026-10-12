set_option pp.fullNames true

namespace FormalSmoke

def ControlledInvalidProposition : Prop := False

instance : Decidable ControlledInvalidProposition := isFalse id

theorem controlledInvalidWitness : ControlledInvalidProposition := by
  decide

end FormalSmoke
