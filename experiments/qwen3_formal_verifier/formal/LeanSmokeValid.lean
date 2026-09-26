namespace FormalSmoke

def StepReady (fact : Bool) : Prop := fact = true

theorem validStepReady : StepReady true := rfl

#print axioms FormalSmoke.validStepReady

end FormalSmoke
