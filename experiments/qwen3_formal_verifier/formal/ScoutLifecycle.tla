------------------------------ MODULE ScoutLifecycle ------------------------------
EXTENDS Naturals, Sequences, FiniteSets

Prefix(sequence, cursor) ==
  IF cursor = 0 THEN <<>> ELSE SubSeq(sequence, 1, cursor)

SequenceElements(sequence) == {sequence[index] : index \in DOMAIN sequence}

HasKind(kinds, kind) == kind \in SequenceElements(kinds)

IndexOf(sequence, value) ==
  CHOOSE index \in 1..Len(sequence) : sequence[index] = value

BeforeIfPresent(kinds, before, after) ==
  \/ ~HasKind(kinds, after)
  \/ /\ HasKind(kinds, before)
     /\ IndexOf(kinds, before) < IndexOf(kinds, after)

ForwardBeforeBackward(kinds, cursor) ==
  LET observed == Prefix(kinds, cursor)
  IN /\ BeforeIfPresent(observed, "forward.started", "forward.completed")
     /\ BeforeIfPresent(observed, "forward.completed", "backward.started")
     /\ BeforeIfPresent(observed, "backward.started", "backward.completed")

GradientReadyBeforeOptimizer(kinds, cursor) ==
  BeforeIfPresent(
    Prefix(kinds, cursor),
    "gradient.ready",
    "optimizer.mutated"
  )

OptimizerBeforeStepEnd(kinds, cursor) ==
  BeforeIfPresent(
    Prefix(kinds, cursor),
    "optimizer.mutated",
    "step.completed"
  )

CausalOrder(eventIds, eventPredecessors, cursor) ==
  \A index \in 1..cursor :
    \A predecessor \in eventPredecessors[eventIds[index]] :
      /\ predecessor \in SequenceElements(Prefix(eventIds, cursor))
      /\ IndexOf(eventIds, predecessor) < index

=============================================================================
