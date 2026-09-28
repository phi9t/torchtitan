------------------------------ MODULE SingleRankValid ------------------------------
EXTENDS TraceLifecycle, SingleRankFacts

VARIABLE cursor

vars == <<cursor>>

Init == cursor = 0

Next ==
  \/ /\ cursor < Len(EventKinds)
     /\ cursor' = cursor + 1
  \/ /\ cursor = Len(EventKinds)
     /\ UNCHANGED cursor

Spec == Init /\ [][Next]_vars

SingleRankForwardBeforeBackward ==
  ForwardBeforeBackward(EventKinds, cursor)

SingleRankGradientReadyBeforeOptimizer ==
  GradientReadyBeforeOptimizer(EventKinds, cursor)

SingleRankOptimizerBeforeStepEnd ==
  OptimizerBeforeStepEnd(EventKinds, cursor)

SingleRankCausalOrder ==
  CausalOrder(EventIds, EventPredecessors, cursor)

=============================================================================
