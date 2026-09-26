------------------------------ MODULE ScoutAValid ------------------------------
EXTENDS ScoutLifecycle, ScoutAFacts

VARIABLE cursor

vars == <<cursor>>

Init == cursor = 0

Next ==
  \/ /\ cursor < Len(EventKinds)
     /\ cursor' = cursor + 1
  \/ /\ cursor = Len(EventKinds)
     /\ UNCHANGED cursor

Spec == Init /\ [][Next]_vars

ScoutAForwardBeforeBackward ==
  ForwardBeforeBackward(EventKinds, cursor)

ScoutAGradientReadyBeforeOptimizer ==
  GradientReadyBeforeOptimizer(EventKinds, cursor)

ScoutAOptimizerBeforeStepEnd ==
  OptimizerBeforeStepEnd(EventKinds, cursor)

ScoutACausalOrder ==
  CausalOrder(EventIds, EventPredecessors, cursor)

=============================================================================
