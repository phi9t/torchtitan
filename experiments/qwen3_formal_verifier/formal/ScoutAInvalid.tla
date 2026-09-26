------------------------------ MODULE ScoutAInvalid ------------------------------
EXTENDS ScoutLifecycle, ScoutABadFacts

VARIABLE cursor

vars == <<cursor>>

Init == cursor = 0

Next ==
  \/ /\ cursor < Len(EventKinds)
     /\ cursor' = cursor + 1
  \/ /\ cursor = Len(EventKinds)
     /\ UNCHANGED cursor

Spec == Init /\ [][Next]_vars

ScoutAGradientReadyBeforeOptimizer ==
  GradientReadyBeforeOptimizer(EventKinds, cursor)

=============================================================================
