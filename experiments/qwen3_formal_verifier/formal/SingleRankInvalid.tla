------------------------------ MODULE SingleRankInvalid ------------------------------
EXTENDS TraceLifecycle, SingleRankBadFacts

VARIABLE cursor

vars == <<cursor>>

Init == cursor = 0

Next ==
  \/ /\ cursor < Len(EventKinds)
     /\ cursor' = cursor + 1
  \/ /\ cursor = Len(EventKinds)
     /\ UNCHANGED cursor

Spec == Init /\ [][Next]_vars

SingleRankGradientReadyBeforeOptimizer ==
  GradientReadyBeforeOptimizer(EventKinds, cursor)

=============================================================================
