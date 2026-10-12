------------------------------ MODULE DeviceMeshInvalid ------------------------------
EXTENDS MeshTopology, DeviceMeshBadFacts

VARIABLE cursor

vars == <<cursor>>

Init == cursor = 0

Next ==
  \/ /\ cursor < 1
     /\ cursor' = cursor + 1
  \/ /\ cursor = 1
     /\ UNCHANGED cursor

Spec == Init /\ [][Next]_vars

DeviceMeshCollectiveProducerCorrelation ==
  \/ cursor = 0
  \/ CollectiveProducerCorrelation(CollectiveWorkIds, CollectiveProducers)

=============================================================================
