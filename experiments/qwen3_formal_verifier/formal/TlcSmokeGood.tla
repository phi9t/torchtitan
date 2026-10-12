--------------------------- MODULE TlcSmokeGood ---------------------------
VARIABLE phase

Init == phase = "ready"
Next == UNCHANGED phase
Spec == Init /\ [][Next]_<<phase>>

SmokeInvariant == phase = "ready"
=============================================================================
