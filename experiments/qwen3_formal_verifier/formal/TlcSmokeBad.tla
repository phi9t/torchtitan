---------------------------- MODULE TlcSmokeBad ----------------------------
VARIABLE phase

Init == phase = "broken"
Next == UNCHANGED phase
Spec == Init /\ [][Next]_<<phase>>

SmokeInvariant == phase = "ready"
=============================================================================
