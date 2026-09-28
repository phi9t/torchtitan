----------------------------- MODULE SingleRankRefineBad -----------------------------
(***************************************************************************)
(* Negative control for the refinement bridge.                             *)
(*                                                                         *)
(* The corruption must be REJECTED FOR THE RIGHT REASON. An earlier        *)
(* version swapped "gradient.ready" with "optimizer.mutated", which put    *)
(* the mutation at position 6 where the optimizer had not started. The     *)
(* model refused it, but because the lifecycle was out of order, not       *)
(* because gradients were unready -- so it exercised a different guard     *)
(* than the one under test.                                                *)
(*                                                                         *)
(* This version substitutes "gradient.missing" for "gradient.ready" and    *)
(* changes nothing else. Every earlier event stays admissible, the         *)
(* optimizer legitimately starts (OptimizerStart accepts resolved-missing  *)
(* gradients), and the trace dies exactly at OptimizerMutate, whose guard  *)
(* demands grad = "ready". That isolates MutationRequiresReadyGradients.   *)
(*                                                                         *)
(* Deriving from the observed facts rather than hand-writing a literal     *)
(* keeps the control from drifting away from the real trace and then       *)
(* being refused for some unrelated reason.                                *)
(***************************************************************************)
EXTENDS Naturals, Sequences, SingleRankModel, SingleRankFacts

(***************************************************************************)
(* IndexOf is partial: applied to an absent value it makes TLC abandon the *)
(* search with an evaluation error, which is not a checking result at all. *)
(* These two indices are read while Observed is being constructed, so no   *)
(* invariant could catch the absence first -- hence the guard here, with 0 *)
(* as the out-of-range sentinel, and ControlEventsArePresent below to turn *)
(* the sentinel into a named invariant violation.                          *)
(***************************************************************************)
ReadyIndex ==
  IF HasKind(EventKinds, "gradient.ready")
  THEN IndexOf(EventKinds, "gradient.ready")
  ELSE 0

MutatedIndex ==
  IF HasKind(EventKinds, "optimizer.mutated")
  THEN IndexOf(EventKinds, "optimizer.mutated")
  ELSE 0

\* The observed trace with gradient readiness downgraded to missing.
Observed ==
  [ i \in DOMAIN EventKinds |->
      IF i = ReadyIndex THEN "gradient.missing" ELSE EventKinds[i] ]

\* Both control events must exist in the observed trace, or ReadyIndex and
\* MutatedIndex are the 0 sentinel and the corruption is not a corruption.
\* Checked first in all three configurations so this is the reported failure.
ControlEventsArePresent ==
  /\ ReadyIndex \in DOMAIN EventKinds
  /\ MutatedIndex \in DOMAIN EventKinds

(***************************************************************************)
(* Guards against this control silently becoming trivial: it must differ   *)
(* from the observed trace at exactly one position, that position must be  *)
(* the gradient event, and the mutation must still come after it.          *)
(***************************************************************************)
CorruptionIsIsolated ==
  /\ Len(Observed) = Len(EventKinds)
  /\ { i \in DOMAIN EventKinds : Observed[i] # EventKinds[i] } = {ReadyIndex}
  /\ ReadyIndex < MutatedIndex

IsPrefixOfObserved(s) ==
  /\ Len(s) <= Len(Observed)
  /\ \A i \in DOMAIN s : s[i] = Observed[i]

ConstrainedNext ==
  /\ Next
  /\ IsPrefixOfObserved(emitted')

RSpec == Init /\ [][ConstrainedNext]_vars

\* No violation means the corrupted trace was never fully emitted: rejected.
ObservedTraceIsNotAdmitted ==
  Len(emitted) < Len(Observed)

\* Violating this is the witness that the model accepted every event up to the
\* mutation, proving rejection happens no EARLIER than the mutation guard.
RejectionHappensBeforeTheMutation ==
  Len(emitted) < MutatedIndex - 1

\* Reachability of the prefix plus non-admission of the whole trace still
\* leaves one gap: the model might admit the mutation and refuse the final
\* "step.completed" instead. This closes it directly -- the mutation is never
\* emitted at all, so the refusing action is OptimizerMutate and nothing later.
MutationIsNeverEmitted ==
  \A i \in DOMAIN emitted : emitted[i] # "optimizer.mutated"

=============================================================================
