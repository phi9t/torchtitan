------------------------------ MODULE TraceLifecycle ------------------------------
EXTENDS Naturals, Sequences, FiniteSets

Prefix(sequence, cursor) ==
  IF cursor = 0 THEN <<>> ELSE SubSeq(sequence, 1, cursor)

SequenceElements(sequence) == {sequence[index] : index \in DOMAIN sequence}

HasKind(kinds, kind) == kind \in SequenceElements(kinds)

(***************************************************************************)
(* IndexOf is PARTIAL: it is defined only when the value occurs in the     *)
(* sequence. Applying it to an absent value makes TLC abort with           *)
(* "Attempted to compute the value of an expression of form CHOOSE x \in   *)
(* S: P, but no element of S satisfied P." and exit 75 -- an evaluation    *)
(* crash, not a model-checking result. Every call site below therefore     *)
(* establishes HasKind first, and callers outside this module must do the  *)
(* same.                                                                   *)
(*                                                                         *)
(* The second conjunct pins the FIRST occurrence. Without it the result is *)
(* whichever index TLC's CHOOSE happens to return for a repeated value:    *)
(* CHOOSE is deterministic but unspecified in TLA+, so the previous        *)
(* definition agreed with Lean's occursBeforeIfPresent (which scans        *)
(* left to right) only by relying on an implementation detail of TLC. The  *)
(* single-rank kind sequence has no repeats, but the DPxTP port does --    *)
(* collective kinds recur once per rank -- so the ambiguity is about to be *)
(* reachable.                                                              *)
(***************************************************************************)
IndexOf(sequence, value) ==
  CHOOSE index \in 1..Len(sequence) :
    /\ sequence[index] = value
    /\ \A earlier \in 1..index - 1 : sequence[earlier] # value

(***************************************************************************)
(* Written as nested conditionals rather than a disjunction so that the    *)
(* guard is part of the SEMANTICS, not of TLC's left-to-right evaluation   *)
(* order. The truth table is unchanged: "after" absent -> TRUE; "after"    *)
(* present and "before" absent -> FALSE; both present -> the first         *)
(* occurrence of "before" precedes the first occurrence of "after".        *)
(***************************************************************************)
BeforeIfPresent(kinds, before, after) ==
  IF ~HasKind(kinds, after)
  THEN TRUE
  ELSE IF ~HasKind(kinds, before)
       THEN FALSE
       ELSE IndexOf(kinds, before) < IndexOf(kinds, after)

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

(***************************************************************************)
(* The IF guards IndexOf for the same reason as BeforeIfPresent: the       *)
(* membership conjunct is over the PREFIX, while IndexOf searches the      *)
(* whole sequence, so neither conjunct may be assumed to have been         *)
(* evaluated first.                                                        *)
(***************************************************************************)
CausalOrder(eventIds, eventPredecessors, cursor) ==
  \A index \in 1..cursor :
    \A predecessor \in eventPredecessors[eventIds[index]] :
      /\ predecessor \in SequenceElements(Prefix(eventIds, cursor))
      /\ IF HasKind(eventIds, predecessor)
         THEN IndexOf(eventIds, predecessor) < index
         ELSE FALSE

=============================================================================
