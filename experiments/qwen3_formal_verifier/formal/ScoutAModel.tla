-------------------------------- MODULE ScoutAModel --------------------------------
(***************************************************************************)
(* An abstract model of ONE single-rank TorchTitan training step.          *)
(*                                                                         *)
(* This module is deliberately different in kind from ScoutAValid. That    *)
(* module walks a cursor along a constant sequence of observed events and  *)
(* evaluates predicates over it, so TLC explores one path and can say      *)
(* nothing about any execution that was not observed. This module is a     *)
(* transition system: its actions have guards, several of them are         *)
(* concurrently enabled, and TLC explores every admitted interleaving.     *)
(* An invariant that holds here holds for ALL behaviours the protocol      *)
(* permits, not for one trace.                                             *)
(*                                                                         *)
(* Two genuine sources of nondeterminism are modelled:                     *)
(*                                                                         *)
(*   1. Gradient readiness and backward completion may occur in either     *)
(*      order. Gradients are reduced asynchronously, so "gradient.ready"   *)
(*      legitimately lands before or after "backward.completed".           *)
(*   2. Gradients may fail to materialise at all. The observed run took     *)
(*      the ready branch, but the missing branch is a real execution and   *)
(*      the safety argument has to cover it.                               *)
(***************************************************************************)
EXTENDS Naturals, Sequences, ScoutLifecycle

(***************************************************************************)
(* RequireReadyGradients is the one load-bearing guard, exposed as a       *)
(* constant so the negative model can relax exactly it and nothing else.   *)
(* The valid configuration sets it TRUE; the negative configuration sets   *)
(* it FALSE and TLC must then find a real counterexample by exploration,   *)
(* rather than by anyone editing a fact.                                   *)
(***************************************************************************)
CONSTANT RequireReadyGradients

VARIABLES
  emitted,    \* sequence of event kinds emitted so far
  stepOpen,
  batchSeen,
  fwd,        \* "none" | "started" | "completed"
  bwd,        \* "none" | "started" | "completed"
  grad,       \* "unknown" | "ready" | "missing"
  opt,        \* "none" | "started" | "mutated" | "skipped"
  done

vars == <<emitted, stepOpen, batchSeen, fwd, bwd, grad, opt, done>>

Kinds ==
  { "step.started", "batch.observed",
    "forward.started", "forward.completed",
    "backward.started", "backward.completed",
    "gradient.ready", "gradient.missing",
    "optimizer.started", "optimizer.mutated",
    "step.completed" }

TypeOK ==
  /\ stepOpen \in BOOLEAN
  /\ batchSeen \in BOOLEAN
  /\ fwd \in {"none", "started", "completed"}
  /\ bwd \in {"none", "started", "completed"}
  /\ grad \in {"unknown", "ready", "missing"}
  /\ opt \in {"none", "started", "mutated", "skipped"}
  /\ done \in BOOLEAN
  /\ \A i \in DOMAIN emitted : emitted[i] \in Kinds

Init ==
  /\ emitted = <<>>
  /\ stepOpen = FALSE
  /\ batchSeen = FALSE
  /\ fwd = "none"
  /\ bwd = "none"
  /\ grad = "unknown"
  /\ opt = "none"
  /\ done = FALSE

StepStart ==
  /\ ~stepOpen
  /\ stepOpen' = TRUE
  /\ emitted' = Append(emitted, "step.started")
  /\ UNCHANGED <<batchSeen, fwd, bwd, grad, opt, done>>

ObserveBatch ==
  /\ stepOpen
  /\ ~batchSeen
  /\ batchSeen' = TRUE
  /\ emitted' = Append(emitted, "batch.observed")
  /\ UNCHANGED <<stepOpen, fwd, bwd, grad, opt, done>>

ForwardStart ==
  /\ batchSeen
  /\ fwd = "none"
  /\ fwd' = "started"
  /\ emitted' = Append(emitted, "forward.started")
  /\ UNCHANGED <<stepOpen, batchSeen, bwd, grad, opt, done>>

ForwardComplete ==
  /\ fwd = "started"
  /\ fwd' = "completed"
  /\ emitted' = Append(emitted, "forward.completed")
  /\ UNCHANGED <<stepOpen, batchSeen, bwd, grad, opt, done>>

BackwardStart ==
  /\ fwd = "completed"
  /\ bwd = "none"
  /\ bwd' = "started"
  /\ emitted' = Append(emitted, "backward.started")
  /\ UNCHANGED <<stepOpen, batchSeen, fwd, grad, opt, done>>

\* Gradient readiness races backward completion: both orders are admitted.
GradientReady ==
  /\ bwd \in {"started", "completed"}
  /\ grad = "unknown"
  /\ grad' = "ready"
  /\ emitted' = Append(emitted, "gradient.ready")
  /\ UNCHANGED <<stepOpen, batchSeen, fwd, bwd, opt, done>>

GradientMissing ==
  /\ bwd \in {"started", "completed"}
  /\ grad = "unknown"
  /\ grad' = "missing"
  /\ emitted' = Append(emitted, "gradient.missing")
  /\ UNCHANGED <<stepOpen, batchSeen, fwd, bwd, opt, done>>

BackwardComplete ==
  /\ bwd = "started"
  /\ bwd' = "completed"
  /\ emitted' = Append(emitted, "backward.completed")
  /\ UNCHANGED <<stepOpen, batchSeen, fwd, grad, opt, done>>

OptimizerStart ==
  /\ bwd = "completed"
  /\ grad \in {"ready", "missing"}
  /\ opt = "none"
  /\ opt' = "started"
  /\ emitted' = Append(emitted, "optimizer.started")
  /\ UNCHANGED <<stepOpen, batchSeen, fwd, bwd, grad, done>>

\* The load-bearing guard: parameters may only be mutated once gradients are
\* known to be ready. Relaxing this guard is what the negative model does.
OptimizerMutate ==
  /\ opt = "started"
  /\ (RequireReadyGradients => grad = "ready")
  /\ opt' = "mutated"
  /\ emitted' = Append(emitted, "optimizer.mutated")
  /\ UNCHANGED <<stepOpen, batchSeen, fwd, bwd, grad, done>>

OptimizerSkip ==
  /\ opt = "started"
  /\ grad = "missing"
  /\ opt' = "skipped"
  /\ UNCHANGED <<emitted, stepOpen, batchSeen, fwd, bwd, grad, done>>

StepComplete ==
  /\ opt \in {"mutated", "skipped"}
  /\ ~done
  /\ done' = TRUE
  /\ emitted' = Append(emitted, "step.completed")
  /\ UNCHANGED <<stepOpen, batchSeen, fwd, bwd, grad, opt>>

Terminated ==
  /\ done
  /\ UNCHANGED vars

Next ==
  \/ StepStart
  \/ ObserveBatch
  \/ ForwardStart
  \/ ForwardComplete
  \/ BackwardStart
  \/ GradientReady
  \/ GradientMissing
  \/ BackwardComplete
  \/ OptimizerStart
  \/ OptimizerMutate
  \/ OptimizerSkip
  \/ StepComplete
  \/ Terminated

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* Safety properties. These are consequences of the protocol above, and    *)
(* TLC checks them against every reachable state of every interleaving.    *)
(***************************************************************************)

\* Parameters are never mutated unless gradients were ready.
MutationRequiresReadyGradients ==
  (opt = "mutated") => (grad = "ready")

\* The emitted evidence never shows a mutation without prior ready evidence.
MutationFollowsGradientEvidence ==
  BeforeIfPresent(emitted, "gradient.ready", "optimizer.mutated")

\* Gradients resolve exactly once: ready and missing are mutually exclusive.
GradientResolutionUnique ==
  ~(HasKind(emitted, "gradient.ready") /\ HasKind(emitted, "gradient.missing"))

\* Forward/backward bracketing holds in every admitted order.
ForwardBracketsBackward ==
  ForwardBeforeBackward(emitted, Len(emitted))

\* Step completion is terminal: nothing is emitted after it.
StepCompletionIsLast ==
  HasKind(emitted, "step.completed") =>
    emitted[Len(emitted)] = "step.completed"

=============================================================================
