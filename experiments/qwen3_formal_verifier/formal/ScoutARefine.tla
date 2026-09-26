------------------------------- MODULE ScoutARefine -------------------------------
(***************************************************************************)
(* Trace refinement: is the OBSERVED run an admitted behaviour of the      *)
(* abstract model?                                                         *)
(*                                                                         *)
(* This is the bridge that makes runtime evidence mean something. The      *)
(* model in ScoutAModel constrains what a training step may do; this       *)
(* module asks whether what the real four-rank... single-rank step         *)
(* actually did lies inside that space. Without it the model and the trace *)
(* are two unrelated artifacts.                                            *)
(*                                                                         *)
(* Method. The model's transitions are restricted so that the emitted      *)
(* sequence must stay a prefix of the observed sequence. The observed run  *)
(* is admitted exactly when a state whose emitted sequence is the whole    *)
(* observed sequence is REACHABLE.                                         *)
(*                                                                         *)
(* TLC proves reachability by refutation, so the witness is a violation of *)
(* ObservedTraceIsNotAdmitted. A reported violation of that invariant      *)
(* therefore means the refinement HOLDS, and the counterexample TLC prints *)
(* is the step-by-step alignment of model actions to observed events. The  *)
(* runner translates this polarity into a positive result token so the     *)
(* evidence does not read backwards.                                       *)
(***************************************************************************)
EXTENDS Naturals, Sequences, ScoutAModel, ScoutAFacts

Observed == EventKinds

IsPrefixOfObserved(s) ==
  /\ Len(s) <= Len(Observed)
  /\ \A i \in DOMAIN s : s[i] = Observed[i]

\* Every model transition, further restricted to stay on the observed path.
ConstrainedNext ==
  /\ Next
  /\ IsPrefixOfObserved(emitted')

RSpec == Init /\ [][ConstrainedNext]_vars

\* Violating this invariant is the refinement witness. See the header.
ObservedTraceIsNotAdmitted ==
  Len(emitted) < Len(Observed)

=============================================================================
