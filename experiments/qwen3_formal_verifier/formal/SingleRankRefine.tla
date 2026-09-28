------------------------------- MODULE SingleRankRefine -------------------------------
(***************************************************************************)
(* Trace refinement: is the OBSERVED run an admitted behaviour of the      *)
(* abstract model?                                                         *)
(*                                                                         *)
(* The model in SingleRankModel constrains what a training step may do; this   *)
(* module asks whether what the real single-rank step actually did lies    *)
(* inside that space. Without it the model and the trace would be two      *)
(* unrelated artifacts.                                                    *)
(*                                                                         *)
(* Reach, stated precisely. single_rank.py requires the normalized event kinds *)
(* to equal one 10-element literal, so EventKinds is that same constant on *)
(* every run and a deviating run is refused by the Python validator before *)
(* any facts are exported. What this module establishes is therefore that  *)
(* THAT constant sequence lies in the abstract model's language: an        *)
(* agreement between two independently written artifacts, not a per-run    *)
(* check that would catch a lifecycle deviation. Catching the deviation is *)
(* the validator's job; this check shows the model and the validator agree *)
(* on what a legal step looks like.                                        *)
(*                                                                         *)
(* Method. The model's transitions are restricted so that the emitted      *)
(* sequence must stay a prefix of the observed sequence. The observed run  *)
(* is admitted exactly when a state whose emitted sequence is the whole    *)
(* observed sequence is REACHABLE.                                         *)
(*                                                                         *)
(* TLC proves reachability by refutation, so the witness is a violation of *)
(* ObservedTraceIsNotAdmitted. A reported violation of that invariant      *)
(* therefore means the refinement HOLDS. The runner translates this        *)
(* polarity into a positive result token so the evidence does not read     *)
(* backwards.                                                              *)
(*                                                                         *)
(* What TLC prints is a state-by-state trace in which every step carries   *)
(* the same action label, ConstrainedNext, because that is RSpec's only    *)
(* action. It does NOT print which model action produced each observed     *)
(* event: that is inferable from the state variables in the printed        *)
(* states, and is not something the checker states.                        *)
(***************************************************************************)
EXTENDS Naturals, Sequences, SingleRankModel, SingleRankFacts

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

(***************************************************************************)
(* Non-vacuity. An EMPTY observed sequence is admitted by every model --   *)
(* Init already satisfies emitted = Observed -- and produces exactly the   *)
(* same TLC outcome as a REFUSED trace: no violation, exit 0. The runner   *)
(* could not tell the two apart and printed "not admitted" for the         *)
(* vacuous case, which states the opposite of what happened. This          *)
(* invariant mentions no VARIABLES, so TLC refutes a false one as a        *)
(* constant expression with exit 151 and its own message, giving the       *)
(* runner a distinguishable outcome to report correctly.                   *)
(***************************************************************************)
ObservedIsNonEmpty ==
  Len(Observed) > 0

=============================================================================
