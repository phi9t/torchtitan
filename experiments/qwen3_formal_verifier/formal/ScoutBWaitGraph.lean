/-
ScoutBWaitGraph -- the general protocol theorem: per-communicator order
agreement plus an acyclic communicator wait-for graph implies neither a
rendezvous mismatch nor a stream-ordered circular wait, for ALL topologies,
states and schedule lengths.

HOW THIS PAIRS WITH TLC, AND WHY THE PAIR IS STRONGER THAN EITHER HALF.

  - This file proves one direction: order agreement AND acyclicity give no
    mismatch and no circular wait. Unbounded in schedule length, unbounded in
    instance size.
  - ScoutBModelDivergent.cfg refutes the converse reading: with operations
    still agreeing at every position and NCCL's own guard still on, relaxing
    communicator-site agreement makes DeadlockFreedom FALSE. So order
    agreement alone is not sufficient, and the acyclicity hypothesis here is
    not a decoration added to make a proof go through.
  - `acyclicityIsLoadBearing` below closes the same gap inside Lean, on a
    two-communicator witness that satisfies every other hypothesis of the
    theorem and fails only acyclicity.

WHAT IS PROVED, AND WHAT IS NOT. `DeadlockFreedom == ~StuckByCircularWait` is
NOT proved unconditionally. Whether the wait-for graph of a given set of guards
is acyclic is an instance question, and it is the question TLC answers -- at
MaxIssues = 2 for the shipped instance. What Lean carries is the implication,
which is the part that is independent of the bound. Read together: TLC says the
shipped guards admit no circular wait up to the bound and that relaxing
site agreement admits one; Lean says that wherever the wait graph is acyclic no
circular wait exists at any bound. The unbounded, unconditional statement
remains open and is recorded as such.

THE WAIT-FOR GRAPH. `blockedBy T s c d` is TLA's `d \in BlockedBy(c)`: some
member of `c` has an earlier, uncompleted issue on `d`, on the same stream as
that member's front issue on `c`. `WaitClosed T s K` is the set-level form of
TLA's `CircularWaitAt(c)`: every element of `K` is fully pending with agreeing
operations, and `K` is closed under `blockedBy`. TLA takes `K` to be
`BlockingClosure(c)`, the least such set, so `\E K, WaitClosed T s K /\ K c` and
`CircularWaitAt(c)` pick out the same states -- one direction by taking K to be
the closure, the other because the closure is contained in any closed superset.

THE MEASURE. Acyclicity is supplied as a rank function strictly decreasing
along `blockedBy`. That is the standard finite-acyclicity witness and it is
what makes the induction go through without any well-foundedness machinery from
Mathlib. It is also the form the Phase 4 observation suggests: `issued` and
`doneOn` are monotone non-decreasing and `Complete` is the only action removing
from `running`, so the reachable STATE graph is acyclic; the graph that has to
be acyclic here is the per-state COMMUNICATOR wait graph, which is a different
object and is exactly what the guards control.

No `simp`, no `omega`, nothing classical. Every witness below is extracted from
a FALSE Bool-valued guard by case analysis, which is why the file needs no
excluded middle: see the AXIOM DISCIPLINE note in ScoutBProtocol.lean.
-/
import ScoutBInductiveInvariant

namespace Qwen3Formal.ScoutBProtocol

section WaitGraph

variable {R C O S : Type} [DecidableEq R] [DecidableEq C] [DecidableEq O]
variable [DecidableEq S]

set_option linter.unusedSectionVars false

/-- TLA's `d \in BlockedBy(c)`. See divergence 8 in ScoutBProtocol.lean for the
one conjunct of `BlockingPairs` that is dropped and why that direction is
safe. -/
def blockedBy (T : Topology R C O S) (s : State R C O) (c d : C) : Prop :=
  T.requireStreamOrder = true ∧
    ∃ (r : R) (k j : Nat) (ek ej : Issue C O),
      T.member c r = true ∧
      idxOfOn c (front s c) (s.issued r) = some k ∧
      atPos (s.issued r) k = some ek ∧
      atPos (s.issued r) j = some ej ∧
      1 <= j ∧ j < k ∧
      T.streamOf ej = T.streamOf ek ∧
      completedB s r j = false ∧
      ej.comm = d

/-- The set-level form of TLA's `CircularWaitAt(c)`. -/
def WaitClosed (T : Topology R C O S) (s : State R C O) (K : C -> Prop) :
    Prop :=
  ∀ c, K c ->
    c ∈ T.comms ∧ fullyPendingB T s c = true ∧ opsAgreeAtFrontB T s c = true ∧
      (∀ d, blockedBy T s c d -> K d)

/-! ## Reading a stuck state -/

theorem stuckNotStartAllowed {T : Topology R C O S} {s : State R C O} {c : C}
    (h : stuckB T s = true) (hc : c ∈ T.comms) : startAllowedB T s c = false :=
  notTrue (allMem (andRight h) c hc)

theorem stuckNotRunning {T : Topology R C O S} {s : State R C O} {c : C}
    (h : stuckB T s = true) (hc : c ∈ T.comms) : s.running c = false :=
  notTrue (allMem (andLeft (andLeft h)) c hc)

theorem fullyPendingFront {T : Topology R C O S} {s : State R C O} {c : C}
    (h : fullyPendingB T s c = true) (r : R) (hr : T.member c r = true) :
    front s c <= commCount s r c :=
  ofDecide (impFromOr (allMem h r (T.membersInRanks c r hr)) hr)

/-! ## The bridge: a stuck, fully pending, agreeing communicator has a blocker

This is the substantive step. Under `stuckB` the communicator cannot start;
`fullyPendingB` supplies every member's issue, so the count conjunct of
`MemberReady` holds; `opsAgreeAtFrontB` with NCCL's guard on supplies the
matching conjunct of `StartAllowed`. The only conjunct left to fail is
`AtStreamHead`, and its failure IS a blocking edge. -/

theorem blockerOfStuckPending (T : Topology R C O S) (s : State R C O) (c : C)
    (hmatch : T.requireMatchedIssueOrder = true)
    (hstream : T.requireStreamOrder = true)
    (hstuck : stuckB T s = true) (hc : c ∈ T.comms)
    (hfp : fullyPendingB T s c = true)
    (hagree : opsAgreeAtFrontB T s c = true) :
    ∃ d, blockedBy T s c d := by
  have hstart : startAllowedB T s c = false := stuckNotStartAllowed hstuck hc
  -- The matching conjunct holds, so the readiness conjunct is the false one.
  have hthird : ((!T.requireMatchedIssueOrder) || opsAgreeAtFrontB T s c)
      = true := by rw [hmatch]; exact hagree
  have hfirst : (!s.running c) = true := by
    rw [stuckNotRunning hstuck hc]; rfl
  have hready : (T.ranks.all fun r => (!T.member c r) || memberReadyB T s c r)
      = false := by
    cases andFalseCases hstart with
    | inl h12 =>
        cases andFalseCases h12 with
        | inl h1 =>
            exact absurd hfirst
              (fun hc' => Bool.noConfusion (h1.symm.trans hc'))
        | inr h2 => exact h2
    | inr h3 =>
        exact absurd hthird
          (fun hc' => Bool.noConfusion (h3.symm.trans hc'))
  have ⟨r, hrmem, hrfalse⟩ := allFalseWitness hready
  have hr : T.member c r = true := notFalse (orFalseLeft hrfalse)
  have hnotready : memberReadyB T s c r = false := orFalseRight hrfalse
  -- The count conjunct holds, so the stream-head conjunct is the false one.
  have hcount : decide (front s c <= commCount s r c) = true :=
    toDecide (fullyPendingFront hfp r hr)
  have hmatchPart :
      (match idxOfOn c (front s c) (s.issued r) with
       | none => true
       | some k => atStreamHeadB T s r k) = false := by
    cases andFalseCases hnotready with
    | inl hbad =>
        exact absurd hcount (fun hc' => Bool.noConfusion (hbad.symm.trans hc'))
    | inr hgood => exact hgood
  match hidx : idxOfOn c (front s c) (s.issued r) with
  | none =>
      rw [hidx] at hmatchPart
      exact Bool.noConfusion (show true = false from hmatchPart)
  | some k =>
      rw [hidx] at hmatchPart
      have hhead : atStreamHeadB T s r k = false := hmatchPart
      have hinner :
          (match atPos (s.issued r) k with
           | none => true
           | some ek =>
               allLt k (fun j =>
                 match atPos (s.issued r) j with
                 | none => true
                 | some ej =>
                     (!decide (T.streamOf ej = T.streamOf ek))
                       || completedB s r j)) = false := by
        have h0 : ((!T.requireStreamOrder) ||
            (match atPos (s.issued r) k with
             | none => true
             | some ek =>
                 allLt k (fun j =>
                   match atPos (s.issued r) j with
                   | none => true
                   | some ej =>
                       (!decide (T.streamOf ej = T.streamOf ek))
                         || completedB s r j))) = false := hhead
        exact orFalseRight h0
      match hek : atPos (s.issued r) k with
      | none =>
          rw [hek] at hinner
          exact Bool.noConfusion (show true = false from hinner)
      | some ek =>
          rw [hek] at hinner
          have ⟨j, hjk, hjfalse⟩ := allLt_false_witness k _ hinner
          match hej : atPos (s.issued r) j with
          | none =>
              rw [hej] at hjfalse
              exact Bool.noConfusion (show true = false from hjfalse)
          | some ej =>
              rw [hej] at hjfalse
              refine ⟨ej.comm, hstream, r, k, j, ek, ej, hr, hidx, hek, hej,
                ?_, hjk, ?_, orFalseRight hjfalse, rfl⟩
              · match j with
                | 0 =>
                    exact absurdNoneSome
                      (show (none : Option (Issue C O)) = some ej from
                        (atPos_zero (s.issued r)).symm.trans hej)
                | _ + 1 => exact Nat.succ_le_succ (Nat.zero_le _)
              · exact ofDecide (notFalse (orFalseLeft hjfalse))

/-! ## The abstract acyclicity argument

A relation with a strictly decreasing rank has no set in which every element
has a successor. Stated for an arbitrary relation so that nothing about
collectives is smuggled into the induction. -/

theorem noChainInAcyclicRelation {A : Type} (blocks : A -> A -> Prop)
    (rank : A -> Nat) (hdec : ∀ x y, blocks x y -> rank y < rank x)
    (inChain : A -> Prop)
    (hsucc : ∀ x, inChain x -> ∃ y, inChain y ∧ blocks x y)
    (x0 : A) (h0 : inChain x0) : False := by
  have key : ∀ n x, inChain x -> rank x <= n -> False := by
    intro n
    induction n with
    | zero =>
        intro x hx hle
        have ⟨y, hy, hb⟩ := hsucc x hx
        exact absurd (Nat.lt_of_lt_of_le (hdec x y hb) hle)
          (Nat.not_succ_le_zero (rank y))
    | succ m ih =>
        intro x hx hle
        have ⟨y, hy, hb⟩ := hsucc x hx
        exact ih y hy
          (Nat.le_of_lt_succ (Nat.lt_of_lt_of_le (hdec x y hb) hle))
  exact key (rank x0) x0 h0 (Nat.le_refl _)

/-! ## The protocol theorem -/

/-- An acyclic wait-for graph admits no stream-ordered circular wait, at any
schedule length and on any instance. `T.maxIssues` appears nowhere. -/
theorem noCircularWaitOfAcyclicWaitGraph (T : Topology R C O S)
    (s : State R C O) (rank : C -> Nat)
    (hmatch : T.requireMatchedIssueOrder = true)
    (hstream : T.requireStreamOrder = true)
    (hacyclic : ∀ c d, blockedBy T s c d -> rank d < rank c)
    (hstuck : stuckB T s = true) (K : C -> Prop)
    (hclosed : WaitClosed T s K) (c0 : C) (h0 : K c0) : False := by
  refine noChainInAcyclicRelation (blockedBy T s) rank hacyclic K ?_ c0 h0
  intro c hcK
  have ⟨hmem, hfp, hag, hnext⟩ := hclosed c hcK
  have ⟨d, hd⟩ :=
    blockerOfStuckPending T s c hmatch hstream hstuck hmem hfp hag
  exact ⟨d, hnext d hd, hd⟩

/-- TLA's `DeadlockFreedom` for this state, under the acyclicity hypothesis. -/
theorem deadlockFreeOfAcyclicWaitGraph (T : Topology R C O S)
    (s : State R C O) (rank : C -> Nat)
    (hmatch : T.requireMatchedIssueOrder = true)
    (hstream : T.requireStreamOrder = true)
    (hacyclic : ∀ c d, blockedBy T s c d -> rank d < rank c) :
    ¬ ∃ (K : C -> Prop) (c0 : C),
        stuckB T s = true ∧ WaitClosed T s K ∧ K c0 := by
  intro ⟨K, c0, hstuck, hclosed, h0⟩
  exact noCircularWaitOfAcyclicWaitGraph T s rank hmatch hstream hacyclic
    hstuck K hclosed c0 h0

/-- Ticket 14's sentence, in one theorem: for all rank sets, communicator
assignments and issue sequences, if per-communicator issue order agreement is
required and the communicator wait-for graph is acyclic, then no rendezvous
mismatches and no stream-ordered circular wait.

WHY THE FIRST COMPONENT IS THE WHOLE OF `Safety` AND NOT JUST
`RendezvousOpAgreement`. That conjunct compares two `Option O`, so on a state
where a communicator is running but some member has not issued far enough on
it, both sides are `none` and it holds vacuously. `RendezvousMembership` is
what forces both sides to `some` -- see divergence 3 in ScoutBProtocol.lean --
so returning `RendezvousOpAgreement` alone would project away exactly the
conjunct that gives it content, and the theorem would read stronger than it is.
`Safety T s` follows from the same hypotheses at no cost, so it is what is
returned. `.1` of the result is still `RendezvousOpAgreement` for a caller that
wants only that, but the caller then also has `.2.1` to go with it. -/
theorem orderAgreementAndAcyclicWaitGraphExcludeBothHazards
    (T : Topology R C O S) (s : State R C O) (rank : C -> Nat)
    (hmatch : T.requireMatchedIssueOrder = true)
    (hstream : T.requireStreamOrder = true)
    (hacyclic : ∀ c d, blockedBy T s c d -> rank d < rank c)
    (hreach : Reachable T s) :
    Safety T s ∧
      ¬ ∃ (K : C -> Prop) (c0 : C),
          stuckB T s = true ∧ WaitClosed T s K ∧ K c0 :=
  ⟨safetyOfReachable T hmatch s hreach,
    deadlockFreeOfAcyclicWaitGraph T s rank hmatch hstream hacyclic⟩

end WaitGraph

/-! ## The negative: acyclicity is load-bearing, not decorative

Two ranks, two communicators, each communicator holding both ranks, one stream,
and the two ranks issuing the two communicators in OPPOSITE order. That is the
shape ScoutBModelDivergent.cfg reaches once communicator-site agreement is
relaxed, so `requireUniformProgramComms` is false here and every other guard is
on, exactly as in that configuration.

The witness satisfies every hypothesis of
`deadlockFreeOfAcyclicWaitGraph` except acyclicity: it is stuck, NCCL's
matching requirement is on, stream order is on, and
`cyclicWitnessIsWaitClosed` gives the closed set. `cyclicWitnessHasTwoCycle`
exhibits the 2-cycle and `cyclicWitnessAdmitsNoRankFunction` shows no rank
function can exist, so the hypothesis that fails is precisely the dropped one.

What is NOT claimed: that this state is reachable in this topology. Reachability
of an analogous state in the shipped 2x2 instance is TLC's result, under
ScoutBModelDivergent.cfg. The purpose here is to show the hypothesis carries
weight, which needs a state, not a trajectory. -/

def cyclicTopology : Topology Bool Bool Unit Unit where
  ranks := [false, true]
  comms := [false, true]
  member := fun _ _ => true
  opsAll := [()]
  admissibleOp := fun _ _ => true
  streamOf := fun _ => ()
  maxIssues := 2
  requireMatchedIssueOrder := true
  requireUniformProgramOps := true
  requireUniformProgramComms := false
  requireStreamOrder := true
  membersInRanks := by
    intro _ r _
    cases r with
    | false => exact List.Mem.head _
    | true => exact List.Mem.tail _ (List.Mem.head _)
  admissibleInOps := by
    intro _ op _
    cases op
    exact List.Mem.head _
  commsComplete := by
    intro c _ _
    cases c with
    | false => exact List.Mem.head _
    | true => exact List.Mem.tail _ (List.Mem.head _)

/-- Rank `false` issues communicator `false` then `true`; rank `true` issues
them the other way round. Nothing has completed, nothing is running. -/
def cyclicState : State Bool Bool Unit where
  issued := fun r =>
    if r then [Issue.mk true (), Issue.mk false ()]
    else [Issue.mk false (), Issue.mk true ()]
  running := fun _ => false
  doneOn := fun _ => 0

theorem cyclicWitnessIsStuck : stuckB cyclicTopology cyclicState = true := by
  decide

/-- Non-vacuity, in the spirit of ScoutBModelReach.cfg: the guards are not
false by construction. With both ranks issuing the two communicators in the
SAME order the first communicator CAN start and the state is not stuck, so
`cyclicWitnessIsStuck` reports a property of the opposite issue order rather
than of an unsatisfiable guard. -/
def sameOrderState : State Bool Bool Unit where
  issued := fun _ => [Issue.mk false (), Issue.mk true ()]
  running := fun _ => false
  doneOn := fun _ => 0

theorem waitGraphWitnessIsNotVacuous :
    startAllowedB cyclicTopology sameOrderState false = true ∧
      stuckB cyclicTopology sameOrderState = false := by
  decide

theorem cyclicWitnessIsWaitClosed :
    WaitClosed cyclicTopology cyclicState (fun _ => True) := by
  intro c _
  refine ⟨?_, ?_, ?_, fun _ _ => True.intro⟩
  · cases c with
    | false => exact List.Mem.head _
    | true => exact List.Mem.tail _ (List.Mem.head _)
  · cases c with
    | false => decide
    | true => decide
  · cases c with
    | false => decide
    | true => decide

theorem cyclicWitnessHasTwoCycle :
    blockedBy cyclicTopology cyclicState false true ∧
      blockedBy cyclicTopology cyclicState true false := by
  refine ⟨⟨rfl, true, 2, 1, Issue.mk false (), Issue.mk true (), ?_, ?_, ?_,
      ?_, ?_, ?_, ?_, ?_, ?_⟩,
    ⟨rfl, false, 2, 1, Issue.mk true (), Issue.mk false (), ?_, ?_, ?_, ?_, ?_,
      ?_, ?_, ?_, ?_⟩⟩
  all_goals first
    | rfl
    | decide

theorem cyclicWitnessAdmitsNoRankFunction :
    ¬ ∃ rank : Bool -> Nat,
        ∀ c d, blockedBy cyclicTopology cyclicState c d -> rank d < rank c := by
  intro ⟨rank, hdec⟩
  have ⟨h01, h10⟩ := cyclicWitnessHasTwoCycle
  exact absurd (Nat.lt_trans (hdec false true h01) (hdec true false h10))
    (Nat.lt_irrefl (rank true))

/-- Dropping acyclicity breaks the theorem. Lean refutes the weakened
statement, so the hypothesis is doing work. -/
theorem acyclicityIsLoadBearing :
    ¬ (∀ (T : Topology Bool Bool Unit Unit) (s : State Bool Bool Unit),
         T.requireMatchedIssueOrder = true -> T.requireStreamOrder = true ->
         ¬ ∃ (K : Bool -> Prop) (c0 : Bool),
             stuckB T s = true ∧ WaitClosed T s K ∧ K c0) := by
  intro h
  exact h cyclicTopology cyclicState rfl rfl
    ⟨fun _ => True, false, cyclicWitnessIsStuck, cyclicWitnessIsWaitClosed,
      True.intro⟩

#print axioms blockerOfStuckPending
#print axioms noChainInAcyclicRelation
#print axioms noCircularWaitOfAcyclicWaitGraph
#print axioms deadlockFreeOfAcyclicWaitGraph
#print axioms orderAgreementAndAcyclicWaitGraphExcludeBothHazards
#print axioms cyclicWitnessIsStuck
#print axioms waitGraphWitnessIsNotVacuous
#print axioms cyclicWitnessIsWaitClosed
#print axioms cyclicWitnessHasTwoCycle
#print axioms cyclicWitnessAdmitsNoRankFunction
#print axioms acyclicityIsLoadBearing

end Qwen3Formal.ScoutBProtocol
