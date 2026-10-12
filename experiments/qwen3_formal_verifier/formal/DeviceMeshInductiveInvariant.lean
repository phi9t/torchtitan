/-
DeviceMeshInductiveInvariant -- the three inductive-invariant obligations for
DeviceMeshModel.tla, discharged in Lean over ALL topologies, states and schedule
lengths.

WHY THIS EXISTS. TLC checks DeviceMeshModel.cfg exhaustively over 38321 distinct
states at MaxIssues = 2. The observed run is 108 collectives per rank. The
shipped claim was therefore a conjunction -- the guards are safe for schedules
up to the bound, AND separately the observed run satisfies the per-communicator
agreement property -- and that conjunction is weaker than "the protocol is
safe". This file closes that gap for the three state-predicate safety
invariants by an induction rather than an enumeration, so `maxIssues` appears
nowhere in the statement of `safetyOfReachable`.

THE THREE OBLIGATIONS, each its own theorem so that a missing one cannot
silently weaken the claim:

  initiation   Init T s -> Inv T s
  consecution  Inv T s -> Step T s s' -> Inv T s'
  sufficiency  Inv T s -> Safety T s

and the composition `invOfReachable` / `safetyOfReachable`.

WHICH SAFETY PROPERTY, AND WHY. `Inv` is `RendezvousMembership /\ CommFifo /\
RendezvousOpAgreement /\ TypeOK`. Those are four of the six invariants
DeviceMeshModel.cfg names. The two that are NOT here are `DeadlockFreedom` and
`StuckImpliesAllDone`, and the reason is stated rather than left to inference:

  - `StuckImpliesAllDone` CANNOT be made bound-free, and the reason is
    syntactic rather than a limitation of this proof: `AllDone` is defined by
    `Len(issued[r]) = MaxIssues`, so the property mentions the bound in its
    own statement. It HOLDS in DeviceMeshModel.cfg at MaxIssues = 2, which is a
    genuine result about that configuration; it is not a property there is an
    unbounded version of. DeviceMeshModel.tla explains separately why it is also
    the wrong thing to read as deadlock freedom wherever it fails.
  - `DeadlockFreedom == ~StuckByCircularWait` is not a state predicate over
    per-communicator arithmetic; it quantifies over the least fixpoint of the
    blocking relation. It is the hardest and most valuable of the six, and it
    is addressed in DeviceMeshWaitGraph.lean CONDITIONALLY: an acyclic wait-for
    graph implies no circular wait, for all topologies and states. Whether the
    graph is acyclic under a given set of guards is what TLC decides, and
    DeviceMeshModelDivergent.cfg shows it is not acyclic once communicator-site
    agreement is relaxed. Unconditional deadlock freedom for all schedule
    lengths is NOT proved here.

`TypeOK` is the cheap one and is carried only because it is free; the claim
does not rest on it.

WHAT THE HYPOTHESIS `requireMatchedIssueOrder = true` IS DOING.
`RendezvousOpAgreement` is not preserved by `Start` unless NCCL's own matching
requirement is on: that guard is exactly what makes a started rendezvous agree
on the operation. DeviceMeshModelUnguarded.cfg is the TLC negative for the same
fact, and it refutes `RendezvousOpAgreement` when the guard is dropped. So the
hypothesis is load-bearing and its necessity is checked, not assumed. None of
the other three `require*` flags is needed: the invariant survives with the
SPMD-program guards and the stream-order guard set either way, which is a
stronger statement than the configuration TLC checks.

NO STRENGTHENING CONJUNCT WAS NEEDED, but the conjunction itself is the
strengthening: `commFifoAloneIsNotInductive` exhibits a topology, a state and a
single `Complete` step that preserves `CommFifo` in the pre-state and breaks it
in the post-state. The witness state is unreachable -- it has a running
communicator nobody issued -- which is precisely why `RendezvousMembership`
has to travel with `CommFifo`.

No `simp`, no `omega`, nothing classical: see the AXIOM DISCIPLINE note in
DeviceMeshProtocol.lean.
-/
import DeviceMeshProtocol

namespace Qwen3Formal.DeviceMeshProtocol

section Proofs

variable {R C O S : Type} [DecidableEq R] [DecidableEq C] [DecidableEq O]
variable [DecidableEq S]

set_option linter.unusedSectionVars false

/-! ## Elementary facts about the `Issue` post-state -/

theorem issuedCandidateSelf (s : State R C O) (r : R) (e : Issue C O) :
    (issueCandidate s r e).issued r = s.issued r ++ [e] := ifPos rfl

theorem issuedCandidateOther (s : State R C O) (r : R) (e : Issue C O) (r' : R)
    (h : ¬ (r' = r)) : (issueCandidate s r e).issued r' = s.issued r' := ifNeg h

theorem commCountLeCandidate (s : State R C O) (r0 : R) (e : Issue C O) (r : R)
    (c : C) : commCount s r c <= commCount (issueCandidate s r0 e) r c := by
  cases h : decide (r = r0) with
  | true =>
      have hp : r = r0 := ofDecide h
      show countOn c (s.issued r)
        <= countOn c ((issueCandidate s r0 e).issued r)
      rw [hp, issuedCandidateSelf]
      exact countOn_le_append c (s.issued r0) [e]
  | false =>
      show countOn c (s.issued r)
        <= countOn c ((issueCandidate s r0 e).issued r)
      rw [issuedCandidateOther s r0 e r (notOfDecideFalse h)]
      exact Nat.le_refl _

theorem opAtOnCandidate (s : State R C O) (r0 : R) (e : Issue C O) (r : R)
    (c : C) (n : Nat) (h1 : 1 <= n) (h2 : n <= commCount s r c) :
    opAtOn c n ((issueCandidate s r0 e).issued r)
      = opAtOn c n (s.issued r) := by
  cases h : decide (r = r0) with
  | true =>
      have hp : r = r0 := ofDecide h
      have h2' : n <= countOn c (s.issued r0) := hp ▸ h2
      rw [hp, issuedCandidateSelf]
      exact opAtOn_append_of_le c (s.issued r0) n [e] h1 h2'
  | false =>
      rw [issuedCandidateOther s r0 e r (notOfDecideFalse h)]

/-! ## Reading the guards

Each lemma pulls exactly one conjunct out of a Bool-valued guard. `&&` is
left-associative, so the projections below mirror the order the conjuncts are
written in `issueAllowedB` and `startAllowedB`. -/

theorem issueAllowedMember {T : Topology R C O S} {s : State R C O} {r : R}
    {c : C} {op : O} (h : issueAllowedB T s r c op = true) :
    T.member c r = true :=
  andLeft (andLeft (andLeft (andLeft h)))

theorem issueAllowedOp {T : Topology R C O S} {s : State R C O} {r : R} {c : C}
    {op : O} (h : issueAllowedB T s r c op = true) :
    T.admissibleOp c op = true :=
  andRight (andLeft (andLeft (andLeft h)))

theorem startAllowedReady {T : Topology R C O S} {s : State R C O} {c : C}
    (h : startAllowedB T s c = true) (r : R) (hr : T.member c r = true) :
    memberReadyB T s c r = true :=
  impFromOr
    (allMem (andRight (andLeft h)) r (T.membersInRanks c r hr)) hr

theorem startAllowedFront {T : Topology R C O S} {s : State R C O} {c : C}
    (h : startAllowedB T s c = true) (r : R) (hr : T.member c r = true) :
    s.doneOn c + 1 <= commCount s r c :=
  ofDecide (andLeft (startAllowedReady h r hr))

theorem startAllowedAgree {T : Topology R C O S} {s : State R C O} {c : C}
    (hmatch : T.requireMatchedIssueOrder = true)
    (h : startAllowedB T s c = true) (r1 r2 : R)
    (h1 : T.member c r1 = true) (h2 : T.member c r2 = true) :
    opAtOn c (s.doneOn c + 1) (s.issued r1)
      = opAtOn c (s.doneOn c + 1) (s.issued r2) :=
  ofDecide
    (impFromOr
      (allMem
        (allMem (impFromOr (andRight h) hmatch) r1 (T.membersInRanks c r1 h1))
        r2 (T.membersInRanks c r2 h2))
      (andBoth h1 h2))

/-! ## Obligation 1 of 3: initiation -/

theorem initiation (T : Topology R C O S) (s : State R C O) (h : Init T s) :
    Inv T s := by
  have ⟨hiss, hrun, hdone⟩ := h
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro c hc
    exact Bool.noConfusion ((hrun c).symm.trans hc)
  · intro c r _
    rw [hdone c]
    exact Nat.zero_le _
  · intro c hc
    exact Bool.noConfusion ((hrun c).symm.trans hc)
  · intro r k e hk
    rw [hiss r] at hk
    exact absurdNoneSome (show (none : Option (Issue C O)) = some e from hk)

/-! ## Obligation 2 of 3: consecution -/

theorem consecution (T : Topology R C O S) (s s' : State R C O)
    (hmatch : T.requireMatchedIssueOrder = true) (hinv : Inv T s)
    (hstep : Step T s s') : Inv T s' := by
  have ⟨hmem, hfifo, hagree, htype⟩ := hinv
  cases hstep with
  | issue r0 c0 op0 hguard =>
      refine ⟨?_, ?_, ?_, ?_⟩
      · intro c hc r hr
        exact Nat.le_trans (hmem c hc r hr) (commCountLeCandidate s r0 _ r c)
      · intro c r hr
        exact Nat.le_trans (hfifo c r hr) (commCountLeCandidate s r0 _ r c)
      · intro c hc r1 r2 h1 h2
        show opAtOn c (s.doneOn c + 1)
              ((issueCandidate s r0 (Issue.mk c0 op0)).issued r1)
          = opAtOn c (s.doneOn c + 1)
              ((issueCandidate s r0 (Issue.mk c0 op0)).issued r2)
        rw [opAtOnCandidate s r0 _ r1 c (s.doneOn c + 1)
              (Nat.succ_le_succ (Nat.zero_le _)) (hmem c hc r1 h1),
            opAtOnCandidate s r0 _ r2 c (s.doneOn c + 1)
              (Nat.succ_le_succ (Nat.zero_le _)) (hmem c hc r2 h2)]
        exact hagree c hc r1 r2 h1 h2
      · intro r k e hk
        cases hrr : decide (r = r0) with
        | true =>
            have hp : r = r0 := ofDecide hrr
            rw [hp, issuedCandidateSelf] at hk
            cases atPos_append_singleton_inv hk with
            | inl hl => exact hp ▸ htype r0 k e hl
            | inr hr =>
                rw [hr]
                exact ⟨issueAllowedOp hguard, hp ▸ issueAllowedMember hguard⟩
        | false =>
            rw [issuedCandidateOther s r0 _ r (notOfDecideFalse hrr)] at hk
            exact htype r k e hk
  | start c0 hguard =>
      refine ⟨?_, ?_, ?_, ?_⟩
      · intro c hc r hr
        cases orCases hc with
        | inl hold => exact hmem c hold r hr
        | inr hnew =>
            have hp : c = c0 := ofDecide hnew
            exact hp ▸ startAllowedFront hguard r (hp ▸ hr)
      · exact hfifo
      · intro c hc r1 r2 h1 h2
        cases orCases hc with
        | inl hold => exact hagree c hold r1 r2 h1 h2
        | inr hnew =>
            have hp : c = c0 := ofDecide hnew
            exact hp ▸ startAllowedAgree hmatch hguard r1 r2 (hp ▸ h1) (hp ▸ h2)
      · exact htype
  | complete c0 hrun =>
      refine ⟨?_, ?_, ?_, ?_⟩
      · intro c hc r hr
        have hne : ¬ (c = c0) := notOfDecideFalse (notTrue (andRight hc))
        show (if c = c0 then s.doneOn c + 1 else s.doneOn c) + 1
          <= commCount s r c
        rw [ifNeg hne]
        exact hmem c (andLeft hc) r hr
      · intro c r hr
        cases hcc : decide (c = c0) with
        | true =>
            have hp : c = c0 := ofDecide hcc
            show (if c = c0 then s.doneOn c + 1 else s.doneOn c)
              <= commCount s r c
            rw [ifPos hp]
            exact hmem c (hp ▸ hrun) r hr
        | false =>
            show (if c = c0 then s.doneOn c + 1 else s.doneOn c)
              <= commCount s r c
            rw [ifNeg (notOfDecideFalse hcc)]
            exact hfifo c r hr
      · intro c hc r1 r2 h1 h2
        have hne : ¬ (c = c0) := notOfDecideFalse (notTrue (andRight hc))
        show opAtOn c ((if c = c0 then s.doneOn c + 1 else s.doneOn c) + 1)
              (s.issued r1)
          = opAtOn c ((if c = c0 then s.doneOn c + 1 else s.doneOn c) + 1)
              (s.issued r2)
        rw [ifNeg hne]
        exact hagree c (andLeft hc) r1 r2 h1 h2
      · exact htype
  | terminated _ => exact ⟨hmem, hfifo, hagree, htype⟩

/-! ## Obligation 3 of 3: sufficiency -/

theorem sufficiency (T : Topology R C O S) (s : State R C O) (h : Inv T s) :
    Safety T s :=
  ⟨h.2.2.1, h.1, h.2.1, h.2.2.2⟩

/-! ## Composition: safety at every schedule length

Note what does NOT appear in the statement: `T.maxIssues`. The result holds for
every topology, including ones whose `maxIssues` is 108 or larger, and for
every reachable state of each. -/

theorem invOfReachable (T : Topology R C O S)
    (hmatch : T.requireMatchedIssueOrder = true) (s : State R C O)
    (h : Reachable T s) : Inv T s := by
  induction h with
  | init s0 hi => exact initiation T s0 hi
  | step s0 s1 _ hst ih => exact consecution T s0 s1 hmatch ih hst

theorem safetyOfReachable (T : Topology R C O S)
    (hmatch : T.requireMatchedIssueOrder = true) (s : State R C O)
    (h : Reachable T s) : Safety T s :=
  sufficiency T s (invOfReachable T hmatch s h)

end Proofs

/-! ## The conjunction is the strengthening

A negative, so that `Inv` being a plain conjunction is not mistaken for "any
one of these is inductive". One `Complete` step from a state satisfying
`CommFifo` lands in a state that does not. The witness has a running
communicator that nobody issued, so it violates `RendezvousMembership`; that
is exactly the conjunct whose absence lets `CommFifo` break. -/

def unitTopology : Topology Unit Unit Unit Unit where
  ranks := [()]
  comms := [()]
  member := fun _ _ => true
  opsAll := [()]
  admissibleOp := fun _ _ => true
  streamOf := fun _ => ()
  maxIssues := 1
  requireMatchedIssueOrder := true
  requireUniformProgramOps := true
  requireUniformProgramComms := true
  requireStreamOrder := true
  membersInRanks := by
    intro _ r _
    cases r
    exact List.Mem.head _
  admissibleInOps := by
    intro _ op _
    cases op
    exact List.Mem.head _
  commsComplete := by
    intro c _ _
    cases c
    exact List.Mem.head _

def runningWithNoIssues : State Unit Unit Unit where
  issued := fun _ => []
  running := fun _ => true
  doneOn := fun _ => 0

theorem commFifoAloneIsNotInductive :
    ∃ (T : Topology Unit Unit Unit Unit) (s s' : State Unit Unit Unit),
      CommFifo T s ∧ Step T s s' ∧ ¬ CommFifo T s' := by
  refine ⟨unitTopology, runningWithNoIssues,
    completeState runningWithNoIssues (), ?_, ?_, ?_⟩
  · intro c r _
    exact Nat.zero_le _
  · exact Step.complete runningWithNoIssues () rfl
  · intro h
    exact Bool.noConfusion (toDecide (h () () rfl))

#print axioms initiation
#print axioms consecution
#print axioms sufficiency
#print axioms invOfReachable
#print axioms safetyOfReachable
#print axioms commFifoAloneIsNotInductive

end Qwen3Formal.DeviceMeshProtocol
