/-
ScoutBProtocol -- a Lean 4 encoding of the ScoutBModel.tla protocol, with the
guards computable and the safety invariants stated as propositions.

WHAT THIS FILE IS FOR, AND WHAT IT IS NOT.

Every other Lean module in this package evaluates a Bool-valued predicate
against literal trace data and closes it with `rfl` or `decide`. Those are
kernel-checked EVALUATIONS of the observed run: genuine, decidable arithmetic
on constants, and silent about any other trace. This file is different in kind.
It carries no trace data at all. It encodes the abstract protocol of
ScoutBModel.tla and states theorems universally quantified over topologies,
states and schedules, so that nothing in its conclusions mentions MaxIssues.

The theorems live in ScoutBInductiveInvariant.lean (the three inductive-
invariant obligations) and ScoutBWaitGraph.lean (the wait-graph result and its
load-bearing-hypothesis negative). This file is the encoding plus the
elementary lemmas they need.

-----------------------------------------------------------------------------
ENCODING FIDELITY. THIS IS A SECOND REVIEWABLE CLAIM.

There is no mechanical link between this file and ScoutBModel.tla. A Lean
proof about a different model than the one TLC checks is worth nothing, so the
correspondence is written out below definition by definition and a reviewer is
expected to check it by inspection. Where the encoding is not a literal
transcription the divergence is named, with the direction it moves the claim.

  TLA+ (ScoutBModel.tla)        Lean (this file)
  ----------------------------  --------------------------------------------
  Ranks                         Topology.ranks : List R
  CommIds                       Topology.comms : List C
  CommMembers[c]                Topology.member c : R -> Bool
  CommOps[c]                    Topology.admissibleOp c : O -> Bool
  Ops                           Topology.opsAll : List O
  StreamOfIssue(e)              Topology.streamOf : Issue C O -> S
  MaxIssues                     Topology.maxIssues : Nat
  RequireMatchedIssueOrder      Topology.requireMatchedIssueOrder : Bool
  RequireUniformProgramOps      Topology.requireUniformProgramOps : Bool
  RequireUniformProgramComms    Topology.requireUniformProgramComms : Bool
  RequireStreamOrder            Topology.requireStreamOrder : Bool

  issued                        State.issued : R -> List (Issue C O)
  running                       State.running : C -> Bool
  doneOn                        State.doneOn : C -> Nat

  issued[r][k]                  atPos (s.issued r) k, 1-based, Option-valued
  Len(issued[r])                (s.issued r).length
  CommCount(r, c)               commCount s r c = countOn c (s.issued r)
  CommPos(r, k)                 commPos (s.issued r) k
  Front(c)                      front s c = s.doneOn c + 1
  Completed(r, k)               completedB s r k
  IdxOf(r, c, n)                idxOfOn c n (s.issued r)
  OpAt(r, c, n)                 opAtOn c n (s.issued r)
  AtStreamHead(r, k)            atStreamHeadB T s r k
  MemberReady(c, r)             memberReadyB T s c r
  FullyPending(c)               fullyPendingB T s c
  OpsAgreeAtFront(c)            opsAgreeAtFrontB T s c
  SameSite(c1, c2)              sameSiteB T c1 c2
  UniformProgramOpsOK(f)        uniformOpsOKB T f
  UniformProgramCommsOK(f)      uniformCommsOKB T f
  IssueCandidate(r, c, op)      issueCandidate s r (Issue.mk c op)
  IssueAllowed(r, c, op)        issueAllowedB T s r c op
  StartAllowed(c)               startAllowedB T s c
  AllDone                       allDoneB T s
  Init                          Init T s
  Issue / Start / Complete /    the four constructors of Step
    Terminated
  Next                          Step T s s'
  Spec (safety part)            Reachable T s
  TypeOK                        TypeOK T s          (see divergence 7)
  RendezvousOpAgreement         RendezvousOpAgreement T s
  RendezvousMembership          RendezvousMembership T s
  CommFifo                      CommFifo T s
  Stuck                         stuckB T s
  BlockedBy(c)                  blockedBy T s c d   (ScoutBWaitGraph.lean)

NAMED DIVERGENCES, each with the direction it moves the claim.

1. SETS AS CHARACTERISTIC FUNCTIONS. `running \subseteq CommIds` becomes
   `running : C -> Bool`, and `\A r \in CommMembers[c] : P(r)` becomes
   `\A r : member c r = true -> P(r)`. Equivalent. The Bool-valued form is
   what makes the guards computable, which is what lets the wait-graph proof
   extract a witness from a failed guard without any classical axiom -- see
   AXIOM DISCIPLINE below.

2. FINITENESS. TLA's Ranks and CommIds are finite sets; here they are Lists,
   and three proof fields tie the types back to them:
   `Topology.membersInRanks` requires every member of a communicator to appear
   in `ranks`, `Topology.admissibleInOps` requires every admissible operation
   to appear in `opsAll`, and `Topology.commsComplete` requires every
   communicator with a member to appear in `comms`. Bounded quantifiers in the
   guards run over `ranks`, `comms` and `opsAll` and are guarded by `member` or
   `admissibleOp`, which is the same quantification given those fields.
   Instantiating a Topology therefore requires discharging all three, so a
   `comms` list that omitted a live communicator -- which would make `stuckB`
   and `allDoneB` quantify over too little -- is rejected at construction
   rather than resting on an unstated side condition.

3. PARTIAL FUNCTIONS. TLA's `issued[r][k]`, `IdxOf` (a CHOOSE) and `OpAt` are
   undefined outside their domains. Here they are Option-valued and total, and
   out-of-domain reads are `none`.

   FOR THE GUARDS the direction is uniform. `Completed(r, k)` and
   `AtStreamHead(r, k)` are `true` when k is outside the domain, which is where
   TLA never evaluates them, and `OpsAgreeAtFront(c)` compares two Options so
   it is satisfied where TLA is undefined. That makes a guard EASIER to satisfy,
   hence the reachable set LARGER, hence the invariant claim STRONGER.
   `idxOfOn_isSome_of_le` proves the index TLA chooses does exist whenever the
   guard preceding its use holds, so the `none` branch of `memberReadyB` is
   unreachable under that guard.

   FOR `RendezvousOpAgreement` THE DIRECTION IS THE OTHER WAY, and the blanket
   argument above does NOT cover it. It is an invariant, not a guard, and it
   compares two Options. A state with `running c = true` in which some member
   has issued fewer than `doneOn c + 1` collectives on `c` makes both sides
   `none`, so that conjunct holds vacuously -- weaker than a strict reading of
   TLA's, where `OpAt` would be an undefined term rather than a satisfied
   equation. What rescues it is the CONJUNCTION: both `Inv` and `Safety` carry
   `RendezvousMembership`, which gives every member `doneOn c + 1 <=
   commCount s r c`, and `idxOfOn_isSome_of_le` then forces
   `idxOfOn c (doneOn c + 1) (issued r)` to be `some k` while `idxOfOn_spec`
   forces `opAtOn` to `some` of the operation at that k. So on every state
   satisfying `Inv`, both sides of the equation are `some` and the conjunct has
   its strict content. This is why `Safety` conjoins the two and why
   `orderAgreementAndAcyclicWaitGraphExcludeBothHazards` returns the whole of
   `Safety T s` rather than projecting out `RendezvousOpAgreement`: projecting
   it out would discard exactly the conjunct that makes it non-vacuous.

4. TLA's `IdxOf(r, c, n) == CHOOSE k \in DOMAIN issued[r] : issued[r][k].comm =
   c /\ CommPos(r, k) = n` is encoded as the recursive `idxOfOn`, which returns
   the position of the n-th issue of `c`. `idxOfOn_spec` proves the encoded
   index satisfies the first two conjuncts of TLA's CHOOSE predicate and that
   `opAtOn` reads exactly the operation at that index, so `OpAt` and `opAtOn`
   agree. The third conjunct, `CommPos(r, k) = n`, is checkable by inspection
   of `idxOfOn` and `commPos` and is not proved here.

   Note this makes the correspondence FORCED rather than merely compatible:
   `CommPos(r, k)` counts occurrences of `c` in the first k positions, so it
   strictly increases along the positions at which `c` is issued. The CHOOSE
   predicate therefore has at most one solution, and where a solution exists
   TLA's CHOOSE must return `idxOfOn`'s answer rather than some other index
   satisfying the same predicate.

5. `\A j \in 1..(k-1)` becomes `allLt k`, which ranges over `0..(k-1)`. Index 0
   is outside every sequence, so `atPos l 0 = none` and the extra conjunct is
   `true` by definition rather than by assumption.

6. `Next`'s Issue disjunct quantifies `op \in Ops`; `Step.issue` quantifies
   `op : O` and leaves `admissibleOp c op` to the guard, exactly as
   `IssueAllowed` does. Same transitions.

7. TypeOK is encoded WITHOUT two of its conjuncts. `running \subseteq CommIds`
   is discharged by Lean's types, since `running : C -> Bool`. The range bound
   `doneOn \in [CommIds -> 0..MaxIssues]` is dropped deliberately: it is the
   ONLY part of TypeOK that mentions MaxIssues, and carrying it would put the
   bound back into the very claim this file exists to remove it from.
   `issued[r][k].comm \in CommIds` is NOT discharged by the types either, since
   `C` is any type and `comms` is a list over it; that conjunct is simply
   dropped, and `commsComplete` is what ties `C`'s inhabitants back to `comms`
   where it matters. What `TypeOK` keeps is the part with content: every
   recorded issue is an admissible operation on a communicator the issuing rank
   belongs to.

8. `blockedBy` differs from `d \in BlockedBy(c)` in two places, and fidelity
   rests on both differences being INERT where the theorem uses the relation --
   not on any direction-of-strength argument. Enlarging the wait relation makes
   `hacyclic` harder to supply AND makes `WaitClosed` harder to satisfy, so it
   weakens the theorem on both sides; that inference is not available here and
   is not used.

   `BlockingPairs(c)` is guarded by `FullyPending(c)`, returning the empty set
   without it, and `blockedBy` has no such guard. INERT on the chain:
   `WaitClosed` requires `fullyPendingB` of every element it contains, so TLA's
   empty-set branch never fires where the proof evaluates the relation, and
   `blockerOfStuckPending` is only ever applied to a chain element.

   `BlockingPairs(c)` also bounds its second component by `1..MaxIssues` as well
   as by `DOMAIN issued[p[1]]`, and the encoding keeps only the domain
   condition. INERT on every reachable state: `issueAllowedB` requires
   `(s.issued r).length < T.maxIssues` before appending, so no sequence ever
   exceeds `maxIssues` and no domain index exceeds it either. That is what
   `ModelBounded` records on the TLA+ side.

   Separately, `deadlockFreeOfAcyclicWaitGraph` carries no `Reachable`
   hypothesis, so as an intermediate theorem it is not in bijection with TLA's
   `DeadlockFreedom`, which is checked over reachable states only. It is the
   stronger statement -- it holds of unreachable states too -- and
   `orderAgreementAndAcyclicWaitGraphExcludeBothHazards` is the one that adds
   `Reachable` and therefore the one to read against TLC's result.

9. Liveness is not encoded. `EveryIssuedCollectiveCompletes`, `LiveSpec` and
   `Fairness` have no counterpart here, and nothing in this file claims
   anything about them. Liveness remains a TLC result at MaxIssues = 2.

10. `Spec == Init /\ [][Next]_vars` admits a stuttering step from any state,
    because `[N]_vars` is `N \/ UNCHANGED vars`. `Reachable` has no stuttering
    constructor other than `Step.terminated`. A stuttering step leaves the
    state unchanged, so it adds no reachable state and the two notions of
    reachability have the same image. Omitting it changes nothing about the
    invariant claim.

-----------------------------------------------------------------------------
AXIOM DISCIPLINE. `#print axioms` must report no axioms at all for every
theorem, which the runner enforces. That rules out more than `sorry`:

  - `omega` depends on propext and Quot.sound;
  - `simp` depends on propext;
  - anything classical (`by_cases` on an undecidable proposition,
    `Classical.em`, `Or.elim (Classical.em _)`) depends on Classical.choice;
  - most `List` lemmas in core are proved by `simp`, so `List.length_append`,
    `List.mem_cons` and `List.getElem?_cons_succ` all drag in propext.

So the list helpers here are defined and proved from scratch, arithmetic goes
through the axiom-free `Nat` lemmas, and every guard is Bool-valued so that a
failed guard yields a witness by case analysis on `Bool` rather than by
excluded middle. Do not reach for `simp` or `omega` in this file or in the two
theorem files; the runner will reject the result.
-/

namespace Qwen3Formal.ScoutBProtocol

/-! ## Prelude: if, decide, Bool and Nat helpers -/

theorem ifPos {p : Prop} [inst : Decidable p] {A : Type} {a b : A} (h : p) :
    (if p then a else b) = a := by
  cases inst with
  | isTrue _ => rfl
  | isFalse hn => exact absurd h hn

theorem ifNeg {p : Prop} [inst : Decidable p] {A : Type} {a b : A} (h : ¬ p) :
    (if p then a else b) = b := by
  cases inst with
  | isTrue hp => exact absurd hp h
  | isFalse _ => rfl

theorem ofDecide {p : Prop} [inst : Decidable p] (h : decide p = true) : p := by
  cases inst with
  | isTrue hp => exact hp
  | isFalse _ => exact Bool.noConfusion h

theorem toDecide {p : Prop} [inst : Decidable p] (h : p) : decide p = true := by
  cases inst with
  | isTrue _ => rfl
  | isFalse hn => exact absurd h hn

theorem absurdNoneSome {A : Type} {a : A} {P : Prop}
    (h : (none : Option A) = some a) : P := by
  cases h

theorem andLeft {a b : Bool} (h : (a && b) = true) : a = true := by
  cases a with
  | false => exact Bool.noConfusion h
  | true => rfl

theorem andRight {a b : Bool} (h : (a && b) = true) : b = true := by
  cases a with
  | false => exact Bool.noConfusion h
  | true => exact h

theorem andBoth {a b : Bool} (ha : a = true) (hb : b = true) :
    (a && b) = true := by
  rw [ha, hb]; rfl

theorem orCases {a b : Bool} (h : (a || b) = true) : a = true ∨ b = true := by
  cases a with
  | false => exact Or.inr h
  | true => exact Or.inl rfl

theorem impFromOr {a b : Bool} (h : ((!a) || b) = true) (ha : a = true) :
    b = true := by
  rw [ha] at h; exact h

theorem falseOfNotTrue {b : Bool} (h : ¬ (b = true)) : b = false := by
  cases b with
  | false => rfl
  | true => exact absurd rfl h

theorem andFalseCases {a b : Bool} (h : (a && b) = false) :
    a = false ∨ b = false := by
  cases a with
  | false => exact Or.inl rfl
  | true => exact Or.inr h

theorem orFalseLeft {a b : Bool} (h : (a || b) = false) : a = false := by
  cases a with
  | false => rfl
  | true => exact Bool.noConfusion h

theorem orFalseRight {a b : Bool} (h : (a || b) = false) : b = false := by
  cases a with
  | false => exact h
  | true => exact Bool.noConfusion h

theorem notTrue {a : Bool} (h : (!a) = true) : a = false := by
  cases a with
  | false => rfl
  | true => exact Bool.noConfusion h

theorem notOfDecideFalse {p : Prop} [Decidable p] (h : decide p = false) :
    ¬ p :=
  fun hp => Bool.noConfusion (h.symm.trans (toDecide hp))

theorem notFalse {a : Bool} (h : (!a) = false) : a = true := by
  cases a with
  | false => exact Bool.noConfusion h
  | true => rfl

theorem zeroAdd (n : Nat) : 0 + n = n := by
  induction n with
  | zero => rfl
  | succ k ih => exact congrArg Nat.succ ih

/-- `a + b + 1 = a + 1 + b`, proved without `omega`. -/
theorem addSuccComm (a b : Nat) : a + b + 1 = a + 1 + b := by
  induction b with
  | zero => rfl
  | succ n ih => exact congrArg Nat.succ ih

/-! ## Prelude: lists, indexed from one as TLA+ sequences are -/

/-- `atPos l k` is TLA's `l[k]`: 1-based, `none` outside the domain. -/
def atPos {A : Type} : List A -> Nat -> Option A
  | [], _ => none
  | _ :: _, 0 => none
  | e :: _, 1 => some e
  | _ :: t, n + 2 => atPos t (n + 1)

theorem atPos_zero {A : Type} (l : List A) : atPos l 0 = none := by
  cases l with
  | nil => rfl
  | cons _ _ => rfl

theorem atPos_cons_one {A : Type} (e : A) (t : List A) :
    atPos (e :: t) 1 = some e := rfl

theorem atPos_cons_succ {A : Type} (e : A) (t : List A) (n : Nat) :
    atPos (e :: t) (n + 2) = atPos t (n + 1) := rfl

theorem atPos_cons_of_pos {A : Type} (e : A) (t : List A) (j : Nat)
    (h : 1 <= j) : atPos (e :: t) (j + 1) = atPos t j := by
  match j with
  | 0 => exact absurd h (Nat.not_succ_le_zero 0)
  | _ + 1 => rfl

theorem atPos_append {A : Type} {l : List A} {k : Nat} {a : A} :
    ∀ (m : List A), atPos l k = some a -> atPos (l ++ m) k = some a := by
  induction l generalizing k with
  | nil =>
      intro m h
      exact absurdNoneSome (show (none : Option A) = some a from h)
  | cons b t ih =>
      intro m h
      match k with
      | 0 => exact absurdNoneSome (show (none : Option A) = some a from h)
      | 1 =>
          have hb : some b = some a := h
          rw [Option.some.inj hb]; rfl
      | n + 2 => exact ih m h

theorem atPos_append_singleton_inv {A : Type} {l : List A} {e a : A} {k : Nat} :
    atPos (l ++ [e]) k = some a -> atPos l k = some a ∨ a = e := by
  induction l generalizing k with
  | nil =>
      intro h
      match k with
      | 0 => exact absurdNoneSome (show (none : Option A) = some a from h)
      | 1 => exact Or.inr (Option.some.inj h).symm
      | _ + 2 => exact absurdNoneSome (show (none : Option A) = some a from h)
  | cons b t ih =>
      intro h
      match k with
      | 0 => exact absurdNoneSome (show (none : Option A) = some a from h)
      | 1 => exact Or.inl h
      | n + 2 =>
          have h' : atPos (t ++ [e]) (n + 1) = some a := h
          exact (ih h').elim (fun hl => Or.inl hl) (fun hr => Or.inr hr)

/-- `takeN l k` is TLA's `SubSeq(l, 1, k)`. -/
def takeN {A : Type} : List A -> Nat -> List A
  | [], _ => []
  | _, 0 => []
  | e :: t, n + 1 => e :: takeN t n

/-- Bounded conjunction over `0..n-1`. Index 0 is outside every sequence, so
this is TLA's `\A j \in 1..n-1` whenever the body is `true` at 0. -/
def allLt : Nat -> (Nat -> Bool) -> Bool
  | 0, _ => true
  | n + 1, f => allLt n f && f n

theorem allLt_false_witness :
    ∀ (n : Nat) (f : Nat -> Bool),
      allLt n f = false -> ∃ j, j < n ∧ f j = false := by
  intro n
  induction n with
  | zero => intro f h; exact Bool.noConfusion h
  | succ m ih =>
      intro f h
      cases hm : allLt m f with
      | false =>
          have ⟨j, hj, hfj⟩ := ih f hm
          exact ⟨j, Nat.lt_trans hj (Nat.lt_succ_self m), hfj⟩
      | true =>
          have h' : (true && f m) = false := by rw [← hm]; exact h
          exact ⟨m, Nat.lt_succ_self m, h'⟩

theorem allMem {A : Type} {f : A -> Bool} :
    ∀ {l : List A}, l.all f = true -> ∀ a, a ∈ l -> f a = true := by
  intro l
  induction l with
  | nil => intro _ a ha; cases ha
  | cons b t ih =>
      intro h a ha
      have h' : (f b && t.all f) = true := h
      cases ha with
      | head => exact andLeft h'
      | tail _ hm => exact ih (andRight h') a hm

theorem allFalseWitness {A : Type} {f : A -> Bool} :
    ∀ {l : List A}, l.all f = false -> ∃ a, a ∈ l ∧ f a = false := by
  intro l
  induction l with
  | nil => intro h; exact Bool.noConfusion h
  | cons b t ih =>
      intro h
      have h' : (f b && t.all f) = false := h
      cases andFalseCases h' with
      | inl hb => exact ⟨b, List.Mem.head _, hb⟩
      | inr ht =>
          have ⟨a, ham, hfa⟩ := ih ht
          exact ⟨a, List.Mem.tail _ ham, hfa⟩

/-! ## Issues, counting, indexing -/

/-- One collective issue: TLA's `[comm |-> c, op |-> op]`. -/
structure Issue (C O : Type) where
  comm : C
  op : O
deriving DecidableEq

section Defs

variable {R C O S : Type}

/-- TLA's `CommCount(r, c)`, as a fold over the issue sequence. -/
def countOn [DecidableEq C] (c : C) : List (Issue C O) -> Nat
  | [] => 0
  | e :: t => if e.comm = c then countOn c t + 1 else countOn c t

theorem countOn_append [DecidableEq C] (c : C) :
    ∀ (l m : List (Issue C O)),
      countOn c (l ++ m) = countOn c l + countOn c m := by
  intro l
  induction l with
  | nil => intro m; exact (zeroAdd (countOn c m)).symm
  | cons e t ih =>
      intro m
      cases he : decide (e.comm = c) with
      | true =>
          have hp : e.comm = c := ofDecide he
          show (if e.comm = c then countOn c (t ++ m) + 1
                  else countOn c (t ++ m))
            = (if e.comm = c then countOn c t + 1 else countOn c t)
                + countOn c m
          rw [ifPos hp, ifPos hp, ih m]
          exact (addSuccComm (countOn c t) (countOn c m)).symm ▸ rfl
      | false =>
          have hp : ¬ (e.comm = c) := fun hc =>
            Bool.noConfusion (he ▸ toDecide hc)
          show (if e.comm = c then countOn c (t ++ m) + 1
                  else countOn c (t ++ m))
            = (if e.comm = c then countOn c t + 1 else countOn c t)
                + countOn c m
          rw [ifNeg hp, ifNeg hp, ih m]

theorem countOn_le_append [DecidableEq C] (c : C) (l m : List (Issue C O)) :
    countOn c l ≤ countOn c (l ++ m) := by
  rw [countOn_append]; exact Nat.le_add_right _ _

/-- TLA's `OpAt(r, c, n)`: the operation of the n-th issue of `c`. -/
def opAtOn [DecidableEq C] (c : C) (n : Nat) : List (Issue C O) -> Option O
  | [] => none
  | e :: t =>
      if e.comm = c then (if n = 1 then some e.op else opAtOn c (n - 1) t)
      else opAtOn c n t

/-- TLA's `IdxOf(r, c, n)`: the 1-based position of the n-th issue of `c`. -/
def idxOfOn [DecidableEq C] (c : C) (n : Nat) : List (Issue C O) -> Option Nat
  | [] => none
  | e :: t =>
      if e.comm = c then
        (if n = 1 then some 1
         else match idxOfOn c (n - 1) t with
              | none => none
              | some k => some (k + 1))
      else
        match idxOfOn c n t with
        | none => none
        | some k => some (k + 1)

/-- TLA's `CommPos(r, k)`: how many of the first k issues share k's
communicator. -/
def commPos [DecidableEq C] (l : List (Issue C O)) (k : Nat) : Nat :=
  match atPos l k with
  | none => 0
  | some e => countOn e.comm (takeN l k)

end Defs

/-! ## Appending an issue leaves earlier reads alone

This is the lemma that makes `RendezvousOpAgreement` survive an `Issue` step:
a rank appending to its own schedule cannot change which operation it already
holds at a position it has already reached. -/

section Reads

variable {C O : Type} [DecidableEq C]

theorem opAtOn_append_of_le (c : C) :
    ∀ (l : List (Issue C O)) (n : Nat) (m : List (Issue C O)),
      1 ≤ n -> n ≤ countOn c l -> opAtOn c n (l ++ m) = opAtOn c n l := by
  intro l
  induction l with
  | nil =>
      intro n m h1 h2
      exact absurd (Nat.le_trans h1 h2) (Nat.not_succ_le_zero 0)
  | cons e t ih =>
      intro n m h1 h2
      cases he : decide (e.comm = c) with
      | true =>
          have hp : e.comm = c := ofDecide he
          have hcnt : countOn c (e :: t) = countOn c t + 1 := ifPos hp
          match n with
          | 0 => exact absurd h1 (Nat.not_succ_le_zero 0)
          | 1 =>
              show (if e.comm = c then (if (1 : Nat) = 1 then some e.op
                      else opAtOn c 0 (t ++ m)) else opAtOn c 1 (t ++ m))
                = (if e.comm = c then (if (1 : Nat) = 1 then some e.op
                      else opAtOn c 0 t) else opAtOn c 1 t)
              rw [ifPos hp, ifPos hp]
              rfl
          | j + 2 =>
              have hne : ¬ ((j + 2 : Nat) = 1) := fun hc => Nat.noConfusion
                (Nat.succ.inj hc)
              have hb : j + 1 ≤ countOn c t := by
                have : j + 2 ≤ countOn c t + 1 := hcnt ▸ h2
                exact Nat.le_of_succ_le_succ this
              show (if e.comm = c then (if (j + 2 : Nat) = 1 then some e.op
                      else opAtOn c (j + 1) (t ++ m))
                    else opAtOn c (j + 2) (t ++ m))
                = (if e.comm = c then (if (j + 2 : Nat) = 1 then some e.op
                      else opAtOn c (j + 1) t) else opAtOn c (j + 2) t)
              rw [ifPos hp, ifPos hp, ifNeg hne, ifNeg hne]
              exact ih (j + 1) m (Nat.succ_le_succ (Nat.zero_le j)) hb
      | false =>
          have hp : ¬ (e.comm = c) := fun hc =>
            Bool.noConfusion (he ▸ toDecide hc)
          have hcnt : countOn c (e :: t) = countOn c t := ifNeg hp
          show (if e.comm = c then (if n = 1 then some e.op
                  else opAtOn c (n - 1) (t ++ m)) else opAtOn c n (t ++ m))
            = (if e.comm = c then (if n = 1 then some e.op
                  else opAtOn c (n - 1) t) else opAtOn c n t)
          rw [ifNeg hp, ifNeg hp]
          exact ih n m h1 (hcnt ▸ h2)

theorem idxOfOn_pos (c : C) :
    ∀ (l : List (Issue C O)) (n k : Nat), idxOfOn c n l = some k -> 1 <= k := by
  intro l
  induction l with
  | nil =>
      intro n k h
      exact absurdNoneSome (show (none : Option Nat) = some k from h)
  | cons e t ih =>
      intro n k h
      have h0 : (if e.comm = c then
                   (if n = 1 then some 1
                    else match idxOfOn c (n - 1) t with
                         | none => none
                         | some j => some (j + 1))
                 else match idxOfOn c n t with
                      | none => none
                      | some j => some (j + 1)) = some k := h
      cases he : decide (e.comm = c) with
      | true =>
          rw [ifPos (ofDecide he)] at h0
          cases hn : decide (n = 1) with
          | true =>
              rw [ifPos (ofDecide hn)] at h0
              exact Nat.le_of_eq (Option.some.inj h0)
          | false =>
              rw [ifNeg (fun hc => Bool.noConfusion (hn ▸ toDecide hc))] at h0
              match hj : idxOfOn c (n - 1) t with
              | none =>
                  rw [hj] at h0
                  exact absurdNoneSome
                    (show (none : Option Nat) = some k from h0)
              | some j =>
                  rw [hj] at h0
                  have hk : j + 1 = k :=
                    Option.some.inj (show some (j + 1) = some k from h0)
                  exact Nat.le_trans (Nat.succ_le_succ (Nat.zero_le j))
                    (Nat.le_of_eq hk)
      | false =>
          rw [ifNeg (fun hc => Bool.noConfusion (he ▸ toDecide hc))] at h0
          match hj : idxOfOn c n t with
          | none =>
              rw [hj] at h0
              exact absurdNoneSome (show (none : Option Nat) = some k from h0)
          | some j =>
              rw [hj] at h0
              have hk : j + 1 = k :=
                Option.some.inj (show some (j + 1) = some k from h0)
              exact Nat.le_trans (Nat.succ_le_succ (Nat.zero_le j))
                (Nat.le_of_eq hk)

/-- The index TLA's CHOOSE picks does exist whenever the guard that precedes
every use of it holds. This is what makes divergence 3 of the header safe: the
`none` branch of `memberReadyB` is unreachable under its own first conjunct. -/
theorem idxOfOn_isSome_of_le (c : C) :
    ∀ (l : List (Issue C O)) (n : Nat), 1 ≤ n -> n ≤ countOn c l ->
      ∃ k, idxOfOn c n l = some k := by
  intro l
  induction l with
  | nil =>
      intro n h1 h2
      exact absurd (Nat.le_trans h1 h2) (Nat.not_succ_le_zero 0)
  | cons e t ih =>
      intro n h1 h2
      cases he : decide (e.comm = c) with
      | true =>
          have hp : e.comm = c := ofDecide he
          have hcnt : countOn c (e :: t) = countOn c t + 1 := ifPos hp
          match n with
          | 0 => exact absurd h1 (Nat.not_succ_le_zero 0)
          | 1 =>
              refine ⟨1, ?_⟩
              show (if e.comm = c then (if (1 : Nat) = 1 then some 1
                      else match idxOfOn c 0 t with
                           | none => none | some k => some (k + 1))
                    else match idxOfOn c 1 t with
                         | none => none | some k => some (k + 1)) = some 1
              rw [ifPos hp]
              exact ifPos rfl
          | j + 2 =>
              have hne : ¬ ((j + 2 : Nat) = 1) := fun hc => Nat.noConfusion
                (Nat.succ.inj hc)
              have hb : j + 1 ≤ countOn c t := by
                have : j + 2 ≤ countOn c t + 1 := hcnt ▸ h2
                exact Nat.le_of_succ_le_succ this
              have ⟨k, hk⟩ := ih (j + 1) (Nat.succ_le_succ (Nat.zero_le j)) hb
              refine ⟨k + 1, ?_⟩
              show (if e.comm = c then (if (j + 2 : Nat) = 1 then some 1
                      else match idxOfOn c (j + 1) t with
                           | none => none | some i => some (i + 1))
                    else match idxOfOn c (j + 2) t with
                         | none => none | some i => some (i + 1)) = some (k + 1)
              rw [ifPos hp, ifNeg hne, hk]
      | false =>
          have hp : ¬ (e.comm = c) := fun hc =>
            Bool.noConfusion (he ▸ toDecide hc)
          have hcnt : countOn c (e :: t) = countOn c t := ifNeg hp
          have ⟨k, hk⟩ := ih n h1 (hcnt ▸ h2)
          refine ⟨k + 1, ?_⟩
          show (if e.comm = c then (if n = 1 then some 1
                  else match idxOfOn c (n - 1) t with
                       | none => none | some i => some (i + 1))
                else match idxOfOn c n t with
                     | none => none | some i => some (i + 1)) = some (k + 1)
          rw [ifNeg hp, hk]

/-- Fidelity for divergence 4: the position `idxOfOn` returns really holds an
issue of `c`, and `opAtOn` reads exactly that issue's operation. So `OpAt` and
`opAtOn` denote the same thing. -/
theorem idxOfOn_spec (c : C) :
    ∀ (l : List (Issue C O)) (n k : Nat), idxOfOn c n l = some k ->
      ∃ e, atPos l k = some e ∧ e.comm = c ∧ opAtOn c n l = some e.op := by
  intro l
  induction l with
  | nil =>
      intro n k h
      exact absurdNoneSome (show (none : Option Nat) = some k from h)
  | cons e t ih =>
      intro n k h
      have h0 : (if e.comm = c then
                   (if n = 1 then some 1
                    else match idxOfOn c (n - 1) t with
                         | none => none
                         | some j => some (j + 1))
                 else match idxOfOn c n t with
                      | none => none
                      | some j => some (j + 1)) = some k := h
      cases he : decide (e.comm = c) with
      | true =>
          have hp : e.comm = c := ofDecide he
          rw [ifPos hp] at h0
          cases hn : decide (n = 1) with
          | true =>
              have hn1 : n = 1 := ofDecide hn
              rw [ifPos hn1] at h0
              have hk : (1 : Nat) = k := Option.some.inj h0
              refine ⟨e, ?_, hp, ?_⟩
              · rw [← hk]; rfl
              · show (if e.comm = c then (if n = 1 then some e.op
                        else opAtOn c (n - 1) t) else opAtOn c n t) = some e.op
                rw [ifPos hp, ifPos hn1]
          | false =>
              have hne : ¬ (n = 1) := fun hc =>
                Bool.noConfusion (hn ▸ toDecide hc)
              rw [ifNeg hne] at h0
              match hj : idxOfOn c (n - 1) t with
              | none =>
                  rw [hj] at h0
                  exact absurdNoneSome
                    (show (none : Option Nat) = some k from h0)
              | some j =>
                  rw [hj] at h0
                  have hk : j + 1 = k :=
                    Option.some.inj (show some (j + 1) = some k from h0)
                  have ⟨e', hat, hcm, hop⟩ := ih (n - 1) j hj
                  have hj1 : 1 <= j := idxOfOn_pos c t (n - 1) j hj
                  refine ⟨e', ?_, hcm, ?_⟩
                  · rw [← hk, atPos_cons_of_pos e t j hj1]; exact hat
                  · show (if e.comm = c then (if n = 1 then some e.op
                            else opAtOn c (n - 1) t) else opAtOn c n t)
                      = some e'.op
                    rw [ifPos hp, ifNeg hne]; exact hop
      | false =>
          have hp : ¬ (e.comm = c) := fun hc =>
            Bool.noConfusion (he ▸ toDecide hc)
          rw [ifNeg hp] at h0
          match hj : idxOfOn c n t with
          | none =>
              rw [hj] at h0
              exact absurdNoneSome (show (none : Option Nat) = some k from h0)
          | some j =>
              rw [hj] at h0
              have hk : j + 1 = k :=
                Option.some.inj (show some (j + 1) = some k from h0)
              have ⟨e', hat, hcm, hop⟩ := ih n j hj
              have hj1 : 1 <= j := idxOfOn_pos c t n j hj
              refine ⟨e', ?_, hcm, ?_⟩
              · rw [← hk, atPos_cons_of_pos e t j hj1]; exact hat
              · show (if e.comm = c then (if n = 1 then some e.op
                        else opAtOn c (n - 1) t) else opAtOn c n t) = some e'.op
                rw [ifNeg hp]; exact hop

end Reads

/-! ## The model: topology, state, guards, transitions -/

/-- TLA's CONSTANTS, with `membersInRanks` making `Ranks` genuinely the
carrier of `CommMembers` (divergence 2). -/
structure Topology (R C O S : Type) where
  ranks : List R
  comms : List C
  member : C -> R -> Bool
  opsAll : List O
  admissibleOp : C -> O -> Bool
  streamOf : Issue C O -> S
  maxIssues : Nat
  requireMatchedIssueOrder : Bool
  requireUniformProgramOps : Bool
  requireUniformProgramComms : Bool
  requireStreamOrder : Bool
  membersInRanks : ∀ c r, member c r = true -> r ∈ ranks
  admissibleInOps : ∀ c op, admissibleOp c op = true -> op ∈ opsAll
  commsComplete : ∀ c r, member c r = true -> c ∈ comms

/-- TLA's VARIABLES. -/
structure State (R C O : Type) where
  issued : R -> List (Issue C O)
  running : C -> Bool
  doneOn : C -> Nat

section Model

variable {R C O S : Type} [DecidableEq R] [DecidableEq C] [DecidableEq O]
variable [DecidableEq S]

def commCount (s : State R C O) (r : R) (c : C) : Nat := countOn c (s.issued r)

def front (s : State R C O) (c : C) : Nat := s.doneOn c + 1

def minN (a b : Nat) : Nat := if a ≤ b then a else b

/-- TLA's `Completed(r, k)`. `true` outside the domain; see divergence 3. -/
def completedB (s : State R C O) (r : R) (k : Nat) : Bool :=
  match atPos (s.issued r) k with
  | none => true
  | some e => decide (commPos (s.issued r) k ≤ s.doneOn e.comm)

/-- TLA's `AtStreamHead(r, k)`. -/
def atStreamHeadB (T : Topology R C O S) (s : State R C O) (r : R) (k : Nat) :
    Bool :=
  (!T.requireStreamOrder) ||
    (match atPos (s.issued r) k with
     | none => true
     | some ek =>
         allLt k (fun j =>
           match atPos (s.issued r) j with
           | none => true
           | some ej =>
               (!decide (T.streamOf ej = T.streamOf ek)) || completedB s r j))

/-- TLA's `MemberReady(c, r)`. -/
def memberReadyB (T : Topology R C O S) (s : State R C O) (c : C) (r : R) :
    Bool :=
  decide (front s c ≤ commCount s r c) &&
    (match idxOfOn c (front s c) (s.issued r) with
     | none => true
     | some k => atStreamHeadB T s r k)

/-- TLA's `FullyPending(c)`. -/
def fullyPendingB (T : Topology R C O S) (s : State R C O) (c : C) : Bool :=
  T.ranks.all fun r =>
    (!T.member c r) || decide (front s c ≤ commCount s r c)

/-- TLA's `OpsAgreeAtFront(c)`. -/
def opsAgreeAtFrontB (T : Topology R C O S) (s : State R C O) (c : C) : Bool :=
  T.ranks.all fun r1 =>
    T.ranks.all fun r2 =>
      (!(T.member c r1 && T.member c r2)) ||
        decide (opAtOn c (front s c) (s.issued r1)
          = opAtOn c (front s c) (s.issued r2))

/-- TLA's `SameSite(c1, c2)`. -/
def sameSiteB (T : Topology R C O S) (c1 c2 : C) : Bool :=
  decide (c1 = c2) ||
    T.ranks.all fun r => !(T.member c1 r && T.member c2 r)

/-- TLA's `UniformProgramOpsOK(candidate)`. -/
def uniformOpsOKB (T : Topology R C O S) (f : R -> List (Issue C O)) : Bool :=
  T.ranks.all fun r1 =>
    T.ranks.all fun r2 =>
      allLt (minN (f r1).length (f r2).length + 1) fun k =>
        match atPos (f r1) k with
        | none => true
        | some e1 =>
            match atPos (f r2) k with
            | none => true
            | some e2 => decide (e1.op = e2.op)

/-- TLA's `UniformProgramCommsOK(candidate)`. -/
def uniformCommsOKB (T : Topology R C O S) (f : R -> List (Issue C O)) : Bool :=
  T.ranks.all fun r1 =>
    T.ranks.all fun r2 =>
      allLt (minN (f r1).length (f r2).length + 1) fun k =>
        match atPos (f r1) k with
        | none => true
        | some e1 =>
            match atPos (f r2) k with
            | none => true
            | some e2 => sameSiteB T e1.comm e2.comm

/-- TLA's `IssueCandidate(r, c, op)`. -/
def issueCandidate (s : State R C O) (r : R) (e : Issue C O) : State R C O :=
  { s with
    issued := fun r' => if r' = r then s.issued r' ++ [e] else s.issued r' }

/-- TLA's `IssueAllowed(r, c, op)`. -/
def issueAllowedB (T : Topology R C O S) (s : State R C O) (r : R) (c : C)
    (op : O) : Bool :=
  T.member c r && T.admissibleOp c op &&
    decide ((s.issued r).length < T.maxIssues) &&
    ((!T.requireUniformProgramOps) ||
      uniformOpsOKB T (issueCandidate s r ⟨c, op⟩).issued) &&
    ((!T.requireUniformProgramComms) ||
      uniformCommsOKB T (issueCandidate s r ⟨c, op⟩).issued)

/-- TLA's `StartAllowed(c)`. -/
def startAllowedB (T : Topology R C O S) (s : State R C O) (c : C) : Bool :=
  (!s.running c) &&
    (T.ranks.all fun r => (!T.member c r) || memberReadyB T s c r) &&
    ((!T.requireMatchedIssueOrder) || opsAgreeAtFrontB T s c)

/-- The post-state of TLA's `Start(c)`. -/
def startState (s : State R C O) (c : C) : State R C O :=
  { s with running := fun d => s.running d || decide (d = c) }

/-- The post-state of TLA's `Complete(c)`. -/
def completeState (s : State R C O) (c : C) : State R C O :=
  { s with
    running := fun d => s.running d && !decide (d = c),
    doneOn := fun d => if d = c then s.doneOn d + 1 else s.doneOn d }

/-- TLA's `AllDone`. -/
def allDoneB (T : Topology R C O S) (s : State R C O) : Bool :=
  (T.comms.all fun c => !s.running c) &&
    (T.ranks.all fun r => decide ((s.issued r).length = T.maxIssues)) &&
    (T.comms.all fun c =>
      T.ranks.all fun r =>
        (!T.member c r) || decide (commCount s r c = s.doneOn c))

/-- TLA's `Init`. -/
def Init (_T : Topology R C O S) (s : State R C O) : Prop :=
  (∀ r, s.issued r = []) ∧ (∀ c, s.running c = false) ∧ (∀ c, s.doneOn c = 0)

/-- TLA's `Next`, one constructor per disjunct. -/
inductive Step (T : Topology R C O S) : State R C O -> State R C O -> Prop where
  | issue (s : State R C O) (r : R) (c : C) (op : O)
      (h : issueAllowedB T s r c op = true) :
      Step T s (issueCandidate s r ⟨c, op⟩)
  | start (s : State R C O) (c : C) (h : startAllowedB T s c = true) :
      Step T s (startState s c)
  | complete (s : State R C O) (c : C) (h : s.running c = true) :
      Step T s (completeState s c)
  | terminated (s : State R C O) (h : allDoneB T s = true) : Step T s s

/-- The safety part of TLA's `Spec`: `Init /\ [][Next]_vars`, as reachability.
`Step.terminated` covers the stuttering disjunct. -/
inductive Reachable (T : Topology R C O S) : State R C O -> Prop where
  | init (s : State R C O) (h : Init T s) : Reachable T s
  | step (s s' : State R C O) (h : Reachable T s) (hs : Step T s s') :
      Reachable T s'

/-- TLA's `Stuck`. -/
def stuckB (T : Topology R C O S) (s : State R C O) : Bool :=
  (T.comms.all fun c => !s.running c) &&
    (T.ranks.all fun r =>
      T.comms.all fun c =>
        T.opsAll.all fun op => !issueAllowedB T s r c op) &&
    (T.comms.all fun c => !startAllowedB T s c)

/-! ## The safety invariants, as propositions

These are the TLA+ invariants of ScoutBModel.tla, stated over all topologies
and all states. None of them mentions `maxIssues`; see divergence 7 for the
one conjunct of `TypeOK` that does and is therefore left out. -/

/-- TLA's `TypeOK`, minus its `MaxIssues` range conjunct. -/
def TypeOK (T : Topology R C O S) (s : State R C O) : Prop :=
  ∀ r k e, atPos (s.issued r) k = some e ->
    T.admissibleOp e.comm e.op = true ∧ T.member e.comm r = true

/-- TLA's `RendezvousMembership`. -/
def RendezvousMembership (T : Topology R C O S) (s : State R C O) : Prop :=
  ∀ c, s.running c = true ->
    ∀ r, T.member c r = true -> s.doneOn c + 1 ≤ commCount s r c

/-- TLA's `CommFifo`. -/
def CommFifo (T : Topology R C O S) (s : State R C O) : Prop :=
  ∀ c r, T.member c r = true -> s.doneOn c ≤ commCount s r c

/-- TLA's `RendezvousOpAgreement`. -/
def RendezvousOpAgreement (T : Topology R C O S) (s : State R C O) : Prop :=
  ∀ c, s.running c = true ->
    ∀ r1 r2, T.member c r1 = true -> T.member c r2 = true ->
      opAtOn c (s.doneOn c + 1) (s.issued r1)
        = opAtOn c (s.doneOn c + 1) (s.issued r2)

/-- The safety property this file establishes for every reachable state, at
every schedule length. -/
def Safety (T : Topology R C O S) (s : State R C O) : Prop :=
  RendezvousOpAgreement T s ∧ RendezvousMembership T s ∧ CommFifo T s
    ∧ TypeOK T s

/-- The inductive invariant. It is the CONJUNCTION of the invariants above and
nothing more: no auxiliary conjunct was needed. The conjunction is the
strengthening -- `ScoutBInductiveInvariant.lean` exhibits a single `Complete`
step that preserves `CommFifo` alone and breaks it, so no single conjunct is
inductive by itself. -/
def Inv (T : Topology R C O S) (s : State R C O) : Prop :=
  RendezvousMembership T s ∧ CommFifo T s ∧ RendezvousOpAgreement T s
    ∧ TypeOK T s

end Model

/-! ## Fidelity lemmas, reported in their own result tokens

These are the encoding-fidelity claims of divergences 3 and 4 that are PROVED
rather than left to inspection. -/

#print axioms idxOfOn_spec
#print axioms idxOfOn_isSome_of_le
#print axioms opAtOn_append_of_le

end Qwen3Formal.ScoutBProtocol
