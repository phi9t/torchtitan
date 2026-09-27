# 14 — Lean theorems over quantified inputs

**What to build:** Replace the Lean evaluation certificates with theorems that
are universally quantified over inputs, so the Lean layer states something
about the protocol rather than about one trace. Ticket 13 — the bounded-checking
gap — is folded in here and discharged in Lean rather than by pinning Apalache
or TLAPS; 13 records that decision and its reasons.

**Blocked by:** 11 — DPxTP collective protocol model.

**Status:** done

## What the Lean layer proved before this ticket

`ScoutDistributed.lean` and `ScoutLifecycle.lean` define `Bool`-valued
predicates and close them with `rfl` or `by decide` against literal trace data.
That is a kernel-checked *evaluation*: it proves the checked-in trace satisfies
the predicate. It is genuine -- the axiom-free check is real and the negatives
fail as they should -- but it is decidable arithmetic on constants, so it says
nothing about any other trace, and a reader who sees "Lean 4 proof" will assume
more.

(This ticket originally named the second file `ScoutALifecycle.lean`. There is
no such file and never was; the module is `ScoutLifecycle.lean`, shared by the
Scout A and Scout B suites. Corrected here.)

Ticket 12's P0 removed the worst instance, a predicate asserting a phase
serialization that does not occur. What remained was honest but narrow.

## What was built

Three new Lean modules, carrying no trace data at all:

- `ScoutBProtocol.lean` — an encoding of `ScoutBModel.tla`: `Topology`,
  `State`, the guards as computable `Bool`s, `Init`, `Step` (one constructor per
  `Next` disjunct), `Reachable`, and the safety invariants as `Prop`s. Its
  header carries the definition-by-definition correspondence to the TLA+ module
  and ten named divergences, each with the direction it moves the claim.
- `ScoutBInductiveInvariant.lean` — the three inductive obligations, the
  composition to every reachable state, and a negative showing the conjunction
  is required.
- `ScoutBWaitGraph.lean` — the general protocol theorem and the
  load-bearing-hypothesis negative.

Plus `ScoutBProtocolInvalid.lean` (a controlled negative Lean must reject),
`run_lean_scout_b_protocol.sh`, and the `lean_scout_b_protocol_test` target in
both `tier0_formal_tests` and `scout_b_formal_tests`.

### The inductive invariant, and which safety property it covers

`Inv = RendezvousMembership /\ CommFifo /\ RendezvousOpAgreement /\ TypeOK`,
and:

    initiation           Init T s -> Inv T s
    consecution          Inv T s -> Step T s s' -> Inv T s'
    sufficiency          Inv T s -> Safety T s
    safetyOfReachable    Reachable T s -> Safety T s

`safetyOfReachable` mentions no bound: `Topology.maxIssues` does not appear in
its statement, so `MaxIssues` is out of the safety claim.

`consecution` takes one guard hypothesis, `requireMatchedIssueOrder = true`.
That is NCCL's own matching requirement, and it is what makes a started
rendezvous agree on the operation; `ScoutBModelUnguarded.cfg` is the TLC
negative for the same fact. The other three `require*` flags are not needed,
which is a stronger statement than the configuration TLC checks.

What was NOT made unconditional, and why:

- `StuckImpliesAllDone` cannot be made bound-free for a syntactic reason
  rather than a limitation of the proof: `AllDone` is defined by
  `Len(issued[r]) = MaxIssues`, so the bound is in the property's own
  statement. It holds in `ScoutBModel.cfg` at `MaxIssues = 2`, which is a real
  result about that configuration; there is no unbounded version of it to
  prove.
- `DeadlockFreedom == ~StuckByCircularWait` is proved only CONDITIONALLY, under
  an acyclicity hypothesis on the wait-for graph. Whether that graph is acyclic
  is an instance-and-guards question, which is what TLC answers. The
  unconditional, unbounded statement remains open.

`TypeOK` is carried because it is free. Its `doneOn \in [CommIds ->
0..MaxIssues]` conjunct is deliberately omitted: it is the only part of `TypeOK`
that mentions the bound, and keeping it would put the bound back into the claim.

### The general protocol theorem

`orderAgreementAndAcyclicWaitGraphExcludeBothHazards`: for all topologies,
states and schedule lengths, if per-communicator issue order agreement is
required and the communicator wait-for graph is acyclic, then the state has no
rendezvous mismatch and no stream-ordered circular wait.

It pairs with `ScoutBModelDivergent.cfg`, which refutes the other direction:
with operations still agreeing at every position, relaxing only
communicator-site agreement makes `DeadlockFreedom` false. Proving one
direction in Lean and refuting the other in TLC is a stronger pair than either
alone.

### The negatives

- `acyclicityIsLoadBearing` — Lean refutes the statement with the acyclicity
  hypothesis dropped, on a two-communicator witness that satisfies every other
  hypothesis. `cyclicWitnessHasTwoCycle` exhibits the cycle and
  `cyclicWitnessAdmitsNoRankFunction` shows no rank function can exist, so the
  hypothesis that fails is exactly the dropped one.
- `waitGraphWitnessIsNotVacuous` — the same topology with both ranks issuing in
  the SAME order is not stuck and the communicator can start, so the witness
  reports a property of the issue order rather than of an unsatisfiable guard.
- `commFifoAloneIsNotInductive` — one `Complete` step preserves `CommFifo` in
  the pre-state and breaks it in the post-state, so the conjunction in `Inv` is
  the strengthening rather than decoration.
- `ScoutBProtocolInvalid.lean` — the uncompilable form of negative, matching
  the shape of `ScoutAInvalid` and `ScoutBInvalid`.

## Acceptance criteria

This ticket's own:

- [x] At least one `theorem` with a genuine `∀` over structures, not a `rfl`
      over literals. Twenty of them, over `Topology`, `State` and `Step`.
- [x] The existing evaluation certificates stay, and are relabelled as
      *evaluation of the observed trace* in the RESULT TOKENS as well as in
      comments: `SCOUT_A_LEAN_VALID`, `SCOUT_A_LEAN_NEGATIVE`,
      `SCOUT_B_LEAN_VALID`, `SCOUT_B_LEAN_MUTATION` and `SCOUT_B_LEAN_NEGATIVE`
      now carry `kind=evaluation scope=observed-trace`, while every new token
      carries `kind=theorem scope=all-topologies-all-schedules bound=none`.
      The distinction is visible in the sealed log.
- [x] `#print axioms` remains clean for every theorem; no `sorry`, no new
      axioms; `formal_classify_lean_valid` was not widened. This forced the
      proofs away from `omega` (propext, Quot.sound), `simp` (propext), most
      core `List` lemmas (propext) and anything classical (Classical.choice);
      the discipline is recorded in `ScoutBProtocol.lean`'s header.
- [x] A negative whose hypotheses are shown to be load-bearing: see above.
- [x] Lean 4 stays on its pinned toolchain; no Mathlib.

Folded in from ticket 13:

- [x] `Inv` stated explicitly and separately from the reachability-checked
      invariants.
- [x] Initiation, consecution and sufficiency each checked and each reported as
      its own result token.
- [x] `MaxIssues` absent from the safety claim's statement.
- [x] CPU-only, no fresh GPU run, in the trace-free tier-0 suite.
- [x] Where an invariant could not be made inductive, recorded as a finding
      with the reason rather than left silent.
- [x] No new toolchain pinned, so no new `MODULE.bazel` pin to corroborate.

## Fidelity review outcome

An independent review found the encoding SOUND WITH CAVEATS: all twenty
theorems axiom-free on the reviewer's own Lean run, `safetyOfReachable`'s type
free of `maxIssues` with the `#print allDoneB` control confirming that is a
property of the statement rather than the printer, all divergences faithful,
`Step` exactly one constructor per `Next` disjunct, `blockerOfStuckPending`'s
case analysis exhaustive against `StartAllowed`'s three conjuncts, all three
negatives proving genuine negations, and the comment-stripped `ScoutBModel.tla`
diff empty, so no TLC regression is possible from this change. Four things were
fixed afterwards:

- **The flagship theorem returned a vacuous-capable component.**
  `RendezvousOpAgreement` compares two `Option O`, so it holds vacuously on a
  state where a communicator is running but a member is short of an issue;
  `RendezvousMembership` is what forces both sides to `some`.
  `orderAgreementAndAcyclicWaitGraphExcludeBothHazards` projected that conjunct
  away with `.1`. It now returns the whole of `Safety T s`, and divergence 3
  names this site explicitly with the conjunction argument rather than relying
  on its blanket direction-of-strength claim, which does NOT cover it.
- **Seven of the twenty tokens overclaimed**, and a test enshrined it.
  `by decide` over one fixed topology with `maxIssues := 2` (or `:= 1`) was
  labelled `scope=all-topologies-all-schedules bound=none`. There are now three
  vocabularies -- `kind=theorem`, `kind=witness scope=fixed-instance bound=<n>`
  and `kind=theorem-negative scope=fixed-instance bound=<n>` -- plus a
  `conditional=acyclic-wait-for-graph` field on the three results that have a
  hypothesis. The test's blanket `all("bound=none" ...)` assertion is replaced
  by a per-token vocabulary check in both directions.
- **Divergence 8's first argument was unsound.** Enlarging `blockedBy` makes
  `hacyclic` harder to supply AND `WaitClosed` harder to satisfy, so it weakens
  the theorem on both sides; "harder hypothesis, so no weakening" does not
  follow. The divergence now leads with the argument that works -- both
  differences are INERT where the proof evaluates the relation -- and records
  that `deadlockFreeOfAcyclicWaitGraph` carries no `Reachable` hypothesis, so
  the theorem to read against TLC's `DeadlockFreedom` is the flagship one.
- **`Topology` had no `comms`-completeness obligation**, so `stuckB` and
  `allDoneB`, which quantify only over `T.comms`, rested on an unstated side
  condition. `Topology.commsComplete` is now a proof field, and the header no
  longer claims Lean's types discharge `issued[r][k].comm \in CommIds` -- they
  do not; `TypeOK` simply drops that conjunct.

Also from the review: `StuckImpliesAllDone` was named by no token at all, so
`SCOUT_B_MODEL_SAFETY` now carries `invariants=`, derived from the cfg by
`formal_cfg_invariants`. The review also strengthened divergence 4 -- `CommPos`
strictly increases along a communicator's positions, so the CHOOSE predicate has
a UNIQUE solution and TLA's CHOOSE is FORCED to `idxOfOn`'s answer rather than
merely compatible with it -- which is now recorded there.

## The remaining reviewable claim

The encoding fidelity of `ScoutBProtocol.lean` against `ScoutBModel.tla` is a
SECOND claim, and there is no mechanical link between the two files. A Lean
proof about a different model than the one TLC checks is worth nothing. The
correspondence is therefore written out definition by definition in that file's
header, with ten named divergences; three of them (that the encoded index
reads the right issue, that the index exists whenever its guard holds, and that
appending an issue leaves earlier reads alone) are discharged as Lean lemmas
with their own result tokens. The rest must be checked by inspection.

### Gate evidence

Full nine-stage Scout B gate, all stages exit 0, sealed and verified, after the
four fidelity-review findings were fixed.

- Scout B evidence ID:
  `sha256:c370f7163b86b8d90916d37f93fb12b15517d5d859a1fbd3c7cc41d414bbca0f`
- Scout B source ID:
  `sha256:2e84b97e26aace9081ff25d2c8d550f0964467a488c95c098bd8287359c66e08`
- Nested Scout A evidence ID:
  `sha256:c331886cd5d27f4aa6904740e7efdfd95d1cd45ebf363319ec6fc7ffa554eea4`

The vocabulary split is visible in the sealed log, which was the point of the
relabelling:

```
consecution                        kind=theorem  scope=all-topologies-all-schedules bound=none
blockerOfStuckPending              kind=theorem  scope=all-topologies-all-schedules bound=none
cyclicWitnessIsStuck               kind=witness  scope=fixed-instance bound=2
cyclicWitnessHasTwoCycle           kind=witness  scope=fixed-instance bound=2
commFifoAloneIsNotInductive        kind=witness  scope=fixed-instance bound=1
acyclicityIsLoadBearing   kind=theorem-negative  scope=fixed-instance bound=2
```

A reader of the log can now tell a theorem over all topologies and schedules
from a `decide` over one fixed topology with a declared bound. Before the fix
all of these carried `bound=none`, and a test asserted that they must.
