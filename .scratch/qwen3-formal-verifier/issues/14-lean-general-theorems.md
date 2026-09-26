# 14 — Lean theorems over quantified inputs

**What to build:** Replace the Lean evaluation certificates with theorems that
are universally quantified over inputs, so the Lean layer states something
about the protocol rather than about one trace.

**Blocked by:** 11 — DPxTP collective protocol model.

**Status:** ready-for-agent

## What the Lean layer proves today

`ScoutDistributed.lean` and `ScoutALifecycle.lean` define `Bool`-valued
predicates and close them with `rfl` or `by decide` against literal trace
data. That is a kernel-checked *evaluation*: it proves the checked-in trace
satisfies the predicate. It is genuine -- the axiom-free check is real and the
negatives fail as they should -- but it is decidable arithmetic on constants,
so it says nothing about any other trace, and a reader who sees "Lean 4 proof"
will assume more.

Ticket 12's P0 removed the worst instance, a predicate asserting a phase
serialization that does not occur. What remains is honest but narrow.

## What to prove instead

At least one theorem of the form: for **all** rank sets, communicator
assignments and issue sequences, if per-communicator issue order agrees and
the communicator wait-for graph is acyclic, then no rendezvous mismatches.
That is a statement about the protocol, and it composes with the TLA+ work:
TLC checks reachability up to a bound, Lean carries the general implication.

A natural first target is the necessary-but-not-sufficient result the model
negatives already demonstrate by refutation: order agreement alone does not
imply deadlock freedom. Proving one direction in Lean and refuting the other
in TLC is a stronger pair than either alone.

## Acceptance criteria

- [ ] At least one `theorem` with a genuine `∀` over structures, not a `rfl`
      over literals.
- [ ] The existing evaluation certificates stay, and are relabelled in
      comments and in the result tokens as *evaluation of the observed trace*,
      distinct from general theorems. The distinction must be visible in the
      sealed log, not only in source comments.
- [ ] `#print axioms` remains clean; no `sorry`, no new axioms. The gate's
  `formal_classify_lean_valid` already rejects `sorryAx` and any axiom
  dependency -- keep it that way rather than widening it.
- [ ] A negative: a near-miss statement (for example dropping the acyclicity
  hypothesis) that Lean rejects, so the theorem's hypotheses are shown to be
  load-bearing rather than decorative.
- [ ] Lean 4 stays on its pinned toolchain; no Mathlib dependency added
      without recording the size and hermeticity cost in `MODULE.bazel`.
