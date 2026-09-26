# 05 — Refine single-rank step semantics

**What to build:** Deepen the stable contract's single-rank model so observed
Qwen3 forward, backward, gradient, optimizer, and step-completion facts are
checked by explicit transition semantics and matching Lean propositions.

**Blocked by:** 04 — Promote the versioned trace contract.

**Status:** ready-for-agent

- [ ] The refinement defines the supported phase state machine and the exact
  evidence that advances each transition.
- [ ] TLA+ and Lean independently express matching invariants for forward before
  backward, gradient readiness before optimizer mutation, and optimizer
  completion before step end.
- [ ] Each invariant maps to raw runtime evidence and to a normalized fact;
  configuration intent alone cannot satisfy it.
- [ ] One controlled mutation at a time demonstrates the named rejection for
  phase order, causal order, missing gradient readiness, and premature step end.
- [ ] Generated modules contain facts only; semantic definitions and proofs are
  maintained separately and cannot be regenerated to bless invalid input.
- [ ] This ticket owns the single-rank lifecycle module and does not redefine
  distributed topology, placement, or collective semantics owned by ticket 06;
  both refinements depend only on ticket 04's stable contract until ticket 07
  composes them.
- [ ] The accepted Scout A trace remains green and the Scout B trace remains
  compatible with the refined single-rank projection.
- [ ] Tests distinguish invariant violation, malformed input, checker failure,
  and missing evidence without generalizing beyond the exercised mutations.
- [ ] From the final state, the supported Scout A and Scout B commands
  re-execute the real 1x1 and 2x2 Trainer paths and both formal backends;
  checked-in or replayed fixtures alone do not satisfy this regression gate.
- [ ] Focused/broader tests, lint, and separate Standards and Spec reviews by a
  fresh clean-context Codex reviewer pass from the same final worktree state.
  Missing review output or unresolved blocking findings block the ticket.
