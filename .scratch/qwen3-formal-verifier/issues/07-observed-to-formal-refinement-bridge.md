# 07 — Observed-to-formal refinement bridge

**What to build:** Demonstrate that the stable normalized traces from real 1x1
and 2x2 Qwen3 steps are legal behaviors of the independently maintained formal
models, with an auditable claims matrix and bounded negative evidence.

**Blocked by:** 05 — Refine single-rank step semantics; 06 — Refine distributed
semantics.

**Status:** ready-for-agent

## Status note (2026-09-26): single-rank half delivered by ticket 10

Ticket 10 delivered the 1x1 bridge: `SingleRankRefine.tla` plus
`SingleRankRefineBad.tla`, the latter refused for the right reason under four
invariants (corruption isolated, not admitted, refusal not later than the
mutation, mutation never emitted).

The 2x2 bridge is delivered by ticket 12 as `DeviceMeshRefine.tla`: the observed
per-rank issue order of all four ranks, all 108 issues each, runs to completion
under every guard of `DeviceMeshModel`, with a same-communicator transposition
refused at the rendezvous guard and a bounded-skew fragment establishing
confluence by exhaustive search. It is tracked separately because the
distributed case
has a trap this ticket's wording does not anticipate: the observed per-rank
`started`/`completed` sub-order is **synthetic**, appended contiguously post
hoc from the Flight Recorder with timestamps dropped. A bridge replaying
per-rank event order would certify a full serialization that never happened,
and would pass, since a serialized schedule satisfies every guard. Only the
`enqueued` sub-order is faithful.


- [ ] The bridge consumes a validated stable trace and produces checker inputs
  without changing the trace facts or formal semantics.
- [ ] A documented refinement relation maps the normalized initial state and
  each observed event to an abstract state/action. The bridge emits an auditable
  step-indexed witness, and PASS requires both formal backends to establish the
  initial-state predicate and every successive transition; final-state
  invariant checks alone do not justify the "legal behavior" claim.
- [ ] Every checked TLA+ invariant and Lean proposition maps to required runtime
  events, normalized fields, and a named verification result.
- [ ] Stable digests bind raw evidence, normalized trace, generated facts,
  maintained semantics, tool versions, and checker outputs into one attempt.
- [ ] Valid observed 1x1 and 2x2 traces are accepted by their respective models.
- [ ] The bridge rejects schema/version mismatch, incomplete provenance,
  checker-input drift, and a fact/semantics digest mismatch before reporting a
  formal PASS.
- [ ] The claims matrix states exactly which controlled mutations each checker
  rejects and explicitly lists unmodeled behavior.
- [ ] At least one controlled mutation that preserves the checked final-state
  facts but breaks an intermediate transition is rejected by the refinement
  relation, demonstrating that the bridge checks behavior rather than only a
  final snapshot.
- [ ] Replay from the immutable attempt bundle reproduces deterministic facts
  and equivalent checker classifications.
- [ ] From the final state, the supported Single-rank and Device-mesh commands
  re-execute the real 1x1 and 2x2 Trainer paths and both formal backends;
  checked-in or replayed fixtures alone do not satisfy this regression gate.
- [ ] Focused/broader tests, artifact sync, lint, and separate Standards and
  Spec reviews by a fresh clean-context Codex reviewer pass from the same final
  worktree state. Missing review output or unresolved blocking findings block
  the ticket.
