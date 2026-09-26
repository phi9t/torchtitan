# 06 — Refine distributed semantics

**What to build:** Deepen the stable contract's distributed model so an
observed 2x2 DPxTP Qwen3 step is checked for topology, placement, collective
work, executor, producer, and causal consistency by both formal backends.

**Blocked by:** 04 — Promote the versioned trace contract.

**Status:** ready-for-agent

## Status note (2026-09-26): partly delivered by ticket 11

Ticket 11 delivered the protocol layer: `ScoutBModel.tla` with per-communicator
FIFO, rendezvous membership and op agreement, stream-head ordering and deadlock
freedom, checked over every interleaving; `PerCommunicatorIssueOrderAgreement`
checked against the real 108-collective run; and negatives that TLC discovers,
including the one showing order agreement is necessary but **not** sufficient.

Two corrections from that work that this ticket's wording predates:

- The global communicator key is `process_group.canonical_id`. There are eight
  communicator instances, not four; `runtime_pg_id` is per-rank local numbering
  and denotes different communicators on different ranks.
- The exported collective order was lexicographic by `work_id`, not issue
  order. Issue order comes from each work's `collective.enqueued` event.

What remains from this ticket:

- Placement is still two booleans per rank, not the observed 74 per-rank
  placements. Tracked as ticket 17.
- Collective payload sizes and dtypes are unmodelled. Tracked as ticket 15.
- Producer correlation is inferred by positional zip and presented as observed.
  Tracked as ticket 18 (labelling) and ticket 09 (the joined-key fix).


- [ ] TLA+ and Lean independently express matching DP-group and TP-group shape,
  rank-coordinate, tensor-placement, work-membership, and shard-agreement
  invariants.
- [ ] Collective work has an explicit enqueue/start/complete state machine with
  rank participation, executor assignment, and producer correlation.
- [ ] Cross-rank relations rely only on observed group membership,
  synchronization, or causal evidence and never on a manufactured total order.
- [ ] Controlled mutations separately exercise wrong group membership, wrong TP
  shard, missing participant, invalid work transition, missing producer, and
  premature step completion.
- [ ] Each checker result is tied to the exact mutation it exercises; no broad
  failure-class or arbitrary-topology claim is made.
- [ ] The accepted Scout B trace remains green, and incomplete four-rank bundles
  remain ineligible for export.
- [ ] Generated facts, maintained semantics, raw evidence, and normalization
  provenance remain distinct and auditable.
- [ ] This ticket owns the distributed topology, placement, and collective
  module and consumes ticket 04's stable contract without redefining the
  single-rank lifecycle semantics owned by ticket 05; ticket 07 composes the
  independently reviewable refinements.
- [ ] From the final state, the supported Scout A and Scout B commands
  re-execute the real 1x1 and 2x2 Trainer paths and both formal backends;
  checked-in or replayed fixtures alone do not satisfy this regression gate.
- [ ] Focused/distributed tests, lint, and separate Standards and Spec reviews
  by a fresh clean-context Codex reviewer pass from the same final worktree
  state. Missing review output or unresolved blocking findings block the ticket.
