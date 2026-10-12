# 09 — Kineto and flight-recorder collective evidence bridge

**What to build:** Corroborate Device-mesh's observed collective lifecycle
against a second independent runtime source -- the PyTorch Kineto profiler and
the NCCL flight recorder -- and carry that join through the sealed evidence
contract as a bounded, schema-validated diagnostic.

**Blocked by:** 03 — Device-mesh: 2x2 DPxTP Qwen3 tracer bullet.

**Status:** ready-for-agent

## Why this is a separate ticket

Ticket 03's accepted bundles and its Codex audit cover the `-final` worktree
source. A draft of this bridge already exists in the sibling worktree
`.worktrees/qwen3-formal-verifier` (branch
`phi9t/traecode/qwen3-formal-verifier`), written after ticket 03's re-review and
never gated or reviewed. Merging it into ticket 03 would put an unreviewed
882-line delta inside a closed verdict, so it is scoped here instead. The draft
is prior art to reconcile, not an accepted implementation; it also bumps the
Device-mesh runtime, stage, and evidence manifest schemas to `-b.v1`, which this
ticket must justify or drop.

It also responds to a recorded lower-severity finding: exact graph completeness
is currently owned by Python before export, with TLA+ and Lean validating
supplied rows structurally rather than reconstructing the expected graph. A
second independent runtime source narrows that gap without overstating it.

## Acceptance criteria

- [ ] The bridge observes the same single bounded four-rank DP-shard-2/TP-2
  Qwen3 step as Device-mesh, preserving its model, precision, optimizer, seed,
  deterministic input, batch, sequence length, and checkpoint policy. It does
  not change the comparison profile to make the join succeed.
- [ ] Kineto collective records and NCCL flight-recorder entries join to
  observed raw collective events on an explicit, documented key. A record that
  cannot be joined is a validation failure, not a silently dropped row.
- [ ] The diagnostic is bounded in both record count and serialized bytes, and
  exceeding either bound fails rather than truncating silently.
- [ ] The diagnostic carries its own schema, and its identity must equal the
  identity of the raw trace it corroborates.
- [ ] Every bridged record binds back to the concrete raw event IDs it
  corroborates, so a finding can be joined to runtime events.
- [ ] Profiler capture overhead on the observed step is measured and reported,
  separately from the Tier 0 budget claim, before any promotion is proposed.
- [ ] The bridge is optional evidence: Device-mesh's gate, trace identity, and
  both formal checkers must still pass with the bridge disabled or unavailable.
  An unavailable profiler is never reported as a successful join.
- [ ] Manifest schema changes, if retained, are justified against the shape they
  actually change; an unchanged shape does not get a new version string.
- [ ] Synthetic or replayed profiler traces remain unit fixtures only and are
  not reported as runtime corroboration evidence.
- [ ] Focused and distributed integration tests, artifact synchronization,
  no-fetch cache reuse, lint, and separate Standards and Spec reviews by a fresh
  clean-context Codex reviewer pass from the same final worktree state.
- [ ] The report states the bounded claim: one 2x2 execution corroborated by a
  second source, not proof of arbitrary-topology collective correctness.
