# 04 — Promote the versioned trace contract

**What to build:** Use the accepted 1x1 and 2x2 scout evidence to replace the
provisional trace with a stable versioned contract that preserves raw evidence,
normalization provenance, distributed completeness, and compatibility rules
through both formal backends.

**Blocked by:** 03 — Scout B: 2x2 DPxTP Qwen3 tracer bullet.

**Status:** ready-for-agent

- [ ] Before implementation, a human-approved Phase 2 design checkpoint compares
  both immutable scout attempt bundles and records the stable vocabulary,
  compatibility boundary, migration policy, and unsupported observations; no
  provisional `scout.v0` shape becomes stable merely because it already exists.
- [ ] The stable vocabulary defines run attempt, process/rank/mesh, device,
  clock/order, step/phase, lineage, event, tensor, work, executor, producer, and
  causal identities without claiming unsupported global order.
- [ ] Raw observed evidence, deterministic normalized trace, and compact
  generated formal fixtures have explicit and testable provenance relationships.
- [ ] Schema versioning states reader/writer compatibility, required fields,
  rejection behavior, and migration rules from `qwen3.formal.scout.v0`.
- [ ] Missing, duplicate, conflicting, or partial distributed evidence fails
  validation before export.
- [ ] The 1x1 and 2x2 scout bundles migrate to the stable contract and still
  pass their independently maintained TLA+ and Lean checks.
- [ ] Unknown extensions remain preservable without weakening required-field or
  invariant validation.
- [ ] The contract remains experiment-owned and introduces no core import of
  Bazel, TLC, Lean, or experiment packages.
- [ ] Tests cover round-trip determinism, migration, compatibility rejection,
  provenance, partial bundles, and unsupported ordering through public APIs.
- [ ] From the final state, the supported Scout A and Scout B commands
  re-execute the real 1x1 and 2x2 Trainer paths and both formal backends;
  checked-in or replayed fixtures alone do not satisfy this regression gate.
- [ ] Focused/broader tests, artifact sync, lint, and separate Standards and
  Spec reviews by a fresh clean-context Codex reviewer pass from the same final
  worktree state. Missing review output or unresolved blocking findings block
  the ticket.
