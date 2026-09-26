# 08 — Measure and decide the promotion boundary

**What to build:** Measure the completed vehicle against TorchTitan's
observability evidence budget and record an evidence-backed decision about
whether any trace or observer interface is ready to be proposed for core
promotion.

**Blocked by:** 07 — Observed-to-formal refinement bridge.

**Status:** ready-for-agent

- [ ] Matched runs separately report median steady-state throughput delta, GPU
  memory delta, host CPU, artifact bytes per GPU-hour, and detection latency.
- [ ] Performance comparisons use the same model, checkpoint/initialization,
  data, global batch, tokens per optimizer step, topology, precision, seed, and
  deterministic settings, with at least ten measured training steps after
  initialization and warmup.
- [ ] Capture costs are separated by raw observation, normalization/export,
  TLC, Lean, and evidence persistence rather than collapsed into one number.
- [ ] Detection latency names the injected event or mutation, its injection
  timestamp, the responsible checker, and the timestamp of the classified
  result; provisioning time, skipped checks, and generic process failure are
  reported separately and cannot be counted as successful detection.
- [ ] The report evaluates Tier 0's one-percent median throughput budget and
  labels any unmet gate without weakening it.
- [ ] The decision states whether the experiment-owned boundary remains the
  correct seam or a separate core-observer design should be opened; it does not
  silently move experiment machinery into core.
- [ ] Any proposed core seam has explicit compatibility, dependency-direction,
  default-off/default-on, failure-policy, and checkpoint/evidence implications
  and requires separate human design approval before implementation.
- [ ] Large traces, caches, checker states, and profiles remain ignored evidence
  artifacts while compact manifests, summaries, and claims remain reviewable.
- [ ] Performance-evidence validation rejects mismatched run profiles, missing
  warmup or samples, fewer than ten measured steps, missing metrics, non-finite
  values, and summaries that cannot be reproduced from the referenced attempt
  artifacts.
- [ ] From the final state, the supported Scout A and Scout B commands
  re-execute the real 1x1 and 2x2 Trainer paths and both formal backends;
  checked-in or replayed fixtures alone do not satisfy this regression gate.
- [ ] Focused/broader tests, performance-evidence validation, lint, and
  separate Standards and Spec reviews by a fresh clean-context Codex reviewer
  pass from the same final worktree state. Missing review output or unresolved
  blocking findings block the ticket.
