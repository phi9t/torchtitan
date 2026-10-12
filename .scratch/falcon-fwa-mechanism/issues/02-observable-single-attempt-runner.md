# 02 - Emit one observable mechanism attempt

Type: task
Status: ready-for-agent
Blocked by: 01
External gate: Campaign B 09 - native Falcon attempt-bundle contract (human-enforced)
Parent: ../spec.md

## What to build

Run exactly one mechanism arm and seed through a native evidence attempt while
capturing bounded, scale-aware state diagnostics that do not change model
outputs or gradients.

## Acceptance criteria

- [ ] One invocation owns exactly one logical arm and seed; matrix orchestration
  launches separate invocations rather than sharing one run/attempt identity.
- [ ] M0-M4 are materialized from the approved registry with only their declared
  alignment, normalization, or scale-compensation knob changed.
- [ ] Fixed probe batches record per-layer/head count, mean, standard deviation,
  and p05/p50/p95 for key energy, beta, actual lambda, eta, gamma, e-folding
  horizon, raw/canonical state norm, update norm, and read norm, plus clamp
  fraction.
- [ ] Screen probes occur at steps 0, 100, 500, 1,000, and 2,000;
  confirmation adds steps 4,000 and 8,000.
- [ ] Delayed sentinel step zero is excluded from write/eta/gamma summaries, and
  scale-canonicalized quantities compare M2 and M4 in common coordinates.
- [ ] Enabling diagnostics preserves outputs and gradients at the approved fp32
  tolerance and records its own elapsed time and artifact bytes.
- [ ] A small smoke invocation writes a complete native attempt bundle with
  configuration, data, source, clock, phase, lineage, diagnostic, artifact, and
  outcome evidence.
- [ ] Rootfs tests prove single-run ownership, required-field validation,
  diagnostic formulas/aggregation, instrumentation invariance, and failure
  outcomes; the Falcon suite and changed-file lint pass.

## Exclusions

- No matrix loop inside the single-attempt runner.
- No actual-shape GPU preflight or 2,000-step science training.
