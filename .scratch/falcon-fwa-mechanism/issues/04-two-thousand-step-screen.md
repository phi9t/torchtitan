# 04 - Run and decide the 2k mechanism screen

Type: task
Status: ready-for-agent
Blocked by: 03
Parent: ../spec.md

## What to build

Run M0-M4 for three matched seeds at 2,000 optimizer steps and emit one
validator-backed stop, walk-back, or promotion decision within the approved
screen budget.

## Acceptance criteria

- [ ] Fifteen logical training runs produce fifteen independent attempt bundles
  using the fixed science model, data, optimizer, batch, region registry, probe
  milestones, seeds, and data order.
- [ ] Total actual-shape preflight, training, retry, diagnostic, checkpoint, and
  evaluation device time remains at or below one B200 GPU-hour.
- [ ] Each contrast uses per-seed means of paired fixed-region CE differences;
  a directional effect requires all three signs to agree and grand-mean
  magnitude of at least 0.01 CE.
- [ ] M2/M4 practical equivalence requires every paired seed mean within
  `[-0.01, 0.01]` CE and the grand mean within `[-0.005, 0.005]` CE.
- [ ] Region ranges are reported only as content-heterogeneity diagnostics, not
  as uncertainty or an equivalence test.
- [ ] A failed M2/M4 seam invariant or practical-equivalence gate produces an
  implementation walk-back rather than a distinct-mechanism claim.
- [ ] If scale compensation explains the normalization contrast, the decision
  requires M2/M4 practical equivalence and lower CE for both M2-versus-M0 and
  M4-versus-M0 under the directional rule, then stops that branch as
  parameterization/preconditioning with no larger normalization run.
- [ ] Alignment promotion considers only M1-versus-M0 and M3-versus-M2; only the
  minimum arms needed for a surviving declared contrast advance. Null,
  ambiguous, over-budget, and incomplete outcomes terminate explicitly.

## Exclusions

- No post-hoc threshold changes, replacement arms, GDN, Falcon-1, or paper
  variants.
- No 8,000-step run before the signed screen decision exists.
