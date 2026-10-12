# 10 - Produce one uncontaminated addition attempt

Type: task
Status: ready-for-agent
Blocked by: 09
Parent: ../spec.md

## What to build

Produce one complete addition smoke attempt whose training stream cannot see
any registered evaluation problem and whose result bundle exposes every metric
needed for the repaired Campaign B evaluation.

## Acceptance criteria

- [ ] Three fixed evaluation draws are mutually disjoint and contain eight
  commutation-aware problem identities per width for ID widths 1-16 and OOD
  widths 17-24.
- [ ] Every draw has a deterministic orientation, immutable manifest, stable
  digest, rendered prompt/target, and enough identity information to reproduce
  it exactly.
- [ ] The deterministic online training stream samples only widths 1-16 and
  rejects both orientations of every evaluation identity.
- [ ] Teacher-forced evaluation reports whole-suffix exact accuracy, suffix
  token accuracy, micro and width-macro aggregates, least-significant-first
  output-position accuracy, and outgoing-carry-count strata.
- [ ] Suffix masks exclude prompts and padding; carry count includes the
  outgoing carry from the most-significant operand column.
- [ ] One A0 smoke run of at most four optimizer steps at the dry-run shape
  completes end to end as a native Falcon attempt bundle with claim label
  `smoke`; if a GPU is needed, total device time is at most 0.01 B200-hours.
- [ ] Rootfs tests prove determinism, mutual disjointness, width balance,
  commutation exclusion, manifest stability, masking, aggregation, and bundle
  validation; the Falcon suite and changed-file lint pass.

## Exclusions

- No 2,000-step Campaign B evaluation runs.
- No GDN feasibility decision or transfer claim.
