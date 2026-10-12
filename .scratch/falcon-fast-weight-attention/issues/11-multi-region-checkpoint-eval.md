# 11 - Evaluate one checkpoint across fixed LM regions

Type: task
Status: ready-for-agent
Blocked by: 09
Parent: ../spec.md

## What to build

Restore an existing hero checkpoint read-only, evaluate it over three fixed
nonoverlapping held-out token regions, and emit traceable per-region and
aggregate evidence in a new evaluation attempt.

## Acceptance criteria

- [ ] A versioned region registry fixes three nonoverlapping ranges, token
  counts, validation-source digest, and aggregate weighting before evaluation.
- [ ] Evaluation restores an existing checkpoint into a separate output
  location and cannot mutate the historical hero dump or checkpoint.
- [ ] Results report token-weighted CE per region, aggregate CE, derived PPL,
  finite-value status, checkpoint lineage, and the region-registry digest.
- [ ] One existing A0 hero checkpoint completes the path end to end in a valid
  native evaluation-attempt bundle.
- [ ] Re-running in the same declared deterministic environment produces
  bitwise-identical per-region token sums, counts, CE, and aggregate values in
  a distinct attempt without duplicating the logical evaluation identity.
- [ ] Rootfs tests cover region overlap, wrong source/checkpoint digests,
  aggregate weighting, immutable restore behavior, and bundle validation; the
  Falcon suite and changed-file lint pass.

## Exclusions

- No hero retraining or checkpoint selection by evaluation score.
- No evaluation of all three hero arms; ticket 13 owns the campaign matrix.
