# 05 - Confirm the surviving contrast or stop

Type: task
Status: ready-for-agent
Blocked by: 04
Parent: ../spec.md

## What to build

Execute the screen's exact terminal decision: confirm only a surviving contrast
at 8,000 steps or record an explicit no-run closeout, then state what mechanism
the local evidence supports.

## Acceptance criteria

- [ ] The input decision is validated against the immutable 2,000-step bundles
  and cannot add an arm or contrast that the screen did not promote.
- [ ] A promoted contrast runs only the minimum required arms for seeds 0, 1,
  and 2 with unchanged model, data, optimizer, batch, evaluator, and metrics.
- [ ] Confirmation succeeds only when all three paired seed means retain the
  screen's direction and the 8,000-step grand-mean magnitude is at least 0.01
  CE. Otherwise the contrast closes as non-replication.
- [ ] All confirmation attempts, retries, diagnostics, checkpoints, and
  evaluation sum to at most three B200 GPU-hours.
- [ ] If nothing promotes, a complete no-run decision artifact closes the
  mechanism campaign and unblocks the external Campaign B closeout gate.
- [ ] Focused rootfs tests cover exact threshold boundaries, direction reversal,
  missing seeds/regions, excess compute, undeclared-arm rejection, and the
  no-run terminal branch before any confirmation spend.
- [ ] The final decision classifies the evidence as
  parameterization/preconditioning, an alignment effect, non-replication, or an
  implementation walk-back; state metrics explain but cannot overrule the scale
  invariant.
- [ ] Facts, inferences, hypotheses, omissions, compute, disk use, failed
  attempts, and every source bundle are explicit in the mechanism report and
  campaign ledgers.
- [ ] The closeout recommends only whether a new paper-alignment design gate is
  warranted; it does not create or authorize that implementation campaign.

## Exclusions

- No hero-scale or paper-scale run, speculative `ctxeta`, Falcon-3A, or retuning
  of Campaign B's frozen A5 arm.
- No commit, push, pull request, or merge without separate authority.
