# 13 - Close Campaign B with repaired evaluation

Type: task
Status: ready-for-agent
Blocked by: 10, 11, 12
External gate: Falcon mechanism 05 - terminal mechanism decision (human-enforced)
Parent: ../spec.md

## What to build

Run the approved replicated evaluation matrix, validate its evidence, and
publish the final Campaign B table with claims no stronger than the data.

## Preserved Step-6 intent

- Evaluate held-out LM CE/PPL for all three Step-5 hero arms using three fixed
  regions as the declared replication axis.
- Run separate addition training for the selected mixers and report replicated
  ID and width-OOD teacher-forced suffix metrics.
- Preserve the A5-versus-GDN LM no-regression comparison.
- Label any optional zero-shot hero-checkpoint addition probe as harsher than
  the paper metric and non-blocking.
- Close with a written, traceable evidence table; teacher-forced accuracy is not
  generation, and this local campaign is not paper Table 1.

## Acceptance criteria

- [ ] A0 and A5 train for 2,000 optimizer steps at seeds 0, 1, and 2 using
  matched deterministic online streams and a predeclared final checkpoint.
- [ ] Before launch, a bundle-backed preflight projects the complete A0/A5
  matrix at no more than one B200 GPU-hour. Actual preflight, training, failed
  attempts, retries, diagnostics, checkpoints, and evaluation remain within
  that aggregate ceiling, with at most one retry per logical run; exceeding
  either bound requires a new human decision.
- [ ] GDN runs the same matrix only when ticket 12 authorizes it; otherwise the
  final table carries the exact omission record.
- [ ] Every final addition checkpoint is evaluated on all three fixed draws and
  reports exact, token, width-macro, output-position, and carry-count metrics.
- [ ] A0, A5, and A1 hero checkpoints are evaluated over the same three fixed LM
  regions without changing their historical artifacts.
- [ ] Practical LM no-regression means aggregate A5 CE is no more than 0.01
  worse than aggregate A1 CE across the fixed regions. A larger gap is recorded
  as `lm_regression`, not hidden or treated as an incomplete evaluation.
- [ ] A campaign validator rejects missing seeds/regions/draws, duplicate
  logical runs, contaminated splits, mismatched configurations, excess compute,
  invalid claim labels, and unexplained omissions.
- [ ] Focused rootfs validator tests pin every coverage, contamination,
  compute/retry, no-regression, claim-label, and omission boundary before the
  full evaluation matrix launches.
- [ ] The closeout separates facts, inferences, hypotheses, and omissions;
  links every number to an attempt or import record; and updates the campaign
  map and progress ledger.
- [ ] If proper OOD exact-suffix accuracy is zero throughout, Campaign B closes
  as completed replicated evaluation with conclusion `no_transfer`, not as a
  successful transfer gate.
- [ ] No paper-aligned implementation or larger hero is authorized by this
  ticket; the terminal mechanism decision only recommends whether a new design
  gate is warranted.

## Exclusions

- Terminal-Bench, the eight-task LM evaluation suite, paper-scale training, and
  hero replication by new training seeds.
- Retuning A5, guessing `ctxeta`, or treating a missing GDN control as a tie.
