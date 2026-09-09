# Step 5: local hero run

Type: task
Status: resolved
Blocked by: 05, 07
Parent: ../spec.md

## Requirements

- One larger run of the ticket-07 winner plus Transformer and
  GDN-style controls at the ticket-05 budget.
- Freeze Step 4 knobs. Do not search.
- Keep global batch and tokens/step matched across the three arms.
- Single node. Rootfs. Run-attempt evidence with `claim_label=
  representative_training`.
- If results disagree with the paper, do not retune here; open a
  new Step 4.

## Exclusions

- 49.2B / Table 1 claims
- Context parallel
- Falcon-2/3 or new alignments
- Promotion into `torchtitan/models/`

## Verification

- Three run-attempt reports (or an explicit skipped control with
  reason) and a short comparison note.
- Finite losses; checkpoints or an explicit no-ckpt decision from
  disk budget.

## Comments

- 2026-09-06: Pause/resume is required. Hero dumps use full train-state
  checkpoints (`interval=500`, `keep_latest_k=2`,
  `last_save_model_only=False`). `experiments/falcon/run.sh pause --arm A5`
  writes `{dump}/PAUSE`; the job finishes the current step, saves, and
  exits. Re-run the same `hero` / `hero-all` command to resume (stale
  PAUSE is cleared on start). `--fresh` refuses if a checkpoint exists.
  Disk floor 5G. Budget locked at 20k steps / 328M tokens, three arms on
  three B200s.

## Answer

All three matched local hero runs completed at 20,000 optimizer steps and
327,680,000 tokens per arm. A0 softmax reached val CE 4.4731 / PPL 87.63 in
894.08 s; frozen A5 reached 4.5543 / 95.04 in 1,166.27 s; A1 GDN reached
4.6000 / 99.48 in 36,525.88 s. Every outcome is finite and each dump contains
a full `step-20000` train-state checkpoint.

A5 stays frozen and was not tuned again. It trails A0 by 7.41 PPL and beats the
single A1 hero by 4.44 PPL while A1 takes 31.32x as long. Claim label:
`representative_training`, one seed, local 29.9M-parameter fp32 science shape,
plain FineWeb, 327.68M tokens. This is not replicated evaluation, paper-scale
training, or a Table 1 reproduction.

Raw outcomes:
`experiments/falcon/results/hero/hero_20k/{A0,A1,A5}_seed0/hero_outcome.json`.
Full report: `../sdd/task-08-report.md`.
