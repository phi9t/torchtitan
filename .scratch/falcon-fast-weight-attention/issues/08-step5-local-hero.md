# Step 5: local hero run

Type: task
Status: open
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
