# Step 6: eval, transfer, campaign close

Type: task
Status: open
Blocked by: 06, 08
Parent: ../spec.md

## Requirements

- Held-out LM CE/PPL for the three Step 5 arms, 2–3 seeds or 3 shards.
- Separate addition trains (paper protocol) for those mixers: ID val
  acc and OOD teacher-forced suffix acc, replicated.
- Optional extra: zero-shot addition on the LM hero checkpoint,
  labeled as harsher and not the paper metric.
- No-regression: Falcon LM PPL stays in band with the GDN-style
  control (not addition-only).
- Close the campaign with a table and claim labels
  `replicated_eval` + `transfer_eval`.

## Exclusions

- Terminal-Bench, lm-eval 8-task suite (optional later, not required)
- Calling teacher-forced addition "generation"
- Calling the local hero "Table 1"

## Verification

- Written close-out note linked from `map.md`.
- Every number has a run-attempt or report path.
