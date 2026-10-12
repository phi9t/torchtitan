# Ticket 05 brief

Read first: `.scratch/falcon-fast-weight-attention/issues/05-data-readiness.md`
Parent spec: `.scratch/falcon-fast-weight-attention/spec.md`

## Fit

Research-only. Unblocks Step 4/5 budgets. Do not train. Do not download
a 50B corpus.

## Do

Inventory FineWeb-Edu and C4 already on this host. Reuse loaders under
`experiments/qwen3_fineweb_hsdp_tp/`, `torchtitan/hf_datasets/`, and any
modded_nanogpt FineWeb manifests. Measure free disk (`df -h` on
`/data02` and the repo). Recommend concrete Step 4 and Step 5 token
budgets, seq_len, and global batch that keep tokens/step matched across
Falcon / GDN-style / Transformer controls.

If nothing fits, name the blocker and the smallest fetch that would
unblock Step 4 — do not fetch it.

## Write

Append `## Answer` to
`.scratch/falcon-fast-weight-attention/issues/05-data-readiness.md`
and copy the same note to
`.scratch/falcon-fast-weight-attention/sdd/task-05-report.md`.

Set ticket Status to resolved only if the answer is complete.

## Do not

- Edit `torchtitan/experiments/falcon/` Python
- Download datasets
- Commit
