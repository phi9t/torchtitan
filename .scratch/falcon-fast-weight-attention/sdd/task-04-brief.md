# Ticket 04 brief

Read first: `.scratch/falcon-fast-weight-attention/issues/04-step2-systems.md`
Depends on ticket 03 (`alignment` switch on `falcon_recurrent_forward`).

## Do

1. Add `falcon_masked_parallel_forward` with the same signature as the
   recurrent oracle (including `alignment`).
2. TDD: recurrent ≡ masked-parallel on tiny random and worked-example
   tensors for delayed + same-step, falcon1 + falcon1a, fp32.
3. Debug Trainer artifact via rootfs:
   `MODULE=falcon CONFIG=falcon_tiny_overfit NGPU=1 COMM_MODE=fake_backend`
   or an equivalent few-step path. Finite loss.
4. Report file: `.scratch/falcon-fast-weight-attention/sdd/task-04-report.md`

## Do not

Commit. Chunk-parallel/FLA. FineWeb. Mini-K3 / qwen3_5 edits.
