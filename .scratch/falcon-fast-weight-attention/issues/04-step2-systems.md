# Step 2: systems / view identity

Type: task
Status: resolved
Blocked by: 03
Parent: ../spec.md

## Requirements

- Implement masked-parallel Falcon-1 / 1A that is equivalent to the
  recurrent oracle on tiny `(B, L, N, K, V)` cases (fp32, documented
  tolerance).
- Cover delayed and same-step alignments from ticket 03.
- Produce a debug Trainer artifact: `falcon_tiny_overfit` (or a slightly
  larger debug config) completes N>1 steps with finite loss under
  rootfs + fake-backend or 1 GPU.
- Record that there is no train–inference stack; the identity gate
  replaces Mercor's logprob-diff < 0.03.

## Exclusions

- Chunk-parallel / WY / FLA (later acceleration, not a science blocker)
- Context parallel
- Compile/FSDP required for pass (optional note only)
- Step 4 data or ablations

## Verification

- Recurrent ≡ masked-parallel tests in rootfs.
- Trainer debug report checked in or written under
  `experiments/falcon/results/` (gitignored) with a pointer in the
  ticket Answer.
- Claim labels: `deterministic_comparison`, `smoke`.

## Answer

Added `falcon_masked_parallel_forward` (same signature as the recurrent oracle,
including `alignment`). Rootfs identity gate recurrent == masked-parallel is
green for `{falcon1,falcon1a} x {delayed,same_step}` in fp32 (tol 1e-5 random /
1e-6 worked example). Debug Trainer artifact: `falcon_tiny_overfit` runs 5 steps
under fake backend with finite loss
(`experiments/falcon/smoke_trainer.py`; log in
`experiments/falcon/results/task04_smoke_trainer.log`, gitignored). Rootfs
`test_falcon_*.py`: 31 passed. The identity gate replaces Mercor's logprob-diff
< 0.03; there is no separate train-inference stack. Report:
`sdd/task-04-report.md`. Claim labels: `deterministic_comparison`, `smoke`.
