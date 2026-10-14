# S06 brief: Step-3 overfit gate + make slow arms fast enough for the matrix
Read `../spec.md`, `s05-report.md` (binding numbers), `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY).
You own `torchtitan/experiments/glm53_arch_speedrun/` and `experiments/glm53_arch_speedrun/`. Do NOT commit. 1 GPU per
run, 24 GiB cap, `shared_gpu` labels, never kill foreign processes, do not start the scheduler.

1. Step-3 gate FIRST (cheap): for B0, A1..A6 and G at seq 1024: fixed small FineWeb bank (e.g. 64 sequences x 1024),
   >= 300 steps, report CE collapse (< 0.5 or > 90% drop), lr=0 control flat, and that arm curves are not identical.
   Fast arms can run in parallel on different GPUs. Write the gate table.
2. Throughput (S05 seq-4096 numbers: B0 216k tok/s; A1 mHC 21k; A4 indexer 2.4k; A5 2.3k; G 7k):
   - A4/A5 sparse attention: at speedrun scale it is mathematically equivalent to dense attention with a mask that keeps
     only each query's selected keys (plus causal). Implement that path (SDPA bool mask or flex_attention mask from the
     top-k indices; indexer scoring as batched einsum on GPU, topk on GPU; IndexPool on GPU) and prove fwd+bwd parity vs
     the S04 gather reference at seq 1024 and 4096. Keep the aux-loss path. Target >= 0.5x B0 tokens/s.
   - A1 mHC: profile (torch.profiler, short) and fix the hot spot (likely uncompiled Sinkhorn loop, fp32 copies, or 4x
     residual traffic): compile the mHC region, fuse, avoid per-iteration allocations. Target >= 0.5x B0.
   - G: re-measure after both; target >= 0.3x B0. KDA (A3) at 135k is acceptable.
   For each arm, raise microbatch back up if memory allows under 24 GiB.
3. Re-run the 2-step smoke table (all arms, seq 1024/4096) with the new paths.
Report `.scratch/glm53-arch-speedrun/sdd/s06-report.md` ending with literal lines `S06_STEP3_GATE=PASS|FAIL` and
`S06_SLOW_ARMS=<comma list of arms still < 0.3x B0 or none>`.
