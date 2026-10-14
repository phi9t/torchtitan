# K02 brief: profile-first -- why are indexer arms 20x slower than B0? fix the real hot spot
Read `env-facts.md` (MANDATORY), `task-k01-report.md` (key fact: masked-dense attention is only ~2.4 ms/call at L=4096, so
attention is NOT the bottleneck; the K01 Triton kernel did not change A4/A5 tokens/s ~4.6k vs B0 ~100k),
`.scratch/glm53-arch-speedrun/sdd/s06-report.md`, `.claude/skills/local_compile_perf/SKILL.md`,
`torchtitan/experiments/glm53_arch_speedrun/elements/attention.py` (LightningIndexer, IndexPool, score_pools,
indexer_auxiliary_loss, TopKSparseCausalAttention), `torchtitan/experiments/glm53_arch_speedrun/model.py`.
Do NOT commit. 1 GPU, <= 24 GiB, never kill other jobs. S10 is concurrently editing files in this package and running
cells: make your changes in `elements/attention.py` only behind new config values or fully parity-preserving
refactors, and coordinate via small diffs; do not touch scheduler/runner/queue files.
1. Profile A4 (and A5, G) at seq 4096 with torch.profiler for 3 steps after warmup (CUDA time per op + CPU gaps, kernel
   counts, memory). Produce a ranked table: indexer scoring, top-k, IndexPool, aux-loss teacher/KL, masked attention,
   chunk loops, host syncs, recompute under AC, compile graph breaks.
2. Fix the top items with parity-preserving changes, e.g.: compile the indexer region (local_compile), fuse score+mask+topk,
   compute the aux-loss teacher from the attention LSE instead of a second dense pass, avoid fp32 [B,H,L,L]
   materialization, remove Python chunk loops / .item() syncs, reuse the selection across layers if the spec allows (it does
   not by default -- keep per-layer). Keep the aux objective and selection semantics bitwise or within bf16 tolerance, with
   tests.
3. Target A4/A5 >= 0.5x B0 and G >= 0.3x B0 at seq 4096 (B0 ~100k tok/s shared). Report before/after tables and the
   profiler traces location. If the remaining cost truly needs a custom kernel, specify it precisely (inputs, outputs,
   flops/bytes) for a K03 ticket.
Report `.scratch/glm53-flash-sft-ladder/sdd/task-k02-report.md`.
