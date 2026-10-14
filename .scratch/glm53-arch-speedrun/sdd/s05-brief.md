# S05 brief: make B0 a real speedrun baseline, wire the GLM elements in, run the Step-3 gate
Read `../spec.md` (incl. "Ablation design"), `s01-report.md`, `s04-report.md`, `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md`
(MANDATORY). You now own `torchtitan/experiments/glm53_arch_speedrun/` (model, config, elements) and
`experiments/glm53_arch_speedrun/` (runner, queue, scheduler). Do NOT commit. 24 GiB per-process GPU cap, 1 GPU per
run, never kill foreign processes, label timing `shared_gpu`. Do not start the scheduler yourself.

Problem: S01's B0 ran 8.4M tokens (16 steps of 524k tokens) to val CE 6.96 -- too early for architecture comparisons --
at 53k tokens/s using only 4.2 GiB (local batch 4 x 1024, 128 grad-accum micro-steps).
1. Throughput: raise local batch to fill the 24 GiB cap at seq 1024 and 4096, confirm torch.compile + bf16 autocast +
   flash/SDPA attention + fused loss are active, cut tokens/step to 262,144 (or justify another value) so a run has
   hundreds of optimizer steps. Target >= 200k tokens/s/GPU for B0 at seq 1024 on a quiet GPU; report what you get.
2. Budget: choose the per-run token budget so one B0 run is ~15-25 min on one GPU (likely 200-400M tokens) and val CE
   is well past the early-training regime (report the curve; modded-nanogpt-classic reaches ~3.3-3.5 by ~1B tokens).
   Re-run B0 x3 seeds at seq 4096 (the matrix shape) and x1 at seq 1024; report mean/std and the declared target.
   Update `queue/cells.jsonl` budget fields via the generator, and the doc's Calibration section.
3. Wire elements into the model via config fields: mHC residual, NoPE, KDA 3:1 hybrid (Attention Gym KDA), indexer
   top-k sparse attention with the S04 aux loss collected into the training loss (log it separately), IndexPool, MTP
   (aux loss logged separately), all dose knobs. Every one of the 48 F1-F5 cells must build and run 2 steps at seq 4096.
   Muon/AdamW partition must still cover every param exactly once for every arm (test it).
4. Step-3 gate: for B0, A1-A6 and G: overfit a fixed small FineWeb bank (CE collapse), lr=0 flat, curves diverge
   across arms; plus tokens/s and peak memory per arm at seq 4096 and 1024. Flag arms > 2x slower than B0 for a side table.
Report `.scratch/glm53-arch-speedrun/sdd/s05-report.md` ending with the literal lines `S05_STEP3_GATE=PASS|FAIL` and
`S05_BUDGET_TOKENS=<n>`; the scheduler will be pointed at this report.
