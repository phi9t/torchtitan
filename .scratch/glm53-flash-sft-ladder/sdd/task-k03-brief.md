# K03 brief: make the indexer arms fast -- aux-loss subsampling + compiled indexer region (+ fused scoring if needed)
Read `env-facts.md` (MANDATORY), `task-k02-report.md` (profile: aux-loss teacher/KL ~75% of A4 time, indexer scoring
~16%, attention ~1%; host syncs fixed), `task-k01-report.md`, `.claude/skills/local_compile_perf/SKILL.md`,
`torchtitan/experiments/glm53_arch_speedrun/elements/attention.py`, `torchtitan/experiments/glm53_arch_speedrun/model.py`,
`.scratch/glm53-arch-speedrun/spec.md` (ablation design: indexer aux weight / warmup are declared dose knobs).
Do NOT commit. 1 GPU, <= 24 GiB, never kill other jobs. S10 runs cells concurrently using this package: every change must be
behind new config fields whose DEFAULTS reproduce current behavior bitwise, except pure performance refactors proven
bitwise/within-bf16 by tests. Do not touch runner/scheduler/queue files.
1. Aux-loss query subsampling: new knob `indexer_aux_query_stride` (default 1 = current) and/or
   `indexer_aux_num_queries` (random per step, seeded): compute the teacher/KL only on the sampled query rows (unbiased
   estimator of the mean-over-rows loss). Test: estimator mean matches full loss over many seeds; gradients' expectation.
2. Compile: make the indexer scoring + IndexPool + top-k + aux-loss teacher a `local_compile` region (follow the skill:
   measure, find graph breaks, fix them), so fp32 elementwise work on [B,Hidx,L,L] fuses; consider computing scores in
   bf16 with fp32 accumulation if the reference/HF GLM does (cite).
3. Teacher cost: compute the head-summed dense attention distribution for the sampled rows in one batched matmul +
   logsumexp (no Python chunk loop), reusing q/k already projected for attention.
4. Scoring: if still > 25% after 1-3, check DeepGEMM (`fp8_mqa_logits` / DSA indexer logits kernels) and
   FlashMLA for a reusable fused indexer-logits kernel on sm_100; otherwise write a precise K04 kernel spec.
5. Measure A4/A5/G tokens/s at seq 4096 (2-step smokes + 20-step) for stride in {1, 8, 16} and report val-CE impact on a
   short run (e.g. 33.5M tokens, 2 seeds) -- stride is a dose knob; the ablation declares which value it uses.
Targets: A4/A5 >= 0.5x B0, G >= 0.3x B0 at a declared stride. Report `sdd/task-k03-report.md`.
