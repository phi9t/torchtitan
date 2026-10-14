# S07b brief: ladder fixes before epoch runs -- diagnose L2 MoE underperformance, make L3/L4 fast, rerun at full budget
Read `../spec.md` (Architecture ladder + Epoch rule), `s07-report.md`, `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md`
(MANDATORY, HDFS-strict), `.scratch/glm53-flash-sft-ladder/sdd/task-17-report.md` (GLM grouped experts:
`GLM53_FLASH_EXPERTS_IMPL=grouped`, 39.5x on the MoE layer), `task-k02-report.md` / K03 (indexer aux-loss cost; K03 adds
an aux query-stride knob and compiled indexer region). You own `torchtitan/experiments/glm53_arch_ladder/` and
`experiments/glm53_arch_ladder/`. Do NOT commit. 1 GPU/run, 24 GiB cap, <= 3 concurrent, never kill other jobs.
1. L2 Qwen3-MoE was worse than dense at matched active params (4.044 vs L1 3.869, L0 3.903) -- likely a harness issue.
   Check, with evidence: which optimizer each MoE param gets (Muon on 3D [E,out,in] expert weights? per-expert
   orthogonalization vs flattened? router/gate on AdamW? LR/WD per group), router load balance over training (expert
   token fractions, entropy, dropped tokens, aux-loss/bias update actually active), expert init scale, grad norms per
   group, capacity/dropless, top-k and shared expert settings vs qwen3 MoE reference. Fix real bugs; run short A/B
   (e.g. 33.5M tokens, 2 seeds) to confirm before a full-budget rerun. If MoE is legitimately worse at this scale, show it
   (e.g. LR sweep 3 points) rather than assert it.
2. L4 mini-GLM: set `GLM53_FLASH_EXPERTS_IMPL=grouped` (T17) and adopt K03's knobs when available (declare the aux
   query stride used); L3 Kimi: profile 1 step and fix the top hot spot if it is ladder-owned (or report). Target each
   >= 0.3x L0 tokens/s.
3. Full-budget reruns (268,435,456 tokens, seeds 0,1) for L2 (after fix), L3, L4; publish bundles to HDFS with md5; update
   the ladder table in `docs/research/2026-10-14-glm-5.3-flash-arch-speedrun-ablations.md`.
Report `.scratch/glm53-arch-speedrun/sdd/s07b-report.md`. S09 (epoch runs) is blocked on this.
