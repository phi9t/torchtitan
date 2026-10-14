# S07 brief: architecture ladder L1-L4 at speedrun scale (qwen3 -> qwen3-moe -> mini-kimi-k3 -> mini-glm-5.3-flash)
Read `../spec.md` ("Architecture ladder" section is binding), `s01-report.md`, `s05-report.md`,
`.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY), `.scratch/mini-kimi-k3-replication/{map.md,model-route.md}`,
`torchtitan/models/{qwen3,kimi_k3}/`, `torchtitan/experiments/glm53_flash/{flavors.py,model.py}`. Do NOT commit.

Ownership: new package `torchtitan/experiments/glm53_arch_ladder/` (config_registry with one config per rung, mini
flavors built from the real model families' config builders, a shared FineWeb data/val setup reused from
`torchtitan/experiments/glm53_arch_speedrun/` by import -- do not edit that package, S06 owns it) and
`experiments/glm53_arch_ladder/` (runner `run_rung.sh`, logs/results gitignored). Do not edit `torchtitan/models/*` or
`torchtitan/experiments/glm53_flash/`; if a family cannot be built small via config, wrap/subclass in your package and
report the needed upstream hook.

GPU rules: 1 GPU per run, 24 GiB per-process cap, at most 3 concurrent runs, NEVER use GPUs in Exclusive_Process
compute mode or with foreign processes (check `nvidia-smi --query-gpu=index,compute_mode` + compute-apps before each
launch; GPUs 2,3 are currently exclusive/foreign), never kill foreign processes, `shared_gpu` label. If the 8-GPU GLM
rung-4 job (`torchtitan.experiments.glm53_flash.config_registry`) is running, still OK to co-run within the cap.

Steps:
1. Build L1-L4 mini configs; print param tables (total, non-embedding, active non-embedding) and hit the matching rule.
   Use GPT-2 vocab (50257, padded) for all rungs so val CE is comparable.
2. Per rung: unit test (build, fwd+bwd, Muon/AdamW partition covers every param exactly once), 2-step smoke, then a
   300-step overfit check on a fixed small bank (CE collapse, lr=0 flat) -- Step-3 gate per rung.
3. Full runs: L1, L2, L3, L4 x seeds {0,1} at the S05 budget. If a rung is < 0.3x L0 tokens/s, run it at a declared
   reduced budget as a side table instead of stalling (and say so). Publish each run bundle to HDFS
   `torchtitan/runs/glm53-arch-ladder/<run_id>/` with md5.
4. Table: rung, params, val CE mean +/- std at budget (and at matched intermediate token counts), tokens/s, memory,
   deltas vs L0 and vs previous rung with the 2x seed-std rule.
Write `.scratch/glm53-arch-speedrun/sdd/s07-report.md` and add a "Architecture ladder" results section to
`docs/research/2026-10-14-glm-5.3-flash-arch-speedrun-ablations.md`.
