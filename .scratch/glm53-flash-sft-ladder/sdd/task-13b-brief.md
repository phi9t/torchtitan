# T13b brief: get GLM EP to a "good state" -- explain the grad-norm gap, fair baselines, EP8, full-depth timing
Read `env-facts.md` (MANDATORY), `task-13-report.md`, `task-17-report.md`, `task-09a-report.md` + `task-09b-report.md`
(same symptom: matching step-1 loss, grad norm off by ~3-6%), `.claude/skills/numerics_debugging/SKILL.md`,
`torchtitan/experiments/glm53_flash_ep/`, `torchtitan/distributed/utils.py` (grad-norm / clip across meshes),
`torchtitan/models/common/moe_sharding.py`, `MOE_SHARDING.md`. You own `torchtitan/experiments/glm53_flash_ep/`. Do NOT
commit. Never kill foreign processes. No big MoE runs except step 4 below (<= 15 steps).
1. Root-cause the grad-norm gap. T13: step-1 loss 10.81232 (plain FSDP) vs 10.81224 (EP4) but grad norm 1.789 vs 1.850.
   With identical data/seed/deterministic mode on the first-8-layer config, dump per-parameter gradients after step 1
   (full tensors gathered to CPU fp32; LoRA params + any non-frozen params) for: (a) FSDP-only + loop experts,
   (b) FSDP-only + grouped experts (`GLM53_FLASH_EXPERTS_IMPL=grouped`), (c) EP4, (d) EP8. Compare per-param norms and
   max |delta|. Also compute the global grad norm offline from the dumps and compare with each run's logged value. Decide:
   metric bug (e.g. sparse/edp mesh params counted twice or missed, Partial vs Replicate reduction, fp32 vs bf16 norm
   accumulation) vs real gradient difference. Fix it in the EP package if it is ours; if it is in core
   (`clip_grad_norm_` across dense + sparse meshes), write a minimal core repro test + proposed patch in the report.
   Also explain T09a/T09b's gaps if the same mechanism applies.
2. Fair speed baselines on the same config: (b) vs (c) vs (d), >= 10 steps each, ms/step + tokens/s + peak memory.
3. Numerics over 20 steps for (b) vs (d): loss and grad-norm curves (TensorBoard full precision), max |delta|.
4. Full depth (45 layers), EP8, grouped, LoRA, seq 512, 32,768 tokens/step: <= 15 steps when all 8 GPUs have >= 150 GiB
   free -- load time (rank-local loader reads only local experts?), ms/step, tokens/s, peak memory/rank, host RSS/rank.
   This is a timing/memory probe, not the rung.
Report `sdd/task-13b-report.md` with a clear "EP good state?" verdict and what remains (DeepEP via N01/T18).
