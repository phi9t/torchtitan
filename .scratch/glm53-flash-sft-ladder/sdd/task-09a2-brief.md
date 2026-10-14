# T09a2 brief: finish T09a (4-GPU weight streaming) -- grad-norm discrepancy, then full depth
Read `task-09-common.md` (binding), `task-09a-brief.md`, `task-09a-report.md`. Package `torchtitan/experiments/glm53_flash_4gpu_stream/`
is yours. Do NOT commit.
1. Root-cause first: truncated-depth step 1 had IDENTICAL loss (10.79574) but grad norm 1.7734 (stream) vs 1.8828 (plain).
   Offload moves bytes, not math, so this is either a grad-norm computation difference under CPUOffloadPolicy (e.g.
   which params/dtypes/devices enter the norm, fp32 vs bf16 reduction) or missing/altered LoRA grads. Compare per-parameter
   LoRA grads after step 1 (deterministic mode, same seed/data) between plain and stream on the truncated config;
   report max |delta| per param and which params differ. Fix it in your package if it is a real bug; if it is only the
   metric, prove it and explain.
2. Full depth: run the 4-GPU 12-step stream evidence through the lock (`flock -w 14400 ...glm-fullsize.lock`); WAIT for
   it -- do not give up because it is held (T09c/T09b hold/queue it). Sweep local batch {4, 8, 16} (AC on/off as memory
   allows), record step time, H2D GiB/s, GPU peak, host RSS per rank, loss/grad norm. Report `task-09a2-report.md`.
