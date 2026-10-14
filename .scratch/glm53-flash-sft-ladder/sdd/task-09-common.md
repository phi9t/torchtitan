# T09 common rules: GLM-5.3-Flash LoRA SFT on 4x B200 (three parallel directions)

Repo: /data02/home/philip.yang/workspace/torchtitan (branch ultron/mainline). Do NOT commit.
Read first: `env-facts.md` (MANDATORY), `task-08b-report.md` (rank-local loader, 8-GPU probe numbers),
`task-07b-report.md` (memory floor table), `torchtitan/experiments/glm53_flash/{checkpoint.py,sharded_hf_load.py,config_registry.py,model.py,sharding.py}`.

Problem: at 4 GPUs the bf16 frozen base is 145.9 GiB/GPU; the 8-GPU probe measured 72.99 GiB static + 58.6 GiB
other (all-gather of the largest unit 13.8 GiB, prefetch, activations under FullAC, Triton workspace, fragmentation)
= 131.56 GiB peak. B200 has ~178 GiB. Goal: a full-depth GLM-5.3-Flash LoRA SFT step loop on 4 GPUs.

## Ownership (three workers run concurrently; T08d runs the 8-GPU rung)
- Do NOT edit any existing file under `torchtitan/experiments/glm53_flash/` (T08d owns it) or other tickets' dirs.
- Your code goes ONLY in your own new package (named in your brief) with its own `config_registry.py`
  that imports/builds on `torchtitan.experiments.glm53_flash`. Subclass or wrap; if a shared change is truly required,
  describe it in the report as a proposed patch.
- Tests: your own `tests/` inside your package; also run `pytest -q torchtitan/experiments/glm53_flash/tests` to show no breakage.

## GPU etiquette (shared host; foreign jobs come and go on all 8 GPUs)
- Never kill or signal processes you did not start. Never touch other users' processes.
- Iterate cheaply first: unit tests, debug/mini flavors, and a TRUNCATED-DEPTH FULL-WIDTH config (e.g. first 8 layers:
  dense 0-2, MoE 3-7, including sparse layers 3 and 7; loader skips missing layers) on 1-4 GPUs.
- Full-depth 4-GPU launches: pick 4 GPUs with enough free memory (nvidia-smi), set CUDA_VISIBLE_DEVICES, and take the lock
  `flock -w 7200 /data05/philip.yang/glm53-ladder-cache/glm-fullsize.lock <launch cmd>` so only one full-size GLM job runs
  at a time. Do NOT launch full-depth while a T08d GLM torchrun is running (`ps -eo args | grep glm53_flash.config_registry`);
  wait and poll every 60 s instead. Keep each full-depth leg <= 45 min.
- Host RAM: `host_ram_guard.sh` kills GLM ranks if available RAM < 600 GB; sample per-rank RSS every 30 s into your log.
- Disk: /data05 has ~190 GB free. Do not download the FP8 HF checkpoint. Weights are staged at
  `/data05/philip.yang/glm53-ladder-cache/model-weights/zai-org/GLM-5.3-Flash-BF16` (rootfs path in T08b config print).
- Reuse the warmed Triton cache from `.scratch/glm53-flash-sft-ladder/evidence/task-08b/triton_cache_prewarm/`.

## Required evidence (same for every direction)
1. Focused unit tests for your mechanism (pass), plus the existing GLM tests still pass.
2. Truncated-depth parity: your path vs the plain bf16 path, same weights/inputs/seed: loss and grad norm for >= 3 steps
   (bitwise if your mechanism promises identity; otherwise max |delta| with a stated tolerance and why).
3. Full-depth 4-GPU run with the T08b data/config (seq 512, 32768 tokens/step, seed 42, LoRA rank 16): >= 12 steps,
   peak GPU memory per rank, host RSS per rank, step time for steps 3-12, loss/grad norm curve. Compare step-1 loss to the
   8-GPU probe (1.67569) and to T08d seed-42 if available.
4. Report `.scratch/glm53-flash-sft-ladder/sdd/task-09<x>-report.md`: mechanism, memory breakdown, numbers, claim labels
   (`smoke`, `deterministic_comparison` only if bitwise), environment_class, what remains for a 94-step 4-GPU rung.
Evidence goes under `.scratch/glm53-flash-sft-ladder/evidence/task-09<x>/` (large files -> /data05/philip.yang/glm53-ladder-cache/).
