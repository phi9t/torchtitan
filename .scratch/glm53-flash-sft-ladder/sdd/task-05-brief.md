# T05 brief: rung 1 -- Qwen3 8B LoRA SFT on countdown-sft/v1 (generalise the rung runner)

Repo: /data02/home/philip.yang/workspace/torchtitan (branch ultron/mainline). Do NOT commit.
Read first: `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY), `spec.md`, `sdd/task-02-brief.md` and
`sdd/task-02-report.md` (rung 0 = the template you must follow), `experiments/glm53_flash_sft_ladder/run_rung0.sh`,
`torchtitan/experiments/glm53_flash_sft_ladder/{config_registry.py,parity.py}`.

## Requirements
1. Generalise rung 0's runner into `experiments/glm53_flash_sft_ladder/run_rung.sh` driven by `MODEL_KEY` (e.g. `qwen3_0_6b`,
   `qwen3_1_7b`, `qwen3_8b`, later `qwen3_30b_a3b`) without breaking `run_rung0.sh` (keep it as a thin wrapper or leave it
   untouched). Add `qwen3_8b_countdown_lora` to the experiment config registry with the SAME frozen knobs as rung 0
   (rank 16, alpha 64, targets wqkv/wo/w13/w2, lr 1e-4, 64x512 tokens/step, seq 512, 94 steps) -- no re-tuning.
2. Inputs: Qwen3-8B is already staged with `.staged.json` at `/data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-8B`
   (HDFS `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218`).
   Run `stage.sh` anyway (it must be a fast no-op) and record URIs + manifest md5 in the run manifest.
3. Gates (same as rung 0 minus LoRA-vs-full and the GPU-count ladder): HF logits parity (parity.py), 2 seeds (42, 43) of
   LoRA SFT on 2-4 GPUs, resume-from-HDFS-after-cache-wipe gate for one seed (bitwise loss check steps after resume),
   >=10 steady steps throughput/memory (`shared_gpu` label), adapter export, dev eval base vs adapter (500 problems,
   32 rollouts) with vLLM `--gpu-memory-utilization` sized to fit next to the shared job, publish adapters/eval/logs to
   HDFS, delete published local checkpoints.
4. Compare with rung 0 in a table: base vs adapter dev pass@1 / pass@32 / strict pass@1 for 0.6B, 1.7B, 8B (2 seeds each,
   mean and per-seed). Declare: does SFT improvement hold / grow with scale? Ties inside seed noise are ties.

## Exclusions
No knob changes, no core edits, no commit, never kill other processes, keep /data05 usage < 60 GB for your outputs.

## Report
`.scratch/glm53-flash-sft-ladder/sdd/task-05-report.md` with gate pass/fail, numbers, HDFS URIs, claim labels
(`replicated_eval` for the 2-seed table, `deterministic_comparison` for resume), no commit.
