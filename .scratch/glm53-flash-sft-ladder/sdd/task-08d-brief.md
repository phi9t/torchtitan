# T08d brief: rung 4 full run -- GLM-5.3-Flash LoRA SFT, resume, eval, publish

Repo: /data02/home/philip.yang/workspace/torchtitan (branch ultron/mainline). Do NOT commit.
Read first: `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY), `sdd/task-08-brief.md` (requirements 1-6),
`sdd/task-08b-brief.md` (requirements 3-5 are THIS ticket), `sdd/task-08b-report.md` (what works now, committed in db23be429).

T08b proved the unblock: rank-local HF load (21.7 s, rank RSS 3.75 GiB), one optimizer step, loss 1.676, peak 131.6 GiB/GPU.
This ticket finishes the rung. Do NOT stop at a probe again. Completing the gates below is the whole job.

## GPU gate (this replaces "stop because GPUs are shared")
- The GLM run needs all 8 GPUs with >= 150 GiB free each. Before every 8-GPU launch, poll `nvidia-smi` every 60 s until that
  holds (write a small wait helper; log each poll decision compactly). WAIT; do not give up while GPUs are busy. T06 (our own
  Qwen3-30B worker) is finishing on GPUs 0,1,4,5. A foreign root process may appear on any GPU: never kill or signal it, just wait.
- Record `environment_class` honestly per leg (`exclusive` only if no foreign GPU procs during the leg, else `shared_gpu`).
- Reuse the warmed Triton cache from T08b (`.scratch/glm53-flash-sft-ladder/evidence/task-08b/triton_cache_prewarm/`) to avoid the
  KDA l2norm autotune OOM; keep the host RAM guard + 30 s per-rank RSS sampling (rank RSS < 120 GiB).
- `/data05` has ~135 GB free: keep checkpoints trainable-only (small); prune local payloads after verified HDFS publish.

## Gates (all required; seed 42 then seed 43; 94 steps; 64x512 tokens/step; config knobs unchanged from T08b)
1. Seed 42 run with `GLM53_FLASH_DETERMINISTIC=1` if it works (else report why and use max |delta|): full 94 steps, checkpoint
   interval 31. Record load time, >= 10 steady steps of throughput (tps, step time) and per-GPU peak memory, full loss curve.
2. Resume gate: a second seed-42 leg that stops after step 62 (same mechanism as rung 0 `run_rung0.sh`), publish the step-62
   trainable-only checkpoint to HDFS (`torchtitan/checkpoints/glm53-ladder/rung4/<run_id>/<attempt_id>/step62`, verify md5),
   delete the local checkpoint, stage it back from HDFS, resume to 94, compare TensorBoard losses 63-94 against gate 1
   (`torchtitan/experiments/glm53_flash_sft_ladder/compare_resume_loss.py` or equivalent).
3. Seed 43 run: 94 steps (no resume leg needed).
4. Adapter export for both seeds (LoRA tensors + PEFT-style config with rank/alpha/targets, HF key names).
5. Eval on Countdown dev: base, seed-42 adapter, seed-43 adapter. First check (quickly, <= 30 min) whether vLLM in the default
   rootfs env can serve `GLM-5.3-Flash-BF16` + LoRA on 8 GPUs; if not, write `torchtitan/experiments/glm53_flash/generate.py`
   (TorchTitan-native sampling with the same sharded model + rank-local loader + adapter load; temperature 0.8, top-p 0.95,
   max 192 new tokens, GLM chat template with the countdown prompt; batch all prompts per forward). Declare the subset up front
   (e.g. first 100 dev problems x 4 samples), use the same subset for all three, score with the countdown verifier, report
   pass@1 and strict-format pass@1 with bootstrap CIs. Claim label `replicated_eval` only if both seeds beat base.
6. Publish adapters, eval outputs, logs, TensorBoard, config prints, manifests to HDFS under
   `torchtitan/runs/glm53-ladder/rung4/<run_id>/<attempt_id>/` with `_COMPLETE` and md5 verification (use
   `experiments/glm53_flash_sft_ladder/publish.sh`).

## Exclusions
No core edits (report proposed core changes). No commit. No knob changes. Never kill foreign processes. Do not touch
`torchtitan/experiments/glm53_flash_bf16_diag/` or the Qwen ladder files owned by T06.

## Report
`.scratch/glm53-flash-sft-ladder/sdd/task-08d-report.md`: every gate with PASS/FAIL/NOT RUN and evidence path, HDFS URIs,
load time, rank RSS max, steady throughput, per-GPU memory, loss curves, resume comparison, eval table, environment_class per leg.
