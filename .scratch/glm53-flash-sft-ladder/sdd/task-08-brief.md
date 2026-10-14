# T08 brief: rung 4 -- GLM-5.3-Flash LoRA SFT on countdown-sft/v1 (8x B200)

Repo: /data02/home/philip.yang/workspace/torchtitan (branch ultron/mainline). Do NOT commit.
Read first: `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY, incl. GPU UPDATE), `spec.md` (Done definition),
`map.md` (D5 indexer frozen, D7 CPU-offload fallback), `sdd/task-07a-report.md`, `sdd/task-07b-report.md` (memory plan),
`sdd/task-07c-report.md` (real-weight parity + GLM tokenizer/data path), `sdd/task-02-report.md` (rung template),
`torchtitan/experiments/glm53_flash/*`, `experiments/glm53_flash_sft_ladder/*`.

## Preconditions (check, stop and report if any fails)
- T07c parity gate passed in fp32 (truncated slices) and full key coverage has zero unknown keys.
- GLM BF16 checkpoint fully staged: `/data05/philip.yang/glm53-ladder-cache/model-weights/zai-org/GLM-5.3-Flash-BF16/.staged.json`
  exists (HDFS `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/zai-org/GLM-5.3-Flash-BF16/a5b45eb41df6402735dedc900be14a42e8d5e538`
  has `_COMPLETE`). Verify MD5 of 3 random shards against MANIFEST.jsonl.
- `nvidia-smi`: decide mode. If all 8 GPUs have >150 GiB free (T07b analytical floor 86.9 GiB/GPU incl. largest all-gather, before activations) -> mode A: FSDP8 BF16 frozen base on GPU (no offload).
  Otherwise -> mode B: FSDP8 + `training.enable_cpu_offload` (host RAM 3.9 TB). Record the mode.

## Requirements
0b. bf16 numerics check (open item from T07c): T07c's bf16 20-27 slice had top-1 0.43 / KL 0.27 vs the HF bf16 oracle
   (fp32 matched). Using `torchtitan/experiments/glm53_flash/hf_parity.py`, rerun the 20-27 bf16 slice on the 4 COUNTDOWN
   prompts only (exclude the synthetic 4096 sequence) and report per-sample top-1/KL. Then check the dtype each of these runs in,
   for TT vs HF: router scores/top-k, indexer scores, KDA gates/beta/recurrence state, mHC Sinkhorn, RMSNorm. If HF upcasts
   any of them to fp32 and TT does not, upcast in TT (experiment seam) and show the bf16 slice improves. Time-box to 45 min; if
   unresolved, record it as an OPEN item with numbers and continue to training (fp32 parity is the identity gate).
0. BLOCKER FIX FIRST (found by T07b): the glm53_flash experiment LoRA path adds LoRA modules but does NOT freeze every
   non-LoRA parameter. Fix it in the experiment seam so that ONLY LoRA A/B params have requires_grad=True (embeddings,
   norms, router, indexer, experts, mHC params all frozen), the optimizer only receives trainable params, and FSDP does not
   allocate grads for frozen params. Add a unit test asserting the trainable-param set == LoRA params for the debug flavor
   and that the count for the full flavor equals 67,190,784 (T07b count) computed on meta device.
1. Config `glm53_flash_countdown_lora` (exists from T07a/T07c): LoRA rank 16, alpha 64 (campaign knobs; targets as declared in
   T07a), lr 1e-4, FullAC, seq 512 first (task data is short), tokens/step 64x512 (same as all rungs), 94 steps, indexer frozen,
   MTP off, sparse attention fast path, HF BF16 initial load via the experiment adapter, `debug.seed` from env.
   LoRA params and optimizer state in fp32 if the experiment supports a frozen-bf16-base/fp32-LoRA split; if `training.dtype`
   forces everything to bf16, say so and keep bf16 (documented gap from the research doc).
2. Load + pre-train parity: after loading the full checkpoint in TorchTitan, compute logits on 4 countdown prompts and compare
   with the T07c oracle numbers for the same prompts if T07c produced full-model reference logits; otherwise compare against
   vLLM if vLLM 0.27 serves `zai-org/GLM-5.3-Flash-BF16` (try; do not install a new vLLM into system env), else record
   "full-model parity: truncated-slice only" honestly.
3. Training: seed 42, 94 steps; record loss curve, grad norm, memory per GPU, tokens/s for >=10 steady steps
   (label `exclusive_foreign_free` or `shared_gpu`). Then seed 43 if time allows (each run's wall clock decides; report ETA).
4. Resume gate: checkpoint (LoRA + optimizer + dataloader + RNG; base weights are frozen -- check whether the checkpointer can save
   only trainable state; if it saves all 600 GB, use `last_save_model_only`/exclude-frozen options or document) at step 62,
   publish that checkpoint to HDFS (`torchtitan/checkpoints/glm53-ladder/rung4/<run_id>/<attempt_id>/`), delete local copy,
   stage it back, resume to 94; compare losses 63-94 bitwise (deterministic mode on) or explain the gap.
5. Eval: base vs adapter on dev (500 problems). Preferred: vLLM with LoRA if it serves the model; fallback: TorchTitan-native
   greedy/sampled generation script in the experiment (`torchtitan/experiments/glm53_flash/generate.py`) with KV-free
   recompute if needed -- on a SUBSET (e.g. 100 problems x 8 samples) if full eval is too slow, clearly labelled. Use the
   countdown verifier from `torchtitan/experiments/countdown_search_distill` for scoring.
6. Publish adapter (PEFT-format if possible), eval outputs, logs, run manifest (staged URIs + md5s) to HDFS.

## Exclusions
No knob search. No core edits unless a clear core bug blocks loading (then stop and report). No commit. Never kill foreign
processes. Keep extra /data05 usage < 50 GB besides the staged checkpoint.

## Report
`.scratch/glm53-flash-sft-ladder/sdd/task-08-report.md` per contract: mode A/B, load time, parity, loss curve, memory, throughput,
resume result, eval table base vs adapter (n, samples, CI), HDFS URIs, claim labels, honest gaps, no commit.
