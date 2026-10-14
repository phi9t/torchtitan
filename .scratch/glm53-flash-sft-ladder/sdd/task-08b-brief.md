# T08b brief: rung 4 unblock -- rank-local sharded HF load + trainable-only checkpoint, then GLM-5.3-Flash LoRA SFT

Repo: /data02/home/philip.yang/workspace/torchtitan (branch ultron/mainline). Do NOT commit.
Read first: `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY), `sdd/task-08-brief.md` (requirements 1-6 still apply),
`sdd/task-08-report.md` (what failed and why), `map.md` (Incident 12:56Z), `torchtitan/experiments/glm53_flash/*`.

## Hard safety rules
- Host RAM is shared (3.9 TB). Start `experiments/glm53_flash_sft_ladder/host_ram_guard.sh` is ALREADY running (kills GLM ranks if
  available < 600 GB). In addition, your launcher must sample per-rank RSS every 30 s into the run log. Per-rank RSS must stay
  < 120 GB. Never enable `training.enable_cpu_offload` for the full model.
- Another worker (T08c) is diagnosing bf16 numerics and will NOT edit model.py; you own `torchtitan/experiments/glm53_flash/`.
- T06 (Qwen3-30B) may occupy ~45 GiB on GPUs 0,1,4,5. Check nvidia-smi; you may use all 8 GPUs if each has >= 120 GiB free.

## Requirements
1. Rank-local sharded HF loader in the experiment (e.g. `torchtitan/experiments/glm53_flash/sharded_hf_load.py`):
   build the model on meta, apply FSDP (mode A, params on GPU as DTensors), `to_empty(device="cuda")`, then for every native
   param/buffer compute the local shard (`DTensor` placements + `compute_local_shape_and_global_offset`), map it to the HF
   key(s) via the existing state-dict adapter mapping (grouped experts: the local expert range maps to a subset of per-expert
   HF keys), and read ONLY those slices with `safetensors.safe_open(...).get_slice(...)` from the staged checkpoint, copying into
   the local shard. LoRA params keep their init. Never build a full tensor of a sharded param on any rank. Verify: after load,
   compare a handful of params (embedding rows, one expert, one KDA proj, one indexer weight) against direct safetensors reads;
   and the T07c truncated-slice forward (layers 0-3) on GPU matches the TT fp32/bf16 numbers from T07c.
   Wire it so the full config uses it instead of the DCP HF initial load (`initial_load_in_hf=False` + an experiment hook after
   model init, or the cleanest seam you find; document it).
2. Trainable-only checkpointing: resumable checkpoints must contain only LoRA params, optimizer state, lr scheduler, dataloader,
   RNG and train step -- not the frozen base. Prefer existing CheckpointManager options (search for model-state filtering /
   `exclude_from_loading` / trainable-only); otherwise implement an experiment-level state-dict filter. On resume, the base is
   re-loaded with the sharded loader. Unit-test the filter on the debug flavor.
3. Then run the rung: T08 requirements 1, 3, 4, 5, 6 (seed 42 first; 94 steps; tokens/step 64x512; FullAC; indexer frozen;
   MTP off; fast sparse path). Load time, steady throughput (>=10 steps), per-GPU memory, loss curve. Resume gate at step 62 via
   HDFS publish of the trainable-only checkpoint (small) -> delete local -> stage -> resume to 94, compare losses 63-94
   (bitwise if deterministic is feasible; otherwise report max |delta| and why).
4. Eval (T08 req 5): base vs adapter on Countdown dev. Check first whether vLLM 0.27 (default env) can serve
   `zai-org/GLM-5.3-Flash-BF16` (+ LoRA). If not, write `torchtitan/experiments/glm53_flash/generate.py` (TorchTitan-native
   sampling using the same sharded model across 8 GPUs, temperature 0.8, top-p 0.95, max 192 new tokens, the GLM chat template with
   the countdown prompt) and evaluate a declared subset (e.g. 100 dev problems x 4 samples) for base and adapter, scored with the
   countdown verifier. Label the subset honestly.
5. Publish adapter (LoRA tensors + PEFT-style config), eval outputs, logs, manifest to HDFS
   (`torchtitan/runs/glm53-ladder/rung4/<run_id>/<attempt_id>/`).

## Exclusions
No core edits (report proposed core changes). No commit. No knob changes. Never kill foreign processes.

## Report
`.scratch/glm53-flash-sft-ladder/sdd/task-08b-report.md` per contract, including per-rank RSS max, load time, and every gate.

## Note added at launch (13:30Z)
- T08 left an in-progress `torchtitan/experiments/glm53_flash/checkpoint.py` (trainable-only LoRA checkpoint policy) committed as-is;
  review, finish and test it for requirement 2 rather than starting over. Debug/LoRA tests: 20 passed, 1 skipped.
