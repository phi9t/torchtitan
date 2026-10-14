# T07b brief: rung 3 -- mini GLM hybrid learnability on Countdown + fast-path training

Repo: /data02/home/philip.yang/workspace/torchtitan (branch ultron/mainline). Do NOT commit.
Read first: `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY), `.scratch/glm53-flash-sft-ladder/spec.md`,
`.scratch/glm53-flash-sft-ladder/sdd/task-07a-report.md`, `torchtitan/experiments/glm53_flash/*`,
`.claude/skills/running-experiment-campaigns/SKILL.md` (Step 3 overfit gate, Falcon-hardened rules).

## Context
T07a landed the GLM-5.3-Flash experiment seam (commit bf5b4e8c5). Rung 3 proves the hybrid mechanisms (KDA, IndexPool +
indexer sparse MLA, mHC, MoE) LEARN on the campaign task before we spend GPU on the 313B model. The mini model is randomly
initialised, so tokenizer identity does not matter here: use the staged Qwen3 tokenizer at
`/data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-0.6B` (set vocab accordingly) and the countdown-sft/v1 data
staged by T02 at `/data05/philip.yang/glm53-ladder-cache/datasets/countdown-sft/v1` (if absent, stage it yourself with
`experiments/glm53_flash_sft_ladder/stage.sh hdfs://harunava/user/tiger/waystone/torchtitan/datasets/countdown-sft/v1 <dir>`).
Another worker (T02) is concurrently editing `torchtitan_recipes/models/qwen3.py`, countdown files and
`experiments/glm53_flash_sft_ladder/run_rung0.sh` -- do not touch those.

## Requirements
1. Wire real Countdown chat data into `glm53_flash_mini_countdown_lora` and add `glm53_flash_mini_countdown_full`
   (full training, no LoRA; random init so LoRA alone cannot learn) in `torchtitan/experiments/glm53_flash/config_registry.py`,
   using the same `ChatProcessor` + packing pattern as `torchtitan_recipes/models/qwen3.py::_qwen3_countdown_lora_sft`
   (copy the pattern; do not import from it if it is being edited). Seq 512, tokens/step 64x512 like rung 0.
2. Make the training configs use the attn-gym FAST sparse path (`attn_impl="fast"`) by default, keeping `reference` selectable
   via env. Verify fast vs reference loss curves for 20 steps agree closely on the debug model (same seed).
3. Overfit gate (Step 3, claim `learnability_smoke`): mini flavor, full training, fixed bank of the first 64 countdown samples,
   repeated (num epochs enough for ~300 steps), 2 GPUs: train CE must collapse (report curve; target < 0.1 or a clear
   >95% drop). Control: lr=0 run must stay flat. Run the gate for (a) the default mini model and (b) an ablation where sparse
   layers use dense causal attention instead of the indexer path (env switch), to show the sparse path is not the blocker.
   Losses across (a)/(b) must not be identical (Falcon silent-collapse rule).
4. Finite-grad + memory/throughput probe at seq 4096 and 8192 for the mini flavor with FullAC on 1-2 GPUs, fast path:
   report peak memory and tokens/s (label `shared_gpu`).
5. Full-size memory planning without weights: build the full `glm53_flash` flavor on meta device and compute exact parameter
   counts per component; then, with `COMM_MODE=fake_backend` / the fake_distributed_backend skill
   (`.claude/skills/fake_distributed_backend/` or run_train.sh fake modes), estimate per-rank memory for 4 and 8 GPUs under
   FSDP-only LoRA (bf16 frozen base, fp32 LoRA states), seq 8192, FullAC, if the experiment seam supports it; otherwise
   compute analytically and say so. State whether 8x B200 at ~179 GiB (exclusive) and the CURRENT shared state (~95 GiB free)
   can host it.
6. Publish rung-3 artifacts (logs, loss curves json, tiny checkpoints not needed) to
   `torchtitan/runs/glm53-ladder/rung3/<run_id>/` with `experiments/glm53_flash_sft_ladder/publish.sh`.

## Exclusions
No core edits. No commit. GPUs are shared (env-facts): stay < 60 GiB/GPU, prefer GPUs 4-7 (T02 may use 0-3; check nvidia-smi).
Never kill other processes.

## Report
`.scratch/glm53-flash-sft-ladder/sdd/task-07b-report.md`: configs, curves (step 1/50/100/.../final), gate pass/fail,
fast-vs-reference agreement, memory/throughput table, full-size memory plan with arithmetic, HDFS URIs, claim labels, no commit.
