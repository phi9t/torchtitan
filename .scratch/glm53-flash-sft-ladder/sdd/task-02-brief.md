# T02 brief: rung 0 -- Qwen3 0.6B / 1.7B LoRA SFT on countdown-sft/v1, HDFS-backed

Repo: /data02/home/philip.yang/workspace/torchtitan (branch ultron/mainline). Do NOT commit.
Read first: `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY, overrides anything stale below), `.scratch/glm53-flash-sft-ladder/spec.md` (all of it), `.scratch/glm53-flash-sft-ladder/sdd/task-01-report.md`
(storage wrappers you must use), `experiments/glm53_flash_sft_ladder/README.md`, `.claude/CLAUDE.md`,
`experiments/countdown_search_distill/README.md`, `torchtitan_recipes/models/qwen3.py` (`_qwen3_countdown_lora_sft`),
`experiments/countdown_search_distill/{run_common.sh,run_train.sh,run_export_adapters.sh,run_eval.sh}`.

## Environment facts
- Another user's job holds ~84 GiB on EVERY GPU (70-95% util). Never kill or signal it. You have ~95 GiB per GPU.
  vLLM must be launched with a low `gpu_memory_utilization` (~0.35; vLLM computes it against total memory, so check
  it fits beside 84 GiB) -- add an optional `--gpu-memory-utilization` flag to
  `torchtitan/experiments/countdown_search_distill/cli.py evaluate-vllm` (default None = unchanged behaviour) and a
  `VLLM_GPU_MEMORY_UTILIZATION` env passthrough in `run_eval.sh`. Keep that change minimal and backward compatible,
  add/adjust a unit test in `tests/unit_tests/test_countdown_search_distill.py` if one covers arg parsing.
- Throughput measured now is labelled `shared_gpu` (not performance evidence).
- Local cache root: `/data05/philip.yang/glm53-ladder-cache` (872 GB free disk, shared). Other disks are nearly full.
- All Python/torchrun/vLLM inside `scripts/rootfs/enter_rootfs.sh` (existing countdown runners already re-enter it).
  The waystone CLI runs on the host (orchestration).

## Requirements
1. Stage inputs from HDFS only (via T01 `stage.sh`): `torchtitan/datasets/countdown-sft/v1` and model weights for
   `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-1.7B` under `torchtitan/model-weights/...` (if 1.7B is not ingested yet, run
   T01's `hf_ingest.sh --repo Qwen/Qwen3-1.7B --revision main` first). Do NOT use `./assets/hf/Qwen3-1.7B` directly.
2. Add experiment-owned training configs (do not edit core): `torchtitan/experiments/glm53_flash_sft_ladder/__init__.py`
   + `config_registry.py`, reusing `_qwen3_countdown_lora_sft` from `torchtitan_recipes.models.qwen3` (or a small
   generalisation there if truly needed; prefer not). Configs read env vars for data file, hf_assets_path, dump
   folder, seed-independent knobs. Configs: `qwen3_0_6b_countdown_lora`, `qwen3_1_7b_countdown_lora`,
   `qwen3_0_6b_countdown_full` (no LoRA transform, lr 1e-5, for the LoRA-vs-full sanity check).
   Seeds/determinism are set via env-read config fields, not CLI flags (see env-facts). Frozen knobs: LoRA rank 16, alpha 64, targets wqkv/wo/w13/w2, lr 1e-4, 64x512 tokens/step, seq 512, 94 steps,
   checkpoint interval 31 (so steps 31, 62, 93/94 exist). Seeds via env -> `config.debug.seed`.
   Add a runner `experiments/glm53_flash_sft_ladder/run_rung0.sh` that does stage -> train -> export adapter ->
   eval -> publish, re-entering rootfs for Python stages (mirror countdown's `countdown_enter_rootfs_if_needed`).
   Use `CUDA_VISIBLE_DEVICES` to pack runs: e.g. 2 GPUs per training run so 4 runs go in parallel.
3. Gates (each pass/fail in the report):
   a. Config dry-run: each config with `COMM_MODE=fake_backend` or 1-2 steps, finite loss.
   b. HF logits parity: TorchTitan Qwen3-0.6B forward vs HF transformers forward on 4 countdown prompts, bf16; report
      max abs logit diff and top-1 agreement (expect top-1 agreement ~100%, small diffs). Put the script under
      `torchtitan/experiments/glm53_flash_sft_ladder/parity.py` so it can be reused for later rungs.
   c. LoRA vs full sanity at 0.6B seed 42: both losses fall; record final losses.
   d. GPU-count ladder at 0.6B seed 42: NGPU=2,4,8 with identical tokens/step; final train losses should agree
      closely (report max |delta loss| over steps). NOTE NGPU=8 shares GPUs with the other job -- fine.
   e. Resume gate (0.6B seed 42, NGPU=2, env -> `config.debug.deterministic=True`): run A uninterrupted 94 steps. Run B: stop after the
      step-62 checkpoint, `publish.sh` the checkpoint dir to `torchtitan/checkpoints/glm53-ladder/<run_id>/<attempt_id>/`,
      DELETE the local checkpoint dir, `stage.sh` it back, resume to 94. Compare losses for steps 63-94 with full
      precision (TensorBoard or `scripts/loss_compare.py`); bitwise identical expected. If not bitwise, root-cause
      (dataloader state? RNG?) and report.
   f. Eval: base vs adapter on dev (500 problems, 32 rollouts, existing sampling config) for 0.6B and 1.7B, seeds 42 and
      43 (adapters: 2 models x 2 seeds = 4; base: 2 models). Report pass@1, pass@32, strict pass@1 with the
      existing bootstrap intervals if the CLI provides them. Sanity anchor: 1.7B formatting adapter seed 43 got dev
      pass@1 0.278 in the prior replication (different train run, so not expected to be identical).
   g. Publish: adapters, eval dirs, train logs/TB, run manifests to `torchtitan/runs/glm53-ladder/<run_id>/...` and
      checkpoints to `torchtitan/checkpoints/glm53-ladder/...` via `publish.sh`; then delete local checkpoints that are
      published (keep space free). Every run manifest records staged HDFS URIs + md5s.
4. Run `pre-commit run --files <changed files>` inside rootfs and focused tests.

## Exclusions
No new knobs beyond those listed. No core changes. No commit. Do not touch the other user's processes.
Do not run 8B or larger models (later rungs).

## Report
`.scratch/glm53-flash-sft-ladder/sdd/task-02-report.md` per the campaign report contract: what ran (configs, seeds, GPUs,
tokens/step, steps), each gate pass/fail with numbers, results table, HDFS URIs, honesty notes, claim_label
(`smoke` for plumbing, `deterministic_comparison` for resume, `replicated_eval` for the 2-seed dev table), "no commit".

## Orchestrator addendum (supersedes conflicting text above)
- Staged inputs already exist (MD5-verified, with `.staged.json`):
  - Qwen3-0.6B: `/data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-0.6B` (HDFS `.../model-weights/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca`)
  - Qwen3-1.7B: `/data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-1.7B` (HDFS `.../model-weights/Qwen/Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`)
  - Dataset: stage `hdfs://harunava/user/tiger/waystone/torchtitan/datasets/countdown-sft/v1` with `stage.sh` into
    `/data05/philip.yang/glm53-ladder-cache/datasets/countdown-sft/v1`.
  Record these URIs + manifest md5 in every run manifest.
- Training/Trainer/FSDP: `TORCHTITAN_ROOTFS_ENV=glm53_train` + `unset CC CXX` (torch nightly). vLLM eval: default env.
  The PEFT export step: try default env; if it imports Trainer/FSDP code that needs nightly, use glm53_train.
- HDFS is ~20-25 MB/s per stream: publish checkpoints only for the resume gate and the final adapters (adapters are small);
  do not publish every intermediate full checkpoint of every run. Delete local checkpoints after publish.
- A background ingest (T03) is writing ~720 GB to /data05; keep your own /data05 usage under ~60 GB at any time.
- GPU packing: the shared job uses ~84 GiB/GPU; Qwen3 0.6B/1.7B LoRA at 64x512 tokens/step is small. Run up to 4 jobs
  concurrently on disjoint GPU pairs. vLLM with gpu_memory_utilization so that its total budget fits next to 84 GiB
  (e.g. 0.4 of 179 GiB = 72 GiB) -- verify free memory with nvidia-smi right before launching.
