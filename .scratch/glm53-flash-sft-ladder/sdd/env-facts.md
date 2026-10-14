# Environment facts (verified 2026-10-14 by orchestrator) -- read before any ticket

- Rootfs deps were out of sync with requirements.txt; fixed in T00 (spmd_types@2ae49fc, attn-gym[linear,cudnn]==0.0.16,
  torch_remat, torch_checkpointing, grain 0.2.18, renderers 0.1.11; torch 2.13.0+cu132 kept; cutlass-dsl 4.6->4.8).
  Pre-change freeze: `.scratch/glm53-flash-sft-ladder/evidence/rootfs-pip-freeze-before-t00.txt`.
  Now importable in rootfs: torchtitan.models.{qwen3,deepseek_v4,kimi_k3}, common.attention.kda, attn_gym, vllm 0.27.0, flashinfer.
- Rootfs is OFFLINE by default. For network (pip/uv installs, HF downloads, git) prefix the host call with
  `TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- ...`. Install with
  `uv pip install --system --break-system-packages -c <(echo torch==2.13.0+cu132) ...` (never let torch change).
- CLI changed upstream: `torchtitan/train.py` accepts ONLY `--module --config [--override TARGET=JSON] [--comm-backend]
  [--output-dir] [--resume-step N|-1] [--print-config]`. Dotted overrides like `--training.steps` / `--debug.seed` do NOT exist.
  Put knobs in config functions, read from env vars (pattern: `torchtitan_recipes/models/qwen3.py` `_countdown_*` helpers).
  Seed: `config.debug.seed`, determinism: `config.debug.deterministic` (torchtitan/config/configs.py:163-176).
  The CLAUDE.md examples using `--debug.seed=42` are stale.
- Default llama3 debug module is `torchtitan_recipes.tests.models.llama3` (run_train.sh MODULE default).
- GPUs: another user's job (`pretrain_commq.py`, ziqi.yin) holds ~84 GiB on all 8 B200s at 70-95% util. NEVER kill/signal it.
  ~95 GiB free per GPU. Throughput is `shared_gpu`, not performance evidence. vLLM needs low gpu_memory_utilization.
- Disk: cache at `/data05/philip.yang/glm53-ladder-cache` (872 GB free shared disk). /data01,03,04,06,07 >=93% full; /data02 ~200 GB.
- HDFS: waystone wrapper `/data02/home/philip.yang/workspace/waystone/scripts/waystone`; auth is kept fresh by cron
  (`~/.local/bin/waystone-auth-keepalive.sh`). If calls hang, the doas token is stale -> run that script once.
- traecli workers launch via `experiments/glm53_flash_sft_ladder/trae_worker.sh <ticket> <brief>`.

## TRAINING ENV (critical, added after T00b)
- Repo code needs torch NIGHTLY (e.g. `MixedPrecisionPolicy(param_dtype_override_fn=...)`); rootfs system torch 2.13.0 cannot run
  `torchtitan/train.py` for any model. A training venv with `torch==2.16.0.dev20261013+cu132` layered over system site-packages
  lives in the rootfs env `glm53_train`. ALWAYS launch training/tests that touch FSDP/Trainer with:
  `TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && ...'`
- vLLM eval (countdown `run_eval.sh`, `evaluate-vllm`) must use the DEFAULT env (no TORCHTITAN_ROOTFS_ENV) -- vLLM 0.27 is built
  for torch 2.13. Export/adapter conversion: try the default env first.
- Host exports `CC/CXX` pointing at a spack gcc that does not exist inside the rootfs; Triton compile fails with
  `FileNotFoundError ... spack .../bin/gcc`. Always `unset CC CXX` inside the rootfs command.
- Verified: `deepseek_v4_debugmodel` 2-GPU trains 10 steps in glm53_train env (loss 4.24 -> 3.65).
- pre-commit: run only the relevant hooks on changed files (flake8, ufmt, pydoclint, codespell); the pyrefly hook currently fails
  repo-wide (52 pre-existing errors) AND auto-edits unrelated core files -- if you run it, `git checkout` those files afterwards.
