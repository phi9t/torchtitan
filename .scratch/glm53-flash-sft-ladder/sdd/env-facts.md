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

## GLM53_EP RUNTIME (R01b, verified 2026-10-14)
- `glm53_ep` is the declared GLM Flash native-extension/post-DeepEP L1 env. It is separate from live `glm53_train` and lives under
  the env-specific rootfs state `.cache/torchtitan-rootfs/glm53_ep/venvs/b200-runtime`.
- Select the opt-in native L0 rootfs from the managed store, not the live legacy rootfs:
  `--rootfs-store /data05/philip.yang/torchtitan-rootfs-store`.
  Current selected native store id:
  `rootfs-e0c18cff68fc8ed06ca494cf00c807cca3d714819e00d5b824d89269079f4ec5`.
- Sync command:
  `experiments/glm53_flash_sft_ladder/runtime/sync_python_env.sh --rootfs-store /data05/philip.yang/torchtitan-rootfs-store`.
- Launch command pattern:
  `TORCHTITAN_ROOTFS_ENV=glm53_ep scripts/rootfs/enter_rootfs.sh --rootfs-store /data05/philip.yang/torchtitan-rootfs-store -- bash -lc 'unset CC CXX CUDAHOSTCXX; cd /workspace/torchtitan && ...'`
- R01b evidence: `.scratch/glm53-flash-sft-ladder/evidence/r01b/`.
  Runtime passes torch/CUDA/sm_100, Triton import, `attn_gym.linear.kda`, TorchTitan's `attn_gym` bridge, NVSHMEM wheel
  presence, CUDA headers/libs, ccache, CUTLASS, RDMA/NUMA, Nsight CLI visibility, and linker probes. GLM Flash tests,
  GLM Flash EP tests, GLM arch speedrun tests, and the 2-step synthetic `glm53_flash_debugmodel` train pass in `glm53_ep`.
- `torchao` and `flash-attn` are intentionally excluded from the `glm53_ep` L1 lock. `glm53_train` does not need either for
  the verified GLM paths, installing `torchao==0.18.0` trips a Triton 3.9 autotune-key validation error during optional
  NVFP4 import, and `flash-attn` is not referenced by the verified GLM/speedrun paths.
- Direct `fla.ops.kda` import still fails before KDA on `fla.ops.cp.chunk_delta_h` because Triton 3.9 rejects an autotune key
  containing `BT` when the kernel args omit `BT`; `verify_runtime.py` records this as a diagnostic, not a GLM runtime gate.

## GPU UPDATE 2026-10-14 09:44Z
- The other user's `pretrain_commq.py` job has ENDED; GPUs are free apart from campaign workers. Always check `nvidia-smi`
  before launching (it may restart). Coordinate: T02 uses GPUs 0-3, T07b uses 4-5, T07c/T05 use 6-7 unless free otherwise.
  Throughput taken while no foreign job is present is labelled `exclusive_foreign_free` (still shared between our workers).

## HDFS-STRICT INPUT RULE (user 2026-10-14, binding for every ticket in both campaigns)
"All training will be using HDFS strictly."
- Every training/eval/parity input (datasets, model weights, tokenizers, checkpoints to resume from) must come from a
  waystone HDFS URI that has `_COMPLETE` and `MANIFEST.jsonl` with md5s. Local disk is ONLY a cache populated by
  `experiments/glm53_flash_sft_ladder/stage.sh` / `lib/waystone.sh` get (writes `.staged.json` with the source URI).
- Before launch, the runner must call `experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh <local_dir>...` (H01 adds
  it) which fails unless each input dir has a `.staged.json` whose files match the HDFS manifest (size + md5).
- No ad hoc local copies (e.g. tokenizers copied into `.scratch/.../inputs/`, raw HF downloads, files from other users'
  dirs). If something is not on HDFS yet, ingest it to HDFS first (`hf_ingest.sh` / `publish.sh`), then stage it.
- Every run artifact (attempt.json / report) records the HDFS URIs (and manifest md5) of all its inputs.
- Outputs (checkpoints, adapters, eval, bundles) are published to HDFS with md5 verification, as before.
- Clarification (user): all HDFS access stays via waystone (`waystone ... put|get --verify-md5`, `layout-profile`,
  `storage-prefix`, or the repo wrappers built on it). Do not call `hdfs dfs`/`hadoop fs` or other clients directly.
  No Primus/Forge work is in scope.
