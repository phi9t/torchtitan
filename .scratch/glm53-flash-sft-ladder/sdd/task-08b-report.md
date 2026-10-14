# T08b report: rank-local GLM-5.3-Flash load unblock

Status: PARTIAL. The original full-HF-load host-RAM blocker is fixed in the
experiment seam, trainable-only checkpoint state is wired and unit-tested, and a
full GLM one-step LoRA SFT smoke run completed. The 94-step rung, HDFS resume
gate, eval, adapter export, and HDFS publish were not completed.

Surface: repo-local GLM experiment seam under
`torchtitan/experiments/glm53_flash`. No core files were edited. No commit was
made.

## Summary

T08 failed because the DCP/HF initial load path materialized full model-shaped
state per rank and was killed around 263 GiB RSS/rank. T08b replaces that path
for the GLM full LoRA config with an experiment-local rank-local safetensors
loader. The loader iterates live parameters/buffers after FSDP, computes each
DTensor local shape and global offset, reads only the needed HF safetensors
slices, maps grouped experts to per-expert HF keys, and leaves LoRA tensors at
their initialized values.

The full config now stays in mode A: GPU-resident sharded params,
`training.enable_cpu_offload=False`, `checkpointer.initial_load_in_hf=False`,
and `Glm53FlashLoRACheckpointManager` loads the frozen HF base with the
rank-local loader. Resumable model state is exposed through
`TrainableModelWrapper`, so checkpoint model state contains only `.lora_a` and
`.lora_b` tensors while optimizer, scheduler, dataloader, RNG, and train-step
state remain owned by the regular checkpoint manager.

The full one-step probe proved the unblock:

- rank-local HF load completed in 21.68 seconds;
- loaded 1,330 non-LoRA tensors and skipped 906 LoRA tensors;
- loaded 78,331,751,136 local bytes across the rank process;
- verified direct safetensor equality for embedding, KDA/indexer attention, and
  grouped expert slices;
- max sampled Python rank RSS was 3.75 GiB, far below the 120 GiB/rank limit;
- step 1 completed with loss 1.67569, grad norm 3.2812, and GPU peak memory
  131.56 GiB.

This is smoke/unblock evidence only. At the time of the decision, another T06
Qwen3-30B run still occupied GPUs 0, 1, 4, and 5 with about 44.9 GiB each,
leaving about 137.8 GiB free on those four GPUs. The GLM one-step probe peaked
at 131.56 GiB, leaving only a narrow shared-GPU margin, and the first uncached
probe hit a Triton autotune CUDA OOM. I did not launch a 94-step/resume/eval
campaign into that shared state because it would not produce valid performance
evidence and had little safety margin for a long run.

## Code changes

- `torchtitan/experiments/glm53_flash/sharded_hf_load.py`
  - Added `load_rank_local_hf_safetensors`.
  - Uses `compute_local_shape_and_global_offset` for DTensor local shards.
  - Reads tensor slices through `safetensors.safe_open(...).get_slice(...)`.
  - Handles one-to-one tensors, flattened `.dt_bias`, grouped
    `gate_up_proj`, and grouped `down_proj`.
  - Skips LoRA A/B tensors so adapter initialization is preserved.
  - Verifies selected loaded slices against direct safetensors reads.
- `torchtitan/experiments/glm53_flash/checkpoint.py`
  - Replaced DCP/HF frozen-base materialization with
    `load_rank_local_hf_safetensors`.
  - Preserved `TrainableModelWrapper` for trainable-only model state.
  - Added `verify_frozen_base_load` config.
- `torchtitan/experiments/glm53_flash/config_registry.py`
  - Full GLM Countdown LoRA config uses mode A only.
  - `GLM53_FLASH_FULL_CPU_OFFLOAD=1` now raises instead of enabling unsafe
    full-model CPU offload.
  - `initial_load_in_hf=False`; the experiment checkpoint hook owns HF base
    loading.
  - Added `GLM53_FLASH_SEED`, `GLM53_FLASH_DETERMINISTIC`, and TensorBoard
    enablement for resume comparison evidence.
- `torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py`
  - Added tiny safetensors loader tests for one-to-one tensors, grouped expert
    local slices, LoRA skip behavior, and verification.
  - Added coverage for seed/deterministic/TensorBoard config defaults.
  - Kept trainable-only model wrapper coverage on the debug flavor.

## Config evidence

Resolved config artifact:
`.scratch/glm53-flash-sft-ladder/evidence/task-08b-config-print.log`.

Important resolved values:

- `hf_assets_path=/project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16`
- `training.max_context_length=512`
- `training.num_tokens_per_train_step=32768`
- `training.steps=94`
- `training.enable_cpu_offload=False`
- `training.dtype=bfloat16`
- optimizer pattern `.*\.lora_[ab]$`
- checkpointer: `initial_load_in_hf=False`, `interval=31`,
  `last_save_model_only=False`, `load_frozen_base_from_hf=True`,
  `trainable_model_only=True`, `verify_frozen_base_load=True`
- `debug.seed=42`
- `debug.deterministic=False`
- `metrics.enable_tensorboard=True`

## Full GLM probes

### Probe 1: uncached Triton path

Run id: `t08b-full-probe-20261014T135736Z`

Artifacts:

- `.scratch/glm53-flash-sft-ladder/evidence/task-08b/logs/full_probe_step1_t08b-full-probe-20261014T135736Z.log`
- `.scratch/glm53-flash-sft-ladder/evidence/task-08b/logs/host_ram_guard_t08b-full-probe-20261014T135736Z.log`

Result:

- FSDP model initialized: `313,393,990,654` total parameters.
- Optimizer matched 906 LoRA params with pattern `.*\.lora_[ab]$`.
- Model CUDA memory before training: 72.99 GiB.
- Rank-local HF load completed successfully:
  - `tensors_loaded=1330`
  - `tensors_skipped_lora=906`
  - `bytes_loaded=78331751136`
  - verified keys:
    `tok_embeddings.weight`, `layers.0.self_attn.q_proj.weight`,
    `layers.3.self_attn.indexer.wk.weight`,
    `layers.3.mlp.experts.gate_up_proj`
  - load time: 26.49 seconds
- Failed later in backward with
  `RuntimeError: Triton Error [CUDA]: out of memory` inside the KDA l2norm
  backward autotuner.
- Host RSS remained low; the prior T08 host materialization problem did not
  recur.

### Triton prewarm

Artifact:
`.scratch/glm53-flash-sft-ladder/evidence/task-08b/triton_prewarm.log`.

The first prewarm call used the wrong `_l2norm_fwd_cuda` signature and failed
with `AttributeError("'NoneType' object has no attribute 'to'")`. The corrected
prewarm used `_l2norm_fwd_cuda(x, 1.0e-6, None)` and `_l2norm_bwd_cuda(...)` on
shape `[1, 512, 64, 128]`, dtype bf16, and succeeded. The second full probe used
the warmed cache under
`.scratch/glm53-flash-sft-ladder/evidence/task-08b/triton_cache_prewarm/`.

### Probe 2: warmed Triton cache

Run id: `t08b-full-probe-cache-20261014T140128Z`

Artifacts:

- `.scratch/glm53-flash-sft-ladder/evidence/task-08b/logs/full_probe_step1_t08b-full-probe-cache-20261014T140128Z.log`
- `.scratch/glm53-flash-sft-ladder/evidence/task-08b/logs/host_ram_guard_t08b-full-probe-cache-20261014T140128Z.log`
- output directory:
  `.scratch/glm53-flash-sft-ladder/evidence/task-08b/full_probe_step1/t08b-full-probe-cache-20261014T140128Z/`

Result:

- TensorBoard enabled at
  `.scratch/glm53-flash-sft-ladder/evidence/task-08b/full_probe_step1/t08b-full-probe-cache-20261014T140128Z/tb/20261014-1401`.
- FSDP model initialized: `313,393,990,654` total parameters.
- Optimizer matched 906 LoRA params with pattern `.*\.lora_[ab]$`.
- Model CUDA memory before training: 72.99 GiB.
- Trainer initialized with 512 tokens per DP rank, 32,768 tokens per train
  step, gradient accumulation 8, max context length 512, total steps 1.
- Rank-local HF load completed:
  - `tensors_loaded=1330`
  - `tensors_skipped_lora=906`
  - `bytes_loaded=78331751136`
  - verified keys:
    `tok_embeddings.weight`, `layers.0.self_attn.q_proj.weight`,
    `layers.3.self_attn.indexer.wk.weight`,
    `layers.3.mlp.experts.gate_up_proj`
  - load time: 21.68 seconds
- Step 1 completed:
  - loss: 1.67569
  - grad norm: 3.2812
  - memory: 131.56 GiB
  - tps: 5
  - tflops: 0.28
  - MFU: 0.01%
- Training completed and the process group was destroyed.
- Host RAM guard maximum sampled Python rank RSS:
  - `max_rss_gb=3.75`

The reported throughput is not performance evidence: this was a one-step smoke
run on shared GPUs with foreign work active on four devices.

## Gate status

| Gate | Status | Evidence / reason |
| --- | --- | --- |
| Rank-local HF load | Passed | Full warmed-cache probe loaded and verified selected slices in 21.68 s with max sampled rank RSS 3.75 GiB. |
| No full tensor materialization on rank | Passed for observed probes | Loader iterates live tensors and reads only local safetensors slices; host RSS stayed below 4 GiB/rank instead of T08's 263 GiB/rank failure. |
| LoRA tensors keep init | Passed in focused test | Tiny loader test asserts LoRA tensor is unchanged and skipped count is recorded. |
| Grouped expert slice mapping | Passed in focused test and full probe verification | Unit test covers local `gate_up_proj` expert slice; full probe verifies one grouped expert tensor. |
| T07c truncated-slice forward | Inherited, not rerun in T08b | Prior T07c fp32 artifacts remain the deterministic comparison evidence: `slice-0-3-fp32.json` and `slice-20-27-fp32.json`; T08b added direct safetensor load verification but did not rerun the HF slice harness. |
| Trainable-only checkpoint model state | Passed in unit test; not end-to-end resume-proven | `TrainableModelWrapper` exposes only `.lora_a`/`.lora_b`; full checkpoint/resume gate was not run. |
| Mode A, no CPU offload | Passed | Config print shows `enable_cpu_offload=False`; config raises if `GLM53_FLASH_FULL_CPU_OFFLOAD=1`. |
| Seed 42, 94 configured steps, 32,768 tokens/step, seq 512 | Configured | Config print shows seed 42, steps 94, tokens/step 32,768, max context 512. |
| One full optimizer step | Passed | Warmed-cache probe completed step 1. |
| >=10 steady training steps / throughput | Not run | Shared GPUs left only about 137.8 GiB free on four devices while step 1 peaked at 131.56 GiB; throughput would not be valid evidence. |
| Step-62 HDFS resume gate | Not run | Requires long training plus checkpoint publish/stage/resume; no step-62 checkpoint exists. |
| Resume loss comparison 63-94 | Not run | Dependent on the resume gate. |
| Countdown dev base-vs-adapter eval | Not run | No step-94 adapter exists. |
| Adapter export / PEFT-style config | Not run | No step-94 adapter exists. |
| HDFS publish of adapter/eval/logs/manifest | Not run | No adapter/eval bundle exists; diagnostic evidence remains local. |

## Verification

Passed in rootfs training env:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'unset CC CXX; cd /workspace/torchtitan && python -m py_compile \
    torchtitan/experiments/glm53_flash/sharded_hf_load.py \
    torchtitan/experiments/glm53_flash/checkpoint.py \
    torchtitan/experiments/glm53_flash/config_registry.py \
    torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py && \
  pytest -q torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py -x && \
  for hook in trailing-whitespace end-of-file-fixer check-ast check-merge-conflict ufmt flake8 pydoclint codespell; do \
    pre-commit run "$hook" --files \
      torchtitan/experiments/glm53_flash/sharded_hf_load.py \
      torchtitan/experiments/glm53_flash/checkpoint.py \
      torchtitan/experiments/glm53_flash/config_registry.py \
      torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py || exit $?; \
  done && \
  git diff --check -- \
    torchtitan/experiments/glm53_flash/sharded_hf_load.py \
    torchtitan/experiments/glm53_flash/checkpoint.py \
    torchtitan/experiments/glm53_flash/config_registry.py \
    torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py'
```

Result:

```text
22 passed, 1 skipped, 14 warnings
```

Selected hooks passed:

- `trailing-whitespace`
- `end-of-file-fixer`
- `check-ast`
- `check-merge-conflict`
- `ufmt`
- `flake8`
- `pydoclint`
- `codespell`

`git diff --check` passed on the touched GLM files.

Full `pre-commit run --all-files` was not run because the environment facts say
the repo-wide pyrefly hook currently fails and can auto-edit unrelated files.

## Remaining work

The next rung attempt should run only when the full 8-GPU set is exclusive, or
when the GLM training peak memory is reduced enough to leave real margin on
shared GPUs. The next attempt should reuse this rank-local loader, run at least
10 steady steps for throughput evidence, then run the step-62 HDFS resume gate,
export the LoRA adapter, evaluate base vs adapter on the declared Countdown dev
subset, and publish the final adapter/eval/log/manifest bundle to HDFS.
