# T08 report: GLM-5.3-Flash LoRA SFT rung 4

Status: BLOCKED before training step 1. No commit was made.

Surface: repo-local GLM experiment seam under `torchtitan/experiments/glm53_flash`.
No core files were edited.

## Summary

Fixed the T07b LoRA blocker: GLM LoRA configs now freeze every non-LoRA
parameter, the optimizer matches only LoRA A/B parameters, and focused tests
prove the trainable set and full-size trainable count.

The full rung did not complete. A one-step, save-disabled, mode-B full
GLM-5.3-Flash load/train probe was killed before model-load completion after
host RSS reached about 263 GiB per rank. The active TorchTitan DCP/HF load path
materializes full model-shaped state per rank; CPU offload helps GPU residency
but does not avoid this host-side materialization. The stock checkpoint path
also saves the flattened full model state for resumable checkpoints, so a
normal step-62 resume checkpoint would include the frozen base and violate the
ticket's local-space intent. No 94-step training, resume gate, adapter export,
eval, or adapter HDFS publish was produced.

## Code changes

- `torchtitan/experiments/glm53_flash/model.py`
  - Added `apply_lora_trainable_policy`.
  - For LoRA configs, only parameters ending in `.lora_a` or `.lora_b` retain
    `requires_grad=True`.
- `torchtitan/experiments/glm53_flash/config_registry.py`
  - LoRA configs use a LoRA-only optimizer pattern: `.*\.lora_[ab]$`.
  - Full GLM Countdown config now defaults to the campaign knobs:
    seq len 512, 32,768 tokens/step, 94 steps, lr 1e-4.
  - Full GLM config enables HF initial load via `CheckpointManager.Config`.
  - CPU-offload mode sets `debug.distinct_seed_mesh_axes=["pp", "dp_shard"]`,
    required by Trainer validation.
  - Added `GLM53_FLASH_FULL_CHECKPOINT_LOAD_ONLY` for load/train probes that
    must not write full-base checkpoints.
- `torchtitan/experiments/glm53_flash/hf_parity.py`
  - Added `--no-long-sequence` so the bf16 open-item rerun can include exactly
    the four Countdown prompts and exclude the synthetic 4096-token sequence.
- `torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py`
  - Added trainable-set and full-size trainable-count coverage.
  - Added full Countdown LoRA config default coverage.

## Preconditions

Passed:

- T07c key coverage artifact has `unknown_hf_keys=[]`,
  `missing_native_keys=[]`, `unexpected_native_keys=[]`, and
  `adapter_error=null`.
- GLM BF16 checkpoint is staged at
  `/data05/philip.yang/glm53-ladder-cache/model-weights/zai-org/GLM-5.3-Flash-BF16`.
- Staged metadata:
  - source URI:
    `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/zai-org/GLM-5.3-Flash-BF16/a5b45eb41df6402735dedc900be14a42e8d5e538`
  - `manifest_md5=365857f8b1d7cb0841992fff91240ccf`
  - `file_count=130`
  - `staged_at=2026-10-14T10:23:44Z`
- HDFS `_COMPLETE` exists at the source URI.
- Random sampled shard MD5 checks passed:

| shard | bytes | md5 |
| --- | ---: | --- |
| `model-00021-of-00120.safetensors` | 5353623672 | `b663fca5a9212c96771395385915bf55` |
| `model-00084-of-00120.safetensors` | 5362973032 | `0a48122ddf6ff94c0f4067ebd1bd596f` |
| `model-00110-of-00120.safetensors` | 5368753872 | `5ab409d218022691ff1b6d8292afa90a` |

Mode decision: mode B, shared GPU. At launch, GPUs 0/1/4/5 had about 138 GiB
free because another in-campaign Qwen3-30B run was using about 44 GiB on each;
GPUs 2/3/6/7 were effectively free. The ticket requires mode A only when all 8
GPUs have >150 GiB free, so the full GLM probe used
`GLM53_FLASH_FULL_CPU_OFFLOAD=1`.

## LoRA blocker fix

Focused test result:

```text
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'unset CC CXX; cd /workspace/torchtitan && pytest -q torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py -x'

17 passed, 1 skipped, 14 warnings in 9.57s
```

The new tests assert:

- debug LoRA trainable-param set equals all and only `.lora_a`/`.lora_b`;
- full GLM rank-16 LoRA trainable parameter count is exactly `67,190,784`
  on meta device;
- full config uses seq 512, 32,768 tokens/step, 94 steps, lr 1e-4,
  HF initial load, checkpoint interval 31, and LoRA-only optimizer.

## BF16 parity open item

Command used the pinned HF oracle env and excluded the synthetic long sequence:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_ref \
TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST=/data05/philip.yang/glm53-ladder-cache \
scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'unset CC CXX; cd /workspace/torchtitan && python -m torchtitan.experiments.glm53_flash.hf_parity slice-parity \
    --checkpoint /project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16 \
    --data-file /project/formal-cache/datasets/countdown-sft/v1/train/formatting.jsonl \
    --layer-start 20 --num-layers 8 --dtype bfloat16 --device cpu --no-long-sequence'
```

Artifact:
`.scratch/glm53-flash-sft-ladder/evidence/task-08/slice-20-27-bf16-countdown-only.json`.

| sample | seq len | top-1 | KL HF->TT | final max abs | max layer abs | sparse selection mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cd-14-16-49-t47` | 76 | 0.447368 | 0.207257 | 6.28125 | 0.875 | 0 |
| `cd-24-8-22-t54` | 76 | 0.539474 | 0.179499 | 6.02832 | 0.875 | 0 |
| `cd-17-95-3-t75` | 76 | 0.434211 | 0.269434 | 8.96875 | 2.25 | 0 |
| `cd-24-49-15-t58` | 76 | 0.460526 | 0.182266 | 6.75 | 0.875 | 0 |

Result: unresolved/open. Excluding the long synthetic sample did not improve the
middle-slice bf16 issue; min top-1 remains `0.434211` and max KL remains
`0.269434`. The visible dtype audit did not find a clear requested category
where HF upcasts and TT does not: router scores/top-k, indexer scores, KDA
forget gate/dt/A_log path, recurrence-state math, mHC mixing, and RMSNorm all
use fp32 in the visible TT and HF reference paths. HF reported that
`causal_conv1d` is absent and it is using its slower reference PyTorch
implementation.

FP32 T07c remains the identity gate: the prior fp32 slice had top-1 agreement
1.0 and zero sparse selection-mask mismatches, with the already-documented
middle-slice router-sensitive outlier.

## Full GLM training attempt

Resolved config artifact:
`.scratch/glm53-flash-sft-ladder/evidence/task-08/full-config-print.txt`.

Important resolved values:

- `hf_assets_path=/project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16`
- `training.max_context_length=512`
- `training.num_tokens_per_train_step=32768`
- `training.steps=94`
- `training.enable_cpu_offload=True`
- `training.dtype=bfloat16`
- optimizer pattern `.*\.lora_[ab]$`
- checkpointer: `initial_load_in_hf=True`, `interval=31`,
  `last_save_model_only=False`
- probe-only checkpointer: `load_only=True`

Attempt 1:

- Log:
  `.scratch/glm53-flash-sft-ladder/evidence/task-08/logs/full_probe_step1_t08-rung4-probe-20261014T124514Z.log`
- Result: failed before model construction with:
  `ValueError: training.enable_cpu_offload requires 'dp_shard' in debug.distinct_seed_mesh_axes`.
- Fix: full GLM CPU-offload config now sets `["pp", "dp_shard"]`.

Attempt 2:

- Run id: `t08-rung4-probe-20261014T124628Z`
- Log:
  `.scratch/glm53-flash-sft-ladder/evidence/task-08/logs/full_probe_step1_t08-rung4-probe-20261014T124628Z.log`
- Config: 8 GPUs, mode B CPU offload, checkpoint saves disabled with
  `GLM53_FLASH_FULL_CHECKPOINT_LOAD_ONLY=1`, `GLM53_FLASH_FULL_STEPS=1`.
- Outcome: killed before model-load completion / before step 1.
- Observed resource samples:
  - about 5 minutes in: ~136 GiB RSS per rank;
  - about 9 minutes in: ~263 GiB RSS per rank, all eight ranks near one CPU
    core each;
  - then the exec returned code 137.
- No step loss, grad norm, throughput, or GPU peak memory can be reported from
  this attempt because no optimizer step completed.

Root-cause assessment:

- The model is created on meta and then materialized on CPU when
  `training.enable_cpu_offload=True`.
- DCP/HF loading builds a flattened full model-shaped state dict for loading;
  it is not trainable-only and is per-rank before the run reaches training.
- CPU offload avoids full GPU residency but does not avoid this host-side
  materialization.
- The same checkpoint manager would save a full flattened model state for a
  resumable interval checkpoint. That would include the frozen base and is not
  acceptable for the rung-4 step-62 resume gate under the ticket's local-space
  constraint.

Cleanup note: the killed probe released GPU memory, but the bwrap subtree left
uninterruptible/defunct process entries for the probe PIDs. A targeted TERM and
KILL against only the failed probe PIDs did not reap them. They were not holding
GPU memory afterward.

## Eval and publish status

Eval was not run. No adapter exists because training did not reach step 94, so
there is no PEFT adapter, base-vs-adapter eval table, or adapter HDFS URI.

The diagnostic evidence bundle is compact and local under
`.scratch/glm53-flash-sft-ladder/evidence/task-08/`.

Published diagnostic HDFS bundle:
`hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-ladder/rung4/t08-rung4-diagnostics-20261014T130432Z/attempt-01`.

It includes:

- bf16 countdown-only parity artifact;
- resolved config print;
- full-load probe logs;
- run id marker.

## Verification

Passed:

```text
python -m py_compile \
  torchtitan/experiments/glm53_flash/hf_parity.py \
  torchtitan/experiments/glm53_flash/model.py \
  torchtitan/experiments/glm53_flash/config_registry.py \
  torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py
```

Passed:

```text
pytest -q torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py -x
17 passed, 1 skipped, 14 warnings
```

Passed selected hooks on the four modified GLM files:

- `ufmt`
- `flake8`
- `pydoclint`
- `codespell`
- `trailing-whitespace`
- `end-of-file-fixer`
- `check-ast`
- `check-merge-conflict`

Passed:

```text
git diff --check -- torchtitan/experiments/glm53_flash/model.py \
  torchtitan/experiments/glm53_flash/config_registry.py \
  torchtitan/experiments/glm53_flash/hf_parity.py \
  torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py
```

Full `pre-commit run --all-files` was not run because the environment facts say
the pyrefly hook currently fails repo-wide and may auto-edit unrelated files.

## Next patch proposal

Do not retry the full rung until the full-checkpoint load and checkpointing
contract is narrowed. The next implementation should add an experiment-owned
frozen-base LoRA checkpoint/load path that:

1. initializes the frozen GLM base from HF shards without materializing a full
   per-rank dense CPU model/checkpoint state;
2. stores only LoRA parameters plus optimizer, scheduler, dataloader, RNG, and
   train-step state at resume checkpoints;
3. reconstructs the full training state on resume by loading the HF base first,
   then applying LoRA and non-model states from the compact resume bundle;
4. exports PEFT-format adapters directly from LoRA tensors.

Until that lands, T08 cannot satisfy the 94-step training, HDFS resume, eval,
or adapter-publish gates.
