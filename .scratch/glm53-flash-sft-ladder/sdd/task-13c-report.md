# T13c report: full-depth GLM-5.3-Flash EP8 readiness probe

Date: 2026-10-14

Verdict: GO for the rung-4 EP8 rerun. The full-depth 45-layer EP8 path ran 15
deterministic steps, saved a trainable-only step-10 checkpoint, published and
staged that checkpoint through Waystone HDFS, resumed from the staged checkpoint,
and matched uninterrupted losses exactly for steps 11-15. Rank-local HF loading
also confirmed the intended 36-expert slice per rank across all 42 sparse
layers.

One follow-up remains before treating adapter export as a reusable production
command: the existing `torchtitan.experiments.glm53_flash.export_glm_lora`
entrypoint matched optimizer-state keys containing `.lora_a` / `.lora_b` and
failed on this DCP checkpoint. For this ticket I produced the adapter with a
strict artifact-local exporter that accepts only model keys ending exactly in
`.lora_a` or `.lora_b`; the resulting PEFT adapter passed schema/key/shape
checks and was published to HDFS.

## Evidence root

`.scratch/glm53-flash-sft-ladder/evidence/task-13c/t13c-20261014T203617Z`

Key artifacts:

- `logs/full_depth_gpu_gate.csv`: full-depth GPU gate polling log.
- `logs/train_full15.log`: uninterrupted 15-step full-depth EP8 run.
- `analysis/load_probe_summary.json`: per-rank rank-local loader summary.
- `analysis/host_rss_glm53_ep_summary.json`: filtered host RSS summary for the
  actual GLM EP rank workers.
- `analysis/checkpoint_metadata_full15.json`: DCP model key metadata for
  step-10 and step-15 checkpoints.
- `analysis/checkpoint_non_lora_keys_step10.json`: dataloader, scheduler, and
  train-state keys in step-10.
- `analysis/checkpoint_optimizer_key_probe_step10.json`: optimizer key summary.
- `analysis/resume_loss_compare_11_15.json`: TensorBoard loss comparison.
- `analysis/adapter_key_shape_check.json`: adapter schema/key/shape check.
- `adapters/full15_step15_strict/export_summary.json`: adapter export summary.

## Inputs

All training inputs came from HDFS-staged Waystone artifacts under
`/data05/philip.yang/glm53-ladder-cache`, mounted inside rootfs as
`/project/formal-cache`.

| Input | HDFS source | manifest md5 | files | mode |
| --- | --- | --- | ---: | --- |
| Countdown SFT | `hdfs://harunava/user/tiger/waystone/torchtitan/datasets/countdown-sft/v1` | `66f71a353be516264fcc02fc08116ddb` | 8 | quick |
| GLM-5.3-Flash BF16 | `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/zai-org/GLM-5.3-Flash-BF16/a5b45eb41df6402735dedc900be14a42e8d5e538` | `365857f8b1d7cb0841992fff91240ccf` | 130 | quick |

## GPU gate

The ticket gate was all 8 GPUs with at least 120 GiB free and no
`Exclusive_Process` compute mode. The full-depth leg waited twice, then launched
at `2026-10-14T20:39:55Z` with minimum free memory `126998` MiB and all GPUs in
`Default` compute mode.

| timestamp | decision | min free MiB | active compute processes | compute mode |
| --- | --- | ---: | ---: | --- |
| `2026-10-14T20:37:53Z` | wait_low_free | 108422 | 8 | all Default |
| `2026-10-14T20:38:54Z` | wait_low_free | 108358 | 8 | all Default |
| `2026-10-14T20:39:55Z` | launch | 126998 | 8 | all Default |

The resume leg launched at `2026-10-14T20:51:08Z` with the same 120 GiB
threshold satisfied. The run was shared-host evidence, not a pristine exclusive
performance campaign; the gate did not find an `Exclusive_Process` holder.

## Launch config

Resolved training shape:

- 45 GLM-5.3-Flash layers.
- EP degree 8 for sparse experts, FSDP for non-expert parameters.
- grouped expert implementation.
- LoRA rank 16, alpha 64, dropout 0.
- sequence length 512, local batch size 1, 32,768 tokens per optimizer step.
- seed 42 with deterministic mode enabled.

Entrypoint:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_ep \
TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST=/data05/philip.yang/glm53-ladder-cache \
scripts/rootfs/enter_rootfs.sh \
  --rootfs-store /data05/philip.yang/torchtitan-rootfs-store -- bash -lc '
    unset CC CXX CUDAHOSTCXX
    cd /workspace/torchtitan
    torchrun --nproc_per_node=8 \
      -m torchtitan.train \
      --module torchtitan.experiments.glm53_flash_ep.config_registry \
      --config glm53_flash_ep_countdown_lora_truncated \
      --output-dir <run-output-dir>'
```

Environment:

```bash
GLM53_FLASH_COUNTDOWN_DATA_ROOT=/project/formal-cache/datasets/countdown-sft/v1
GLM53_FLASH_COUNTDOWN_DATA_FILE=/project/formal-cache/datasets/countdown-sft/v1/train/formatting.jsonl
GLM53_FLASH_COUNTDOWN_SPLIT=train
GLM53_FLASH_COUNTDOWN_MAX_DOCS=64
GLM53_FLASH_COUNTDOWN_HF_ASSETS_PATH=/project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16
GLM53_FLASH_EXPERTS_IMPL=grouped
GLM53_FLASH_EP_DEGREE=8
GLM53_FLASH_EP_TRUNCATED_NUM_LAYERS=45
GLM53_FLASH_FULL_STEPS=15
GLM53_FLASH_FULL_SEQ_LEN=512
GLM53_FLASH_FULL_LOCAL_BATCH_SIZE=1
GLM53_FLASH_FULL_TOKENS_PER_STEP=32768
GLM53_FLASH_FULL_CHECKPOINT_INTERVAL=10
GLM53_FLASH_FULL_KEEP_LATEST_K=10
GLM53_FLASH_SEED=42
GLM53_FLASH_DETERMINISTIC=1
GLM53_FLASH_LOSS_CHUNKS=8
GLM53_FLASH_EP_VERIFY_FROZEN_BASE_LOAD=1
TRITON_CACHE_DIR=/workspace/torchtitan/.scratch/glm53-flash-sft-ladder/evidence/task-08b/triton_cache_prewarm
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

Resume used the same module/config/env with `--resume-step 10`, output directory
`.scratch/glm53-flash-sft-ladder/evidence/task-13c/t13c-20261014T203617Z/runs/resume_from10`,
and a Waystone-staged checkpoint at `checkpoint/step-10`.

## Rank-local loader evidence

The dedicated load-only probe produced one JSON file per rank. Each rank loaded
its expected contiguous 36-expert slice across sparse layers 3-44, for 42 sparse
layers total. All ranks skipped the same 906 LoRA tensors during frozen-base HF
load.

| rank | experts per sparse layer | sparse layers | expert keys | tensors loaded | bytes loaded | load seconds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0-35 | 42 | 4536 | 1330 | 78331751136 | 21.145 |
| 1 | 36-71 | 42 | 4536 | 1330 | 78331751136 | 21.118 |
| 2 | 72-107 | 42 | 4536 | 1330 | 78331751136 | 21.136 |
| 3 | 108-143 | 42 | 4536 | 1240 | 78331750776 | 21.179 |
| 4 | 144-179 | 42 | 4536 | 1229 | 78331747960 | 21.134 |
| 5 | 180-215 | 42 | 4536 | 1229 | 78331747960 | 21.217 |
| 6 | 216-251 | 42 | 4536 | 1229 | 78331747960 | 21.184 |
| 7 | 252-287 | 42 | 4536 | 1229 | 78331747960 | 21.124 |

Checks passed:

- 8 rank files present.
- EP8 expert ownership matches 36 local experts per rank.
- All ranks saw 42 sparse layers and 4536 expert keys.
- All ranks skipped 906 LoRA tensors.
- Loaded byte range was narrow: `78331747960` to `78331751136`.

The small tensor-count and byte-count differences come from non-expert HF shard
membership, not from expert ownership; the expert range check is exact.

## Full-depth training metrics

Uninterrupted run status: `exit_status=0`, finished
`2026-10-14T20:49:08Z`.

Steps 3-15:

- mean wall time from metric timestamps: `29013.4` ms/optimizer step
- min/max wall time: `28484.0` / `32391.0` ms/optimizer step
- mean printed `tps`: `142.77`
- printed `tps` range: `141` to `144`
- peak printed CUDA memory: `81.46` GiB/rank
- filtered rank-worker host RSS: max `40.711` GiB/rank, range
  `28.429` to `40.711` GiB/rank, max matched-rank RSS sum `276.837` GiB

Per-step scalars:

| step | loss | grad norm | printed tps | CUDA GiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.67393 | 3.3166 | 69 | 81.44 |
| 2 | 1.21999 | 2.6693 | 144 | 81.46 |
| 3 | 0.66938 | 2.0616 | 144 | 81.46 |
| 4 | 0.35819 | 2.1213 | 144 | 81.46 |
| 5 | 0.17780 | 1.8200 | 143 | 81.46 |
| 6 | 0.11197 | 0.3109 | 143 | 81.46 |
| 7 | 0.09790 | 0.2019 | 142 | 81.46 |
| 8 | 0.09499 | 0.1740 | 142 | 81.46 |
| 9 | 0.09373 | 0.2296 | 142 | 81.46 |
| 10 | 0.09168 | 0.1888 | 143 | 81.46 |
| 11 | 0.08103 | 0.1729 | 143 | 81.46 |
| 12 | 0.07985 | 0.1326 | 144 | 81.46 |
| 13 | 0.07745 | 0.1317 | 143 | 81.46 |
| 14 | 0.07739 | 0.1195 | 142 | 81.46 |
| 15 | 0.07477 | 0.1398 | 141 | 81.46 |

Loop-path comparison: T08d attempt-03 seed42 reported step 1 `1.6781`,
step 2 `1.6739`, and step 8 `0.2885`. T13c is close at step 1 only
(`1.67393`), then diverges downward from step 2 onward. That is not a failure of
the EP readiness probe; this run uses a different code path/data order and was
not intended as a bitwise loop-path parity run.

## Trainable-only checkpoint and HDFS resume

The uninterrupted run saved DCP checkpoints at step 10 and step 15:

| checkpoint | local size | model LoRA keys | non-LoRA layer keys | total DCP keys |
| --- | ---: | ---: | ---: | ---: |
| step 10 | 439M | 13590 | 0 | 13627 |
| step 15 | 439M | 13590 | 0 | 13627 |

Step-10 non-LoRA keys were expected runtime state, not frozen base weights:

- dataloader keys: 34
- `lr_scheduler.last_epoch`: 1
- `train_state.ntokens_seen` and `train_state.step`: 2
- optimizer keys: 12684 (`state`: 2720, `param_groups`: 9966)

HDFS round trip:

| artifact | result |
| --- | --- |
| published checkpoint | `hdfs://harunava/user/tiger/waystone/torchtitan/checkpoints/glm53-ladder/task-13c/t13c-20261014T203617Z/attempt-01/step10` |
| publish files | 9 |
| publish throughput | 12.472009460798159 MB/s |
| local step-10 after publish | deleted before staging |
| staged checkpoint | `.scratch/glm53-flash-sft-ladder/evidence/task-13c/t13c-20261014T203617Z/runs/resume_from10/checkpoint/step-10` |
| staged size | 439M |
| staged strict check | manifest md5 `be28f4c326792ea15a7da5791786394d`, 9 files |
| stage throughput | 14.68489312199403 MB/s |

Resume status: `exit_status=0`, finished `2026-10-14T20:55:15Z`.
Resume loaded the rank-local frozen base in `21.551` s and loaded the staged
checkpoint in `17.25` s. Filtered resume rank-worker host RSS peaked at
`42.220` GiB/rank, with max matched-rank RSS sum `319.809` GiB.

Loss comparison used full-precision TensorBoard values for
`loss_metrics/global_avg_loss`:

| step | uninterrupted | resumed | delta | exact equal |
| ---: | ---: | ---: | ---: | --- |
| 11 | 0.08102714270353317 | 0.08102714270353317 | 0.0 | true |
| 12 | 0.07984541356563568 | 0.07984541356563568 | 0.0 | true |
| 13 | 0.07745101302862167 | 0.07745101302862167 | 0.0 | true |
| 14 | 0.07739074528217316 | 0.07739074528217316 | 0.0 | true |
| 15 | 0.07476678490638733 | 0.07476678490638733 | 0.0 | true |

Result: all exact equal, max absolute delta `0.0`.

## Adapter export

The first export attempt with
`python -m torchtitan.experiments.glm53_flash.export_glm_lora` failed because
the reader accepted optimizer-state keys such as
`optimizer.state.layers.0.self_attn.q_proj.lora_a.exp_avg` as LoRA model keys.

The strict artifact-local export filtered to checkpoint keys ending exactly in
`.lora_a` or `.lora_b`:

| field | value |
| --- | ---: |
| source model LoRA tensors | 906 |
| exported tensors | 906 |
| module pairs | 453 |
| skipped optimizer LoRA-state keys | 12684 |
| adapter size | 129M |
| key/shape errors | 0 |
| T08 export schema match | true |

Target modules matched the T08 export path:

```text
q_proj, k_proj, v_proj, o_proj, q_a_proj, q_b_proj,
kv_a_proj_with_mqa, kv_b_proj, f_a_proj, f_b_proj, g_a_proj, g_b_proj,
shared_experts.gate_proj, shared_experts.up_proj, shared_experts.down_proj
```

Adapter HDFS artifact:

`hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-ladder/task-13c/t13c-20261014T203617Z/attempt-01/adapter-full15-step15`

Publish summary: 3 files, `HDFS_PUT_MBPS=11.103914088189281`.

## Warnings and caveats

- The training logs include FSDP2 warnings about wrapped modules returning view
  tensors. The run completed, checkpointed, resumed, and matched losses exactly;
  this warning should still be tracked separately before broader hardening.
- Throughput is shared-host readiness evidence, not exclusive performance
  evidence. The gate verified free memory and compute mode, but other compute
  processes were present.
- Adapter export is proven for this artifact via the strict exporter. Patch the
  reusable export module before relying on the original command for future DCP
  checkpoints that include optimizer state.

## Verification

All Python/model/checkpoint/log parsing ran inside `TORCHTITAN_ROOTFS_ENV=glm53_ep`
with the repo-local rootfs store.

Passed:

- HDFS strict input check for dataset and model.
- Full-depth GPU gate with 120 GiB threshold.
- 15-step full-depth EP8 training.
- Per-rank loader probe and summary generation.
- Trainable-only checkpoint metadata/key probes.
- Waystone publish/delete/stage/strict-check for step-10 checkpoint.
- Resume from staged step-10 checkpoint to step 15.
- TensorBoard loss comparison for steps 11-15: exact equality.
- Strict PEFT adapter export and key/shape validation.
- Waystone adapter publish.

No source changes were made for T13c, and no commit was created.
