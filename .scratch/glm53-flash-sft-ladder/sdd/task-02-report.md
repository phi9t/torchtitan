# T02 report: rung 0 Qwen3 Countdown LoRA SFT

Status: PASS. No commit was made.

Surface: repo-local research program. Training configs live under
`torchtitan/experiments/glm53_flash_sft_ladder/`; host orchestration and
ignored artifacts live under `experiments/glm53_flash_sft_ladder/`.

Run:

- `run_id`: `t02-rung0-20261014T091354Z`
- `attempt_id`: `attempt-01`
- local result root:
  `experiments/glm53_flash_sft_ladder/results/t02-rung0-20261014T091354Z/attempt-01`
- `environment_class`: `shared_gpu`
- claim labels: `smoke`, `deterministic_comparison`, `replicated_eval`

## What changed

- Added experiment-owned rung-0 configs:
  `qwen3_0_6b_countdown_lora`, `qwen3_1_7b_countdown_lora`,
  `qwen3_0_6b_countdown_full`.
- Reused and minimally generalized `_qwen3_countdown_lora_sft` for optional
  learning rate, no-LoRA full fine-tune, checkpoint interval, and fixed-shape
  varlen metadata.
- Added reusable HF-vs-TorchTitan Qwen3 parity script and a stop-after-step
  training entrypoint for deterministic resume testing.
- Added `run_rung0.sh` host orchestrator for stage, mirror, dry-run, parity,
  train, resume, export, eval, and publish.
- Added optional vLLM `--gpu-memory-utilization` support and runner env
  passthrough, with focused unit tests.

## Fixed rung config

| field | value |
|---|---|
| models | Qwen3 0.6B, Qwen3 1.7B |
| dataset | `countdown-sft/v1` formatting train and dev |
| LoRA | rank 16, alpha 64, targets `wqkv`, `wo`, `w13`, `w2` |
| LoRA LR | `1e-4` |
| full fine-tune LR | `1e-5` |
| sequence length | 512 |
| local batch | 64 |
| tokens per optimizer step | 32768 |
| steps | 94 |
| checkpoint interval | 31 |
| dataloader varlen metadata | `max_num_documents=64` |
| TensorBoard | enabled for full-precision resume comparison |

Seeds and determinism are env-read config fields. Trainer/FSDP work used
`TORCHTITAN_ROOTFS_ENV=glm53_train` plus `unset CC CXX`. vLLM eval used the
default rootfs with `CUDA_VISIBLE_DEVICES=0` and
`VLLM_GPU_MEMORY_UTILIZATION=0.35`.

## Staged inputs

All inputs were staged from HDFS through the T01 storage wrappers and mirrored
into the repo-local ignored rootfs-visible cache.

| input | HDFS URI | files | manifest md5 |
|---|---|---:|---|
| dataset | `hdfs://harunava/user/tiger/waystone/torchtitan/datasets/countdown-sft/v1` | 8 | `66f71a353be516264fcc02fc08116ddb` |
| Qwen3 0.6B | `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca` | 10 | `441136422c0f62fc97fb83ccf3212b6a` |
| Qwen3 1.7B | `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 12 | `501ad940034aaede84d8af89613ae65c` |

The rootfs-visible mirror used for runs:

- `experiments/glm53_flash_sft_ladder/cache/rung0-inputs/datasets/countdown-sft/v1`
- `experiments/glm53_flash_sft_ladder/cache/rung0-inputs/model-weights/Qwen/Qwen3-0.6B`
- `experiments/glm53_flash_sft_ladder/cache/rung0-inputs/model-weights/Qwen/Qwen3-1.7B`

## Gates

### A. Config dry-run

PASS. Each config ran two steps with finite, falling loss.

| config | step 1 loss | step 2 loss |
|---|---:|---:|
| `dry_qwen3_0_6b_lora` | 1.76624 | 0.67804 |
| `dry_qwen3_1_7b_lora` | 2.67164 | 1.03533 |
| `dry_qwen3_0_6b_full` | 1.76624 | 0.85567 |

Claim label: `smoke`.

### B. HF logits parity

PASS. The reusable parity script compared TorchTitan Qwen3 0.6B against HF
Transformers on four Countdown prompts in bf16.

| metric | value |
|---|---:|
| prompts | 4 |
| valid tokens | 114 |
| max abs logit diff | 0.40625 |
| top-1 agreement | 0.9912281036376953 |
| dtype | `bfloat16` |

Artifact:
`experiments/glm53_flash_sft_ladder/results/t02-rung0-20261014T091354Z/attempt-01/parity/qwen3_0_6b.json`.

Claim label: `smoke`.

### C. LoRA vs full sanity

PASS. For Qwen3 0.6B seed 42 on 2 GPUs, both losses fell over 94 steps.

| run | step 1 loss | step 94 loss |
|---|---:|---:|
| LoRA | 1.75004 | 0.01867 |
| full fine-tune | 1.75004 | 0.04586 |

Claim label: `smoke`.

### D. GPU-count ladder

PASS as a shared-GPU smoke comparison. Qwen3 0.6B LoRA seed 42 used identical
32768 tokens/step on 2, 4, and 8 GPUs.

| GPUs | step 1 loss | step 94 loss |
|---:|---:|---:|
| 2 | 1.75004 | 0.01867 |
| 4 | 1.74397 | 0.01404 |
| 8 | 1.75215 | 0.01632 |

Using rounded console loss values, the maximum per-step loss spread across
2/4/8 GPU runs was 0.03088 at step 2. The final step spread was 0.00463.

Claim label: `smoke`. Throughput from this run is not performance evidence
because the environment class was `shared_gpu`.

### E. Resume gate

PASS. Qwen3 0.6B LoRA seed 42 on 2 GPUs ran with
`GLM53_LADDER_DETERMINISTIC=1`.

Procedure:

1. Run A: uninterrupted 94 steps with `training.steps=94`.
2. Run B stop leg: stop after step 62 while preserving `training.steps=94`.
3. Publish the step-62 checkpoint to HDFS.
4. Delete the local step-62 checkpoint directory.
5. Stage the checkpoint back from HDFS.
6. Resume from step 62 to step 94.
7. Compare full-precision TensorBoard scalar losses for steps 63-94.

HDFS checkpoint:
`hdfs://harunava/user/tiger/waystone/torchtitan/checkpoints/glm53-ladder/t02-rung0-20261014T091354Z/attempt-01/resume_stop62_step62`.

Publish/stage stats:

- checkpoint publish: `FILES=3`, `HDFS_PUT_MBPS=25.274757248920018`
- checkpoint stage-back: `FILES=3`, `SKIPPED=0`,
  `HDFS_GET_MBPS=31.075417564318457`

Comparison:

- tag: `loss_metrics/global_avg_loss`
- compared steps: 63-94
- `num_steps`: 32
- `all_exact_equal`: true
- `max_abs_delta`: 0

Artifact:
`experiments/glm53_flash_sft_ladder/results/t02-rung0-20261014T091354Z/attempt-01/logs/resume_loss_compare.json`.

Claim label: `deterministic_comparison`.

### F. Dev eval

PASS. Evaluated base and adapter runs on the 500-problem dev split with 32
rollouts. Existing bootstrap intervals from the CLI are reported below.

| eval | n | pass@1 | pass@1 CI | pass@32 | pass@32 CI | strict pass@1 |
|---|---:|---:|---|---:|---|---:|
| `qwen3_0_6b_base` | 500 | 0.004 | [0, 0.01] | 0.172 | [0.14, 0.204] | 0 |
| `qwen3_0_6b_seed42` | 500 | 0.254 | [0.218, 0.286] | 0.794 | [0.752, 0.828] | 0.254 |
| `qwen3_0_6b_seed43` | 500 | 0.326 | [0.28, 0.364] | 0.84 | [0.8, 0.87] | 0.324 |
| `qwen3_1_7b_base` | 500 | 0.024 | [0.012, 0.04] | 0.47 | [0.42, 0.514] | 0.006 |
| `qwen3_1_7b_seed42` | 500 | 0.37 | [0.334, 0.408] | 0.908 | [0.882, 0.93] | 0.362 |
| `qwen3_1_7b_seed43` | 500 | 0.228 | [0.194, 0.266] | 0.914 | [0.886, 0.936] | 0.212 |

Claim label: `replicated_eval`.

### G. Export and publish

PASS. Four LoRA adapters exported successfully.

| adapter | exported tensors | source tensors | rank | alpha |
|---|---:|---:|---:|---:|
| `qwen3_0_6b_seed42` | 392 | 224 | 16 | 64 |
| `qwen3_0_6b_seed43` | 392 | 224 | 16 | 64 |
| `qwen3_1_7b_seed42` | 392 | 224 | 16 | 64 |
| `qwen3_1_7b_seed43` | 392 | 224 | 16 | 64 |

Published HDFS artifacts:

| artifact | HDFS URI | files | HDFS put MB/s |
|---|---|---:|---:|
| adapters | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-ladder/t02-rung0-20261014T091354Z/attempt-01/adapters` | 12 | 2.922881604420043 |
| eval | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-ladder/t02-rung0-20261014T091354Z/attempt-01/eval` | 36 | 1.6217504155675388 |
| logs | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-ladder/t02-rung0-20261014T091354Z/attempt-01/logs` | 33 | 0.006589117419025115 |
| checkpoint-free bundle | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-ladder/t02-rung0-20261014T091354Z/attempt-01/bundle` | 141 | 1.1252874544451323 |
| resume checkpoint | `hdfs://harunava/user/tiger/waystone/torchtitan/checkpoints/glm53-ladder/t02-rung0-20261014T091354Z/attempt-01/resume_stop62_step62` | 3 | 25.274757248920018 |

Every publish path wrote `MANIFEST.jsonl` and `_COMPLETE`.

Local checkpoint payloads were pruned after publish. The local train artifact
tree dropped from 23 GiB before pruning to 389 MiB after pruning. Logs,
TensorBoard, structured evidence, adapters, eval outputs, summaries, and
manifests remain local.

## Verification

Passed:

```bash
bash -n experiments/glm53_flash_sft_ladder/run_rung0.sh experiments/countdown_search_distill/run_eval.sh
```

Passed inside rootfs:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'unset CC CXX; cd /workspace/torchtitan && pytest -q tests/unit_tests/test_countdown_search_distill.py'
```

Result: `46 passed, 14 warnings in 9.25s`.

Passed inside rootfs, by explicit hook id on changed T02 files:

- `trailing-whitespace`
- `check-ast`
- `check-merge-conflict`
- `check-added-large-files`
- `end-of-file-fixer`
- `insert-license`
- `ufmt`
- `flake8`
- `pydoclint`
- `codespell`
- `check-no-pii`

`pyrefly-check` was intentionally not run because the task environment facts
say it fails repo-wide and may auto-edit unrelated files.

## Honesty notes

- This is not convergence or performance evidence. It is rung-0 plumbing,
  deterministic resume, and replicated dev eval evidence under shared GPU
  conditions.
- The GPU-count ladder used rounded console loss values for cross-run spread.
  The resume gate used full-precision TensorBoard scalars.
- vLLM eval used `gpu_memory_utilization=0.35`; logs reported about 57-60 GiB
  KV cache. One eval attempt exposed a stale shared vLLM/TorchInductor AOT
  cache reload path. The runner now sets per-eval `VLLM_CACHE_ROOT` for future
  runs, and the successful retry outputs were published.
- No 8B or larger model was run.
- No commit was made.
