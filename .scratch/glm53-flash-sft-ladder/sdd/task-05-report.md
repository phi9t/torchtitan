# T05 report: rung 1 Qwen3 8B Countdown LoRA SFT

Status: PASS. No commit was made.

Surface: repo-local research program. Training configs live under
`torchtitan/experiments/glm53_flash_sft_ladder/`; host orchestration and
ignored artifacts live under `experiments/glm53_flash_sft_ladder/`.

Run:

- `run_id`: `t05-rung1-20261014T104800Z`
- `attempt_id`: `attempt-01`
- `model_key`: `qwen3_8b`
- local result root:
  `experiments/glm53_flash_sft_ladder/results/t05-rung1-20261014T104800Z/attempt-01`
- `environment_class`: `shared_gpu`
- claim labels: `smoke`, `deterministic_comparison`, `replicated_eval`

## What changed

- Added `experiments/glm53_flash_sft_ladder/run_rung.sh`, a generic rung runner
  driven by `MODEL_KEY` for `qwen3_0_6b`, `qwen3_1_7b`, `qwen3_8b`, and a
  `qwen3_30b_a3b` placeholder that requires `MODEL_URI`.
- Left `experiments/glm53_flash_sft_ladder/run_rung0.sh` untouched.
- Added `qwen3_8b_countdown_lora` to the ladder config registry with the frozen
  rung-0 LoRA knobs.
- Made the ladder checkpoint retention env-controllable with
  `GLM53_LADDER_KEEP_LATEST_K`, defaulting to the previous value outside the
  runner.
- Fixed Qwen3 LoRA export for grouped-query 8B checkpoints by reading
  `num_attention_heads`, `num_key_value_heads`, and `head_dim` from the HF
  `config.json`.
- Added focused tests for the 8B ladder config and grouped-query export
  metadata.

## Fixed rung config

| field | value |
|---|---|
| model | Qwen3 8B |
| dataset | `countdown-sft/v1` formatting train and dev |
| LoRA | rank 16, alpha 64, targets `wqkv`, `wo`, `w13`, `w2` |
| LoRA LR | `1e-4` |
| sequence length | 512 |
| local batch | 64 |
| tokens per optimizer step | 32768 |
| steps | 94 |
| checkpoint interval | 31 |
| train GPUs | 2 |
| resume GPUs | 2 |
| eval | dev split, 500 problems, 32 rollouts |
| vLLM memory cap | `VLLM_GPU_MEMORY_UTILIZATION=0.35` |

Trainer/FSDP work used `TORCHTITAN_ROOTFS_ENV=glm53_train` plus `unset CC CXX`.
vLLM eval used the default rootfs environment.

## Staged inputs

`stage.sh` was run as required and was a fast no-op for already staged inputs.

| input | HDFS URI | files | skipped | manifest md5 |
|---|---|---:|---:|---|
| dataset | `hdfs://harunava/user/tiger/waystone/torchtitan/datasets/countdown-sft/v1` | 8 | 8 | `66f71a353be516264fcc02fc08116ddb` |
| Qwen3 8B | `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218` | 15 | 15 | `6ba6a83877bc9d0f826c547c5d75c291` |

Rootfs-visible mirror:

- `experiments/glm53_flash_sft_ladder/cache/qwen3_8b-inputs/datasets/countdown-sft/v1`
- `experiments/glm53_flash_sft_ladder/cache/qwen3_8b-inputs/model-weights/Qwen/Qwen3-8B`

## Gates

### A. Config dry-run

PASS. Qwen3 8B ran two steps with finite, falling loss.

| config | step 1 loss | step 2 loss |
|---|---:|---:|
| `dry_qwen3_8b_lora` | 2.66890 | 1.68418 |

Claim label: `smoke`.

### B. HF logits parity

PASS. The parity script compared TorchTitan Qwen3 8B against HF Transformers on
four Countdown prompts in bf16.

| metric | value |
|---|---:|
| prompts | 4 |
| valid tokens | 114 |
| max abs logit diff | 0.375 |
| top-1 agreement | 1.0 |
| dtype | `bfloat16` |

Artifact:
`experiments/glm53_flash_sft_ladder/results/t05-rung1-20261014T104800Z/attempt-01/parity/qwen3_8b.json`.

Claim label: `smoke`.

### C. LoRA SFT seeds

PASS. Two Qwen3 8B LoRA SFT seeds completed 94 steps on 2 GPUs with the frozen
rung knobs.

| run | step 1 loss | step 94 loss |
|---|---:|---:|
| seed 42 | 2.65779 | 0.00665 |
| seed 43 | 2.65779 | 0.00489 |

Claim label: `smoke`.

### D. Resume gate

PASS. Qwen3 8B seed 42 used `GLM53_LADDER_DETERMINISTIC=1`.

Procedure:

1. Run A: uninterrupted 94 steps with `training.steps=94`.
2. Run B stop leg: stop after step 62 while preserving `training.steps=94`.
3. Publish the step-62 checkpoint to HDFS.
4. Delete the local step-62 checkpoint directory.
5. Stage the checkpoint back from HDFS.
6. Resume from step 62 to step 94.
7. Compare full-precision TensorBoard scalar losses for steps 63-94.

HDFS checkpoint:
`hdfs://harunava/user/tiger/waystone/torchtitan/checkpoints/glm53-ladder/t05-rung1-20261014T104800Z/attempt-01/resume_stop62_qwen3_8b_step62`.

Publish/stage stats:

- checkpoint publish: `FILES=3`, `HDFS_PUT_MBPS=24.72752620871551`
- checkpoint stage-back: `FILES=3`, `SKIPPED=0`,
  `HDFS_GET_MBPS=28.791807885687366`

Comparison:

- tag: `loss_metrics/global_avg_loss`
- compared steps: 63-94
- `num_steps`: 32
- `all_exact_equal`: true
- `max_abs_delta`: 0

Artifact:
`experiments/glm53_flash_sft_ladder/results/t05-rung1-20261014T104800Z/attempt-01/logs/resume_loss_compare.json`.

Claim label: `deterministic_comparison`.

### E. Throughput and memory

PASS. Throughput is labelled `shared_gpu` and is not performance evidence.
Steady window below is steps 84-93, satisfying the >=10 steady-step requirement.

| run | steady steps | avg tps | min/max tps | avg memory | avg TFLOPS | avg MFU |
|---|---:|---:|---:|---:|---:|---:|
| seed 42 | 10 | 6150.1 | 5655 / 6330 | 15.06 GiB | 286.20 | 12.72% |
| seed 43 | 10 | 6172.8 | 4916 / 6323 | 15.06 GiB | 287.26 | 12.77% |

Claim label: `smoke`.

### F. Adapter export

PASS. Both Qwen3 8B adapters exported to PEFT format after resolving the HF
grouped-query attention shape from `config.json`.

| adapter | exported tensors | source tensors | rank | alpha | heads | kv heads | head dim |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen3_8b_seed42` | 504 | 288 | 16 | 64 | 32 | 8 | 128 |
| `qwen3_8b_seed43` | 504 | 288 | 16 | 64 | 32 | 8 | 128 |

Claim label: `smoke`.

### G. Dev eval

PASS. Base and two adapters were evaluated on the 500-problem dev split with 32
rollouts.

| eval | n | pass@1 | pass@1 CI | pass@32 | pass@32 CI | strict pass@1 |
|---|---:|---:|---|---:|---|---:|
| `qwen3_8b_base` | 500 | 0.028 | [0.016, 0.044] | 0.436 | [0.388, 0.476] | 0.012 |
| `qwen3_8b_seed42` | 500 | 0.474 | [0.430, 0.516] | 0.906 | [0.878, 0.930] | 0.474 |
| `qwen3_8b_seed43` | 500 | 0.488 | [0.446, 0.532] | 0.912 | [0.886, 0.936] | 0.486 |

Claim label: `replicated_eval`.

### H. Publish and pruning

PASS. Published HDFS paths contain `MANIFEST.jsonl` and `_COMPLETE`.

| artifact | HDFS URI | files | HDFS put MB/s |
|---|---|---:|---:|
| adapters | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-ladder/t05-rung1-20261014T104800Z/attempt-01/adapters` | 6 | 7.608031825968777 |
| eval | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-ladder/t05-rung1-20261014T104800Z/attempt-01/eval` | 18 | 2.767664009961765 |
| logs | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-ladder/t05-rung1-20261014T104800Z/attempt-01/logs` | 19 | 0.0047851663854948144 |
| checkpoint-free bundle | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-ladder/t05-rung1-20261014T104800Z/attempt-01/bundle` | 63 | 0.9744885363841057 |
| resume checkpoint | `hdfs://harunava/user/tiger/waystone/torchtitan/checkpoints/glm53-ladder/t05-rung1-20261014T104800Z/attempt-01/resume_stop62_qwen3_8b_step62` | 3 | 24.72752620871551 |

Local checkpoint payloads were pruned after publish. Logs, TensorBoard,
structured evidence, adapters, eval outputs, summaries, and manifests remain
local. The local T05 result root is 681 MiB after pruning; T05 temp publish and
stage directories on `/data05` are under 1 MiB.

## Rung comparison

Claim label: `replicated_eval`.

| model | run | pass@1 | pass@32 | strict pass@1 |
|---|---|---:|---:|---:|
| Qwen3 0.6B | base | 0.004 | 0.172 | 0.000 |
| Qwen3 0.6B | adapter mean | 0.290 | 0.817 | 0.289 |
| Qwen3 0.6B | seed 42 | 0.254 | 0.794 | 0.254 |
| Qwen3 0.6B | seed 43 | 0.326 | 0.840 | 0.324 |
| Qwen3 1.7B | base | 0.024 | 0.470 | 0.006 |
| Qwen3 1.7B | adapter mean | 0.299 | 0.911 | 0.287 |
| Qwen3 1.7B | seed 42 | 0.370 | 0.908 | 0.362 |
| Qwen3 1.7B | seed 43 | 0.228 | 0.914 | 0.212 |
| Qwen3 8B | base | 0.028 | 0.436 | 0.012 |
| Qwen3 8B | adapter mean | 0.481 | 0.909 | 0.480 |
| Qwen3 8B | seed 42 | 0.474 | 0.906 | 0.474 |
| Qwen3 8B | seed 43 | 0.488 | 0.912 | 0.486 |

SFT improvement holds at all three scales. The 8B pass@1 and strict pass@1
adapter means are materially higher than both rung-0 adapter means, so the
pass@1 improvement grows at 8B. Pass@32 stays high at 8B and is effectively tied
with 1.7B inside seed noise; the 8B pass@32 delta from base is smaller than
0.6B because the 8B base is already stronger at pass@32.

## Verification

Passed:

```bash
bash -n experiments/glm53_flash_sft_ladder/run_rung.sh \
  experiments/glm53_flash_sft_ladder/run_rung0.sh \
  experiments/glm53_flash_sft_ladder/stage.sh \
  experiments/glm53_flash_sft_ladder/publish.sh
```

Passed inside rootfs:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'unset CC CXX; cd /workspace/torchtitan && pytest -q tests/unit_tests/test_countdown_search_distill.py -k "glm53_ladder_qwen3_configs_read_env or qwen3_fused_qkv_lora_b_split_preserves_grouped_order or lora_export_config_reads_grouped_query_shape_from_hf_config or convert_torchtitan_lora_tensors_emits_peft_qwen3_keys"'
```

Result: `4 passed, 43 deselected, 14 warnings in 5.65s`.

HDFS completion check passed for adapters, eval, logs, bundle, and the resume
checkpoint.

## Honesty notes

- This is not performance evidence; throughput is labelled `shared_gpu`.
- No hyperparameter re-tuning was performed.
- The original export failure was
  `ValueError: fused QKV LoRA B output dimension mismatch: got 6144, expected 4096`.
  The cause was a 1.7B-shaped exporter default; the fix infers Qwen3 grouped
  query dimensions from the HF config.
- The export summary records the checkpoint source path that was used during
  export. Those local checkpoint payloads were intentionally deleted after
  publish.
- No core trainer or model code was changed.
- No commit was made.
