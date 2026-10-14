# T06 report: rung 2 Qwen3 30B-A3B MoE Countdown LoRA SFT

Status: PASS for the rung mechanics, replicated dev eval, deterministic resume
gate, router-stat check, adapter export, vLLM MoE LoRA serving path, and HDFS
publish. HF logits parity is not numerical-identity evidence for this MoE rung:
the top-1 agreement is high, but the max logit delta is too large to call exact.

I did not create or amend a commit. Current `HEAD` at report creation is
`04497f671` on `ultron/mainline`; this report and the source changes remain
uncommitted.

Surface: repo-local research program. Training configs live under
`torchtitan/experiments/glm53_flash_sft_ladder/`; host orchestration and ignored
artifacts live under `experiments/glm53_flash_sft_ladder/`.

Run:

- `run_id`: `t06-rung2-20261014T121604Z`
- `attempt_id`: `attempt-01`
- `model_key`: `qwen3_30b_a3b`
- local result root:
  `/data05/philip.yang/glm53-ladder-results/t06-rung2-20261014T121604Z/attempt-01`
- rootfs result root:
  `/project/formal-cache/t06-rung2-20261014T121604Z/attempt-01`
- `environment_class`: `shared_gpu`
- claim labels: `smoke`, `deterministic_comparison`, `replicated_eval`

## What changed

- Added `qwen3_30b_a3b_countdown_lora` to the ladder config registry with the
  frozen rank 16, alpha 64, LR `1e-4`, sequence length 512, 94-step recipe.
- Added 30B-A3B Arm A and Arm B support in `run_rung.sh`:
  Arm A targets attention `wqkv` and `wo`; Arm B additionally targets routed
  expert `w13` and `w2` grouped linears.
- Used FSDP plus Qwen3 MoE expert parallelism with
  `parallelism.expert_parallel_degree=4`.
- Kept CPU offload disabled; the 4-GPU runs fit in shared GPU memory.
- Fixed 30B parity memory pressure by loading/converting the TorchTitan model on
  CPU and sequencing HF and TorchTitan GPU forwards.
- Added deterministic resume loss comparison tooling for full-precision
  TensorBoard scalars.
- Added router-stat collection for base and adapter routes on dev prompts using
  fp32 router logits.
- Extended Qwen3 LoRA PEFT export for grouped MoE expert tensors, and made PEFT
  adapter metadata infer its target modules from exported tensors.
- Added vLLM MoE backend and tensor-parallel passthrough in the Countdown eval
  CLI and runner.
- Added focused tests for the 30B config/env path, Arm A versus Arm B LoRA
  targets, grouped expert export, PEFT target inference, and vLLM passthrough.

## Fixed rung config

| field | value |
|---|---|
| model | Qwen3 30B-A3B |
| dataset | `countdown-sft/v1` formatting train and dev |
| LoRA Arm A | rank 16, alpha 64, targets `wqkv`, `wo` |
| LoRA Arm B | rank 16, alpha 64, targets `wqkv`, `wo`, `routed_experts.w13`, `routed_experts.w2` |
| LoRA LR | `1e-4` |
| sequence length | 512 |
| local batch | 64 |
| tokens per optimizer step | 32768 |
| steps | 94 |
| checkpoint interval | 31 |
| train GPUs | 4 |
| resume GPUs | 4 |
| expert parallel degree | 4 |
| CPU offload | disabled |
| eval | dev split, 500 problems, 32 rollouts |
| vLLM eval | `VLLM_MOE_BACKEND=triton`, `VLLM_TENSOR_PARALLEL_SIZE=4`, `VLLM_GPU_MEMORY_UTILIZATION=0.20` |

Trainer/FSDP work used `TORCHTITAN_ROOTFS_ENV=glm53_train` plus
`unset CC CXX`. vLLM eval used the default rootfs environment plus
`unset CC CXX`.

## Staged inputs

`stage.sh` was run as required and was a fast no-op for already staged inputs.

| input | HDFS URI | files | skipped | manifest md5 |
|---|---|---:|---:|---|
| dataset | `hdfs://harunava/user/tiger/waystone/torchtitan/datasets/countdown-sft/v1` | 8 | 8 | `66f71a353be516264fcc02fc08116ddb` |
| Qwen3 30B-A3B | `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-30B-A3B/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | 26 | 26 | `31dc7602093b6a1bac1e7c6f9d9ec140` |

Rootfs-visible mirrors:

- `/project/formal-cache/datasets/countdown-sft/v1`
- `/project/formal-cache/model-weights/Qwen/Qwen3-30B-A3B`

## Gates

### A. Config dry-run

PASS. Qwen3 30B-A3B Arm A ran two steps on 4 GPUs with finite, falling loss.

| config | step 1 loss | step 2 loss | memory |
|---|---:|---:|---:|
| `dry_qwen3_30b_a3b_armA_lora` | 2.84552 | 2.31731 | 39.78 GiB |

Claim label: `smoke`.

### B. HF logits parity

WEAK PASS as a smoke check only. The parity script compared TorchTitan Qwen3
30B-A3B against HF Transformers on four Countdown prompts in bf16. It proved
the HF checkpoint can be loaded into the TorchTitan MoE path and produces mostly
aligned next-token argmaxes, but it does not prove numerical identity.

| metric | value |
|---|---:|
| prompts | 4 |
| valid tokens | 114 |
| max abs logit diff | 3.65625 |
| top-1 agreement | 0.9122806787490845 |
| dtype | `bfloat16` |

Artifact:
`/data05/philip.yang/glm53-ladder-results/t06-rung2-20261014T121604Z/attempt-01/parity/qwen3_30b_a3b.json`.

Claim label: `smoke`.

### C. LoRA SFT seeds

PASS. Arm A completed two seeds and Arm B completed one seed with the frozen
rung knobs on 4 GPUs.

| run | targets | step 1 loss | step 94 loss |
|---|---|---:|---:|
| Arm A seed 42 | attention | 2.84552 | 0.05633 |
| Arm A seed 43 | attention | 2.84552 | 0.05581 |
| Arm B seed 42 | attention + routed experts | 2.84552 | 0.02034 |

Claim label: `smoke`.

### D. Resume gate

PASS. Arm A seed 42 used `GLM53_LADDER_DETERMINISTIC=1`.

Procedure:

1. Run A: uninterrupted 94 steps with `training.steps=94`.
2. Run B stop leg: stop after step 62 while preserving `training.steps=94`.
3. Publish the step-62 checkpoint to HDFS.
4. Delete the local step-62 checkpoint directory.
5. Stage the checkpoint back from HDFS.
6. Resume from step 62 to step 94.
7. Compare full-precision TensorBoard scalar losses for steps 63-94.

HDFS checkpoint:
`hdfs://harunava/user/tiger/waystone/torchtitan/checkpoints/glm53-ladder/t06-rung2-20261014T121604Z/attempt-01/resume_stop62_qwen3_30b_a3b_armA_step62`.

Publish/stage stats:

- checkpoint publish: `FILES=5`, `HDFS_PUT_MBPS=27.89170263307187`
- checkpoint stage-back: `FILES=5`, `SKIPPED=0`,
  `HDFS_GET_MBPS=29.46034815576696`

Comparison:

- tag: `loss_metrics/global_avg_loss`
- compared steps: 63-94
- `num_steps`: 32
- `all_exact_equal`: true
- `max_abs_delta`: 0

Artifact:
`/data05/philip.yang/glm53-ladder-results/t06-rung2-20261014T121604Z/attempt-01/logs/resume_loss_compare.json`.

Claim label: `deterministic_comparison`.

### E. Throughput and memory

PASS as smoke evidence. Throughput is labelled `shared_gpu` and is not
performance evidence. Steady window below is steps 84-93, satisfying the >=10
steady-step requirement.

| run | steady steps | avg tps | min/max tps | avg memory | avg TFLOPS | avg MFU |
|---|---:|---:|---:|---:|---:|---:|
| Arm A seed 42 | 10 | 502.2 | 499 / 505 | 39.78 GiB | 9.80 | 0.44% |
| Arm A seed 43 | 10 | 484.8 | 431 / 507 | 39.78 GiB | 9.46 | 0.42% |
| Arm B seed 42 | 10 | 448.5 | 365 / 463 | 40.50 GiB | 8.86 | 0.39% |

Claim label: `smoke`.

### F. Router stats

PASS. Router histograms were collected on the 500-problem dev split using fp32
router logits. No expert collapse was detected before or after SFT.

| run | prompts | valid tokens | top-k | experts | max fraction | normalized entropy | zero experts | collapse |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| base | 500 | 34892 | 8 | 128 | 0.021654916122129237 | 0.9856424979506061 | 0 | false |
| Arm A seed 42 | 500 | 34892 | 8 | 128 | 0.02092192515476327 | 0.9856891789558403 | 0 | false |
| Arm B seed 42 | 500 | 34892 | 8 | 128 | 0.02057576772612633 | 0.9870672022303536 | 0 | false |

Artifacts:

- `/data05/philip.yang/glm53-ladder-results/t06-rung2-20261014T121604Z/attempt-01/router_stats/qwen3_30b_a3b_base.json`
- `/data05/philip.yang/glm53-ladder-results/t06-rung2-20261014T121604Z/attempt-01/router_stats/qwen3_30b_a3b_armA_seed42.json`
- `/data05/philip.yang/glm53-ladder-results/t06-rung2-20261014T121604Z/attempt-01/router_stats/qwen3_30b_a3b_armB_seed42.json`

Claim label: `smoke`.

### G. Adapter export

PASS. Arm A exported attention-only PEFT adapters; Arm B exported attention
plus per-expert PEFT tensors from TorchTitan grouped expert LoRA weights.

| adapter | exported tensors | source tensors | rank | alpha | target modules | heads | kv heads | head dim |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| `qwen3_30b_a3b_armA_seed42` | 384 | 192 | 16 | 64 | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 32 | 4 | 128 |
| `qwen3_30b_a3b_armA_seed43` | 384 | 192 | 16 | 64 | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 32 | 4 | 128 |
| `qwen3_30b_a3b_armB_seed42` | 37248 | 384 | 16 | 64 | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | 32 | 4 | 128 |

Claim label: `smoke`.

### H. Dev eval

PASS. Base and adapters were evaluated on the 500-problem dev split with 32
rollouts. vLLM 0.27 served Qwen3 MoE LoRA successfully after using the default
rootfs, unsetting stale host compilers, forcing the Triton MoE backend, and
using TP=4.

| eval | n | pass@1 | pass@1 CI | pass@32 | pass@32 CI | strict pass@1 |
|---|---:|---:|---|---:|---|---:|
| `qwen3_30b_a3b_base` | 500 | 0.006 | [0.000, 0.012] | 0.150 | [0.118, 0.184] | 0.000 |
| `qwen3_30b_a3b_armA_seed42` | 500 | 0.440 | [0.394, 0.484] | 0.922 | [0.898, 0.944] | 0.440 |
| `qwen3_30b_a3b_armA_seed43` | 500 | 0.476 | [0.430, 0.526] | 0.932 | [0.908, 0.954] | 0.472 |
| `qwen3_30b_a3b_armB_seed42` | 500 | 0.600 | [0.546, 0.644] | 0.924 | [0.900, 0.944] | 0.594 |

Claim label: `replicated_eval` for Arm A. Arm B is a one-seed extension, so it
is directional rung evidence, not replicated Arm B evidence.

### I. Publish and pruning

PASS. Published HDFS paths contain `MANIFEST.jsonl` and `_COMPLETE`.

| artifact | HDFS URI | files | HDFS put MB/s |
|---|---|---:|---:|
| adapters | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-ladder/t06-rung2-20261014T121604Z/attempt-01/adapters` | 9 | 22.39594543774196 |
| eval | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-ladder/t06-rung2-20261014T121604Z/attempt-01/eval` | 30 | 2.2706571207756436 |
| router stats | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-ladder/t06-rung2-20261014T121604Z/attempt-01/router_stats` | 3 | 0.04143494434729793 |
| logs | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-ladder/t06-rung2-20261014T121604Z/attempt-01/logs` | 37 | 0.05113516137499016 |
| checkpoint-free bundle | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-ladder/t06-rung2-20261014T121604Z/attempt-01/bundle` | 171 | 0.7846992500011366 |
| resume checkpoint | `hdfs://harunava/user/tiger/waystone/torchtitan/checkpoints/glm53-ladder/t06-rung2-20261014T121604Z/attempt-01/resume_stop62_qwen3_30b_a3b_armA_step62` | 5 | 27.89170263307187 |

Local checkpoint payloads were pruned after publish where safe. Logs,
TensorBoard, structured evidence, adapters, eval outputs, summaries, router
stats, and manifests remain local under `/data05`.

## Rung comparison

Claim label: `replicated_eval` for two-seed adapter means.

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
| Qwen3 30B-A3B | base | 0.006 | 0.150 | 0.000 |
| Qwen3 30B-A3B | Arm A adapter mean | 0.458 | 0.927 | 0.456 |
| Qwen3 30B-A3B | Arm A seed 42 | 0.440 | 0.922 | 0.440 |
| Qwen3 30B-A3B | Arm A seed 43 | 0.476 | 0.932 | 0.472 |
| Qwen3 30B-A3B | Arm B seed 42 | 0.600 | 0.924 | 0.594 |

The MoE rung de-risks the GLM-5.3-Flash path: HF MoE checkpoint loading, FSDP
plus EP training, attention-only LoRA, grouped expert LoRA, router-stat capture,
resume-from-HDFS, PEFT export, and vLLM MoE LoRA eval all completed. Arm A
improves dev pass@1 over the 30B-A3B base across both seeds. Arm B improves
pass@1 further in its single seed while preserving router balance, but it needs
a second seed before making a replicated Arm B claim.

## Verification

Final source verification is pending after this report write and should include:

```bash
bash -n experiments/glm53_flash_sft_ladder/run_rung.sh \
  experiments/glm53_flash_sft_ladder/stage.sh \
  experiments/glm53_flash_sft_ladder/publish.sh \
  experiments/countdown_search_distill/run_eval.sh
```

Focused pytest in the training rootfs:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train \
TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST=/data05/philip.yang/glm53-ladder-results \
scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'unset CC CXX; cd /workspace/torchtitan && pytest -q tests/unit_tests/test_countdown_search_distill.py -k "qwen3_30b or moe or target_modules or lora_export"'
```

Narrow pre-commit hooks should be run only on the changed T06 files. Per
environment facts, `pyrefly-check` should not be run because it has
pre-existing repo-wide failures and can auto-edit unrelated files.

## Honesty notes

- Throughput is labelled `shared_gpu`; it is not performance evidence.
- The parity result is not exact numerical parity. It is only a smoke-level
  checkpoint-load/logit-alignment gate for this MoE rung.
- The final vLLM path used the default rootfs with `unset CC CXX`,
  `VLLM_MOE_BACKEND=triton`, `VLLM_TENSOR_PARALLEL_SIZE=4`, and
  `VLLM_GPU_MEMORY_UTILIZATION=0.20`. Earlier attempts without that combination
  hit FlashInfer cubin download/JIT issues and stale host compiler leakage.
- Arm A adapter metadata was repaired after source export was fixed because the
  original step-94 DCP directories had already been pruned.
- `GLM53_LADDER_KEEP_LATEST_K=1` is invalid for TorchTitan checkpointing; the
  run used keep-latest 2 and pruned later.
- No core trainer or model code was changed.
- No commit was created or amended.
