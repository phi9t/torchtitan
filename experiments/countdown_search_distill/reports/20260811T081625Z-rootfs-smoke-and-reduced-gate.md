# Countdown Rootfs Smoke And Reduced Gate Report

Timestamp: 2026-08-11T08:16:25Z

## Command Discipline

All validation, dependency repair, asset download, generation, scoring,
training, evaluation, and summarization commands were run through:

```bash
scripts/rootfs/enter_rootfs.sh -- <command>
```

The experiment runners also re-exec through the same rootfs when
`TORCHTITAN_IN_ROOTFS` is not set.

## Environment

Rootfs imports and GPU visibility were verified after repair:

- `torch=True`
- `vllm=True`
- `datasets=True`
- `transformers=True`
- `spmd_types=True`
- `torchaudio=False`
- `torch_version=2.13.0+cu132`
- `cuda_available=True`
- `cuda_device_count=8`

`torchaudio` was removed because the vLLM install pulled a CUDA 13.0 build that
was incompatible with the rootfs Torch CUDA 13.2 build.

Qwen3-1.7B assets were downloaded under `assets/hf/Qwen3-1.7B` and vLLM loaded
the model from that path.

## Runtime Fixes

The run required the following local fixes:

- Added a constructive all-number Countdown pool generation path for the smoke
  and pilot generation filters.
- Disabled expensive exact recoverability checks during bulk vLLM scoring.
- Made `evaluate-vllm` default to rootfs-compatible vLLM settings:
  `attention_backend=TRITON_ATTN`, FlashInfer sampler off, FlashInfer autotune
  off, and `max_model_len=2048`.
- Reworked evaluation JSON loading so summary/preflight commands preserve saved
  verification results instead of recomputing exact reachability for every
  rollout.
- Added a smoke-only canonical raw fallback dataset so plumbing training can run
  even when the base model produces no verified successes.
- Disabled HF weight loading for `qwen3_debugmodel_countdown_lora_smoke`, which
  uses tokenizer-only test assets and should initialize the debug model fresh.

## Validation

Rootfs focused tests:

```text
17 passed, 14 warnings
```

Rootfs shell syntax:

```text
bash -n experiments/countdown_search_distill/run_*.sh
```

## Smoke Run

The smoke path completed its intended plumbing coverage:

- calibration: 50 problems, 32 rollouts each;
- collection: train 100, dev 50, IID 100, OOD 50;
- training: `debug_smoke`, 2 steps, checkpoint writes at steps 1 and 2;
- evaluation: dev, IID, and OOD base evaluations through vLLM.

Training evidence:

- `experiments/countdown_search_distill/data/train/raw.jsonl` has 100 rows.
- `experiments/countdown_search_distill/results/train/debug_smoke/checkpoint/step-1/__0_0.distcp`
- `experiments/countdown_search_distill/results/train/debug_smoke/checkpoint/step-2/__0_0.distcp`
- step 1 loss: 8.01678
- step 2 loss: 7.50526

Base evaluation summaries:

| Split | Problems | pass@1 | pass@32 |
| --- | ---: | ---: | ---: |
| calibration | 50 | 0.0 | 0.0 |
| dev | 50 | 0.0 | 0.0 |
| iid_test | 100 | 0.0 | 0.0 |
| ood_test | 50 | 0.0 | 0.0 |

## Reduced Preflight

Reduced adapter training was not launched. The preflight command wrote
`experiments/countdown_search_distill/data/reduced_preflight.json` and failed
the intended gate:

```text
calibration_selected=false
train_matched=0
dev_matched=0
min_train_matched=300
min_dev_matched=100
```

This is a clean stop condition, not an environment failure. The current
problem/prompt regime produced no base-elicitable bucket, so there is no
matched supervision set for the reduced pilot.

## Follow-Up

Before running reduced training, adjust the generation or prompting regime until
calibration lands in the target band:

```text
pass@1  in [0.05, 0.20]
pass@32 in [0.35, 0.70]
```

Candidate knobs are lower target range, lower solution depth, no all-number
requirement, fewer operands, prompt formatting changes, or higher rollout
budget during calibration.
