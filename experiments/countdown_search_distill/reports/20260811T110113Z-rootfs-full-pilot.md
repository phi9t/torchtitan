# Countdown Rootfs Full Pilot Report

Date: 2026-08-11

## Scope

This report records the corrected full Countdown scaffold-to-policy pilot. All
real Python setup, generation, training, evaluation, and summarization used the
TorchTitan bwrap rootfs entrypoint:

```bash
scripts/rootfs/enter_rootfs.sh -- <command>
```

The full pilot was run only after smoke and reduced mode completed. A first
full attempt exposed mode collision in the training dump folders, so the
launcher was fixed and the full pilot was rerun with output scoped under
`experiments/countdown_search_distill/results/train/full/`.

## Environment

Rootfs imports and GPU visibility were verified before the pilot:

- `torch=True`
- `vllm=True`
- `datasets=True`
- `transformers=True`
- `spmd_types=True`
- `torch_version=2.13.0+cu132`
- `cuda_available=True`
- `cuda_device_count=8`

Qwen3-1.7B assets were present at `assets/hf/Qwen3-1.7B`.

Focused validation passed under rootfs:

```bash
python -m pytest tests/unit_tests/test_countdown_search_distill.py -q
bash -n experiments/countdown_search_distill/run_*.sh
```

Result: `23 passed, 14 warnings`.

## Command

Corrected full pilot command:

```bash
MODE=full NGPU=8 PASS1_MIN=0.0 PASS1_MAX=0.20 PASS32_MIN=0.35 PASS32_MAX=0.70 \
  experiments/countdown_search_distill/run_full_pilot.sh
```

## Calibration And Preflight

Selected regime: `3n_10_50_d2_all`

Calibration summary:

| Problems | Rollouts | pass@1 | pass@32 | easy | elicitable | unreached |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 32 | 0.018 | 0.474 | 9 | 228 | 263 |

Full preflight passed:

| Check | Observed | Threshold |
| --- | ---: | ---: |
| train matched coverage | 947 | 300 |
| dev matched coverage | 237 | 100 |

## Training Artifacts

All full arms wrote fresh mode-scoped checkpoints:

| Arm | Checkpoints |
| --- | --- |
| `raw` | `step-31`, `step-62`, `step-93`, `step-94` |
| `clean` | `step-31`, `step-62`, `step-93`, `step-94` |
| `hindsight` | `step-31`, `step-62`, `step-93`, `step-94` |
| `curriculum` | `step-31`, `step-62`, `step-93`, `step-94` |

Final checkpoint paths:

- `experiments/countdown_search_distill/results/train/full/raw/checkpoint/step-94/`
- `experiments/countdown_search_distill/results/train/full/clean/checkpoint/step-94/`
- `experiments/countdown_search_distill/results/train/full/hindsight/checkpoint/step-94/`
- `experiments/countdown_search_distill/results/train/full/curriculum/checkpoint/step-94/`

## Base Evaluation

The full pilot evaluated the base Qwen3-1.7B path on dev, IID, and OOD splits.

| Split | Problems | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 500 | 0.018 | 0.038 | 0.088 | 0.174 | 0.278 | 0.474 |
| iid_test | 1000 | 0.018 | 0.047 | 0.088 | 0.161 | 0.272 | 0.466 |
| ood_test | 500 | 0.024 | 0.052 | 0.092 | 0.194 | 0.308 | 0.484 |

Bucket counts:

| Split | easy | elicitable | unreached |
| --- | ---: | ---: | ---: |
| dev | 9 | 228 | 263 |
| iid_test | 18 | 448 | 534 |
| ood_test | 12 | 230 | 258 |

Evaluation artifacts:

- `experiments/countdown_search_distill/results/eval/dev/summary.json`
- `experiments/countdown_search_distill/results/eval/iid_test/summary.json`
- `experiments/countdown_search_distill/results/eval/ood_test/summary.json`

## Interpretation

The corrected full pilot is a successful environment and training-path run. It
does not yet establish scaffold-to-policy improvement, because adapter
evaluation was intentionally not run. The available trained artifacts are
TorchTitan internal checkpoints, not validated vLLM-ready LoRA adapter
directories.

Next step: validate an export path from the TorchTitan LoRA checkpoints to a
vLLM adapter directory, then evaluate each exported arm on dev/IID/OOD and on
the base-elicitable dev bucket.
