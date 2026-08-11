# Countdown Rootfs Full Adapter Evaluation Report

Date: 2026-08-11

## Scope

This report records the completed full Countdown scaffold-to-policy experiment:
smoke, reduced pilot, corrected full pilot, LoRA export, and adapter evaluation.
All real Python setup, generation, training, export, evaluation, validation, and
summarization used the TorchTitan bwrap rootfs entrypoint:

```bash
scripts/rootfs/enter_rootfs.sh -- <command>
```

## Environment And Validation

Rootfs imports and GPU visibility were verified before the run:

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

Result: `26 passed, 14 warnings`.

## Full Pilot

The corrected full pilot command was:

```bash
MODE=full NGPU=8 PASS1_MIN=0.0 PASS1_MAX=0.20 PASS32_MIN=0.35 PASS32_MAX=0.70 \
  experiments/countdown_search_distill/run_full_pilot.sh
```

Full preflight passed with 947 train matched examples and 237 dev matched
examples against thresholds of 300 and 100. All four full arms wrote fresh
mode-scoped checkpoints at `step-94`.

## Adapter Export

The TorchTitan LoRA checkpoints were exported to PEFT/vLLM adapter directories
before evaluation. TorchTitan internal checkpoints were not passed directly as
`LORA_ADAPTER`.

| Arm | Adapter directory | Source checkpoint |
| --- | --- | --- |
| `raw` | `experiments/countdown_search_distill/results/adapters/full/raw/` | `experiments/countdown_search_distill/results/train/full/raw/checkpoint/step-94/` |
| `clean` | `experiments/countdown_search_distill/results/adapters/full/clean/` | `experiments/countdown_search_distill/results/train/full/clean/checkpoint/step-94/` |
| `hindsight` | `experiments/countdown_search_distill/results/adapters/full/hindsight/` | `experiments/countdown_search_distill/results/train/full/hindsight/checkpoint/step-94/` |
| `curriculum` | `experiments/countdown_search_distill/results/adapters/full/curriculum/` | `experiments/countdown_search_distill/results/train/full/curriculum/checkpoint/step-94/` |

Each export summary reported `source_tensor_count=282`,
`exported_tensor_count=394`, `rank=32`, and `alpha=64`.

## Evaluation Matrix

Every split/arm summary below was validated to contain the expected number of
problems and 32 rollouts per problem.

| Split | Arm | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | easy | elicitable | unreached |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | `base` | 0.018 | 0.038 | 0.088 | 0.174 | 0.278 | 0.474 | 9 | 228 | 263 |
| dev | `raw` | 0.208 | 0.326 | 0.494 | 0.714 | 0.838 | 0.896 | 104 | 344 | 52 |
| dev | `clean` | 0.182 | 0.330 | 0.546 | 0.736 | 0.862 | 0.916 | 91 | 367 | 42 |
| dev | `hindsight` | 0.126 | 0.244 | 0.424 | 0.590 | 0.778 | 0.894 | 63 | 384 | 53 |
| dev | `curriculum` | 0.178 | 0.302 | 0.480 | 0.676 | 0.836 | 0.896 | 89 | 359 | 52 |
| iid_test | `base` | 0.018 | 0.047 | 0.088 | 0.161 | 0.272 | 0.466 | 18 | 448 | 534 |
| iid_test | `raw` | 0.169 | 0.319 | 0.505 | 0.695 | 0.839 | 0.904 | 169 | 735 | 96 |
| iid_test | `clean` | 0.181 | 0.340 | 0.533 | 0.748 | 0.857 | 0.914 | 181 | 733 | 86 |
| iid_test | `hindsight` | 0.141 | 0.244 | 0.419 | 0.621 | 0.793 | 0.882 | 141 | 741 | 118 |
| iid_test | `curriculum` | 0.156 | 0.286 | 0.463 | 0.668 | 0.824 | 0.896 | 156 | 740 | 104 |
| ood_test | `base` | 0.024 | 0.052 | 0.092 | 0.194 | 0.308 | 0.484 | 12 | 230 | 258 |
| ood_test | `raw` | 0.136 | 0.252 | 0.430 | 0.610 | 0.792 | 0.884 | 68 | 374 | 58 |
| ood_test | `clean` | 0.168 | 0.310 | 0.490 | 0.678 | 0.810 | 0.902 | 84 | 367 | 49 |
| ood_test | `hindsight` | 0.136 | 0.240 | 0.402 | 0.572 | 0.756 | 0.864 | 68 | 364 | 68 |
| ood_test | `curriculum` | 0.124 | 0.238 | 0.394 | 0.596 | 0.784 | 0.874 | 62 | 375 | 63 |

## Interpretation

All trained arms improved strongly over base on dev, IID, and OOD. The original
full-run justification gate required at least one reduced-pilot arm to improve
dev pass@1 by 5 absolute points while keeping dev pass@32 within 2 absolute
points of base. In the completed full adapter evaluation, every arm improves
dev pass@1 by at least 10.8 absolute points and improves dev pass@32 by at least
42.0 absolute points.

`clean` is the strongest overall arm by pass@32 on all three splits and by
pass@1 on IID/OOD. `raw` has the best dev pass@1. `hindsight` increases the
number of dev and IID elicitable problems most, but does not translate that into
the best pass@1 or pass@32.

The next useful experiment step is not environment repair. It is a targeted
analysis of solved/unsolved examples, base-elicitable subset behavior, and why
`clean` generalizes best on IID/OOD while `raw` wins dev pass@1.
