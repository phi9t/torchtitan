# Countdown Search-Distill Results

## 2026-08-12 Clean-Arm Replication And Rank-Size Sweep

The clean-arm replication and rank-size sweep completed under the bwrap rootfs.
The sweep used isolated data/results roots under
`experiments/countdown_search_distill/sweeps/replication/` and covered seed 43
with train sizes 1000 and 2000 and LoRA ranks 32 and 16.

All four cells produced TorchTitan checkpoints, PEFT/vLLM adapter exports,
dev/IID/OOD adapter matrices, and report inputs. Validation passed for every
cell. The detailed report is:

```text
experiments/countdown_search_distill/reports/20260812T043000Z-replication-rank-size-sweep.md
```

Aggregate clean-adapter results:

| Label | Dev pass@1 | Dev pass@32 | IID pass@1 | IID pass@32 | OOD pass@1 | OOD pass@32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `seed43_train1000_rank32` | 0.168 | 0.894 | 0.179 | 0.885 | 0.280 | 0.902 |
| `seed43_train1000_rank16` | 0.186 | 0.880 | 0.157 | 0.876 | 0.316 | 0.906 |
| `seed43_train2000_rank32` | 0.174 | 0.900 | 0.205 | 0.887 | 0.276 | 0.890 |
| `seed43_train2000_rank16` | 0.192 | 0.918 | 0.163 | 0.905 | 0.264 | 0.914 |

Best sweep cell: `seed43_train2000_rank16`. It is strongest on dev pass@1,
dev pass@32, IID pass@32, OOD pass@32, and OOD base-elicitable pass@1. On its
OOD base-elicitable subset, `clean` reaches 0.351 pass@1 and 1.000 pass@32 over
279 problems.

Decision: the clean scaffold-to-policy effect has replicated under a fresh
seed/split draw. The next Countdown-specific step is to replicate and sweep the
`formatting` champion because it remains the strongest strict-output arm from
the original clean full split.

## 2026-08-12 Formatting Arm Completion

The full clean-split matrix now includes the `formatting` arm. Training,
adapter export, dev/IID/OOD vLLM evaluation, summary refresh, matrix validation,
and report-input generation all ran through the bwrap rootfs.

Fresh artifacts:

- `results/train/full/formatting/checkpoint/step-94/`
- `results/adapters/full/formatting/`
- `results/eval/adapters/full/{dev,iid_test,ood_test}/formatting/summary.json`
- `results/eval/adapters/full/adapter_matrix_full.json`
- `results/manifests/report_input_20260812T001300Z-full-formatting-continuation.json`
- `reports/20260812T001300Z-formatting-arm-results-and-audit.md`

The report input passes all registry checks: required manifest stages, selected
split registry, no split overlap, base summaries present, and selected adapter
matrix.

| Split | Arm | pass@1 | pass@32 | strict pass@1 | strict pass@32 |
| --- | --- | ---: | ---: | ---: | ---: |
| dev | base | 0.026 | 0.468 | 0.008 | 0.148 |
| dev | clean | 0.180 | 0.904 | 0.080 | 0.758 |
| dev | formatting | 0.374 | 0.900 | 0.370 | 0.900 |
| iid_test | base | 0.033 | 0.509 | 0.008 | 0.167 |
| iid_test | clean | 0.208 | 0.922 | 0.108 | 0.790 |
| iid_test | formatting | 0.361 | 0.908 | 0.351 | 0.907 |
| ood_test | base | 0.024 | 0.484 | 0.004 | 0.166 |
| ood_test | clean | 0.180 | 0.890 | 0.076 | 0.658 |
| ood_test | formatting | 0.294 | 0.902 | 0.276 | 0.900 |

Decision: `formatting` is now the strongest performance arm and should be the
champion for the next Countdown iteration. `clean` remains useful as the
conciseness/style control because formatting often emits repeated answer text
after a valid strict trace.

The report input now also includes artifact-derived `analysis`:
base-elicitable subset metrics and deterministic representative examples for
wins, regressions, unchanged failures, and format failures. On the
base-elicitable OOD subset, `formatting` reaches 0.370 pass@1, 0.978 pass@32,
0.343 strict pass@1, and 0.978 strict pass@32 over 230 problems. See
`reports/20260812T002500Z-base-elicitable-and-examples.md`.

## 2026-08-11 Rootfs Smoke, Reduced Pilot, Clean-Split Full Eval

All real Python setup, generation, training, evaluation, and summarization for
this run used the TorchTitan bwrap rootfs entrypoint:

```bash
scripts/rootfs/enter_rootfs.sh -- <command>
```

Rootfs dependency preflight passed after installing the repo requirements and
vLLM in the rootfs. The verified runtime state was:

- `torch=True`
- `vllm=True`
- `datasets=True`
- `transformers=True`
- `spmd_types=True`
- `torch_version=2.13.0+cu132`
- `cuda_available=True`
- `cuda_device_count=8`

Qwen3-1.7B assets were downloaded to `assets/hf/Qwen3-1.7B` and used by vLLM
for all generation and evaluation.

## Smoke Outcome

Smoke plumbing completed. Calibration, collection, debug smoke training, and
base dev/IID/OOD evaluation all ran under the rootfs. The smoke run was a
plumbing check only; it was not used as hypothesis evidence.

Focused validation also passed under the rootfs:

```bash
python -m pytest tests/unit_tests/test_countdown_search_distill.py -q
bash -n experiments/countdown_search_distill/run_*.sh
```

Result: `33 passed, 14 warnings`.

The clean-split implementation added a split registry, problem-key exclusion
between generated splits, and matrix validation for adapter evaluation outputs.
The final validation artifact is:

```text
experiments/countdown_search_distill/data/split_registry.json
```

It reports `selected=true`, `no_problem_key_overlap=true`, and zero overlaps
across train, dev, IID, and OOD split problem keys.

## Calibration And Gates

The selected full calibration regime is recorded in
`experiments/countdown_search_distill/data/calibration_regime.json`:

| Regime | Problems | Rollouts | pass@1 | pass@32 | Buckets |
| --- | ---: | ---: | ---: | ---: | --- |
| `3n_10_50_d2_all` | 500 | 32 | 0.018 | 0.474 | easy 9, elicitable 228, unreached 263 |

The reduced and full gates both passed:

| Mode | Selected | train matched | dev matched | Thresholds |
| --- | --- | ---: | ---: | --- |
| reduced | true | 474 | 133 | train >= 300, dev >= 100 |
| full | true | 947 | 237 | train >= 300, dev >= 100 |

## Reduced Pilot

The reduced pilot completed collection, preflight, and training for `raw`,
`hindsight`, and `curriculum`, then ran base dev/IID/OOD evaluations. Reduced
mode was used to validate the experiment path before launching the full pilot.

## Clean-Split Full Run

The first full pilot exposed a split hygiene flaw: train, dev, and IID were
generated with overlapping number-target keys. The launcher now validates the
split registry after collection and before training for reduced/full modes.
Generation excludes prior split keys in order: dev excludes train, IID excludes
train+dev, and OOD excludes train+dev+IID.

Clean-split full command:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'MODE=full NGPU=8 RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-full-clean-splits \
   experiments/countdown_search_distill/run_full_pilot.sh'
```

The clean-split run completed all manifest stages:

| Stage | Return code | Duration |
| --- | ---: | ---: |
| `validate_splits` | 0 | 1s |
| `train_raw` | 0 | 28s |
| `train_clean` | 0 | 28s |
| `train_hindsight` | 0 | 28s |
| `train_curriculum` | 0 | 28s |
| `base_eval_dev` | 0 | 110s |
| `base_eval_iid_test` | 0 | 192s |
| `base_eval_ood_test` | 0 | 109s |
| `export_adapters` | 0 | 0s |
| `eval_adapters` | 0 | 2742s |

The full training checkpoints already existed from the earlier full run and
were loaded/resumed to completion rather than retrained from scratch in this
clean-split rerun. The adapter exports also already existed and were skipped.
The base evaluations and mode-scoped full adapter evaluations below were
regenerated against the clean splits.

| Arm | Final checkpoint |
| --- | --- |
| `raw` | `results/train/full/raw/checkpoint/step-94/` |
| `clean` | `results/train/full/clean/checkpoint/step-94/` |
| `hindsight` | `results/train/full/hindsight/checkpoint/step-94/` |
| `curriculum` | `results/train/full/curriculum/checkpoint/step-94/` |

## Full Adapter Export

The full TorchTitan LoRA checkpoints were exported to PEFT/vLLM adapter
directories before adapter evaluation. TorchTitan internal checkpoints were not
passed directly as `LORA_ADAPTER`.

Exported adapter directories:

| Arm | Adapter directory | Source checkpoint |
| --- | --- | --- |
| `raw` | `results/adapters/full/raw/` | `results/train/full/raw/checkpoint/step-94/` |
| `clean` | `results/adapters/full/clean/` | `results/train/full/clean/checkpoint/step-94/` |
| `hindsight` | `results/adapters/full/hindsight/` | `results/train/full/hindsight/checkpoint/step-94/` |
| `curriculum` | `results/adapters/full/curriculum/` | `results/train/full/curriculum/checkpoint/step-94/` |

Each export wrote `adapter_config.json`, `adapter_model.safetensors`, and
`export_summary.json`. The export summaries report `rank=32`, `alpha=64`, and
target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj`, and `lm_head`.

## Base And Adapter Evaluations

The full pilot evaluated the base Qwen3-1.7B path and all four exported LoRA
adapters on dev, IID, and OOD splits with 32 rollouts per problem.

| Split | Problems | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 500 | 0.026 | 0.046 | 0.084 | 0.150 | 0.284 | 0.468 |
| iid_test | 1000 | 0.033 | 0.059 | 0.099 | 0.191 | 0.310 | 0.509 |
| ood_test | 500 | 0.024 | 0.052 | 0.092 | 0.194 | 0.308 | 0.484 |

Bucket counts:

| Split | easy | elicitable | unreached |
| --- | ---: | ---: | ---: |
| dev | 13 | 221 | 266 |
| iid_test | 33 | 476 | 491 |
| ood_test | 12 | 230 | 258 |

Adapter metrics:

| Split | Arm | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | easy | elicitable | unreached |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | `raw` | 0.162 | 0.310 | 0.466 | 0.658 | 0.816 | 0.876 | 81 | 357 | 62 |
| dev | `clean` | 0.180 | 0.344 | 0.522 | 0.694 | 0.836 | 0.904 | 90 | 362 | 48 |
| dev | `hindsight` | 0.162 | 0.268 | 0.424 | 0.630 | 0.794 | 0.880 | 81 | 359 | 60 |
| dev | `curriculum` | 0.182 | 0.310 | 0.492 | 0.674 | 0.812 | 0.890 | 91 | 354 | 55 |
| iid_test | `raw` | 0.169 | 0.317 | 0.506 | 0.699 | 0.840 | 0.915 | 169 | 746 | 85 |
| iid_test | `clean` | 0.208 | 0.375 | 0.574 | 0.739 | 0.858 | 0.922 | 208 | 714 | 78 |
| iid_test | `hindsight` | 0.152 | 0.263 | 0.426 | 0.619 | 0.796 | 0.890 | 152 | 738 | 110 |
| iid_test | `curriculum` | 0.163 | 0.296 | 0.503 | 0.699 | 0.837 | 0.921 | 163 | 758 | 79 |
| ood_test | `raw` | 0.126 | 0.224 | 0.404 | 0.624 | 0.770 | 0.870 | 63 | 372 | 65 |
| ood_test | `clean` | 0.180 | 0.316 | 0.470 | 0.670 | 0.816 | 0.890 | 90 | 355 | 55 |
| ood_test | `hindsight` | 0.142 | 0.242 | 0.414 | 0.578 | 0.764 | 0.864 | 71 | 361 | 68 |
| ood_test | `curriculum` | 0.142 | 0.260 | 0.384 | 0.596 | 0.762 | 0.878 | 71 | 368 | 61 |

Adapter deltas versus base:

| Split | Arm | delta pass@1 | delta pass@32 | delta elicitable |
| --- | --- | ---: | ---: | ---: |
| dev | `raw` | +0.136 | +0.408 | +136 |
| dev | `clean` | +0.154 | +0.436 | +141 |
| dev | `hindsight` | +0.136 | +0.412 | +138 |
| dev | `curriculum` | +0.156 | +0.422 | +133 |
| iid_test | `raw` | +0.136 | +0.406 | +270 |
| iid_test | `clean` | +0.175 | +0.413 | +238 |
| iid_test | `hindsight` | +0.119 | +0.381 | +262 |
| iid_test | `curriculum` | +0.130 | +0.412 | +282 |
| ood_test | `raw` | +0.102 | +0.386 | +142 |
| ood_test | `clean` | +0.156 | +0.406 | +125 |
| ood_test | `hindsight` | +0.118 | +0.380 | +131 |
| ood_test | `curriculum` | +0.118 | +0.394 | +138 |

## Decision

The clean-split full evaluation establishes scaffold-to-policy improvement for
every trained arm on dev, IID, and OOD. All arms clear the original full-run
justification gate by more than 5 absolute pass@1 points while improving,
rather than preserving, pass@32. `clean` is the best overall arm by IID/OOD
pass@1 and by dev/IID/OOD pass@32; `curriculum` is narrowly highest on dev
pass@1.

The next scientific step is a focused analysis of failure modes and matched
base-elicitable subsets, followed by a reasoning-first benchmark expansion with
the same hermetic rootfs and split/evaluation registry discipline.
