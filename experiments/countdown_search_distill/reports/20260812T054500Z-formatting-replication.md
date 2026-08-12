# Countdown Formatting Replication

Date: 2026-08-12

## Scope

This report records the isolated replication of the Countdown `formatting`
champion. The run used the same seed/split draw and cost point as the strongest
clean-arm sweep cell, but trained and evaluated only the `formatting` arm:

```bash
SWEEP_GROUP=formatting_replication \
LABEL_PREFIX=formatting_ \
RUN_ID_PREFIX=20260812T-formatting-rep \
SEEDS=43 \
TRAIN_SIZES=2000 \
LORA_RANKS=16 \
ARMS=formatting \
NGPU=8 \
experiments/countdown_search_distill/run_replication_sweep.sh
```

The command was launched through the bwrap rootfs:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc '<command>'
```

All generation, split validation, TorchTitan training, PEFT adapter export,
vLLM base evaluation, vLLM LoRA evaluation, matrix validation, and report-input
generation stayed inside the rootfs. The final command exited with code 0 and
wrote:

```text
experiments/countdown_search_distill/sweeps/formatting_replication/replication_sweep_full.jsonl
```

## Artifacts

The run root is:

```text
experiments/countdown_search_distill/sweeps/formatting_replication/formatting_seed43_train2000_rank16/
```

Key artifacts:

| Artifact | Path |
| --- | --- |
| Runtime preflight | `results/runtime_preflight.json` |
| Final TorchTitan checkpoint | `results/train/full/formatting/checkpoint/step-94/` |
| Exported PEFT/vLLM adapter | `results/adapters/full/formatting/` |
| Adapter export summary | `results/adapters/full/formatting/export_summary.json` |
| Adapter matrix | `results/eval/adapters/full/adapter_matrix_full.json` |
| Report input | `results/manifests/report_input_20260812T-formatting-rep-formatting_seed43_train2000_rank16.json` |

The report input checks all passed:

| Check | Result |
| --- | --- |
| `required_manifest_stages_succeeded` | true |
| `split_registry_selected` | true |
| `split_registry_no_overlap` | true |
| `base_summaries_present` | true |
| `adapter_matrix_selected` | true |

The adapter export summary records rank 16, alpha 64, 282 source tensors, and
394 exported tensors. vLLM evaluation used the exported PEFT adapter directory;
the TorchTitan internal checkpoint was not passed directly to vLLM.

## Aggregate Results

The base model and formatting adapter were evaluated with 32 rollouts per
problem on 500 dev, 1000 IID, and 500 OOD problems.

| Split | Model | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | base | 0.024 | 0.052 | 0.102 | 0.190 | 0.298 | 0.470 |
| dev | formatting | 0.278 | 0.464 | 0.648 | 0.804 | 0.878 | 0.916 |
| iid_test | base | 0.027 | 0.065 | 0.121 | 0.212 | 0.336 | 0.493 |
| iid_test | formatting | 0.293 | 0.458 | 0.663 | 0.797 | 0.873 | 0.911 |
| ood_test | base | 0.056 | 0.108 | 0.206 | 0.362 | 0.502 | 0.616 |
| ood_test | formatting | 0.302 | 0.520 | 0.704 | 0.818 | 0.878 | 0.914 |

Strict final-line results:

| Split | Model | strict pass@1 | strict pass@2 | strict pass@4 | strict pass@8 | strict pass@16 | strict pass@32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | base | 0.006 | 0.008 | 0.016 | 0.032 | 0.070 | 0.136 |
| dev | formatting | 0.268 | 0.454 | 0.636 | 0.794 | 0.874 | 0.914 |
| iid_test | base | 0.008 | 0.021 | 0.032 | 0.057 | 0.113 | 0.184 |
| iid_test | formatting | 0.287 | 0.447 | 0.651 | 0.789 | 0.865 | 0.907 |
| ood_test | base | 0.016 | 0.032 | 0.070 | 0.130 | 0.216 | 0.346 |
| ood_test | formatting | 0.302 | 0.518 | 0.702 | 0.812 | 0.876 | 0.914 |

The formatting replication improves dev pass@1 by 25.4 points over base while
keeping dev pass@32 44.6 points above base. It also preserves the desired
search-compression profile: first-sample behavior improves strongly and
residual pass@32 remains high.

## Base-Elicitable Subsets

Base-elicitable subsets are the problems where base Qwen3-1.7B solves the
problem within 32 samples. These are the direct scaffold-compression targets.

| Split | Model | Problems | pass@1 | pass@32 | strict pass@1 | strict pass@32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dev | base | 223 | 0.000 | 1.000 | 0.000 | 0.283 |
| dev | formatting | 223 | 0.323 | 0.996 | 0.305 | 0.991 |
| iid_test | base | 466 | 0.000 | 1.000 | 0.000 | 0.365 |
| iid_test | formatting | 466 | 0.378 | 0.985 | 0.371 | 0.981 |
| ood_test | base | 280 | 0.000 | 1.000 | 0.000 | 0.554 |
| ood_test | formatting | 280 | 0.382 | 0.993 | 0.382 | 0.993 |

On OOD base-elicitable problems, the formatting adapter compresses 38.2% of
base-scaffold-reachable problems into first-sample strict solutions while
retaining 99.3% strict pass@32.

## Bucket Movement

| Split | Model | Easy | Elicitable | Unreached |
| --- | --- | ---: | ---: | ---: |
| dev | base | 12 | 223 | 265 |
| dev | formatting | 139 | 319 | 42 |
| iid_test | base | 27 | 466 | 507 |
| iid_test | formatting | 293 | 618 | 89 |
| ood_test | base | 28 | 280 | 192 |
| ood_test | formatting | 151 | 306 | 43 |

The adapter converts many base-elicitable problems into easy problems and
substantially reduces unreached counts across all splits.

## Failure Modes

The strict-format objective worked, but did not eliminate every known quality
issue.

| Split | Model | Success with strict final | Success missing strict final | Failure with strict final | Failure missing strict final |
| --- | --- | ---: | ---: | ---: | ---: |
| dev | base | 75 | 318 | 664 | 14943 |
| dev | formatting | 4217 | 128 | 2713 | 8942 |
| iid_test | base | 240 | 698 | 1354 | 29708 |
| iid_test | formatting | 9152 | 267 | 4976 | 17605 |
| ood_test | base | 279 | 758 | 527 | 14436 |
| ood_test | formatting | 5515 | 129 | 2485 | 7871 |

The remaining qualitative issue is post-answer repetition. Representative wins
often contain a valid strict trace early, followed by repeated answer text or
fence tokens. This usually does not harm exact verification because the first
valid `FINAL:` line is present, but it is undesirable for policy quality and
should be addressed with stop-quality shaping or shorter answer targets.

Example win from dev:

```text
Problem: numbers 1, 13, 78 target 66
Base: missing FINAL line after a long search attempt.
Formatting:
78 - 13 = 65
65 + 1 = 66
FINAL: 66
```

Example regression from dev:

```text
Problem: numbers 5, 33, 27 target 65
Base first sample solved arithmetically.
Formatting emitted a valid early trace but then repeated answer text; the
selected sampled annotation classified the first inspected rollout as missing
strict final after continuation noise.
```

Example unchanged failure from dev:

```text
Problem: numbers 1, 26, 51 target 25
Canonical solution:
51 - 26 = 25
25 * 1 = 25
FINAL: 25
Formatting attempted an unavailable-number trace and failed verification.
```

## Decision

The formatting strategy has now replicated under an isolated seed-43,
train-2000, rank-16 run. It remains the Countdown performance champion for
pass@1 and strict pass@1, and the strict-output gain is no longer only a
single original clean-split result.

Countdown has enough replicated evidence to support the next reasoning-lane
step: run local exact-verifier transfer tasks with the same rootfs, split
registry, report-input, and strict-format reporting discipline. The next
Countdown-specific improvement should target stop-quality shaping for
post-answer repetition rather than another immediate rank sweep.
