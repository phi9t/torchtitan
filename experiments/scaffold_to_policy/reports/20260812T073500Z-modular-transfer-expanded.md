# Modular Sequences Expanded Transfer

Date: 2026-08-12

## Scope

This report records the first larger `modular_sequences`
scaffold-to-policy transfer run. It follows the minimal transfer smoke but uses
enough dev and OOD examples to evaluate the local synthetic reasoning transfer
gate:

- held-out pass@1 movement;
- base-elicitable subset pass@1 movement;
- retention of base-elicitable pass@8/pass@32;
- representative wins, regressions, and unchanged failures.

This remains a local synthetic exact-verifier result. It is not a public
benchmark result.

## Command

The run used Qwen3-1.7B through the bwrap rootfs for generation, training,
adapter export, adapter evaluation, and report input creation:

```bash
RUN_ID=20260812T073500Z-modular-transfer-expanded \
DATA_ROOT=experiments/scaffold_to_policy/data/modular_sequences_transfer_expanded \
RESULTS_ROOT=experiments/scaffold_to_policy/results/modular_sequences_transfer_expanded \
TRAIN_PROBLEMS=64 \
DEV_PROBLEMS=32 \
OOD_PROBLEMS=32 \
NUM_ROLLOUTS=8 \
EVAL_ROLLOUTS=8 \
MAX_NEW_TOKENS=512 \
TRAIN_STEPS=24 \
CHECKPOINT_STEP=step-24 \
NGPU=1 \
MIN_TRAIN_EXAMPLES=20 \
LORA_RANK=16 \
LORA_ALPHA=32 \
experiments/scaffold_to_policy/run_modular_sequences_transfer_smoke.sh
```

The entrypoint re-executes itself through:

```bash
scripts/rootfs/enter_rootfs.sh -- experiments/scaffold_to_policy/run_modular_sequences_transfer_smoke.sh
```

## Artifacts

Generated artifacts are ignored by git and live under:

```text
experiments/scaffold_to_policy/data/modular_sequences_transfer_expanded/
experiments/scaffold_to_policy/results/modular_sequences_transfer_expanded/
```

Key artifacts:

```text
experiments/scaffold_to_policy/results/modular_sequences_transfer_expanded/eval/base/train_summary.json
experiments/scaffold_to_policy/results/modular_sequences_transfer_expanded/eval/base/dev_summary.json
experiments/scaffold_to_policy/results/modular_sequences_transfer_expanded/eval/base/ood_test_summary.json
experiments/scaffold_to_policy/results/modular_sequences_transfer_expanded/train/modular_sequences_raw/checkpoint/step-24/
experiments/scaffold_to_policy/results/modular_sequences_transfer_expanded/adapters/raw/export_summary.json
experiments/scaffold_to_policy/results/modular_sequences_transfer_expanded/eval/adapters/raw/dev_summary.json
experiments/scaffold_to_policy/results/modular_sequences_transfer_expanded/eval/adapters/raw/ood_test_summary.json
experiments/scaffold_to_policy/results/modular_sequences_transfer_expanded/manifests/report_input_20260812T073500Z-modular-transfer-expanded.json
```

Report-input checks:

| Check | Result |
| --- | --- |
| `split_registry_selected` | true |
| `summaries_present` | true |
| `summary_split_counts_match` | true |

The run-scoped stage manifest has eight fresh rootfs-active stages:
`generate_splits`, `collect_train`, `build_dataset`, `eval_base`,
`train_raw`, `export_raw`, `eval_adapter_raw`, and `build_report_input`.

## Training Scaffold

The train scaffold evaluated 64 problems with 8 rollouts each. It found 44
verifier-successful training examples, satisfying the `MIN_TRAIN_EXAMPLES=20`
gate.

| Split | Problems | Rollouts | pass@1 | pass@4 | pass@8 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| train | 64 | 512 | 0.328 | 0.609 | 0.688 | easy 21, elicitable 23, unreached 20 |

TorchTitan trained a rank-16, alpha-32 LoRA adapter for 24 steps on one B200.
The final checkpoint was:

```text
experiments/scaffold_to_policy/results/modular_sequences_transfer_expanded/train/modular_sequences_raw/checkpoint/step-24/
```

The adapter was exported to PEFT/vLLM format with 394 exported tensors across
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`,
and `lm_head`.

## Held-Out Results

| Arm | Split | Problems | Rollouts | pass@1 | pass@2 | pass@4 | pass@8 | pass@32 | Buckets |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | dev | 32 | 256 | 0.531 | 0.531 | 0.625 | 0.656 | 0.656 | easy 17, elicitable 4, unreached 11 |
| raw LoRA | dev | 32 | 256 | 0.625 | 0.688 | 0.781 | 0.781 | 0.781 | easy 20, elicitable 5, unreached 7 |
| base | ood_test | 32 | 256 | 0.375 | 0.469 | 0.531 | 0.688 | 0.688 | easy 12, elicitable 10, unreached 10 |
| raw LoRA | ood_test | 32 | 256 | 0.594 | 0.719 | 0.750 | 0.781 | 0.781 | easy 19, elicitable 6, unreached 7 |

Strict-format pass@k matched pass@k in this task because the verifier only
marks samples successful when they contain the exact strict final-answer line.

## Base-Elicitable Subsets

Base-elicitable means the base model solved a problem within the rollout budget
but not on the first sample. This is the direct search-compression target.

| Arm | Split | Problems | pass@1 | pass@8 | pass@32 |
| --- | --- | ---: | ---: | ---: | ---: |
| base | dev | 4 | 0.000 | 1.000 | 1.000 |
| raw LoRA | dev | 4 | 0.750 | 1.000 | 1.000 |
| base | ood_test | 10 | 0.000 | 1.000 | 1.000 |
| raw LoRA | ood_test | 10 | 0.800 | 1.000 | 1.000 |

This clears the local synthetic transfer gate: first-sample behavior improved
on held-out dev and OOD, and base-elicitable pass@8/pass@32 did not collapse.

## Failure Modes

The two dominant failure classes remained:

- missing `FINAL` lines, especially in verbose base generations;
- arithmetic slips in modular reduction.

The adapter reduced missing final lines materially:

| Split | Arm | Missing `FINAL` | Successes |
| --- | --- | ---: | ---: |
| dev | base | 96 | 126 |
| dev | raw LoRA | 30 | 171 |
| ood_test | base | 89 | 93 |
| ood_test | raw LoRA | not in top failures | 162 |

The adapter still makes arithmetic mistakes. Several regressions are strict
`FINAL` lines with wrong modular reductions rather than format failures.

## Examples

Dev win:

```text
x_0 = 119
For i = 1..4:
x_i = (4 * x_(i-1) + 3 * i + 111) mod 192
Answer: 107
```

The base first sample computed most of the trace but missed the final line and
only solved the problem on rollout 3. The adapter solved it on rollout 1:

```text
x_1 = 590 mod 192 = 14
x_2 = 173 mod 192 = 173
x_3 = 812 mod 192 = 44
x_4 = 299 mod 192 = 107

FINAL: 107
```

OOD win:

```text
x_0 = 67
For i = 1..4:
x_i = (10 * x_(i-1) + 11 * i + 95) mod 122
Answer: 23
```

The base solved this only on rollout 6. The adapter solved it on the first
sample:

```text
x_1 = 776 mod 122 = 44
x_2 = 557 mod 122 = 69
x_3 = 818 mod 122 = 86
x_4 = 999 mod 122 = 23

FINAL: 23
```

Dev regression:

```text
x_0 = 2
For i = 1..4:
x_i = (2 * x_(i-1) + 17 * i + 10) mod 120
Answer: 24
```

The base solved it on the first sample. The adapter initially made a modular
reduction error at `x_3`:

```text
x_3 = 273 mod 120 = 3
x_4 = 84 mod 120 = 84

FINAL: 84
```

It recovered on rollout 2, so this is a pass@1 regression but not a pass@8
collapse.

OOD unchanged failure:

```text
x_0 = 27
For i = 1..4:
x_i = (14 * x_(i-1) + 3 * i + 67) mod 80
Answer: 43
```

Both base and adapter produced the same wrong trajectory ending in:

```text
x_4 = 1003 mod 80 = 3

FINAL: 3
```

The verifier correctly rejected both because the final answer is 43.

## Interpretation

This run provides the first positive reasoning-lane transfer result outside
Countdown. The adapter improves both first-sample held-out accuracy and
base-elicitable search compression on a second exact-verifier task.

The result is still limited:

- it uses one synthetic task family;
- it uses one split draw;
- dev has only 4 base-elicitable problems;
- training uses only the `raw` arm;
- public benchmark normalization has not yet been exercised.

The next spec stage should add GSM-style final-answer normalization with a
fixture-backed smoke, then run a small public or public-shaped GSM evaluation
only if the no-tool exact-verifier semantics are preserved.
