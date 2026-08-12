# Countdown Formatting Arm Results And Audit

Date: 2026-08-12

## Executive Summary

The full Countdown scaffold-to-policy matrix now includes the `formatting` arm.
The arm was trained under the bwrap rootfs on the clean full train split,
exported to a PEFT/vLLM LoRA directory, evaluated on dev/IID/OOD with 32
rollouts per problem, and folded into the full five-arm adapter matrix.

The formatting arm changes the result profile. It is the strongest pass@1 arm
on all three splits and is the only arm that nearly closes the strict
`FINAL: <target>` compliance gap. On OOD it improves base pass@1 from 0.024 to
0.294 and strict-format pass@1 from 0.004 to 0.276. Its OOD pass@32 is 0.902,
slightly above the previous `clean` champion at 0.890.

The main caveat is qualitative: the formatting arm often emits repeated answer
phrases after a valid trace. The verifier accepts the first valid trace plus
strict final line, but the policy is overfit to answer-shape repetition. The
next improvement should optimize concision and stop behavior, not only
arithmetic correctness.

## Run And Provenance

All real Python, training, export, evaluation, and summarization work used the
TorchTitan rootfs entrypoint:

```bash
scripts/rootfs/enter_rootfs.sh -- <command>
```

Fresh continuation artifacts:

| Artifact | Path |
| --- | --- |
| Formatting train data | `experiments/countdown_search_distill/data/train/formatting.jsonl` |
| Formatting checkpoint | `experiments/countdown_search_distill/results/train/full/formatting/checkpoint/step-94/` |
| Exported adapter | `experiments/countdown_search_distill/results/adapters/full/formatting/` |
| Dev eval summary | `experiments/countdown_search_distill/results/eval/adapters/full/dev/formatting/summary.json` |
| IID eval summary | `experiments/countdown_search_distill/results/eval/adapters/full/iid_test/formatting/summary.json` |
| OOD eval summary | `experiments/countdown_search_distill/results/eval/adapters/full/ood_test/formatting/summary.json` |
| Full adapter matrix | `experiments/countdown_search_distill/results/eval/adapters/full/adapter_matrix_full.json` |
| Continuation manifest | `experiments/countdown_search_distill/results/manifests/20260812T001300Z-full-formatting-continuation.jsonl` |
| Report input | `experiments/countdown_search_distill/results/manifests/report_input_20260812T001300Z-full-formatting-continuation.json` |

Registry checks from the final report input:

| Check | Result |
| --- | --- |
| `required_manifest_stages_succeeded` | true |
| `split_registry_selected` | true |
| `split_registry_no_overlap` | true |
| `base_summaries_present` | true |
| `adapter_matrix_selected` | true |

The full adapter matrix now has 15 rows: three splits by five arms.

## Training And Export

Formatting data was generated from matched clean train evaluations. It contains
947 rows. Each target answer is a compact parsed operation trace ending in exact
`FINAL: <target>`.

Example training row:

```text
94 - 59 = 35
6 + 35 = 41
FINAL: 41
```

Training command:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'MODE=full ARM=formatting NGPU=8 \
   TRAIN_RESULT_ROOT=experiments/countdown_search_distill/results/train/full \
   experiments/countdown_search_distill/run_train.sh'
```

Training completed at step 94. Loss dropped from about 2.74 at step 1 to 0.02197
at step 94. The final checkpoint was saved at:

```text
experiments/countdown_search_distill/results/train/full/formatting/checkpoint/step-94/
```

Export command:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'MODE=full ARMS=formatting \
   TRAIN_RESULT_ROOT=experiments/countdown_search_distill/results/train/full \
   ADAPTER_RESULT_ROOT=experiments/countdown_search_distill/results/adapters/full \
   CHECKPOINT_STEP=step-94 \
   experiments/countdown_search_distill/run_export_adapters.sh'
```

The export wrote `adapter_config.json`, `adapter_model.safetensors`, and
`export_summary.json`. Export summary: rank 32, alpha 64, 394 exported tensors,
target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj`, and `lm_head`.

## Results

Base Qwen3-1.7B:

| Split | pass@1 | pass@32 | strict pass@1 | strict pass@32 |
| --- | ---: | ---: | ---: | ---: |
| dev | 0.026 | 0.468 | 0.008 | 0.148 |
| iid_test | 0.033 | 0.509 | 0.008 | 0.167 |
| ood_test | 0.024 | 0.484 | 0.004 | 0.166 |

Full adapter matrix:

| Split | Arm | pass@1 | pass@32 | strict pass@1 | strict pass@32 |
| --- | --- | ---: | ---: | ---: | ---: |
| dev | raw | 0.162 | 0.876 | 0.086 | 0.716 |
| dev | clean | 0.180 | 0.904 | 0.080 | 0.758 |
| dev | formatting | 0.374 | 0.900 | 0.370 | 0.900 |
| dev | hindsight | 0.162 | 0.880 | 0.084 | 0.716 |
| dev | curriculum | 0.182 | 0.890 | 0.088 | 0.724 |
| iid_test | raw | 0.169 | 0.915 | 0.069 | 0.771 |
| iid_test | clean | 0.208 | 0.922 | 0.108 | 0.790 |
| iid_test | formatting | 0.361 | 0.908 | 0.351 | 0.907 |
| iid_test | hindsight | 0.152 | 0.890 | 0.074 | 0.745 |
| iid_test | curriculum | 0.163 | 0.921 | 0.077 | 0.809 |
| ood_test | raw | 0.126 | 0.870 | 0.052 | 0.626 |
| ood_test | clean | 0.180 | 0.890 | 0.076 | 0.658 |
| ood_test | formatting | 0.294 | 0.902 | 0.276 | 0.900 |
| ood_test | hindsight | 0.142 | 0.864 | 0.090 | 0.598 |
| ood_test | curriculum | 0.142 | 0.878 | 0.072 | 0.640 |

OOD deltas versus base:

| Arm | delta pass@1 | delta pass@32 | delta strict pass@1 | delta strict pass@32 |
| --- | ---: | ---: | ---: | ---: |
| raw | +0.102 | +0.386 | +0.048 | +0.460 |
| clean | +0.156 | +0.406 | +0.072 | +0.492 |
| formatting | +0.270 | +0.418 | +0.272 | +0.734 |
| hindsight | +0.118 | +0.380 | +0.086 | +0.432 |
| curriculum | +0.118 | +0.394 | +0.068 | +0.474 |

Formatting OOD bucket counts:

| easy | elicitable | unreached |
| ---: | ---: | ---: |
| 147 | 304 | 49 |

This is a large movement from base OOD counts of 12 easy, 230 elicitable, and
258 unreached.

## Format Breakdown

Strict format means an arithmetic success also includes a line exactly equal to
`FINAL: <target>`.

Formatting arm rollout-level breakdown:

| Split | success with strict final | success missing strict final | failure with strict final | failure missing strict final |
| --- | ---: | ---: | ---: | ---: |
| dev | 5909 | 143 | 1921 | 8027 |
| iid_test | 11582 | 368 | 3683 | 16367 |
| ood_test | 4579 | 212 | 1904 | 9305 |

The key change is not just higher pass@1. For OOD, only 212 successful
formatting rollouts were missing a strict final line, compared with 1659 for
`clean` and 1442 for `raw`.

## Examples

### Strict Formatting Win

Problem:

```text
numbers: 54, 85, 75
target: 64
canonical solution:
85 - 75 = 10
10 + 54 = 64
FINAL: 64
```

Base sample 0 failed with `missing FINAL line` after free-form reasoning.

Formatting sample 0 succeeded:

```text
The final answer is 64.
54 + 85 = 139
139 - 75 = 64
FINAL: 64
The final answer is 64.
The final answer is 64.
...
```

This illustrates both the win and the caveat. The first valid strict trace is
correct, but the generation continues with repeated answer text.

### Output Contract Repair

Problem:

```text
numbers: 16, 7, 31
target: 54
canonical solution:
31 + 7 = 38
38 + 16 = 54
FINAL: 54
```

`clean` sample 0 was arithmetically successful but did not emit a strict final
line before drifting into prompt-like text:

```text
Example:
16 + 7 = 23
23 + 31 = 54
```

Formatting sample 0 emitted strict final lines:

```text
The answer is 54.
16 + 31 = 47
47 + 7 = 54
FINAL: 54
The answer is 54.
16 + 7 = 23
31 + 23 = 54
FINAL: 54
...
```

The behavior is less concise than desired, but it satisfies the strict contract
early in the sample.

### Regression Caveat

Problem:

```text
numbers: 49, 35, 82
target: 68
canonical solution:
82 - 49 = 33
33 + 35 = 68
FINAL: 68
```

`clean` sample 0 solved with the expected trace:

```text
82 - 49 = 33
33 + 35 = 68
FINAL: 68
```

Formatting sample 0 failed by repeating:

```text
The answer is 68
The answer is 68
The answer is 68
...
```

This is why the next formatting iteration should add stop/length shaping or
negative examples for answer-only repetition.

## Infrastructure Review

The rootfs path is now doing the right job. TorchTitan training, vLLM
generation, adapter export, summary refresh, matrix validation, and report
input generation all ran in the bwrap environment. This kept host Python out of
the evidence path and gave one package/runtime surface for both training and
inference.

The TorchTitan to vLLM boundary is explicit. Internal DCP checkpoints stay in:

```text
experiments/countdown_search_distill/results/train/full/<arm>/checkpoint/step-94/
```

vLLM receives only exported PEFT adapter directories:

```text
experiments/countdown_search_distill/results/adapters/full/<arm>/
```

The registry layer is now useful enough for audit:

- run manifests record canonical stages and fresh formatting continuation work;
- split registry selection and no-overlap checks are required;
- adapter matrix validation checks expected problem counts, rollout counts,
  pass@k curve coverage, and bucket totals;
- report input records artifact paths and hashes for small artifacts;
- strict-format metrics and format breakdowns are first-class in summaries,
  adapter matrix rows, and report-input compact rows.

Two infrastructure issues remain:

- The continuation manifest had to be assembled after a direct training launch
  from the previous session. Future long runs should always enter through a
  manifest stage wrapper, even for one-arm continuation work.
- `ARMS=<list>` overrides were added to export/eval wrappers during this run.
  That makes scoped reruns practical, but a more general manifest-driven runner
  would be cleaner for repeated ablations.

## Agentic Execution Trace Review

What worked well:

- The process waited for the active training PTY to finish instead of launching
  a competing job.
- The final checkpoint was exported before adapter eval, preserving the
  TorchTitan/vLLM boundary.
- Evaluation was scoped to the missing formatting arm first, then the full
  five-arm matrix was regenerated explicitly.
- Summary refresh reused existing rollout/evaluation artifacts instead of
  spending GPU time to regenerate old arms.
- The reporting code was patched when it became clear that strict-format fields
  existed in summaries but were missing from matrix/report compact rows.

What needs to improve:

- One-arm continuation should have been started through `countdown_run_stage`
  so the manifest row existed naturally.
- The first `ARMS=formatting` export attempt exposed a shell bug: the default
  array assignment overwrote the scalar environment override. The wrapper now
  captures `REQUESTED_ARMS` first, but this should have had a test earlier.
- The formatting arm was allowed to optimize strict final-line presence without
  a stop-quality metric. That created repeated answer text despite strong
  verifier metrics.

## Completion Audit

| Spec objective | Evidence | Status |
| --- | --- | --- |
| Add strict output-format compliance metrics | `strict_format_pass_at_k` and `format_breakdown` in summaries, matrix rows, and report input | complete |
| Add a formatting arm | `formatting` dataset builder, Qwen3 config, runner arm mapping, export/eval matrix support | complete |
| Train formatting on full mode | `results/train/full/formatting/checkpoint/step-94/` | complete |
| Export formatting before vLLM eval | `results/adapters/full/formatting/{adapter_config.json,adapter_model.safetensors,export_summary.json}` | complete |
| Evaluate dev/IID/OOD at 32 rollouts | formatting summaries for dev, IID, and OOD under `results/eval/adapters/full/` | complete |
| Validate five-arm adapter matrix | `adapter_matrix_full.json`, selected true, 15 rows | complete |
| Generate audited report input | `report_input_20260812T001300Z-full-formatting-continuation.json`, all checks true | complete |
| Keep real Python/GPU work inside rootfs | all commands above ran via `scripts/rootfs/enter_rootfs.sh -- ...` | complete |
| Keep generated data/results out of git | result and data paths remain ignored by `.gitignore` | complete |

## Decision

The formatting arm is worth keeping and promoting into the next Countdown
iteration. It is no longer just an output-contract ablation: it substantially
improves first-sample arithmetic success while also fixing strict final-line
compliance. The previous `clean` arm remains a better reference for concise
behavior, but it is no longer the best overall result.

Next experiments should treat `formatting` as the new performance champion and
`clean` as the style/conciseness control. The next training target should add a
stop-quality constraint: short parsed traces, one strict final line, and no
post-final repetition.
