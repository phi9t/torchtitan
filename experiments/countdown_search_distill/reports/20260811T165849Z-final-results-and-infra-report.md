# Countdown Final Results And Experiment Infrastructure Report

Date: 2026-08-11

## Executive Summary

The Countdown scaffold-to-policy experiment completed end to end under the
TorchTitan bwrap rootfs. The path covered runtime repair, Qwen3-1.7B asset
download, smoke, reduced pilot, full pilot, TorchTitan LoRA training, LoRA
export to PEFT/vLLM format, and adapter evaluation with vLLM.

The strongest scientifically reliable result is on the OOD target-range split,
which has no overlap with train by problem ID or number-target key. On OOD,
every trained arm improves substantially over the frozen Qwen3-1.7B base model:

| Arm | OOD pass@1 | OOD pass@32 | pass@1 delta | pass@32 delta |
| --- | ---: | ---: | ---: | ---: |
| base | 0.024 | 0.484 | - | - |
| raw | 0.136 | 0.884 | +0.112 | +0.400 |
| clean | 0.168 | 0.902 | +0.144 | +0.418 |
| hindsight | 0.136 | 0.864 | +0.112 | +0.380 |
| curriculum | 0.124 | 0.874 | +0.100 | +0.390 |

`clean` is the best current arm by OOD pass@1 and pass@32. It is also best by
IID pass@1/pass@32, but IID is not held out in this run.

Important validity caveat: the generated dev and IID splits exactly overlap
with train because `run_collect.sh` used the same seed for train, dev, and IID
generation under the same problem regime. Those numbers demonstrate training
path, export path, and capacity, but they should not be presented as clean
held-out generalization evidence. OOD is the reliable held-out split from this
run.

## Scientific Question

The experiment asks whether best-of-N Countdown behavior from a frozen Qwen3-1.7B
base model can be compressed into one-shot policy behavior using TorchTitan LoRA
SFT.

The intended evidence chain is:

1. Select a Countdown regime where base pass@1 is low but pass@32 is meaningfully
   higher.
2. Collect best-of-32 rollouts from the base model and exactly verify each
   rollout.
3. Train LoRA adapters from verified successful traces using several target
   constructions.
4. Export TorchTitan LoRA checkpoints to PEFT/vLLM adapter directories.
5. Evaluate base and adapters with the same verifier and pass@k metrics.
6. Compare pass@1 lift while monitoring pass@32, bucket movement, and split
   behavior.

## Runtime And Infrastructure

All real Python setup, generation, training, export, evaluation, validation, and
summarization ran through the repo-local bwrap rootfs entrypoint:

```bash
scripts/rootfs/enter_rootfs.sh -- <command>
```

The rootfs isolates the experiment from host Python state while bind-mounting the
workspace and NVIDIA device nodes. It sets rootfs-specific CUDA paths and exposes
the local GPUs inside the sandbox. The relevant contract is now enforced by:

```bash
experiments/countdown_search_distill/run_preflight.sh
```

The runtime preflight writes:

```text
experiments/countdown_search_distill/results/runtime_preflight.json
```

Observed preflight state:

| Check | Result |
| --- | --- |
| `TORCHTITAN_IN_ROOTFS=1` | true |
| `torch` import | true |
| `vllm` import | true |
| `datasets` import | true |
| `transformers` import | true |
| `spmd_types` import | true |
| CUDA available | true |
| CUDA device count positive | true |
| Qwen3 config/tokenizer/safetensors present | true |
| FlashInfer sampler disabled | true |
| FlashInfer autotune disabled | true |

vLLM is used for frozen-base and adapter generation. The experiment defaults use
the safer B200/rootfs path:

- `COUNTDOWN_VLLM_ATTENTION_BACKEND=TRITON_ATTN` by default.
- `COUNTDOWN_VLLM_FLASHINFER_AUTOTUNE=0` by default.
- `COUNTDOWN_VLLM_USE_FLASHINFER_SAMPLER=0` by default.
- `max_lora_rank=32` when LoRA is enabled.

TorchTitan is used for LoRA SFT. The Qwen3 configs live in:

```text
torchtitan/models/qwen3/config_registry.py
```

The trained LoRA checkpoints are TorchTitan DCP checkpoints. They are not passed
directly to vLLM. Instead:

```bash
MODE=full experiments/countdown_search_distill/run_export_adapters.sh
```

exports them to PEFT/vLLM adapter directories:

```text
experiments/countdown_search_distill/results/adapters/full/{raw,clean,hindsight,curriculum}/
```

Each exported adapter contains:

- `adapter_config.json`
- `adapter_model.safetensors`
- `export_summary.json`

The full adapters all reported:

| Field | Value |
| --- | ---: |
| source LoRA tensor count | 282 |
| exported PEFT tensor count | 394 |
| rank | 32 |
| alpha | 64 |

Adapter evaluation is now a first-class resumable stage:

```bash
MODE=full experiments/countdown_search_distill/run_eval_adapters.sh
```

It skips existing valid summaries by default and validates the matrix with:

```text
experiments/countdown_search_distill/results/eval/adapter_matrix_full.json
```

The full matrix validation passed for 12 adapter summaries:

- 3 splits: `dev`, `iid_test`, `ood_test`
- 4 arms: `raw`, `clean`, `hindsight`, `curriculum`
- expected split sizes: 500, 1000, 500
- expected rollout count: 32 per problem
- pass@k curve keys present
- bucket counts sum to split size

## Experiment Arms

The full run trained four LoRA arms:

| Arm | Training target |
| --- | --- |
| `raw` | shortest verified base-model success trace |
| `clean` | deterministic cleaned rewrite of a verified success |
| `hindsight` | hint-conditioned prompt plus verified answer |
| `curriculum` | mixture of hint-present, hint-dropout, and hint-absent rows |

Final full checkpoints:

| Arm | Checkpoint |
| --- | --- |
| raw | `results/train/full/raw/checkpoint/step-94/` |
| clean | `results/train/full/clean/checkpoint/step-94/` |
| hindsight | `results/train/full/hindsight/checkpoint/step-94/` |
| curriculum | `results/train/full/curriculum/checkpoint/step-94/` |

## Full Metrics

Base model:

| Split | Problems | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | easy | elicitable | unreached |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 500 | 0.018 | 0.038 | 0.088 | 0.174 | 0.278 | 0.474 | 9 | 228 | 263 |
| iid_test | 1000 | 0.018 | 0.047 | 0.088 | 0.161 | 0.272 | 0.466 | 18 | 448 | 534 |
| ood_test | 500 | 0.024 | 0.052 | 0.092 | 0.194 | 0.308 | 0.484 | 12 | 230 | 258 |

Adapter metrics:

| Split | Arm | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | easy | elicitable | unreached |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | raw | 0.208 | 0.326 | 0.494 | 0.714 | 0.838 | 0.896 | 104 | 344 | 52 |
| dev | clean | 0.182 | 0.330 | 0.546 | 0.736 | 0.862 | 0.916 | 91 | 367 | 42 |
| dev | hindsight | 0.126 | 0.244 | 0.424 | 0.590 | 0.778 | 0.894 | 63 | 384 | 53 |
| dev | curriculum | 0.178 | 0.302 | 0.480 | 0.676 | 0.836 | 0.896 | 89 | 359 | 52 |
| iid_test | raw | 0.169 | 0.319 | 0.505 | 0.695 | 0.839 | 0.904 | 169 | 735 | 96 |
| iid_test | clean | 0.181 | 0.340 | 0.533 | 0.748 | 0.857 | 0.914 | 181 | 733 | 86 |
| iid_test | hindsight | 0.141 | 0.244 | 0.419 | 0.621 | 0.793 | 0.882 | 141 | 741 | 118 |
| iid_test | curriculum | 0.156 | 0.286 | 0.463 | 0.668 | 0.824 | 0.896 | 156 | 740 | 104 |
| ood_test | raw | 0.136 | 0.252 | 0.430 | 0.610 | 0.792 | 0.884 | 68 | 374 | 58 |
| ood_test | clean | 0.168 | 0.310 | 0.490 | 0.678 | 0.810 | 0.902 | 84 | 367 | 49 |
| ood_test | hindsight | 0.136 | 0.240 | 0.402 | 0.572 | 0.756 | 0.864 | 68 | 364 | 68 |
| ood_test | curriculum | 0.124 | 0.238 | 0.394 | 0.596 | 0.784 | 0.874 | 62 | 375 | 63 |

## Interpretation

The OOD result supports the central hypothesis: verified best-of-N behavior can
be compressed into much stronger one-shot behavior with LoRA SFT. The frozen
base model solves only 2.4% of OOD problems on the first rollout and 48.4% with
32 rollouts. The best adapter, `clean`, solves 16.8% on the first rollout and
90.2% with 32 rollouts.

The pass@32 gains are also meaningful. They show the adapters are not merely
reshuffling search mass into the first sample while losing total reachable
coverage. Instead, the adapters move many previously unreached problems into
the easy or elicitable buckets.

On OOD:

| Arm | easy | elicitable | unreached |
| --- | ---: | ---: | ---: |
| base | 12 | 230 | 258 |
| raw | 68 | 374 | 58 |
| clean | 84 | 367 | 49 |
| hindsight | 68 | 364 | 68 |
| curriculum | 62 | 375 | 63 |

`clean` has the best OOD pass@1, pass@32, and lowest unreached count. It is the
strongest candidate for follow-up.

## Validity Caveat: Dev/IID Split Overlap

The dev and IID splits are not held out in the current artifacts. Direct
inspection shows:

| Split pair | Problem ID overlap | Number-target overlap |
| --- | ---: | ---: |
| train-dev | 500 | 500 |
| train-iid_test | 1000 | 1000 |
| dev-iid_test | 500 | 500 |
| train-ood_test | 0 | 0 |
| dev-ood_test | 0 | 0 |
| iid_test-ood_test | 0 | 0 |

Cause: `run_collect.sh` used the same default seed for train, dev, and IID under
the same regime. OOD uses a disjoint target range, so it does not overlap.

Consequence:

- Dev/IID results are useful for verifying the full stack and measuring
  train-regime capacity.
- Dev/IID results are not clean held-out generalization evidence.
- OOD is the reliable held-out evidence from this run.
- The next run should use split-specific seeds or global de-duplication.

The base-elicitable dev gate still holds numerically on the 228 base-elicitable
dev problems:

| Arm | Base-elicitable dev pass@1 | Base-elicitable dev pass@32 |
| --- | ---: | ---: |
| raw | 0.276 | 0.978 |
| clean | 0.219 | 0.991 |
| hindsight | 0.162 | 0.969 |
| curriculum | 0.237 | 0.982 |

But because dev overlaps train, that gate should be rerun on a clean held-out
dev split before being used as scientific justification.

## Representative Examples

Examples below come from OOD, the non-overlapping split.

### Example 1: Base Unreached, Clean Solves On First Rollout

Problem:

```text
Numbers: 71, 39, 85
Target: 53
```

Base first rollout failed with `missing FINAL line` and did not solve this
problem in 32 attempts.

Clean adapter first rollout:

```text
71 - 39 = 32
85 - 32 = 53
```

Verifier status: success. This is the desired scaffold-to-policy behavior:
base search failed in the sampled budget, while the adapter produced a valid
trace immediately.

### Example 2: Base Unreached, Raw And Clean Both Find A Simple Difference

Problem:

```text
Numbers: 40, 47, 6
Target: 81
```

Base first rollout started with the right intermediate value but failed verifier
state tracking:

```text
40 + 47 = 87
87 + 6 = 93
...
```

Verifier error: `operation consumes unavailable numbers: 40 + 47 = 87`.

Raw adapter success:

```text
40 + 47 = 87
87 - 6 = 81
FINAL: 81
```

Clean adapter success:

```text
40 + 47 = 87
87 - 6 = 81
```

Verifier status: success.

### Example 3: Base Unreached, Clean Produces A Direct Two-Step Solution

Problem:

```text
Numbers: 99, 100, 67
Target: 66
```

Base first rollout failed with `missing FINAL line` and did not solve in 32
attempts.

Clean adapter first rollout:

```text
100 - 67 = 33
99 - 33 = 66
```

Verifier status: success.

### Example 4: Base Unreached, Clean Solves With Add-Then-Subtract

Problem:

```text
Numbers: 69, 95, 93
Target: 71
```

Clean adapter first rollout:

```text
69 + 95 = 164
164 - 93 = 71
```

Verifier status: success. This is representative of many OOD gains: the adapter
is not just copying a single arithmetic template, but frequently finds valid
two-step combinations across the shifted target range.

### Example 5: Regression From Base pass@1

Problem:

```text
Numbers: 59, 80, 73
Target: 52
```

Base first rollout solved:

```text
80 - 59 = 21
73 - 21 = 52
```

Several adapters reasoned toward the same solution but failed the strict output
format by omitting a final line. Raw adapter first rollout:

```text
80 - 59 = 21
73 - 21 = 52
```

Verifier error: `missing FINAL line`.

This illustrates the remaining failure mode: adapters often discover valid
arithmetic but still lose credit on output-format compliance. Future training or
decoding constraints should target final-answer formatting explicitly.

### Example 6: Regression With Verifier-State Confusion

Problem:

```text
Numbers: 96, 62, 28
Target: 62
```

Base first rollout solved by using the target number as an intermediate target:

```text
96 + 28 = 124
124 - 62 = 62
Final: 62
```

Clean adapter first rollout failed verifier state tracking. It mixed example
text and operations, then attempted an invalid operation sequence:

```text
96 - 62 = 34
96 - 28 = 68
...
```

Verifier error: `operation consumes unavailable numbers: 96 - 28 = 68`.

This suggests the adapters still sometimes generate explanatory/example text
that contaminates the operation trace.

## Failure Modes

The most common qualitative failure modes visible in sampled examples are:

- Missing `FINAL` line despite correct arithmetic appearing in the text.
- Reusing unavailable numbers after producing an intermediate.
- Including example text or repeated code fences before/after the valid trace.
- Misstating the problem rules in prose, especially around whether numbers may
  be reused.
- Correct intermediate discovery followed by a malformed final answer.

The exact verifier is doing useful work here: many superficially plausible
outputs are rejected for state-trace violations that would be easy to miss by
manual inspection.

## Experiment Process Improvements Implemented

The run exposed several infrastructure issues, which are now encoded into the
experiment surface:

| Improvement | Artifact |
| --- | --- |
| Rootfs runtime preflight | `run_preflight.sh`, `preflight-runtime` |
| Mode-scoped full/reduced checkpoints | `run_full_pilot.sh`, `TRAIN_RESULT_ROOT=results/train/{mode}` |
| PEFT/vLLM LoRA export | `lora_export.py`, `run_export_adapters.sh` |
| Adapter eval matrix runner | `run_eval_adapters.sh` |
| Matrix validator | `validate-eval-matrix` |
| JSONL stage manifest | `run_common.sh`, `TORCHTITAN_COUNTDOWN_MANIFEST` |
| Completion audit | `reports/20260811T121857Z-completion-audit.md` |

Final validation command:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'bash -n experiments/countdown_search_distill/run_*.sh && python -m pytest tests/unit_tests/test_countdown_search_distill.py -q'
```

Result:

```text
30 passed, 14 warnings
```

Additional rootfs checks also passed:

- runtime preflight against real rootfs/assets
- full adapter matrix validation
- export/eval wrapper skip-mode
- manifest smoke check

## Conclusions

The completed run is a real end-to-end success for the infrastructure and a
promising scientific result on OOD. The reliable claim is:

> On a non-overlapping OOD target-range split, TorchTitan LoRA SFT from verified
> Countdown traces substantially improves Qwen3-1.7B pass@1 and pass@32, with
> the `clean` target construction performing best.

The overclaim to avoid is:

> Dev and IID establish held-out generalization.

They do not in this run because they overlap train exactly.

## Recommended Next Experiments

1. Regenerate train/dev/IID with split-specific seeds and global de-duplication.
2. Rerun base plus adapter eval on clean dev/IID using the already exported full
   adapters.
3. Add a formatting-focused training or decoding variant to reduce missing
   `FINAL` failures.
4. Run at least 2-3 seeds for the best arms (`clean`, `raw`) after split hygiene
   is fixed.
5. Add a failure-mode table from annotations: missing final, unavailable
   operand, wrong final, non-exact division, and success.
6. Consider constrained post-processing only if it is applied equally to base
   and adapters, or report it as a separate intervention.
