# Modular Sequences Transfer Smoke

Date: 2026-08-12

## Scope

This report records the first end-to-end scaffold-to-policy transfer smoke for
the `modular_sequences` exact-verifier reasoning task. It is a plumbing result,
not a scientific transfer claim.

The run used the calibrated modular recurrence band:

- steps: 3-5
- modulus: 37-257
- prompt: chat template
- verifier: exact strict `FINAL: <integer>`

## Command

The completed minimal run used Qwen3-1.7B through the bwrap rootfs for
generation, TorchTitan training, LoRA export, adapter evaluation, and report
input creation:

```bash
RUN_ID=20260812T071500Z-modular-transfer-smoke \
DATA_ROOT=experiments/scaffold_to_policy/data/modular_sequences_transfer_smoke_min \
RESULTS_ROOT=experiments/scaffold_to_policy/results/modular_sequences_transfer_smoke_min \
TRAIN_PROBLEMS=16 \
DEV_PROBLEMS=4 \
OOD_PROBLEMS=4 \
NUM_ROLLOUTS=8 \
EVAL_ROLLOUTS=4 \
MAX_NEW_TOKENS=384 \
TRAIN_STEPS=2 \
CHECKPOINT_STEP=step-2 \
NGPU=1 \
MIN_TRAIN_EXAMPLES=4 \
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
experiments/scaffold_to_policy/data/modular_sequences_transfer_smoke_min/
experiments/scaffold_to_policy/results/modular_sequences_transfer_smoke_min/
```

Key artifacts:

```text
experiments/scaffold_to_policy/results/modular_sequences_transfer_smoke_min/eval/base/train_summary.json
experiments/scaffold_to_policy/results/modular_sequences_transfer_smoke_min/eval/base/dev_summary.json
experiments/scaffold_to_policy/results/modular_sequences_transfer_smoke_min/eval/base/ood_test_summary.json
experiments/scaffold_to_policy/results/modular_sequences_transfer_smoke_min/train/modular_sequences_raw/checkpoint/step-2/
experiments/scaffold_to_policy/results/modular_sequences_transfer_smoke_min/adapters/raw/export_summary.json
experiments/scaffold_to_policy/results/modular_sequences_transfer_smoke_min/eval/adapters/raw/dev_summary.json
experiments/scaffold_to_policy/results/modular_sequences_transfer_smoke_min/eval/adapters/raw/ood_test_summary.json
experiments/scaffold_to_policy/results/modular_sequences_transfer_smoke_min/manifests/report_input_20260812T071500Z-modular-transfer-smoke.json
```

Report-input checks:

| Check | Result |
| --- | --- |
| `split_registry_selected` | true |
| `summaries_present` | true |
| `summary_split_counts_match` | true |

The adapter export summary recorded rank 16, alpha 32, 282 source tensors, and
394 exported tensors across `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, `down_proj`, and `lm_head`.

## Results

The train scaffold collected 16 problems with 8 rollouts each. It found 9
verifier-successful training examples, enough to exercise the TorchTitan SFT
path with the configured minimal threshold of 4 examples.

Base train summary:

| Split | Problems | Rollouts | pass@1 | pass@4 | pass@8 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| train | 16 | 128 | 0.438 | 0.562 | 0.562 | easy 7, elicitable 2, unreached 7 |

Held-out summaries:

| Arm | Split | Problems | Rollouts | pass@1 | pass@4 | pass@8 | Buckets |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| base | dev | 4 | 16 | 0.250 | 0.250 | 0.250 | easy 1, elicitable 0, unreached 3 |
| raw LoRA | dev | 4 | 16 | 0.500 | 0.500 | 0.500 | easy 2, elicitable 0, unreached 2 |
| base | ood_test | 4 | 16 | 0.250 | 0.500 | 0.500 | easy 1, elicitable 1, unreached 2 |
| raw LoRA | ood_test | 4 | 16 | 0.250 | 0.250 | 0.250 | easy 1, elicitable 0, unreached 3 |

Strict-format pass@k matched pass@k in this run because the verifier accepts
only strict final-answer lines as successes.

## Examples

Adapter dev win over base:

```text
x_0 = 242
For i = 1..4:
x_i = (8 * x_(i-1) + 16 * i + 120) mod 257
Answer: 173
```

The base first sample missed the answer after an early modular arithmetic
error and then ran out without a final line. The adapter first sample produced
a verifier-clean trace:

```text
x_1 = (8*242 + 16*1 + 120) mod 257 = 2072 mod 257 = 16
x_2 = (8*16 + 16*2 + 120) mod 257 = 280 mod 257 = 23
x_3 = (8*23 + 16*3 + 120) mod 257 = 352 mod 257 = 95
x_4 = (8*95 + 16*4 + 120) mod 257 = 944 mod 257 = 173

FINAL: 173
```

OOD regression:

```text
x_0 = 118
For i = 1..3:
x_i = (3 * x_(i-1) + 9 * i + 156) mod 216
Answer: 192
```

The base solved it on the first sample:

```text
x_1 = 519 mod 216 = 87
x_2 = 435 mod 216 = 3
x_3 = 192 mod 216 = 192

FINAL: 192
```

The adapter first sample made the same style of reduction mistake seen in the
base model on other problems:

```text
x_1 = 519 mod 216 = 93
x_2 = 453 mod 216 = 21
x_3 = 246 mod 216 = 30
FINAL: 30
```

Shared persistent failure:

```text
x_0 = 85
For i = 1..4:
x_i = (3 * x_(i-1) + 15 * i + 78) mod 212
Answer: 63
```

Both base and adapter carried an off-by-one recurrence state at `x_3`:

```text
x_1 = 348 mod 212 = 136
x_2 = 516 mod 212 = 92
x_3 = 400 mod 212 = 188
x_4 = 702 mod 212 = 106

FINAL: 106
```

The deterministic verifier rejected this because the correct `x_3` is 187 and
the final answer is 63.

## Interpretation

The transfer pipeline is now proven end to end:

- rootfs-managed generation and evaluation;
- split registry validation;
- training data construction from verifier-successful rollouts only;
- TorchTitan LoRA SFT on Qwen3-1.7B;
- PEFT/vLLM adapter export;
- vLLM LoRA evaluation;
- run-scoped manifest and report input.

The result should not be used as evidence that modular recurrence reasoning
improved. The sample is intentionally tiny, the adapter trained for only two
steps over 9 examples, and OOD regressed. The useful conclusion is that the
mechanics are ready for a larger modular transfer run with enough held-out
problems to measure base-elicitable gains and OOD retention.
