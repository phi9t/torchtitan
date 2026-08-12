# Modular Sequences vLLM Calibration

Date: 2026-08-12

## Scope

This report records the second local exact-verifier reasoning task in the
scaffold-to-policy ladder. `modular_sequences` generates modular recurrence
problems and checks only a strict final integer line:

```text
FINAL: <integer>
```

The run is a calibration smoke, not a training result. Its purpose is to find a
nontrivial base-model difficulty band before using this task for
scaffold-to-policy transfer.

## Final Calibration Run

The useful calibration run used Qwen3-1.7B through vLLM inside the bwrap rootfs:

```bash
RUN_ID=20260812T064500Z-modular-sequences-small-calibration \
DATA_ROOT=experiments/scaffold_to_policy/data/modular_sequences_small_calibration \
RESULTS_ROOT=experiments/scaffold_to_policy/results/modular_sequences_small_calibration \
NUM_ROLLOUTS=8 \
MAX_NEW_TOKENS=512 \
MIN_STEPS=3 \
MAX_STEPS=5 \
MIN_MODULUS=37 \
MAX_MODULUS=257 \
PROMPT_VARIANT=chat \
experiments/scaffold_to_policy/run_modular_sequences_vllm_smoke.sh
```

The script re-executed through:

```bash
scripts/rootfs/enter_rootfs.sh -- experiments/scaffold_to_policy/run_modular_sequences_vllm_smoke.sh
```

## Artifacts

Generated artifacts are ignored by git and live under:

```text
experiments/scaffold_to_policy/data/modular_sequences_small_calibration/
experiments/scaffold_to_policy/results/modular_sequences_small_calibration/
```

The report input is:

```text
experiments/scaffold_to_policy/results/modular_sequences_small_calibration/manifests/report_input_20260812T064500Z-modular-sequences-small-calibration.json
```

Report-input checks:

| Check | Result |
| --- | --- |
| `split_registry_selected` | true |
| `summaries_present` | true |
| `summary_split_counts_match` | true |

## Results

The calibration generated 16 train, 8 dev, and 8 OOD synthetic recurrence
problems. It evaluated dev and OOD with 8 rollouts per problem.

| Split | Problems | Rollouts | pass@1 | pass@4 | pass@8 | strict pass@1 | strict pass@8 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 8 | 64 | 0.250 | 0.500 | 0.625 | 0.250 | 0.625 | easy 2, elicitable 3, unreached 3 |
| ood_test | 8 | 64 | 0.500 | 0.625 | 0.625 | 0.500 | 0.625 | easy 4, elicitable 1, unreached 3 |

This is the first local reasoning smoke in this directory with a useful
difficulty profile: it has first-sample successes, search-elicitable cases,
and unreached cases in the same run.

## Failure Modes

The main failures were:

- missing `FINAL` line after verbose traces;
- arithmetic slips in intermediate recurrence values;
- wrong strict final values after otherwise verifier-compliant formatting.

Dev failure counts:

| Failure | Count |
| --- | ---: |
| `success` | 15 |
| `missing FINAL line` | 33 |
| wrong final value | 16 |

OOD failure counts:

| Failure | Count |
| --- | ---: |
| `success` | 24 |
| `missing FINAL line` | 18 |
| wrong final value | 22 |

## Examples

Elicitable dev problem, solved on rollout 4:

```text
x_0 = 155
For i = 1..5:
x_i = (10 * x_(i-1) + 10 * i + 126) mod 195
Answer: 131
```

The first three samples did not emit a final line. Rollout 4 completed the
trace and passed:

```text
x_1 = ... = 126
x_2 = ... = 41
x_3 = ... = 176
x_4 = ... = 171
x_5 = ... = 131

FINAL: 131
```

Unreached dev problem:

```text
x_0 = 169
For i = 1..5:
x_i = (7 * x_(i-1) + 16 * i + 95) mod 181
Answer: 119
```

Several samples produced strict `FINAL` lines, but the arithmetic diverged
early. One representative failure started with an incorrect `x_1`:

```text
x_1 = (1183 + 16 + 95) mod 181
x_1 = 1300 mod 181
x_1 = 33
...
FINAL: 45
```

The verifier correctly rejected it because the deterministic answer is 119.

## Calibration Attempts

Two earlier calibrations define the useful boundary:

| Run | Settings | Result |
| --- | --- | --- |
| `20260812T061500Z-modular-sequences-vllm-smoke` | 10-16 steps, modulus 997-9973, 8 rollouts, 384 tokens | dev/OOD both 0.000 pass@8; mostly missing `FINAL` |
| `20260812T062000Z-modular-sequences-vllm-calibration` | 4-7 steps, modulus 97-997, 8 rollouts, 512 tokens | dev 0.000 pass@8; OOD 0.375 pass@8 |
| `20260812T063500Z-modular-sequences-concise-calibration` | 4-7 steps, modulus 97-997, concise prompt, 256 tokens | dev/OOD both 0.000 pass@8; every rollout missed `FINAL` |
| `20260812T064500Z-modular-sequences-small-calibration` | 3-5 steps, modulus 37-257, 8 rollouts, 512 tokens | dev 0.625 pass@8; OOD 0.625 pass@8 |

The concise prompt did not help because Qwen3 still expanded into verbose
symbolic traces and hit the token budget before the final line. The smaller
recurrence settings produced the desired mix of easy, elicitable, and unreached
cases while preserving exact verification.

## Interpretation

`modular_sequences` is a better next local transfer candidate than
`arithmetic_words`. The arithmetic-words smoke was saturated at pass@1, while
the calibrated modular task exposes both formatting and recurrence-arithmetic
failure modes.

The next experimental step should use the small-calibration band as the first
scaffold collection setting, then generate best-of-N successful traces for
training a small policy adapter. The first promotion gate should be held-out
improvement on the dev base-elicitable subset without collapsing pass@8.
