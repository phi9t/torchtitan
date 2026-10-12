# GSM-Style vLLM Smoke

Date: 2026-08-12

## Scope

This report records the first no-tool real-model GSM-style smoke in the
scaffold-to-policy reasoning lane. It evaluates Qwen3-1.7B with vLLM on the
checked-in GSM-style fixtures and verifies generations with the exact
`gsm_style_normalized_final_v1` verifier.

This is infrastructure evidence only. It is not a public GSM8K or GSM-style
benchmark result because the run uses local checked-in fixtures rather than a
pinned public dataset revision.

## Command

The run used the bwrap rootfs:

```bash
RUN_ID=20260812T080500Z-gsm-style-vllm-smoke \
DATA_ROOT=experiments/scaffold_to_policy/data/gsm_style_vllm_smoke \
RESULTS_ROOT=experiments/scaffold_to_policy/results/gsm_style_vllm_smoke \
NUM_ROLLOUTS=4 \
MAX_NEW_TOKENS=256 \
experiments/scaffold_to_policy/run_gsm_style_vllm_smoke.sh
```

The entrypoint re-executes itself through:

```bash
scripts/rootfs/enter_rootfs.sh -- experiments/scaffold_to_policy/run_gsm_style_vllm_smoke.sh
```

## Runtime Condition

The smoke keeps the public-reasoning condition no-tool: vLLM receives only the
problem prompt and system instruction. There is no retrieval, calculator,
Python execution, or LLM judge.

The default prompt variant uses the Qwen chat template with
`enable_thinking=False`. The output contract asks the model to end with:

```text
FINAL: <answer>
```

The verifier accepts recognized final-answer markers and normalizes the final
value before exact comparison.

## Artifacts

Checked-in fixture sources:

```text
experiments/scaffold_to_policy/fixtures/gsm_style_train.jsonl
experiments/scaffold_to_policy/fixtures/gsm_style_dev.jsonl
experiments/scaffold_to_policy/fixtures/gsm_style_ood_test.jsonl
```

Generated artifacts are ignored by git and live under:

```text
experiments/scaffold_to_policy/data/gsm_style_vllm_smoke/
experiments/scaffold_to_policy/results/gsm_style_vllm_smoke/
```

The report input is:

```text
experiments/scaffold_to_policy/results/gsm_style_vllm_smoke/manifests/report_input_20260812T080500Z-gsm-style-vllm-smoke.json
```

Report-input checks:

| Check | Result |
| --- | --- |
| `split_registry_selected` | true |
| `summaries_present` | true |
| `summary_split_counts_match` | true |

## Results

The smoke evaluates 3 dev and 3 OOD fixture problems, with 4 sampled rollouts
per problem.

| Split | Problems | Rollouts | pass@1 | pass@2 | pass@4 | strict pass@1 | strict pass@4 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 3 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | easy 3, elicitable 0, unreached 0 |
| ood_test | 3 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | easy 3, elicitable 0, unreached 0 |

Failure breakdown:

| Split | Successes | Other failures |
| --- | ---: | --- |
| dev | 12 | none |
| ood_test | 11 | `could not normalize final answer '<answer>'`: 1 |

The single OOD failure occurred on a non-first-sample rollout for the fraction
problem. The model computed the right expression but emitted the placeholder
`Final: <answer>`, which the verifier rejected.

## Examples

Dev integer example:

```text
Problem: A class has 18 red pencils and 24 blue pencils. How many pencils are
there in all?

Generation:
18 + 24 = 42
FINAL: 42
```

Dev decimal normalization example:

```text
Problem: A rope is 15 meters long. It is cut into two equal pieces. How long
is each piece?

Generation:
15 divided by 2 equals 7.5
FINAL: 7.5

Normalized answer: 15/2
```

OOD fraction example:

```text
Problem: A tank is 3/4 full. Rain adds another 1/2 tank. How full is the tank?

Generation:
Trace:
Start with 3/4 full.
Add 1/2 tank.
Convert to common denominator: 3/4 + 2/4 = 5/4.

FINAL: 5/4
```

OOD rejected placeholder example:

```text
Trace:
- Initial volume: 3/4
- Rain adds: 1/2
- Total volume: 3/4 + 1/2

Convert to common denominator:
- 3/4 + 2/4 = 5/4

Final: <answer>
```

The rejection is correct: the final value cannot be normalized to the expected
answer.

## Interpretation

This clears the first GSM-style real-model plumbing gate:

- the bwrap rootfs can run vLLM generation for Qwen3-1.7B on GSM-style prompts;
- checked-in GSM-style fixtures can be prepared, split-validated, evaluated,
  summarized, and converted into a report input;
- exact final-answer normalization handles integer, decimal, fraction, boxed,
  comma, and currency-style answers on real model outputs.

The run is deliberately too small and too easy for scientific claims. The next
reasoning step should import or construct a pinned larger GSM-style numeric
subset with the same no-tool condition and verifier, then add a MATH-style
normalization smoke only after GSM-style dataset provenance and report
selection are stable.
