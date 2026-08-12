# GSM8K Public vLLM Smoke

Date: 2026-08-12

## Scope

This report records the first pinned public GSM8K no-tool smoke in the
scaffold-to-policy reasoning lane. It imports a small slice of `openai/gsm8k`,
evaluates Qwen3-1.7B with vLLM, and verifies generations with the exact
`gsm_style_normalized_final_v1` verifier.

This is a public-dataset plumbing and calibration run, not a benchmark claim.
The run uses 16 total examples and 4 rollouts per problem.

## Command

The run used the bwrap rootfs:

```bash
RUN_ID=20260812T083000Z-gsm8k-public-vllm-smoke \
DATA_ROOT=experiments/scaffold_to_policy/data/gsm8k_public_vllm_smoke \
RESULTS_ROOT=experiments/scaffold_to_policy/results/gsm8k_public_vllm_smoke \
DEV_PROBLEMS=8 \
OOD_PROBLEMS=8 \
NUM_ROLLOUTS=4 \
MAX_NEW_TOKENS=384 \
experiments/scaffold_to_policy/run_gsm8k_public_vllm_smoke.sh
```

The entrypoint re-executes itself through:

```bash
scripts/rootfs/enter_rootfs.sh -- experiments/scaffold_to_policy/run_gsm8k_public_vllm_smoke.sh
```

## Dataset

Dataset provenance:

| Field | Value |
| --- | --- |
| Dataset | `openai/gsm8k` |
| Subset | `main` |
| Revision | `740312add88f781978c0658806c59bc2815b9866` |
| Source split | `test` |
| Dev slice | offset 0, limit 8 |
| OOD slice | offset 256, limit 8 |

The importer converts GSM8K rows into the shared `gsm_style` JSONL format by
extracting the final answer after the `####` marker and preserving the original
GSM8K rationale as provenance.

Split provenance artifacts:

```text
experiments/scaffold_to_policy/data/gsm8k_public_vllm_smoke/dev_provenance.json
experiments/scaffold_to_policy/data/gsm8k_public_vllm_smoke/ood_test_provenance.json
```

Problem-id hashes:

| Split | Problems | Problem-id hash |
| --- | ---: | --- |
| dev | 8 | `290cc804806e998d7beb4ef9c18c33b89519adf09a20fb5fc05ababd0fe4b6e9` |
| ood_test | 8 | `953e2108958a687fe0e42e80a1fbd3e3eee05bccc7932ba53374ceefc891d6c7` |

## Runtime Condition

The condition is no-tool. vLLM receives only the problem prompt and a system
instruction. The run does not use retrieval, a calculator, Python execution, or
an LLM judge.

The default prompt variant uses the Qwen chat template with
`enable_thinking=False`. The output contract asks the model to end with:

```text
FINAL: <answer>
```

The verifier accepts recognized final-answer markers, normalizes the final
value, and compares it exactly to the normalized GSM8K answer.

## Artifacts

Generated artifacts are ignored by git and live under:

```text
experiments/scaffold_to_policy/data/gsm8k_public_vllm_smoke/
experiments/scaffold_to_policy/results/gsm8k_public_vllm_smoke/
```

The report input is:

```text
experiments/scaffold_to_policy/results/gsm8k_public_vllm_smoke/manifests/report_input_20260812T083000Z-gsm8k-public-vllm-smoke.json
```

Report-input checks:

| Check | Result |
| --- | --- |
| `split_registry_selected` | true |
| `summaries_present` | true |
| `summary_split_counts_match` | true |

The report input records scaffold budget `4`, matching the run's rollout count.

## Results

| Split | Problems | Rollouts | pass@1 | pass@2 | pass@4 | strict pass@1 | strict pass@4 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 8 | 32 | 0.625 | 0.625 | 0.750 | 0.625 | 0.750 | easy 5, elicitable 1, unreached 2 |
| ood_test | 8 | 32 | 0.625 | 0.750 | 0.750 | 0.625 | 0.750 | easy 5, elicitable 1, unreached 2 |

The strict-format curve matches the pass@k curve because every successful
answer was produced through a recognized final-answer line.

Failure breakdown:

| Split | Successes | Dominant failures |
| --- | ---: | --- |
| dev | 17 | wrong final value for house-profit, discount-glasses, sheep-ratio, feed, restart-download |
| ood_test | 17 | wrong final value for tree-weight, fruit-count, flour, pennies-to-dollars; two placeholder `<answer>` final lines |

## Examples

Easy dev example:

```text
Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast every
morning and bakes muffins for her friends every day with four. She sells the
remainder at the farmers' market daily for $2 per fresh duck egg. How much in
dollars does she make every day at the farmers' market?

Generation:
Janet's ducks lay 16 eggs per day.
She eats 3 eggs for breakfast, so she has 16 - 3 = 13 eggs left.
She bakes muffins with 4 eggs, so she has 13 - 4 = 9 eggs left.
She sells 9 eggs at $2 each, so she makes 9 x 2 = $18.

FINAL: 18
```

Elicitable dev example:

```text
Problem: Toulouse has twice as many sheep as Charleston. Charleston has 4 times
as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle
have together if Seattle has 20 sheep?

First sample:
Toulouse has 20 x 4 = 80 sheep.
Charleston has 20 x 2 = 40 sheep.
Toulouse + Charleston + Seattle = 80 + 40 + 20 = 140 sheep.

FINAL: 140

Fourth sample:
Charleston = 4 x 20 = 80
Toulouse = 2 x 80 = 160
Total = 160 + 80 + 20 = 260

FINAL: 260
```

Unreached dev example:

```text
Problem: Kylar went to the store to buy glasses for his new apartment. One
glass costs $5, but every second glass costs only 60% of the price. Kylar wants
to buy 16 glasses. How much does he need to pay for them?

Generation:
First glass: $5
Second glass: 60% of $5 = $3
Total for 16 glasses: (15 glasses x $3) + $5 = $45 + $5 = $50

FINAL: $50
```

The correct answer is 64. The model repeatedly treated every glass after the
first as discounted rather than alternating regular and discounted prices.

Elicitable OOD example:

```text
Problem: Two cups of flour are needed to make a dozen cookies. Carla is making
36 cookies today and 30 cookies tomorrow. How many cups of flour will Carla
need to bake the cookies today and tomorrow?

First sample:
66 cookies / 12 = 5.5
Final: 5.5 cups of flour

Second sample:
66 cookies / 12 = 5.5 dozen
5.5 dozen x 2 cups/dozen = 11 cups of flour

FINAL: 11
```

Placeholder format failure:

```text
Harry has 11 - 3 = 8 more trees than Ferdinand.

FINAL: <answer>
```

The verifier correctly rejected the placeholder because the final value cannot
be normalized.

## Interpretation

This clears the first public reasoning plumbing gate:

- a pinned HF dataset revision can be imported inside the bwrap rootfs;
- public GSM8K rows are converted into the shared exact-verifier problem
  format with per-split provenance;
- the no-tool Qwen3-1.7B vLLM path produces nontrivial easy, elicitable, and
  unreached buckets on public GSM-style problems;
- report inputs now record the actual scaffold budget.

The run is too small for benchmark claims. The next useful step is a larger
public GSM8K calibration with enough elicitable examples to build a small
scaffold-to-policy transfer dataset, followed by a held-out base-vs-adapter
evaluation. MATH-style normalization should remain behind that larger GSM8K
calibration because symbolic answer forms will add verifier complexity.
