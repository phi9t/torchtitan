# MATH Public vLLM Smoke

Date: 2026-08-12

## Scope

This report records the first harder public reasoning smoke in the
scaffold-to-policy ladder. It imports a pinned slice of the MATH-style algebra
dataset, evaluates Qwen3-1.7B with vLLM in a no-tool condition, and verifies
outputs with `math_style_normalized_final_v1`.

This is a plumbing and calibration run, not a benchmark claim. The run uses 16
total examples and 4 rollouts per problem.

## Command

The run used the bwrap rootfs:

```bash
RUN_ID=20260812T091500Z-math-public-vllm-smoke \
DATA_ROOT=experiments/scaffold_to_policy/data/math_public_vllm_smoke \
RESULTS_ROOT=experiments/scaffold_to_policy/results/math_public_vllm_smoke \
DEV_PROBLEMS=8 \
OOD_PROBLEMS=8 \
NUM_ROLLOUTS=4 \
MAX_NEW_TOKENS=512 \
experiments/scaffold_to_policy/run_math_public_vllm_smoke.sh
```

The entrypoint re-executes itself through:

```bash
scripts/rootfs/enter_rootfs.sh -- experiments/scaffold_to_policy/run_math_public_vllm_smoke.sh
```

After the first pass, observed normalization gaps were fixed and the saved
vLLM outputs were rescored inside the rootfs. The final report input uses the
rescored summaries.

## Dataset

Dataset provenance:

| Field | Value |
| --- | --- |
| Dataset | `EleutherAI/hendrycks_math` |
| Subset | `algebra` |
| Revision | `21a5633873b6a120296cce3e2df9d5550074f4a3` |
| Source split | `test` |
| Dev slice | offset 0, limit 8 |
| OOD slice | offset 256, limit 8 |

The original target dataset, `hendrycks/competition_math`, exposed metadata
but failed to load through `datasets` in the rootfs. The working public source
for this smoke is therefore `EleutherAI/hendrycks_math` at the revision above.

Problem-id hashes:

| Split | Problems | Problem-id hash |
| --- | ---: | --- |
| dev | 8 | `18adbb850413f2a05eea9c0affecc15f2cae561e9576508a5f85342f94c8f5e6` |
| ood_test | 8 | `9b437fe6e61f3e6ded89f095f83f9b983fdc02acd2901a8085aef9496c3ab80c` |

## Verifier

The condition is no-tool. vLLM receives only the problem prompt and a system
instruction. The run does not use retrieval, a calculator, Python execution, or
an LLM judge.

The verifier extracts a final value from `FINAL:` or boxed answer markers,
normalizes integers, decimals, simple fractions, LaTeX fractions, and light
literal symbolic forms, then exact-matches against the normalized dataset
answer.

Known limitations:

- no symbolic algebra equivalence;
- no interval, set, matrix, or multi-answer semantic matching;
- literal symbolic cleanup is intentionally conservative.

## Artifacts

Generated artifacts are ignored by git and live under:

```text
experiments/scaffold_to_policy/data/math_public_vllm_smoke/
experiments/scaffold_to_policy/results/math_public_vllm_smoke/
```

The final report input is:

```text
experiments/scaffold_to_policy/results/math_public_vllm_smoke/manifests/report_input_20260812T091500Z-math-public-vllm-smoke.json
```

Report-input checks:

| Check | Result |
| --- | --- |
| `split_registry_selected` | true |
| `summaries_present` | true |
| `summary_split_counts_match` | true |

## Results

| Split | Problems | Rollouts | pass@1 | pass@2 | pass@4 | strict pass@1 | strict pass@4 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 8 | 32 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 | easy 6, elicitable 0, unreached 2 |
| ood_test | 8 | 32 | 0.750 | 0.750 | 0.875 | 0.750 | 0.875 | easy 6, elicitable 1, unreached 1 |

Failure breakdown:

| Split | Successes | Dominant failures |
| --- | ---: | --- |
| dev | 23 | 4 missing final answers; 4 interval-normalization failures; 1 wrong final value |
| ood_test | 22 | 10 placeholder `<answer>` final lines across three problems |

## Examples

Easy dev example:

```text
Problem: How many vertical asymptotes does the graph of
y = 2/(x^2+x-6) have?

Generation:
1. Factor the denominator: x^2 + x - 6 = (x + 3)(x - 2)
2. Vertical asymptotes occur where the denominator is zero.
3. Solve (x + 3)(x - 2) = 0: x = -3 or x = 2.

FINAL: 2
```

Easy dev arithmetic example:

```text
Problem: What is the positive difference between 120% of 30 and 130% of 20?

Generation:
120% of 30 = 36
130% of 20 = 26
|36 - 26| = 10

FINAL: 10
```

Unreached dev example:

```text
Problem: Find x such that ceil(x) + x = 23/7. Express x as a common fraction.

First generation:
The model starts a case analysis but does not finish within the generation
budget with a final answer line.

Verifier result: missing final answer
Expected answer: 9/7
```

Verifier-limitation example:

```text
Problem: For what values of x is it true that x^2 - 5x - 4 <= 10?
Express your answer in interval notation.

Expected dataset answer: x \in [-2,7]
Normalized expected answer: xin-27
Model final answer: [-2, 7]
Normalized model answer: -27
Verifier result: rejected
```

This is a known limitation of the conservative exact normalizer, not evidence
that the model's interval reasoning was wrong.

Easy OOD example:

```text
Problem: Paula invests $10,000 for 5 years at simple interest of 10% per year.
How much is her investment worth?

Generation:
I = 10000 * 0.10 * 5 = 5000
A = 10000 + 5000 = 15000

FINAL: 15000
```

Elicitable OOD example:

```text
Problem: The symbols triangle, square, diamond, clubsuit represent different
integers from 1 to 9. What is the value of square?

First three samples ended with:
FINAL: <answer>

Fourth sample ended with:
FINAL: 3
```

Unreached OOD example:

```text
Problem: A is 40% greater than B and 30% less than C. What is B:C?

The model derives C = 2B, so B/C = 1/2, but every rollout ends with the
placeholder:

FINAL: <answer>
```

The verifier correctly rejects these placeholders.

## Interpretation

This clears the first harder public reasoning smoke:

- pinned public MATH-style rows import inside the bwrap rootfs;
- the exact verifier handles boxed answers, fractions, decimals, currency
  escapes, and simple literal symbolic answers;
- saved vLLM outputs can be rescored after verifier fixes without rerunning
  generation;
- the run yields easy, elicitable, unreached, and verifier-limitation examples.

The run remains too small for a benchmark claim or adapter-training conclusion.
The next useful reasoning step is a larger MATH slice with explicit filtering
for answer forms the verifier can faithfully score, or a second verifier track
for intervals/sets if those problems are needed.
