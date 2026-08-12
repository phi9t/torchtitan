# GSM-Style Verifier Smoke

Date: 2026-08-12

## Scope

This report records the first GSM-style final-answer normalization smoke in the
scaffold-to-policy reasoning lane. It is a verifier and artifact-contract smoke,
not a model evaluation.

The purpose is to prepare the next public-reasoning stage by testing exact
answer normalization before importing or evaluating a public GSM-style dataset.

## Command

The run used the bwrap rootfs:

```bash
RUN_ID=20260812T075500Z-gsm-style-smoke \
DATA_ROOT=experiments/scaffold_to_policy/data/gsm_style_smoke \
RESULTS_ROOT=experiments/scaffold_to_policy/results/gsm_style_smoke \
experiments/scaffold_to_policy/run_gsm_style_smoke.sh
```

The entrypoint re-executes itself through:

```bash
scripts/rootfs/enter_rootfs.sh -- experiments/scaffold_to_policy/run_gsm_style_smoke.sh
```

## Artifacts

Checked-in fixture sources:

```text
experiments/scaffold_to_policy/fixtures/gsm_style_train.jsonl
experiments/scaffold_to_policy/fixtures/gsm_style_dev.jsonl
experiments/scaffold_to_policy/fixtures/gsm_style_ood_test.jsonl
```

Generated artifacts are ignored by git and live under:

```text
experiments/scaffold_to_policy/data/gsm_style_smoke/
experiments/scaffold_to_policy/results/gsm_style_smoke/
```

The report input is:

```text
experiments/scaffold_to_policy/results/gsm_style_smoke/manifests/report_input_20260812T075500Z-gsm-style-smoke.json
```

Report-input checks:

| Check | Result |
| --- | --- |
| `split_registry_selected` | true |
| `summaries_present` | true |
| `summary_split_counts_match` | true |

## Verifier Contract

The verifier is `gsm_style_normalized_final_v1`.

It accepts final answers from:

- strict `FINAL: <answer>` lines;
- GSM8K-style `#### <answer>` lines.

It normalizes:

- commas and currency markers;
- `\boxed{...}` answers;
- integers and negative integers;
- decimals, represented internally as reduced fractions;
- simple fractions and LaTeX `\frac{a}{b}` forms.

It does not use tools, retrieval, or an LLM judge.

## Results

The fixture smoke evaluates 3 dev and 3 OOD problems, with 4 fixture rollouts
per problem. The first rollout is intentionally correct, while later rollouts
exercise equivalent formatting, wrong answers, and missing final answers.

| Split | Problems | Rollouts | pass@1 | pass@4 | strict pass@1 | Failure coverage |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 3 | 12 | 1.000 | 1.000 | 1.000 | success, wrong normalized answer, missing final |
| ood_test | 3 | 12 | 1.000 | 1.000 | 1.000 | success, wrong normalized answer, missing final |

The fixtures covered these answer forms:

| Form | Example |
| --- | --- |
| GSM8K marker | `#### 42` |
| decimal | `FINAL: 7.5` -> `15/2` |
| boxed | `FINAL: \boxed{36}` |
| currency/comma | `FINAL: $1,250` |
| simple fraction | `FINAL: 5/4` |
| negative integer | `FINAL: -4` |

## Interpretation

This clears the GSM-style normalization smoke gate. The next step is a small
no-tool real-model GSM-style evaluation using the same exact verifier and
artifact contract. Public benchmark claims should wait until the dataset
revision, split provenance, and no-tool prompt condition are pinned in the run
registry.
